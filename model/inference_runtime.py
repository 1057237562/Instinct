"""Inference-only fused kernels shared across native model layers.

Compile stable numerical kernels, leaving streaming, cache growth and Python
control flow outside Dynamo. Parameters/state_dict and training remain intact.

Mode ``full`` additionally compiles the whole Transformer trunk with dynamic
shapes and warms up the prefill and decode graphs at load time. CUDA graphs
(``reduce-overhead``) are deliberately NOT used: the decode loop grows the KV
cache with ``torch.cat`` every step, which cannot be captured.
"""
import sys
import torch
import torch.nn.functional as F
from torch import nn


def rms_kernel(x, weight, eps):
    value = x.float()
    return (weight * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps))).to(x.dtype)


def swiglu_kernel(gate, up):
    return F.silu(gate) * up


class FusedKernel:
    def __init__(self, function, compiler=None):
        self.function = function
        self.compiler = compiler or torch.compile
        self.compiled = None
        self.failed = False

    def __call__(self, *args):
        if not self.failed:
            try:
                if self.compiled is None:
                    self.compiled = self.compiler(self.function, fullgraph=True, dynamic=True)
                return self.compiled(*args)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                self.failed = True
                self.compiled = None
                print(f'[Inference] {self.function.__name__} compile unavailable; using eager: {type(exc).__name__}: {str(exc)[:200]}', flush=True)
        return self.function(*args)


def select_inference_dtype(device: str) -> torch.dtype:
    """BF16 on Ampere+ CUDA GPUs, FP16 elsewhere (the previous default).

    MoE decode needs BF16 weights to stay on the grouped-GEMM expert path:
    FP16 silently falls back to the per-expert loop, which synchronizes with
    the host once per expert per decoded token.
    """
    if device.startswith("cuda") and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float16


class CompiledTrunk(nn.Module):
    """Whole-trunk torch.compile wrapper that keeps diagnostic hooks eager.

    Per-call Python callbacks (the chat logit-lens ``layer_callback`` closures,
    ``exit_check_fn``) re-key Dynamo's guards and would burn the recompile
    limit; ``return_intermediate`` changes the output structure outright.
    Those calls fall back to the original module; normal generation stays on
    the compiled path. ``forward`` itself is compiler-disabled so Dynamo only
    ever traces the wrapped InstinctModel, never this dispatch branch.
    """

    _EAGER_KWARGS = ("layer_callback", "exit_check_fn", "return_intermediate")

    def __init__(self, compiled, original):
        super().__init__()
        self.compiled = compiled
        self.original = original

    @torch.compiler.disable
    def forward(self, *args, **kwargs):
        if any(key in kwargs for key in self._EAGER_KWARGS):
            return self.original(*args, **kwargs)
        return self.compiled(*args, **kwargs)


def _trunk_compile_eligible(config) -> bool:
    """Full-trunk compilation is only validated on the standard topology."""
    return (
        getattr(config, "model_architecture", "standard") == "standard"
        and getattr(config, "residual_type", "standard") == "standard"
    )


def _warmup_compiled_model(model):
    """Trigger both graph shapes at load time: prefill (seq>1, no cache) and
    decode (seq=1, growing cache), so compilation cost lands in the loader
    instead of the first user message."""
    vocab = getattr(model.config, "vocab_size", 6400)
    prompt = torch.randint(
        3, max(vocab - 100, 4), (1, 8), device=next(model.parameters()).device
    )
    model.generate(input_ids=prompt, max_new_tokens=2, do_sample=False, eos_token_id=None)


def _compile_model_trunk(model) -> bool:
    """Compile the Transformer trunk; restore the eager module on any failure."""
    original = model.model
    try:
        model.model = CompiledTrunk(torch.compile(original, dynamic=True), original)
        _warmup_compiled_model(model)
        return True
    except Exception as exc:
        model.model = original
        print(f"[Inference] full-trunk compile unavailable "
              f"({type(exc).__name__}: {str(exc)[:160]}); using fused kernels only.",
              flush=True)
        return False


def optimize_inference(model, mode='auto'):
    if mode == 'off' or next(model.parameters()).device.type != 'cuda':
        return model
    if not type(model).__module__.startswith('model.model_instinct'):
        return model
    if sys.platform == 'win32' and not sys.flags.utf8_mode:
        print('[Inference] Windows compilation requires PYTHONUTF8=1 before launch; using eager.', flush=True)
        return model
    if mode == 'reduce-overhead':
        print('[Inference] CUDA graphs cannot capture the growing decode KV cache; using full-trunk compile instead.', flush=True)
        mode = 'full'
    rms, gate = FusedKernel(rms_kernel), FusedKernel(swiglu_kernel)
    norms = gates = stacks = 0
    for layer in model.modules():
        if type(layer).__name__ == 'RMSNorm' and type(layer).__module__.startswith('model.model_instinct'):
            layer._inference_norm = rms
            norms += 1
        if type(layer).__name__ == 'FeedForward' and getattr(model.config, 'hidden_act', '') == 'silu':
            layer._inference_gate = gate
            gates += 1
        if type(layer).__name__ == 'MOEFeedForward':
            # Marks the expert list for moe_dispatch's stacked-weight cache.
            # Trainers never call optimize_inference, so training keeps
            # rebuilding the stack inside the autograd graph.
            layer.experts._inference_stack_cache = {}
            stacks += 1
    detail = f', {stacks} expert-weight stacks' if stacks else ''
    trunk = mode == 'full' and _trunk_compile_eligible(model.config) and _compile_model_trunk(model)
    if trunk:
        print(f'[Inference] Compiled kernels enabled: {norms} RMSNorm, {gates} SwiGLU{detail}, '
              f'full-trunk torch.compile (prefill+decode warmed up).', flush=True)
    else:
        print(f'[Inference] Compiled kernels enabled: {norms} RMSNorm, {gates} SwiGLU{detail}; first use compiles/caches kernels.', flush=True)
    return model
