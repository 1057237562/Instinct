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

from model.grouped_mm import inference_backend
from model.moe_dispatch import refresh_inference_stacks
from model.static_cache import STATIC_CACHE_INITIAL_DECODE_TOKENS

# Warm only the initial decode bucket.  The output limit is preserved by
# growing to larger buckets if a response actually reaches this headroom.
WARMUP_MAX_NEW_TOKENS = STATIC_CACHE_INITIAL_DECODE_TOKENS


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


def warmup_decode(model):
    """Compile the decode step at load time, and record its CUDA graph.

    The prompt shape is deliberately left to the first real call. It is dynamic in
    length, and Inductor re-specializes for short prompts (a 16-token prompt and a
    256-token one compile separately, ~90 s each on the 512-dim MoE), so warming a
    fake prompt only adds compiles the session never reuses.
    """
    device = next(model.parameters()).device
    vocab = getattr(model.config, "vocab_size", 6400)
    prompt = torch.randint(3, max(vocab - 100, 4), (1, 1), device=device)
    with torch.inference_mode():
        state = model._decode_state_for(prompt, None, WARMUP_MAX_NEW_TOKENS,
                                       use_cache=True, early_exit=False, kwargs={})
        if state is None:
            return
        forward_kwargs = {'logits_to_keep': 1}
        state.step_ids.fill_(int(prompt[0, 0]))
        state.step_position.fill_(0)
        model(state.step_ids, None, state.cache, use_cache=True,
              position_ids=state.step_position, **forward_kwargs)
        if state.graph is None and not state.capture_failed:
            recorded = model._capture_decode_step(state.cache, state.step_ids,
                                                  state.step_position, forward_kwargs)
            if recorded is None:
                state.capture_failed = True
            else:
                state.graph = recorded


def _compile_model_trunk(model, mode='full') -> bool:
    """Compile the Transformer trunk; restore the eager module on any failure.

    ``reduce-overhead`` captures the compiled trunk as a CUDA graph. That is only
    viable because generation now writes into a fixed-shape KV cache: a graph
    that grows with ``torch.cat`` every step would need a fresh capture per
    length. Prompt shapes stay symbolic, so Inductor is told to skip capturing
    them instead of accumulating one pool per prompt length.
    """
    original = model.model
    compile_kwargs = {'dynamic': True}
    if mode == 'reduce-overhead':
        compile_kwargs['mode'] = 'reduce-overhead'
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    try:
        model.model = CompiledTrunk(torch.compile(original, **compile_kwargs), original)
        warmup_decode(model)
        return True
    except Exception as exc:
        model.model = original
        print(f"[Inference] full-trunk compile unavailable "
              f"({type(exc).__name__}: {str(exc)[:160]}); using fused kernels only.",
              flush=True)
        return False


def _prime_traced_path(model):
    """Settle one-time probes that would otherwise break the compiled graph.

    Both the flash-attn probe and the RoPE-cache check run inside the traced
    trunk: the failed import breaks the graph once per attention layer, and the
    tensor-conditioned cache check breaks it once per forward. Doing them eagerly
    keeps the graph whole, which is what the MoE decode path needs.
    """
    from model.flash_attn_4 import resolve_flash_attn

    resolve_flash_attn()
    trunk = getattr(model.model, "original", model.model)
    ensure = getattr(trunk, "ensure_rope_caches", None)
    if ensure is not None:
        ensure()


def optimize_inference(model, mode='auto'):
    if mode == 'off' or next(model.parameters()).device.type != 'cuda':
        return model
    if not type(model).__module__.startswith('model.model_instinct'):
        return model
    if sys.platform == 'win32' and not sys.flags.utf8_mode:
        print('[Inference] Windows compilation requires PYTHONUTF8=1 before launch; using eager.', flush=True)
        return model
    rms, gate = FusedKernel(rms_kernel), FusedKernel(swiglu_kernel)
    norms = gates = stacks = 0
    parameters = next(model.parameters())
    stack_modules = []
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
            # Both are resolved here rather than in the forward: the backend
            # lookup must not run inside Dynamo's traced region, and the
            # version-counter check that validates the stacks cannot be guarded
            # there at all. The eager forward refreshes them every token.
            layer._inference_backend = inference_backend(parameters.device, parameters.dtype)
            refresh_inference_stacks([layer])
            stack_modules.append(layer)
            stacks += 1
    # Kept on the wrapper (never inside the compiled trunk) for the per-forward
    # refresh in InstinctForCausalLM.forward.
    model._inference_stack_modules = stack_modules or None
    detail = f', {stacks} expert-weight stacks' if stacks else ''
    _prime_traced_path(model)
    # Fixed-shape decode: generate() uses a preallocated in-place cache so every
    # step has the same shape, which is what the captured graph needs. Only
    # inference-optimized models opt in; trainers and eval scripts keep the
    # growing cache until they call this.
    model._static_cache_ok = True
    trunk = mode in ('full', 'reduce-overhead') and _trunk_compile_eligible(model.config) and _compile_model_trunk(model, mode)
    if trunk:
        captured = ' (decode step captured as a CUDA graph)' if mode == 'reduce-overhead' else ''
        print(f'[Inference] Compiled kernels enabled: {norms} RMSNorm, {gates} SwiGLU{detail}, '
              f'full-trunk torch.compile (prefill+decode warmed up){captured}.', flush=True)
    else:
        print(f'[Inference] Compiled kernels enabled: {norms} RMSNorm, {gates} SwiGLU{detail}; first use compiles/caches kernels.', flush=True)
    return model
