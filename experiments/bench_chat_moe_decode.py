"""Reproduce the chat WebUI decode path to locate MoE throughput loss.

``scripts/web_demo.py`` loads through ``select_inference_dtype`` (bf16 on
Ampere+) and ``optimize_inference(..., 'auto')``, so ``bf16 --mode auto`` is the
faithful reproduction; fp16 is kept as a control because MoE dispatch only
selects the grouped-GEMM path for BF16 activations.

``--mode`` trades load time against steady state: 'auto' skips the trunk compile
(load is instant, decode ~3.7 ms/token on the 512-dim MoE), 'full' compiles the
whole trunk (~50 s and a second compile for the first real prompt, decode ~2.7
ms/token and a 2x faster prefill).

Variants run round-robin and each variant's median is reported: this machine
loses up to 1.7x between the first and last measurement of one process, so a
block-per-variant ordering would report that drift as a speedup. ``--profile``
is a separate, non-interleaved diagnostic.

Run while nothing else uses the GPU:
    python experiments/bench_chat_moe_decode.py --dtype bf16 --mode full
    python experiments/bench_chat_moe_decode.py --dtype bf16 --tokens 64 \\
        --ablate untraced_experts --ablate legacy_cache
    python experiments/bench_chat_moe_decode.py --dtype bf16 --profile

Ablations, in the order they were used to find the losses:
    untraced_experts  pre-optimization MoE dispatch (autograd path, graph break
                      per layer) -- isolates the traceable inference dispatch
    legacy_cache      growing torch.cat KV cache -- isolates the preallocated one
    skip_experts      no expert dispatch at all -- upper bound on MoE cost
    force_expert_loop per-expert Python loop with a host sync per expert
    native_grouped_mm / cached_grouped_mm  other plan backends
    bf16_kv_cache / fp32_kv_cache          unquantized KV storage
    cached_expert_weights                  expert stack cache on vs off
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: F401 -- Windows DLL order
import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.inference_runtime import optimize_inference
from model.moe_dispatch import _can_use_grouped_mm
from model.grouped_mm import _auto_backend

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / 'checkpoints' / 'pretrain_20260919_174858_512_moe.json'
WEIGHT = REPO / 'out' / 'pretrain_20260919_174858_512_moe.pth'

DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}
ABLATIONS = ('cached_expert_weights', 'bf16_kv_cache', 'fp32_kv_cache', 'force_expert_loop',
             'native_grouped_mm', 'cached_grouped_mm', 'skip_experts', 'legacy_cache',
             'untraced_experts')


def build_model(dtype, device='cuda', config_path=None, weight_path=None, mode='full'):
    """Load exactly as scripts/web_demo.py does, minus the tokenizer."""
    with open(config_path or CONFIG, 'r', encoding='utf-8') as handle:
        config = InstinctConfig(**json.load(handle))
    model = InstinctForCausalLM(config)
    state = torch.load(weight_path or WEIGHT, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=False)
    del state
    model = model.to(dtype).eval().to(device)
    return optimize_inference(model, mode)


def trunk_of(model):
    """The Transformer trunk, unwrapped from ``mode='full'``'s CompiledTrunk."""
    return getattr(model.model, 'original', model.model)


def report_forward_path(model):
    """Which expert kernel the loaded model's dtype actually selects."""
    block = next(iter(trunk_of(model).layers))
    if not hasattr(block.mlp, 'experts'):
        print('[path] dense FFN (no expert dispatch)')
        return
    experts = list(block.mlp.experts)
    x = torch.randn(64, model.config.hidden_size, device='cuda',
                    dtype=experts[0].gate_proj.weight.dtype)
    print(f"[path] expert weights dtype             : {experts[0].gate_proj.weight.dtype}")
    print(f"[path] grouped GEMM plan backend        : {_auto_backend(torch.device('cuda'))}")
    print(f"[path] grouped GEMM used by forward     : {_can_use_grouped_mm(x, experts)}")


class Ablation:
    """Apply a suspected per-token cost patch and give back a restore handle."""

    def __init__(self, model, name):
        self.model = model
        self.name = name

    def __enter__(self):
        import model.moe_dispatch as dispatch

        self.restore = []
        if self.name == 'cached_expert_weights':
            # optimize_inference already installs the stack cache, so this only
            # verifies it is live: an uncached stack recopies every expert weight
            # on each MoE call, which eval weights do not need.
            original = dispatch._stacked_expert_weights

            def uncached(experts, weight_name, dtype):
                cache = getattr(experts, '_inference_stack_cache', None)
                try:
                    delattr(experts, '_inference_stack_cache')
                    return original(experts, weight_name, dtype)
                finally:
                    if cache is not None:
                        experts._inference_stack_cache = cache

            dispatch._stacked_expert_weights = uncached
            self.restore.append(lambda: setattr(dispatch, '_stacked_expert_weights', original))
        elif self.name == 'force_expert_loop':
            # Same dtype, different dispatch: isolates the per-expert Python loop
            # (and its blocking nonzero calls) from fp16 arithmetic itself.
            original = dispatch._can_use_grouped_mm
            dispatch._can_use_grouped_mm = lambda x, experts: False
            self.restore.append(lambda: setattr(dispatch, '_can_use_grouped_mm', original))
        elif self.name in ('bf16_kv_cache', 'fp32_kv_cache'):
            # Quantized caches are dequantized and re-quantized over the whole
            # (growing) sequence every step; this isolates that cost from the
            # fp8/fp16/fp32 storage dtype itself.
            target = self.name.split('_')[0]
            for layer in trunk_of(self.model).layers:
                attention = layer.self_attn
                previous = attention.kv_cache_dtype
                attention.kv_cache_dtype = target
                self.restore.append(lambda a=attention, p=previous: setattr(a, 'kv_cache_dtype', p))
        elif self.name == 'legacy_cache':
            # Strains the growing torch.cat cache against the preallocated one.
            previous = getattr(self.model, '_static_cache_ok', False)
            self.model._static_cache_ok = False
            self.restore.append(lambda: setattr(self.model, '_static_cache_ok', previous))
        elif self.name == 'untraced_experts':
            # Clears the inference dispatch installed by optimize_inference, so
            # the MoE block falls back to the autograd path that breaks the
            # compiled graph once per layer (the pre-optimization behaviour).
            import model.model_instinct as backbone

            trunk = getattr(self.model.model, 'original', self.model.model)
            for layer in trunk.layers:
                if type(layer.mlp).__name__ != 'MOEFeedForward':
                    continue
                previous = (layer.mlp._inference_backend, layer.mlp._inference_stacked)
                layer.mlp._inference_backend = None
                layer.mlp._inference_stacked = None
                self.restore.append(
                    lambda mlp=layer.mlp, state=previous: (
                        setattr(mlp, '_inference_backend', state[0]),
                        setattr(mlp, '_inference_stacked', state[1]),
                    )
                )
        elif self.name in ('native_grouped_mm', 'cached_grouped_mm'):
            # _auto_backend prefers Triton on this platform; compare the other
            # two plan backends at identical routing to separate launch/wrapper
            # overhead from the GEMM itself.
            forced = self.name.split('_')[0]
            original = dispatch.make_grouped_mm_plan

            def forcing(offsets, *, backend=None, _forced=forced, _original=original):
                return _original(offsets, backend=_forced)

            dispatch.make_grouped_mm_plan = forcing
            self.restore.append(lambda: setattr(dispatch, 'make_grouped_mm_plan', original))
        elif self.name == 'skip_experts':
            # Upper bound on everything the MoE dispatch costs per layer: the
            # router, the routing metadata, the expert GEMMs and the scatter-add.
            import model.model_instinct as backbone

            original = backbone.routed_moe_forward

            def passthrough(x, gate, experts, *, num_experts_per_tok, norm_topk_prob, act_fn):
                flat = x.reshape(-1, x.shape[-1])
                zero_scores = flat.new_zeros((flat.shape[0], 1))
                return x, zero_scores, zero_scores.new_zeros((flat.shape[0], 1), dtype=torch.long)

            backbone.routed_moe_forward = passthrough
            self.restore.append(lambda: setattr(backbone, 'routed_moe_forward', original))
        elif self.name != 'baseline':
            raise ValueError(f'unknown ablation: {self.name}')
        return self

    def __exit__(self, *exc):
        for undo in reversed(self.restore):
            undo()
        return False


def make_runner(model, prompt, tokens):
    input_ids = torch.randint(10, model.config.vocab_size, (1, prompt), device='cuda')
    attention_mask = torch.ones_like(input_ids)

    def run(steps):
        model.generate(input_ids=input_ids, attention_mask=attention_mask,
                       max_new_tokens=steps, eos_token_id=None, temperature=0.9,
                       top_p=0.85, top_k=50, do_sample=True)

    def warm():
        run(2)

    def long_run():
        run(tokens)

    def prefill():
        with torch.inference_mode():
            model(input_ids, attention_mask=attention_mask, use_cache=True,
                  logits_to_keep=1, past_key_values=None)

    return warm, long_run, prefill


def time_once(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def measure(model, prompt, tokens, rounds):
    """Interleave short/long runs so drift hits both, then derive per-token cost."""
    warm, long_run, prefill = make_runner(model, prompt, tokens)
    warm()
    short_samples, long_samples, prefill_samples = [], [], []
    for _ in range(rounds):
        short_samples.append(time_once(warm))
        long_samples.append(time_once(long_run))
        prefill_samples.append(time_once(prefill))
    decode = (statistics.median(long_samples) - statistics.median(short_samples)) / (tokens - 2)
    return decode, statistics.median(prefill_samples)


def report(label, decode, prefill, prompt, tokens):
    print(f"[timing] {label:26s}: {decode * 1e3:7.1f} ms/token ({1.0 / decode:6.2f} tok/s) "
          f"| prefill {prefill * 1e3:6.1f} ms ({prompt / prefill:7.1f} tok/s) "
          f"over {tokens} decoded tokens")


def profile_generate(model, prompt, tokens):
    from torch.profiler import ProfilerActivity, profile

    warm, long_run, _ = make_runner(model, prompt, tokens)
    warm()
    torch.cuda.synchronize()
    wall = time_once(long_run)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        long_run()
    kernels = [event for event in prof.events() if event.device_type.name == 'CUDA' and event.name]
    print(f"[profile] wall time / token                : {wall / tokens * 1e3:8.1f} ms")
    print(f"[profile] CUDA kernel launches / token  : {len(kernels) / tokens:.1f}")
    average = prof.key_averages()
    print(f"[profile] CUDA busy / token             : "
          f"{sum(event.self_device_time_total for event in average) / tokens / 1e3:.1f} ms "
          f"(host-side aten total {sum(event.self_cpu_time_total for event in average) / tokens / 1e3:.1f} ms)")
    print('[profile] top kernels by self CUDA time:')
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    print('[profile] top ops by self host time:')
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))


def count_host_syncs(model, prompt, tokens):
    """Sync debug mode reports every call site that blocks the host."""
    counts = {}
    warm, long_run, _ = make_runner(model, prompt, tokens)
    warm()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        torch.cuda.set_sync_debug_mode('warn')
        try:
            long_run()
        finally:
            torch.cuda.set_sync_debug_mode('default')
    for item in caught:
        text = str(item.message).splitlines()[0][:100]
        key = f'{Path(item.filename).name}:{item.lineno} :: {text}'
        counts[key] = counts.get(key, 0) + 1
    print(f"[sync] blocking ops per token ({tokens} decoded tokens, top sites):")
    for key, value in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
        print(f'    {value / tokens:8.1f}/token  {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', default='fp16', choices=sorted(DTYPES))
    parser.add_argument('--compare', default='',
                        help='comma-separated dtypes measured interleaved in one process')
    parser.add_argument('--ablate', action='append', default=[], choices=['baseline', *ABLATIONS],
                        help='measure one patched cost (bf16 dtype only)')
    parser.add_argument('--prompt', type=int, default=128)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--mode', default='auto',
                        help="optimize_inference mode; 'auto' matches scripts/web_demo.py, "
                             "'full' compiles the trunk (~50 s) for a faster prefill")
    parser.add_argument('--trace-moe', action='store_true',
                        help='drop the torch.compiler.disable barrier on the grouped expert forward '
                             'before tracing (the barrier guards a training-only TMA issue)')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--config', default=str(CONFIG), help='model config json (default: MoE)')
    parser.add_argument('--weight', default=str(WEIGHT), help='weights .pth (default: MoE)')
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.trace_moe:
        import model.moe_dispatch as dispatch

        dispatch._grouped_expert_forward = dispatch._grouped_expert_forward.__wrapped__
        print('[trace-moe] compiler.disable barrier removed; the MoE block will be traced')
    names = args.compare.split(',') if args.compare else [args.dtype]
    variants = []
    for name in names:
        variants.append((name, 'baseline'))
        variants += [(name, ablation) for ablation in args.ablate if ablation != 'baseline']
    variants.append((names[0], 'baseline'))  # re-measure the start to expose drift

    models = {}
    for name, _ in variants:
        if name not in models:
            models[name] = build_model(DTYPES[name], config_path=args.config,
                                       weight_path=args.weight, mode=args.mode)
            report_forward_path(models[name])
            print(f"[model] {name}: {sum(p.numel() for p in models[name].parameters()) / 1e6:.1f}M "
                  f"params, {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB allocated")

    # Round-robin over variants and take the median per variant: this machine
    # loses ~1.7x between the first and last measurement of one process, so a
    # block-per-variant ordering would report drift as a speedup.
    samples = {key: [] for key in variants}
    for _ in range(args.rounds):
        for key in variants:
            name, ablation = key
            model = models[name]
            with Ablation(model, ablation):
                decode, prefill = measure(model, args.prompt, args.tokens, 1)
            samples[key].append((decode, prefill))

    measurements = {
        key: (statistics.median(item[0] for item in values),
              statistics.median(item[1] for item in values))
        for key, values in samples.items()
    }
    for key in variants:
        name, ablation = key
        decode, prefill = measurements[key]
        report(f'{name} {ablation}', decode, prefill, args.prompt, args.tokens)

    baseline = {}
    for (name, ablation) in variants:
        if ablation == 'baseline':
            baseline.setdefault(name, measurements[(name, ablation)][0])
    for (name, ablation) in variants:
        if ablation != 'baseline':
            decode = measurements[(name, ablation)][0]
            print(f"[ablate] {name} {ablation:24s}: {baseline[name] / decode:.2f}x vs baseline")
    if variants[0] == variants[-1]:
        first, last = measurements[variants[0]][0], measurements[variants[-1]][0]
        print(f"[drift] {variants[0][0]} baseline re-measured: {first * 1e3:.1f} -> "
              f"{last * 1e3:.1f} ms/token ({last / first:.2f}x)")

    if args.profile:
        name = args.dtype if args.dtype in models else next(iter(models))
        model = models[name]
        try:
            profile_generate(model, args.prompt, min(args.tokens, 32))
        except Exception as exc:  # profiler is best-effort diagnostics
            print(f'[profile] unavailable: {exc}')
    count_host_syncs(models[args.dtype if args.dtype in models else next(iter(models))],
                     args.prompt, 4)


if __name__ == '__main__':
    main()
