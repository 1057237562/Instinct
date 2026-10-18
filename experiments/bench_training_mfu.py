"""Full synthetic training-step benchmark: forward, backward, clipping, optimizer.

Uses fresh random weights and token IDs; never reads/writes training corpora or
checkpoints. Fixed packed documents isolate compute from the real loader. Each
invocation should run in a separate process to release optimizer/compiler memory.
MFU is an explicitly labelled 6*N_active estimate plus useful causal attention;
recomputation and optimizer FLOPs are excluded. Supply a dense precision-specific
peak, never the advertised sparse FP4 AI TOPS. Compare wall tokens/s first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trainer.compile_cache  # noqa: F401 -- before torch
import datasets  # noqa: F401 -- Windows DLL import order
import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from trainer.trainer_cli import step_with_scaler
from trainer.trainer_utils import apply_torchao_fp8_training, build_optimizer


def gram_candidate(update, *, ns_coefficients, ns_steps, eps, dtype):
    """Gram NS, restarting every 3 iterations; omit unused Gram/identity GEMMs.

    Experimental only. Square/tiny shapes keep standard NS. Validate against the
    independently written Gram and standard references before using this path.
    """
    from tests.test_gram_newton_schulz import standard_zeropower
    if update.size(-2) == update.size(-1) or min(update.shape[-2:]) < 32 or ns_steps == 0:
        return standard_zeropower(update, ns_coefficients=ns_coefficients,
                                  ns_steps=ns_steps, eps=eps, dtype=dtype)
    x = update.to(dtype)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    x = x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp(min=eps)
    a, b, c = ns_coefficients
    eye = torch.eye(x.size(-2), device=x.device, dtype=x.dtype)
    gram = torch.bmm(x, x.transpose(-2, -1))
    q = None
    for i in range(ns_steps):
        z = torch.baddbmm(gram, gram, gram, beta=b, alpha=c) + a * eye
        q = z if q is None else torch.bmm(q, z)
        if i + 1 == ns_steps:
            break
        if (i + 1) % 3 == 0:
            x = torch.bmm(q, x)
            gram = torch.bmm(x, x.transpose(-2, -1))
            q = None
        else:
            gram = torch.bmm(torch.bmm(z, gram), z)
    x = torch.bmm(q, x)
    return x.transpose(-2, -1) if transposed else x


def install_experimental_ns(method, dtype):
    """Process-local candidate only; standard BF16 leaves production untouched.

    Stacked inputs retain production's BF16 cast. Changing that cast is a
    separate precision experiment and is deliberately excluded here.
    """
    if method == 'standard' and dtype == 'bf16':
        return
    import trainer.batched_muon as module
    from tests.test_gram_newton_schulz import standard_zeropower
    fn = gram_candidate if method == 'gram' else standard_zeropower

    def candidate(update, **options):
        return fn(update, dtype=torch.float16 if dtype == 'fp16' else torch.bfloat16,
                  **options)

    module._batched_zeropower = candidate


def install_cast_before_stack():
    """Experimental exact-value cast before packing expert weights.

    Avoids a temporary FP32 stack; each element is still rounded once to BF16.
    Gradients must be checked because autograd now casts each expert separately.
    """
    import model.moe_dispatch as dispatch
    original = dispatch._stacked_expert_weights

    def candidate(experts, name, dtype):
        if getattr(experts, '_inference_stack_cache', None) is not None:
            return original(experts, name, dtype)
        return torch.stack([getattr(expert, name).weight.to(dtype=dtype)
                            for expert in experts], dim=0).transpose(1, 2)

    dispatch._stacked_expert_weights = candidate


def install_traceable_training_moe():
    """Experimental: remove the training barrier only with explicit Triton GEMMs.

    Never use native grouped-MM's Blackwell weight-gradient lowering here.
    This is deliberately process-local until compilation and numerical gates pass.
    """
    import model.moe_dispatch as dispatch
    os.environ['INSTINCT_GROUPED_MM_BACKEND'] = 'triton'
    dispatch._grouped_expert_forward = dispatch._grouped_expert_forward.__wrapped__


def useful_flops(config, total_params, batch, seq, segments):
    # Count inactive expert matrices out of the 6N approximation (tied head once).
    expert = (config.num_hidden_layers * config.num_experts * 3
              * config.hidden_size * config.moe_intermediate_size) if config.use_moe else 0
    active = total_params - expert + expert * config.num_experts_per_tok / config.num_experts if expert else total_params
    boundaries = [i * seq // segments for i in range(segments + 1)]
    pairs = batch * sum((b - a) * (b - a + 1) // 2
                        for a, b in zip(boundaries, boundaries[1:]))
    attention = 12 * config.num_hidden_layers * config.num_attention_heads * config.head_dim * pairs
    return active, 6 * active * batch * seq + attention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'trainer/config_instinct_v1_moe.json'))
    p.add_argument('--layers', type=int, default=32)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--seq', type=int, default=2048)
    p.add_argument('--segments', type=int, default=4)
    p.add_argument('--param-dtype', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--checkpoint', type=int, choices=[0, 1, 2], default=1)
    p.add_argument('--selective-checkpoint', choices=['0', '1', '2', '3', 'down', 'gate_down'],
                   help='Cache selected expert GEMMs while retaining FFN checkpointing')
    p.add_argument('--attention', choices=['auto', 'sdpa', 'flex'], default='auto')
    p.add_argument('--grouped', choices=['auto', 'cached', 'native', 'triton'], default='auto')
    p.add_argument('--fp8', choices=['off', 'tensorwise'], default='off')
    p.add_argument('--compile-mode', choices=['off', 'default', 'reduce-overhead', 'max-autotune-no-cudagraphs'], default='off')
    p.add_argument('--ns-dtype', choices=['bf16', 'fp16'], default='bf16')
    p.add_argument('--ns-backend', choices=['torch', 'triton', 'triton_graph', 'auto'])
    p.add_argument('--fused-updates', type=int, choices=[0,1])
    p.add_argument('--fused-pack', type=int, choices=[0,1])
    p.add_argument('--fused-swiglu', type=int, choices=[0,1])
    p.add_argument('--ns-method', choices=['standard', 'gram'], default='standard')
    p.add_argument('--compile-ns', action='store_true', help='Compile only the process-local NS function')
    p.add_argument('--cast-before-stack', action='store_true', help='Avoid temporary FP32 expert stacks')
    p.add_argument('--compile-moe', action='store_true', help='Experimental traceable Triton training dispatch')
    p.add_argument('--workspace-mb', type=float, default=64)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--peak-tflops', type=float, help='Dense peak, explicit reference precision')
    p.add_argument('--peak-label', default='BF16/FP32-accumulate dense')
    p.add_argument('--output', required=True)
    p.add_argument('--trace', action='store_true', help='Extra diagnostic step after timing')
    args = p.parse_args()
    if args.selective_checkpoint is not None:
        os.environ['INSTINCT_MOE_SELECTIVE_CHECKPOINT'] = args.selective_checkpoint
    if min(args.layers, args.batch, args.seq, args.segments, args.warmup, args.steps, args.repeats) < 1:
        p.error('shape, warmup and measurement counts must be positive')
    if args.segments > args.seq or args.peak_tflops is not None and args.peak_tflops <= 0:
        p.error('segments <= seq and peak > 0 required')
    if args.ns_method != 'standard' or args.ns_dtype != 'bf16' or args.compile_ns:
        if args.ns_backend not in (None, 'torch'):
            p.error('experimental NS replacements require --ns-backend torch')
        # The optimizer-owned production graph intentionally calls the fixed
        # production transform; do not silently bypass an experimental patch.
        args.ns_backend = 'torch'
    os.environ['INSTINCT_PACKED_ATTENTION_BACKEND'] = args.attention
    os.environ['INSTINCT_GROUPED_MM_BACKEND'] = args.grouped
    os.environ['INSTINCT_MUON_WORKSPACE_MB'] = str(args.workspace_mb)
    if args.ns_backend is not None:
        os.environ['INSTINCT_MUON_NS_BACKEND'] = args.ns_backend
    args.ns_backend = os.environ.get('INSTINCT_MUON_NS_BACKEND', 'auto')
    if args.fused_updates is not None:
        os.environ['INSTINCT_MUON_FUSED_UPDATES'] = str(args.fused_updates)
    args.fused_updates = os.environ.get('INSTINCT_MUON_FUSED_UPDATES', 'auto')
    if args.fused_pack is not None:
        os.environ['INSTINCT_MOE_FUSED_PACK'] = str(args.fused_pack)
    args.fused_pack = os.environ.get('INSTINCT_MOE_FUSED_PACK', 'auto')
    if args.fused_swiglu is not None:
        os.environ['INSTINCT_MOE_FUSED_SWIGLU'] = str(args.fused_swiglu)
    args.fused_swiglu = os.environ.get('INSTINCT_MOE_FUSED_SWIGLU', 'auto')
    install_experimental_ns(args.ns_method, args.ns_dtype)
    if args.cast_before_stack:
        install_cast_before_stack()
    if args.compile_moe:
        if args.compile_mode == 'off':
            p.error('--compile-moe requires model compilation')
        install_traceable_training_moe()
    if args.compile_ns:
        import trainer.batched_muon as muon_module
        muon_module._batched_zeropower = torch.compile(
            muon_module._batched_zeropower, fullgraph=True, mode='reduce-overhead')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    raw = json.loads(Path(args.config).read_text(encoding='utf-8'))
    raw.update(num_hidden_layers=args.layers, use_grad_checkpoint=args.checkpoint,
               norm_topk_prob=True, param_dtype=args.param_dtype)
    config = InstinctConfig(**raw)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'args': vars(args), 'config': raw, 'status': 'running',
              'environment': {key: value for key, value in os.environ.items()
                              if key.startswith(('INSTINCT_MOE_', 'INSTINCT_MUON_',
                                                 'INSTINCT_CAUSAL_ATTENTION_', 'INSTINCT_FLEX_'))},
              'scope': 'synthetic full compute step; excludes loader, LR schedule, logs, saving',
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(), 'seed': args.seed,
              'git_head': subprocess.check_output(['git', '--git-dir', str(ROOT / '.git'),
                                                   'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip(),
              'input_config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ['trainer/batched_muon.py', 'trainer/trainer_utils.py',
                                             'model/model_instinct.py', 'model/grouped_mm_triton.py',
                                             'model/moe_dispatch.py', 'model/packed_attention.py',
                                             'model/checkpointing.py', 'experiments/bench_training_mfu.py',
                                             'tests/test_gram_newton_schulz.py', 'trainer/muon_kernels.py',
                                             'model/expert_weight_pack.py', 'model/swiglu_triton.py',
                                             'model/cuda_fastpath.py', 'model/checkpoint_swiglu_triton.py',
                                             'model/expert_linear_triton.py']}}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        model = InstinctForCausalLM(config).cuda().train()
        model.to(dtype=torch.float32 if args.param_dtype == 'fp32' else torch.bfloat16)
        params = sum(x.numel() for x in model.parameters())
        optimizer = build_optimizer(model.named_parameters(), lr=5e-4, optimizer='muon')
        model = apply_torchao_fp8_training(model, SimpleNamespace(
            fp8_training=args.fp8, fp8_filter='auto', dtype='bfloat16', device='cuda',
            use_compile=int(args.compile_mode != 'off')))
        if args.compile_mode != 'off':
            model = torch.compile(model, mode=args.compile_mode)
        data = torch.randint(3, config.vocab_size, (args.batch, args.seq), device='cuda')
        labels = data.clone()
        sequence_ids = torch.empty_like(data)
        for i in range(args.segments):
            start, end = i * args.seq // args.segments, (i + 1) * args.seq // args.segments
            sequence_ids[:, start:end] = i
            # Production loss shifts labels; label at each document start must be ignored.
            labels[:, start] = -100
        if args.segments == 1:
            sequence_ids = None  # ordinary fused causal attention fast path
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        optimizer.zero_grad(set_to_none=True)

        def step(markers=None):
            if args.compile_mode != 'off':
                torch.compiler.cudagraph_mark_step_begin()
            if markers is not None:
                markers[0].record()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(data, labels=labels, sequence_ids=sequence_ids)
                loss = out.loss + out.aux_loss
                del out  # Match the production trainer's output lifetime.
            if markers is not None:
                markers[1].record()
            loss.backward()
            if markers is not None:
                markers[2].record()
            step_with_scaler(scaler, optimizer, model.parameters(), 1.0)
            if markers is not None:
                markers[3].record()
            return loss.detach()

        print('WARMUP', vars(args), flush=True)
        start = time.perf_counter()
        first = None
        for i in range(args.warmup):
            loss = step()
            if i == 0:
                first = float(loss)
            if (i + 1) % 5 == 0:
                print(f'warmup {i + 1}/{args.warmup}', flush=True)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        windows, phase_samples = [], []
        for repeat in range(args.repeats):
            events = [[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                      for _ in range(args.steps)]
            torch.cuda.synchronize()
            start = time.perf_counter()
            for markers in events:
                loss = step(markers)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - start) * 1000 / args.steps
            windows.append(wall_ms)
            phase_samples.extend([[m[0].elapsed_time(m[1]), m[1].elapsed_time(m[2]),
                                   m[2].elapsed_time(m[3])] for m in events])
            print(f'window {repeat + 1}: {wall_ms:.2f} ms/step', flush=True)
        active, flops = useful_flops(config, params, args.batch, args.seq, args.segments)
        wall_ms = statistics.median(windows)
        result.update(status='ok', params=params, active_params_estimate=active,
                      useful_flops_per_step_estimate=flops, warmup_seconds=warmup_seconds,
                      window_wall_ms=windows, wall_ms=wall_ms,
                      tokens_per_second=args.batch * args.seq / (wall_ms / 1000),
                      loss_first=first, loss_last=float(loss),
                      phase_mean_ms=dict(zip(['forward', 'backward', 'optimizer'],
                                            [statistics.mean(row[i] for row in phase_samples) for i in range(3)])),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      mfu_estimate_pct=(100 * flops / (wall_ms / 1000) / (args.peak_tflops * 1e12)
                                        if args.peak_tflops else None))
        if args.trace:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                step()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(output.with_suffix('.trace.json')))
            output.with_suffix('.operators.txt').write_text(
                prof.key_averages().table(sort_by='self_device_time_total', row_limit=40), encoding='utf-8')
        print(json.dumps({k: result[k] for k in ['wall_ms', 'tokens_per_second', 'phase_mean_ms',
                                               'peak_allocated_gib', 'mfu_estimate_pct']}, indent=2), flush=True)
    except Exception as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
