"""Audit checkpoint forward/backward cost without updating any model weights.

CPU: count executed matrix FLOPs on a small slice. CUDA: time a fixed-shape
forward+backward window and measure allocator peaks. Optional resident-state
mode reserves real Muon/Adam state and existing gradients without updates.
There is no data loader or training cursor, so this is not whole-training MFU. Run each
mode in a fresh process with the same arguments and no concurrent GPU work.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trainer.compile_cache  # noqa: F401 -- before torch
import datasets  # noqa: F401 -- Windows DLL ordering
import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from trainer.trainer_utils import cast_training_parameters, apply_torchao_fp8_training, build_optimizer


def reserve_training_state(model):
    """Initialize real optimizer state/grad buffers without taking any update."""
    optimizer = build_optimizer(model.named_parameters(), lr=5e-4, optimizer='muon')
    for opt in getattr(optimizer, 'optimizers', [optimizer]):
        if hasattr(opt, 'reserve_momentum'):
            opt.reserve_momentum()
        elif isinstance(opt, torch.optim.AdamW):
            for group in opt.param_groups:
                for p in group['params']:
                    opt.state[p].update(step=torch.tensor(0.), exp_avg=torch.zeros_like(p),
                                        exp_avg_sq=torch.zeros_like(p))
        else:
            raise ValueError('resident probe requires the batched Muon backend')
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    return optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'trainer/config_instinct_v2.json'))
    parser.add_argument('--mode', type=int, choices=[0, 1, 2, 3, 4], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seq', type=int, default=128)
    parser.add_argument('--segments', type=int, default=1)
    parser.add_argument('--param-dtype', choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compile-mode', default='default', choices=['default', 'max-autotune-no-cudagraphs'])
    parser.add_argument('--resident-state', action='store_true')
    parser.add_argument('--allocator-limit-gib', type=float, default=0.)
    parser.add_argument('--workspace-reserve-mb', type=int, default=0)
    parser.add_argument('--mode4-policy', choices=['recipe', 'kv_gate', 'attention_gate', 'projections'], default='recipe')
    parser.add_argument('--mode4-gate-layers', type=int)
    parser.add_argument('--fp8', choices=['off', 'tensorwise'], default='off')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=307)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.mode == 4:
        from functools import partial
        import model.model_instinct as model_module
        from model.checkpointing import dense_checkpoint_options
        model_module.dense_checkpoint_options = partial(dense_checkpoint_options,
            mode4_policy=args.mode4_policy, mode4_gate_layers=args.mode4_gate_layers)
    if min(args.layers, args.batch, args.seq, args.segments, args.warmup, args.steps, args.repeats) < 1:
        parser.error('dimensions and measurement counts must be positive')
    if args.segments > args.seq or args.seq < 2:
        parser.error('require 1 <= segments <= seq and seq >= 2')
    if args.fp8 != 'off' and (args.device != 'cuda' or args.dtype != 'bf16' or not args.compile):
        parser.error('FP8 requires CUDA, BF16 activations and --compile')
    if args.allocator_limit_gib < 0 or args.workspace_reserve_mb < 0:
        parser.error('memory limits must be nonnegative')
    if args.allocator_limit_gib:
        if args.device != 'cuda':
            parser.error('allocator limit requires CUDA')
        torch.cuda.set_per_process_memory_fraction(min(1., args.allocator_limit_gib * 2**30 /
            torch.cuda.get_device_properties(0).total_memory))
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    raw = json.loads(Path(args.config).read_text(encoding='utf-8'))
    raw.update(num_hidden_layers=args.layers, use_grad_checkpoint=args.mode,
               param_dtype=args.param_dtype)
    config = InstinctConfig(**raw)
    model = cast_training_parameters(InstinctForCausalLM(config), args.param_dtype)
    model = model.to(args.device).train()
    optimizer = reserve_training_state(model) if args.resident_state else None
    workspace = torch.empty(args.workspace_reserve_mb * 2**20, dtype=torch.uint8, device=args.device)
    model = apply_torchao_fp8_training(model, SimpleNamespace(
        fp8_training=args.fp8, fp8_filter='auto', dtype='bfloat16',
        device=args.device, use_compile=int(args.compile)))
    if args.resident_state:
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    resident_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    resident_bytes += sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
    if optimizer is not None:
        resident_bytes += sum(t.numel() * t.element_size()
            for opt in getattr(optimizer, 'optimizers', [optimizer])
            for state in opt.state.values() for t in state.values()
            if isinstance(t, torch.Tensor) and t.device.type == args.device)
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, dynamic=False)
    tokens = torch.randint(3, config.vocab_size, (args.batch, args.seq), device=args.device)
    labels = tokens.clone()
    ids = None
    if args.segments > 1:
        ids = torch.empty_like(tokens)
        for i in range(args.segments):
            start, end = i * args.seq // args.segments, (i + 1) * args.seq // args.segments
            ids[:, start:end] = i
            labels[:, start] = -100

    def step():
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        # With existing gradients, the next accumulation forward overlaps all
        # gradient storage. Clear values in place, preserving that live storage.
        model.zero_grad(set_to_none=not args.resident_state)
        with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.dtype == 'bf16'):
            out = model(tokens, labels=labels, sequence_ids=ids)
            loss = out.loss
            del out
        loss.backward()
        return loss.detach()

    report = dict(status='running', args=vars(args), config=config.to_dict(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  scope='fixed weights, synthetic forward+backward; no optimizer updates, loader or saves; resident state optional',
                  resident_optimizer=args.resident_state,
                  resident_tensor_gib=resident_bytes / 2**30,
                  environment={key: value for key, value in os.environ.items()
                               if key.startswith(('INSTINCT_', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'))})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        if args.device == 'cpu':
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as counter:
                loss = step()
            report['executed_matrix_flops'] = {
                str(op): count for op, count in counter.get_flop_counts()['Global'].items()}
        else:
            report['gpu'] = torch.cuda.get_device_name()
            start = time.perf_counter()
            for i in range(args.warmup):
                loss = step()
                print(f'warmup {i + 1}/{args.warmup}', flush=True)
            torch.cuda.synchronize()
            report['warmup_seconds'] = time.perf_counter() - start
            torch.cuda.reset_peak_memory_stats()
            windows = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.steps):
                    loss = step()
                torch.cuda.synchronize()
                windows.append((time.perf_counter() - start) * 1000 / args.steps)
            report.update(window_ms=windows, median_ms=statistics.median(windows),
                          input_tokens_per_second=args.batch * args.seq * 1000 / statistics.median(windows),
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        report.update(status='ok', loss=float(loss))
    except Exception as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('config', 'environment')}, indent=2))


if __name__ == '__main__':
    main()
