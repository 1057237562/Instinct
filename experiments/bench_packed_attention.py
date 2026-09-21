"""Compare packed SDPA and FlexAttention, including GQA backward.

Run with training paused: python experiments/bench_packed_attention.py
No dataset/checkpoint is read or changed. This is not whole-model throughput.
"""

import argparse
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: F401 -- Windows DLL import order
import trainer.compile_cache  # noqa: F401 -- before torch
import torch

from model.attention_mask import prepare_sdpa_attention_bias
from model.flash_attn_4 import flash_attention
from model.packed_attention import build_packed_flex_mask, packed_flex_attention
from model.sequence_packing import merge_packed_attention_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, default=2928)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--segments', type=int, default=4)
    parser.add_argument('--steps', type=int, default=12)
    args = parser.parse_args()
    if min(args.length, args.batch, args.segments, args.steps) < 1:
        parser.error('all sizes must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(5)
    ids = (torch.arange(args.length, device='cuda') * args.segments // args.length)
    ids = ids[None].expand(args.batch, -1).contiguous()
    q = torch.randn(args.batch, args.length, 16, 32, device='cuda', dtype=torch.bfloat16,
                    requires_grad=True)
    k = torch.randn(args.batch, args.length, 4, 32, device='cuda', dtype=torch.bfloat16,
                    requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    grad = torch.randn_like(q)
    dense = prepare_sdpa_attention_bias(merge_packed_attention_mask(ids, None), q,
                                       query_length=args.length)
    blocked = build_packed_flex_mask(ids)
    backends = {
        'sdpa': lambda: flash_attention(q, k, v, attention_mask=dense),
        'flex': lambda: packed_flex_attention(q, k, v, blocked),
    }
    results = {}
    for name, fn in backends.items():
        y = fn()
        grads = torch.autograd.grad(y, (q, k, v), grad)
        results[name] = (y.detach(), [g.detach() for g in grads])
    for label, a, b in [('output', results['flex'][0], results['sdpa'][0]),
                        *[(n, a, b) for n, a, b in zip(('dQ', 'dK', 'dV'),
                          results['flex'][1], results['sdpa'][1])]]:
        relative = ((a.float() - b.float()).norm() / b.float().norm()).item()
        print(f'{label}: relative_L2_error={relative:.6f}', flush=True)
        if relative > 0.02:
            raise AssertionError(f'{label} relative error exceeded 2%')
    del results, y, grads, a, b
    for fn in backends.values():
        for _ in range(3):
            torch.autograd.grad(fn(), (q, k, v), grad)
    torch.cuda.synchronize()
    samples = {name: [] for name in backends}
    for i in range(args.steps):
        order = list(backends) if i % 2 == 0 else list(reversed(backends))
        for name in order:
            start = time.perf_counter()
            torch.autograd.grad(backends[name](), (q, k, v), grad)
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
    for name in backends:
        print(f'{name}: median_forward_backward_ms={statistics.median(samples[name]):.3f}', flush=True)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.autograd.grad(backends[name](), (q, k, v), grad)
        torch.cuda.synchronize()
        print(f'{name}: extra_peak_mib={(torch.cuda.max_memory_allocated() - baseline) / 2**20:.3f}', flush=True)
    # Metadata is shared across layers. Report its warm per-batch cost separately.
    build_packed_flex_mask(ids)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        build_packed_flex_mask(ids)
    torch.cuda.synchronize()
    print(f'block_mask_build_ms={(time.perf_counter() - start) * 200:.3f}', flush=True)


if __name__ == '__main__':
    main()
