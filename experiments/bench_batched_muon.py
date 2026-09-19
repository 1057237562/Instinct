"""Isolated optimizer benchmark; no datasets/checkpoints or model training.

Run from the repo root: python experiments/bench_batched_muon.py --device cuda
"""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: F401 -- Windows DLL import order
import trainer.compile_cache  # noqa: F401 -- before torch
import torch

from trainer.batched_muon import BatchedMuon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--matrices', type=int, default=32)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workspace-mb', type=float, default=64)
    parser.add_argument('--profile', action='store_true', help='Count launches in one extra step')
    args = parser.parse_args()
    if args.matrices < 1 or args.steps < 1 or args.warmup < 1:
        parser.error('matrices, steps and warmup must be positive')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if not hasattr(torch.optim, 'Muon'):
        parser.error('This benchmark needs native torch.optim.Muon as a reference')

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    results = []
    for backend in ('native', 'batched'):
        torch.manual_seed(42)
        # Equal numbers of tall/wide expert matrices and square attention weights.
        shapes = [(1664, 512), (512, 1664), (512, 512)]
        params = [torch.nn.Parameter(torch.randn(shapes[i % 3], device=device))
                  for i in range(args.matrices)]
        for p in params:
            p.grad = torch.randn_like(p)
        kwargs = dict(lr=5e-4, adjust_lr_fn='match_rms_adamw')
        opt = (torch.optim.Muon(params, **kwargs) if backend == 'native' else
               BatchedMuon(params, batch_size=args.batch_size,
                           workspace_mb=args.workspace_mb, **kwargs))
        for _ in range(args.warmup):
            opt.step()
        sync()
        if device.type == 'cuda':
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            opt.step()
        sync()
        row = dict(backend=backend, matrices=args.matrices,
                   step_ms=1000 * (time.perf_counter() - start) / args.steps)
        if device.type == 'cuda':
            row['temporary_peak_mib'] = (torch.cuda.max_memory_allocated(device) - baseline) / 2**20
        if backend == 'batched':
            row['chunk_sizes'] = [opt._chunk_size(p) for p in params[:3]]
        if args.profile and device.type == 'cuda':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                opt.step()
                sync()
            row['kernel_launch_calls'] = sum(
                event.count for event in prof.key_averages()
                if event.key in ('cudaLaunchKernel', 'cuLaunchKernel',
                                 'cudaLaunchKernelExC', 'cuLaunchKernelEx')
            )
        results.append(row)
        del opt, params, p
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(json.dumps(dict(results=results,
                          speedup=results[0]['step_ms'] / results[1]['step_ms']), indent=2))


if __name__ == '__main__':
    main()
