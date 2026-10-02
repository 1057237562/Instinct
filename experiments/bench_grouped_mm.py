"""Short MoE dispatch benchmark without datasets or checkpoint I/O.

Run while training is stopped: python experiments/bench_grouped_mm.py --profile
Compares identical BF16 routed FFNs, including first-order backward.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: F401 -- Windows DLL order
import trainer.compile_cache  # noqa: F401 -- before torch
import torch

from model.model_instinct import InstinctConfig, MOEFeedForward
from model.checkpointing import checkpoint_ffn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=11712)
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--checkpoint', action='store_true', help='Replay FFN as in training mode 1')
    args = parser.parse_args()
    if min(args.tokens, args.steps, args.warmup) < 1:
        parser.error('tokens, steps and warmup must be positive')
    torch.set_num_threads(1)
    rows = []
    original_backend = os.environ.get('INSTINCT_GROUPED_MM_BACKEND')
    try:
        backends = ('native', 'cached', 'triton')
        torch.manual_seed(7)
        cfg = InstinctConfig(hidden_size=512, num_experts=8, num_experts_per_tok=1,
                             moe_intermediate_size=1664, use_moe=True)
        model = MOEFeedForward(cfg).cuda().train()
        x = torch.randn(1, args.tokens, 512, device='cuda', requires_grad=True)

        def step():
            model.zero_grad(set_to_none=True)
            x.grad = None
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if args.checkpoint:
                    output, aux = checkpoint_ffn(model, x)
                else:
                    output = model(x)
                    aux = model.aux_loss
                loss = output.float().square().mean() + aux
            loss.backward()

        for backend in backends:
            os.environ['INSTINCT_GROUPED_MM_BACKEND'] = backend
            for _ in range(args.warmup):
                step()
        torch.cuda.synchronize()
        timings = {backend: [] for backend in backends}
        # Alternate order to reduce bias from changing clocks/background load.
        for iteration in range(args.steps):
            shift = iteration % len(backends)
            for backend in backends[shift:] + backends[:shift]:
                os.environ['INSTINCT_GROUPED_MM_BACKEND'] = backend
                start = time.perf_counter()
                step()
                torch.cuda.synchronize()
                timings[backend].append(1000 * (time.perf_counter() - start))
        for backend in backends:
            os.environ['INSTINCT_GROUPED_MM_BACKEND'] = backend
            model.zero_grad(set_to_none=True)
            x.grad = None
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            step()
            torch.cuda.synchronize()
            row = dict(backend=backend, checkpoint=args.checkpoint,
                       step_ms=statistics.median(timings[backend]),
                       mean_ms=statistics.mean(timings[backend]),
                       peak_above_inputs_mib=(torch.cuda.max_memory_allocated() - baseline) / 2**20)
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    step()
                    torch.cuda.synchronize()
                keys = {e.key: e.count for e in prof.key_averages()}
                row['gpu_to_host_copies'] = sum(e.count for e in prof.key_averages()
                                               if 'Memcpy DtoH' in e.key)
                row['native_grouped_mm_calls'] = keys.get('aten::_grouped_mm', 0)
                row['kernel_launch_calls'] = sum(keys.get(k, 0) for k in (
                    'cudaLaunchKernel', 'cuLaunchKernel', 'cuLaunchKernelEx', 'cudaLaunchKernelExC'))
            rows.append(row)
    finally:
        if original_backend is None:
            os.environ.pop('INSTINCT_GROUPED_MM_BACKEND', None)
        else:
            os.environ['INSTINCT_GROUPED_MM_BACKEND'] = original_backend
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
