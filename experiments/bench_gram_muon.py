"""Benchmark standard vs Gram Newton-Schulz on the real BatchedMuon buckets.

Run with training paused: python experiments/bench_gram_muon.py [--iters N]
No dataset/checkpoint is read or changed. This is not whole-model throughput.

Shapes mirror run pretrain_20260919_174858 (hidden 512, 32 layers, 8 experts
top-1): per layer the Muon buckets are 24 expert matrices [512, 1664], q/o
[512, 512] x2, k/v [128, 512] x2, router [8, 512]. The optimizer groups all 32
layers per shape, so the benchmark measures each full-bucket stack once —
exactly what `BatchedMuon.step` would execute per training step.

The numerics section stresses half-precision Gram on ill-conditioned inputs
(low rank, duplicated rows, wild scales) against the fp64 iteration, to decide
whether the stabilization restart can be dropped for speed.
"""

import argparse
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: F401 -- Windows DLL import order
import trainer.compile_cache  # noqa: F401 -- before torch
import torch

from tests.test_gram_newton_schulz import gram_zeropower, standard_zeropower
from trainer.batched_muon import _NS_COEFFICIENTS, _batched_zeropower

# (count, shape, per-layer multiplicity) for hidden=512, L=32, E=8, top-1.
BUCKETS = [
    (24 * 32, (512, 1664), "experts gate/up/down"),
    (2 * 32, (512, 512), "q_proj + o_proj"),
    (2 * 32, (128, 512), "k_proj + v_proj"),
    (1 * 32, (8, 512), "routers"),
]

STEP_MS = 600.0          # stable-window step time of run pretrain_20260919_174858
TOKENS_PER_S = 17947.0   # same window


def bench(fn, *, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def bucket_flops(count, shape, variant):
    """Absolute NS matmul FLOPs; restart policy must match the variant."""
    m, n = min(shape), max(shape)
    alpha = max(shape) / min(shape)
    if variant.startswith("standard"):
        return 5 * (4 * alpha + 2) * m ** 3 * count
    per_iter = 8 * m ** 3                      # A@A + aI fold + Q@Z + Z@A@Z
    rectangular = 2 * (2 * alpha * m ** 3)     # init XX^T + final Q@X
    restarts = 0 if "r0" in variant else (2 if "r2" in variant else 1)
    restart_cost = restarts * 2 * (2 * alpha * m ** 3)
    return 5 * per_iter * count + (rectangular + restart_cost) * count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--numerics-only", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "needs CUDA"
    torch.manual_seed(0)
    device = "cuda"
    print(f"device: {torch.cuda.get_device_name(device)}\n")

    # ---- numerics: can fp16 Gram drop the restart? ----------------------
    print("numerics vs fp64 standard iteration (rel err, small batches)")
    stressors = {
        "gaussian": lambda: torch.randn(8, 64, 200, dtype=torch.float64, device=device),
        "low-rank r1": lambda: (torch.randn(8, 64, 1, dtype=torch.float64, device=device)
                                @ torch.randn(8, 1, 200, dtype=torch.float64, device=device)),
        "low-rank r8": lambda: (torch.randn(8, 64, 8, dtype=torch.float64, device=device)
                                @ torch.randn(8, 8, 200, dtype=torch.float64, device=device)),
        "dup rows": lambda: torch.randn(8, 1, 200, dtype=torch.float64, device=device).repeat(1, 64, 1),
        "row scales 1e-3..1e3": lambda: torch.randn(8, 64, 200, dtype=torch.float64, device=device)
                                        * torch.logspace(-3, 3, 64, device=device).view(1, -1, 1),
    }
    for name, make in stressors.items():
        update = make()
        ref = standard_zeropower(update, ns_steps=5)

        def rel(x):
            d = torch.linalg.vector_norm((x.double() - ref).flatten())
            n = torch.linalg.vector_norm(ref.flatten())
            return (d / n.clamp(min=1e-30)).item()

        std_bf16 = rel(standard_zeropower(update, ns_steps=5, dtype=torch.bfloat16))
        g_fp16_r0 = rel(gram_zeropower(update, ns_steps=5, dtype=torch.float16))
        g_fp16_r2 = rel(gram_zeropower(update, ns_steps=5, dtype=torch.float16, restart_after=2))
        bad = " <-- fp16 no-restart worse than 3x production error" \
            if not (g_fp16_r0 < 3 * max(std_bf16, 1e-4)) else ""
        print(f"  {name:<22} std-bf16 {std_bf16:8.2e}   gram-fp16-r0 {g_fp16_r0:8.2e}"
              f"   gram-fp16-r2 {g_fp16_r2:8.2e}{bad}")
    if args.numerics_only:
        return

    # ---- speed ------------------------------------------------------------
    variants = {
        "standard bf16 (production)": lambda u: _batched_zeropower(
            u, ns_coefficients=_NS_COEFFICIENTS, ns_steps=5, eps=1e-7),
        "standard fp16": lambda u: standard_zeropower(
            u, ns_steps=5, dtype=torch.float16),
        "gram bf16 r2": lambda u: gram_zeropower(
            u, ns_steps=5, dtype=torch.bfloat16, restart_after=2),
        "gram fp16 r2": lambda u: gram_zeropower(
            u, ns_steps=5, dtype=torch.float16, restart_after=2),
        "gram fp16 r0": lambda u: gram_zeropower(
            u, ns_steps=5, dtype=torch.float16),
    }

    totals = {name: 0.0 for name in variants}
    flops = {name: 0.0 for name in variants}
    header = (f"{'bucket':<28}{'ms':>8}")
    for name in variants:
        header += f"{name.split(' (')[0]:>16}"
    print("\n" + header)
    for count, shape, label in BUCKETS:
        update = torch.randn(count, *shape, device=device)
        row = f"{label + ' ' + str(shape):<28}"
        for name, fn in variants.items():
            med = bench(lambda: fn(update), warmup=args.warmup, iters=args.iters)
            totals[name] += med
            flops[name] += bucket_flops(count, shape, name)
            row += f"{med:>16.2f}"
        print(row + "   (ms/std column is the production baseline)")

    prod = totals["standard bf16 (production)"]
    print(f"\nNS totals per training step (production baseline {prod:.1f} ms):")
    for name, total in totals.items():
        if name == "standard bf16 (production)":
            continue
        step = STEP_MS - prod + total
        print(f"  {name:<26} {total:7.1f} ms  ({flops[name] / 1e12:5.2f} TFLOP,"
              f" {flops[name] / (total / 1e3) / 1e12:5.1f} TFLOPS)"
              f"  speedup {prod / total:.2f}x  step ~{step:.0f} ms"
              f"  tokens/s {TOKENS_PER_S * STEP_MS / step:.0f}"
              f" (+{100 * (prod - total) / step:.0f}%)")


if __name__ == "__main__":
    main()
