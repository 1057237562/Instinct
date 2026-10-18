"""Muon mantissa slowdown: implementation overhead vs fundamental cost.

Sibling of ``experiments/bench_muon_mantissa_speed.py`` (which is NOT touched).
That script showed, at one small size set, that ``BatchedMuon(use_mantissa=True)``
("A") is ~1.54x slower than the fp32-master Muon ("B") for an isolated optimizer
step. This script answers *why*: how much of that gap is recoverable
implementation overhead (allocations, extra kernel launches, per-matrix Python
loops, int32 upcasts) and how much is fundamental (unavoidable extra elementwise
bandwidth / compute that the mantissa representation needs).

Three sections
--------------
1. SIZE SWEEP. Same shape family at growing widths + two controlled points with
   equal total elements but different matrix counts. Report median ms/step A vs
   B, ratio A/B, ns/elem, and -- crucially -- the *absolute delta* (A-B) ms and
   ns/elem. A delta that stays ~constant while elements grow => fixed per-step
   overhead (implementation). A delta that grows ~linearly with elements =>
   fundamental per-element work. A flat ns/elem delta => bandwidth-bound.

2. PROFILER. ``torch.profiler`` (CUDA activities only, record_shapes=False) on a
   single ``opt.step()`` per config: total kernel count, top kernels by self CUDA
   time, total CUDA time, and the A-only kernels (bit ops / casts / copies).

3. OPTIMIZED PROTOTYPE. A local (non-production) subclass that keeps the same
   numerics but batches the mantissa bit-manipulation across the shape-homogeneous
   chunk (the production ``step()`` already buckets params by exact shape, so a
   chunk is same-shape), removes the redundant ``bitwise_and`` in the split, and
   reuses persistent scratch/grad buffers. Correctness is asserted bit-for-bit
   against the production optimizer before timing. Reports how much of the A/B
   gap it closes = the recoverable ceiling.

Run from the repo root::

    python experiments/bench_muon_mantissa_scaling.py

Windows import order is mandatory: ``datasets`` -> ``trainer.compile_cache`` ->
``torch``. Nothing in production is modified.
"""

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import datasets  # noqa: F401,E402 -- Windows DLL conflict workaround (before torch)
import trainer.compile_cache  # noqa: F401,E402 -- Inductor cache config (before torch)
import torch  # noqa: E402

from trainer.batched_muon import (  # noqa: E402
    BatchedMuon,
    _batched_zeropower,
    _reconstruct_precise,
)

MIB = 2 ** 20
GIB = 2 ** 30
FP32_BUDGET = 2 * GIB  # task cap: keep the fp32-master footprint <= ~2 GiB
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


# --------------------------------------------------------------------------- #
# Shape families
# --------------------------------------------------------------------------- #
def family_shapes(width):
    """Representative Muon shape family: 2 squares + 2 tall rectangles.

    The 3x aspect ratio matches the real V1 dense (768 -> 2432) and MoE
    (512 -> 1664) hidden/intermediate relationships (2.7x - 3.25x).
    """
    return [(width, width), (width, width), (width, 3 * width), (width, 3 * width)]


# Ordered smallest -> largest by total elements, then two controlled points at
# equal total elements but 4 vs 1 matrix (per-matrix overhead isolation).
SWEEP_POINTS = [
    ("P1_fam_w768", family_shapes(768)),      # ~4.7M elements, 4 matrices
    ("P2_fam_w1600", family_shapes(1600)),    # ~20.5M
    ("P3_fam_w2432", family_shapes(2432)),    # ~47.3M
    ("P4_fam_w3264", family_shapes(3264)),    # ~85.2M
    ("P5_sq4096_x4", [(4096, 4096)] * 4),     # 67.1M elements, 4 matrices
    ("P6_sq8192_x1", [(8192, 8192)]),         # 67.1M elements, 1 matrix
]


def total_elements(shapes):
    return sum(rows * cols for rows, cols in shapes)


def fp32_estimate_bytes(shapes):
    """Conservative fp32-master footprint: 12 B/elem (param+grad+momentum),
    plus per-matrix Newton-Schulz workspace (8 B per short^2), plus slack."""
    elems = total_elements(shapes)
    gram = sum(8 * min(rows, cols) ** 2 for rows, cols in shapes)
    return elems * 12 + gram + 64 * MIB


def fit_point(shapes, budget_bytes):
    """Shrink a point (half dims -> drop duplicates -> drop a matrix) until the
    fp32 estimate fits ``budget_bytes``. Returns (shapes, estimate, notes)."""
    current = list(shapes)
    notes = []
    while len(current) > 1:
        estimate = fp32_estimate_bytes(current)
        if estimate <= budget_bytes:
            break
        seen = set()
        deduped = [s for s in current if not (s in seen or seen.add(s))]
        if len(deduped) < len(current):
            notes.append(f"dropped duplicate shapes -> {len(deduped)} matrices")
            current = deduped
            continue
        halved = [
            (max(64, (rows // 2) // 64 * 64), max(64, (cols // 2) // 64 * 64))
            for rows, cols in current
        ]
        if halved != current:
            notes.append(f"halved dims -> {halved}")
            current = halved
        else:
            notes.append(f"dropped one matrix -> {len(current)}")
            current = current[:-1]
    return current, fp32_estimate_bytes(current), notes


# --------------------------------------------------------------------------- #
# Optimizer construction / measurement helpers
# --------------------------------------------------------------------------- #
def build_params(shapes, dtype, device, seed):
    torch.manual_seed(seed)
    params = []
    for shape in shapes:
        param = torch.nn.Parameter(torch.randn(shape, device=device, dtype=dtype) * 0.02)
        param.grad = torch.randn(shape, device=device, dtype=dtype)
        params.append(param)
    return params


def make_optimizer(params, use_mantissa, cls=BatchedMuon):
    """Production-equivalent Muon defaults (matches trainer_utils.build_optimizer
    and the existing bench script)."""
    return cls(
        params, lr=5e-4, weight_decay=0.1, momentum=0.95, nesterov=True,
        ns_coefficients=NS_COEFFICIENTS, ns_steps=5, eps=1e-7,
        adjust_lr_fn="match_rms_adamw", batch_size=16, workspace_mb=64,
        use_mantissa=use_mantissa,
    )


def time_steps(opt, iters, device):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        opt.step()
        ends[i].record()
    torch.cuda.synchronize(device)
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _check_mutation(tag, params, use_mantissa, opt):
    """Assert one step actually mutates the true (reconstructed) parameter."""
    probe = params[0].detach().clone()
    if use_mantissa:
        recon_before = _reconstruct_precise(
            probe, torch.zeros_like(probe, dtype=torch.uint16)
        )
    else:
        recon_before = probe
    before = [p.detach().clone() for p in params[:16]]
    opt.step()
    if use_mantissa:
        recon_after = _reconstruct_precise(
            params[0].detach(), opt.state[params[0]]["mantissa"]
        )
        changed = sum(0 if torch.equal(b, p.detach()) else 1 for b, p in zip(before, params[:16]))
        assert not torch.equal(recon_before, recon_after), (
            f"{tag}: optimizer step did not mutate the reconstructed parameter"
        )
        assert changed > 0, f"{tag}: optimizer step did not mutate any parameter"
    else:
        changed = sum(0 if torch.equal(b, p.detach()) else 1 for b, p in zip(before, params[:16]))
        assert not torch.equal(recon_before, params[0].detach()), (
            f"{tag}: optimizer step did not mutate the probe parameter"
        )
        assert changed > 0, f"{tag}: optimizer step did not mutate any parameter"
    return changed


def bench_isolated(tag, shapes, expected_dtype, use_mantissa, device,
                   warmup, iters, seed, cls=BatchedMuon):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    params = build_params(shapes, expected_dtype, device, seed)
    assert all(p.dtype == expected_dtype for p in params), (
        f"{tag}: not all params are {expected_dtype}"
    )
    opt = make_optimizer(params, use_mantissa, cls=cls)

    changed = _check_mutation(tag, params, use_mantissa, opt)
    assert params[0].dtype == expected_dtype, f"{tag}: parameter dtype changed after step"

    for _ in range(warmup):
        opt.step()
    torch.cuda.synchronize(device)
    persistent = torch.cuda.memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)
    times = time_steps(opt, iters, device)
    peak = torch.cuda.max_memory_allocated(device)

    elems = total_elements(shapes)
    median = statistics.median(times)
    result = {
        "tag": tag, "elements": elems, "n_params": len(params),
        "median_ms": median, "mean_ms": statistics.fmean(times),
        "min_ms": min(times), "max_ms": max(times),
        "ns_per_elem": median * 1e6 / elems,
        "peak_mib": peak / MIB, "persistent_mib": persistent / MIB,
        "mutated": changed,
    }
    del opt, params
    gc.collect()
    torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# Profiler
# --------------------------------------------------------------------------- #
def profile_one_step(opt, device):
    """Profile a single opt.step(); return kernel count, total CUDA us, top keys."""
    from torch.profiler import ProfilerActivity, profile

    opt.step()  # warm (kernel selection / buffer init)
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        opt.step()
        torch.cuda.synchronize(device)

    cuda_events = [
        e for e in prof.events()
        if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA
    ]
    key_avgs = list(prof.key_averages())
    total_us = sum(k.self_device_time_total for k in key_avgs)
    n_kernels = sum(k.count for k in key_avgs if k.self_device_time_total > 0)
    top = sorted(key_avgs, key=lambda k: k.self_device_time_total, reverse=True)
    top = [k for k in top if k.self_device_time_total > 0][:12]
    top_tuples = [(k.key, k.count, k.self_device_time_total) for k in top]
    return {
        "n_cuda_events": len(cuda_events),
        "n_kernels": n_kernels,
        "n_keys": len([k for k in key_avgs if k.self_device_time_total > 0]),
        "total_cuda_us": total_us,
        "all_keys": {k.key: (k.count, k.self_device_time_total) for k in key_avgs},
        "top": top_tuples,
    }


# --------------------------------------------------------------------------- #
# Optimized prototype (local; production untouched)
# --------------------------------------------------------------------------- #
class OptBatchedMuon(BatchedMuon):
    """Same numerics as ``BatchedMuon(use_mantissa=True)`` but the mantissa
    bookkeeping is batched over the shape-homogeneous chunk and buffer-reused.

    ``BatchedMuon.step()`` buckets parameters by exact shape, so every chunk
    passed to ``_step_chunk`` is shape-homogeneous. That lets this variant:
      * stack params / mantissas once and run the reconstruct+split bit ops on
        the whole (n, r, c) chunk instead of per matrix;
      * drop the redundant ``bitwise_and`` (an int32 -> uint16 cast already
        truncates to the low 16 bits);
      * reuse persistent scratch and grad buffers (no per-step allocations).

    Correctness is asserted bit-for-bit against production in the script before
    any timing. This is an experiment-only upper bound on what is recoverable;
    production ``trainer/batched_muon.py`` is never modified.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scratch = {}

    def _get_scratch(self, shape, n, device):
        key = (tuple(shape), n)
        buf = self._scratch.get(key)
        if buf is None:
            r, c = shape
            buf = {
                "p_bf16": torch.empty((n, r, c), dtype=torch.bfloat16, device=device),
                "m_u16": torch.empty((n, r, c), dtype=torch.uint16, device=device),
                "p_i32": torch.empty((n, r, c), dtype=torch.int32, device=device),
                "m_i32": torch.empty((n, r, c), dtype=torch.int32, device=device),
                "bits": torch.empty((n, r, c), dtype=torch.int32, device=device),
                "hi_i32": torch.empty((n, r, c), dtype=torch.int32, device=device),
                "high": torch.empty((n, r, c), dtype=torch.bfloat16, device=device),
                "low": torch.empty((n, r, c), dtype=torch.uint16, device=device),
                "grad_fp32": torch.empty((n, r, c), dtype=torch.float32, device=device),
            }
            self._scratch[key] = buf
        return buf

    def _step_chunk(self, params, group):
        if not self.use_mantissa:
            return super()._step_chunk(params, group)

        n = len(params)
        shape = tuple(params[0].shape)
        device = params[0].device
        db = self._get_scratch(shape, n, device)

        # --- grad promotion into a reused fp32 buffer (no per-step alloc) --- #
        grads = []
        for i, p in enumerate(params):
            target = db["grad_fp32"][i]
            target.copy_(p.grad)  # cast into the persistent fp32 buffer
            grads.append(target)

        buffers = []
        mantissas = []
        for i, p in enumerate(params):
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grads[i])
            if state["momentum_buffer"].dtype != torch.float32:
                state["momentum_buffer"] = state["momentum_buffer"].to(torch.float32)
            buffers.append(state["momentum_buffer"])
            if "mantissa" not in state:
                state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
            mantissas.append(state["mantissa"])

        momentum = group["momentum"]
        torch._foreach_lerp_(buffers, grads, 1 - momentum)
        if group["nesterov"]:
            updates = torch._foreach_lerp(grads, buffers, momentum)
            stacked = torch.stack(updates).to(torch.bfloat16)
            del updates
        else:
            stacked = torch.stack(buffers).to(torch.bfloat16)
        update = _batched_zeropower(
            stacked,
            ns_coefficients=group.get("ns_coefficients", NS_COEFFICIENTS),
            ns_steps=group["ns_steps"], eps=group["eps"],
        )
        del stacked

        lr = group["lr"]
        rows, cols = params[0].shape
        if group["adjust_lr_fn"] == "match_rms_adamw":
            adjusted_lr = lr * 0.2 * math.sqrt(max(rows, cols))
        else:
            adjusted_lr = lr * math.sqrt(max(1.0, rows / cols))

        update_f32 = update.to(torch.float32, memory_format=torch.contiguous_format)

        if n == 1:
            # Production's chunk size collapses to 1 for large matrices, so there is
            # nothing to batch; avoid the extra stack copies and reconstruct in place.
            p, m = params[0], mantissas[0]
            p_i32, m_i32, bits = db["p_i32"][0], db["m_i32"][0], db["bits"][0]
            p_i32.copy_(p.view(torch.int16))                # int16 -> int32
            m_i32.copy_(m)                                  # uint16 -> int32
            torch.bitwise_left_shift(p_i32, 16, out=bits)
            torch.bitwise_or(bits, m_i32, out=bits)
            prec = bits.view(torch.float32)                 # bit-exact fp32 view
            prec.mul_(1 - lr * group["weight_decay"])
            prec.add_(update_f32[0], alpha=-adjusted_lr)
            torch.bitwise_right_shift(bits, 16, out=db["hi_i32"][0])
            db["high"][0].view(torch.int16).copy_(db["hi_i32"][0])
            db["low"][0].copy_(bits)
            p.copy_(db["high"][0])
            m.copy_(db["low"][0])
            return

        # ---- batched reconstruct: (n,r,c) bf16 + (n,r,c) u16 -> (n,r,c) f32 --- #
        # bits = (high16 << 16) | low16, assembled with one buffer for the whole chunk.
        torch.stack(params, out=db["p_bf16"])
        torch.stack(mantissas, out=db["m_u16"])
        db["p_i32"].copy_(db["p_bf16"].view(torch.int16))   # int16 -> int32
        db["m_i32"].copy_(db["m_u16"])                      # uint16 -> int32
        bits = db["bits"]
        torch.bitwise_left_shift(db["p_i32"], 16, out=bits)
        torch.bitwise_or(bits, db["m_i32"], out=bits)
        prec = bits.view(torch.float32)                     # (n,r,c) fp32, bit-exact

        # ---- fused decay + update on the reconstructed stack --------------- #
        prec.mul_(1 - lr * group["weight_decay"])
        prec.add_(update_f32, alpha=-adjusted_lr)

        # ---- batched split + scatter (no bitwise_and needed) --------------- #
        torch.bitwise_right_shift(bits, 16, out=db["hi_i32"])
        # int32 -> int16 truncates to the high word; int32 -> uint16 to the low word.
        db["high"].view(torch.int16).copy_(db["hi_i32"])
        db["low"].copy_(bits)
        for i, (p, m) in enumerate(zip(params, mantissas)):
            p.copy_(db["high"][i])
            m.copy_(db["low"][i])


def check_optimized_matches_production(shapes, device, seed):
    """Bit-for-bit equivalence of OptBatchedMuon vs production, one step."""
    torch.manual_seed(seed)
    p_prod = build_params(shapes, torch.bfloat16, device, seed)
    torch.manual_seed(seed)
    p_opt = build_params(shapes, torch.bfloat16, device, seed)
    opt_prod = make_optimizer(p_prod, True, cls=BatchedMuon)
    opt_opt = make_optimizer(p_opt, True, cls=OptBatchedMuon)
    opt_prod.step()
    opt_opt.step()
    max_diff = 0.0
    exact = True
    for a, b in zip(p_prod, p_opt):
        ra = _reconstruct_precise(a.detach(), opt_prod.state[a]["mantissa"]).view(torch.int32)
        rb = _reconstruct_precise(b.detach(), opt_opt.state[b]["mantissa"]).view(torch.int32)
        if not torch.equal(ra, rb):
            exact = False
            max_diff = max(max_diff, (ra.float() - rb.float()).abs().max().item())
    return exact, max_diff


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_sweep_table(rows):
    print()
    print("-" * 132)
    print("SIZE SWEEP (isolated optimizer step). A=mantissa(bf16 params), B=fp32 master.")
    print("-" * 132)
    print(f"{'point':<14}{'mats':>5}{'elements':>13}"
          f"{'A_med_ms':>10}{'B_med_ms':>10}{'ratio':>8}"
          f"{'A_ns/e':>9}{'B_ns/e':>9}{'d_ns/e':>9}{'d_ms':>9}")
    print("-" * 132)
    for r in rows:
        a, b = r["A"], r["B"]
        ratio = a["median_ms"] / b["median_ms"]
        d_ms = a["median_ms"] - b["median_ms"]
        d_ns = a["ns_per_elem"] - b["ns_per_elem"]
        print(f"{r['name']:<14}{a['n_params']:>5}{a['elements']:>13,}"
              f"{a['median_ms']:>10.4f}{b['median_ms']:>10.4f}{ratio:>8.4f}"
              f"{a['ns_per_elem']:>9.4f}{b['ns_per_elem']:>9.4f}{d_ns:>9.4f}{d_ms:>9.4f}")
    print("-" * 132)


def print_profile(sec):
    print()
    print("-" * 132)
    print("PROFILER: single opt.step()  (CUDA activities, record_shapes=False)")
    print("-" * 132)
    a, b = sec["A"], sec["B"]
    print(f"total CUDA events : A={a['n_cuda_events']:<6} B={b['n_cuda_events']:<6} "
          f"(delta {a['n_cuda_events'] - b['n_cuda_events']:+d})")
    print(f"kernel launches   : A={a['n_kernels']:<6} B={b['n_kernels']:<6} "
          f"(delta {a['n_kernels'] - b['n_kernels']:+d})")
    print(f"distinct op keys  : A={a['n_keys']:<6} B={b['n_keys']:<6}")
    print(f"total CUDA time   : A={a['total_cuda_us']:>9.1f} us  B={b['total_cuda_us']:>9.1f} us  "
          f"(delta {a['total_cuda_us'] - b['total_cuda_us']:+.1f} us)")
    for tag, cfg in (("A_mantissa", a), ("B_fp32", b)):
        print(f"\n  top kernels by self CUDA time -- {tag}:")
        print(f"    {'count':>7}  {'self_us':>10}  {'us/launch':>10}  kernel")
        for key, count, us in cfg["top"]:
            per = us / count if count else 0.0
            print(f"    {count:>7}  {us:>10.1f}  {per:>10.2f}  {key}")

    only_a = []
    for key, (count, us) in a["all_keys"].items():
        _, bus = b["all_keys"].get(key, (0, 0.0))
        if bus <= 0.0 and us > 0.0:
            only_a.append((key, count, us))
    only_a.sort(key=lambda t: t[2], reverse=True)
    print("\n  kernels present in A but not B (the mantissa bookkeeping):")
    print(f"    {'count':>7}  {'self_us':>10}  kernel")
    for key, count, us in only_a[:20]:
        print(f"    {count:>7}  {us:>10.1f}  {key}")
    total_only_a = sum(us for _, _, us in only_a)
    print(f"    -> {len(only_a)} distinct A-only kernels, {total_only_a:.1f} us total "
          f"({total_only_a / max(a['total_cuda_us'], 1e-9) * 100:.1f}% of A CUDA time)")


def print_opt(opt_rows):
    print()
    print("-" * 132)
    print("OPTIMIZED PROTOTYPE vs production (same shapes, same protocol)")
    print("-" * 132)
    print(f"{'point':<14}{'A_prod_ms':>11}{'A_opt_ms':>11}{'B_ms':>10}"
          f"{'ratio_prod':>11}{'ratio_opt':>10}{'recovered':>11}")
    print("-" * 132)
    for r in opt_rows:
        print(f"{r['name']:<14}{r['A_prod']:>11.4f}{r['A_opt']:>11.4f}{r['B']:>10.4f}"
              f"{r['ratio_prod']:>11.4f}{r['ratio_opt']:>10.4f}{r['recovered_pct']:>10.1f}%")
    print("-" * 132)
    print("recovered% = fraction of the production A/B gap removed by the optimized "
          "variant, per point.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-point", default="P3_fam_w2432",
                        help="sweep point used for the profiler section")
    parser.add_argument("--opt-points", nargs="*",
                        default=["P1_fam_w768", "P2_fam_w1600", "P4_fam_w3264", "P6_sq8192_x1"],
                        help="sweep points used for the optimized-prototype section")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if args.warmup < 20 or args.iters < 50:
        parser.error("--warmup must be >= 20 and --iters must be >= 50")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        parser.error("this benchmark requires a CUDA device")

    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    props = torch.cuda.get_device_properties(device)
    total_vram = props.total_memory
    print("=" * 132)
    print("Muon mantissa slowdown: implementation overhead vs fundamental cost")
    print(f"device     : {torch.cuda.get_device_name(device)}")
    print(f"total VRAM : {total_vram} bytes ({total_vram / GIB:.2f} GiB)")
    print(f"torch      : {torch.__version__}   CUDA runtime: {torch.version.cuda}")
    print(f"protocol   : warmup={args.warmup}, iters={args.iters}, seed={args.seed}, "
          f"torch.cuda.Event + synchronize")
    print(f"fp32 cap   : {FP32_BUDGET / GIB:.2f} GiB per point")
    print("=" * 132)

    # ---- build + fit the sweep points ---------------------------------------- #
    points = []
    print("Shape-set preparation:")
    for name, shapes in SWEEP_POINTS:
        fitted, est, notes = fit_point(shapes, FP32_BUDGET)
        elems = total_elements(fitted)
        maxdim = max(max(r, c) for r, c in fitted)
        note = ("  [fit: " + "; ".join(notes) + "]") if notes else ""
        print(f"  {name:<14} {len(fitted)} matrices, {elems:>13,} elements, max_dim={maxdim:<6} "
              f"fp32_est={est / MIB:8.1f} MiB  fits={est <= FP32_BUDGET}{note}")
        assert est <= FP32_BUDGET, f"{name}: fp32 estimate {est} exceeds cap"
        points.append((name, fitted, est))
    print()

    # ---- Section 1: size sweep ---------------------------------------------- #
    sweep_rows = []
    for name, shapes, est in points:
        gc.collect()
        torch.cuda.empty_cache()
        res_b = bench_isolated(f"{name}/B_fp32", shapes, torch.float32, False, device,
                               args.warmup, args.iters, args.seed)
        res_a = bench_isolated(f"{name}/A_mantissa", shapes, torch.bfloat16, True, device,
                               args.warmup, args.iters, args.seed)
        sweep_rows.append({"name": name, "A": res_a, "B": res_b})
    print_sweep_table(sweep_rows)

    # ---- Section 2: profiler ------------------------------------------------- #
    pname, pshapes, _ = next(p for p in points if p[0] == args.profile_point)
    print()
    print(f"[profiler section uses point {pname}: {len(pshapes)} matrices, "
          f"{total_elements(pshapes):,} elements]")
    prof = {}
    for tag, use_mantissa, dtype in (("B", False, torch.float32),
                                     ("A", True, torch.bfloat16)):
        gc.collect()
        torch.cuda.empty_cache()
        params = build_params(pshapes, dtype, device, args.seed)
        opt = make_optimizer(params, use_mantissa)
        prof[tag] = profile_one_step(opt, device)
        del opt, params
        gc.collect()
        torch.cuda.empty_cache()
    print_profile(prof)

    # ---- Section 3: optimized prototype -------------------------------------- #
    print()
    print("Correctness check (optimized prototype vs production, one step):")
    ok_multi, diff_multi = check_optimized_matches_production(
        [(256, 256)] * 6, device, args.seed
    )
    ok_single, diff_single = check_optimized_matches_production(
        [(1024, 1024)], device, args.seed
    )
    print(f"  multi-matrix chunk (6x [256,256])   : bit-exact={ok_multi} max|diff|={diff_multi}")
    print(f"  single-matrix chunk (1x [1024,1024]): bit-exact={ok_single} max|diff|={diff_single}")
    assert ok_multi and ok_single, "optimized prototype is not bit-exact vs production"

    opt_rows = []
    for name, shapes, est in points:
        if name not in args.opt_points:
            continue
        gc.collect()
        torch.cuda.empty_cache()
        res_b = bench_isolated(f"{name}/B_fp32", shapes, torch.float32, False, device,
                               args.warmup, args.iters, args.seed)
        res_prod = bench_isolated(f"{name}/A_prod", shapes, torch.bfloat16, True, device,
                                  args.warmup, args.iters, args.seed)
        res_opt = bench_isolated(f"{name}/A_opt", shapes, torch.bfloat16, True, device,
                                 args.warmup, args.iters, args.seed, cls=OptBatchedMuon)
        gap_prod = res_prod["median_ms"] - res_b["median_ms"]
        gap_opt = res_opt["median_ms"] - res_b["median_ms"]
        recovered = (1 - gap_opt / gap_prod) * 100 if gap_prod > 0 else 0.0
        opt_rows.append({
            "name": name,
            "A_prod": res_prod["median_ms"], "A_opt": res_opt["median_ms"],
            "B": res_b["median_ms"],
            "ratio_prod": res_prod["median_ms"] / res_b["median_ms"],
            "ratio_opt": res_opt["median_ms"] / res_b["median_ms"],
            "recovered_pct": recovered,
        })
    print_opt(opt_rows)

    # ---- Verdict ------------------------------------------------------------- #
    print()
    print("=" * 132)
    print("VERDICT")
    print("=" * 132)
    ratios = [r["A"]["median_ms"] / r["B"]["median_ms"] for r in sweep_rows]
    deltas_ms = [r["A"]["median_ms"] - r["B"]["median_ms"] for r in sweep_rows]
    deltas_ns = [r["A"]["ns_per_elem"] - r["B"]["ns_per_elem"] for r in sweep_rows]
    elems = [r["A"]["elements"] for r in sweep_rows]
    print("ratio vs size      : " + "  ".join(
        f"{r['name'].split('_')[0]}={rt:.3f}" for r, rt in zip(sweep_rows, ratios)))
    print("delta A-B (ms)     : " + "  ".join(
        f"{r['name'].split('_')[0]}={d:+.3f}" for r, d in zip(sweep_rows, deltas_ms)))
    print("delta A-B (ns/elem): " + "  ".join(
        f"{r['name'].split('_')[0]}={d:+.4f}" for r, d in zip(sweep_rows, deltas_ns)))
    print(f"element span       : {min(elems):,} .. {max(elems):,} "
          f"({max(elems) / min(elems):.1f}x)")
    if opt_rows:
        print("optimized recovery : " + "  ".join(
            f"{r['name'].split('_')[0]}={r['recovered_pct']:.1f}%" for r in opt_rows))
    print("=" * 132)

    if args.json:
        payload = {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "sweep": [
                {
                    "name": r["name"], "elements": r["A"]["elements"],
                    "A_median_ms": r["A"]["median_ms"], "B_median_ms": r["B"]["median_ms"],
                    "ratio": r["A"]["median_ms"] / r["B"]["median_ms"],
                    "A_ns_per_elem": r["A"]["ns_per_elem"],
                    "B_ns_per_elem": r["B"]["ns_per_elem"],
                }
                for r in sweep_rows
            ],
            "profile": {
                "point": pname,
                "A": {k: prof["A"][k] for k in
                      ("n_kernels", "n_cuda_events", "total_cuda_us", "top")},
                "B": {k: prof["B"][k] for k in
                      ("n_kernels", "n_cuda_events", "total_cuda_us", "top")},
            },
            "optimized": opt_rows,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"json written to {args.json}")


if __name__ == "__main__":
    main()
