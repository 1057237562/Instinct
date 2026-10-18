"""Wall-clock benchmark: Muon mantissa mode vs fp32-master mode (SMALL models only).

Goal
----
Measure whether ``BatchedMuon(use_mantissa=True)`` (bf16 params + parallel uint16
low-half + fp32 momentum + fp32 grad promotion) is FASTER or SLOWER than the
default fp32-master Muon (fp32 params + fp32 grads + fp32 momentum), by how
much, and what the peak memory of each is. Scope is deliberately small so the
whole script fits comfortably on an 8 GB consumer GPU (e.g. RTX 5060 Ti 8 GB).

Why isolated optimizer steps are the primary measurement
--------------------------------------------------------
``_batched_zeropower`` casts its input to bf16 in *both* modes, so the
Newton-Schulz matmul work is identical. The mantissa path only adds per-step
work: bf16->fp32 gradient promotion, ``_reconstruct_precise`` (bit ops + fp32
alloc), fp32 ``foreach`` decay/add, ``_split_precise`` and two ``copy_`` calls.
Timing repeated ``opt.step()`` calls isolates exactly that overhead with no
forward/backward noise.

Config A "mantissa": params bf16, ``.grad`` bf16, ``use_mantissa=True``.
Config B "fp32":     params fp32, ``.grad`` fp32, ``use_mantissa=False``.
Both use the SAME shape multiset, representative of real Muon layers.

Secondary end-to-end: a tiny dense model (``tests/helpers.make_tiny_config``),
forward + backward + ``opt.step()`` via the production ``build_optimizer`` Muon
wiring. Best-effort: failures are printed verbatim and do not invalidate the
isolated numbers.

Timing uses ``torch.cuda.Event`` + ``torch.cuda.synchronize`` (never time.time).

Run from the repo root::

    python experiments/bench_muon_mantissa_speed.py
"""

import argparse
import gc
import json
import os
import statistics
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))
import datasets  # noqa: F401,E402 -- Windows DLL import order (must precede torch)
import trainer.compile_cache  # noqa: F401,E402 -- before torch
import torch  # noqa: E402

from model.model_instinct import InstinctForCausalLM  # noqa: E402
from tests.helpers import make_tiny_config  # noqa: E402
from trainer.batched_muon import BatchedMuon, _reconstruct_precise  # noqa: E402
from trainer.trainer_utils import build_optimizer  # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)

# Small 2D matrices representative of real Muon layers: hidden x hidden and
# hidden x (moe_)intermediate shapes around the V1/V2 widths. "A few of each".
BASE_SHAPES = [
    (512, 512), (512, 512),
    (512, 1728), (512, 1728),
    (640, 640), (640, 640),
    (640, 1728), (640, 1728),
]


def _total_elements(shapes):
    return sum(rows * cols for rows, cols in shapes)


def _fp32_estimate_bytes(shapes):
    """Conservative fp32-master footprint: params+grads+momentum (12 B/elem)
    plus per-matrix bf16 Gram/polynomial workspace (8 B per short^2) plus a
    fixed foreach/allocator overhead allowance."""
    elems = _total_elements(shapes)
    gram = sum(8 * min(rows, cols) ** 2 for rows, cols in shapes)
    return elems * 12 + gram + 64 * MIB


def fit_shapes(shapes, total_vram_bytes):
    """Shrink the shape set until the fp32 footprint fits a 1/4-VRAM budget.

    Returns ``(shapes, estimate_bytes, shrank)``. With the default set the
    estimate is tens of MB, so this normally returns immediately unchanged.
    """
    budget = max(256 * MIB, total_vram_bytes // 4)
    current = list(shapes)
    shrank = []
    while len(current) > 1:
        estimate = _fp32_estimate_bytes(current)
        if estimate <= budget:
            break
        # First drop duplicate shapes (keep the first of each unique shape).
        seen = set()
        deduped = [s for s in current if not (s in seen or seen.add(s))]
        if len(deduped) < len(current):
            shrank.append(f"dropped duplicate shapes -> {len(deduped)} matrices")
            current = deduped
            continue
        # Then halve each dimension (rounded down to a multiple of 64, min 64).
        halved = [
            (max(64, (rows // 2) // 64 * 64), max(64, (cols // 2) // 64 * 64))
            for rows, cols in current
        ]
        if halved == current:
            current = current[:-1]
            shrank.append(f"dropped one matrix -> {len(current)}")
        else:
            current = halved
            shrank.append(f"halved dims -> {current}")
    return current, _fp32_estimate_bytes(current), shrank


def _tensors_equal(a, b):
    """Bitwise tensor equality (thin wrapper so untyped list indexing stays out
    of type-checked ``torch.equal`` call sites)."""
    return torch.equal(a, b)


def build_params(shapes, dtype, device, seed):
    torch.manual_seed(seed)
    params = []
    for shape in shapes:
        param = torch.nn.Parameter(
            torch.randn(shape, device=device, dtype=dtype) * 0.02
        )
        param.grad = torch.randn(shape, device=device, dtype=dtype)
        params.append(param)
    return params


def make_optimizer(params, use_mantissa):
    """Production-equivalent Muon defaults (see trainer_utils.build_optimizer)."""
    return BatchedMuon(
        params, lr=5e-4, weight_decay=0.1, momentum=0.95, nesterov=True,
        ns_coefficients=NS_COEFFICIENTS, ns_steps=5, eps=1e-7,
        adjust_lr_fn="match_rms_adamw", batch_size=16, workspace_mb=64,
        use_mantissa=use_mantissa,
    )


def time_steps(opt, iters, device):
    """Per-step elapsed ms via CUDA events; one event pair per measured step."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        opt.step()
        ends[i].record()
    torch.cuda.synchronize(device)
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def bench_isolated(tag, shapes, expected_dtype, use_mantissa, device,
                   warmup, iters, seed):
    """Build one config, sanity-check it, then time repeated ``opt.step()``."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    params = build_params(shapes, expected_dtype, device, seed)
    assert all(p.dtype == expected_dtype for p in params), (
        f"{tag}: not all params are {expected_dtype}"
    )
    opt = make_optimizer(params, use_mantissa)

    # Sanity: a single step must actually mutate the parameter (guards against a
    # no-op measurement). In mantissa mode the bf16 high bits can legitimately
    # keep their value for a sub-ulp update, so the true parameter is the
    # reconstructed (bf16, uint16) fp32 value; assert that changed instead.
    probe_bf16 = params[0].detach().clone()
    probe_recon = (
        _reconstruct_precise(
            probe_bf16, torch.zeros_like(probe_bf16, dtype=torch.uint16)
        )
        if use_mantissa else None
    )
    before = [(p.detach().clone(), p) for p in params[:16]]
    opt.step()
    bf16_changed = not _tensors_equal(probe_bf16, params[0].detach())
    if use_mantissa:
        recon_after = _reconstruct_precise(
            params[0].detach(), opt.state[params[0]]["mantissa"]
        )
        changed_probe = sum(
            0 if _tensors_equal(bb, p.detach()) else 1 for bb, p in before
        )
        recon_changed = not _tensors_equal(probe_recon, recon_after)
        print(f"  {tag}: probe bf16-high-bits changed={bf16_changed}, "
              f"fp32-reconstruction changed={recon_changed}, "
              f"changed(bf16-view)={changed_probe}/{len(before)}")
        assert recon_changed, (
            f"{tag}: optimizer step did not mutate the reconstructed parameter"
        )
    else:
        changed_probe = sum(
            0 if _tensors_equal(bb, p.detach()) else 1 for bb, p in before
        )
        print(f"  {tag}: probe param changed={not _tensors_equal(probe_bf16, params[0].detach())}, "
              f"changed={changed_probe}/{len(before)}")
        assert not _tensors_equal(probe_bf16, params[0].detach()), (
            f"{tag}: optimizer step did not mutate the probe parameter"
        )
        assert changed_probe > 0, f"{tag}: optimizer step did not mutate any parameter"
    assert params[0].dtype == expected_dtype, (
        f"{tag}: parameter dtype changed after step"
    )
    del probe_bf16, probe_recon, before

    for _ in range(warmup):
        opt.step()
    torch.cuda.synchronize(device)
    persistent = torch.cuda.memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)
    times = time_steps(opt, iters, device)
    peak = torch.cuda.max_memory_allocated(device)

    elems = _total_elements(shapes)
    result = {
        "tag": tag,
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "ns_per_elem_per_step": statistics.median(times) * 1e6 / elems,
        "persistent_mib": persistent / MIB,
        "peak_mib": peak / MIB,
        "n_params": len(params),
        "elements": elems,
    }
    del opt, params
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _forward_backward_step(model, opt, tokens, autocast_ctx):
    for p in model.parameters():
        p.grad = None
    with autocast_ctx:
        out = model(tokens, labels=tokens)
        loss = out.loss
        aux = getattr(out, "aux_loss", None)
        if aux is not None:
            loss = loss + aux
    loss.backward()
    opt.step()


def bench_e2e(device, seq, micro_batch, warmup, iters, seed):
    """Forward + backward + opt.step() on a tiny dense model, A vs B."""
    results = []
    for tag, use_mantissa, param_dtype in (
        ("A_mantissa", True, torch.bfloat16),
        ("B_fp32", False, torch.float32),
    ):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(seed)

        config = make_tiny_config(use_moe=False)
        model = InstinctForCausalLM(config).to(dtype=param_dtype).to(device)
        model.train()

        os.environ["INSTINCT_MUON_MANTISSA"] = "1" if use_mantissa else "0"
        opt = build_optimizer(
            list(model.named_parameters()), lr=5e-4, optimizer="muon"
        )

        generator = torch.Generator(device=device).manual_seed(seed)
        tokens = torch.randint(
            0, config.vocab_size, (micro_batch, seq),
            generator=generator, device=device,
        )
        autocast_ctx = torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=True
        )
        for _ in range(warmup):
            _forward_backward_step(model, opt, tokens, autocast_ctx)
        torch.cuda.synchronize(device)
        persistent = torch.cuda.memory_allocated(device)

        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _forward_backward_step(model, opt, tokens, autocast_ctx)
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated(device)

        results.append({
            "tag": tag,
            "mean_ms": statistics.fmean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "persistent_mib": persistent / MIB,
            "peak_mib": peak / MIB,
            "seq": seq,
            "micro_batch": micro_batch,
            "param_dtype": str(param_dtype).replace("torch.", ""),
        })
        del model, opt, tokens
        gc.collect()
        torch.cuda.empty_cache()
    return results


def _print_isolated(result, reference_median=None):
    ratio = ""
    if reference_median is not None:
        ratio = f"  ratio_vs_B={result['median_ms'] / reference_median:.4f}"
    print(
        f"  {result['tag']:<12} mean={result['mean_ms']:8.4f} ms  "
        f"median={result['median_ms']:8.4f} ms  "
        f"min={result['min_ms']:8.4f}  max={result['max_ms']:8.4f}  "
        f"ns/elem/step={result['ns_per_elem_per_step']:8.4f}  "
        f"persistent={result['persistent_mib']:8.1f} MiB  "
        f"peak={result['peak_mib']:8.1f} MiB{ratio}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-e2e", action="store_true",
                        help="Skip the end-to-end forward+backward+step section")
    parser.add_argument("--e2e-warmup", type=int, default=5)
    parser.add_argument("--e2e-iters", type=int, default=20)
    parser.add_argument("--e2e-seq", type=int, default=256)
    parser.add_argument("--e2e-micro-batch", type=int, default=2)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if args.warmup < 20 or args.iters < 50:
        parser.error("--warmup must be >= 20 and --iters must be >= 50")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        parser.error("this benchmark requires a CUDA device")

    device = torch.device(args.device)
    torch.set_num_threads(1)

    props = torch.cuda.get_device_properties(device)
    total_vram = props.total_memory
    print("=" * 100)
    print("Muon mantissa vs fp32-master speed benchmark (small models)")
    print(f"device      : {torch.cuda.get_device_name(device)}")
    print(f"total VRAM  : {total_vram} bytes ({total_vram / GIB:.2f} GiB)")
    print(f"torch       : {torch.__version__}   CUDA runtime: {torch.version.cuda}")
    print(f"protocol    : warmup={args.warmup}, iters={args.iters}, "
          f"seed={args.seed}, timing=torch.cuda.Event + synchronize")
    print("=" * 100)

    # --- VRAM guard: assert the default set fits, else shrink automatically. ---
    default_est = _fp32_estimate_bytes(BASE_SHAPES)
    budget = max(256 * MIB, total_vram // 4)
    print(f"VRAM guard  : default fp32 estimate={default_est / MIB:.1f} MiB, "
          f"budget (1/4 VRAM)={budget / MIB:.1f} MiB, "
          f"fits={default_est <= budget}")
    shapes, est_bytes, shrank = fit_shapes(BASE_SHAPES, total_vram)
    if shrank:
        print("              -> auto-shrunk:")
        for step in shrank:
            print(f"                 {step}")
    assert est_bytes <= budget, (
        f"cannot fit benchmark set within VRAM: estimate={est_bytes} budget={budget}"
    )
    elems = _total_elements(shapes)
    print(f"shape set   : {len(shapes)} matrices, {elems:,} elements, "
          f"fp32 estimate={est_bytes / MIB:.1f} MiB "
          f"({est_bytes / max(elems, 1):.2f} B/elem incl. workspace)")
    print(f"shapes      : {shapes}")
    print()

    print("--- Isolated optimizer step (PRIMARY) ---")
    result_b = bench_isolated(
        "B_fp32", shapes, torch.float32, False, device,
        args.warmup, args.iters, args.seed,
    )
    result_a = bench_isolated(
        "A_mantissa", shapes, torch.bfloat16, True, device,
        args.warmup, args.iters, args.seed,
    )
    _print_isolated(result_a, reference_median=result_b["median_ms"])
    _print_isolated(result_b, reference_median=result_b["median_ms"])
    ratio_median = result_a["median_ms"] / result_b["median_ms"]
    ratio_mean = result_a["mean_ms"] / result_b["mean_ms"]
    print(
        f"  ratio A/B median = {ratio_median:.4f} ({(ratio_median - 1) * 100:+.2f}%)   "
        f"mean = {ratio_mean:.4f} ({(ratio_mean - 1) * 100:+.2f}%)   "
        f"ns/elem A={result_a['ns_per_elem_per_step']:.4f} vs "
        f"B={result_b['ns_per_elem_per_step']:.4f}   "
        f"peak delta={result_a['peak_mib'] - result_b['peak_mib']:+.1f} MiB"
    )

    e2e_results = None
    if not args.skip_e2e:
        print()
        print(f"--- End-to-end forward+backward+step (secondary): "
              f"seq={args.e2e_seq}, micro_batch={args.e2e_micro_batch}, "
              f"tiny dense config ---")
        try:
            e2e_results = bench_e2e(
                device, args.e2e_seq, args.e2e_micro_batch,
                args.e2e_warmup, args.e2e_iters, args.seed,
            )
            for result in e2e_results:
                print(
                    f"  {result['tag']:<12} mean={result['mean_ms']:8.4f} ms  "
                    f"median={result['median_ms']:8.4f} ms  "
                    f"min={result['min_ms']:8.4f}  max={result['max_ms']:8.4f}  "
                    f"persistent={result['persistent_mib']:8.1f} MiB  "
                    f"peak={result['peak_mib']:8.1f} MiB  "
                    f"({result['param_dtype']})"
                )
            e2e_ratio = e2e_results[0]["median_ms"] / e2e_results[1]["median_ms"]
            print(f"  ratio A/B (median) = {e2e_ratio:.4f} ({(e2e_ratio - 1) * 100:+.2f}%)")
        except Exception as exc:  # noqa: BLE001 -- best-effort section
            print(f"  END-TO-END FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            e2e_results = None
            torch.cuda.empty_cache()

    print()
    print("=" * 100)
    verdict = "FASTER" if ratio_median < 1.0 else "SLOWER"
    print(
        f"VERDICT: mantissa Muon step is {verdict} than fp32-master Muon by "
        f"{abs(ratio_median - 1) * 100:.2f}% "
        f"(median {result_a['median_ms']:.4f} ms vs {result_b['median_ms']:.4f} ms, "
        f"ratio {ratio_median:.4f}); "
        f"ns/elem/step {result_a['ns_per_elem_per_step']:.4f} vs "
        f"{result_b['ns_per_elem_per_step']:.4f}; "
        f"peak {result_a['peak_mib']:.1f} MiB vs {result_b['peak_mib']:.1f} MiB."
    )
    print("=" * 100)

    if args.json:
        payload = {
            "device": torch.cuda.get_device_name(device),
            "total_vram_bytes": total_vram,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "protocol": {"warmup": args.warmup, "iters": args.iters, "seed": args.seed},
            "shapes": shapes,
            "elements": elems,
            "isolated": {
                "A_mantissa": result_a,
                "B_fp32": result_b,
                "ratio_median": ratio_median,
                "ratio_mean": ratio_mean,
            },
            "e2e": e2e_results,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json written to {args.json}")


if __name__ == "__main__":
    main()
