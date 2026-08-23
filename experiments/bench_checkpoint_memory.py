"""
GPU peak-memory / throughput benchmark for gradient-checkpointing modes (plan T11).

Compares the three ``use_grad_checkpoint`` modes of ``InstinctForCausalLM``:

    Mode 0  no checkpointing        (full activations saved for backward)
    Mode 1  selective recompute     (attention core QK^T/softmax/dropout@V + FFN block
                                     re-run in backward instead of stored)
    Mode 2  whole-layer checkpoint  (torch.utils.checkpoint per block)

Each mode runs a real training step on a synthetic batch, matching
``trainer/train_pretrain.py:35-46``: bf16 autocast forward (labels -> loss) ->
backward -> AdamW step. Per mode we report:

    peak MB   = torch.cuda.max_memory_allocated() after reset_peak_memory_stats()
    steps/s   = steps / elapsed wall time (time.perf_counter, CUDA-synced)
    save %    = (peak0 - peakX) / peak0
    overhead% = (sps0 - spsX) / sps0

Built-in acceptance assertions (only when ``--mode all`` on a real GPU):
    * S=2048 eager (``--flash 0 --seq 2048``):
        Mode 1 peak save >= 45%, Mode 2 peak save >= 85%
        throughput overhead: Mode 1 <= 15%, Mode 2 <= 50%
    * flash path (``--flash 1``):
        |peak1 - peak0| / peak0 <= 10%
        (the fused attention path is unchanged — only the FFN is saved)

Exit code is non-zero when any assertion fails. Thresholds are deliberately
conservative; measured values are always printed next to each threshold.

No GPU (or ``--cpu-only``): prints ``SKIP`` and exits 0.
"""
import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM

warnings.filterwarnings("ignore")

# ---- conservative acceptance thresholds (plan T11) ----
S2048_EAGER_S1_SAVE = 0.45      # Mode 1 peak saving at S=2048 eager
S2048_EAGER_S2_SAVE = 0.85      # Mode 2 peak saving at S=2048 eager
S2048_EAGER_S1_OVERHEAD = 0.15  # Mode 1 throughput overhead
S2048_EAGER_S2_OVERHEAD = 0.50  # Mode 2 throughput overhead
FLASH_PEAK_DIFF = 0.10          # |peak1 - peak0| / peak0 on the flash path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bs", type=int, default=8, help="batch size")
    p.add_argument("--seq", type=int, default=2048, help="sequence length (tokens)")
    p.add_argument("--layers", type=int, default=8, help="num_hidden_layers")
    p.add_argument("--hidden", type=int, default=768, help="hidden size")
    p.add_argument("--moe", type=int, default=0, choices=[0, 1], help="use MoE architecture")
    p.add_argument("--mode", type=str, default="all", choices=["0", "1", "2", "all"],
                   help="gradient-checkpointing mode(s) to benchmark")
    p.add_argument("--steps", type=int, default=5, help="training steps per mode after warmup")
    p.add_argument("--flash", type=int, default=0, choices=[0, 1],
                   help="1 = flash attention (config.flash_attn=True)")
    p.add_argument("--compile", type=int, default=0, choices=[0, 1],
                   help="是否使用 torch.compile 编译整个 CausalLM（0=否，1=是）")
    p.add_argument("--compile_mode", type=str, default="default",
                   choices=["default", "reduce-overhead", "max-autotune",
                            "max-autotune-no-cudagraphs"],
                   help="torch.compile 模式（default=Triton 编译；reduce-overhead=叠加 "
                        "CUDA graph；max-autotune=极限调优，编译极慢）")
    p.add_argument("--cpu-only", action="store_true",
                   help="force no-GPU path: print SKIP and exit 0")
    return p.parse_args()


def build_model(mode, args):
    cfg = InstinctConfig(
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=args.seq * 2,
        vocab_size=6400,
        flash_attn=bool(args.flash),
        use_moe=bool(args.moe),
        use_grad_checkpoint=int(mode),
        dropout=0.0,
    )
    model = InstinctForCausalLM(cfg).to("cuda")
    model.train()  # Mode 1 / Mode 2 checkpoint guards run only under self.training
    if args.compile == 1:
        model = torch.compile(model, mode=args.compile_mode)
    return model


def run_step(model, opt, scaler, input_ids, labels, autocast_ctx):
    opt.zero_grad(set_to_none=True)
    with autocast_ctx:
        out = model(input_ids, labels=labels)
        loss = out.loss + out.aux_loss
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return float(loss.item())


def bench_mode(mode, args):
    torch.cuda.empty_cache()
    t_build = time.perf_counter()
    model = build_model(mode, args)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # bf16 autocast needs no scaling
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())

    def make_batch():
        ids = torch.randint(0, 6400, (args.bs, args.seq), device="cuda")
        return ids, ids.clone()

    # warmup under a fresh peak counter; with torch.compile the first forward
    # triggers graph compilation, so a 2nd warmup step isolates steady-state
    # compiled peak/speed (excluding compilation-time allocations) below.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    ids, labels = make_batch()
    run_step(model, opt, scaler, ids, labels, autocast_ctx)  # triggers compile (if on)
    torch.cuda.synchronize()
    compile_time = time.perf_counter() - t_build
    if args.compile == 1:
        ids, labels = make_batch()
        run_step(model, opt, scaler, ids, labels, autocast_ctx)  # steady-state compiled step
        torch.cuda.synchronize()

    # measured run
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        ids, labels = make_batch()
        run_step(model, opt, scaler, ids, labels, autocast_ctx)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    steps_per_sec = args.steps / elapsed if elapsed > 0 else float("nan")

    del model, opt, scaler, ids, labels
    torch.cuda.empty_cache()
    return {"mode": int(mode), "peak_mb": peak_mb,
            "steps_per_sec": steps_per_sec, "params": n_params,
            "compile_time": compile_time}


def main():
    args = parse_args()
    gpu_ok = torch.cuda.is_available() and not args.cpu_only

    if not gpu_ok:
        print("SKIP: no CUDA GPU available (or --cpu-only) — gradient-checkpointing "
              "memory bench not run")
        print(f"SKIP gpu_available={torch.cuda.is_available()} cpu_only={args.cpu_only}")
        return 0

    modes = ["0", "1", "2"] if args.mode == "all" else [args.mode]
    results = {}
    for m in modes:
        print(f"\n[bench] mode {m} (bs={args.bs} seq={args.seq} "
              f"compile={args.compile}/{args.compile_mode}) ...")
        results[m] = bench_mode(m, args)
        r = results[m]
        print(f"[mode {m}] peak={r['peak_mb']:.1f} MB  steps/s={r['steps_per_sec']:.3f}  "
              f"params={r['params'] / 1e6:.1f}M"
              + (f"  compile_time={r['compile_time']:.1f}s" if args.compile == 1 else ""))

    peak0 = results["0"]["peak_mb"] if "0" in results else None
    sps0 = results["0"]["steps_per_sec"] if "0" in results else None

    print("\n=== Gradient-checkpointing bench ===")
    print(f"config: bs={args.bs} seq={args.seq} layers={args.layers} hidden={args.hidden} "
          f"moe={args.moe} flash={args.flash} steps={args.steps} "
          f"compile={args.compile}({args.compile_mode}) "
          f"device={torch.cuda.get_device_name(0)}")
    print(f"{'mode':>4}  {'peak MB':>9}  {'save%':>6}  {'steps/s':>8}  {'overhead%':>9}")
    for m in ("0", "1", "2"):
        if m not in results:
            continue
        r = results[m]
        save = "" if (peak0 is None or m == "0") else f"{(peak0 - r['peak_mb']) / peak0 * 100:5.1f}%"
        ovh = "" if (sps0 is None or m == "0") else f"{(sps0 - r['steps_per_sec']) / sps0 * 100:7.1f}%"
        print(f"{m:>4}  {r['peak_mb']:9.1f}  {save:>6}  {r['steps_per_sec']:8.3f}  {ovh:>9}")

    if args.mode != "all":
        return 0

    failures = []

    def check(name, ok, measured, threshold):
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}  (measured={measured:.1%}, threshold={threshold:.1%})")
        if not ok:
            failures.append(name)

    if args.flash == 0 and args.seq == 2048:
        print("\n=== Assertions (S=2048 eager) ===")
        s1, s2 = results["1"], results["2"]
        save1 = (peak0 - s1["peak_mb"]) / peak0
        save2 = (peak0 - s2["peak_mb"]) / peak0
        ovh1 = (sps0 - s1["steps_per_sec"]) / sps0
        ovh2 = (sps0 - s2["steps_per_sec"]) / sps0
        check("Mode 1 peak saving >= 45%", save1 >= S2048_EAGER_S1_SAVE,
              save1, S2048_EAGER_S1_SAVE)
        check("Mode 2 peak saving >= 85%", save2 >= S2048_EAGER_S2_SAVE,
              save2, S2048_EAGER_S2_SAVE)
        check("Mode 1 throughput overhead <= 15%", ovh1 <= S2048_EAGER_S1_OVERHEAD,
              ovh1, S2048_EAGER_S1_OVERHEAD)
        check("Mode 2 throughput overhead <= 50%", ovh2 <= S2048_EAGER_S2_OVERHEAD,
              ovh2, S2048_EAGER_S2_OVERHEAD)
    elif args.flash == 1:
        print("\n=== Assertions (flash path) ===")
        diff = abs(peak0 - results["1"]["peak_mb"]) / peak0
        check("flash: |peak1 - peak0| / peak0 <= 10%", diff <= FLASH_PEAK_DIFF,
              diff, FLASH_PEAK_DIFF)
    else:
        print("\n(no acceptance assertions defined for this config)")
        return 0

    if failures:
        print(f"\nASSERTIONS FAILED: {failures}")
        return 1
    print("\nALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
