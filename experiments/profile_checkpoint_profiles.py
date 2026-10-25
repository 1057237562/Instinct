"""Compare PyTorch Dense checkpoint profiles without a GPU, or measure on CUDA.

FakeTensor executes real forward/autograd operators without allocating their
tensor data. Storage lifetimes count aliases once. This is NOT a GPU allocator,
kernel workspace, fragmentation, torch.compile, FA4, or throughput simulator.
The default is Instinct V2 Dense: L16/H768/Q8/KV2/V6400, B32/S4096, BF16
autocast, FP32 parameters/gradients/AdamW moments, an accumulated gradient and
the previous microbatch's returned logits/hidden states still live at entry.
"""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Keep the repository's Windows pyarrow/torch import order.
import datasets  # noqa: F401
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count, sdpa_backward_flop_count

from model.model_instinct import InstinctConfig, InstinctForCausalLM

GIB = 2 ** 30
PROFILES = ("off", "legacy1", "legacy2", "ckpt1", "ckpt2", "ckpt3", "ckpt4")


class StorageTracker(TorchDispatchMode):
    """Live tensor storage bytes, not saved-tensor sums or module-hook estimates.

    Weak storage references also account for tensors retained by SAC caches.
    Views share _cdata; freed storage removes its size. CUDA's minimum allocation
    rounding is applied, but cached allocator blocks/workspaces are not modeled.
    """
    def __init__(self):
        super().__init__()
        self.storages = {}
        self.current = self.peak = 0

    def add(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        storage = tensor.untyped_storage()
        key = storage._cdata
        size = (storage.nbytes() + 511) // 512 * 512
        if key in self.storages:
            _, previous = self.storages[key]
            self.current += size - previous
        else:
            self.current += size
        def remove(ref):
            entry = self.storages.pop(key, None)
            if entry is not None:
                self.current -= entry[1]
        self.storages[key] = (weakref.ref(storage, remove), size)
        self.peak = max(self.peak, self.current)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for tensor in tree_flatten((args, kwargs))[0]:
            self.add(tensor)
        result = func(*args, **kwargs)
        for tensor in tree_flatten(result)[0]:
            self.add(tensor)
        return result


def flop_counter():
    # Torch 2.8 does not register CPU flash SDPA in FlopCounterMode. Use the
    # exact same formulas as its CUDA flash entries: full square attention,
    # FMA=2 FLOPs, flash backward includes QK recomputation. Causal tile skipping
    # and elementwise/reduction/optimizer operations are deliberately excluded.
    registry = FlopCounterMode(display=False, custom_mapping={
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu:
            lambda q, k, v, *a, **kw: sdpa_flop_count(q, k, v),
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward:
            lambda g, q, k, v, *a, **kw: sdpa_backward_flop_count(g, q, k, v),
    }).flop_registry
    return OperatorFlopCounter(registry)


class OperatorFlopCounter(TorchDispatchMode):
    """PyTorch's FLOP formulas without ModuleTracker's backward module hooks.

    Module hooks can prolong checkpoint-recomputed activation lifetimes. Do not
    use module-attributed FlopCounterMode/MemTracker inside a memory experiment.
    """
    def __init__(self, registry):
        super().__init__()
        self.registry = registry
        self.flop_counts = {"Global": defaultdict(int)}

    def get_total_flops(self):
        return sum(self.flop_counts["Global"].values())

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        rule = self.registry.get(func._overloadpacket)
        if rule:
            self.flop_counts["Global"][func._overloadpacket] += rule(*args, **kwargs, out_val=result)
        return result


def make_config(profile, config):
    values = dict(config)
    values.update(checkpoint_profile="legacy" if profile.startswith("legacy") else profile,
                  use_grad_checkpoint=int(profile[-1]) if profile.startswith("legacy") else 0)
    return InstinctConfig(**values)


def simulate(profile, config, batch, seq):
    with FakeTensorMode():
        model = InstinctForCausalLM(make_config(profile, config)).train()
        optimizer = torch.optim.AdamW(model.parameters(), foreach=True)
        ids = torch.ones(batch, seq, dtype=torch.long)
        labels = ids.clone()
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
            optimizer.state[parameter].update(
                step=torch.tensor(1.), exp_avg=torch.zeros_like(parameter),
                exp_avg_sq=torch.zeros_like(parameter))
        # Trainer assigns `res = model(...)`: the previous res survives until
        # the new forward returns, even though its backward already ran.
        previous_output = (
            torch.empty(batch, seq, model.config.vocab_size, dtype=torch.bfloat16),
            torch.empty(batch, seq, model.config.hidden_size, dtype=torch.float32),
        )
        tracker = StorageTracker()
        for tensor in tree_flatten((list(model.parameters()), list(model.buffers()),
                                   [p.grad for p in model.parameters()],
                                   list(optimizer.state.values()), ids, labels, previous_output))[0]:
            tracker.add(tensor)
        del tensor
        persistent_bytes = tracker.current
        counter = flop_counter()
        # Only the data-dependent RoPE repair predicate is bypassed. Its actual
        # buffers, shapes, dtype, RoPE ops and storage remain in the simulation.
        with patch.object(model.model, "_ensure_rope_buffers", lambda device: None), tracker:
            with counter:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = model(ids, labels=labels)
                previous_output = None
                forward_peak = tracker.peak
                forward_live = tracker.current
                forward_flops = counter.get_total_flops()
                output.loss.backward()
            backward_peak = tracker.peak
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            optimizer.step()
        return dict(profile=profile, peak_bytes=tracker.peak,
                    persistent_bytes=persistent_bytes, forward_live_bytes=forward_live,
                    forward_peak_bytes=forward_peak, backward_peak_bytes=backward_peak,
                    forward_flops=forward_flops, step_flops=counter.get_total_flops(),
                    op_flops={str(k): v for k, v in counter.flop_counts["Global"].items()})


def measure_cuda(profile, config, batch, seq, steps):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no real GPU measurement was performed")
    torch.cuda.empty_cache()
    model = InstinctForCausalLM(make_config(profile, config)).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), foreach=True)
    ids = torch.randint(config["vocab_size"], (batch, seq), device="cuda")
    labels = ids.clone()
    output = None
    # FA4 is disabled for the whole comparison so every row uses PyTorch SDPA.
    with patch("model.flash_attn_4._get_fa4", return_value=None):
        def step():
            nonlocal output
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(ids, labels=labels)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            optimizer.step()
            # Retain gradient buffers, as in an accumulation microbatch.
            optimizer.zero_grad(set_to_none=False)
        for _ in range(2):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / steps
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        with flop_counter() as counter:
            step()
    return dict(profile=profile, peak_bytes=peak, peak_reserved_bytes=reserved,
                step_seconds=elapsed, tokens_per_second=batch * seq / elapsed,
                step_flops=counter.get_total_flops())


def add_ratios(rows, overhead_gib, reserve_fraction):
    by_name = {r["profile"]: r for r in rows if "peak_bytes" in r}
    baseline, minimum = by_name.get("off"), by_name.get("ckpt4")
    for row in by_name.values():
        row["peak_gib"] = row["peak_bytes"] / GIB
        row["required_capacity_gib"] = (row["peak_gib"] + overhead_gib) / (1 - reserve_fraction)
        if baseline:
            row["extra_compute_percent"] = 100 * (row["step_flops"] / baseline["step_flops"] - 1)
            row["memory_saved_percent"] = 100 * (1 - row["peak_bytes"] / baseline["peak_bytes"])
        if minimum:
            saved = (minimum["step_flops"] - row["step_flops"]) / 1e12
            extra_memory = row["peak_gib"] - minimum["peak_bytes"] / GIB
            row["tflops_saved_vs_ckpt4"] = saved
            row["compute_saved_percent_vs_ckpt4"] = 100 * saved * 1e12 / minimum["step_flops"]
            row["tflops_saved_per_total_gib_vs_ckpt4"] = saved / row["peak_gib"]
            row["tflops_saved_per_extra_gib_vs_ckpt4"] = saved / extra_memory if extra_memory > 0 else None
        row["pareto_efficient"] = not any(
            other["peak_bytes"] <= row["peak_bytes"] and other["step_flops"] <= row["step_flops"]
            and (other["peak_bytes"] < row["peak_bytes"] or other["step_flops"] < row["step_flops"])
            for other in by_name.values())
    for high, low in zip(("ckpt1", "ckpt2", "ckpt3"), ("ckpt2", "ckpt3", "ckpt4")):
        if high in by_name and low in by_name:
            h, l = by_name[high], by_name[low]
            delta = h["peak_gib"] - l["peak_gib"]
            h["marginal_tflops_per_gib_vs_next_tier"] = (l["step_flops"] - h["step_flops"]) / 1e12 / delta if delta > 0 else None


def choose_profile(rows, capacity_gib):
    fitting = [row for row in rows if row.get("profile", "").startswith("ckpt")
               and row.get("required_capacity_gib", float("inf")) <= capacity_gib]
    if not fitting:
        return None  # Reduce microbatch/sequence; never claim ckpt4 always fits.
    return min(fitting, key=lambda row: (row["step_flops"], row["peak_bytes"]))["profile"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["simulate", "cuda"], default="simulate")
    parser.add_argument("--config", type=Path, help="Optional Dense model JSON; replaces preset fields")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--attention", choices=["sdpa", "math"], default="sdpa")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--overhead-gib", type=float, default=2.0,
                        help="User-observed external overhead allowance, NOT measured PyTorch overhead")
    parser.add_argument("--reserve-fraction", type=float, default=0.10)
    parser.add_argument("--capacities", nargs="+", type=float, default=[8, 10, 12, 16, 24])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.seq, args.steps, *args.capacities) <= 0 or args.overhead_gib < 0 or not 0 <= args.reserve_fraction < 1:
        parser.error("Positive shapes/steps/capacities, nonnegative overhead and reserve in [0,1) required")
    config = dict(hidden_size=768, num_hidden_layers=16, num_attention_heads=8,
                  num_key_value_heads=2, vocab_size=6400, dropout=0.,
                  max_position_embeddings=32768, tie_word_embeddings=True)
    if args.config:
        config.update(json.loads(args.config.read_text(encoding="utf-8")))
    if args.layers is not None:
        config["num_hidden_layers"] = args.layers
    config["flash_attn"] = args.attention == "sdpa"
    if config.get("use_moe") or config.get("model_architecture", "standard") != "standard":
        parser.error("This profiler supports Dense only")
    if config["num_hidden_layers"] <= 0 or args.seq > config["max_position_embeddings"]:
        parser.error("Positive layers and seq <= max_position_embeddings required")
    if config.get("dropout", 0) != 0:
        parser.error("This calibration uses dropout=0; CPU and CUDA fused dropout backends differ")
    if config.get("param_dtype", "fp32") != "fp32":
        parser.error("This calibration uses FP32 parameters and AdamW states; param_dtype must be fp32")
    if args.mode == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no real GPU measurements performed")
    rows = []
    for profile in args.profiles:
        gc.collect()
        try:
            row = simulate(profile, config, args.batch, args.seq) if args.mode == "simulate" else measure_cuda(profile, config, args.batch, args.seq, args.steps)
        except torch.cuda.OutOfMemoryError:
            row = dict(profile=profile, error="CUDA OOM")
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
    add_ratios(rows, args.overhead_gib, args.reserve_fraction)
    result = dict(mode=args.mode, torch_version=torch.__version__, batch=args.batch, seq=args.seq,
                  config=config, dtype="bf16_autocast_fp32_parameters_adamw_foreach",
                  overhead_gib=args.overhead_gib, reserve_fraction=args.reserve_fraction,
                  flop_convention="GEMM + full-square SDPA FMA=2; excludes pointwise/reductions; not execution time",
                  limitations="Fake CPU SDPA tensor lifetimes approximate CUDA SDPA; no allocator cache, opaque kernel workspace, compile, FA4, DDP or early-exit losses",
                  rows=rows, recommendations={str(c): choose_profile(rows, c) for c in args.capacities})
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
