"""Bounded synthetic activation audit for the current V2 recurrent backbone.

No optimizer, corpus, or production checkpoint is used. Experimental schedules
reuse the real modules, and must pass loss/all-parameter gradient parity first.
Saved-storage accounting is a separate instrumented forward, not a peak-memory
measurement; CUDA peaks/timing come from uninstrumented forward + backward.
Run from the repository root, with PYTHONUTF8=1 on Windows.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import datasets  # noqa: F401 -- Windows DLL import ordering
import trainer.compile_cache  # noqa: F401 -- before torch
import torch
from torch.utils.checkpoint import checkpoint

from model.model_recurrsive_instinct import InstinctConfig, InstinctForCausalLM, _compute_lm_loss
from model.rope import select_rope_cache


VARIANTS = (
    "native0", "native1", "native2", "side_checkpoint", "all_checkpoint",
    "segment4", "nested2", "nested4", "nested8",
)


def recompute(fn, *args):
    return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True)


def experimental_loss(model, ids, labels, total, backward, variant):
    """Dense/standard/unpacked/no-cache only; equivalent finite-depth objective."""
    m = model.model
    h = m.dropout(m.embed_tokens(ids)) * model.config.embedding_scale
    pos = select_rope_cache(
        m.freqs_cos, m.freqs_sin, m.freqs_cos_short, m.freqs_sin_short,
        model.config.rope_scaling, start_pos=0, seq_length=ids.shape[1],
        position_ids=None,
    )

    def side(layer, value):
        return recompute(lambda x: layer(x, pos, use_cache=False)[0], value)

    for layer in m.prelude:
        h = side(layer, h)
    embedded = h
    state = m._initialize_state(embedded)

    def core(value, injected):
        value = m._inject_input(value, injected, 0)
        for layer in m.recurrent_block:
            value = layer(value, pos, use_cache=False)[0]
        return value

    with torch.no_grad():
        for _ in range(total - backward):
            state = core(state, embedded)

    group = int(variant.removeprefix("nested")) if variant.startswith("nested") else 4
    if variant in ("side_checkpoint", "all_checkpoint"):
        for _ in range(backward):
            state = recompute(core, state, embedded)
    else:
        for start in range(0, backward, group):
            size = min(group, backward - start)

            def segment(value, injected, size=size):
                for _ in range(size):
                    value = (recompute(core, value, injected)
                             if variant.startswith("nested") else core(value, injected))
                return value

            state = recompute(segment, state, embedded)

    h = m.recurrent_norm(state)
    for layer in m.coda:
        h = side(layer, h)
    h = m.norm(h)
    if variant == "side_checkpoint":
        return _compute_lm_loss(model.lm_head(h), labels)
    return recompute(lambda x: _compute_lm_loss(model.lm_head(x), labels), h)


def loss_for(model, ids, labels, total, backward, variant):
    if variant.startswith("native"):
        out = model(ids, labels=labels, num_steps=(total - backward, backward))
        return out.loss + out.aux_loss
    return experimental_loss(model, ids, labels, total, backward, variant)


def configure_mode(model, variant):
    mode = int(variant[-1]) if variant.startswith("native") else 0
    model.config.use_grad_checkpoint = mode
    for layer in model.model.layers:
        layer.use_grad_checkpoint = mode
        layer.self_attn.use_grad_checkpoint = mode


def seed():
    torch.manual_seed(991)


def check_parity(variants):
    # FP32 CPU, eager attention: isolate schedule errors from CUDA rounding.
    torch.manual_seed(123)
    cfg = InstinctConfig(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128,
        vocab_size=128, max_position_embeddings=32, prelude_layers=1,
        recurrent_layers=2, coda_layers=1, loop_iters=8,
        max_recurrence=16, mean_backprop_depth=8, flash_attn=False,
        recurrence_sampling="fixed", state_init_std=0.1, dropout=0.0,
    )
    model = InstinctForCausalLM(cfg).train()
    ids = torch.randint(0, 128, (1, 12))
    # Include both truncated and full BPTT, with non-divisible group lengths.
    results = []
    for total, backward in ((11, 8), (5, 5)):
        reference = None
        for variant in dict.fromkeys(("native0", *variants)):
            configure_mode(model, variant)
            model.zero_grad(set_to_none=True)
            seed()
            loss = loss_for(model, ids, ids, total, backward, variant)
            loss.backward()
            grads = {n: p.grad.clone() if p.grad is not None else None
                     for n, p in model.named_parameters()}
            if reference is None:
                reference = (loss.detach(), grads)
            torch.testing.assert_close(loss.detach(), reference[0], rtol=1e-5, atol=1e-6)
            max_error = 0.0
            for name, grad in grads.items():
                expected = reference[1][name]
                if (grad is None) != (expected is None):
                    raise AssertionError(f"Gradient presence differs: {variant}: {name}")
                if grad is not None:
                    torch.testing.assert_close(grad, expected, rtol=1e-4, atol=2e-6)
                    max_error = max(max_error, (grad - expected).abs().max().item())
            results.append(dict(variant=variant, total=total, backward=backward,
                                max_abs_grad_error=max_error, passed=True))
    return results


def storage_key(t):
    storage = t.untyped_storage()
    return (str(t.device), storage.data_ptr(), storage.nbytes())


def saved_storage(model, run):
    excluded = {storage_key(t) for t in (*model.parameters(), *model.buffers())}
    seen = set()
    counts = {"save_calls": 0, "nonparameter_unique_storage_bytes": 0}

    def pack(t):
        counts["save_calls"] += 1
        key = storage_key(t)
        if key not in excluded and key not in seen:
            seen.add(key)
            counts["nonparameter_unique_storage_bytes"] += key[2]
        # Avoid retaining the original autograd graph through the pack hook.
        return t.detach()

    seed()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = run()
    del loss
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--total", type=int, default=16)
    p.add_argument("--backward", type=int, default=8)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    variants = args.variants.split(",")
    if any(v not in VARIANTS for v in variants):
        p.error("Unknown variant")
    if not 1 <= args.backward <= args.total or min(args.seq, args.batch, args.steps) < 1:
        p.error("Require 1 <= backward <= total and positive dimensions/steps")
    if args.warmup < 0:
        p.error("warmup must be non-negative")
    torch.set_num_threads(1)
    report = dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  scope="Dense V2, synthetic IDs, eager BF16 autocast, FP32 parameters; no optimizer or updates",
                  limitations="Saved-storage hooks are a separate forward-only diagnostic, include autocast weight copies, and do not equal activation peak. Peaks include model + gradients, exclude optimizer/compile/packing. Timings are short probes, not production throughput.",
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  parity=check_parity(variants), results=[])
    print("CPU loss/all-parameter gradient parity passed", flush=True)
    if not torch.cuda.is_available():
        report["gpu_status"] = "unavailable"
    else:
        report["gpu"] = torch.cuda.get_device_name()
        if args.config:
            config_data = json.loads(args.config.read_text(encoding="utf-8"))
        else:
            config_data = dict(hidden_size=384, num_hidden_layers=4,
                               num_attention_heads=6, num_key_value_heads=2,
                               head_dim=64, intermediate_size=1024, vocab_size=6400,
                               prelude_layers=1, recurrent_layers=2, coda_layers=1,
                               loop_iters=32, mean_backprop_depth=8, max_recurrence=256)
        if config_data.get("use_moe", False) or config_data.get("residual_type", "standard") != "standard":
            raise ValueError("Audit supports dense standard residuals only")
        config_data.update(dropout=0.0, use_grad_checkpoint=0, flash_attn=True)
        report["config"] = config_data
        torch.manual_seed(123)
        model = InstinctForCausalLM(InstinctConfig(**config_data)).cuda().train()
        report["parameters"] = sum(p.numel() for p in model.parameters())
        ids = torch.randint(0, model.config.vocab_size, (args.batch, args.seq), device="cuda")

        def sync():
            torch.cuda.synchronize()

        def run(variant):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return loss_for(model, ids, ids, args.total, args.backward, variant)

        for variant in variants:
            configure_mode(model, variant)
            row = dict(variant=variant)
            try:
                model.zero_grad(set_to_none=True)
                row["saved_forward"] = saved_storage(model, lambda: run(variant))
                gc.collect()
                sync()
                for _ in range(args.warmup):
                    model.zero_grad(set_to_none=True)
                    seed()
                    loss = run(variant)
                    loss.backward()
                    del loss
                sync()
                times, peaks, forward_peaks, reserved = [], [], [], []
                for _ in range(args.steps):
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    seed()
                    sync()
                    row["baseline_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    loss = run(variant)
                    sync()
                    forward_peaks.append(torch.cuda.max_memory_allocated())
                    loss.backward()
                    sync()
                    times.append(1000 * (time.perf_counter() - start))
                    peaks.append(torch.cuda.max_memory_allocated())
                    reserved.append(torch.cuda.max_memory_reserved())
                    row["loss"] = loss.item()
                    row["finite_gradients"] = all(
                        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None
                    )
                    del loss
                row.update(median_ms=statistics.median(times),
                           forward_peak_mib=max(forward_peaks) / 2**20,
                           forward_backward_peak_mib=max(peaks) / 2**20,
                           reserved_peak_mib=max(reserved) / 2**20,
                           sample_ms=times)
                print(json.dumps(row), flush=True)
            except torch.OutOfMemoryError as exc:
                row["oom"] = str(exc).splitlines()[0]
                print(f"{variant}: OOM", flush=True)
            report["results"].append(row)
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
