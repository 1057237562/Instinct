# Fused Muon mantissa updates — 2026-10-18

The BF16 + uint16 weight representation now has a fused CUDA update path. The
default split-weight mode still uses FP32 momentum. Its numerical behavior matches the
existing foreach mantissa implementation; this optimization does not claim that
BF16 training itself is equivalent to training with FP32 master parameters.

The subsequently added **optional** BF16 momentum storage mode is documented in
[the BF16 momentum report](muon_bf16_momentum.md). The comparisons on this page
use FP32 momentum unless explicitly stated otherwise.

## Implementation

`trainer/muon_kernels.py` now promotes BF16 gradients inside the momentum kernel,
performs both lerps in FP32, and writes the BF16 NS input directly. The weight
update kernel loads the two weight halves, reconstructs FP32 in registers,
performs separately rounded FP32 decay followed by the same FMA addition, then
writes the high/low halves in place. No full-sized promoted gradient, precise
weight tensor, split intermediate or per-parameter copy kernel is required.

The raw CUDA stores increment parameter/mantissa tensor version counters so
cached inference expert stacks are invalidated. The two halves' storage and
Parameter identities stay unchanged. Mantissa format, checkpoint restore,
momentum precision, chunk sizes, NS coefficients/iterations, learning-rate
scaling and missing-gradient semantics remain the same.

Automatic dispatch is limited to contiguous validated shapes `(512,512)`,
`(128,512)`, `(1664,512)` and `(512,1664)` on PyTorch 2.13 / CUDA SM12 with Triton.
Other layouts, shapes, CPUs and devices retain the reference. Set
`INSTINCT_MUON_FUSED_UPDATES=0` to force the reference or `1` for explicit CUDA
experiments. The mantissa precision choice remains opt-in; ordinary FP32 training
does not switch precision. No extra persistent weights/state are introduced.

## Experiments actually run

Host: RTX 5070 Ti, torch 2.13.0+cu132, CUDA 13.2. Source hashes are in the JSON
artifacts, including the experimental script. No production corpus, weights or
checkpoint files were used. All batches contain seeded synthetic numeric tokens.

1. Re-ran the original 42,213,376-element two-block V1 benchmark before editing.
   At 64 MiB workspace, the unfused mantissa step took 13.053 ms; FP32 took
   7.322 ms. `experiments/muon_precision_before_fusion.json` also confirms exact
   reconstructed FP32 trajectories with shared gradients and FP32 momentum.
2. Initial fused run: 7.714 ms (40.9% less time), peak CUDA allocation
   486.959 -> 452.084 MiB. The experimental BF16-momentum adapter also improved
   from 14.634 to 8.670 ms; it still quantizes momentum and remains experimental.
   Artifact: `experiments/muon_precision_after_fusion.json`.
3. Alternating repeated A/B timings, profiler events, and 100 shared-gradient
   steps using all matrices in two V1 blocks. Every high half, low half and
   momentum tensor is compared at every step; absent gradients and changing LR
   are included. Artifact: `experiments/muon_mantissa_fusion_validation.json`.
   Three alternating rounds measured 16.001–16.512 ms for the reference and
   8.333–8.533 ms for fusion under deterministic settings. The profile recorded
   CUDA events decreasing from 1,859 to 793 and CPU operator events from 4,012
   to 399. These event counts include the NS graph and are not interpreted as
   exclusive kernel time. All 100 shared-gradient steps were bitwise equal.
4. Full **32-layer / 678,726,144-parameter** V1 MoE, normalized top-2, gradient
   checkpointing on, BF16 autocast, FP32 AdamW parameters, clipping at 1.0,
   identical seeded initialization and batches, LR 5e-4 -> 5e-5. Run reference,
   reference repeat and fused for 50 steps each, batch 1 x 256 tokens. This
   experiment uses deterministic algorithms, an eager model trunk and the same
   auto NS path for all arms. Exclude the first 10 steps from timing statistics.
   Artifact: `experiments/muon_mantissa_fusion_full_training.json`.

Full-model results:

| Phase | Whole step median (ms) | Optimizer median (ms) | Bitwise equal to reference |
|---|---:|---:|---|
| Foreach reference | 511.421 | 265.368 | yes |
| Foreach repeat | 473.879 | 241.354 | yes |
| Fused | 345.838 | 118.330 | yes |

The fused full step uses 27.0–32.4% less time than the two reference runs; its
optimizer uses 51.0–55.4% less time. These are measured configurations, not a
claim about arbitrary GPUs, sequence lengths or compiled production throughput.

All 50 recorded losses, final model bytes, complete optimizer tensor bytes and
held-out validation loss match **exactly** across all three runs. Final hashes:

- Model: `ced157140cd79ab04a96f25cc508d19b7b17fff320563e1177a3500209a7aa8d`
- Optimizer: `8ce184392a24569215d4dd811eeb14d83995f15542474bc54b05c3eb07011db3`

Training loss finished at 1.2433137894. Validation loss changed from 8.7729167938
to 8.8198528290 in **all** phases: the small repeated synthetic training set is
not a real pretraining quality benchmark. This experiment validates equivalence
of the optimized update, not generalization or BF16-vs-FP32 quality.

### Normal training settings with a larger batch

A second full-model run disables `torch.use_deterministic_algorithms`, uses
**batch 2 x 2048 tokens**, and takes 30 steps per phase (last 20 timed). The same
eager trunk, automatic Flex/NS backends, FP32 momentum, BF16 autocast, checkpointing,
clipping and LR decay are used. The script pins the cuBLAS workspace setting and
disables TF32 for reproducibility. Artifact:
`experiments/muon_mantissa_fusion_training_perf.json`.

| Phase | Whole step median (ms) | Optimizer median (ms) | Peak allocated GiB |
|---|---:|---:|---:|
| Foreach reference | 426.380 | 204.940 | 8.2491 |
| Foreach repeat | 408.294 | 193.479 | 8.2484 |
| Fused | 319.233 | 108.136 | 8.2475 |

This reduces whole-step time by **21.8–25.1%** and optimizer time by
**44.1–47.2%**, with no measured peak-memory increase. The activation/backward
peak dominates here, so the isolated optimizer's temporary-memory reduction does
not translate into a comparable reduction in the complete training peak.

Despite disabling deterministic algorithms, these particular three runs again
produced exactly equal per-step losses and final model/optimizer fingerprints.
Validation loss was 9.0642175674 -> 8.9819860458 in all three phases. This observed
equality is not a guarantee of deterministic training on other shapes/backends.

This tests the actual V1 model shape and optimizer wiring, using synthetic data.
Real corpus loading, packing mixtures, a compiled model trunk and longer schedules
can change total throughput; none of those costs is included in these claims.

### Comparison with the original FP32 master-parameter configuration

The 21.8–25.1% improvement above is relative to **unfused mantissa**, not FP32.
A subsequent ABBA comparison uses fresh subprocesses for FP32 / mantissa /
mantissa / FP32. Both use BF16 activation autocast, FP32 momentum, automatic
fused/NS kernels, the same full V1 model, synthetic inputs, initialization seed,
LR schedule, gradient checkpointing and batch 2 x 2048. Each runs 30 steps and
times the last 20. FP32 keeps its production 256 MiB workspace default; mantissa
keeps 64 MiB. The model trunk is eager in both arms. The mantissa initialization
uses the normal BF16 cast, so it does not retain the original FP32 low bits.

Reproducer: `experiments/compare_muon_fp32_mantissa.py`; complete results:
`experiments/muon_fp32_vs_mantissa.json`.

| Metric | Original FP32 | Fused mantissa, FP32 momentum |
|---|---:|---:|
| Per-run median whole step, ms | 323.241 / 337.290 | 332.903 / 322.595 |
| Mean of those two medians, ms | 330.265 | 327.749 |
| Mean optimizer median, ms | 107.309 | 114.794 |
| Maximum peak allocated GiB | 8.3236 | 8.2528 |
| Final held-out synthetic loss | 8.983993 / 8.983993 | 8.980528 / 8.981986 |

Whole-step timing ranges overlap: the 0.76% mean difference is too small to
claim a stable speedup from two runs. The mantissa optimizer is about 7.0%
slower, and full-training peak allocation falls by only **72.45 MiB / 0.85%**
in this configuration. FP32 repeats were exact here; independent mantissa
processes were not exact with deterministic algorithms disabled. The small
loss differences on two synthetic validation batches do not establish superior
quality or long-run equivalence. FP32 remains the production default.

Separate 12-step memory probes isolate forward, backward, clipping and update
peaks; these instrumented runs are not used for timing. Artifacts:
`experiments/muon_fp32_memory_phases.json` and
`experiments/muon_mantissa_memory_phases.json`.

| Phase peak, GiB | FP32 | Mantissa |
|---|---:|---:|
| Forward | 8.3236 | 8.2528 |
| Backward | 8.3155 | 8.2450 |
| Optimizer | 7.9780 | 6.6715 |

The observed overall peak is in forward, after `zero_grad(set_to_none=True)`.
Weights plus optimizer tensors total exactly 5,443,058,184 bytes in **both**
arms: hidden FP32 weights + FP32 momentum use 8 B/element, and split BF16
weights + uint16 low bits + FP32 momentum also use 8 B/element. Gradient tensors
are 2.5285 GiB versus 1.2704 GiB, saving 1.2581 GiB when resident. This explains
why the optimizer phase saves about 1.3 GiB but the full-training peak barely
changes. Shapes, microbatch size and gradient accumulation can move the peak;
theoretical tensor savings alone cannot predict it.

```powershell
python experiments/compare_muon_fp32_mantissa.py --output experiments/muon_fp32_vs_mantissa.json
python experiments/compare_muon_fp32_mantissa.py --arm fp32 --steps 12 --memory-phases --output experiments/muon_fp32_memory_phases.json
python experiments/compare_muon_fp32_mantissa.py --arm mantissa --steps 12 --memory-phases --output experiments/muon_mantissa_memory_phases.json
```

## Regression coverage and use

Kernel tests cover BF16 input gradients, FP32 lerp rounding at five momentum
settings, Nesterov on/off, transposed NS results, masked tails, randomly sampled
finite FP32 bit patterns, subnormals, signed zeros, and separate decay rounding.
Optimizer tests include real expert shapes, autocast, changing LR including zero,
missing gradients, preallocation, exact resume, noncontiguous fallback and
inference-stack invalidation.

The targeted suite completed with **125 passed**, including the CUDA cases;
`git diff --check` passed.

For a compatible opt-in mantissa run, auto fusion is the default. Keep using
`--optimizer muon --param_dtype bf16 --dtype bfloat16` with
`INSTINCT_MUON_MANTISSA=1`; the shared initializer keeps AdamW parameters FP32.
These fusion results preserve the selected precision. The later
`INSTINCT_MUON_MOMENTUM_DTYPE=bf16` option is separate and defaults to `fp32`.

```powershell
$env:PYTHONUTF8='1'
python -m pytest tests/test_muon_kernels.py tests/test_muon_mantissa.py tests/test_batched_muon.py tests/test_optimizer_precision.py tests/test_autotune.py -q
python experiments/validate_muon_mantissa_fusion.py --sections isolated,shared --output experiments/muon_mantissa_fusion_validation.json
python experiments/validate_muon_mantissa_fusion.py --sections train --output experiments/muon_mantissa_fusion_full_training.json
python experiments/validate_muon_mantissa_fusion.py --sections train --performance-only --steps 30 --batch 2 --seq 2048 --output experiments/muon_mantissa_fusion_training_perf.json
```
