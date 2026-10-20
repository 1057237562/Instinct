# Muon low precision audit — 2026-10-18

The weight split/reconstruction and FP32 update arithmetic are correct under
identical input gradients. The branch does **not** establish FP32-equivalent
training or a globally minimum-memory production optimizer. Keep FP32 as the
default and make reduced storage choices explicit. This page records the initial
audit; the subsequently added optional BF16 momentum factory mode is documented
in [the BF16 momentum report](muon_bf16_momentum.md).

**Follow-up:** the original PyTorch mantissa update has since been fused and
validated against the unfused implementation on the full 32-layer V1 MoE.
See [the fusion report](muon_mantissa_fusion.md) for current timing and exactness
results. The timings below describe the implementation **before** that optimization.
The follow-up also compares against original FP32 using full V1 training:
whole-step timing is essentially tied, and the measured peak falls only 72 MiB
at batch 2 x 2048 because forward, with gradients cleared, is the memory peak.
The 1.258 GiB gradient-storage saving is real but is not the peak-memory saving.

## What was merged and what its evidence establishes

Merge `49a06d5` incorporated `origin/feat/muon-low-precision` through `8011da3`:

- `41e91db`: opt-in BF16 high bits + uint16 low bits, FP32 momentum, optimizer
  state restoration, persistent memory estimator and correctness tests.
- `97153d8`: parameter trajectory experiment with shared-gradient and independent
  training modes; BF16/24-bit momentum **value simulations**.
- `0753972`: isolated optimizer timing, size sweep and an unfused prototype.
- The V2 config, compute/VRAM survey and dataset documentation removals are
  separate changes; they do not validate low precision Muon. The survey describes
  a historical top-1 model. Current V1 uses normalized top-2 routing.

The checked-in `experiments/bench_muon_mantissa_scaling.json` and `.out.txt`
identify an RTX 5060 Ti 8 GB, torch 2.11.0+cu130 and CUDA 13.0. At 4.72 M elements,
mantissa took 4.765 ms versus 2.709 ms (1.759x); at 20.48 M, 23.310 versus
20.356 ms (1.145x); the 8192-square case took 366.271 versus 354.633 ms (1.033x).
These are isolated matrix updates, not full-model training speed. The shrinking
ratio also changes matrix dimensions and NS cost; it cannot be extrapolated just
from a model's parameter count. The additional reconstruct/split kernels explain
overhead in that implementation, not a lower bound on a future fused kernel.

No checked-in trajectory JSON, long-run validation curve or full V1 BF16-momentum
memory trace was found in the merged evidence. The delta script applies Muon to
all 2D parameters, including embeddings, and freezes 1D parameters. Its train
reference uses FP32 forward while the BF16 arms use BF16 forward without the same
autocast policy. These are diagnostic experiments, not the production
Muon-hidden-matrices + AdamW-other-parameters training protocol.

In particular, `_quantize_momentum_bf16` used `fp32_buffer.copy_(buffer.bfloat16())`.
The persistent tensor remained FP32. `mom24` likewise masked an FP32 tensor's low
bits; it did not implement three-byte storage. Their claimed 8/9 B tiers were
storage models, not measured resident representations. The old speed benchmark
also retained a reconstructed sanity-probe tensor while measuring memory; this
audit releases that probe before measurement.

## Numerical limits and production fixes

The representation concatenates the BF16 high 16 bits and uint16 low 16 bits of
an FP32 value. It retains the complete FP32 accumulator, including updates smaller
than a BF16 ULP. This is **truncation toward zero**, not nearest rounding: for
example, FP32 `1.006` is exposed as BF16 `1.0`, whereas a nearest BF16 cast is
`1.0078125`. Internal exactness therefore does not establish identical forward,
backward, gradient accumulation, routing or long-run convergence. Standard BF16
weights without compensation can lose small updates; this issue is also examined
in [Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192).

The audit reproduced a resume bug: loading mantissa state with the switch off
silently cast both FP32 momentum and uint16 bits to BF16, then ignored the bits.
The following protections are now implemented:

- Versioned split-format metadata, recognition of old unversioned mantissa
  states, and checks for low-bit dtype/shape/missing data. The loader avoids
  creating lossy temporary BF16 copies of saved tensors.
- Reject a mode/backend mismatch before dropping low bits, including native
  factory optimizers and `CombinedOptimizer` layout fallback paths.
- Keep non-Muon trainable parameters in FP32 in shared model initialization.
  Direct factory callers supplying BF16 AdamW parameters are rejected. Muon
  compensation never protected those parameters, even at a large starting LR.
- Exempt only compensated matrices from the small-LR guard; retain the guard
  elsewhere. Reject malformed mantissa environment settings and unsupported
  backend/optimizer combinations. Defaults remain off / FP32.
- Include the mantissa switch in the autotune training fingerprint.

At the initial audit, the production split mode kept FP32 momentum. Research BF16 momentum promotes
one chunk to FP32, performs momentum/Nesterov/NS and the weight update, then really
stores the momentum as BF16. Its arithmetic matches the former rounding
simulation. This differs from rounding momentum before computing Nesterov or
performing the lerp itself in BF16. Research checkpoints use a different format
tag and are rejected by the production loader.

Muon should operate on hidden matrices; embeddings and heads need a separate
optimizer, consistent with the [PyTorch Muon documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html).

## New local measurements

Artifacts:

- `experiments/muon_precision_audit_storage.json`
- `experiments/muon_precision_audit_training.json`
- `experiments/muon_precision_audit_workspace64.json` (same workspace target)
- Reproducer: `experiments/audit_muon_precision.py`

Machine: RTX 5070 Ti, torch 2.13.0+cu132, CUDA 13.2. Artifacts record the parent
revision and SHA-256 of the source files at measurement time. The audit was run
on working-copy changes after `49a06d5`; they are not results for that unmodified
commit. Subsequent checkpoint checks do not change optimizer arithmetic.

### Real storage and optimizer peak

Two complete V1 blocks: 42,213,376 hidden-matrix elements, including all eight
experts per block. 20 warmup and 50 timed steps, CUDA events with synchronization.
Uses factory workspace defaults: FP32 256 MiB, BF16 64 MiB, auto NS/fused kernels.
FP32 has its validated fused update path; split weights use the unfused path.
CUDA graph storage and chunk sizes differ. This is a comparison of current
execution policies, not an isolated precision-only speed comparison.

| Representation | Actual tensor B/element, incl. gradients | Tensor MiB | Steady CUDA MiB | Step peak CUDA MiB | Median step ms |
|---|---:|---:|---:|---:|---:|
| FP32 parameter/gradient/momentum | 12 | 483.094 | 634.098 | 660.098 | 6.748 |
| BF16 + uint16, FP32 momentum | 10 | 402.578 | 446.459 | 486.959 | 13.754 |
| BF16 + uint16, actual BF16 momentum | 8 | 322.063 | 363.443 | 408.818 | 14.231 |
| Direct BF16, no compensation | 6 | 241.547 | 312.428 | 332.428 | 6.905 |

Steady/peak CUDA includes optimizer graph allocations beyond owned tensors.
Reserved allocator memory is separately recorded in JSON. These numbers exclude
forward/backward activations and must not be presented as full-training peaks.

A separate run fixing **all workspaces to 64 MiB** measured FP32 at 7.278 ms /
566.598 MiB peak, split/FP32 momentum at 13.793 ms / 486.959 MiB, and split/BF16
momentum at 13.123 ms / 408.818 MiB. Thus the measured memory reduction and
roughly 1.8–1.9x step cost persist with the same workspace target. Timing between
the two split variants varies across runs; these samples do not establish which
one is faster. Neither benchmark includes full training or long-term allocator
behavior.

### Shared-gradient correctness

100 steps, BF16-valued shared gradients, seeded exact initial high/low bits,
weight decay 0.1, Nesterov, mixed square/tall/wide shapes, and missing gradients.

| Mode | LR 5e-4: error / reference displacement | LR 2e-5: error / reference displacement | Exact steps |
|---|---:|---:|---:|
| Split + FP32 momentum | 0 | 0 | 100/100 |
| Split + BF16 momentum | 0.005868 | 0.005866 | 1/100 |
| Direct BF16 | 0.121786 | 1.021878 | 0/100 |

The BF16-momentum error is about 0.59% of the update displacement, **not** a
0.59% loss/accuracy penalty. Internal reconstructed weights were compared, not
just their visible BF16 high halves. Production-shape CUDA tests also compare
actual BF16 gradients against the same values in FP32, and check autocast,
serialization, preallocation and missing-gradient behavior.

### Short training, same autocast and optimizer partition

100 steps for each of 4 arms x 3 seeds x 2 architectures (tiny Dense and normalized
top-2 MoE), BF16 autocast for every arm, FP32 AdamW parameters, checkpointing on,
gradient clipping and LR decay 5e-4 to 5e-5. Training uses repeating period-8
numeric token IDs; validation uses eight held-out batches from the same synthetic
distribution. No training corpus, checkpoint or running trainer was modified.

The independent FP32 repeat exactly reproduced the recorded losses for all six
seed/architecture pairs. All runs had finite losses and improved validation loss.

| Candidate | Dense mean validation-loss increase vs FP32 | MoE mean increase vs FP32 | Largest increase across seeds |
|---|---:|---:|---:|
| Split + FP32 momentum | +0.000297 | +0.000429 | +0.000674 |
| Split + BF16 momentum | +0.000313 | +0.000509 | +0.000698 |

Both candidates were slightly worse than their paired FP32 control on all six
probes. This is a small, synthetic, short-run result, not a statistical equivalence
test or approval for real pretraining. No acceptable V1 loss tolerance was supplied
or inferred. Full V1 data, longer schedules, real validation and complete training
peaks remain necessary before changing defaults.

## Does BF16 + uint16 + BF16 save the most VRAM?

Interpreting the last BF16 as **momentum storage**, this is the 8 B/element tier:
2 B weights + 2 B low bits + 2 B gradients + 2 B momentum. It is smaller than the
10 B split/FP32-momentum tier and 12 B current FP32 baseline. The two weight halves
alone still occupy 4 B; their benefit is preserving accumulation while allowing
BF16 parameter/gradient storage, not compressing an arbitrary FP32 value below
32 bits.

For V1's 675,414,016 Muon elements + 3,312,128 FP32 AdamW elements, analytically:

| Mode | Parameters + gradients + optimizer state | Saving vs FP32 |
|---|---:|---:|
| FP32 | 7.598 GiB | — |
| Split + FP32 momentum | 6.340 GiB | 1.258 GiB |
| Split + BF16 momentum | 5.082 GiB | 2.516 GiB |

These are tensor accounting estimates for the full model, not measured full-model
CUDA peaks. After `zero_grad(set_to_none=True)`, the 10 B mode loses its persistent
advantage over FP32 because both weights+momentum use 8 B; at a step with gradients
resident, its advantage returns. Activations, cast weights, graph pools, temporaries,
fragmentation and gradient accumulation affect the actual peak.

It is not an absolute minimum: direct BF16 is 6 B but loses updates; lower-bit
momentum, stochastic rounding, offload or optimizer changes can change the budget
and the math. None is established here as a superior validated production choice.
The 8 B tier is a useful experiment, not a demonstrated best quality/memory tradeoff.

## Reproduction

```powershell
$env:PYTHONUTF8='1'
python -m pytest tests/test_muon_mantissa.py tests/test_batched_muon.py tests/test_optimizer_precision.py tests/test_autotune.py tests/test_muon_kernels.py -q
python experiments/audit_muon_precision.py --sections storage,trajectory --output experiments/muon_precision_audit_storage.json
python experiments/audit_muon_precision.py --sections train --steps 100 --output experiments/muon_precision_audit_training.json
```

Run GPU measurements without another training workload. For a same-workspace
comparison, add `--workspace-mb 64` to the storage command. No BF16-momentum
production switch or automatic precision migration was introduced by that initial
audit. The later optional BF16 momentum mode has its own validation report above.

Audit validation: 105 targeted tests passed, including CUDA expert shapes and
kernels, exact state serialization, missing gradients, factory guards, and
autotune contracts. `git diff --check` passed. This does not replace a long-run
quality/throughput/peak-memory comparison on the real V1 training corpus.
