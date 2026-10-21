# Optional BF16 momentum for split-weight Muon

BF16 weights + uint16 low bits + BF16 momentum is now an explicit option.
The default remains FP32 parameters and FP32 momentum. Enabling split weights
alone still defaults to FP32 momentum. Existing launch/panel configurations were
not changed and no production training was started.

## Select the option

Config WebUI: open **04 · 精度与性能**, select optimizer `muon` and parameter
precision `bf16`, enable **Muon BF16 补偿权重**, then select momentum `bf16`.
The optional **使用 16 矩阵 / 272 MiB 分块** button changes only the chunk limits.
The controls persist with panel state, templates and pipeline snapshots; the
command preview includes the exact CLI selections. FP32 momentum and disabled
split weights remain the defaults. Disabled controls retain their saved UI
values but do not enable low precision for other optimizers.

All shared trainers also accept `--muon_mantissa 0|1`,
`--muon_momentum_dtype fp32|bf16`, `--muon_batch_size N` and
`--muon_workspace_mb MiB`. Explicit CLI values override their environment
equivalents. Zero chunk/workspace values, or omitted arguments, retain the
environment/platform defaults. Autotune translates the performance aliases
into its existing environment search axes; precision remains a fixed context.

```powershell
$env:INSTINCT_MUON_MANTISSA='1'
$env:INSTINCT_MUON_MOMENTUM_DTYPE='bf16'
# Include with the intended trainer's existing arguments:
# --optimizer muon --param_dtype bf16 --dtype bfloat16
```

`INSTINCT_MUON_MOMENTUM_DTYPE` accepts `fp32` (default) or `bf16`. BF16 requires
the batched Muon mantissa mode; incompatible selections fail explicitly.
The Python API is `BatchedMuon(..., use_mantissa=True, momentum_dtype='bf16')`.
Shared model initialization keeps embeddings, tied heads, norms and biases in
FP32 for AdamW. To select FP32 momentum in a split-weight run, use `fp32` or omit
the momentum setting. These are startup choices, not a live training toggle.

## Exact arithmetic and storage policy

1. Load the stored BF16 EMA and the BF16 gradient as FP32.
2. Compute the new EMA and Nesterov direction in FP32. This step's direction uses
   the **unrounded** new EMA.
3. Store the EMA rounded to BF16 for the next step. NS retains its existing BF16
   matrix operations and coefficients.
4. Reconstruct the FP32 weight from its BF16 high half + uint16 low half, perform
   the existing FP32 decay/addition, and split the result back without losing bits.

The CUDA kernel promotes values in registers and writes BF16 momentum directly.
It does not allocate a persistent FP32 momentum shadow or promote/demote the
whole momentum tensor at every step. CPU, unsupported hardware/shapes and
noncontiguous layouts use a per-chunk FP32 reference workspace. Existing
SM12/PyTorch 2.13 auto dispatch, explicit FMA and separate decay rounding apply.

This preserves the prior research algorithm that rounded momentum after each
step, while avoiding its FP32 storage/rounding copies. It is not equivalent to
keeping momentum in FP32 or computing every EMA operation directly in BF16.
BF16-momentum quantization still changes the optimizer trajectory.

For each Muon element, weights + low bits + gradients + momentum use **8 B**
(2+2+2+2). With gradients cleared, weights + low bits + momentum use **6 B**.
Persistent-memory estimates and preallocation use the selected dtype. Autotune
fingerprints include this setting. Resume metadata distinguishes BF16 and FP32
momentum, preserves raw uint16 bits and rejects a precision switch or corrupt
state before loading. Keep the saved setting for `--from_resume 1`; deliberately
changing precision requires a new stage from weights. Original research-format
checkpoints are not silently migrated.

## Validation actually run (2026-10-18)

Machine: RTX 5070 Ti, torch 2.13.0+cu132, CUDA 13.2.

- CPU/CUDA tests compare the production implementation to the independent old
  FP32-temporary + BF16-rounding adapter. Nesterov on/off, dynamic momentum/LR,
  missing gradients, transpose/noncontiguous layout and stable buffer pointers
  are covered.
- The fused prepare kernel is checked against FP32 arithmetic followed by BF16
  storage at momentum 0, 0.5, 0.8, 0.95 and 1.
- Checkpoint tests round-trip real CPU/CUDA BF16 state through `torch.save/load`,
  verify continuing updates, preallocation and precision-change rejection.
- Factory tests verify that **both old defaults remain unchanged**, explicitly
  selected BF16 storage totals 8 B/element, and the estimator matches allocation.
- 100 shared-gradient steps on two complete V1 blocks compare every weight high
  half, low half and momentum tensor: all are bitwise equal to the BF16 reference.
- Full 32-layer / 678,726,144-parameter normalized top-2 V1 MoE, 50 steps per arm,
  deterministic reference / repeat / fused, batch 1 x 256, BF16 autocast, FP32
  AdamW, gradient checkpointing and clipping. All recorded training losses,
  final model bytes, optimizer-state bytes and validation losses are identical.

Artifact: `experiments/muon_bf16_momentum_validation.json`. Final hashes:

- Model: `a20955cf068996eab102f6505ce2ce5112968a45a455dfc1d7dba2e164155f82`
- Optimizer: `7155b1122f8eb881fab2804f838449958f9cb0dbfbf09bc829ab8ac8c2d30f71`

This equality is to the **BF16-momentum algorithm**, not FP32-momentum training.
The tiny repeated synthetic set overfits: validation goes from 8.7729168 to
8.8196692 in every arm. These tests establish implementation equivalence, not
real-data pretraining quality.

## Comparison with the original FP32 configuration

Four fresh subprocesses in FP32 / BF16-momentum / BF16-momentum / FP32 order.
Each uses the same initial seed and synthetic tokens, full V1 model, batch
2 x 2048, BF16 autocast, FP32 AdamW, gradient checkpointing, clipping and LR
decay from 5e-4 to 5e-5. There are 30 steps per run; the last 20 are timed.
The model trunk is eager; Flex/NS kernels use the existing automatic backends.
Deterministic algorithms are disabled. Factory defaults are retained: FP32
workspace 256 MiB, split-weight workspace 64 MiB.

| Metric | Original FP32 | Optional BF16 weights + uint16 + BF16 momentum |
|---|---:|---:|
| Step median by run, ms | 300.487 / 302.011 | 306.834 / 334.448 |
| Mean of the two medians, ms | 301.249 | 320.641 |
| Mean optimizer median, ms | 99.037 | 110.281 |
| Peak allocated GiB | 8.3236 | 6.9635 |
| Measured Muon tensor bytes/element, including gradients | 12 | 8 |
| Stored momentum tensor dtype | FP32 | BF16 |
| Final synthetic validation loss | 8.983993 | 8.982230 |

Artifact: `experiments/muon_fp32_vs_bf16_momentum.json`.
Measured peak allocation drops by **1.3601 GiB (16.34%)**. The average whole-step
time is **6.44% longer** in these runs; the BF16 timings show run-to-run variation,
so this is not a stable universal slowdown estimate. Both repeats within each
precision mode produced identical losses and final hashes in this particular run.
The short synthetic validation difference does not prove a quality improvement.

The earlier 72 MiB peak saving concerned **FP32 momentum** with split weights.
BF16 momentum removes another two resident bytes per Muon element, so its memory
benefit exists during forward even after gradients have been cleared. The full
training peak is measured here rather than inferred from the tensor-byte count.

## Reproduction

```powershell
$env:PYTHONUTF8='1'
python -m pytest tests/test_muon_bf16_momentum.py tests/test_muon_mantissa.py tests/test_muon_kernels.py tests/test_batched_muon.py tests/test_optimizer_precision.py tests/test_autotune.py -q
python experiments/validate_muon_mantissa_fusion.py --sections shared,train --momentum-dtype bf16 --output experiments/muon_bf16_momentum_validation.json
python experiments/compare_muon_fp32_mantissa.py --momentum-dtype bf16 --output experiments/muon_fp32_vs_bf16_momentum.json
```

Targeted suite: 153 tests passed, including CUDA. `git diff --check` passed.
No datasets, production checkpoints or saved training-panel settings were changed.

## Optional larger chunks

The existing runtime controls can increase low-precision chunks without changing
the BF16 momentum option, checkpoint format or arithmetic. Neither the 64 MiB
low-precision default nor saved training-panel configurations were changed.

```powershell
$env:INSTINCT_MUON_MANTISSA='1'
$env:INSTINCT_MUON_MOMENTUM_DTYPE='bf16'
$env:INSTINCT_MUON_BATCH_SIZE='16'
$env:INSTINCT_MUON_WORKSPACE_MB='272'
# Use with --optimizer muon --param_dtype bf16 --dtype bfloat16
```

For either V1 expert orientation (1664x512 or 512x1664), the current conservative
estimate is 16.625 MiB per matrix. Both the matrix-count cap and workspace limit
apply. 64 MiB/cap16 gives chunk3; 144 MiB/cap8 gives chunk8; 272 MiB/cap16 gives
chunk16; 544 MiB/cap32 gives chunk32. 256 MiB would give chunk15, so it does not
quite match the original FP32 configuration's chunk16. These numbers describe
the present estimator, not a promise of exact CUDA allocation. Other shapes
and small buckets have different effective chunk sizes.

### Measurements on 2026-10-18

Isolated sweep: four V1 blocks, fixed synthetic gradients, 15 warmup + 40 timed
steps per candidate, forward then reverse candidate order. Every candidate's
final weight/low-bit/momentum bytes matched. Optimizer times (ms) were:

| Expert chunk | First pass | Reverse pass |
|---|---:|---:|
| 3 | 14.888 | 15.878 |
| 8 | 14.767 | 14.908 |
| 16 | 14.633 | 14.909 |
| 32 | 14.820 | 14.868 |

32 did not establish a speed advantage over 16 and used more memory.
Artifact: `experiments/muon_low_precision_chunk_sweep.json`.

Full-step comparison: full 32-layer normalized top-2 V1, batch 2x2048, BF16
autocast, checkpointing and clipping, eager model trunk, identical seeds and
synthetic tokens, 30 steps/run with the last 20 timed. First consecutive triplet:

| Mode | Expert chunk | Step median, ms | Optimizer median, ms | Peak allocated GiB |
|---|---:|---:|---:|---:|
| Original FP32 | 16 | 293.870 | 97.611 | 8.3236 |
| BF16 momentum, 64 MiB | 3 | 294.782 | 103.215 | 6.9635 |
| BF16 momentum, 272 MiB | 16 | 286.018 | 94.461 | 7.0181 |

In this consecutive pair, chunk16 reduced optimizer time by **8.48%**, whole-step
time by **2.97%**, and added **55.87 MiB** to peak allocation. It retained **1.3055
GiB** of peak savings against original FP32. Larger chunks do not change the
8 B/element Muon state/gradient representation; their extra cost is workspace
and retained graph storage.

All four low-precision runs, including repeats, produced identical loss arrays,
final model hashes, optimizer hashes and validation loss (8.9822302). This checks
batching equivalence in the measured runs, not real-data pretraining quality.
The reverse chunk16 run was 303.667 ms/step and included a 787.9 ms outlier.
The harness was interrupted afterward; the final chunk3 and FP32 runs resumed
about 37 minutes later and measured 328.126 and 329.634 ms respectively. Raw
results are retained in `experiments/muon_chunk_full_training.json`, including
the interruption note. A six-run average would mix different conditions; the
first pair's 2.97% improvement is not a universal performance guarantee.

Additional CPU/CUDA tests check the chunk controls, chunk3 versus chunk16 with
remainder batches, missing gradients, changing LR/Nesterov and resume into a
different workspace size. Both FP32 and BF16 momentum are covered.
The targeted six-file suite now passes **159 tests**, including CUDA.

```powershell
python experiments/tune_muon_low_precision_chunks.py --output experiments/muon_low_precision_chunk_sweep.json
python experiments/tune_muon_low_precision_chunks.py --training --steps 30 --warmup 10 --output experiments/muon_chunk_full_training.json
```
