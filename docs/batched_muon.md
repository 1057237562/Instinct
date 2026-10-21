# Batched Muon

`--optimizer muon` now defaults to shape-batched Muon in the shared optimizer
factory used by the trainers. Restart/resume training to pick up the change;
an already running process keeps its existing optimizer implementation.

Parameters are grouped by optimizer parameter group, device, dtype and exact
matrix shape. Each small chunk uses foreach momentum/parameter updates and
batched BF16 Newton–Schulz matrix multiplications. The normalization is per
matrix, never across experts. The momentum, Nesterov rule, five default
iterations, coefficients, learning-rate scaling and weight decay are unchanged.
Different GEMM/reduction paths can produce small BF16 rounding differences;
bitwise-identical training trajectories are not promised.

Model parameters are not repacked, and no persistent shadow weights are added.
Momentum stays in each parameter's `momentum_buffer`. Native Muon checkpoints
and the current `CombinedOptimizer` wrapper can resume directly, preserving
momentum and parameter ordering. Older in-repo Muon states without
`ns_coefficients` use the original coefficients. Existing incompatible parameter
group layouts remain subject to the factory's existing resume checks.

Runtime settings (set before starting the trainer/WebUI):

| Environment variable | Default | Meaning |
|---|---|---|
| `INSTINCT_MUON_BACKEND` | `batched` | Set `native` to use the previous native/fallback optimizer |
| `INSTINCT_MUON_BATCH_SIZE` | `16` | Maximum matrices per chunk |
| `INSTINCT_MUON_WORKSPACE_MB` | `64` | Conservative temporary tensor budget in MiB |

On validated Blackwell/PyTorch devices the factory chooses 256 MiB for FP32
parameters unless explicitly overridden; other paths use 64 MiB.

Both limits apply together: increasing `INSTINCT_MUON_BATCH_SIZE` alone cannot
overcome the workspace limit. For the current split-weight estimator, a
1664x512 (or 512x1664) expert needs an estimated 16.625 MiB per matrix. The
64 MiB default therefore allows three matrices; 256 MiB allows fifteen.
To opt into sixteen without changing precision or optimizer mathematics:

```powershell
# Optional, in the shell that launches the low-precision trainer:
$env:INSTINCT_MUON_BATCH_SIZE='16'
$env:INSTINCT_MUON_WORKSPACE_MB='272'
```

This is a workspace target, not an allocation made up front. CUDA graph pools
can retain additional memory across shapes. The setting applies to the next
optimizer construction, including resume; it does not alter a running job.
See [the BF16-momentum chunk measurements](muon_bf16_momentum.md#optional-larger-chunks)
for full-step timing, memory and correctness checks. Defaults remain unchanged.

## Experimental BF16 split weights

Production keeps FP32 parameters by default. `INSTINCT_MUON_MANTISSA=1` is an
explicit experiment requiring `--optimizer muon --param_dtype bf16 --dtype bfloat16` and the
`batched` backend. Shared `init_model` keeps hidden matrices in BF16 and the
AdamW parameters (embeddings, tied heads, norms, biases) in FP32. Custom model
construction should call `cast_training_parameters(model, 'bf16')` with this
environment set before building optimizers or wrapping DDP/compile. The factory
rejects unprotected low precision AdamW parameters and incompatible backends.

Each Muon weight has a BF16 high half and uint16 low half; momentum defaults to FP32.
Reconstructing that pair is bit-exact, but the forward uses a truncation toward
zero, unlike BF16 autocast's nearest rounding. Gradients also reside in BF16.
Equal optimizer updates under shared gradients do not imply equal full training.
The pair itself uses the same four bytes as an FP32 parameter. Including
gradients and momentum costs 10 B/element versus 12 B for FP32; ignoring
gradients, both cost 8 B. CUDA graphs, activations and workspaces are additional.

Resume checkpoints record the split format and require the same mantissa mode.
Original unversioned checkpoints containing `mantissa` are recognized too.
Disabling the mode, using the native backend, changing the optimizer layout or
loading corrupted low bits raises instead of silently losing precision. Keep
both model high bits and optimizer low bits in the full resume checkpoint;
weight-only exports cannot resume the full precision accumulator. Legacy
non-mantissa states can initialize zero low bits, but cannot recover precision
already lost when weights were cast.

### Optional BF16 momentum

`INSTINCT_MUON_MOMENTUM_DTYPE=fp32|bf16` selects momentum storage **only when
mantissa is enabled**. The default is `fp32`; existing launches and configurations
are unchanged. To explicitly select BF16 weights + uint16 low bits + BF16 momentum:

```powershell
$env:INSTINCT_MUON_MANTISSA='1'
$env:INSTINCT_MUON_MOMENTUM_DTYPE='bf16'
# Add to the intended training command:
# --optimizer muon --param_dtype bf16 --dtype bfloat16
```

Momentum/Nesterov arithmetic is FP32. The current update consumes the unrounded
FP32 EMA; only the persistent EMA is rounded to BF16 for the next step. The fused
kernel promotes in registers and stores directly to BF16, avoiding a full FP32
momentum copy. CPU/unsupported cases use a per-chunk FP32 reference workspace.
Non-Muon AdamW parameters remain FP32. Including gradients, the Muon storage
cost is 8 B/element; without gradients it is 6 B.

The checkpoint format records momentum precision and rejects changing it during
resume, including before state has been allocated. Keep the same option when
resuming. Changing precision requires a deliberately new stage from weights.
Historical research checkpoints use their own format and remain rejected.

BF16 momentum quantizes the optimizer trajectory; matching the reference BF16
algorithm does not mean matching FP32-momentum training. See
[BF16 momentum validation and comparison](muon_bf16_momentum.md) and
[the original audit](muon_low_precision_audit.md).

### Fused split-weight updates

For opt-in mantissa mode, `INSTINCT_MUON_FUSED_UPDATES=auto` now uses fused
momentum/update kernels on the same validated PyTorch 2.13 / SM12 CUDA shapes
as FP32 Muon. Gradient promotion happens in registers; a second kernel
reconstructs the split FP32 accumulator, applies decay and the update, and writes
both halves in place. It preserves separate FP32 decay rounding and FMA addition,
FP32 update arithmetic and the original NS transform. Tensor version counters advance so
inference weight stacks invalidate correctly.

The parameter/state layout, chunk budget, learning-rate schedule and `grad=None`
behavior are unchanged. CPU, unsupported shapes/devices and noncontiguous tensors
keep the foreach reference. `INSTINCT_MUON_FUSED_UPDATES=0` explicitly selects
that reference. No optimizer state or Python learning rate is CUDA-graph captured.

Full 32-layer V1 MoE deterministic training, shared-gradient comparisons and
kernel/lifecycle tests validate equivalence to the unfused mantissa algorithm.
See [fusion correctness and performance results](muon_mantissa_fusion.md).

### Instinct V2 fast path

On the validated PyTorch 2.13 / Blackwell stack, `auto` also covers V2's
1536x1536, 384x1536, 5504x1536 and 1536x5504 matrices. Momentum/Nesterov and
split-weight updates are fused. NS uses the Triton polynomial/update kernels;
the 1536x5504 Gram product and single-matrix 384x1536 tail retain cuBLAS's
reference reduction order. Replacing those products with the triangular
kernel changed BF16 NS results, so they are deliberately preserved.

V2 selects `triton` without retaining NS CUDA graphs, preserving memory for
activations on 16 GiB cards. V1 keeps its existing `triton_graph` selection.
Explicit backend overrides and unsupported-device/shape fallbacks still apply.
No checkpoint or optimizer-state migration is required.

Local RTX 5070 Ti validation (`experiments/muon_v2_validation.json`): all 224
V2 hidden matrices, BF16 momentum, mantissa weights, 256 MiB workspace. Across
two trials in alternating order, optimizer median CUDA time fell from
630.9/635.5 ms to 404.6/402.9 ms (about 36% less time). The diagnostic CUDA
event count fell from 9305 to 4229. These are optimizer-only measurements;
with accumulation 8, the isolated saving amortizes to about 29 ms/microstep.
Do not claim the same percentage improvement for whole training.

The 32-layer deterministic BF16 training comparison uses synthetic sequences
of length 128 and 12 updates, with full-layer checkpointing. Reference,
reference-repeat and auto produced identical losses, held-out loss, model
SHA-256 and optimizer-state SHA-256. Shape-level tests additionally cover
random/zero/rank-one NS inputs, tail batches, FP32 and both mantissa momentum
precisions, changing LR/momentum, missing gradients and checkpoint reloads.
The same 32-layer comparison also passes with tensorwise FP8 (160 Linear
layers) and `max-autotune-no-cudagraphs`: all three trajectories are bitwise
equal. See `experiments/muon_v2_fp8_compiled_validation.json`; reproduce with
the command below plus `--sections train --fp8 tensorwise --compile-mode
max-autotune-no-cudagraphs` and a separate output path.

```powershell
python -m pytest tests/test_muon_v2.py tests/test_muon_kernels.py tests/test_muon_bf16_momentum.py tests/test_muon_mantissa.py tests/test_batched_muon.py -q
python experiments/validate_muon_mantissa_fusion.py --config trainer/config_instinct_v2.json --compare-ns --full-optimizer --workspace-mb 256 --momentum-dtype bf16 --sections isolated,shared,train --isolated-trials 2 --isolated-warmup 3 --isolated-steps 10 --shared-steps 12 --steps 12 --batch 1 --seq 128 --checkpoint 2 --output experiments/muon_v2_validation.json
```

The workspace limit excludes parameters, gradients, persistent momentum,
allocator reservation and CUDA library workspaces. At least one matrix must
fit; an unusually large single matrix can exceed the target. For FP32
512×1664 experts, 64 MiB permits four matrices; the validated factory default
of 256 MiB permits sixteen. The limit is a
runtime property, so loading a checkpoint does not restore a larger old limit.

For a short isolated benchmark (run while training is stopped):

```powershell
python experiments/bench_batched_muon.py --device cuda --matrices 96 --steps 20 --profile
```

The benchmark compares native/batched optimizer wall time after warmup and
reports temporary peak allocated memory. It uses expert/attention matrix shapes
without loading data or writing checkpoints. This is optimizer-only speedup;
verify whole-training useful tokens/s separately after resuming.

Targeted checks:

```powershell
python -m pytest tests/test_batched_muon.py tests/test_optimizer_precision.py --skip-gpu -q
python -m pytest tests/test_batched_muon.py -m gpu -q
```
