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

The workspace limit excludes parameters, gradients, persistent momentum,
allocator reservation and CUDA library workspaces. At least one matrix must
fit; an unusually large single matrix can exceed the target. For FP32
512×1664 experts, the default processes four matrices at once. The limit is a
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
