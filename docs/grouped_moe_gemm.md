# Device-offset MoE GEMM

The shared MoE path now chooses a real grouped kernel rather than assuming that
the presence of `torch.nn.functional.grouped_mm` guarantees a fast backend.
The installed PyTorch 2.13 native BF16 dispatcher uses a host-offset fallback on
SM120: each forward/dX/dW call copies offsets to CPU and loops over experts.

`model/grouped_mm.py` creates one routing plan per FFN invocation. Gate, up and
down projections and their backward passes share it. Backend selection:

| `INSTINCT_GROUPED_MM_BACKEND` | Behavior |
|---|---|
| `auto` (default) | Native on non-Windows SM90/SM100; otherwise Triton on NVIDIA SM80+ when installed; cached PyTorch fallback otherwise |
| `triton` | BF16 device-offset kernels for forward, dX and dW; requires CUDA + Triton |
| `cached` | Read offsets once, reuse Python boundaries for all projections and gradients |
| `native` | Original PyTorch grouped-mm, for comparisons or higher-order differentiation |

For the current RTX 5070 Ti / Windows / Triton 3.4 environment, `auto` selects
Triton. Kernels use conventional loads and FP32 dot accumulation, without TMA,
offsets `.item()` / `.cpu()`, device-dependent host loops, or float atomics for
weight-gradient reduction. Empty experts get zero weight gradients. Routing
continues to be stable-sorted and supports top-k assignment. The compact MoE
region remains outside Inductor to avoid the known grouped-weight-gradient TMA
lowering failure; the surrounding model can still use `torch.compile`.

Checkpoint keys, original 2D expert parameters, Muon state layout, and optimizer
updates remain unchanged. There are no persistent packed weight copies or new
trainable parameters. Float rounding can differ from native cuBLAS, including
its optional reduced-precision intermediate reductions. Tests compare outputs
and gradients with both native BF16 and independent FP32 references. The new
custom backward supports first-order training, not higher-order derivatives.

With 32 layers and FFN checkpointing, native grouped-mm performs 384 offsets
readbacks per step. The cached path needs 64 (one per forward/recompute); the
Triton path needs zero. This excludes other trainer/logging synchronization.

Set the environment before launching the trainer (or the WebUI that launches
it). Existing Python processes do not hot-reload the change. No package upgrade
is needed in the current environment. If Triton is absent, `auto` uses cached;
kernel errors are not silently swallowed. `cached` is an explicit escape hatch.

## Validation

```powershell
python -m pytest tests/test_grouped_mm.py tests/test_moe_dispatch.py -q
python -m pytest tests/test_shared_attention_bias.py -k compiled_moe_model --run-slow -q
python experiments/bench_grouped_mm.py --checkpoint --profile
```

The benchmark uses one identical 512-hidden, 1664-FFN, 8-expert module with
11,712 tokens, alternates backend order after warmup, and includes backward.
It neither reads a dataset nor changes checkpoints. Run it with training stopped.
One local 20-sample run on RTX 5070 Ti, with FFN checkpointing:

| Backend | Median ms | Offsets readbacks | All kernel launches | Peak above inputs |
|---|---:|---:|---:|---:|
| Native fallback | 22.18 | 12 | 242 | 328.44 MiB |
| Cached boundaries | 20.74 | 2 | 242 | 328.44 MiB |
| Triton | 19.81 | 0 | 182 | 328.44 MiB |

This is a single-layer microbenchmark, not a whole-training speedup promise.
Background GPU applications were present, so timings are indicative; launch
and copy counts establish the structural improvement. Compare warm steady-state
useful tokens/s after resuming training for the end-to-end result.
