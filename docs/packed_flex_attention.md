# Packed FlexAttention

Packed CUDA BF16 attention now uses PyTorch FlexAttention's Triton backend when
eligible. The current Windows PyTorch build does not contain native FlashAttention
(`torch.backends.cuda.is_flash_attention_available()` is false), so simply calling
its varlen API cannot provide the desired kernel. No dependency upgrades are needed
in the current PyTorch 2.13 / Triton 3.4 environment.

## What changes

- Build a document-aware `BlockMask` once per batch and reuse it across layers.
  Its predicate is exactly `same sequence ID AND key position <= query position`.
- Skip fully masked blocks rather than computing a dense attention matrix first.
- Keep K/V in compact GQA form (4 heads in the current model, not expanded to 16).
- Preserve RoPE positions, padding ID `-1` semantics, label masks, model weights,
  optimizer state and checkpoint keys. Nothing is written to existing checkpoints.
- Preserve BF16 compute with normal floating-point rounding differences. Output
  and gradient parity is tested against the existing SDPA path; training trajectories
  are not promised to be bitwise identical.

The model-entry selector is shared by the standard Dense/MoE and looped backbones.
Unpacked attention and KV-cache generation continue through the previous path.
Arbitrary explicit caller masks, nonzero attention dropout, non-BF16 compute,
CPU, and unsupported head dimensions retain the shared-bias SDPA implementation.
Supported fast-path head dimensions are 16, 32, 64 and 128.

## Selection and rollback

Set `INSTINCT_PACKED_ATTENTION_BACKEND` before starting the trainer/WebUI:

| Value | Behavior |
|---|---|
| `auto` (default) | Use Flex for eligible packed inputs when dependencies exist |
| `flex` | Same eligibility rules; missing Flex/Triton dependencies raise an error |
| `sdpa` | Use the previous shared-bias memory-efficient SDPA path |

Compilation/runtime failures are not silently swallowed. For an unsupported
environment, choose `sdpa` and resume the existing checkpoint. Already running
processes do not hot-reload this change. Initial calls compile mask and attention
kernels; compare throughput after both buckets have warmed up.

Flex kernels and block-mask construction are compiled independently and called
from small compiler-disabled entry points. This avoids coupling their higher-order
operators to the outer model's MoE graph breaks while keeping compiled forward and
backward. Block metadata is per batch, not a cache of previous sequence IDs.

## Local validation

CUDA tests cover output/dQ/dK/dV, non-aligned lengths, GQA, padding, document/future
isolation, changing document boundaries between batches, MoE gradients, checkpoint
modes 1/2, looped models, strict state-dict loading, and outer `torch.compile`.

```powershell
python -m pytest tests/test_packed_attention.py -q
# Full model compile check is opt-in to keep ordinary tests short:
python -m pytest tests/test_packed_attention.py -k compiled_moe_training --run-slow -q
# Run benchmarks with training paused:
python experiments/bench_packed_attention.py --segments 4
python experiments/bench_packed_attention.py --segments 1
```

RTX 5070 Ti, BF16, batch=4, length=2928, Q/KV heads=16/4, head_dim=32,
single-layer forward+backward, alternating backend order after warmup:

| Synthetic document layout | SDPA | Flex | Warm block-mask build (once per batch) |
|---|---:|---:|---:|
| 4 equal-length segments/row | 13.33 ms | 1.73 ms | 0.64 ms |
| 1 segment/row with explicit causal mask | 15.02 ms | 3.60 ms | 0.39 ms |

An additional warm four-segment run measured 13.28 ms vs 1.54 ms. Its extra
forward/backward allocation peak was 116.67 MiB for SDPA vs 30.59 MiB for Flex.
These peaks exclude inputs and prebuilt masks (both masks coexist in the benchmark),
so they are not the whole model's memory requirement or total mask-storage savings.

The second training bucket (`batch=2, length=4096`, four equal segments) measured
12.25 ms vs 1.32 ms, with a 0.46 ms block-mask build and extra allocation peaks
of 81.03 MiB vs 21.00 MiB. Both current bucket shapes passed gradient checks.

In these runs, relative L2 error was 0.09–0.12% for output and 0.27–0.41% for
gradients. This is an attention microbenchmark, not a whole-model speedup claim;
real document lengths, optimizer time, compiler overhead and other GPU workloads
affect training throughput. Unpacked single-document batches are not switched by
this optimization even though the explicit-mask benchmark measures that case.
