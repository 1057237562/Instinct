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

The model-entry selector is shared by the standard Dense/MoE and RecurrsiveInstinct backbones.
KV-cache generation continues through the previous path. V2's standard-residual
training also prepares a plain causal mask for eligible single-document batches.
Arbitrary explicit caller masks, nonzero attention dropout, non-BF16 compute,
CPU, and unsupported head dimensions retain the shared-bias SDPA implementation.
Supported packed fast-path head dimensions are 16, 32, 64 and 128. Native head
dimension 96 is additionally enabled for V2's 16 Q / 4 KV heads on the validated
PyTorch 2.13 / Blackwell SM12 stack; explicit `flex` can select its packed path.

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

Block-mask construction runs outside checkpointed layers. The V1 attention
region compiles independently; V2's 96-wide attention stays traceable inside the
outer compiled checkpoint, keeping attention, FP8 projections and recomputation
in the compiled trunk. Document block metadata is per batch, not a cache of
previous sequence IDs.

## Local validation

CUDA tests cover output/dQ/dK/dV, non-aligned lengths, GQA, padding, document/future
isolation, changing document boundaries between batches, MoE gradients, checkpoint
modes 1/2, RecurrsiveInstinct models, strict state-dict loading, and outer `torch.compile`.

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
affect training throughput. Those V1 explicit-mask measurements do not measure
the V2 single-document training path described below.

## V2 single-document training

`prepare_lm_batch` intentionally drops `sequence_ids` when a batch contains only
one document per row. The 4096-by-1 bucket therefore bypasses the packed selector.
The older layer-local causal selector only enabled V1's 32-wide heads, leaving
V2's 96-wide heads on memory-efficient SDPA even with packed Flex enabled.

The standard model now calls `maybe_prepare_causal_flex_mask` at model entry for
training without a KV cache or explicit caller mask. It uses the existing
`INSTINCT_CAUSAL_ATTENTION_BACKEND=auto|flex|sdpa` control. Automatic selection
requires BF16 compute, zero attention dropout, 16 Q / 4 KV heads of dimension 96,
PyTorch 2.13 and Blackwell SM12. Other residual layouts and inference keep their
previous dispatch.

The shared mask is the original plain causal predicate `query >= key`, with
batch/head-independent metadata cached by sequence length and device. It does
not introduce document boundaries or change padding/label/RoPE semantics.
Mask preparation occurs before the transformer loop; placing a disabled mask
builder inside a checkpointed layer would prevent compiling that whole layer.
The prepared mask reuses the traceable 96-wide Flex implementation.

Validation: `tests/test_causal_attention_v2.py` covers the real loader's
single-document branch, output/dQ/dK/dV at lengths 137 and 4096, future isolation,
eligibility/fallback controls, mask cache reuse, inference/cache/residual
exclusions, and compiled BF16/FP8 training with two full-layer checkpoints.
Real-data performance and quality reports belong in
`experiments/results/v2_causal96/`; these are finite-window comparisons, not
long-run convergence guarantees.

### Local V2 result (2026-10-20)

RTX 5070 Ti, the fixed 5344-token recipe, FP8 tensorwise projections, BF16
parameters/momentum, Muon and full-layer checkpointing. Both runs start from the
same saved step 2152 and consume identical batches: 16 warmup + 128 measured
microsteps (581,408 useful tokens in the measured window). The control keeps
packed Flex enabled and sets only the causal backend to `sdpa`; the candidate
uses `auto` for both. A repeated control uses the same source revision as the
candidate, including the concurrent ETA logging update.

| Warm whole-model measurement | Causal SDPA control | Causal Flex fix |
|---|---:|---:|
| Wall time per microstep | 1165.39 ms | 472.34 ms |
| Useful tokens/s | 3,897.63 | 9,616.52 |
| Mixed FP8/BF16 useful MFU | 16.56% | 40.86% |
| Peak allocated memory | 11.93 GiB | 10.20 GiB |
| Peak reserved memory | 13.48 GiB | 11.81 GiB |

Whole-model speedup was 2.47x. Both the attention kernels and peak memory
footprint change in this comparison, so it does not isolate either contribution.
The two 64-step candidate halves measured 469.15 and 475.40 ms per GPU step.
MFU uses the same nominal non-sparse FP32-accumulation peaks as prior reports:
175.8 TF/s FP8 and 87.9 TF/s BF16. It excludes padding, checkpoint recomputation
and optimizer work. Cold compilation is excluded from the throughput window.

Across 144 microsteps / 18 optimizer updates and 8 held-out sequences, finite
training-loss windows and the validation gate passed. Final held-out NLL was
3.13889267 for the control and 3.13891494 for the candidate (+0.00071%).
The focused regression run passed 34 tests, skipping two optional slow tests.
See `experiments/results/v2_causal96/summary.json` and its paired raw reports.
