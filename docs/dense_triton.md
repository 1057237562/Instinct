# Dense Triton kernel implementation

Experimental branch: `experimental/dense-triton-fp8`, based on
`refactor/code-readability` at `cdac2a9fbb178a9ef741f91a5974aa6a22d22411`.

This implements Instinct Dense's computational kernels and a complete decoder
block with explicit forward/backward. The implementation imports neither
PyTorch nor TorchAO. TorchAO is an optional **test reference**, not a runtime
dependency. The existing model and trainers remain the baseline.

## Implemented

| Component | Forward | Backward / precision |
| --- | --- | --- |
| Linear | Strided, tiled BF16/FP32 GEMM or scaled FP8 GEMM | dX and dW GEMMs; transpose strides avoid copies; masked tails |
| FP8 quantization | Dynamic tensorwise amax, scale, saturating round-to-nearest-even conversion | E4M3 activations/weights; E5M2 upstream gradients |
| RMSNorm | FP32 sum/normalization, output cast to activation dtype | dX plus bounded partial reductions for dWeight |
| RoPE | Instinct's split-half layout; supplied cosine/sine tables | Transpose rotation, including nonzero position offsets |
| Causal GQA | Online softmax, native grouped K/V heads | Recomputed probabilities; separate dQ and dK/dV kernels |
| SwiGLU | Fused SiLU(gate) × up | Fused gate/up derivatives |
| Residual | FP32 addition followed by output cast | Explicit gradient addition in the block |
| Embedding | Token gather | FP32 scatter-add, including repeated IDs and tied-head contributions |
| Cross entropy | Stable per-row loss, ignored labels | Unreduced dlogits and valid-label counts |

`DenseBlock` composes **all seven Linear layers**, four RMSNorms (including
Q/K head norms), RoPE, causal GQA, SwiGLU and both residuals. Its backward
returns dInput and all **11 parameter gradients**, using the existing
`InstinctBlock` parameter names. Defaults match the repository: hidden 768,
8 query heads / 4 KV heads, head dimension 96, FFN width 2432.

GQA stores output and log-sum-exp, not an S×S attention matrix. K/V are never
expanded to query-head count. KV backward reduces each query-head group in
registers rather than writing atomic gradients. The initial tiles are fixed
at 32×32 for attention and 32×64×32 for GEMM; they are not yet tuned per GPU.

## FP8 precision contract

The first implemented recipe corresponds to TorchAO 0.12's NVIDIA `tensorwise`
format/scaling policy:

```
scale = float32(fp8_max / max(float64(amax(abs(x))), 1e-12))
q = round_to_nearest_even(clamp(float32(x) * scale, -fp8_max, fp8_max))
Y  = (Qx @ Qw.T) / Sx / Sw
dX = (Qdy @ Qw)  / Sdy / Sw
dW = (Qdy.T @ Qx) / Sdy / Sx
```

E4M3 uses maximum 448; E5M2 uses maximum 57344. Matrix products accumulate
in FP32. Master weights remain FP32 when supplied as FP32; there is no FP8
optimizer state. Attention, normalization, RoPE, activations and residuals
remain BF16/FP32 rather than being indiscriminately quantized.

This is **not a claim of bitwise TorchAO equivalence or equal convergence**:

- Native FP8 casts and Tensor Core execution still require GPU validation.
- The GEMM requests accurate accumulation (`max_num_imprecise_acc=0`), while
  TorchAO's tensorwise forward can enable fast accumulation.
- dW is stored directly in the master parameter's dtype. For FP32 parameters,
  this avoids a possible intermediate BF16 gradient rounding.
- Fused SwiGLU keeps intermediate arithmetic in FP32; an unfused BF16 SiLU
  rounds its intermediate output. BF16 comparisons therefore use tolerances.
- `DenseBlock` uses the input activation dtype consistently. The caller chooses
  where to cast embeddings/activations; there is no implicit autocast policy.
- Rowwise scaling, delayed scaling, MXFP8 and FP8 attention are not implemented.

The optional TorchAO test compares all three scaled Linear GEMMs against its
emulated tensorwise recipe using prequantized values; it does not stand in for
native FP8 GPU validation.

## Files and API

- `model/dense_triton/linear.py`: amax/scaling/cast and tiled GEMM.
- `model/dense_triton/attention.py`: online GQA forward and recomputed backward.
- `model/dense_triton/elementwise.py`: normalization, rotary, activation,
  residual, embedding and cross-entropy kernels.
- `model/dense_triton/ops.py`: buffer orchestration and `DenseBlock`.
- `model/dense_triton/runtime.py`: CUDA allocations, explicit-target Triton
  compilation and direct CUDA Driver API launches.

The backend interface is deliberately small: `empty(shape, dtype)` and
`launch(kernel, grid, buffers, **constants)`. `Tensor.view()` creates contiguous
views without copies. Kernel code does not use `torch.Tensor` or autograd.

```python
from model.dense_triton.ops import DenseOps, DenseBlock, BlockConfig
from model.dense_triton.runtime import CudaBackend

with CudaBackend(device=0) as backend:
    ops = DenseOps(backend, fp8=True)
    # Upload weights under InstinctBlock's state-dict names using
    # backend.from_numpy(). Supply x, upstream_grad, cos and sin similarly.
    block = DenseBlock(ops, weights, BlockConfig())
    output, context = block.forward(x, cos, sin)
    grad_input, grad_parameters = block.backward(context, upstream_grad)
    backend.synchronize()
```

See `experiments/smoke_dense_triton.py` for a complete executable example.
The raw launcher pins **Triton 3.4.x**, compiles for an explicit architecture,
loads the cubin and calls `cuLaunchKernel`. It bypasses Triton's normal
PyTorch-backed driver initialization. It rejects unsupported cluster, TMA and
scratch-buffer launch ABIs rather than guessing their calling convention.

Runtime prerequisites: Python, NumPy, Triton 3.4.x, NVIDIA driver. For native
Windows, use a compatible `triton-windows` 3.4 build; native Windows execution
has not been tested here. FP8 requires SM89 or later; an RTX 3090/SM86 must use
the BF16 path (`--fp8 0`). Compiling a cubin offline does not prove that a
particular installed driver can load it.

## Validation commands

Run from the repository root. Production dependencies do not include TorchAO;
the test environment additionally uses the repository's PyTorch/datasets/test
dependencies and optionally TorchAO 0.12 for the oracle test.

```bash
# Compile the default-width Dense forward/backward, no GPU or torch needed:
python -m experiments.compile_dense_triton --arch 89 --fp8 1
python -m experiments.compile_dense_triton --arch 90 --fp8 1
python -m experiments.compile_dense_triton --arch 120 --fp8 1
python -m experiments.compile_dense_triton --arch 86 --fp8 0

# CPU execution of the actual Triton kernels through the interpreter:
TRITON_INTERPRET=1 python -m pytest -q tests/test_dense_triton.py

# GPU kernel/reference tests; unset TRITON_INTERPRET first:
python -m pytest -q tests/test_dense_triton.py

# Fresh process; deliberately blocks any torch/torchao import:
python -m experiments.smoke_dense_triton --fp8 1 --seq_len 17
```

In PowerShell, set `$env:TRITON_INTERPRET='1'` for the interpreter run and
remove it with `Remove-Item Env:TRITON_INTERPRET` before GPU tests.

Validation performed in this development environment:

- Selected test run: **39 passed, 6 skipped** (16 kernel/orchestration tests
  plus 23 existing Dense/checkpoint/flag regression tests). The six skipped
  checks require native FP8/BF16 or the CUDA Driver API.
- Complete FP32 block output, dInput and all parameter gradients checked
  against the repository's `InstinctBlock` and PyTorch autograd.
- Individual forward/backward tests cover width 768 RMSNorm, head dimension 96,
  nonmultiple tile tails, GQA ratios 2 and 4, sequence length 1, padded/all-masked
  attention rows, offset RoPE, duplicate embeddings, and ignored CE targets.
- FP8 recipe GEMMs checked against TorchAO's emulated tensorwise Linear.
- Default-width BF16 block with FP8 Linear: 54 kernel specializations compiled
  for each of SM89, SM90 and SM120, with no torch import. BF16-only: 35
  specializations for SM86. Maximum shared memory: 34,944 bytes.
- Native FP8/BF16 GPU execution, Windows execution, throughput, MFU, peak memory
  and training convergence **not measured**: this environment has no CUDA driver.

Triton 3.4's interpreter FP8 downcast does not implement correct carry/tie
handling. Native cast tests are explicitly skipped in interpreter mode and
remain GPU tests. The implementation uses native RNE casts; it does not replace
them with the faulty interpreter's behavior.

## Supported scope and next steps

The implemented unit is the Dense **kernel layer and decoder block**, not yet
an end-to-end replacement of `train_pretrain.py`. Dropout must be zero;
attention is causal, same-sequence prefill/training with head dimension ≤128;
inputs are contiguous and finite. MoE, KV decoding, distributed training,
automatic checkpointing and higher-order derivatives are outside this version.
RoPE accepts externally prepared tables, including YaRN tables.

Cross entropy expects already-shifted valid labels (or -100); callers reduce
losses and divide gradients by the valid count, skipping an all-ignored batch.
Embedding inputs must be in range. Its atomic backward is nondeterministic.
Tied embedding/head gradients must be accumulated into one FP32 buffer.

The small CUDA backend is an **arena for one forward/backward experiment**:
allocations stay live until context exit. Repeated steps with the current
allocating orchestration will grow memory. Before a training loop, add a static
buffer lifetime plan/reuse and persistent optimizer state. Do not use this
prototype's arena size as a fair comparison with an optimized trainer.

Prioritized work after GPU correctness passes:

1. Benchmark the actual projection/FFN shapes and all three GEMMs. Compare
   tuned Triton against cuBLASLt FP8; use whichever wins on the target GPU.
   Keep quantization scale conventions and backward precision consistent.
2. Fuse quantization with producers where possible, reuse weight casts across
   accumulation microbatches until the optimizer changes the weights, and
   eliminate redundant amax launches. Preserve activation-checkpoint semantics.
3. Add memory reuse and checkpoint/recompute scheduling, then token embedding,
   the block stack, final normalization, BF16 tied LM head, reduced CE and
   fused AdamW into an end-to-end runner. Fuse/chunk head + CE to avoid a full
   retained logits buffer at long sequence lengths.
4. Tune attention tiles for head dimension 96; benchmark backward recomputation
   and grouped KV gradients, then CUDA Graphs once buffer addresses are stable.
5. Compare matched seeds/data, loss curves, clipping frequency and gradient
   errors against BF16 and the chosen TorchAO recipe. Report useful tokens/s,
   allocated/peak memory and MFU with a documented FLOP convention and the
   correct dense BF16/FP8 hardware peak. No estimated speedup is a measurement.

For the original Windows memory question, `experiments/profile_dense_memory.py`
compares a CUDA-context-only process with the existing PyTorch Dense trainer.
Run variants as separate processes. `reserved - allocated` is reusable allocator
cache; device-wide used memory also includes the desktop/driver/other processes.
Removing PyTorch does not remove the CUDA context or guarantee recovery of 2 GiB.

## Reference specifications

- [TorchAO 0.12 float8 configuration](https://github.com/pytorch/ao/blob/v0.12.0/torchao/float8/config.py)
- [TorchAO scale computation](https://github.com/pytorch/ao/blob/v0.12.0/torchao/float8/float8_utils.py)
- [NVIDIA cuBLASLt FP8 scaling](https://docs.nvidia.com/cuda/cublas/index.html#tensorwide-scaling-for-fp8-data-types)
- [Triton 3.4 driver interface](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/backends/driver.py)
- [Triton Windows distribution](https://github.com/triton-lang/triton-windows)
- [PyTorch CUDA memory accounting](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-memory-management)
