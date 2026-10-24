"""FP8 tensorwise quantization and strided GEMMs for forward, dX and dW.

E4M3 (448) for activations/weights, E5M2 (57344) for upstream gradients.
Scales multiply high-precision values on quantization; GEMM divides by them.
Master weights and optimizer state are never stored in FP8.
"""
import triton
import triton.language as tl


@triton.jit
def amax_partial(X, PARTIAL, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < SIZE, other=0).to(tl.float32)
    tl.store(PARTIAL + tl.program_id(0), tl.max(tl.abs(x), 0))


@triton.jit
def scale_from_amax(PARTIAL, SCALE, COUNT: tl.constexpr,
                    FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    amax = tl.max(tl.load(PARTIAL + i, i < COUNT, other=0), 0)
    # Float64 division matches TorchAO's tensorwise scale computation.
    denominator = tl.maximum(amax.to(tl.float64), 1.0e-12)
    scale = (FP8_MAX / denominator).to(tl.float32)
    tl.store(SCALE, scale)


@triton.jit
def quantize(X, SCALE, Q, SIZE: tl.constexpr, FP8_MAX: tl.constexpr,
             BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < SIZE, other=0).to(tl.float32)
    scaled = x * tl.load(SCALE)
    saturated = tl.minimum(tl.maximum(scaled, -FP8_MAX), FP8_MAX)
    tl.store(Q + i, saturated.to(Q.dtype.element_ty, fp_downcast_rounding="rtne"), i < SIZE)


@triton.jit
def matmul(A, B, C, SA, SB, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           AM: tl.constexpr, AK: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
           SCALED: tl.constexpr, BM: tl.constexpr = 32,
           BN_TILE: tl.constexpr = 64, BK_TILE: tl.constexpr = 32):
    """C[M,N] = A[M,K] @ B[K,N], with optional scalar FP8 dequantization.

    Explicit strides cover X @ W.T, dY @ W and dY.T @ X without copies.
    Padding is masked inside the kernel, including odd token counts in dW.
    """
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN_TILE + tl.arange(0, BN_TILE)
    k = tl.arange(0, BK_TILE)
    acc = tl.full((BM, BN_TILE), 0, tl.float32)
    for start in range(tl.cdiv(K, BK_TILE)):
        kk = start * BK_TILE + k
        a = tl.load(A + m[:, None] * AM + kk[None, :] * AK,
                    (m[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B + kk[:, None] * BK + n[None, :] * BN,
                    (kk[:, None] < K) & (n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee", max_num_imprecise_acc=0)
    if SCALED:
        acc = acc / tl.load(SA) / tl.load(SB)
    tl.store(C + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))
