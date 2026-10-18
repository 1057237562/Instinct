"""Experimental FP16 inner accumulator; NS intermediates still round to BF16."""
import triton
import triton.language as tl

@triton.jit
def _gemm_fp16acc(A, B, C, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr,
          B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr,
          C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
          ALPHA: tl.constexpr, BETA: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SYMMETRIC: tl.constexpr,
          TRANSPOSE_OUTPUT: tl.constexpr):
    if SYMMETRIC:
        tile = tl.program_id(0)
        row_tile = ((tl.sqrt(8. * tile + 1.) - 1.) * 0.5).to(tl.int32)
        col_tile = tile - row_tile * (row_tile + 1) // 2
    else:
        row_tile, col_tile = tl.program_id(0), tl.program_id(1)
    row = row_tile * BM + tl.arange(0, BM)
    col = col_tile * BN + tl.arange(0, BN)
    batch = tl.program_id(2)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float16)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        a = tl.load(A + batch*A0 + row[:, None]*A1 + k[None, :]*A2,
                    (row[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(B + batch*B0 + k[:, None]*B1 + col[None, :]*B2,
                    (k[:, None] < K) & (col[None, :] < N), 0)
        acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc, out_dtype=tl.float16)
    acc = acc.to(tl.float32)
    result = ALPHA * acc
    if BETA != 0:
        c = tl.load(C + batch*C0 + row[:, None]*C1 + col[None, :]*C2,
                    (row[:, None] < M) & (col[None, :] < N), 0).to(tl.float32)
        result = tl.fma(ALPHA, acc, BETA * c)
    out_offset = ((batch*N + col[None, :])*M + row[:, None] if TRANSPOSE_OUTPUT
                  else (batch*M + row[:, None])*N + col[None, :])
    tl.store(O + out_offset, result,
             (row[:, None] < M) & (col[None, :] < N))
    if SYMMETRIC and row_tile != col_tile:
        tl.store(O + (batch*M + col[None, :])*N + row[:, None], result,
                 (row[:, None] < M) & (col[None, :] < N))
