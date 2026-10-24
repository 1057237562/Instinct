"""Instinct Dense elementwise kernels; no PyTorch or TorchAO dependency."""
import triton
import triton.language as tl


@triton.jit
def cast(X, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i, i < SIZE, other=0), i < SIZE)


@triton.jit
def _rms_forward(X, W, Y, R, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    tl.store(Y + row * N + col, x * r * w, col < N)
    tl.store(R + row, r)


@triton.jit
def _rms_backward(X, W, R, DY, DX, PARTIAL_DW, M: tl.constexpr,
                  N: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    w = tl.load(W + col, col < N, other=0).to(tl.float32)
    dw = tl.full((BLOCK,), 0, tl.float32)
    for row in range(group, M, GROUPS):
        x = tl.load(X + row * N + col, col < N, other=0).to(tl.float32)
        dy = tl.load(DY + row * N + col, col < N, other=0).to(tl.float32)
        r = tl.load(R + row)
        xhat = x * r
        dxhat = dy * w
        dot = tl.sum(dxhat * xhat, axis=0) / N
        tl.store(DX + row * N + col, r * (dxhat - xhat * dot), col < N)
        dw += dy * xhat
    tl.store(PARTIAL_DW + group * N + col, dw, col < N)


@triton.jit
def _rms_reduce_dw(PARTIAL_DW, DW, N: tl.constexpr, GROUPS: tl.constexpr,
                   BLOCK_G: tl.constexpr, BLOCK_N: tl.constexpr):
    g = tl.arange(0, BLOCK_G)
    col = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    values = tl.load(PARTIAL_DW + g[:, None] * N + col[None, :],
                     (g[:, None] < GROUPS) & (col[None, :] < N), other=0)
    tl.store(DW + col, tl.sum(values, axis=0), col < N)


@triton.jit
def _swiglu_forward(G, U, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + offsets, offsets < SIZE, other=0).to(tl.float32)
    u = tl.load(U + offsets, offsets < SIZE, other=0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    tl.store(Y + offsets, g * sig * u, offsets < SIZE)


@triton.jit
def _swiglu_backward(G, U, DY, DG, DU, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    g = tl.load(G + offsets, mask, other=0).to(tl.float32)
    u = tl.load(U + offsets, mask, other=0).to(tl.float32)
    dy = tl.load(DY + offsets, mask, other=0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    tl.store(DG + offsets, dy * u * sig * (1.0 + g * (1.0 - sig)), mask)
    tl.store(DU + offsets, dy * g * sig, mask)


@triton.jit
def add(A, B, OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i, i < SIZE, other=0).to(tl.float32)
    b = tl.load(B + i, i < SIZE, other=0).to(tl.float32)
    tl.store(OUT + i, a + b, i < SIZE)


@triton.jit
def rotary(X, COS, SIN, Y, SEQ: tl.constexpr, HEADS: tl.constexpr,
           DIM: tl.constexpr, START: tl.constexpr = 0,
           BACKWARD: tl.constexpr = False, BLOCK: tl.constexpr = 128):
    """[batch, sequence, heads, dim], split-half RoPE, including its transpose.

    COS/SIN contain the duplicated half frequencies used by Instinct. Tables
    are provided externally, so existing theta/YaRN tables can also be used.
    """
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    pos = (row // HEADS) % SEQ + START
    other_d = (d + DIM // 2) % DIM
    x = tl.load(X + row * DIM + d, d < DIM, other=0).to(tl.float32)
    other = tl.load(X + row * DIM + other_d, d < DIM, other=0).to(tl.float32)
    cos = tl.load(COS + pos * DIM + d, d < DIM, other=0).to(tl.float32)
    if BACKWARD:
        sin = tl.load(SIN + pos * DIM + other_d, d < DIM, other=0).to(tl.float32)
        sign = tl.where(d < DIM // 2, 1.0, -1.0)
    else:
        sin = tl.load(SIN + pos * DIM + d, d < DIM, other=0).to(tl.float32)
        sign = tl.where(d < DIM // 2, -1.0, 1.0)
    tl.store(Y + row * DIM + d, x * cos + sign * other * sin, d < DIM)


@triton.jit
def embedding(IDS, W, Y, TOKENS: tl.constexpr, DIM: tl.constexpr,
              VOCAB: tl.constexpr, BLOCK: tl.constexpr = 256):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(IDS + row)
    valid = (token >= 0) & (token < VOCAB)
    value = tl.load(W + token * DIM + d, valid & (d < DIM), other=0)
    tl.store(Y + row * DIM + d, value, d < DIM)


@triton.jit
def embedding_backward(IDS, DY, DW, TOKENS: tl.constexpr, DIM: tl.constexpr,
                       VOCAB: tl.constexpr, BLOCK: tl.constexpr = 256):
    """Accumulates into initialized FP32 DW, including an existing tied-head dW.

    Atomic additions handle duplicate token IDs; their order is nondeterministic.
    """
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(IDS + row)
    dy = tl.load(DY + row * DIM + d, d < DIM, other=0).to(tl.float32)
    tl.atomic_add(DW + token * DIM + d, dy,
                  (token >= 0) & (token < VOCAB) & (d < DIM), sem="relaxed")


@triton.jit
def cross_entropy(LOGITS, LABELS, LOSSES, VALID, DLOGITS,
                   VOCAB: tl.constexpr, BLOCK: tl.constexpr):
    """Per-row CE and *unreduced* dlogits; labels must already be shifted.

    Caller divides the gradient and sum(losses) by sum(VALID), excluding -100.
    An all-ignored batch returns zeros here; caller must skip that update.
    """
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    label = tl.load(LABELS + row)
    valid = (label >= 0) & (label < VOCAB)
    x = tl.load(LOGITS + row * VOCAB + col, col < VOCAB, other=-float("inf")).to(tl.float32)
    m = tl.max(x, 0)
    exp = tl.exp(x - m)
    total = tl.sum(exp, 0)
    target = tl.sum(tl.where(col == label, x, 0), 0)
    loss = tl.where(valid, m + tl.log(total) - target, 0)
    grad = tl.where(valid, exp / total - (col == label).to(tl.float32), 0)
    tl.store(LOSSES + row, loss)
    tl.store(VALID + row, valid.to(tl.int32))
    tl.store(DLOGITS + row * VOCAB + col, grad, col < VOCAB)
