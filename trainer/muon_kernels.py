"""CUDA BF16 batched GEMM with a single FP32 alpha/beta epilogue.

Loaded only by the optional Muon CUDA path. No changes to NS coefficients,
normalization or iteration count; each GEMM result is materialized as BF16.
"""
import torch
import triton
import triton.language as tl

_TILE = (64, 64, 32, 4, 3)


@triton.jit
def _lerp(start, end, weight):
    return tl.where(tl.abs(weight) < 0.5,
                    tl.fma(weight, end - start, start),
                    tl.fma(-(1. - weight), end - start, end))


@triton.jit
def _prepare(G, V, U, NUMEL: tl.constexpr, MOM: tl.constexpr,
             NEST: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    matrix = tl.program_id(1)
    momentum = tl.full((), MOM, tl.float32)
    weight = tl.full((), 1. - MOM, tl.float32)
    for i in tl.static_range(len(G)):
        if matrix == i:
            # BF16 gradients/momentum storage still use FP32 EMA/Nesterov math.
            grad = tl.load(G[i] + index, index < NUMEL, 0).to(tl.float32)
            buf = tl.load(V[i] + index, index < NUMEL, 0).to(tl.float32)
            buf = _lerp(buf, grad, weight)
            tl.store(V[i] + index, buf, index < NUMEL)
            # Consume the unrounded FP32 EMA for THIS Nesterov update, even
            # when the persistent buffer above is rounded to BF16 for next step.
            update = _lerp(grad, buf, momentum) if NEST else buf
            tl.store(U + i*NUMEL + index, update, index < NUMEL)


@triton.jit
def _apply(P, U, ROWS: tl.constexpr, COLS: tl.constexpr,
           U0: tl.constexpr, U1: tl.constexpr, U2: tl.constexpr,
           DECAY, ALPHA, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    matrix = tl.program_id(1)
    mask = index < ROWS*COLS
    update = tl.load(U + matrix*U0 + (index//COLS)*U1 + (index%COLS)*U2,mask,0).to(tl.float32)
    for i in tl.static_range(len(P)):
        if matrix == i:
            value = tl.load(P[i]+index,mask,0)
            # Preserve the separate FP32 weight-decay rounding before add_.
            value = value * DECAY
            value = tl.fma(update, ALPHA, value)
            tl.store(P[i]+index,value,mask)


def prepare_update(grads, buffers, *, momentum, nesterov):
    shape = (len(grads), *grads[0].shape)
    output = torch.empty(shape, device=grads[0].device, dtype=torch.bfloat16)
    elements = grads[0].numel()
    _prepare[(triton.cdiv(elements,1024),len(grads))](tuple(grads),tuple(buffers),output,
        elements,momentum,nesterov,1024,enable_fp_fusion=False)
    return output


def apply_update(params, update, *, decay, alpha):
    rows,cols=params[0].shape
    _apply[(triton.cdiv(rows*cols,1024),len(params))](tuple(params),update,rows,cols,
        *update.stride(),decay,alpha,1024,enable_fp_fusion=False)


@triton.jit
def _apply_mantissa(P, LOW, U, ROWS: tl.constexpr, COLS: tl.constexpr,
                    U0: tl.constexpr, U1: tl.constexpr, U2: tl.constexpr,
                    DECAY, ALPHA, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    matrix = tl.program_id(1)
    mask = index < ROWS * COLS
    update = tl.load(U + matrix*U0 + (index//COLS)*U1 + (index%COLS)*U2, mask, 0).to(tl.float32)
    for i in tl.static_range(len(P)):
        if matrix == i:
            high = tl.load(P[i] + index, mask, 0).to(tl.uint16, bitcast=True).to(tl.uint32)
            low = tl.load(LOW[i] + index, mask, 0).to(tl.uint32)
            value = ((high << 16) | low).to(tl.float32, bitcast=True)
            # Match foreach mul_ followed by add_: decay rounds separately,
            # while the scaled addition is an FMA. No BF16 numeric cast here.
            value = value * DECAY
            value = tl.fma(update, ALPHA, value)
            bits = value.to(tl.uint32, bitcast=True)
            high = (bits >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
            tl.store(P[i] + index, high, mask)
            tl.store(LOW[i] + index, (bits & 0xffff).to(tl.uint16), mask)


def apply_mantissa_update(params, mantissas, update, *, decay, alpha):
    """Reconstruct/update/split in registers; preserve each Parameter/storage."""
    rows, cols = params[0].shape
    _apply_mantissa[(triton.cdiv(rows*cols, 1024), len(params))](
        tuple(params), tuple(mantissas), update, rows, cols, *update.stride(),
        decay, alpha, 1024, enable_fp_fusion=False)
    # Raw CUDA stores bypass Tensor.copy_'s version bump. Weight-stack caches
    # and saved-tensor mutation checks must still see this in-place update.
    torch.autograd.graph.increment_version((*params, *mantissas))


@triton.jit
def _gemm(A, B, C, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
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
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        a = tl.load(A + batch*A0 + row[:, None]*A1 + k[None, :]*A2,
                    (row[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(B + batch*B0 + k[:, None]*B1 + col[None, :]*B2,
                    (k[:, None] < K) & (col[None, :] < N), 0)
        acc = tl.dot(a, b, acc)
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


def baddbmm(c, a, b, *, alpha=1., beta=0., symmetric=False, transpose_output=False):
    batch, m, k = a.shape
    n = b.shape[-1]
    output = (torch.empty((batch,n,m),device=a.device,dtype=a.dtype).transpose(-2,-1)
              if transpose_output else torch.empty((batch, m, n), device=a.device, dtype=a.dtype))
    bm, bn, bk, warps, stages = _TILE
    if symmetric:
        bn = bm
    tiles = triton.cdiv(m,bm)
    grid = (tiles*(tiles+1)//2,1,batch) if symmetric else (tiles,triton.cdiv(n,bn),batch)
    _gemm[grid](
        a, b, c, output, m, n, k, *a.stride(), *b.stride(), *c.stride(),
        alpha, beta, bm, bn, bk, symmetric, transpose_output, num_warps=warps, num_stages=stages,
        enable_fp_fusion=False)
    return output


def zeropower(update, *, ns_coefficients, ns_steps, eps):
    x = update.to(torch.bfloat16)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    x = x / torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp(min=eps)
    a, b, c = ns_coefficients
    for iteration in range(ns_steps):
        # V2 FFN's long K and the single-matrix KV tail use different cuBLAS
        # reduction orders. Replacing these Gram products with the triangular
        # kernel changes BF16 results across NS iterations. Keep their reference
        # products; polynomial/update still use fused Triton epilogues.
        gram = (torch.bmm(x, x.transpose(-2, -1))
                if (tuple(x.shape[-2:]) == (1536, 5504)
                    or tuple(x.shape) == (1, 384, 1536))
                else baddbmm(x, x, x.transpose(-2, -1), symmetric=True))
        polynomial = baddbmm(gram, gram, gram, alpha=c, beta=b, symmetric=True)
        x = baddbmm(x, polynomial, x, beta=a,
                   transpose_output=transposed and iteration == ns_steps-1)
    return x.transpose(-2, -1) if transposed else x


class ZeropowerGraphCache:
    """Bounded, optimizer-owned graphs for the pure NS transformation.

    No parameter, gradient pointer, momentum, LR or optimizer step is captured.
    Results are borrowed until the next call on the same stream/shape; the
    optimizer consumes them on that stream before invoking another chunk.
    The cache is runtime-only and is never part of optimizer state_dict.
    """
    def __init__(self, limit=8):
        self.entries = {}
        self.limit = limit

    @torch.no_grad()
    def __call__(self, update, **options):
        if torch.cuda.is_current_stream_capturing():
            return zeropower(update, **options)
        current = torch.cuda.current_stream(update.device)
        key = (update.device, update.dtype, tuple(update.shape), tuple(update.stride()),
               current.cuda_stream, tuple(options['ns_coefficients']),
               options['ns_steps'], options['eps'])
        entry = self.entries.get(key)
        if entry is None:
            # Avoid retaining an unbounded graph pool for dynamic shapes or
            # schedules that change NS settings. Fall back without recapture.
            if len(self.entries) >= self.limit:
                return zeropower(update, **options)
            with torch.cuda.device(update.device):
                source = torch.empty_like(update)
                source.copy_(update)
                stream = torch.cuda.Stream(device=update.device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        result = zeropower(source, **options)
                current.wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    result = zeropower(source, **options)
                entry = (source, result, graph)
                self.entries[key] = entry
        source, result, graph = entry
        source.copy_(update)
        graph.replay()
        return result
