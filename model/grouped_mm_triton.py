"""BF16 ragged GEMMs using device offsets and conventional (non-TMA) loads.

Imported lazily by grouped_mm.py: CPU-only installations do not need Triton.
The persistent forward grid walks expert tiles on device, including empty groups.
Weight gradients assign one output tile to one program, with no floating-point
atomic reduction or host-side routing decisions.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ragged_mm(
    X, W, Ends, Y,
    K: tl.constexpr, N: tl.constexpr, E: tl.constexpr,
    X0: tl.constexpr, X1: tl.constexpr,
    W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    tile = tl.program_id(0)
    problem_start = 0
    row_start = 0
    columns = tl.cdiv(N, BN)
    for expert in range(E):
        row_end = tl.load(Ends + expert)
        rows = row_end - row_start
        tiles = tl.cdiv(rows, BM) * columns
        while (tile >= problem_start) & (tile < problem_start + tiles):
            local = tile - problem_start
            rm = (local // columns) * BM + tl.arange(0, BM)
            rn = (local % columns) * BN + tl.arange(0, BN)
            rk = tl.arange(0, BK)
            acc = tl.full((BM, BN), 0, tl.float32)
            for block in range(tl.cdiv(K, BK)):
                kk = block * BK + rk
                a = tl.load(X + (row_start + rm[:, None]) * X0 + kk[None, :] * X1,
                            (rm[:, None] < rows) & (kk[None, :] < K), other=0)
                b = tl.load(W + expert * W0 + kk[:, None] * W1 + rn[None, :] * W2,
                            (kk[:, None] < K) & (rn[None, :] < N), other=0)
                acc = tl.dot(a, b, acc)
            tl.store(Y + (row_start + rm[:, None]) * N + rn[None, :],
                     acc.to(Y.dtype.element_ty),
                     (rm[:, None] < rows) & (rn[None, :] < N))
            tile += PROGRAMS
        problem_start += tiles
        row_start = row_end


@triton.jit
def _ragged_weight_grad(
    X, DY, Ends, DW,
    K: tl.constexpr, N: tl.constexpr,
    X0: tl.constexpr, X1: tl.constexpr,
    G0: tl.constexpr, G1: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    tile = tl.program_id(0)
    expert = tl.program_id(1)
    end = tl.load(Ends + expert)
    start = tl.load(Ends + expert - 1, expert > 0, other=0)
    rm = (tile // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    rn = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(end - start, BK)):
        rr = start + block * BK + rk
        a = tl.load(X + rr[None, :] * X0 + rm[:, None] * X1,
                    (rr[None, :] < end) & (rm[:, None] < K), other=0)
        b = tl.load(DY + rr[:, None] * G0 + rn[None, :] * G1,
                    (rr[:, None] < end) & (rn[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    # An empty expert also writes its full gradient tile (zeros).
    tl.store(DW + expert * K * N + rm[:, None] * N + rn[None, :],
             acc.to(DW.dtype.element_ty), (rm[:, None] < K) & (rn[None, :] < N))


_PROGRAMS_CACHE = {}


def _programs(device):
    """Persistent-grid size for one device; resolved once, never per launch.

    A plain dict rather than ``lru_cache``: Dynamo warns about (and traces
    through) cache-wrapped callables it meets inside a graph, and this one is
    reached from the traced MoE forward.
    """
    key = str(device)
    if key not in _PROGRAMS_CACHE:
        _PROGRAMS_CACHE[key] = 4 * torch.cuda.get_device_properties(device).multi_processor_count
    return _PROGRAMS_CACHE[key]


def ragged_mm(x, weight, offsets, programs=None):
    tokens, inner = x.shape
    experts, _, columns = weight.shape
    output = torch.empty((tokens, columns), device=x.device, dtype=x.dtype)
    if tokens == 0:
        return output
    if programs is None:
        programs = _programs(x.device)
    with torch.cuda.device(x.device):
        _ragged_mm[(programs,)](
            x, weight, offsets, output, inner, columns, experts,
            *x.stride(), *weight.stride(), programs,
            BM=64, BN=64, BK=32, num_warps=4, num_stages=3,
        )
    return output


def ragged_weight_grad(x, grad, offsets, experts):
    inner, columns = x.size(1), grad.size(1)
    result = torch.empty((experts, inner, columns), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        _ragged_weight_grad[(triton.cdiv(inner, 64) * triton.cdiv(columns, 64), experts)](
            x, grad, offsets, result, inner, columns, *x.stride(), *grad.stride(),
            BM=64, BN=64, BK=32, num_warps=4, num_stages=3,
        )
    return result
