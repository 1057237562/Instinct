"""Pure host-side layout planning, independent of CUDA availability."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MatmulLayout:
    m: int
    n: int
    k: int
    mp: int
    np: int
    kp: int
    a_stride: tuple
    b_transpose_stride: tuple
    a_pack: bool
    b_pack: bool


def plan_layout(a_shape, b_shape, transpose_a=False, transpose_b=False, alignment=16):
    if len(a_shape) != 2 or len(b_shape) != 2 or min(*a_shape, *b_shape) <= 0:
        raise ValueError('GEMM requires nonempty matrices')
    if alignment <= 0:
        raise ValueError('alignment must be positive')
    m, k = a_shape[::-1] if transpose_a else a_shape
    kb, n = b_shape[::-1] if transpose_b else b_shape
    if k != kb:
        raise ValueError('GEMM inner dimensions differ')
    pad = lambda d: (d + alignment - 1) // alignment * alignment
    mp, np, kp = pad(m), pad(n), pad(k)
    a_stride = (1, a_shape[1]) if transpose_a else (a_shape[1], 1)
    # Store B.T as row-major [N,K] so cuBLAS can view it as column-major [K,N].
    bt_stride = (b_shape[1], 1) if transpose_b else (1, b_shape[1])
    return MatmulLayout(m, n, k, mp, np, kp, a_stride, bt_stride,
                         not (a_stride == (k, 1) and (mp, kp) == (m, k)),
                         not (bt_stride == (k, 1) and (np, kp) == (n, k)))
