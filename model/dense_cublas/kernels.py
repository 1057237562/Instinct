"""Packing and scale conversion around cuBLASLt; no framework dependency."""
import triton
import triton.language as tl


@triton.jit
def pack_matrix(X, OUT, ROWS: tl.constexpr, COLS: tl.constexpr,
                PAD_ROWS: tl.constexpr, PAD_COLS: tl.constexpr,
                STRIDE_ROW: tl.constexpr, STRIDE_COL: tl.constexpr, BLOCK: tl.constexpr = 256):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // PAD_COLS, index % PAD_COLS
    value = tl.load(X + row * STRIDE_ROW + col * STRIDE_COL,
                    (row < ROWS) & (col < COLS), other=0.0)
    tl.store(OUT + index, value, index < PAD_ROWS * PAD_COLS)


@triton.jit
def unpack_matrix(X, OUT, ROWS: tl.constexpr, COLS: tl.constexpr,
                  PAD_COLS: tl.constexpr, BLOCK: tl.constexpr = 256):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // COLS, index % COLS
    value = tl.load(X + row * PAD_COLS + col, index < ROWS * COLS, other=0.0)
    tl.store(OUT + index, value, index < ROWS * COLS)


@triton.jit
def inverse_scale(SCALE, INVERSE):
    # Triton quantization stores multiplication scales; cuBLASLt expects their
    # reciprocals (dequantization scales). Keep the conversion on device.
    value = tl.div_rn(1.0, tl.load(SCALE))
    tl.store(INVERSE, value)
