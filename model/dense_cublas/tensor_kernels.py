"""General training primitives. All device arithmetic is Triton, GEMMs use Lt."""
import triton
import triton.language as tl


@triton.jit
def gradient_sum(A, B, C, D, OUT, N: tl.constexpr, COUNT: tl.constexpr,
                 BLOCK: tl.constexpr = 1024):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    value = tl.load(A+i,valid,0).to(tl.float32) + tl.load(B+i,valid,0).to(tl.float32)
    if COUNT >= 3: value = value + tl.load(C+i,valid,0).to(tl.float32)
    if COUNT >= 4: value = value + tl.load(D+i,valid,0).to(tl.float32)
    tl.store(OUT+i,value,valid)


@triton.jit
def resize_last_dim(X, OUT, ROWS: tl.constexpr, SOURCE: tl.constexpr,
                    DEST: tl.constexpr, BLOCK: tl.constexpr = 1024):
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row,col=i//DEST,i%DEST
    value=tl.load(X+row*SOURCE+col,(row<ROWS)&(col<SOURCE),0)
    tl.store(OUT+i,value,i<ROWS*DEST)


@triton.jit
def offset(index, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    result = tl.full(index.shape, 0, tl.int64)
    remaining = index
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        result += (remaining % SHAPE[axis]) * STRIDES[axis]
        remaining = remaining // SHAPE[axis]
    return result


@triton.jit
def fill(OUT, value, N: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + i, value, i < N)


@triton.jit
def masked_fill(X, MASK, OUT, value, SHAPE: tl.constexpr, XS: tl.constexpr, MS: tl.constexpr,
                N: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset(i, SHAPE, XS), i < N, 0)
    mask = tl.load(MASK + offset(i, SHAPE, MS), i < N, 0)
    tl.store(OUT + i, tl.where(mask != 0, x, value), i < N)


@triton.jit
def binary(A, B, OUT, SHAPE: tl.constexpr, AS: tl.constexpr, BS: tl.constexpr,
           N: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offset(i, SHAPE, AS), i < N, 0).to(tl.float32)
    b = tl.load(B + offset(i, SHAPE, BS), i < N, 0).to(tl.float32)
    if OP == 'add': y = a + b
    elif OP == 'mul': y = a * b
    elif OP == 'div': y = a / b
    elif OP == 'max': y = tl.maximum(a, b)
    elif OP == 'min': y = tl.minimum(a, b)
    elif OP == 'gt': y = a > b
    elif OP == 'ge': y = a >= b
    elif OP == 'eq': y = a == b
    tl.store(OUT + i, y, i < N)


@triton.jit
def binary_scalar(A, OUT, value, N: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i, i < N, 0).to(tl.float32)
    if OP == 'add': y = a + value
    elif OP == 'mul': y = a * value
    elif OP == 'div': y = a / value
    elif OP == 'max': y = tl.maximum(a, value)
    elif OP == 'min': y = tl.minimum(a, value)
    elif OP == 'gt': y = a > value
    elif OP == 'ge': y = a >= value
    elif OP == 'eq': y = a == value
    tl.store(OUT + i, y, i < N)


@triton.jit
def scalar_extreme_backward(A, G, OUT, value, N: tl.constexpr, MAXIMUM: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i, i < N, 0).to(tl.float32)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    active = a > value if MAXIMUM else a < value
    tl.store(OUT + i, g * (active.to(tl.float32) + (a == value).to(tl.float32) * .5), i < N)


@triton.jit
def squared_norm_partials(G, OUT, Scale, scale, N: tl.constexpr, OFFSET: tl.constexpr,
                          DEVICE_SCALE: tl.constexpr, BLOCK: tl.constexpr = 1024):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if DEVICE_SCALE: scale = tl.load(Scale)
    g = tl.load(G + i, i < N, 0).to(tl.float32) * scale
    tl.store(OUT + OFFSET + tl.program_id(0), tl.sum(g * g, 0))


@triton.jit
def sum_partials(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr = 1024):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + tl.program_id(0), tl.sum(tl.load(X + i, i < N, 0.).to(tl.float32), 0))


@triton.jit
def cross_entropy_recompute(Logits, Labels, Losses, Valid, DY, G, Denom,
                            VOCAB: tl.constexpr, BLOCK: tl.constexpr, BACKWARD: tl.constexpr,
                            SHIFT: tl.constexpr=False, SEQ: tl.constexpr=1):
    row=tl.program_id(0);col=tl.arange(0,BLOCK)
    source=row
    if SHIFT:source=(row//(SEQ-1))*SEQ+row%(SEQ-1)
    label=tl.load(Labels+source+(1 if SHIFT else 0));valid=(label>=0)&(label<VOCAB)
    x=tl.load(Logits+source*VOCAB+col,col<VOCAB,-float('inf')).to(tl.float32)
    maximum=tl.max(x,0);exponent=tl.exp(x-maximum);total=tl.sum(exponent,0)
    if BACKWARD:
        scale=tl.load(G)/tl.maximum(tl.load(Denom),1.)
        grad=tl.where(valid,(exponent/total-(col==label).to(tl.float32))*scale,0.)
        tl.store(DY+source*VOCAB+col,grad,col<VOCAB)
        if SHIFT:
            if row%(SEQ-1)==SEQ-2:
                tl.store(DY+(source+1)*VOCAB+col,0.,col<VOCAB)
    else:
        target=tl.sum(tl.where(col==label,x,0.),0)
        tl.store(Losses+row,tl.where(valid,maximum+tl.log(total)-target,0.))
        tl.store(Valid+row,valid.to(tl.float32))


@triton.jit
def unary(X, OUT, N: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    if OP == 'neg': y = -x
    elif OP == 'exp': y = tl.exp(x)
    elif OP == 'log': y = tl.log(x)
    elif OP == 'sqrt': y = tl.sqrt(x)
    elif OP == 'rsqrt': y = tl.rsqrt(x)
    elif OP == 'sigmoid': y = 1. / (1. + tl.exp(-x))
    elif OP == 'silu': y = x / (1. + tl.exp(-x))
    elif OP == 'relu': y = tl.maximum(x, 0.)
    elif OP == 'gelu': y = 0.5 * x * (1. + tl.erf(x * 0.7071067811865476))
    elif OP == 'softplus': y = tl.maximum(x, 0.) + tl.log(1. + tl.exp(-tl.abs(x)))
    elif OP == 'abs': y = tl.abs(x)
    tl.store(OUT + i, y, i < N)


@triton.jit
def unary_backward(X, Y, DY, DX, N: tl.constexpr, OP: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    y = tl.load(Y + i, i < N, 0).to(tl.float32)
    dy = tl.load(DY + i, i < N, 0).to(tl.float32)
    sig = 1. / (1. + tl.exp(-x))
    if OP == 'neg': d = -1.
    elif OP == 'exp': d = y
    elif OP == 'log': d = 1. / x
    elif OP == 'sqrt': d = 0.5 / y
    elif OP == 'rsqrt': d = -0.5 * y * y * y
    elif OP == 'sigmoid': d = y * (1. - y)
    elif OP == 'silu': d = sig * (1. + x * (1. - sig))
    elif OP == 'relu': d = (x > 0).to(tl.float32)
    elif OP == 'gelu': d = 0.5 * (1. + tl.erf(x * 0.7071067811865476)) + x * 0.3989422804014327 * tl.exp(-0.5 * x * x)
    elif OP == 'softplus': d = sig
    elif OP == 'abs': d = (x > 0).to(tl.float32) - (x < 0).to(tl.float32)
    tl.store(DX + i, dy * d, i < N)


@triton.jit
def reduce_axis(X, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                OP: tl.constexpr = 'sum', BLOCK: tl.constexpr = 256):
    row = tl.program_id(0)
    m, k = row // K, row % K
    idx = tl.arange(0, BLOCK)
    if OP == 'sum': acc = tl.full((BLOCK,), 0., tl.float32)
    else: acc = tl.full((BLOCK,), -float('inf'), tl.float32)
    for start in range(0, N, BLOCK):
        n = start + idx
        if OP == 'sum':
            x = tl.load(X + (m * N + n) * K + k, n < N, 0).to(tl.float32)
            acc += x
        else:
            x = tl.load(X + (m * N + n) * K + k, n < N, -float('inf')).to(tl.float32)
            acc = tl.maximum(acc, x)
    if OP == 'sum': value = tl.sum(acc, 0)
    else: value = tl.max(acc, 0)
    tl.store(OUT + row, value)


@triton.jit
def strided_copy(X, Y, SHAPE: tl.constexpr, STRIDES: tl.constexpr, OFFSET: tl.constexpr,
                 N: tl.constexpr, REVERSE: tl.constexpr = False, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mapped = offset(i, SHAPE, STRIDES) + OFFSET
    if REVERSE:
        tl.store(Y + mapped, tl.load(X + i, i < N, 0), i < N)
    else:
        tl.store(Y + i, tl.load(X + mapped, i < N, 0), i < N)


@triton.jit
def softmax(X, Y, N: tl.constexpr, LOG: tl.constexpr = False, BLOCK: tl.constexpr = 256):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mx = tl.full((BLOCK,), -float('inf'), tl.float32)
    for start in range(0, N, BLOCK):
        x = tl.load(X + row * N + start + col, start + col < N, -float('inf')).to(tl.float32)
        mx = tl.maximum(mx, x)
    maximum = tl.max(mx, 0)
    maximum = tl.where(maximum == -float('inf'), 0., maximum)
    accum = tl.full((BLOCK,), 0., tl.float32)
    for start in range(0, N, BLOCK):
        x = tl.load(X + row * N + start + col, start + col < N, -float('inf')).to(tl.float32)
        accum += tl.exp(x - maximum)
    total = tl.maximum(tl.sum(accum, 0), 1.e-30)
    for start in range(0, N, BLOCK):
        x = tl.load(X + row * N + start + col, start + col < N, -float('inf')).to(tl.float32)
        if LOG: y = x - maximum - tl.log(total)
        else: y = tl.exp(x - maximum) / total
        tl.store(Y + row * N + start + col, y, start + col < N)


@triton.jit
def softmax_backward(Y, DY, DX, N: tl.constexpr, LOG: tl.constexpr = False, BLOCK: tl.constexpr = 256):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    accum = tl.full((BLOCK,), 0., tl.float32)
    for start in range(0, N, BLOCK):
        y = tl.load(Y + row * N + start + col, start + col < N, 0).to(tl.float32)
        g = tl.load(DY + row * N + start + col, start + col < N, 0).to(tl.float32)
        if LOG: accum += g
        else: accum += g * y
    total = tl.sum(accum, 0)
    for start in range(0, N, BLOCK):
        y = tl.load(Y + row * N + start + col, start + col < N, 0).to(tl.float32)
        g = tl.load(DY + row * N + start + col, start + col < N, 0).to(tl.float32)
        if LOG: dx = g - tl.exp(y) * total
        else: dx = y * (g - total)
        tl.store(DX + row * N + start + col, dx, start + col < N)


@triton.jit
def gather_last(X, IDS, Y, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                 BACKWARD: tl.constexpr = False, BLOCK: tl.constexpr = 256):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(IDS + row, row < ROWS, -100)
    valid = (row < ROWS) & (ids >= 0) & (ids < WIDTH)
    if BACKWARD:
        value = tl.load(X + row, row < ROWS, 0)
        tl.store(Y + row * WIDTH + ids, value, valid)
    else:
        value = tl.load(X + row * WIDTH + ids, valid, 0.)
        tl.store(Y + row, value, row < ROWS)


@triton.jit
def dropout(X, Y, seed, counter, probability, N: tl.constexpr, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    keep = tl.rand(seed, (i + counter).to(tl.uint32)) >= probability
    tl.store(Y + i, tl.where(keep, x / (1. - probability), 0.), i < N)


@triton.jit
def scaled_copy(X, Y, scale, N: tl.constexpr, ADD: tl.constexpr = False, BLOCK: tl.constexpr = 256):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < N, 0).to(tl.float32) * scale
    if ADD: value += tl.load(Y + i, i < N, 0).to(tl.float32)
    tl.store(Y + i, value, i < N)


@triton.jit
def topk_mask(X, MASK, WIDTH: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + row * WIDTH + c, c < WIDTH, -float('inf')).to(tl.float32)
    chosen = tl.full((BLOCK,), 0., tl.float32)
    for _ in range(TOPK):
        maximum = tl.max(x, 0)
        index = tl.min(tl.where((x == maximum) & (c < WIDTH), c, 2147483647), 0)
        chosen += (c == index).to(tl.float32)
        x = tl.where(c == index, -float('inf'), x)
    tl.store(MASK + row * WIDTH + c, chosen, c < WIDTH)
