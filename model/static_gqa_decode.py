"""Length-aware, single-query attention over the compact static KV cache."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode(
    Q, K, V, Start, Position, Finished, Out, Partial, Stats,
    QB: tl.constexpr, QH: tl.constexpr,
    KB: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr,
    VB: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr,
    POS_STRIDE: tl.constexpr, KV_HEADS: tl.constexpr, GROUP: tl.constexpr,
    HEADS: tl.constexpr, DIM: tl.constexpr, CAPACITY: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr,
):
    row_head = tl.program_id(0)
    split = tl.program_id(1)
    row, kv_head = row_head // KV_HEADS, row_head % KV_HEADS
    start = tl.load(Start + row).to(tl.int32)
    end = tl.minimum(tl.load(Position + row * POS_STRIDE).to(tl.int32) + 1, CAPACITY)
    end = tl.where(tl.load(Finished + row), start, end)
    # Partition live tiles, so a large allocated capacity costs no extra KV reads.
    tiles = tl.cdiv(tl.maximum(end - start, 0), BLOCK_T)
    split_tiles = tl.cdiv(tiles, SPLITS)
    first = start + split * split_tiles * BLOCK_T
    last = tl.minimum(first + split_tiles * BLOCK_T, end)
    group = tl.arange(0, 16)
    d = tl.arange(0, BLOCK_D)
    head = kv_head * GROUP + group
    q = tl.load(Q + row * QB + head[:, None] * QH + d[None, :],
                (first < last) & (group[:, None] < GROUP) & (d[None, :] < DIM), other=0)
    # A finite sentinel keeps empty splits well-defined through the final reduction.
    maximum = tl.full((16,), -1.0e30, tl.float32)
    denominator = tl.full((16,), 0, tl.float32)
    numerator = tl.full((16, BLOCK_D), 0, tl.float32)
    for offset in range(first, last, BLOCK_T):
        pos = offset + tl.arange(0, BLOCK_T)
        valid = (pos[:, None] < last) & (d[None, :] < DIM)
        k = tl.load(K + row * KB + pos[:, None] * KT + kv_head * KH + d[None, :], valid, other=0)
        v = tl.load(V + row * VB + pos[:, None] * VT + kv_head * VH + d[None, :], valid, other=0)
        score = tl.dot(q, tl.trans(k)) * (1.4426950408889634 / tl.sqrt(float(DIM)))
        score = tl.where(pos[None, :] < last, score, -1.0e30)
        new_maximum = tl.maximum(maximum, tl.max(score, axis=1))
        correction = tl.exp2(maximum - new_maximum)
        p = tl.exp2(score - new_maximum[:, None])
        numerator = tl.dot(p.to(v.dtype), v, numerator * correction[:, None])
        denominator = denominator * correction + tl.sum(p, axis=1)
        maximum = new_maximum

    output_row = row * HEADS + head
    active_head = group < GROUP
    if SPLITS == 1:
        y = numerator / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(Out + output_row[:, None] * DIM + d[None, :], y,
                 active_head[:, None] & (d[None, :] < DIM))
    else:
        part = output_row * SPLITS + split
        tl.store(Partial + part[:, None] * DIM + d[None, :], numerator,
                 active_head[:, None] & (d[None, :] < DIM))
        tl.store(Stats + part * 2, maximum, active_head)
        tl.store(Stats + part * 2 + 1, denominator, active_head)


@triton.jit
def _merge(Partial, Stats, Out, SPLITS: tl.constexpr, DIM: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    split = tl.arange(0, SPLITS)
    d = tl.arange(0, BLOCK_D)
    part = row * SPLITS + split
    maximum = tl.load(Stats + part * 2)
    denominator = tl.load(Stats + part * 2 + 1)
    weights = tl.exp2(maximum - tl.max(maximum, axis=0))
    total = tl.sum(weights * denominator, axis=0)
    numerators = tl.load(Partial + part[:, None] * DIM + d[None, :], d[None, :] < DIM, other=0)
    y = tl.sum(weights[:, None] * numerators, axis=0) / tl.where(total > 0, total, 1.0)
    tl.store(Out + row * DIM + d, y, d < DIM)


def static_gqa_attention(q, k, v, pad_starts, positions, finished):
    """Q is [B, 1, Hq, D], K/V are [B, capacity, Hkv, D]."""
    batch, _, heads, dim = q.shape
    _, capacity, kv_heads, _ = k.shape
    group = heads // kv_heads
    block_d = triton.next_power_of_2(dim)
    block_t = 64
    # Short batches need more CTAs to occupy the device. Split only on tiles
    # and let each program derive its range from the current device positions.
    splits = min(16, triton.next_power_of_2(triton.cdiv(256, batch * kv_heads)),
                 triton.next_power_of_2(triton.cdiv(capacity, block_t)))
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    partial = stats = out
    if splits > 1:
        partial = torch.empty((batch, heads, splits, dim), device=q.device, dtype=torch.float32)
        stats = torch.empty((batch, heads, splits, 2), device=q.device, dtype=torch.float32)
    position_stride = positions.stride(0) if positions.ndim == 2 and positions.shape[0] > 1 else 0
    _decode[(batch * kv_heads, splits)](
        q, k, v, pad_starts, positions, finished, out, partial, stats,
        q.stride(0), q.stride(2), *k.stride()[:3], *v.stride()[:3],
        position_stride, kv_heads, group, heads, dim, capacity, splits, block_d, block_t,
        num_warps=4,
    )
    if splits > 1:
        _merge[(batch * heads,)](partial, stats, out, splits, dim, block_d, num_warps=4)
    return out
