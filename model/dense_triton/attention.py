"""Causal GQA with online softmax and recomputed backward, deterministic attention dropout.

Layout is [batch, sequence, heads, head_dim], matching Instinct. No repeated
K/V heads and no [sequence, sequence] saved tensor. KV gradients sum query
head groups in registers, avoiding atomics. These are correctness-first tiles;
Dropout addresses use a 64-bit logical [batch, head, query, key] counter,
so masks remain reproducible across tiling/recompute without wrapping at 2**32.
Hardware-specific scheduling/autotuning remains a benchmark-driven follow-up.
"""
import triton
import triton.language as tl


@triton.jit
def forward(Q, K, V, MASK, O, LSE, seed, offset, SEQ: tl.constexpr, HQ: tl.constexpr,
            HKV: tl.constexpr, DIM: tl.constexpr, SCALE: tl.constexpr,
            HAS_MASK: tl.constexpr, DROPOUT: tl.constexpr = 0., BM: tl.constexpr = 32,
            BN: tl.constexpr = 32, BD: tl.constexpr = 128, PACKED: tl.constexpr = False,
            PACKED_SKIP: tl.constexpr = True):
    qh = tl.program_id(1) % HQ
    batch = tl.program_id(1) // HQ
    kh = qh // (HQ // HKV)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.arange(0, BN)
    d = tl.arange(0, BD)
    qo = ((batch * SEQ + m[:, None]) * HQ + qh) * DIM + d[None, :]
    q = tl.load(Q + qo, (m[:, None] < SEQ) & (d[None, :] < DIM), other=0)
    maxval = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM, BD), 0, tl.float32)
    begin = 0
    if PACKED and PACKED_SKIP:
        # Packed metadata stores the contiguous document's start offset.
        documents=tl.load(MASK+batch*SEQ+m,m<SEQ,-1)
        begin=(tl.min(tl.where((m<SEQ)&(documents>=0),documents,SEQ),0)//BN*BN).to(tl.int32)
    for start in range(begin, tl.minimum((tl.program_id(0) + 1) * BM, SEQ), BN):
        nn = start + n
        ko = ((batch * SEQ + nn[None, :]) * HKV + kh) * DIM + d[:, None]
        k = tl.load(K + ko, (nn[None, :] < SEQ) & (d[:, None] < DIM), other=0)
        v = tl.load(V + tl.trans(ko), (nn[:, None] < SEQ) & (d[None, :] < DIM), other=0)
        scores = tl.dot(q, k, input_precision="ieee") * SCALE
        allowed = (m[:, None] >= nn[None, :]) & (nn[None, :] < SEQ)
        if HAS_MASK:
            valid = tl.load(MASK + batch * SEQ + nn, nn < SEQ, other=0)
            if PACKED:
                query_doc = tl.load(MASK + batch * SEQ + m, m < SEQ, other=-1)
                allowed = allowed & (query_doc[:, None] == valid[None, :]) & (query_doc[:, None] >= 0)
            else:
                allowed = allowed & (valid[None, :] != 0)
        scores = tl.where(allowed, scores, -float("inf"))
        new_max = tl.maximum(maxval, tl.max(scores, 1))
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.exp(maxval - safe_max)
        p = tl.where(allowed, tl.exp(scores - safe_max[:, None]), 0.0)
        acc = acc * alpha[:, None]
        if DROPOUT > 0.:
            index = ((batch * HQ + qh) * SEQ + m[:, None]).to(tl.int64) * SEQ + nn[None, :] + offset
            kept = tl.rand(seed, index) >= DROPOUT
            pd = tl.where(kept, p / (1. - DROPOUT), 0.)
        else:
            pd = p
        acc = tl.dot(pd.to(q.dtype), v, acc, input_precision="ieee")
        denom = denom * alpha + tl.sum(p, 1)
        maxval = new_max
    out = acc / tl.where(denom > 0, denom, 1.0)[:, None]
    lse = tl.where(denom > 0, maxval + tl.log(tl.where(denom > 0, denom, 1.0)), float("inf"))
    tl.store(O + qo, out, (m[:, None] < SEQ) & (d[None, :] < DIM))
    tl.store(LSE + (batch * HQ + qh) * SEQ + m, lse, m < SEQ)


@triton.jit
def backward_delta(O, DO, Delta, SEQ: tl.constexpr, HQ: tl.constexpr,
                   DIM: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    batch, token, head = row // (SEQ * HQ), (row // HQ) % SEQ, row % HQ
    d = tl.arange(0, BD)
    o = tl.load(O + row * DIM + d, d < DIM, 0.).to(tl.float32)
    do = tl.load(DO + row * DIM + d, d < DIM, 0.).to(tl.float32)
    tl.store(Delta + (batch * HQ + head) * SEQ + token, tl.sum(o * do, 0))


@triton.jit
def backward_q(Q, K, V, MASK, O, LSE, DO, DQ, seed, offset, SEQ: tl.constexpr,
               HQ: tl.constexpr, HKV: tl.constexpr, DIM: tl.constexpr,
               SCALE: tl.constexpr, HAS_MASK: tl.constexpr, DROPOUT: tl.constexpr = 0.,
               BM: tl.constexpr = 32, BN: tl.constexpr = 32, BD: tl.constexpr = 128,
               PRECOMPUTED_DELTA: tl.constexpr = False, PACKED: tl.constexpr = False,
               PACKED_SKIP: tl.constexpr = True):
    qh = tl.program_id(1) % HQ
    batch = tl.program_id(1) // HQ
    kh = qh // (HQ // HKV)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.arange(0, BN)
    d = tl.arange(0, BD)
    qo = ((batch * SEQ + m[:, None]) * HQ + qh) * DIM + d[None, :]
    mask = (m[:, None] < SEQ) & (d[None, :] < DIM)
    q = tl.load(Q + qo, mask, other=0)
    do = tl.load(DO + qo, mask, other=0)
    if PRECOMPUTED_DELTA:
        delta = tl.load(O + (batch * HQ + qh) * SEQ + m, m < SEQ, 0.)
    else:
        o = tl.load(O + qo, mask, other=0).to(tl.float32)
        delta = tl.sum(do.to(tl.float32) * o, 1)
    lse = tl.load(LSE + (batch * HQ + qh) * SEQ + m, m < SEQ, other=float("inf"))
    dq = tl.full((BM, BD), 0, tl.float32)
    begin = 0
    if PACKED and PACKED_SKIP:
        documents=tl.load(MASK+batch*SEQ+m,m<SEQ,-1)
        begin=(tl.min(tl.where((m<SEQ)&(documents>=0),documents,SEQ),0)//BN*BN).to(tl.int32)
    for start in range(begin, tl.minimum((tl.program_id(0) + 1) * BM, SEQ), BN):
        nn = start + n
        ko = ((batch * SEQ + nn[None, :]) * HKV + kh) * DIM + d[:, None]
        kmask = (nn[None, :] < SEQ) & (d[:, None] < DIM)
        k = tl.load(K + ko, kmask, other=0)
        v = tl.load(V + ko, kmask, other=0)
        allowed = (m[:, None] >= nn[None, :]) & (nn[None, :] < SEQ)
        if HAS_MASK:
            valid = tl.load(MASK + batch * SEQ + nn, nn < SEQ, other=0)
            if PACKED:
                query_doc = tl.load(MASK + batch * SEQ + m, m < SEQ, other=-1)
                allowed = allowed & (query_doc[:, None] == valid[None, :]) & (query_doc[:, None] >= 0)
            else:
                allowed = allowed & (valid[None, :] != 0)
        scores = tl.dot(q, k, input_precision="ieee") * SCALE
        p = tl.where(allowed, tl.exp(scores - lse[:, None]), 0.0)
        dp = tl.dot(do, v, input_precision="ieee")
        if DROPOUT > 0.:
            index = ((batch * HQ + qh) * SEQ + m[:, None]).to(tl.int64) * SEQ + nn[None, :] + offset
            kept = tl.rand(seed, index) >= DROPOUT
            dp = tl.where(kept, dp / (1. - DROPOUT), 0.)
        ds = (p * (dp - delta[:, None]) * SCALE).to(q.dtype)
        dq = tl.dot(ds, tl.trans(k), dq, input_precision="ieee")
    tl.store(DQ + qo, dq, mask)


@triton.jit
def backward_kv(Q, K, V, MASK, O, LSE, DO, DK, DV, seed, offset, SEQ: tl.constexpr,
                HQ: tl.constexpr, HKV: tl.constexpr, DIM: tl.constexpr,
                SCALE: tl.constexpr, HAS_MASK: tl.constexpr, DROPOUT: tl.constexpr = 0.,
                BM: tl.constexpr = 32, BN: tl.constexpr = 32, BD: tl.constexpr = 128,
                PRECOMPUTED_DELTA: tl.constexpr = False, PACKED: tl.constexpr = False,
                PACKED_SKIP: tl.constexpr = True):
    kh = tl.program_id(1) % HKV
    batch = tl.program_id(1) // HKV
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    m = tl.arange(0, BM)
    d = tl.arange(0, BD)
    ko = ((batch * SEQ + n[:, None]) * HKV + kh) * DIM + d[None, :]
    kmask = (n[:, None] < SEQ) & (d[None, :] < DIM)
    k = tl.load(K + ko, kmask, other=0)
    v = tl.load(V + ko, kmask, other=0)
    valid = n < SEQ
    if HAS_MASK:
        key_doc = tl.load(MASK + batch * SEQ + n, n < SEQ, other=-1 if PACKED else 0)
        valid = valid & ((key_doc >= 0) if PACKED else (key_doc != 0))
    dk = tl.full((BN, BD), 0, tl.float32)
    dv = tl.full((BN, BD), 0, tl.float32)
    end = SEQ
    if PACKED and PACKED_SKIP:
        last_doc=tl.max(tl.where(valid,key_doc,-1),0)
        positions=tl.arange(0,triton.next_power_of_2(SEQ))
        documents=tl.load(MASK+batch*SEQ+positions,positions<SEQ,-1)
        end=tl.max(tl.where((positions<SEQ)&(documents==last_doc)&(last_doc>=0),positions+1,0),0).to(tl.int32)
    for group in range(HQ // HKV):
        qh = kh * (HQ // HKV) + group
        # BM == BN in the public launcher: earlier query tiles are all masked.
        for start in range(tl.program_id(0) * BN, end, BM):
            mm = start + m
            qo = ((batch * SEQ + mm[:, None]) * HQ + qh) * DIM + d[None, :]
            qmask = (mm[:, None] < SEQ) & (d[None, :] < DIM)
            q = tl.load(Q + qo, qmask, other=0)
            do = tl.load(DO + qo, qmask, other=0)
            if PRECOMPUTED_DELTA:
                delta = tl.load(O + (batch * HQ + qh) * SEQ + mm, mm < SEQ, 0.)
            else:
                o = tl.load(O + qo, qmask, other=0).to(tl.float32)
                delta = tl.sum(do.to(tl.float32) * o, 1)
            lse = tl.load(LSE + (batch * HQ + qh) * SEQ + mm, mm < SEQ, other=float("inf"))
            allowed = (mm[:, None] >= n[None, :]) & (mm[:, None] < SEQ) & valid[None, :]
            if PACKED:
                query_doc = tl.load(MASK + batch * SEQ + mm, mm < SEQ, other=-1)
                allowed = allowed & (query_doc[:, None] == key_doc[None, :]) & (query_doc[:, None] >= 0)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
            p = tl.where(allowed, tl.exp(scores - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            if DROPOUT > 0.:
                index = ((batch * HQ + qh) * SEQ + mm[:, None]).to(tl.int64) * SEQ + n[None, :] + offset
                kept = tl.rand(seed, index) >= DROPOUT
                dp = tl.where(kept, dp / (1. - DROPOUT), 0.)
                pd = tl.where(kept, p / (1. - DROPOUT), 0.)
            else:
                pd = p
            ds = (p * (dp - delta[:, None]) * SCALE).to(q.dtype)
            dk = tl.dot(tl.trans(ds), q, dk, input_precision="ieee")
            dv = tl.dot(tl.trans(pd.to(q.dtype)), do, dv, input_precision="ieee")
    tl.store(DK + ko, dk, kmask)
    tl.store(DV + ko, dv, kmask)
