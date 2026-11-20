"""Inference-only static KV and fused GQA decode for the native cuBLAS engine.

Loaded by file path by the chat worker, which may use a separate native checkout.
No training methods or global runtime classes are changed.
"""
from types import MethodType

import triton
import triton.language as tl


@triton.jit
def _store_kv(K, V, KC, VC, position, N: tl.constexpr, WIDTH: tl.constexpr,
              BLOCK: tl.constexpr = 256, DEVICE_POSITION: tl.constexpr = False):
    if DEVICE_POSITION:
        position = tl.load(position).to(tl.int32)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(KC + position * WIDTH + i, tl.load(K + i, i < N, 0), i < N)
    tl.store(VC + position * WIDTH + i, tl.load(V + i, i < N, 0), i < N)


@triton.jit
def _norm_rope(X, W, COS, SIN, Y, position, D: tl.constexpr,
               EPS: tl.constexpr, BLOCK: tl.constexpr, DEVICE_POSITION: tl.constexpr = False):
    if DEVICE_POSITION:
        position = tl.load(position).to(tl.int32)
    head = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    other = (d + D // 2) % D
    x = tl.load(X + head * D + d, d < D, 0).to(tl.float32)
    w = tl.load(W + d, d < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    partner = tl.load(X + head * D + other, d < D, 0).to(tl.float32)
    partner_w = tl.load(W + other, d < D, 0).to(tl.float32)
    # Preserve the native runtime's BF16 rounding between RMSNorm and RoPE.
    z = (x * inv * w).to(X.dtype.element_ty).to(tl.float32)
    rotated = (partner * inv * partner_w).to(X.dtype.element_ty).to(tl.float32)
    cos = tl.load(COS + position * D + d, d < D, 0)
    sin = tl.load(SIN + position * D + d, d < D, 0)
    tl.store(Y + head * D + d, z * cos + tl.where(d < D // 2, -rotated, rotated) * sin, d < D)


@triton.jit
def _scores(Q, K, S, length, HQ: tl.constexpr, HK: tl.constexpr,
            D: tl.constexpr, CAP: tl.constexpr, BD: tl.constexpr,
            BT: tl.constexpr = 128, DEVICE_POSITION: tl.constexpr = False):
    if DEVICE_POSITION:
        length = tl.load(length).to(tl.int32) + 1
    head, part = tl.program_id(0), tl.program_id(1)
    t = part * BT + tl.arange(0, BT)
    d = tl.arange(0, BD)
    q = tl.load(Q + head * D + d, d < D, 0).to(tl.float32)
    k = tl.load(K + t[:, None] * HK * D + (head // (HQ // HK)) * D + d[None, :],
                (t[:, None] < length) & (d[None, :] < D), 0).to(tl.float32)
    score = tl.sum(k * q[None, :], 1) * (D ** -0.5)
    tl.store(S + head * CAP + t, score, t < length)


@triton.jit
def _probabilities(S, P, length, CAP: tl.constexpr, BLOCK: tl.constexpr,
                    DEVICE_POSITION: tl.constexpr = False):
    if DEVICE_POSITION:
        length = tl.load(length).to(tl.int32) + 1
    head = tl.program_id(0)
    t = tl.arange(0, BLOCK)
    s = tl.load(S + head * CAP + t, t < length, -float('inf'))
    p = tl.exp(s - tl.max(s, 0))
    p = p / tl.sum(p, 0)
    # The reference native attention stores probabilities in the value dtype.
    tl.store(P + head * CAP + t, p, t < length)


@triton.jit
def _weighted_values(P, V, PARTIAL, length, HQ: tl.constexpr, HK: tl.constexpr,
                     D: tl.constexpr, CAP: tl.constexpr, SPLITS: tl.constexpr,
                     BD: tl.constexpr, BT: tl.constexpr = 128, DEVICE_POSITION: tl.constexpr = False):
    if DEVICE_POSITION:
        length = tl.load(length).to(tl.int32) + 1
    head, part = tl.program_id(0), tl.program_id(1)
    t = part * BT + tl.arange(0, BT)
    d = tl.arange(0, BD)
    p = tl.load(P + head * CAP + t, t < length, 0).to(tl.float32)
    v = tl.load(V + t[:, None] * HK * D + (head // (HQ // HK)) * D + d[None, :],
                (t[:, None] < length) & (d[None, :] < D), 0).to(tl.float32)
    result = tl.sum(p[:, None] * v, 0)
    tl.store(PARTIAL + (head * SPLITS + part) * D + d, result, d < D)


@triton.jit
def _combine(PARTIAL, OUT, parts, D: tl.constexpr, SPLITS: tl.constexpr,
             BS: tl.constexpr, BD: tl.constexpr, DEVICE_POSITION: tl.constexpr = False):
    if DEVICE_POSITION:
        parts = tl.cdiv(tl.load(parts).to(tl.int32) + 1, 128)
    head = tl.program_id(0)
    s, d = tl.arange(0, BS), tl.arange(0, BD)
    v = tl.load(PARTIAL + (head * SPLITS + s[:, None]) * D + d[None, :],
                (s[:, None] < parts) & (d[None, :] < D), 0)
    tl.store(OUT + head * D + d, tl.sum(v, 0), d < D)


def fused_decode(engine, q, keys, values, length, position=None):
    """Single-query unmasked GQA. The cache suffix is never read."""
    from model.dense_triton.runtime import Scalar
    b = engine.backend
    _, _, hq, d = q.shape
    _, cap, hk, _ = keys.shape
    if not 0 < length <= cap or hq % hk:
        raise ValueError('invalid native decode cache length or GQA heads')
    splits = triton.cdiv(cap, 128)
    active = splits if position is not None else triton.cdiv(length, 128)
    scores = b.empty((hq, cap), 'fp32')
    probs = b.empty((hq, cap), values.dtype)
    partial = b.empty((hq, splits, d), 'fp32')
    out = b.empty(q.shape, q.dtype)
    size = position if position is not None else Scalar(length, 'i32')
    dynamic = position is not None
    options = dict(HQ=hq, HK=hk, D=d, CAP=cap, BD=triton.next_power_of_2(d), DEVICE_POSITION=dynamic)
    b.launch(_scores, (hq, active), (q.data, keys.data, scores, size), **options)
    b.launch(_probabilities, (hq,), (scores, probs, size), CAP=cap,
             BLOCK=triton.next_power_of_2(cap), DEVICE_POSITION=dynamic)
    b.launch(_weighted_values, (hq, active), (probs, values.data, partial, size), SPLITS=splits, **options)
    b.launch(_combine, (hq,), (partial, out, position if dynamic else Scalar(active, 'i32')),
             D=d, SPLITS=splits, BS=triton.next_power_of_2(splits), BD=triton.next_power_of_2(d), DEVICE_POSITION=dynamic)
    return engine.wrap(out)


def _attention(self, x, i, mask, cache, new_cache, start, cache_key, packed=None, rotary=None):
    original = self._chat_original_attention
    if (self.training or self.e.recording or new_cache is None or mask is not None
            or packed is not None or x.shape[0] != 1 or (start and x.shape[1] != 1)):
        if cache and isinstance(cache.get(cache_key), dict):
            raise ValueError('static native chat cache requires single-query unmasked inference')
        return original(x, i, mask, cache, new_cache, start, cache_key, packed, rotary)
    from model.dense_triton.runtime import Scalar
    e, c, b = self.e, self.config, self.e.backend
    d, hq, hk = c['head_dim'], c['num_attention_heads'], c['num_key_value_heads']
    position = getattr(self, '_chat_graph_position', None)
    pos_arg = position if position is not None else Scalar(start, 'i32')
    if not start:
        out = original(x, i, mask, cache, new_cache, start, cache_key, packed, rotary)
        pk, pv = new_cache[cache_key]
        k, v = self._unpack_cache(pk), self._unpack_cache(pv)
        capacity = max(x.shape[1], getattr(self, '_chat_cache_capacity', c['max_position_embeddings']))
        keys = e.wrap(b.empty((1, capacity, hk, d), self.kv_cache_dtype))
        values = e.wrap(b.empty((1, capacity, hk, d), self.kv_cache_dtype))
        new_cache[cache_key] = dict(keys=keys, values=values)
    else:
        saved = cache[cache_key]
        if not isinstance(saved, dict):
            return original(x, i, mask, cache, new_cache, start, cache_key, packed, rotary)
        keys, values = saved['keys'], saved['values']
        if start >= keys.shape[1]:
            raise ValueError('native chat KV capacity exhausted')
        p = self.layer_prefixes[i] + 'self_attn.'
        with e.ops.reuse_scope():
            q, k, v = self.projections(x, [p + name + '_proj.weight' for name in ('q', 'k', 'v')])
        cos, sin = rotary or (self.cos, self.sin)
        def normalize(value, heads, name):
            result = b.empty((1, 1, heads, d), value.dtype)
            b.launch(_norm_rope, (heads,),
                     (value.data, self.params[p + name + '_norm.weight'].data, cos.data, sin.data,
                      result, pos_arg), D=d, EPS=c['rms_norm_eps'], BLOCK=triton.next_power_of_2(d),
                      DEVICE_POSITION=position is not None)
            return e.wrap(result)
        q, k = normalize(q, hq, 'q'), normalize(k, hk, 'k')
        v = v.reshape(1, 1, hk, d)
        new_cache[cache_key] = saved
    count = x.shape[1] * hk * d
    b.launch(_store_kv, (triton.cdiv(count, 256),),
             (k.data, v.data, keys.data, values.data, pos_arg), N=count, WIDTH=hk*d,
             DEVICE_POSITION=position is not None)
    if start:
        out = self.linear(fused_decode(e, q, keys, values, start + 1, position).reshape(1, 1, hq*d), p + 'o_proj.weight')
    return out


def grow_cache(model, cache, capacity):
    """Allocate a larger bucket only after the previous graph has been closed."""
    from model.dense_triton.runtime import Scalar
    e, b = model.e, model.e.backend
    length = cache['position']
    for saved in cache.values():
        if not isinstance(saved, dict) or 'keys' not in saved:
            continue
        keys, values = saved['keys'], saved['values']
        _, old_capacity, hk, dim = keys.shape
        if capacity <= old_capacity:
            continue
        new_keys = e.wrap(b.empty((1, capacity, hk, dim), keys.dtype))
        new_values = e.wrap(b.empty((1, capacity, hk, dim), values.dtype))
        count = length * hk * dim
        b.launch(_store_kv, (triton.cdiv(count, 256),),
                 (keys.data, values.data, new_keys.data, new_values.data, Scalar(0, 'i32')), N=count, WIDTH=hk*dim)
        saved.update(keys=new_keys, values=new_values)


class DecodeGraph:
    """One request-owned CUDA graph; token and position are device inputs."""
    def __init__(self, model, cache, token):
        import numpy as np
        from model.dense_triton.runtime import Tensor
        self.model, self.cache = model, cache
        self.b, self.e = model.e.backend, model.e
        self.mark = self.b.mark()
        self.prior_pinned = set(self.b._pinned)
        self.plan = None
        try:
            self.host_inputs = np.asarray([token, cache['position']], np.int64)
            self.inputs = self.b.from_numpy(self.host_inputs)
            self.ids = self.e.wrap(Tensor((1, 1), 'i64', self.inputs.storage, self.b))
            position = Tensor((1,), 'i64', self.inputs.storage + 8, self.b)
            model._chat_graph_position = position
            def forward():
                x = self.e.embedding(model.params['model.embed_tokens.weight'], self.ids).cast(model.activation_dtype)
                rotary = (model.cos, model.sin) if cache.get('rope_long') or model.short_rope is None else model.short_rope
                new_cache = {}
                for i in range(model.config['num_hidden_layers']):
                    layer_mark = self.b.mark()
                    x, _ = model.block(x, i, cache=cache, new_cache=new_cache,
                                       start=cache['position'], rotary=rotary)
                    self.b.release_since(layer_mark, keep=(x.data, *model.cache_tensors(cache)))
                return model.head(x)
            self.plan = self.b.build(forward)
            self.b._pinned.add(self.inputs.storage)
            self.plan.capture()
        except BaseException:
            self.close()
            raise
        finally:
            model.__dict__.pop('_chat_graph_position', None)

    def step(self, token, position):
        self.host_inputs[:] = token, position
        self.b.copy_from_numpy(self.inputs, self.host_inputs)
        return self.plan.replay()

    def close(self):
        if self.plan is not None:
            self.b.synchronize()
            if self.plan.graph_exec:
                self.b._graph_exec_destroy(self.plan.graph_exec)
            if self.plan.graph:
                self.b._graph_destroy(self.plan.graph)
            self.plan.graph_exec = self.plan.graph = None
        self.b.execution_plan = None
        self.b._plan_storage.clear()
        self.b._pinned.intersection_update(self.prior_pinned)
        self.b.release_since(self.mark)
        self.plan = None


def install(model):
    """Enable only the validated standard Dense BF16 single-sequence path."""
    c = model.config
    eligible = (c['model_architecture'] == 'standard' and not c['use_moe']
                and model.activation_dtype == model.kv_cache_dtype == 'bf16'
                and 0 < c['head_dim'] <= 256 and c['head_dim'] % 2 == 0
                and c['num_attention_heads'] % c['num_key_value_heads'] == 0)
    if not eligible or hasattr(model, '_chat_original_attention'):
        return False
    model._chat_original_attention = model.attention
    model.attention = MethodType(_attention, model)
    return True
