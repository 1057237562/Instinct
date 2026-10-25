"""Explicit forward/backward orchestration. No autograd, torch, or TorchAO.

Backend contract: empty(shape, dtype, zero=False), launch(fn, grid, buffers,
**constexprs). Tensors are contiguous runtime.Tensor objects. Caller owns the
saved contexts and frees/reuses their buffers; no hidden global activation tape.
"""
from dataclasses import dataclass
from .runtime import Scalar
import triton

from . import attention as attn
from . import elementwise as ew
from . import linear as lin


class DenseOps:
    def __init__(self, backend, fp8=True):
        self.backend = backend
        self.fp8 = fp8

    def empty(self, shape, dtype):
        return self.backend.empty(shape, dtype)

    def cast(self, x, dtype):
        if x.dtype == dtype:
            return x
        y = self.empty(x.shape, dtype)
        self.backend.launch(ew.cast, (triton.cdiv(x.numel, 256),), (x, y), SIZE=x.numel)
        return y

    def add(self, a, b):
        if a.shape != b.shape:
            raise ValueError('residual shapes differ')
        out = self.empty(a.shape, a.dtype)
        self.backend.launch(ew.add, (triton.cdiv(a.numel, 256),), (a, b, out), SIZE=a.numel)
        return out

    def quantize(self, x, dtype='fp8e4nv'):
        if dtype not in ('fp8e4nv', 'fp8e5') or x.dtype not in ('fp32', 'bf16', 'fp16'):
            raise ValueError('expected high-precision input and E4M3/E5M2 output')
        count = triton.cdiv(x.numel, 1024)
        if count <= 0:
            raise ValueError('tensorwise amax requires a nonempty tensor')
        maximum = 448.0 if dtype == 'fp8e4nv' else 57344.0
        partial = self.empty((count,), 'fp32')
        scale = self.empty((1,), 'fp32')
        out = self.empty(x.shape, dtype)
        self.backend.launch(lin.amax_partial, (count,), (x, partial), SIZE=x.numel, BLOCK=1024)
        while count > 1024:
            reduced = self.empty((triton.cdiv(count,1024),), 'fp32')
            self.backend.launch(lin.amax_partial, (reduced.numel,), (partial,reduced), SIZE=count,BLOCK=1024)
            partial,count = reduced,reduced.numel
        self.backend.launch(lin.scale_from_amax, (1,), (partial, scale), COUNT=count,
                            FP8_MAX=maximum, BLOCK=triton.next_power_of_2(count))
        self.backend.launch(lin.quantize, (triton.cdiv(x.numel,1024),), (x, scale, out),
                            SIZE=x.numel, FP8_MAX=maximum, BLOCK=1024)
        return out, scale

    def mm(self, a, b, transpose_a=False, transpose_b=False,
           scales=None, out_dtype='fp32'):
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError('GEMM requires matrices')
        m, k = a.shape[::-1] if transpose_a else a.shape
        kb, n = b.shape[::-1] if transpose_b else b.shape
        if k != kb:
            raise ValueError('GEMM inner dimensions differ')
        if scales is None and a.dtype != b.dtype:
            raise ValueError('unscaled GEMM inputs must have matching dtypes')
        am, ak = (1, a.shape[1]) if transpose_a else (a.shape[1], 1)
        bk, bn = (1, b.shape[1]) if transpose_b else (b.shape[1], 1)
        c = self.empty((m, n), out_dtype)
        sa, sb = scales if scales is not None else (a, b)  # constexpr-eliminated placeholders
        self.backend.launch(lin.matmul, (triton.cdiv(m, 32), triton.cdiv(n, 64)),
                            (a, b, c, sa, sb), M=m, N=n, K=k, AM=am, AK=ak,
                            BK=bk, BN=bn, SCALED=scales is not None)
        return c

    def linear(self, x, weight):
        if len(weight.shape) != 2 or x.shape[-1] != weight.shape[1]:
            raise ValueError('Linear input/weight shapes do not match')
        if x.dtype not in ('fp32', 'bf16', 'fp16') or weight.dtype not in ('fp32', 'bf16', 'fp16'):
            raise ValueError('Linear expects high-precision activations and master weights')
        matrix = x.view(x.numel // x.shape[-1], x.shape[-1])
        if self.fp8:
            qx, sx = self.quantize(matrix)
            qw, sw = self.quantize(weight)
            scales = (sx, sw)
        else:
            qx, qw = matrix, self.cast(weight, x.dtype)
            scales = None
        y = self.mm(qx, qw, transpose_b=True, scales=scales, out_dtype=x.dtype)
        saved = (x.shape, x.dtype, weight.dtype, qx, qw, scales)
        return y.view(*x.shape[:-1], weight.shape[0]), saved

    def linear_backward(self, saved, dy):
        shape, input_dtype, weight_dtype, qx, qw, scales = saved
        if dy.shape != (*shape[:-1], qw.shape[0]) or dy.dtype != input_dtype:
            raise ValueError('Linear gradient shape/dtype does not match its saved forward')
        dy = dy.view(dy.numel // dy.shape[-1], dy.shape[-1])
        if scales is not None:
            qdy, sdy = self.quantize(dy, 'fp8e5')
            sx, sw = scales
            dx = self.mm(qdy, qw, scales=(sdy, sw), out_dtype=dy.dtype)
            # FP32 dW for FP32 master parameters, avoiding an extra BF16 rounding.
            dw = self.mm(qdy, qx, transpose_a=True, scales=(sdy, sx), out_dtype=weight_dtype)
        else:
            dx = self.mm(dy, qw, out_dtype=dy.dtype)
            dw = self.mm(dy, qx, transpose_a=True, out_dtype=weight_dtype)
        return dx.view(*shape), dw

    def norm(self, x, weight, eps=1e-6):
        n, rows = x.shape[-1], x.numel // x.shape[-1]
        if not 0 < n <= 8192 or weight.shape != (n,):
            raise ValueError('RMSNorm requires width <= 8192 and a matching weight')
        if eps <= 0:
            raise ValueError('RMSNorm epsilon must be positive')
        y, r = self.empty(x.shape, x.dtype), self.empty((rows,), 'fp32')
        self.backend.launch(ew._rms_forward, (rows,), (x, weight, y, r),
                            N=n, EPS=eps, BLOCK=triton.next_power_of_2(n))
        return y, (x, weight, r)

    def norm_backward(self, saved, dy):
        x, w, r = saved
        if dy.shape != x.shape or dy.dtype != x.dtype:
            raise ValueError('RMSNorm gradient must match its input shape and dtype')
        n, rows = x.shape[-1], x.numel // x.shape[-1]
        groups = min(rows, 128)
        dx, dw = self.empty(x.shape, x.dtype), self.empty(w.shape, w.dtype)
        partial = self.empty((groups, n), 'fp32')
        self.backend.launch(ew._rms_backward, (groups,), (x, w, r, dy, dx, partial),
                            M=rows, N=n, GROUPS=groups, BLOCK=triton.next_power_of_2(n))
        self.backend.launch(ew._rms_reduce_dw, (triton.cdiv(n, 64),), (partial, dw),
                            N=n, GROUPS=groups, BLOCK_G=triton.next_power_of_2(groups), BLOCK_N=64)
        return dx, dw

    def rotary(self, x, cos, sin, backward=False, start=0):
        if len(x.shape) != 4 or x.shape[-1] % 2:
            raise ValueError('RoPE requires [batch, seq, heads, even dim]')
        b, s, h, d = x.shape
        if cos.shape != sin.shape or cos.shape[1:] != (d,) or start < 0 or start + s > cos.shape[0]:
            raise ValueError('RoPE tables do not cover the requested positions')
        y = self.empty(x.shape, x.dtype)
        self.backend.launch(ew.rotary, (b * s * h,), (x, cos, sin, y), SEQ=s,
                            HEADS=h, DIM=d, START=start, BACKWARD=backward, BLOCK=triton.next_power_of_2(d))
        return y

    def swiglu(self, gate, up):
        if gate.shape != up.shape or gate.dtype != up.dtype:
            raise ValueError('SwiGLU shapes/dtypes differ')
        out = self.empty(gate.shape, gate.dtype)
        self.backend.launch(ew._swiglu_forward, (triton.cdiv(gate.numel, 256),),
                            (gate, up, out), SIZE=gate.numel, BLOCK=256)
        return out, (gate, up)

    def swiglu_backward(self, saved, dy):
        gate, up = saved
        if dy.shape != gate.shape or dy.dtype != gate.dtype:
            raise ValueError('SwiGLU gradient must match its input shape and dtype')
        dg, du = self.empty(gate.shape, gate.dtype), self.empty(up.shape, up.dtype)
        self.backend.launch(ew._swiglu_backward, (triton.cdiv(gate.numel, 256),),
                            (gate, up, dy, dg, du), SIZE=gate.numel, BLOCK=256)
        return dg, du

    def attention(self, q, k, v, mask=None, dropout=0., seed=0, counter=0, cached=None, packed=False):
        if len(q.shape) != 4 or len(k.shape) != 4 or k.shape != v.shape:
            raise ValueError('GQA requires [batch, seq, heads, dim] inputs')
        b, s, hq, d = q.shape
        if k.shape[:2] != (b, s) or k.shape[3] != d or hq % k.shape[2] or d > 256:
            raise ValueError('GQA shapes mismatch; head_dim must be <= 256')
        if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in ('fp32', 'bf16', 'fp16'):
            raise ValueError('attention uses matching FP32/BF16/FP16 inputs')
        if mask is not None and (mask.shape != (b, s) or mask.dtype not in ('i32', 'i64')):
            raise ValueError('attention mask must be integer [batch, seq]')
        if not 0 <= dropout < 1:
            raise ValueError('attention dropout must be in [0, 1)')
        tile = 16 if d > 128 else 32
        tile = getattr(self.backend,'attention_tile',None) or tile
        if tile not in (16,32,64):raise ValueError('attention tile must be 16, 32 or 64')
        options = dict(BM=tile, BN=tile, DROPOUT=float(dropout), SEQ=s, HQ=hq, HKV=k.shape[2], DIM=d, SCALE=d ** -0.5,
                       HAS_MASK=mask is not None, BD=max(16, triton.next_power_of_2(d)), PACKED=packed,
                       PACKED_SKIP=getattr(self.backend,'packed_skip',True))
        if packed and mask is None: raise ValueError('packed attention requires segment IDs')
        out, lse = cached if cached is not None else (self.empty(q.shape, q.dtype), self.empty((b, hq, s), 'fp32'))
        mask = mask if mask is not None else q
        if cached is None:
            self.backend.launch(attn.forward, (triton.cdiv(s, tile), b * hq), (q, k, v, mask, out, lse, Scalar(seed, 'i32'), Scalar(counter, 'i64')), **options)
        return out, (q, k, v, mask, out, lse, options, seed, counter)

    def attention_backward(self, saved, dy):
        q, k, v, mask, out, lse, options, seed, counter = saved
        if dy.shape != q.shape or dy.dtype != q.dtype:
            raise ValueError('attention gradient must match its output shape and dtype')
        b, s, hq, _ = q.shape
        dq, dk, dv = [self.empty(x.shape, x.dtype) for x in (q, k, v)]
        # dKV visits each query row for many KV tiles. Compute sum(dO*O)
        # once per row instead of repeatedly loading O and reducing head_dim.
        delta = self.empty((b, hq, s), 'fp32')
        self.backend.launch(attn.backward_delta, (b * s * hq,), (out, dy, delta),
                            SEQ=s, HQ=hq, DIM=q.shape[-1], BD=options['BD'])
        self.backend.launch(attn.backward_q, (triton.cdiv(s, options['BM']), b * hq),
                            (q, k, v, mask, delta, lse, dy, dq, Scalar(seed, 'i32'), Scalar(counter, 'i64')),
                            PRECOMPUTED_DELTA=True, **options)
        self.backend.launch(attn.backward_kv, (triton.cdiv(s, options['BN']), b * k.shape[2]),
                            (q, k, v, mask, delta, lse, dy, dk, dv, Scalar(seed, 'i32'), Scalar(counter, 'i64')),
                            PRECOMPUTED_DELTA=True, **options)
        return dq, dk, dv


@dataclass(frozen=True)
class BlockConfig:
    hidden_size: int = 768
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 96
    intermediate_size: int = 2432
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.hidden_size, self.num_attention_heads, self.num_key_value_heads,
               self.head_dim, self.intermediate_size) <= 0:
            raise ValueError('Dense dimensions must be positive')
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2 or self.head_dim > 128:
            raise ValueError('requires divisible GQA head counts and even head_dim <= 128')
        if self.dropout != 0:
            raise ValueError('this kernel implementation currently supports dropout=0 only')


class DenseBlock:
    """Instinct Dense block with an explicit backward and native state-dict names."""
    def __init__(self, ops, weights, config=BlockConfig()):
        self.ops, self.weights, self.config = ops, weights, config
        c = config
        expected = {'input_layernorm.weight': (c.hidden_size,),
                    'post_attention_layernorm.weight': (c.hidden_size,),
                    'self_attn.q_norm.weight': (c.head_dim,), 'self_attn.k_norm.weight': (c.head_dim,)}
        for name, shape in {'q_proj': (c.num_attention_heads * c.head_dim, c.hidden_size),
                            'k_proj': (c.num_key_value_heads * c.head_dim, c.hidden_size),
                            'v_proj': (c.num_key_value_heads * c.head_dim, c.hidden_size),
                            'o_proj': (c.hidden_size, c.num_attention_heads * c.head_dim)}.items():
            expected[f'self_attn.{name}.weight'] = shape
        for name, shape in {'gate_proj': (c.intermediate_size, c.hidden_size),
                            'up_proj': (c.intermediate_size, c.hidden_size),
                            'down_proj': (c.hidden_size, c.intermediate_size)}.items():
            expected[f'mlp.{name}.weight'] = shape
        for name, shape in expected.items():
            if name not in weights or weights[name].shape != shape:
                raise ValueError(f'missing or invalid Dense parameter: {name}, expected {shape}')

    def forward(self, x, cos, sin, mask=None):
        op, c, w = self.ops, self.config, self.weights
        if len(x.shape) != 3 or x.shape[-1] != c.hidden_size:
            raise ValueError('Dense input must have shape [batch, seq, hidden_size]')
        b, s, _ = x.shape
        saved = {'cos': cos, 'sin': sin, 'shape': x.shape}
        def norm(name, value):
            result, saved[name] = op.norm(value, w[name + '.weight'], c.rms_norm_eps)
            return result
        def linear(name, value):
            result, saved[name] = op.linear(value, w[name + '.weight'])
            return result
        xn = norm('input_layernorm', x)
        q = linear('self_attn.q_proj', xn).view(b, s, c.num_attention_heads, c.head_dim)
        k = linear('self_attn.k_proj', xn).view(b, s, c.num_key_value_heads, c.head_dim)
        v = linear('self_attn.v_proj', xn).view(b, s, c.num_key_value_heads, c.head_dim)
        q = op.rotary(norm('self_attn.q_norm', q), cos, sin)
        k = op.rotary(norm('self_attn.k_norm', k), cos, sin)
        a, saved['attention'] = op.attention(q, k, v, mask)
        residual = op.add(x, linear('self_attn.o_proj', a.view(b, s, c.num_attention_heads * c.head_dim)))
        xn = norm('post_attention_layernorm', residual)
        gate, up = linear('mlp.gate_proj', xn), linear('mlp.up_proj', xn)
        ffn, saved['swiglu'] = op.swiglu(gate, up)
        y = op.add(residual, linear('mlp.down_proj', ffn))
        return y, saved

    def backward(self, saved, dy):
        op, c = self.ops, self.config
        if dy.shape != saved['shape']:
            raise ValueError('Dense gradient shape does not match its saved forward')
        grads = {}
        def linear(name, value):
            dx, grads[name + '.weight'] = op.linear_backward(saved[name], value)
            return dx
        def norm(name, value):
            dx, grads[name + '.weight'] = op.norm_backward(saved[name], value)
            return dx
        df = linear('mlp.down_proj', dy)
        dg, du = op.swiglu_backward(saved['swiglu'], df)
        dn = op.add(linear('mlp.gate_proj', dg), linear('mlp.up_proj', du))
        dr = op.add(dy, norm('post_attention_layernorm', dn))
        da = linear('self_attn.o_proj', dr)
        b, s, _ = saved['shape']
        dq, dk, dv = op.attention_backward(saved['attention'], da.view(b, s, c.num_attention_heads, c.head_dim))
        dq = norm('self_attn.q_norm', op.rotary(dq, saved['cos'], saved['sin'], backward=True))
        dk = norm('self_attn.k_norm', op.rotary(dk, saved['cos'], saved['sin'], backward=True))
        dq = linear('self_attn.q_proj', dq.view(b, s, c.num_attention_heads * c.head_dim))
        dk = linear('self_attn.k_proj', dk.view(b, s, c.num_key_value_heads * c.head_dim))
        dv = linear('self_attn.v_proj', dv.view(b, s, c.num_key_value_heads * c.head_dim))
        dx = op.add(dr, norm('input_layernorm', op.add(op.add(dq, dk), dv)))
        return dx, grads
