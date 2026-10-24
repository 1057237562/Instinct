"""Small explicit VJP engine over Triton kernels and cuBLASLt.

There is no Torch import or fallback. Graphs are Python metadata; tensors and
gradients remain on the selected backend. CPU reference backends are test-only.
"""
from contextlib import contextmanager
import math
import numpy as np
import triton

from model.dense_triton.runtime import Scalar
from model.dense_triton import elementwise as ew
from . import tensor_kernels as tk
from .ops import CublasDenseOps


def strides(shape):
    return tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))


class Value:
    def __init__(self, engine, data, requires_grad=False, node=None, name=None):
        self.engine, self.data, self.requires_grad, self.node, self.name = engine, data, requires_grad, node, name

    @property
    def shape(self): return self.data.shape
    @property
    def dtype(self): return self.data.dtype
    @property
    def size(self): return self.data.numel
    @property
    def T(self): return self.permute(1, 0)
    def detach(self): return Value(self.engine, self.data)
    def reshape(self, *shape): return self.engine.reshape(self, shape)
    def permute(self, *axes): return self.engine.permute(self, axes)
    def sum(self, axis=None, keepdims=False): return self.engine.reduce(self, axis, keepdims)
    def mean(self, axis=None, keepdims=False):
        count = self.size if axis is None else math.prod(self.shape[i] for i in ((axis,) if isinstance(axis, int) else axis))
        return self.sum(axis, keepdims) / count
    def cast(self, dtype): return self.engine.cast(self, dtype)
    def exp(self): return self.engine.unary(self, 'exp')
    def log(self): return self.engine.unary(self, 'log')
    def sqrt(self): return self.engine.unary(self, 'sqrt')
    def rsqrt(self): return self.engine.unary(self, 'rsqrt')
    def sigmoid(self): return self.engine.unary(self, 'sigmoid')
    def softplus(self): return self.engine.unary(self, 'softplus')
    def clip(self, low, high): return self.engine.binary(self.engine.binary(self, low, 'max'), high, 'min')
    def __add__(self, x): return self.engine.binary(self, x, 'add')
    def __radd__(self, x): return self + x
    def __neg__(self): return self.engine.unary(self, 'neg')
    def __sub__(self, x): return self + (-x if isinstance(x, Value) else -float(x))
    def __rsub__(self, x): return -self + x
    def __mul__(self, x): return self.engine.binary(self, x, 'mul')
    def __rmul__(self, x): return self * x
    def __truediv__(self, x): return self.engine.binary(self, x, 'div')
    def __rtruediv__(self, x): return self.engine.binary(self.engine.full((1,), x), self, 'div')
    def __matmul__(self, x): return self.engine.mm(self, x)
    def __getitem__(self, key): return self.engine.slice(self, key)


class Engine:
    def __init__(self, backend, fp8=False, seed=42):
        self.backend, self.ops = backend, CublasDenseOps(backend, fp8)
        self.recording, self.seed, self.counter = True, int(seed), 0

    def wrap(self, data, requires_grad=False, name=None):
        return Value(self, data, requires_grad, name=name)

    def array(self, array, dtype=None, requires_grad=False, name=None):
        array = np.asarray(array)
        if array.dtype.kind == 'f': array = array.astype(np.float32, copy=False)
        elif array.dtype.kind in 'biu': array = array.astype(np.int64, copy=False)
        if not array.shape: array = array.reshape(1)
        data = self.backend.from_numpy(array)
        if dtype: data = self.ops.cast(data, dtype)
        return self.wrap(data, requires_grad, name)

    def full(self, shape, value, dtype='fp32'):
        data = self.backend.empty(tuple(shape) or (1,), dtype)
        self.backend.launch(tk.fill, (triton.cdiv(data.numel, 256),), (data, Scalar(value)), N=data.numel)
        return self.wrap(data)

    def numpy(self, value):
        return self.backend.to_numpy(value.data if isinstance(value, Value) else value)

    def item(self, value): return float(self.numpy(value).reshape(-1)[0])

    def _node(self, data, parents, backward):
        required = self.recording and any(x.requires_grad for x in parents)
        return Value(self, data, required, (tuple(parents), backward) if required else None)

    @contextmanager
    def no_grad(self):
        prior, self.recording = self.recording, False
        try: yield
        finally: self.recording = prior

    def cast(self, x, dtype):
        if x.dtype == dtype: return x
        return self._node(self.ops.cast(x.data, dtype), (x,), lambda g: (self.ops.cast(g, x.dtype),))

    def reshape(self, x, shape):
        shape = list(shape)
        if shape.count(-1) == 1:
            index = shape.index(-1)
            shape[index] = x.size // -math.prod(shape)
        if not shape: shape = [1]
        return self._node(x.data.view(*shape), (x,), lambda g: (g.view(*x.shape),))

    def _broadcast_strides(self, shape, output):
        aligned = (1,) * (len(output) - len(shape)) + shape
        return tuple(0 if d == 1 else s for d, s in zip(aligned, strides(aligned)))

    def sum_to(self, x, shape):
        while len(x.shape) > len(shape): x = x.sum(0)
        for axis, dim in enumerate(shape):
            if dim == 1 and x.shape[axis] != 1: x = x.sum(axis, keepdims=True)
        return x.reshape(*shape)

    def binary(self, a, b, op):
        scalar = not isinstance(b, Value)
        if scalar: b = self.full((1,), b)
        shape = tuple(np.broadcast_shapes(a.shape, b.shape))
        dtype = a.dtype if scalar or a.dtype == b.dtype else 'fp32'
        if op in ('gt', 'ge', 'eq'): dtype = 'fp32'
        out = self.backend.empty(shape, dtype)
        self.backend.launch(tk.binary, (triton.cdiv(out.numel, 256),), (a.data, b.data, out),
            SHAPE=shape, AS=self._broadcast_strides(a.shape, shape), BS=self._broadcast_strides(b.shape, shape), N=out.numel, OP=op)
        def backward(g):
            g = self.wrap(g)
            if op == 'add': ga, gb = g, g
            elif op == 'mul': ga, gb = g * b, g * a
            elif op == 'div': ga, gb = g / b, -g * a / (b * b)
            elif op in ('min', 'max'):
                cmp = self.binary(a, b, 'gt') if op == 'max' else self.binary(b, a, 'gt')
                part = cmp + self.binary(a, b, 'eq') * 0.5
                ga, gb = g * part, g * (1. - part)
            else: return None, None
            return self.sum_to(ga, a.shape).data, self.sum_to(gb, b.shape).data
        return self.wrap(out) if op in ('gt', 'ge', 'eq') else self._node(out, (a, b), backward)

    def unary(self, x, op):
        out = self.backend.empty(x.shape, x.dtype)
        self.backend.launch(tk.unary, (triton.cdiv(x.size, 256),), (x.data, out), N=x.size, OP=op)
        def backward(g):
            dx = self.backend.empty(x.shape, x.dtype)
            self.backend.launch(tk.unary_backward, (triton.cdiv(x.size, 256),), (x.data, out, g, dx), N=x.size, OP=op)
            return (dx,)
        return self._node(out, (x,), backward)

    def reduce(self, x, axis=None, keepdims=False, op='sum'):
        axes = tuple(range(len(x.shape))) if axis is None else ((axis,) if isinstance(axis, int) else tuple(axis))
        axes = sorted({a % len(x.shape) for a in axes}, reverse=True)
        out = x
        for a in axes:
            current = out
            shape = list(current.shape)
            m, n, k = math.prod(shape[:a]), shape[a], math.prod(shape[a + 1:])
            shape[a] = 1
            data = self.backend.empty(tuple(shape), 'fp32')
            self.backend.launch(tk.reduce_axis, (m * k,), (current.data, data), M=m, N=n, K=k, OP=op)
            def backward(g, src=current, reduced=data, axis=a):
                gradient = self.full(src.shape, 1.) * self.wrap(g)
                if op == 'max':
                    equal = self.binary(src, self.wrap(reduced), 'eq')
                    gradient = gradient * equal / equal.sum(axis, True)
                return (gradient.data,)
            out = self._node(data, (current,), backward)
            if not keepdims:
                target = tuple(d for i, d in enumerate(shape) if i != a) or (1,)
                out = out.reshape(*target)
        return out

    def _mapped(self, x, shape, source_strides, offset=0):
        out = self.backend.empty(tuple(shape), x.dtype)
        options = dict(SHAPE=tuple(shape), STRIDES=tuple(source_strides), OFFSET=offset, N=out.numel)
        self.backend.launch(tk.strided_copy, (triton.cdiv(out.numel, 256),), (x.data, out), **options)
        def backward(g):
            dx = self.backend.empty(x.shape, 'fp32', zero=True)
            self.backend.launch(tk.strided_copy, (triton.cdiv(out.numel, 256),), (g, dx), REVERSE=True, **options)
            return (dx,)
        return self._node(out, (x,), backward)

    def permute(self, x, axes):
        if sorted(axes) != list(range(len(x.shape))): raise ValueError('invalid permutation')
        return self._mapped(x, tuple(x.shape[a] for a in axes), tuple(strides(x.shape)[a] for a in axes))

    def slice(self, x, key):
        if not isinstance(key, tuple): key = (key,)
        if Ellipsis in key:
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (len(x.shape) - len(key) + 1) + key[i + 1:]
        key = key + (slice(None),) * (len(x.shape) - len(key))
        if len(key) != len(x.shape): raise ValueError('index rank mismatch')
        shape, ss, offset = [], [], 0
        for dim, stride, part in zip(x.shape, strides(x.shape), key):
            if isinstance(part, int):
                index = part + dim if part < 0 else part
                if not 0 <= index < dim: raise IndexError(index)
                offset += index * stride
            elif isinstance(part, slice):
                start, stop, step = part.indices(dim)
                count = len(range(start, stop, step))
                if count < 1: raise ValueError('empty tensor slices are not supported')
                shape.append(count); ss.append(stride * step); offset += start * stride
            else: raise TypeError('only slices and integer indices are supported')
        return self._mapped(x, shape or (1,), ss or (0,), offset)

    def cat(self, values, axis=0):
        shape = list(values[0].shape)
        axis %= len(shape)
        shape[axis] = sum(x.shape[axis] for x in values)
        out = self.backend.empty(tuple(shape), values[0].dtype)
        starts, start = [], 0
        for x in values:
            if any(d != shape[i] for i, d in enumerate(x.shape) if i != axis): raise ValueError('cat dimensions mismatch')
            starts.append(start)
            self.backend.launch(tk.strided_copy, (triton.cdiv(x.size, 256),), (x.data, out),
                SHAPE=x.shape, STRIDES=strides(shape), OFFSET=start * strides(shape)[axis], N=x.size, REVERSE=True)
            start += x.shape[axis]
        def backward(g):
            result = []
            for x, start in zip(values, starts):
                key = [slice(None)] * len(shape); key[axis] = slice(start, start + x.shape[axis])
                result.append(self.slice(self.wrap(g), tuple(key)).data)
            return tuple(result)
        return self._node(out, values, backward)

    def mm(self, a, b):
        if len(a.shape) != 2 or len(b.shape) != 2: raise ValueError('mm requires two matrices')
        dtype = a.dtype if a.dtype == b.dtype else 'fp32'
        aa, bb = self.ops.cast(a.data, dtype), self.ops.cast(b.data, dtype)
        out = self.ops.mm(aa, bb, out_dtype=dtype)
        def backward(g):
            gg = self.ops.cast(g, dtype)
            return (self.ops.mm(gg, bb, transpose_b=True), self.ops.mm(aa, gg, transpose_a=True))
        return self._node(out, (a, b), backward)

    def linear(self, x, weight):
        out, saved = self.ops.linear(x.data, weight.data)
        def backward(g):
            result = self.ops.linear_backward(saved, self.ops.cast(g, x.dtype), x.requires_grad, weight.requires_grad)
            if weight.requires_grad: self.backend.finish_backward()
            return result
        return self._node(out, (x, weight), backward)

    def norm(self, x, weight, eps=1e-6):
        out, saved = self.ops.norm(x.data, weight.data, eps)
        return self._node(out, (x, weight), lambda g: self.ops.norm_backward(saved, self.ops.cast(g, x.dtype)))

    def rotary(self, x, cos, sin, start=0):
        out = self.ops.rotary(x.data, cos.data, sin.data, start=start)
        return self._node(out, (x,), lambda g: (self.ops.rotary(g, cos.data, sin.data, start=start, backward=True),))

    def embedding(self, weight, ids):
        shape = (*ids.shape, weight.shape[1])
        out = self.backend.empty(shape, weight.dtype)
        options = dict(TOKENS=ids.size, DIM=weight.shape[1], VOCAB=weight.shape[0])
        grid = (ids.size, triton.cdiv(weight.shape[1], 256))
        self.backend.launch(ew.embedding, grid, (ids.data, weight.data, out), **options)
        def backward(g):
            dw = self.backend.empty(weight.shape, 'fp32', zero=True)
            self.backend.launch(ew.embedding_backward, grid, (ids.data, g, dw), **options)
            return (dw,)
        return self._node(out, (weight,), backward)

    def attention(self, q, k, v, mask=None, dropout=0., training=True, flash=True, start=0):
        if flash and start == 0 and q.shape[1] == k.shape[1] and q.shape[-1] <= 256:
            probability = dropout if training else 0.
            counter = self.counter
            if probability: self.counter += q.shape[0] * q.shape[2] * q.shape[1] ** 2
            out, saved = self.ops.attention(q.data, k.data, v.data, None if mask is None else mask.data,
                                            dropout=probability, seed=self.seed, counter=counter)
            return self._node(out, (q, k, v), lambda g: self.ops.attention_backward(saved, self.ops.cast(g, q.dtype)))
        b, seq, heads, dim = q.shape
        kv_heads, length = k.shape[2], k.shape[1]
        # The general path also handles arbitrary head dimensions and decode offsets.
        causal = np.arange(length)[None, :] <= (np.arange(seq)[:, None] + start)
        bias = self.array(np.where(causal, 0., -np.inf).astype(np.float32))
        rows = []
        for batch in range(b):
            batch_bias = bias
            if mask is not None:
                # Avoid 0 * -inf: masked_fill has an explicit predicate.
                valid = mask[batch].reshape(1, length) * self.array(causal.astype(np.float32))
                batch_bias = self.masked_fill(self.full((seq, length), 0.), valid, -float('inf'))
            hs = []
            for head in range(heads):
                kh = head // (heads // kv_heads)
                scores = (q[batch, :, head, :].cast('fp32') @ k[batch, :, kh, :].cast('fp32').T) / math.sqrt(dim)
                probs = self.dropout(self.softmax(scores + batch_bias), dropout, training).cast(v.dtype)
                hs.append((probs @ v[batch, :, kh, :]).reshape(seq, 1, dim))
            rows.append(self.cat(hs, 1).reshape(1, seq, heads, dim))
        return self.cat(rows, 0).cast(q.dtype)

    def masked_fill(self, x, mask, value=0.):
        shape = tuple(np.broadcast_shapes(x.shape, mask.shape))
        out = self.backend.empty(shape, x.dtype)
        options = dict(SHAPE=shape, XS=self._broadcast_strides(x.shape, shape), MS=self._broadcast_strides(mask.shape, shape), N=out.numel)
        self.backend.launch(tk.masked_fill, (triton.cdiv(out.numel, 256),), (x.data, mask.data, out, Scalar(value)), **options)
        return self._node(out, (x,), lambda g: (self.sum_to(self.wrap(g) * mask, x.shape).data,))

    def scatter_rows(self, x, ids, rows):
        """Sum rows into destinations; repeated IDs are allowed (MoE dispatch)."""
        out = self.backend.empty((rows, x.shape[1]), x.dtype, zero=True)
        self.backend.launch(ew.embedding_backward, (ids.size, triton.cdiv(x.shape[1], 256)),
            (ids.data, x.data, out), TOKENS=ids.size, DIM=x.shape[1], VOCAB=rows)
        return self._node(out, (x,), lambda g: (self.embedding(self.wrap(g), ids).data,))

    def softmax(self, x, log=False):
        out = self.backend.empty(x.shape, 'fp32')
        rows, width = x.size // x.shape[-1], x.shape[-1]
        self.backend.launch(tk.softmax, (rows,), (x.data, out), N=width, LOG=log)
        def backward(g):
            dx = self.backend.empty(x.shape, 'fp32')
            self.backend.launch(tk.softmax_backward, (rows,), (out, g, dx), N=width, LOG=log)
            return (dx,)
        return self._node(out, (x,), backward)

    def gather_last(self, x, ids):
        if x.shape[:-1] != ids.shape: raise ValueError('gather shape mismatch')
        out = self.backend.empty(ids.shape, x.dtype)
        options = dict(ROWS=ids.size, WIDTH=x.shape[-1])
        grid = (triton.cdiv(ids.size, 256),)
        self.backend.launch(tk.gather_last, grid, (x.data, ids.data, out), **options)
        def backward(g):
            dx = self.backend.empty(x.shape, 'fp32', zero=True)
            self.backend.launch(tk.gather_last, grid, (g, ids.data, dx), BACKWARD=True, **options)
            return (dx,)
        return self._node(out, (x,), backward)

    def log_probs(self, logits, labels): return self.gather_last(self.softmax(logits, log=True), labels)

    def ce(self, logits, labels):
        valid = self.binary(labels, 0., 'ge') * self.binary(self.full((1,), logits.shape[-1]), labels, 'gt')
        return -(self.log_probs(logits, labels) * valid).sum() / self.binary(valid.sum(), 1., 'max')

    def dropout(self, x, probability, training=True):
        if not training or probability == 0: return x
        if not 0 <= probability < 1: raise ValueError('dropout must be in [0, 1)')
        seed, counter = self.seed, self.counter
        self.counter += x.size
        def apply(data):
            out = self.backend.empty(x.shape, x.dtype)
            self.backend.launch(tk.dropout, (triton.cdiv(x.size, 256),),
                (data, out, Scalar(seed, 'i32'), Scalar(counter, 'i64'), Scalar(probability)), N=x.size)
            return out
        return self._node(apply(x.data), (x,), lambda g: (apply(g),))

    def topk_mask(self, x, k):
        if not 1 <= k <= x.shape[-1]: raise ValueError('invalid top-k')
        out = self.backend.empty(x.shape, 'fp32')
        self.backend.launch(tk.topk_mask, (x.size // x.shape[-1],), (x.data, out),
            WIDTH=x.shape[-1], TOPK=k, BLOCK=triton.next_power_of_2(x.shape[-1]))
        return self.wrap(out)

    def vjp(self, output, gradient=None, inputs=None):
        """Reverse traversal, accumulating tied/shared parameters exactly once."""
        ordered, seen, stack = [], set(), [(output, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded: ordered.append(node); continue
            if node in seen: continue
            seen.add(node); stack.append((node, True))
            if node.node:
                stack.extend((p, False) for p in node.node[0] if p.requires_grad)
        with self.no_grad():
            gradients = {output: gradient if gradient is not None else self.full(output.shape, 1.).data}
            for value in reversed(ordered):
                g = gradients.get(value)
                if g is None or value.node is None: continue
                parents, backward = value.node
                local = backward(g)
                if len(local) != len(parents): raise RuntimeError('VJP returned wrong gradient count')
                for parent, grad in zip(parents, local):
                    if not parent.requires_grad or grad is None: continue
                    grad = self.ops.cast(grad, 'fp32')
                    if parent in gradients: grad = (self.wrap(grad) + self.wrap(gradients[parent])).data
                    gradients[parent] = grad
            if inputs is not None: return tuple(gradients.get(x) for x in inputs)
            return {value: grad for value, grad in gradients.items() if value.node is None and value.requires_grad}

    def checkpoint(self, fn, x, parameters):
        """Discard forward temporaries; reproduce RNG and regenerate a local VJP."""
        if not self.recording: return fn(x)
        mark, seed, counter = self.backend.mark(), self.seed, self.counter
        with self.no_grad(): out = fn(x)
        self.backend.release_since(mark, keep=(out.data,))
        parents = (x, *parameters)
        def backward(g):
            mark = self.backend.mark()
            prior = self.recording, self.seed, self.counter
            self.recording, self.seed, self.counter = True, seed, counter
            try:
                leaf = self.wrap(x.data, True)
                regenerated = fn(leaf)
                result = self.vjp(regenerated, g, (leaf, *parameters))
                self.backend.release_since(mark, keep=result)
                return result
            finally:
                self.recording, self.seed, self.counter = prior
        return self._node(out.data, parents, backward)

    def copy(self, target, source, scale=1., add=False):
        if target.shape != source.shape: raise ValueError('copy shape mismatch')
        self.backend.launch(tk.scaled_copy, (triton.cdiv(target.size, 256),),
            (source.data, target.data, Scalar(scale)), N=target.size, ADD=add)
