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
    def __init__(self, engine, data, requires_grad=False, node=None, name=None, gradient_dtype='fp32'):
        self.engine, self.data, self.requires_grad, self.node, self.name = engine, data, requires_grad, node, name
        self.gradient_dtype = gradient_dtype

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


class PendingGradient:
    def __init__(self, data, event):self.data,self.event=data,event


class GradientSum:
    """Ordered FP32 additions that a compatible consumer can perform in registers."""
    def __init__(self,parts):self.parts=tuple(parts)


class Engine:
    def __init__(self, backend, fp8=False, seed=42, fp8_filter='eligible'):
        self.backend, self.ops = backend, CublasDenseOps(backend, fp8)
        self.recording, self.seed, self.counter = True, int(seed), 0
        self.replay_cache = None
        if fp8_filter not in ('auto','eligible'):raise ValueError('unknown FP8 filter')
        self.fp8_filter=fp8_filter

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

    def _node(self, data, parents, backward, gradient_dtype='fp32'):
        required = self.recording and any(x.requires_grad for x in parents)
        return Value(self, data, required, (tuple(parents), backward) if required else None,
                     gradient_dtype=gradient_dtype)

    @contextmanager
    def no_grad(self):
        prior, self.recording = self.recording, False
        try: yield
        finally: self.recording = prior

    def cast(self, x, dtype):
        if x.dtype == dtype: return x
        return self._node(self.ops.cast(x.data, dtype), (x,), lambda g: (self.ops.cast(g, x.dtype),), 'preserve')

    def reshape(self, x, shape):
        shape = list(shape)
        if shape.count(-1) == 1:
            index = shape.index(-1)
            shape[index] = x.size // -math.prod(shape)
        if not shape: shape = [1]
        return self._node(x.data.view(*shape), (x,), lambda g: (g.view(*x.shape),), 'preserve')

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
        if scalar:
            dtype = 'fp32' if op in ('gt', 'ge', 'eq') else a.dtype
            out = self.backend.empty(a.shape, dtype)
            self.backend.launch(tk.binary_scalar, (triton.cdiv(a.size, 256),),
                                (a.data, out, Scalar(float(b))), N=a.size, OP=op)
            def backward(g):
                if op == 'add': return (g,)
                if op in ('mul', 'div'): return (self.binary(self.wrap(g), b, op).data,)
                dx = self.backend.empty(a.shape, 'fp32')
                self.backend.launch(tk.scalar_extreme_backward, (triton.cdiv(a.size, 256),),
                                    (a.data, g, dx, Scalar(float(b))), N=a.size, MAXIMUM=op == 'max')
                return (dx,)
            grad_dtype='preserve' if op=='add' and getattr(self.backend,'compact_gradients',False) else 'fp32'
            return self.wrap(out) if op in ('gt', 'ge', 'eq') else self._node(out, (a,), backward,grad_dtype)
        shape = tuple(np.broadcast_shapes(a.shape, b.shape))
        dtype = a.dtype if scalar or a.dtype == b.dtype else 'fp32'
        if op in ('gt', 'ge', 'eq'): dtype = 'fp32'
        out = self.backend.empty(shape, dtype)
        self.backend.launch(tk.binary, (triton.cdiv(out.numel, 256),), (a.data, b.data, out),
            SHAPE=shape, AS=self._broadcast_strides(a.shape, shape), BS=self._broadcast_strides(b.shape, shape), N=out.numel, OP=op)
        def backward(g):
            g = self.wrap(g)
            need_a, need_b = a.requires_grad, b.requires_grad
            if op == 'add': ga, gb = (g if need_a else None), (g if need_b else None)
            elif op == 'mul': ga, gb = (g * b if need_a else None), (g * a if need_b else None)
            elif op == 'div': ga, gb = (g / b if need_a else None), (-g * a / (b * b) if need_b else None)
            elif op in ('min', 'max'):
                cmp = self.binary(a, b, 'gt') if op == 'max' else self.binary(b, a, 'gt')
                part = cmp + self.binary(a, b, 'eq') * 0.5
                ga, gb = (g * part if need_a else None), (g * (1. - part) if need_b else None)
            else: return None, None
            return (None if ga is None else self.sum_to(ga, a.shape).data,
                    None if gb is None else self.sum_to(gb, b.shape).data)
        grad_dtype='preserve' if op=='add' and getattr(self.backend,'compact_gradients',False) else 'fp32'
        return self.wrap(out) if op in ('gt', 'ge', 'eq') else self._node(out, (a, b), backward,grad_dtype)

    def squared_norm(self, values, scale=1.):
        """Optimizer-only reduction: O(N/1024) scratch, no full-size products."""
        values = list(values)
        if not values: return self.full((1,), 0.)
        sizes = [triton.cdiv(x.size, 1024) for x in values]
        partials = self.backend.empty((sum(sizes),), 'fp32')
        device_scale = isinstance(scale, Value)
        scale_data = scale.data if device_scale else values[0].data
        offset = 0
        for value, blocks in zip(values, sizes):
            self.backend.launch(tk.squared_norm_partials, (blocks,),
                (value.data, partials, scale_data, Scalar(1. if device_scale else scale)),
                N=value.size, OFFSET=offset, DEVICE_SCALE=device_scale)
            offset += blocks
        return self.sum_partials(partials)

    def sum_partials(self,partials):
        # A single CTA scanning ~200k partials becomes another serial tail.
        # Reduce hierarchically so every CTA reads at most 1024 elements.
        while partials.numel > 1:
            reduced = self.backend.empty((triton.cdiv(partials.numel, 1024),), 'fp32')
            self.backend.launch(tk.sum_partials, (reduced.numel,), (partials, reduced), N=partials.numel)
            partials = reduced
        return self.wrap(partials)

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
                dx = self.backend.empty(src.shape, 'fp32')
                ss = list(strides(reduced.shape)); ss[axis] = 0
                self.backend.launch(tk.strided_copy, (triton.cdiv(src.size, 256),), (g, dx),
                                    SHAPE=src.shape, STRIDES=tuple(ss), OFFSET=0, N=src.size)
                gradient = self.wrap(dx)
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

    def cat(self, values, axis=0, dtype=None):
        shape = list(values[0].shape)
        axis %= len(shape)
        shape[axis] = sum(x.shape[axis] for x in values)
        out = self.backend.empty(tuple(shape), dtype or values[0].dtype)
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
        previous=self.ops.fp8
        if self.fp8_filter=='auto':
            n,k=weight.shape
            self.ops.fp8=previous and n%16==0 and k%16==0 and not (k<=4096 and n<=1024) and 'lm_head' not in (weight.name or '') and 'embed_tokens' not in (weight.name or '')
        try:
            return self._selected_linear(x,weight)
        finally:
            self.ops.fp8=previous

    def _selected_linear(self,x,weight):
        cache = self.replay_cache
        selected = cache is not None and weight.name in cache.linear_names
        if selected and cache.replay:
            out = cache.values[weight.name]
            saved = self.ops.linear_saved(x.data, weight.data)
        else:
            out, saved = self.ops.linear(x.data, weight.data)
            if selected:
                cache.values[weight.name] = out
        return self._linear_node(x,weight,out,saved)

    def linears(self,x,weights):
        if self.replay_cache is not None or self.fp8_filter=='auto':
            return [self.linear(x, w) for w in weights]
        return [self._linear_node(x,w,out,saved) for w,(out,saved)
                in zip(weights,self.ops.linears(x.data,[w.data for w in weights]))]

    def _linear_node(self,x,weight,out,saved):
        def backward(g):
            return self._linear_vjp(x,weight,saved,g)
        value=self._node(out, (x, weight), backward, x.dtype)
        value.quantized_gradient=saved[-1] is not None
        return value

    def _linear_vjp(self,x,weight,saved,g):
        dx,dw=self.ops.linear_backward(saved,self.ops.cast(g,x.dtype),x.requires_grad,weight.requires_grad)
        if dw is not None and self.backend.weight_stream!=self.backend.main_stream:
            event=self.backend.create_event()
            with self.backend.on_stream(self.backend.weight_stream):self.backend.record_event(event)
            dw=PendingGradient(dw,event)
        return dx,dw

    def linear_residual_replay(self,x,weight,residual,output):
        """Rebuild only the VJP of a terminal residual + Linear operation.

        output is the original checkpoint result, not a newly evaluated Linear.
        Neither Linear's VJP nor addition's VJP reads the terminal output value.
        """
        if output.shape!=residual.shape or output.shape!=(*x.shape[:-1],weight.shape[0]):
            raise ValueError('terminal replay output/residual shape mismatch')
        previous=self.ops.fp8
        if self.fp8_filter=='auto':
            n,k=weight.shape
            self.ops.fp8=previous and n%16==0 and k%16==0 and not (k<=4096 and n<=1024) and 'lm_head' not in (weight.name or '') and 'embed_tokens' not in (weight.name or '')
        try:saved=self.ops.linear_saved(x.data,weight.data)
        finally:self.ops.fp8=previous
        def backward(g):
            dx,dw=self._linear_vjp(x,weight,saved,g)
            return g,dx,dw
        dtype='preserve' if getattr(self.backend,'compact_gradients',False) else 'fp32'
        return self._node(output,(residual,x,weight),backward,dtype)

    def linear_ce(self, x, weight, labels, chunk_size=256):
        """Chunked projection + fused CE; recompute logits in VJP, never retain T×V."""
        shape=x.shape;flat=x.reshape(-1,shape[-1]);target=labels.reshape(-1)
        rows,vocab=flat.shape[0],weight.shape[0]
        if target.size!=rows or chunk_size<1:raise ValueError('invalid linear CE shape/chunk')
        valid=self.binary(target,0.,'ge')*self.binary(self.full((1,),vocab),target,'gt')
        denom=self.binary(valid.sum(),1.,'max').detach()
        def chunk(begin,end):
            xx=flat[begin:end];yy=target[begin:end]
            logits,saved=self.ops.linear(xx.data,weight.data)
            losses=self.backend.empty((end-begin,),'fp32');counts=self.backend.empty(losses.shape,'fp32')
            dy=self.backend.empty((end-begin,vocab),'fp32')
            self.backend.launch(ew.cross_entropy,(end-begin,),(logits,yy.data,losses,counts,dy),VOCAB=vocab,BLOCK=triton.next_power_of_2(vocab))
            return losses,dy,saved
        loss=self.full((1,),0.)
        with self.no_grad():
            for begin in range(0,rows,chunk_size):
                mark=self.backend.mark()
                losses,_,_=chunk(begin,min(begin+chunk_size,rows))
                self.copy(loss,self.wrap(losses).sum()/denom,add=True)
                self.backend.release_since(mark)
        def backward(g):
            dx=self.full(flat.shape,0.);dw=self.full(weight.shape,0.)
            for begin in range(0,rows,chunk_size):
                end=min(begin+chunk_size,rows);mark=self.backend.mark()
                _,dy,saved=chunk(begin,end)
                dy=(self.wrap(dy)*self.wrap(g)/denom).cast(x.dtype)
                gx,gw=self.ops.linear_backward(saved,dy.data)
                self.backend.finish_backward()
                self.backend.launch(tk.strided_copy,(triton.cdiv((end-begin)*shape[-1],256),),(gx,dx.data),
                    SHAPE=(end-begin,shape[-1]),STRIDES=(shape[-1],1),OFFSET=begin*shape[-1],N=(end-begin)*shape[-1],REVERSE=True)
                self.copy(dw,self.wrap(gw),add=True)
                self.backend.release_since(mark)
            return dx.data.view(*shape),dw.data
        return self._node(loss.data,(x,weight),backward)

    def swiglu(self,gate,up,round_gate=False,up_grad_dtype=None):
        if round_gate:
            from .fused import rounded_swiglu,rounded_swiglu_amax
            if gate.shape!=up.shape or gate.dtype!=up.dtype:raise ValueError('SwiGLU inputs differ')
            out=self.backend.empty(gate.shape,gate.dtype)
            fused=getattr(self.backend,'fused_swiglu_amax',False) and self.ops.fp8
            if fused:
                partial=self.backend.empty((triton.cdiv(gate.size,1024),),'fp32')
                self.backend.launch(rounded_swiglu_amax,(partial.numel,),
                    (gate.data,up.data,out,out,out,out,partial,partial),N=gate.size,BACKWARD=False)
                out.amax_partial=partial
            else:self.backend.launch(rounded_swiglu,(triton.cdiv(gate.size,256),),
                    (gate.data,up.data,out,out,out,out),N=gate.size,BACKWARD=False)
            def backward(g):
                dg=self.backend.empty(gate.shape,gate.dtype)
                du=self.backend.empty(up.shape,up_grad_dtype or 'fp32')
                if fused:
                    mg,mu=[self.backend.empty((triton.cdiv(gate.size,1024),),'fp32') for _ in range(2)]
                    self.backend.launch(rounded_swiglu_amax,(mg.numel,),
                        (gate.data,up.data,out,g,dg,du,mg,mu),N=gate.size,BACKWARD=True)
                    dg.amax_partial,du.amax_partial=mg,mu
                else:self.backend.launch(rounded_swiglu,(triton.cdiv(gate.size,256),),
                        (gate.data,up.data,out,g,dg,du),N=gate.size,BACKWARD=True)
                return dg,du
            return self._node(out,(gate,up),backward,'preserve')
        out,saved=self.ops.swiglu(gate.data,up.data)
        return self._node(out,(gate,up),lambda g:self.ops.swiglu_backward(saved,self.ops.cast(g,gate.dtype)),gate.dtype)

    def norm(self, x, weight, eps=1e-6, out_dtype=None):
        out, saved = self.ops.norm(x.data, weight.data, eps, out_dtype=out_dtype)
        return self._norm_node(out,x,weight,saved)

    def _norm_node(self,out,x,weight,saved):
        def backward(g):
            if isinstance(g,GradientSum):return self.ops.norm_backward_sum(saved,g.parts)
            return self.ops.norm_backward(saved,self.ops.cast(g,saved[0].dtype))
        value=self._node(out,(x,weight),backward,saved[0].dtype)
        value.accept_gradient_sum=bool(getattr(self.backend,'fused_norm_backward',False))
        value.residual_norm_saved=saved
        return value

    def residual_norm(self,x,a,weight,eps=1e-6,out_dtype=None):
        from .fused import residual_norm
        if x.shape!=a.shape or weight.shape!=(x.shape[-1],):raise ValueError('residual norm shape mismatch')
        n=x.shape[-1];rows=x.size//n;dtype=x.dtype if x.dtype==a.dtype else 'fp32'
        z=self.backend.empty(x.shape,dtype);y=self.backend.empty(x.shape,out_dtype or dtype)
        inv=self.backend.empty((rows,),'fp32');maximum=self.backend.empty((rows,),'fp32')
        tile=getattr(self.backend,'norm_forward_rows',1)
        if tile>1:
            from .fused import norm_forward_amax
            self.backend.launch(norm_forward_amax,(triton.cdiv(rows,tile),),(x.data,a.data,weight.data,z,y,inv,maximum),
                M=rows,N=n,EPS=eps,BLOCK=triton.next_power_of_2(n),ROWS=tile,ADD_RESIDUAL=True)
        else:self.backend.launch(residual_norm,(rows,),(x.data,a.data,weight.data,z,y,inv,maximum),
                                 N=n,EPS=eps,BLOCK=triton.next_power_of_2(n))
        y.amax_partial=maximum
        grad_dtype='preserve' if getattr(self.backend,'compact_gradients',False) else 'fp32'
        residual=self._node(z,(x,a),lambda g:(g,g),grad_dtype)
        saved=(z,weight.data,inv)
        normalized=self._norm_node(y,residual,weight,saved)
        return residual,normalized

    def rotary(self, x, cos, sin, start=0):
        out = self.ops.rotary(x.data, cos.data, sin.data, start=start)
        return self._node(out, (x,), lambda g: (self.ops.rotary(g, cos.data, sin.data, start=start, backward=True),))

    def norm_rotary(self,x,weight,cos,sin,eps=1.e-6,start=0):
        out,saved=self.ops.norm_rotary(x.data,weight.data,cos.data,sin.data,eps,start)
        return self._node(out,(x,weight),lambda g:self.ops.norm_rotary_backward(saved,g),'preserve')

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

    def attention(self, q, k, v, mask=None, dropout=0., training=True, flash=True, start=0, packed=False):
        """Packed masks are canonical start offsets from packing.metadata."""
        if flash and start == 0 and q.shape[1] == k.shape[1] and q.shape[-1] <= 256:
            probability = dropout if training else 0.
            counter = self.counter
            if probability: self.counter += q.shape[0] * q.shape[2] * q.shape[1] ** 2
            cache = self.replay_cache
            selected = cache is not None and cache.attention and probability == 0.
            cached = cache.values['attention'] if selected and cache.replay else None
            out, saved = self.ops.attention(q.data, k.data, v.data, None if mask is None else mask.data,
                                            dropout=probability, seed=self.seed, counter=counter, cached=cached, packed=packed)
            if selected and not cache.replay:
                cache.values['attention'] = (saved[4], saved[5])
            return self._node(out, (q, k, v), lambda g: self.ops.attention_backward(saved, self.ops.cast(g, q.dtype)),q.dtype)
        if packed:
            raise ValueError('packed attention requires fused attention, head_dim <= 256 and no decode cache')
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

    def ce(self, logits, labels, shift=False):
        rows,vocab=labels.size,logits.shape[-1]
        if logits.size!=rows*vocab:raise ValueError('CE labels/logits shape mismatch')
        seq=1
        if shift:
            if len(logits.shape)!=3 or labels.shape!=logits.shape[:2] or labels.shape[1]<2:
                raise ValueError('shifted CE requires [batch,seq>=2,vocab] logits and [batch,seq] labels')
            seq=labels.shape[1];rows=labels.shape[0]*(seq-1)
        losses=self.backend.empty((rows,),'fp32');counts=self.backend.empty((rows,),'fp32')
        options=dict(VOCAB=vocab,BLOCK=triton.next_power_of_2(vocab),SHIFT=shift,SEQ=seq)
        self.backend.launch(tk.cross_entropy_recompute,(rows,),
            (logits.data,labels.data,losses,counts,logits.data,losses,counts),BACKWARD=False,**options)
        denom=self.wrap(counts).sum()
        loss=self.wrap(losses).sum()/self.binary(denom,1.,'max')
        def backward(g):
            dx=self.backend.empty(logits.shape,'fp32')
            self.backend.launch(tk.cross_entropy_recompute,(rows,),
                (logits.data,labels.data,losses,counts,dx,g,denom.data),BACKWARD=True,**options)
            return (dx,)
        return self._node(loss.data,(logits,),backward)

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

    def vjp(self, output, gradient=None, inputs=None, leaf_callback=None):
        """Reverse traversal, accumulating tied/shared parameters exactly once."""
        mark=self.backend.mark()
        recycle=(getattr(self.backend,'recycle_vjp',False)
                 and self.backend.weight_stream==self.backend.main_stream)
        requested=set(inputs or ());finished={}
        ordered, seen, stack = [], set(), [(output, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded: ordered.append(node); continue
            if node in seen: continue
            seen.add(node); stack.append((node, True))
            if node.node:
                stack.extend((p, False) for p in node.node[0] if p.requires_grad)
        # A DFS leaf can appear very late in reverse order despite its last
        # actual consumer being finished. Drain leaf gradients at last use,
        # rather than retaining every layer's expert dW until traversal ends.
        early_consume=leaf_callback is not None and self.backend.weight_stream==self.backend.main_stream
        users={}
        if early_consume:
            for value in ordered:
                if value.node:
                    for parent in value.node[0]:
                        if parent.requires_grad and parent.node is None:
                            users[parent]=users.get(parent,0)+1
        residual_users={}
        if getattr(self.backend,'fused_residual_backward',False):
            for value in ordered:
                if value.node:
                    for parent in value.node[0]:
                        if parent.requires_grad:residual_users[parent]=residual_users.get(parent,0)+1
        with self.no_grad():
            gradients = {output: [gradient if gradient is not None else self.full(output.shape, 1.).data]}
            def materialize(value):
                parts=[part for item in gradients.get(value,[]) for part in
                       (item.parts if isinstance(item,GradientSum) else (item,))];result=None
                target='fp32' if value.node is None or value in requested else value.gradient_dtype
                compact=getattr(self.backend,'compact_gradients',False)
                cast=self.ops.cast_for_quantize if getattr(value,'quantized_gradient',False) else self.ops.cast
                if compact and value.node is None:target='preserve'
                if compact and 2<=len(parts)<=4:
                    ready=[]
                    for part in parts:
                        if isinstance(part,PendingGradient):
                            self.backend.wait_event(part.event);part=part.data
                        ready.append(part)
                    if value not in requested and getattr(value,'accept_gradient_sum',False):
                        return GradientSum(ready)
                    result=self.backend.empty(value.shape,'fp32')
                    arguments=(ready+[ready[0]]*4)[:4]
                    self.backend.launch(tk.gradient_sum,(triton.cdiv(value.size,1024),),
                        (*arguments,result),N=value.size,COUNT=len(ready))
                    if target!='preserve':result=cast(result,target)
                    gradients[value]=[result]
                    return result
                for part in parts:
                    if isinstance(part,PendingGradient):
                        self.backend.wait_event(part.event);part=part.data
                    # Keep accumulation in FP32. A single incoming gradient can
                    # go directly to a kernel's declared input precision, without
                    # a BF16 -> FP32 -> BF16 round trip. Metadata views preserve it.
                    dtype='fp32' if len(parts)>1 else (part.dtype if target=='preserve' else target)
                    part=cast(part,dtype) if len(parts)==1 else self.ops.cast(part,dtype)
                    result=part if result is None else (self.wrap(result)+self.wrap(part)).data
                if result is not None and target!='preserve':result=cast(result,target)
                gradients[value]=[] if result is None else [result]
                return result
            for value in reversed(ordered):
                if value not in gradients:continue
                g=materialize(value)
                if value in requested:finished[value]=g
                if value.node is None:
                    consumed=leaf_callback is not None and leaf_callback(value,g)
                    if not consumed and inputs is None:finished[value]=g
                else:
                    parents, backward = value.node
                    saved=getattr(value,'residual_norm_saved',None)
                    residual=parents[0] if saved is not None else None
                    prior=gradients.get(residual,[])
                    parts=g.parts if isinstance(g,GradientSum) else (g,)
                    # Consume the residual only at its last incoming edge. The
                    # fused result is the complete input gradient, including
                    # requested checkpoint leaves; it preserves FP32 add order.
                    fuse=(saved is not None and saved[0].dtype=='fp32'
                          and residual_users.get(residual)==1 and len(prior)==1
                          and not isinstance(prior[0],(PendingGradient,GradientSum)) and 1<=len(parts)<=4)
                    if fuse:
                        local=self.ops.norm_backward_sum(saved,parts,residual=prior[0])
                        gradients.pop(residual)
                    else:local = backward(g)
                    if len(local) != len(parents): raise RuntimeError('VJP returned wrong gradient count')
                    for parent, grad in zip(parents, local):
                        if parent in residual_users:residual_users[parent]-=1
                        if parent.requires_grad and grad is not None:
                            gradients.setdefault(parent,[]).append(grad)
                    if early_consume:
                        for parent in parents:
                            if parent in users:
                                users[parent]-=1
                                if users[parent]==0 and parent not in requested and parent in gradients:
                                    leaf_grad=materialize(parent)
                                    if leaf_callback(parent,leaf_grad):gradients.pop(parent,None)
                gradients.pop(value,None)
                if recycle:
                    pending=[part.data if isinstance(part,PendingGradient) else part
                             for parts in gradients.values() for item in parts
                             for part in (item.parts if isinstance(item,GradientSum) else (item,))]
                    keep=(*pending,*finished.values())
                    statistics=tuple(t.amax_partial for t in keep if t is not None and hasattr(t,'amax_partial'))
                    self.backend.release_since(mark,keep=(*keep,*statistics))
            # Parameter gradients are consumed only after all dX work has been queued.
            if inputs is not None:return tuple(finished.get(x) for x in inputs)
            return finished

    def checkpoint(self, fn, x, parameters, selective=None, replay=None):
        """Discard forward temporaries; reproduce RNG and regenerate a local VJP."""
        if not self.recording: return fn(x)
        mark, seed, counter = self.backend.mark(), self.seed, self.counter
        previous = self.replay_cache
        try:
            self.replay_cache = selective
            with self.no_grad(): out = fn(x)
        finally:
            self.replay_cache = previous
        retained = selective.tensors() if selective is not None else ()
        self.backend.release_since(mark, keep=(out.data, *retained))
        parents = (x, *parameters)
        def backward(g):
            if selective is not None and getattr(selective,'consumed',False):
                raise RuntimeError('optimizer backward already consumed this checkpoint cache')
            mark = self.backend.mark()
            prior = self.recording, self.seed, self.counter, self.replay_cache
            self.recording, self.seed, self.counter = True, seed, counter
            self.replay_cache = selective
            previous_layouts=self.ops.save_activation_layouts
            # Only replay operands need both orientations. Original checkpoint
            # forwards discard their VJP inputs, so a transpose there is waste.
            self.ops.save_activation_layouts=getattr(self.backend,'fused_activation_layouts',False)
            if selective is not None: selective.replay = True
            try:
                leaf = self.wrap(x.data, True)
                regenerated = fn(leaf) if replay is None else replay(leaf,out.data)
                result = self.vjp(regenerated, g, (leaf, *parameters))
                self.backend.release_since(mark, keep=result)
                if selective is not None and getattr(self,'consume_checkpoint_cache',False):
                    self.backend.release_owned(selective.tensors())
                    selective.values.clear();selective.consumed=True
                return result
            finally:
                self.recording, self.seed, self.counter, self.replay_cache = prior
                self.ops.save_activation_layouts=previous_layouts
        grad_dtype='preserve' if getattr(self.backend,'compact_gradients',False) else 'fp32'
        return self._node(out.data, parents, backward,grad_dtype)

    def copy(self, target, source, scale=1., add=False):
        if target.shape != source.shape: raise ValueError('copy shape mismatch')
        if hasattr(target.data,'quant_scale_valid'):target.data.quant_scale_valid=False
        self.backend.launch(tk.scaled_copy, (triton.cdiv(target.size, 256),),
            (source.data, target.data, Scalar(scale)), N=target.size, ADD=add)
