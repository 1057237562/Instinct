"""Drop-in Dense operations: cuBLASLt Linear plus existing Triton kernels."""
from contextlib import contextmanager
import triton

from model.dense_triton.ops import DenseOps, DenseBlock
from .layout import plan_layout
from . import kernels
from . import fused
from model.dense_triton import elementwise as ew


class CublasDenseOps(DenseOps):
    def norm(self,x,weight,eps=1e-6,out_dtype=None):
        if not (getattr(self.backend,'fused_input_norm_amax',False) and out_dtype in ('bf16','fp16')):
            return super().norm(x,weight,eps,out_dtype)
        n=x.shape[-1];rows=x.numel//n
        if not 0<n<=8192 or weight.shape!=(n,) or eps<=0:raise ValueError('invalid RMSNorm shape or epsilon')
        y=self.empty(x.shape,out_dtype);r=self.empty((rows,),'fp32');maximum=self.empty((rows,),'fp32')
        tile=getattr(self.backend,'norm_forward_rows',1)
        self.backend.launch(fused.norm_forward_amax,(triton.cdiv(rows,tile),),(x,x,weight,x,y,r,maximum),
            M=rows,N=n,EPS=eps,BLOCK=triton.next_power_of_2(n),ROWS=tile)
        y.amax_partial=maximum
        return y,(x,weight,r)

    def norm_backward_sum(self,saved,parts,residual=None):
        x,w,r=saved
        if not 1<=len(parts)<=4 or any(p.shape!=x.shape for p in parts):
            raise ValueError('RMSNorm gradient parts must match the input')
        n=x.shape[-1];rows=x.numel//n;groups=min(rows,getattr(self.backend,'norm_groups',None) or 128)
        dx,dw=self.empty(x.shape,x.dtype),self.empty(w.shape,w.dtype)
        partial=self.empty((groups,n),'fp32');args=(list(parts)+[parts[0]]*4)[:4]
        if residual is not None and (x.dtype!='fp32' or residual.shape!=x.shape):
            raise ValueError('fused residual backward requires matching FP32 norm input')
        self.backend.launch(fused.rms_backward_sum,(groups,),(x,w,r,*args,dx,partial,residual or x),
            M=rows,N=n,GROUPS=groups,COUNT=len(parts),BLOCK=triton.next_power_of_2(n),ADD_RESIDUAL=residual is not None)
        block_n=8 if groups>256 else 64
        self.backend.launch(ew._rms_reduce_dw,(triton.cdiv(n,block_n),),(partial,dw),
            N=n,GROUPS=groups,BLOCK_G=triton.next_power_of_2(groups),BLOCK_N=block_n)
        return dx,dw

    def cast_for_quantize(self,x,dtype):
        if x.dtype==dtype or not getattr(self.backend,'fused_cast_amax',False):return self.cast(x,dtype)
        out=self.empty(x.shape,dtype);partial=self.empty((triton.cdiv(x.numel,1024),),'fp32')
        self.backend.launch(fused.cast_amax,(partial.numel,),(x,out,partial),N=x.numel)
        out.amax_partial=partial
        return out

    def attention(self,q,k,v,mask=None,dropout=0.,seed=0,counter=0,cached=None,packed=False,scale=None):
        native=getattr(self.backend,'native_attention',None)
        if native is not None and scale is None:
            return native.attention(q,k,v,mask,dropout,seed,counter,cached,packed,super().attention)
        return super().attention(q,k,v,mask,dropout,seed,counter,cached,packed,scale)

    def attention_backward(self,saved,dy):
        native=getattr(self.backend,'native_attention',None)
        if native is not None:return native.backward(saved,dy,super().attention_backward)
        return super().attention_backward(saved,dy)

    def __init__(self, backend, fp8=True):
        super().__init__(backend, fp8)
        self._reuse = None
        self._role = 'forward'

    @contextmanager
    def reuse_scope(self, cache=None):
        prior, self._reuse = self._reuse, {} if cache is None else cache
        try:
            yield self._reuse
        finally:
            self._reuse = prior

    @staticmethod
    def _key(tensor):
        # Storage identity is stable for CUDA pointers and reference-test buffers.
        pointer = tensor.storage if isinstance(tensor.storage, int) else id(tensor.storage)
        return pointer, tensor.shape, tensor.dtype

    def _cached(self, key, create):
        if self._reuse is None:
            return create()
        if key not in self._reuse:
            result = create()
            event = None
            if self.backend.weight_stream != self.backend.main_stream:
                event = self.backend.create_event()
                self.backend.record_event(event)
            self._reuse[key] = (result, self.backend.stream, event)
        result, producer, event = self._reuse[key]
        if producer != self.backend.stream:
            self.backend.wait_event(event)
        return result

    def quantize(self, x, dtype='fp8e4nv'):
        # Q/K/V share one normalized activation cast; gate/up share another.
        def create():
            dual=bool(getattr(self.backend,'fused_quant_pack',False)
                      and dtype=='fp8e5' and len(x.shape)==2 and all(d%16==0 for d in x.shape))
            if getattr(self.backend,'fused_scale_inverse',False):
                from .quantization import quantize
                return quantize(self,x,dtype,dual)
            return super(CublasDenseOps,self).quantize(x,dtype,dual_layout=dual)
        return self._cached(('quant',self._key(x),dtype),create)

    def norm_rotary(self,x,weight,cos,sin,eps=1.e-6,start=0):
        if len(x.shape)!=4 or x.shape[-1]%2 or weight.shape!=(x.shape[-1],):
            raise ValueError('fused Q/K norm+RoPE requires [batch,seq,heads,even dim]')
        b,s,h,d=x.shape
        packed_tables = start == -1 and cos.shape == (b*s,d)
        if d>8192 or eps<=0 or cos.shape!=sin.shape or cos.shape[1:]!=(d,) or (not packed_tables and (start<0 or start+s>cos.shape[0])):
            raise ValueError('invalid normalization or RoPE tables')
        out=self.empty(x.shape,x.dtype);r=self.empty((b*s*h,),'fp32')
        options=dict(N=d,SEQ=s,HEADS=h,START=start,BLOCK=triton.next_power_of_2(d))
        rows=getattr(self.backend,'norm_rope_rows',1) if d<=128 else 1
        if rows>1:
            self.backend.launch(fused.norm_rotary_forward_rows,(triton.cdiv(b*s*h,rows),),
                (x,weight,cos,sin,out,r),M=b*s*h,EPS=eps,ROWS=rows,**options)
        else:self.backend.launch(fused.norm_rotary_forward,(b*s*h,),(x,weight,cos,sin,out,r),EPS=eps,**options)
        return out,(x,weight,cos,sin,r,options)

    def norm_rotary_backward(self,saved,g):
        x,w,cos,sin,r,options=saved;rows=x.numel//x.shape[-1]
        tile=getattr(self.backend,'norm_rope_rows',1) if x.shape[-1]<=128 else 1
        groups=min(triton.cdiv(rows,tile),getattr(self.backend,'norm_groups',None) or 128)
        dx=self.empty(x.shape,x.dtype);dw=self.empty(w.shape,w.dtype)
        partial=self.empty((groups,x.shape[-1]),'fp32')
        if tile>1:
            self.backend.launch(fused.norm_rotary_backward_rows,(groups,),(x,w,r,cos,sin,g,dx,partial),
                M=rows,GROUPS=groups,ROWS=tile,**options)
        else:self.backend.launch(fused.norm_rotary_backward,(groups,),(x,w,r,cos,sin,g,dx,partial),M=rows,GROUPS=groups,**options)
        block_n=8 if groups>256 else 64
        self.backend.launch(ew._rms_reduce_dw,(triton.cdiv(x.shape[-1],block_n),),(partial,dw),
                            N=x.shape[-1],GROUPS=groups,BLOCK_G=triton.next_power_of_2(groups),BLOCK_N=block_n)
        return dx,dw

    def linears(self,x,weights):
        # Q/K/V and gate/up keep separate GEMMs and parameter identities, while
        # sharing one cast launch. No concatenated master weights or output slices.
        if x.dtype not in ('fp32','bf16','fp16') or any(w.dtype not in ('fp32','bf16','fp16') for w in weights):
            raise ValueError('projections require high-precision activations and weights')
        if (self.fp8 or len(weights) not in (2,3) or len({w.dtype for w in weights})!=1
                or weights[0].dtype==x.dtype):
            return [self.linear(x,w) for w in weights]
        if any(len(w.shape)!=2 or w.shape[1]!=x.shape[-1] for w in weights):
            raise ValueError('projection input/weight shapes differ')
        converted=[self.empty(w.shape,x.dtype) for w in weights]
        inputs=list(weights);outputs=list(converted);sizes=[w.numel for w in weights]
        if len(weights)==2:inputs.append(weights[0]);outputs.append(converted[0]);sizes.append(0)
        self.backend.launch(fused.projection_cast,(sum(triton.cdiv(n,256) for n in sizes),),
                            (*inputs,*outputs),N0=sizes[0],N1=sizes[1],N2=sizes[2])
        matrix=x.view(x.numel//x.shape[-1],x.shape[-1]);result=[]
        for weight,qw in zip(weights,converted):
            y=self.mm(matrix,qw,transpose_b=True,out_dtype=x.dtype)
            saved=(x.shape,x.dtype,weight.dtype,matrix,qw,None)
            result.append((y.view(*x.shape[:-1],weight.shape[0]),saved))
        return result

    def linear_saved(self, x, weight):
        """Rebuild VJP operands without repeating a selectively saved GEMM."""
        matrix = x.view(x.numel // x.shape[-1], x.shape[-1])
        if self.fp8:
            qx, sx = self.quantize(matrix)
            qw, sw = self.quantize(weight)
            scales = sx, sw
        else:
            qx, qw, scales = matrix, self.cast(weight, x.dtype), None
        return x.shape, x.dtype, weight.dtype, qx, qw, scales

    def _inverse(self, scale):
        if scale.dtype != 'fp32' or scale.numel != 1:
            raise ValueError('quantization scale must be a device FP32 scalar')
        paired=getattr(scale,'_native_inverse',None)
        if paired is not None:return paired
        def create():
            out = self.empty((1,), 'fp32')
            self.backend.launch(kernels.inverse_scale, (1,), (scale, out))
            return out
        return self._cached(('inverse', self._key(scale)), create)

    def _pack(self, x, rows, cols, pr, pc, strides, needed):
        if not needed:
            return x
        transposed=getattr(x,'transpose_data',None)
        if (transposed is not None and strides==(1,x.shape[1])
                and (rows,cols)==(pr,pc)==transposed.shape):return transposed
        def create():
            out = self.empty((pr, pc), x.dtype)
            tile = getattr(self.backend, 'pack_tile', None)
            options = dict(ROWS=rows, COLS=cols, PAD_ROWS=pr, PAD_COLS=pc,
                           STRIDE_ROW=strides[0], STRIDE_COL=strides[1])
            if tile and strides[0] == 1:
                self.backend.launch(kernels.pack_matrix_tiled,
                    (triton.cdiv(pr, tile), triton.cdiv(pc, tile)), (x, out), TILE=tile, **options)
            else:
                self.backend.launch(kernels.pack_matrix, (triton.cdiv(pr * pc, 256),),
                                    (x, out), **options)
            return out
        return self._cached(('pack', self._key(x), rows, cols, pr, pc, strides), create)

    def mm(self, a, b, transpose_a=False, transpose_b=False, scales=None, out_dtype='fp32'):
        fp8 = a.dtype.startswith('fp8') or b.dtype.startswith('fp8')
        if fp8 and (scales is None or a.dtype not in ('fp8e4nv', 'fp8e5') or b.dtype not in ('fp8e4nv', 'fp8e5')):
            raise ValueError('FP8 GEMM requires two FP8 operands and multiplication scales')
        if not fp8 and (a.dtype != b.dtype or scales is not None):
            raise ValueError('unscaled cuBLASLt GEMM requires matching high-precision operands')
        if not fp8 and a.dtype not in ('fp32', 'bf16', 'fp16'):
            raise ValueError('GEMM inputs must be FP32/BF16/FP16 or scaled FP8')
        if a.dtype == b.dtype == 'fp8e5':
            raise ValueError('cuBLASLt does not support E5M2 x E5M2 for this path')
        if out_dtype not in ('fp32', 'bf16', 'fp16'):
            raise ValueError('GEMM output must be FP32/BF16/FP16')
        # BF16/FP16/FP32 support all transpose combinations directly. Describe
        # the existing row-major operands as column-major transposes; dX/dW
        # need no physical transpose, padding, or output crop. SM120 FP8 still
        # requires TN, so keep its explicitly packed/aligned path below.
        direct = not fp8 and self.backend.direct_layout
        layout = plan_layout(a.shape, b.shape, transpose_a, transpose_b,
                             alignment=1 if direct else 16)
        l = layout
        gemm_dtype = out_dtype if fp8 or out_dtype in (a.dtype, 'fp32') else 'fp32'
        if direct:
            out = self.empty((l.m, l.n), gemm_dtype)
            self.backend.matmul(b, a, out, None, None, l.n, l.m, l.k,
                                role=self._role, transpose_a=transpose_b, transpose_b=transpose_a)
            return self.cast(out, out_dtype)
        ap = self._pack(a, l.m, l.k, l.mp, l.kp, l.a_stride, l.a_pack)
        bp = self._pack(b, l.n, l.k, l.np, l.kp, l.b_transpose_stride, l.b_pack)
        # Regular cuBLASLt GEMMs support same-type or FP32 outputs. Independent
        # parameter/activation precision may request e.g. FP32 -> BF16 dW:
        # accumulate into FP32 and cast (fused into crop when padding is needed).
        cp = self.empty((l.mp, l.np), gemm_dtype)
        sa, sb = (self._inverse(s) for s in scales) if scales is not None else (None, None)
        # C^T = B^T @ A^T. Column-major TN returns the desired row-major C,
        # avoiding an output transpose (only crop when dimensions were padded).
        self.backend.matmul(bp, ap, cp, sb, sa, l.np, l.mp, l.kp, role=self._role)
        if (l.mp, l.np) == (l.m, l.n):
            return self.cast(cp, out_dtype)
        out = self.empty((l.m, l.n), out_dtype)
        self.backend.launch(kernels.unpack_matrix, (triton.cdiv(l.m * l.n, 256),),
                            (cp, out), ROWS=l.m, COLS=l.n, PAD_COLS=l.np)
        return out

    def linear_backward(self, saved, dy, need_dx=True, need_dw=True):
        shape, input_dtype, weight_dtype, qx, qw, scales = saved
        if dy.shape != (*shape[:-1], qw.shape[0]) or dy.dtype != input_dtype:
            raise ValueError('Linear gradient shape/dtype does not match its forward')
        dy = dy.view(dy.numel // dy.shape[-1], dy.shape[-1])
        qdy, sdy = self.quantize(dy, 'fp8e5') if scales is not None else (dy, None)
        return self.linear_backward_quantized(saved,qdy,sdy,need_dx,need_dw)

    def linear_backward_quantized(self,saved,qdy,sdy,need_dx=True,need_dw=True):
        shape,input_dtype,weight_dtype,qx,qw,scales=saved
        dx_scales = (sdy, scales[1]) if scales is not None else None
        dw_scales = (sdy, scales[0]) if scales is not None else None
        prior = self._role
        try:
            ready = None
            if self.backend.weight_stream != self.backend.main_stream and need_dw:
                ready = self.backend.create_event()
                self.backend.record_event(ready)  # dY is ready before dX, so both GEMMs can overlap.
            self._role = 'dX'
            dx = self.mm(qdy, qw, scales=dx_scales, out_dtype=input_dtype) if need_dx else None
            if not need_dw:
                return None if dx is None else dx.view(*shape), None
            self._role = 'dW'
            with self.backend.on_stream(self.backend.weight_stream):
                if self.backend.weight_stream != self.backend.main_stream:
                    self.backend.wait_event(ready)
                dw = self.mm(qdy, qx, transpose_a=True, scales=dw_scales, out_dtype=weight_dtype)
            return None if dx is None else dx.view(*shape), dw
        finally:
            self._role = prior


class ScheduledDenseBlock(DenseBlock):
    """Caches are scoped to one saved block, so later forwards cannot use stale casts."""
    def forward(self, x, cos, sin, mask=None):
        with self.ops.reuse_scope() as cache:
            y, saved = super().forward(x, cos, sin, mask)
            saved['_cublas_reuse'] = cache
            return y, saved

    def backward(self, saved, dy):
        with self.ops.reuse_scope(saved['_cublas_reuse']):
            return super().backward(saved, dy)
