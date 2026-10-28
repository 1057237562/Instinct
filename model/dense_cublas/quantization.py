"""Native FP8 scale pairs: preserve rounding while avoiding reciprocal launches."""
import triton
import triton.language as tl
from model.dense_triton import linear as lin
from model.dense_triton.runtime import Tensor


@triton.jit
def scale_pair(PARTIAL,PAIR,COUNT:tl.constexpr,FP8_MAX:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK)
    maximum=tl.max(tl.load(PARTIAL+i,i<COUNT,0),0)
    denominator=tl.maximum(maximum.to(tl.float64),1.0e-12)
    scale=(FP8_MAX/denominator).to(tl.float32)
    # The reciprocal must use the rounded FP32 scale, just as inverse_scale.
    tl.store(PAIR,scale)
    tl.store(PAIR+4,tl.div_rn(1.0,scale))


def make_scale(ops,partial,maximum,pair=None):
    while partial.numel>1024:
        reduced=ops.empty((triton.cdiv(partial.numel,1024),),'fp32')
        ops.backend.launch(lin.amax_partial,(reduced.numel,),(partial,reduced),SIZE=partial.numel,BLOCK=1024)
        partial=reduced
    # Keep both device addresses 16-byte aligned, including CUDA 13.1 cuBLAS
    # builds that incorrectly impose matrix alignment on scalar FP8 scales.
    if pair is None:pair=ops.empty((8,),'fp32')
    ops.backend.launch(scale_pair,(1,),(partial,pair),COUNT=partial.numel,
                       FP8_MAX=maximum,BLOCK=triton.next_power_of_2(partial.numel))
    # Both scalars share one allocation. Retaining scale retains the inverse;
    # no auxiliary pointer can be recycled while a saved scale is still live.
    pointer=pair.storage
    first=pointer if isinstance(pointer,int) else pointer.reshape(-1)[:1]
    second=pointer+16 if isinstance(pointer,int) else pointer.reshape(-1)[4:5]
    scale=Tensor((1,),'fp32',first,pair.owner)
    scale._native_inverse=Tensor((1,),'fp32',second,pair.owner)
    return scale


def reserve_weight_scale(ops,weight):
    """Small persistent scalar storage owned alongside a model parameter."""
    weight.quant_scale_storage=ops.empty((8,),'fp32')
    weight.quant_scale_valid=False
    return weight.quant_scale_storage


def quantize(ops,x,dtype,dual_layout):
    if dtype not in ('fp8e4nv','fp8e5') or x.dtype not in ('fp32','bf16','fp16'):
        raise ValueError('expected high-precision input and E4M3/E5M2 output')
    count=triton.cdiv(x.numel,1024)
    if count<=0:raise ValueError('tensorwise amax requires a nonempty tensor')
    maximum=448. if dtype=='fp8e4nv' else 57344.
    # Captured commands must refresh their own scales on every replay. Eager
    # parameter scalars are invalidated by optimizer updates and Engine.copy.
    cached=(dtype=='fp8e4nv' and getattr(ops.backend,'cache_weight_scales',False)
            and getattr(ops.backend,'_commands',None) is None and hasattr(x,'quant_scale_storage'))
    if cached and x.quant_scale_valid:scale=x.quant_scale_value
    else:
        partial=getattr(x,'amax_partial',None)
        if partial is None:
            partial=ops.empty((count,),'fp32')
            ops.backend.launch(lin.amax_partial,(count,),(x,partial),SIZE=x.numel,BLOCK=1024)
        scale=make_scale(ops,partial,maximum,x.quant_scale_storage if cached else None)
        if cached:x.quant_scale_value=scale;x.quant_scale_valid=True
    out=ops.empty(x.shape,dtype)
    if dual_layout:
        rows,cols=x.shape;transposed=ops.empty((cols,rows),dtype)
        ops.backend.launch(lin.quantize_dual,(triton.cdiv(rows,32),triton.cdiv(cols,32)),
            (x,scale,out,transposed),ROWS=rows,COLS=cols,FP8_MAX=maximum)
        out.transpose_data=transposed
    else:
        ops.backend.launch(lin.quantize,(triton.cdiv(x.numel,1024),),(x,scale,out),
            SIZE=x.numel,FP8_MAX=maximum,BLOCK=1024)
    return out,scale
