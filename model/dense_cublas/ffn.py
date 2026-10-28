"""Tensorwise FP8 FFN: reduce statistics, then write GEMM layouts directly."""
import triton
import triton.language as tl
from model.dense_triton import linear as lin
from . import tensor_kernels as tk


@triton.jit
def values(G,U,DY,offset,mask,BACKWARD:tl.constexpr):
    g=tl.load(G+offset,mask,0).to(tl.float32)
    u=tl.load(U+offset,mask,0).to(tl.float32)
    denominator=1.+tl.exp(-g)
    a=(g/denominator).to(G.dtype.element_ty).to(tl.float32)
    y=(a*u).to(G.dtype.element_ty).to(tl.float32)
    dg=tl.full(offset.shape,0.,tl.float32);du=dg
    if BACKWARD:
        dy=tl.load(DY+offset,mask,0).to(tl.float32)
        sig=1./denominator
        derivative=sig*(1.+g*(1.-sig))
        dg=((dy*u)*derivative).to(G.dtype.element_ty).to(tl.float32)
        du=(dy*a).to(U.dtype.element_ty).to(tl.float32)
    return y,dg,du


@triton.jit
def statistics(G,U,DY,MY,MG,MU,N:tl.constexpr,BACKWARD:tl.constexpr,B:tl.constexpr=1024):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N
    y,dg,du=values(G,U,DY,i,mask,BACKWARD)
    tl.store(MY+tl.program_id(0),tl.max(tl.where(mask,tl.abs(y),0.),0))
    if BACKWARD:
        tl.store(MG+tl.program_id(0),tl.max(tl.where(mask,tl.abs(dg),0.),0))
        tl.store(MU+tl.program_id(0),tl.max(tl.where(mask,tl.abs(du),0.),0))


@triton.jit
def quantized_values(G,U,DY,SY,SG,SU,Y,QG,QGT,QU,QUT,
                     ROWS:tl.constexpr,COLS:tl.constexpr,BACKWARD:tl.constexpr,TILE:tl.constexpr=32,
                     FLAT_FORWARD:tl.constexpr=False, COL_TILE:tl.constexpr=0, COL_MAJOR_GRID:tl.constexpr=False):
    if FLAT_FORWARD and not BACKWARD:
        pos=tl.program_id(0)*1024+tl.arange(0,1024);valid=pos<ROWS*COLS
    else:
        width:tl.constexpr=COL_TILE if COL_TILE else TILE
        row=tl.program_id(1 if COL_MAJOR_GRID else 0)*TILE+tl.arange(0,TILE)
        col=tl.program_id(0 if COL_MAJOR_GRID else 1)*width+tl.arange(0,width)
        pos=row[:,None]*COLS+col[None,:];valid=(row[:,None]<ROWS)&(col[None,:]<COLS)
    y,dg,du=values(G,U,DY,pos,valid,BACKWARD)
    yq=tl.minimum(tl.maximum(y*tl.load(SY),-448.),448.).to(Y.dtype.element_ty,fp_downcast_rounding='rtne')
    if BACKWARD:
        tl.store(Y+col[None,:]*ROWS+row[:,None],yq,valid)
        gq=tl.minimum(tl.maximum(dg*tl.load(SG),-57344.),57344.).to(QG.dtype.element_ty,fp_downcast_rounding='rtne')
        uq=tl.minimum(tl.maximum(du*tl.load(SU),-57344.),57344.).to(QU.dtype.element_ty,fp_downcast_rounding='rtne')
        tl.store(QG+pos,gq,valid);tl.store(QGT+col[None,:]*ROWS+row[:,None],gq,valid)
        tl.store(QU+pos,uq,valid);tl.store(QUT+col[None,:]*ROWS+row[:,None],uq,valid)
    else:tl.store(Y+pos,yq,valid)


def _scale(ops,partial,maximum):
    if getattr(ops.backend,'fused_scale_inverse',False):
        from .quantization import make_scale
        return make_scale(ops,partial,maximum)
    while partial.numel>1024:
        reduced=ops.empty((triton.cdiv(partial.numel,1024),),'fp32')
        ops.backend.launch(lin.amax_partial,(reduced.numel,),(partial,reduced),SIZE=partial.numel,BLOCK=1024)
        partial=reduced
    scale=ops.empty((1,),'fp32')
    ops.backend.launch(lin.scale_from_amax,(1,),(partial,scale),COUNT=partial.numel,
                       FP8_MAX=maximum,BLOCK=triton.next_power_of_2(partial.numel))
    return scale


def quantize_swiglu(ops,gate,up,dy=None):
    rows,cols=gate.shape;backward=dy is not None
    block=getattr(ops.backend,'ffn_statistics_block',1024)
    partials=[ops.empty((triton.cdiv(gate.numel,block),),'fp32') for _ in range(3 if backward else 1)]
    py,pg,pu=(partials+[partials[0]]*3)[:3]
    ops.backend.launch(statistics,(py.numel,),(gate,up,dy or gate,py,pg,pu),N=gate.numel,BACKWARD=backward,B=block)
    sy=_scale(ops,py,448.)
    y=ops.empty((cols,rows) if backward else (rows,cols),'fp8e4nv')
    if backward:
        sg,su=(_scale(ops,p,57344.) for p in (pg,pu))
        qg,qu=[ops.empty((rows,cols),'fp8e5') for _ in range(2)]
        qgt,qut=[ops.empty((cols,rows),'fp8e5') for _ in range(2)]
        qg.transpose_data,qu.transpose_data=qgt,qut
    else:sg=su=sy;qg=qgt=qu=qut=y
    tile=getattr(ops.backend,'ffn_quant_tile',32)
    flat=getattr(ops.backend,'ffn_flat_forward',False)
    width=getattr(ops.backend,'ffn_quant_col_tile',tile)
    col_major=getattr(ops.backend,'ffn_col_major_grid',False)
    grid=(triton.cdiv(rows,tile),triton.cdiv(cols,width))
    if col_major:grid=grid[::-1]
    if flat and not backward:grid=(triton.cdiv(rows*cols,1024),)
    ops.backend.launch(quantized_values,grid,
        (gate,up,dy or gate,sy,sg,su,y,qg,qgt,qu,qut),ROWS=rows,COLS=cols,BACKWARD=backward,
        TILE=tile,FLAT_FORWARD=flat,COL_TILE=width,COL_MAJOR_GRID=col_major)
    return (y,sy,qg,sg,qu,su) if backward else (y,sy)


def eligible(model,x,prefix):
    e=model.e
    if (not getattr(e.backend,'fused_ffn',False) or not e.ops.fp8 or x.dtype!='bf16'
        or model.adapters or model.fuse_projections or model.config['hidden_act']!='silu'
        or e.backend.weight_stream!=e.backend.main_stream):return False
    for name in ('gate_proj.weight','up_proj.weight','down_proj.weight'):
        w=model.params[prefix+name];n,k=w.shape
        if w.dtype!='bf16' or n%16 or k%16:return False
        if e.fp8_filter=='auto' and k<=4096 and n<=1024:return False
    return True


def apply(model,x,prefix,*,residual=None,output=None):
    e,ops=model.e,model.e.ops
    wg,wu,wd=[model.params[prefix+n+'_proj.weight'] for n in ('gate','up','down')]
    cache=e.replay_cache
    def project(w):
        selected=cache is not None and w.name in cache.linear_names
        if selected and cache.replay:return cache.values[w.name],ops.linear_saved(x.data,w.data)
        y,saved=ops.linear(x.data,w.data)
        if selected:cache.values[w.name]=y
        return y,saved
    with ops.reuse_scope():
        gate,gs=project(wg);up,us=project(wu)
    rows=gate.numel//gate.shape[-1];gate=gate.view(rows,gate.shape[-1]);up=up.view(rows,up.shape[-1])
    if output is None:
        qa,sa=quantize_swiglu(ops,gate,up)
        qw,sw=ops.quantize(wd.data)
        output=ops.mm(qa,qw,transpose_b=True,scales=(sa,sw),out_dtype=x.dtype).view(*x.shape)
    def backward(g):
        with ops.reuse_scope():
            dy=ops.cast_for_quantize(g,x.dtype).view(rows,x.shape[-1]);qd,sd=ops.quantize(dy,'fp8e5')
            qw,sw=ops.quantize(wd.data)
            prior=ops._role
            try:
                ops._role='dX'
                dh=ops.mm(qd,qw,scales=(sd,sw),out_dtype=x.dtype)
                ya,sy,qg,sg,qu,su=quantize_swiglu(ops,gate,up,dh)
                ops._role='dW'
                dd=ops.mm(qd,ya,transpose_a=True,transpose_b=True,scales=(sd,sy),out_dtype=wd.dtype)
                dxg,dg=ops.linear_backward_quantized(gs,qg,sg)
                dxu,du=ops.linear_backward_quantized(us,qu,su)
                if getattr(e.backend,'fused_norm_backward',False):
                    from .autograd import GradientSum
                    dx=GradientSum((dxg,dxu))
                else:
                    dx=ops.empty(x.shape,'fp32')
                    e.backend.launch(tk.gradient_sum,(triton.cdiv(x.size,1024),),(dxg,dxu,dxg,dxg,dx),N=x.size,COUNT=2)
            finally:ops._role=prior
        result=(dx,dg,du,dd)
        return result if residual is None else (*result,g)
    parents=(x,wg,wu,wd) if residual is None else (x,wg,wu,wd,residual)
    return e._node(output,parents,backward,x.dtype if residual is None else 'preserve')
