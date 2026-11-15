"""Experimental 32-value FP8 blocks with cuBLASLt tiled UE8M0 scales.

Not a production precision default. Scales describe dequantization, unlike
the tensorwise path's multiplication scales. Each orientation needs its own
block quantization; transposing only the FP8 bytes is incorrect.
"""
import triton
import triton.language as tl
from .ffn import values


@triton.jit
def scale_offset(row,block,SF_COLUMNS:tl.constexpr):
    return (row//128)*SF_COLUMNS*128+(block//4)*512+(row%32)*16+((row%128)//32)*4+block%4


@triton.jit
def scale_exponent(maximum,FP8_MAX:tl.constexpr):
    # Both FP8 maxima are 1.75 * 2**SHIFT. Compare the mantissa directly
    # instead of dividing every block maximum and evaluating exp2. Preserve
    # the reference's FP32 division rounding at the subnormal threshold:
    # the first float above 1.75 * 2**(SHIFT-127) still rounds to 2**-127.
    SHIFT:tl.constexpr=8 if FP8_MAX==448. else 15
    bits=maximum.to(tl.uint32,bitcast=True)
    raw_exponent=(bits>>23)&255;mantissa=bits&0x7fffff
    exponent=raw_exponent.to(tl.int32)-SHIFT+(mantissa>0x600000).to(tl.int32)
    exponent=tl.where(bits<=((SHIFT<<23)|0x600001),0,exponent)
    exponent=tl.where(raw_exponent==255,255,exponent).to(tl.uint32)
    inverse_bits=tl.where(exponent<254,(254-exponent)<<23,
                          tl.where(exponent==254,0x400000,0x200000))
    inverse=inverse_bits.to(tl.float32,bitcast=True)
    return exponent.to(tl.uint8),inverse


@triton.jit
def mxfp8_quantize_rows(X,Q,S,ROWS:tl.constexpr,COLS:tl.constexpr,FP8_MAX:tl.constexpr,
                  TILE_ROWS:tl.constexpr=32,TILE_COLS:tl.constexpr=128):
    row=tl.program_id(0)*TILE_ROWS+tl.arange(0,TILE_ROWS)
    col=tl.program_id(1)*TILE_COLS+tl.arange(0,TILE_COLS)
    valid=(row[:,None]<ROWS)&(col[None,:]<COLS)
    x=tl.load(X+row[:,None]*COLS+col[None,:],valid,0).to(tl.float32)
    blocks=tl.reshape(x,(TILE_ROWS,TILE_COLS//32,32))
    maximum=tl.max(tl.abs(blocks),2)
    exponent,inverse=scale_exponent(maximum,FP8_MAX)
    scaled=tl.reshape(blocks*inverse[:,:,None],(TILE_ROWS,TILE_COLS))
    value=tl.minimum(tl.maximum(scaled,-FP8_MAX),FP8_MAX).to(Q.dtype.element_ty,fp_downcast_rounding='rtne')
    tl.store(Q+row[:,None]*COLS+col[None,:],value,valid)
    block=tl.program_id(1)*(TILE_COLS//32)+tl.arange(0,TILE_COLS//32)
    offset=scale_offset(row[:,None],block[None,:],tl.cdiv(COLS,128)*4)
    tl.store(S.to(tl.pointer_type(tl.uint8))+offset,exponent)


@triton.jit
def tile_quantize(x,MAXIMUM:tl.constexpr,TILE:tl.constexpr,TRANSPOSE:tl.constexpr,COL_TILE:tl.constexpr=0):
    COLS:tl.constexpr=COL_TILE if COL_TILE else TILE
    if TRANSPOSE:
        blocks=tl.reshape(x,(TILE//32,32,COLS))
        exponent,inverse=scale_exponent(tl.max(tl.abs(blocks),1),MAXIMUM)
        scaled=tl.reshape(blocks*inverse[:,None,:],(TILE,COLS))
    else:
        blocks=tl.reshape(x,(TILE,COLS//32,32))
        exponent,inverse=scale_exponent(tl.max(tl.abs(blocks),2),MAXIMUM)
        scaled=tl.reshape(blocks*inverse[:,:,None],(TILE,COLS))
    return tl.minimum(tl.maximum(scaled,-MAXIMUM),MAXIMUM),exponent


@triton.jit
def store_tile_scales(S,exponent,row,col,ROWS:tl.constexpr,COLS:tl.constexpr,TILE:tl.constexpr,TRANSPOSE:tl.constexpr,COL_TILE:tl.constexpr=0):
    WIDTH:tl.constexpr=COL_TILE if COL_TILE else TILE
    if TRANSPOSE:
        block=tl.program_id(0)*(TILE//32)+tl.arange(0,TILE//32)
        offsets=scale_offset(col[None,:],block[:,None],tl.cdiv(ROWS,128)*4)
    else:
        block=tl.program_id(1)*(WIDTH//32)+tl.arange(0,WIDTH//32)
        offsets=scale_offset(row[:,None],block[None,:],tl.cdiv(COLS,128)*4)
    tl.store(S.to(tl.pointer_type(tl.uint8))+offsets,exponent)


@triton.jit
def mxfp8_quantize_dual(X,Q,S,QT,ST,ROWS:tl.constexpr,COLS:tl.constexpr,FP8_MAX:tl.constexpr,
                       TILE:tl.constexpr=64,ROW_OUTPUT:tl.constexpr=True):
    row=tl.program_id(0)*TILE+tl.arange(0,TILE);col=tl.program_id(1)*TILE+tl.arange(0,TILE)
    valid=(row[:,None]<ROWS)&(col[None,:]<COLS)
    x=tl.load(X+row[:,None]*COLS+col[None,:],valid,0).to(tl.float32)
    if ROW_OUTPUT:
        value,exponent=tile_quantize(x,FP8_MAX,TILE,False)
        tl.store(Q+row[:,None]*COLS+col[None,:],value.to(Q.dtype.element_ty,fp_downcast_rounding='rtne'),valid)
        store_tile_scales(S,exponent,row,col,ROWS,COLS,TILE,False)
    value,exponent=tile_quantize(x,FP8_MAX,TILE,True)
    tl.store(QT+col[None,:]*ROWS+row[:,None],value.to(QT.dtype.element_ty,fp_downcast_rounding='rtne'),valid)
    store_tile_scales(ST,exponent,row,col,ROWS,COLS,TILE,True)


def new_scales(ops,shape):
    capacity=getattr(ops.backend,'capacity_shape',tuple)
    rows,cols=capacity(shape)
    count=triton.cdiv(rows,128)*triton.cdiv(cols,128)*512
    scales=ops.empty((count//2,),'u16');scales.mx32=True;scales.matrix_shape=shape
    return scales


def quantize_transposed(ops,x,dtype):
    rows,cols=x.shape;out=ops.empty((cols,rows),dtype);scales=new_scales(ops,out.shape)
    ops.backend.launch(mxfp8_quantize_dual,(triton.cdiv(rows,128)*2,triton.cdiv(cols,128)*2),
        (x,out,scales,out,scales),ROWS=rows,COLS=cols,FP8_MAX=448. if dtype=='fp8e4nv' else 57344.,ROW_OUTPUT=False)
    out.mx32_scale=scales
    return out


def quantize(ops,x,dtype='fp8e4nv',dual_layout=False):
    if len(x.shape)!=2 or dtype not in ('fp8e4nv','fp8e5') or x.dtype not in ('fp32','bf16','fp16'):
        raise ValueError('MXFP8 quantization requires a high-precision matrix and E4M3/E5M2 output')
    rows,cols=x.shape
    out=ops.empty(x.shape,dtype)
    # Raw UE8M0 bytes in an aligned existing storage type; no FP8 arithmetic
    # is performed on this scale allocation.
    scales=new_scales(ops,x.shape)
    if dual_layout:
        transposed=ops.empty((cols,rows),dtype);ts=new_scales(ops,transposed.shape)
        ops.backend.launch(mxfp8_quantize_dual,(triton.cdiv(rows,128)*2,triton.cdiv(cols,128)*2),
            (x,out,scales,transposed,ts),ROWS=rows,COLS=cols,FP8_MAX=448. if dtype=='fp8e4nv' else 57344.)
        out.transpose_data=transposed;transposed.mx32_scale=ts
    else:
        ops.backend.launch(mxfp8_quantize_rows,(triton.cdiv(rows,128)*4,triton.cdiv(cols,128)),
            (x,out,scales),ROWS=rows,COLS=cols,FP8_MAX=448. if dtype=='fp8e4nv' else 57344.)
    out.mx32_scale=scales;out.mx32_source=x
    return out,scales


@triton.jit
def cached_swiglu_values(G,U,GS,US,DY,row,col,valid,COLS:tl.constexpr,BACKWARD:tl.constexpr):
    pos=row[:,None]*COLS+col[None,:]
    sf=scale_offset(row[:,None],col[None,:]//32,tl.cdiv(COLS,128)*4)
    gs=tl.load(GS.to(tl.pointer_type(tl.uint8))+sf).to(tl.float32)
    us=tl.load(US.to(tl.pointer_type(tl.uint8))+sf).to(tl.float32)
    g=(tl.load(G+pos,valid,0.).to(tl.float32)*tl.exp2(gs-127.)).to(tl.bfloat16).to(tl.float32)
    u=(tl.load(U+pos,valid,0.).to(tl.float32)*tl.exp2(us-127.)).to(tl.bfloat16).to(tl.float32)
    denominator=1.+tl.exp(-g)
    a=(g/denominator).to(tl.bfloat16).to(tl.float32)
    y=(a*u).to(tl.bfloat16).to(tl.float32)
    dg=tl.full(pos.shape,0.,tl.float32);du=dg
    if BACKWARD:
        dy=tl.load(DY+pos,valid,0).to(tl.float32);sig=1./denominator
        derivative=sig*(1.+g*(1.-sig))
        dg=((dy*u)*derivative).to(tl.bfloat16).to(tl.float32)
        du=(dy*a).to(tl.bfloat16).to(tl.float32)
    return y,dg,du


@triton.jit
def mxfp8_swiglu(G,U,DY,Y,SY,QG,SG,QGT,SGT,QU,SU,QUT,SUT,GS,US,
                 ROWS:tl.constexpr,COLS:tl.constexpr,BACKWARD:tl.constexpr,TILE:tl.constexpr=64,GRAD_MAX:tl.constexpr=57344.,COL_TILE:tl.constexpr=0,CACHED:tl.constexpr=False,SAVE_INPUTS:tl.constexpr=False):
    WIDTH:tl.constexpr=COL_TILE if COL_TILE else TILE
    row=tl.program_id(0)*TILE+tl.arange(0,TILE);col=tl.program_id(1)*WIDTH+tl.arange(0,WIDTH)
    pos=row[:,None]*COLS+col[None,:];valid=(row[:,None]<ROWS)&(col[None,:]<COLS)
    if SAVE_INPUTS:
        tl.static_assert(not BACKWARD and not CACHED)
        g=tl.load(G+pos,valid,0).to(tl.float32);u=tl.load(U+pos,valid,0).to(tl.float32)
        a=(g/(1.+tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        y=(a*u).to(tl.bfloat16).to(tl.float32)
    elif CACHED:y,dg,du=cached_swiglu_values(G,U,GS,US,DY,row,col,valid,COLS,BACKWARD)
    else:y,dg,du=values(G,U,DY,pos,valid,BACKWARD)
    value,exponent=tile_quantize(y,448.,TILE,BACKWARD,WIDTH)
    offset=col[None,:]*ROWS+row[:,None] if BACKWARD else pos
    tl.store(Y+offset,value.to(Y.dtype.element_ty,fp_downcast_rounding='rtne'),valid)
    store_tile_scales(SY,exponent,row,col,ROWS,COLS,TILE,BACKWARD,WIDTH)
    if SAVE_INPUTS:
        value,exponent=tile_quantize(g,448.,TILE,False,WIDTH)
        tl.store(QG+pos,value.to(QG.dtype.element_ty,fp_downcast_rounding='rtne'),valid)
        store_tile_scales(SG,exponent,row,col,ROWS,COLS,TILE,False,WIDTH)
        value,exponent=tile_quantize(u,448.,TILE,False,WIDTH)
        tl.store(QU+pos,value.to(QU.dtype.element_ty,fp_downcast_rounding='rtne'),valid)
        store_tile_scales(SU,exponent,row,col,ROWS,COLS,TILE,False,WIDTH)
    if BACKWARD:
        value,exponent=tile_quantize(dg,GRAD_MAX,TILE,False,WIDTH)
        tl.store(QG+pos,value.to(QG.dtype.element_ty,fp_downcast_rounding='rtne'),valid)
        store_tile_scales(SG,exponent,row,col,ROWS,COLS,TILE,False,WIDTH)
        value,exponent=tile_quantize(dg,GRAD_MAX,TILE,True,WIDTH)
        tl.store(QGT+col[None,:]*ROWS+row[:,None],value.to(QGT.dtype.element_ty,fp_downcast_rounding='rtne'),valid)
        store_tile_scales(SGT,exponent,row,col,ROWS,COLS,TILE,True,WIDTH)
        value,exponent=tile_quantize(du,GRAD_MAX,TILE,False,WIDTH)
        tl.store(QU+pos,value.to(QU.dtype.element_ty,fp_downcast_rounding='rtne'),valid)
        store_tile_scales(SU,exponent,row,col,ROWS,COLS,TILE,False,WIDTH)
        value,exponent=tile_quantize(du,GRAD_MAX,TILE,True,WIDTH)
        tl.store(QUT+col[None,:]*ROWS+row[:,None],value.to(QUT.dtype.element_ty,fp_downcast_rounding='rtne'),valid)
        store_tile_scales(SUT,exponent,row,col,ROWS,COLS,TILE,True,WIDTH)


def quantize_swiglu(ops,gate,up,dy=None,*,save_inputs=False):
    rows,cols=gate.shape;backward=dy is not None
    cached=gate.dtype=='fp8e4nv'
    if cached and (up.dtype!='fp8e4nv' or not hasattr(gate,'mx32_scale') or not hasattr(up,'mx32_scale')):
        raise ValueError('compressed FFN cache needs two independently scaled E4M3 inputs')
    if save_inputs and (cached or backward):raise ValueError('save FFN inputs only in the original BF16 forward')
    gradient_dtype=getattr(ops.backend,'mxfp8_gradient_dtype','fp8e5')
    if gradient_dtype not in ('fp8e4nv','fp8e5'):raise ValueError('unsupported MXFP8 gradient format')
    y=ops.empty((cols,rows) if backward else gate.shape,'fp8e4nv');sy=new_scales(ops,y.shape);y.mx32_scale=sy
    if backward:
        qg,qu=[ops.empty(gate.shape,gradient_dtype) for _ in range(2)]
        qgt,qut=[ops.empty((cols,rows),gradient_dtype) for _ in range(2)]
        sg,su,sgt,sut=[new_scales(ops,t.shape) for t in (qg,qu,qgt,qut)]
        for t,s in ((qg,sg),(qu,su),(qgt,sgt),(qut,sut)):t.mx32_scale=s
        qg.transpose_data=qgt;qu.transpose_data=qut
    elif save_inputs:
        qg,qu=[ops.empty(gate.shape,'fp8e4nv') for _ in range(2)]
        sg,su=[new_scales(ops,gate.shape) for _ in range(2)]
        qg.mx32_scale=sg;qu.mx32_scale=su
        qgt,qut,sgt,sut=qg,qu,sg,su
    else:qg=qgt=qu=qut=y;sg=sgt=su=sut=sy
    tile=getattr(ops.backend,'mxfp8_ffn_tile',64)
    width=getattr(ops.backend,'mxfp8_ffn_col_tile',tile)
    if tile not in (32,64,128) or width not in (32,64,128):raise ValueError('MXFP8 FFN tile must divide scale tiles')
    ops.backend.launch(mxfp8_swiglu,(triton.cdiv(rows,128)*(128//tile),triton.cdiv(cols,128)*(128//width)),
        (gate,up,dy or gate,y,sy,qg,sg,qgt,sgt,qu,su,qut,sut,gate.mx32_scale if cached else gate,up.mx32_scale if cached else up),
        ROWS=rows,COLS=cols,BACKWARD=backward,TILE=tile,
        GRAD_MAX=448. if gradient_dtype=='fp8e4nv' else 57344.,COL_TILE=width,CACHED=cached,SAVE_INPUTS=save_inputs)
    return (y,sy,qg,sg,qu,su) if backward or save_inputs else (y,sy)


@triton.jit
def mxfp8_norm_forward(X,A,W,Z,Y,R,Q,S,M:tl.constexpr,N:tl.constexpr,EPS:tl.constexpr,
                      BLOCK:tl.constexpr,ADD_RESIDUAL:tl.constexpr,ROWS:tl.constexpr=1):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS);d=tl.arange(0,BLOCK)
    valid=(row[:,None]<M)&(d[None,:]<N);offset=row[:,None]*N+d[None,:]
    z=tl.load(X+offset,valid,0).to(tl.float32)
    if ADD_RESIDUAL:
        z=(z+tl.load(A+offset,valid,0).to(tl.float32)).to(Z.dtype.element_ty).to(tl.float32)
        tl.store(Z+offset,z,valid)
    weight=tl.load(W+d,d<N,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(z*z,1)/N+EPS)
    y=(z*inv[:,None]*weight[None,:]).to(Y.dtype.element_ty)
    tl.store(Y+offset,y,valid);tl.store(R+row,inv,row<M)
    blocks=tl.reshape(y.to(tl.float32),(ROWS,BLOCK//32,32))
    exponent,inverse=scale_exponent(tl.max(tl.abs(blocks),2),448.)
    scaled=tl.reshape(blocks*inverse[:,:,None],(ROWS,BLOCK))
    value=tl.minimum(tl.maximum(scaled,-448.),448.).to(Q.dtype.element_ty,fp_downcast_rounding='rtne')
    tl.store(Q+offset,value,valid)
    group=tl.arange(0,BLOCK//32)
    sf_offset=scale_offset(row[:,None],group[None,:],tl.cdiv(N,128)*4)
    tl.store(S.to(tl.pointer_type(tl.uint8))+sf_offset,exponent,group[None,:]<tl.cdiv(N,128)*4)


def norm(ops,x,weight,eps,addition=None,storage=None):
    n=x.shape[-1];rows=x.numel//n
    if not 32<=n<=8192 or weight.shape!=(n,) or eps<=0:raise ValueError('unsupported fused MXFP8 RMSNorm')
    if storage is None:
        z=ops.empty(x.shape,'fp32' if addition is not None and x.dtype!=addition.dtype else x.dtype) if addition is not None else x
        y=ops.empty(x.shape,'bf16');inv=ops.empty((rows,),'fp32')
        q=ops.empty((rows,n),'fp8e4nv');scales=new_scales(ops,q.shape)
    else:z,y,inv,q,scales=storage
    tile=getattr(ops.backend,'mxfp8_norm_forward_rows',1)
    if tile not in (1,2,4,8):raise ValueError('MXFP8 norm row tile must divide scale storage tiles')
    ops.backend.launch(mxfp8_norm_forward,(triton.cdiv(rows,128)*(128//tile),),
        (x,addition or x,weight,z,y,inv,q,scales),M=rows,N=n,EPS=eps,BLOCK=triton.next_power_of_2(n),ADD_RESIDUAL=addition is not None,ROWS=tile)
    q.mx32_scale=scales;q.mx32_source=y.view(rows,n)
    if hasattr(q,'transpose_data'):del q.transpose_data
    y.mx32_quantized=(q,scales)
    return z,y,inv


@triton.jit
def mxfp8_norm_backward(X,W,R,G0,G1,G2,G3,DX,DY_BF,Q,S,PARTIAL_DW,RESIDUAL,
                       M:tl.constexpr,N:tl.constexpr,GROUPS:tl.constexpr,COUNT:tl.constexpr,
                       BLOCK:tl.constexpr,GRAD_MAX:tl.constexpr):
    group=tl.program_id(0);col=tl.arange(0,BLOCK)
    weight=tl.load(W+col,col<N,0).to(tl.float32);dw=tl.full((BLOCK,),0.,tl.float32)
    for row in range(group,tl.cdiv(M,128)*128,GROUPS):
        valid=(row<M)&(col<N);offset=row*N+col
        dy=tl.load(G0+offset,valid,0).to(tl.float32)
        if COUNT>=2:dy=dy+tl.load(G1+offset,valid,0).to(tl.float32)
        if COUNT>=3:dy=dy+tl.load(G2+offset,valid,0).to(tl.float32)
        if COUNT>=4:dy=dy+tl.load(G3+offset,valid,0).to(tl.float32)
        dy=dy.to(X.dtype.element_ty).to(tl.float32)
        x=tl.load(X+offset,valid,0).to(tl.float32);inv=tl.load(R+row,row<M,0)
        xhat=x*inv;dxhat=dy*weight;dot=tl.sum(dxhat*xhat,0)/N
        dx=(inv*(dxhat-xhat*dot)).to(X.dtype.element_ty).to(tl.float32)
        dx=tl.load(RESIDUAL+offset,valid,0).to(tl.float32)+dx
        tl.store(DX+offset,dx,valid)
        rounded=dx.to(tl.bfloat16);tl.store(DY_BF+offset,rounded,valid)
        blocks=tl.reshape(rounded.to(tl.float32),(BLOCK//32,32))
        exponent,inverse=scale_exponent(tl.max(tl.abs(blocks),1),GRAD_MAX)
        scaled=tl.reshape(blocks*inverse[:,None],(BLOCK,))
        value=tl.minimum(tl.maximum(scaled,-GRAD_MAX),GRAD_MAX).to(Q.dtype.element_ty,fp_downcast_rounding='rtne')
        tl.store(Q+offset,value,valid)
        block=tl.arange(0,BLOCK//32);sf_offset=scale_offset(row,block,tl.cdiv(N,128)*4)
        tl.store(S.to(tl.pointer_type(tl.uint8))+sf_offset,exponent,block<tl.cdiv(N,128)*4)
        dw+=dy*xhat
    tl.store(PARTIAL_DW+group*N+col,dw,col<N)


def norm_backward(ops,saved,parts,residual,dx,partial,groups):
    x,w,r=saved;n=x.shape[-1];rows=x.numel//n
    bf=ops.empty(x.shape,'bf16');dtype=getattr(ops.backend,'mxfp8_gradient_dtype','fp8e5')
    q=ops.empty((rows,n),dtype);s=new_scales(ops,q.shape)
    inputs=(list(parts)+[parts[0]]*4)[:4]
    ops.backend.launch(mxfp8_norm_backward,(groups,),(x,w,r,*inputs,dx,bf,q,s,partial,residual),
        M=rows,N=n,GROUPS=groups,COUNT=len(parts),BLOCK=triton.next_power_of_2(n),GRAD_MAX=448. if dtype=='fp8e4nv' else 57344.)
    q.mx32_scale=s;q.mx32_source=bf.view(rows,n)
    bf.mx32_quantized=(q,s);bf.mx32_quantized_kind='fp8e5';dx.mx32_cast_bf16=bf


def auxiliary_storage(tensors):
    """Keep gradient producer buffers alive through nested checkpoint VJPs."""
    pending=list(tensors);seen=set();extra=[]
    while pending:
        tensor=pending.pop()
        if tensor is None or tensor.storage in seen:continue
        seen.add(tensor.storage)
        children=[]
        cast=getattr(tensor,'mx32_cast_bf16',None)
        if cast is not None:children.append(cast)
        children.extend(getattr(tensor,'mx32_quantized',()))
        if hasattr(tensor,'mx32_scale'):
            children.append(tensor.mx32_scale)
            if hasattr(tensor,'transpose_data'):children.append(tensor.transpose_data)
        extra.extend(children);pending.extend(children)
    return tuple(extra)
