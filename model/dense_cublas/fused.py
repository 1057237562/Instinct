"""Fused Dense kernels preserving the explicit low-precision rounding points."""
import triton
import triton.language as tl


@triton.jit
def projection_cast(W0,W1,W2,C0,C1,C2,N0:tl.constexpr,N1:tl.constexpr,N2:tl.constexpr,B:tl.constexpr=256):
    block=tl.program_id(0);i=tl.arange(0,B)
    if block<tl.cdiv(N0,B):
        offset=block*B+i
        tl.store(C0+offset,tl.load(W0+offset,offset<N0),offset<N0)
    elif block<tl.cdiv(N0,B)+tl.cdiv(N1,B):
        offset=(block-tl.cdiv(N0,B))*B+i
        tl.store(C1+offset,tl.load(W1+offset,offset<N1),offset<N1)
    else:
        offset=(block-tl.cdiv(N0,B)-tl.cdiv(N1,B))*B+i
        tl.store(C2+offset,tl.load(W2+offset,offset<N2),offset<N2)


@triton.jit
def norm_rotary_forward(X, W, COS, SIN, Y, R, N: tl.constexpr, EPS: tl.constexpr,
                        SEQ: tl.constexpr, HEADS: tl.constexpr, START: tl.constexpr,
                        BLOCK: tl.constexpr):
    row=tl.program_id(0);d=tl.arange(0,BLOCK);mask=d<N
    other_d=(d+N//2)%N;pos=(row//HEADS)%SEQ+START
    if START == -1: pos=row//HEADS
    x=tl.load(X+row*N+d,mask,0).to(tl.float32)
    w=tl.load(W+d,mask,0).to(tl.float32)
    r=tl.rsqrt(tl.sum(x*x,0)/N+EPS)
    other=tl.load(X+row*N+other_d,mask,0).to(tl.float32)
    other_w=tl.load(W+other_d,mask,0).to(tl.float32)
    # Match RMSNorm's stored BF16/FP16 result before applying RoPE.
    z=(x*r*w).to(X.dtype.element_ty).to(tl.float32)
    partner=(other*r*other_w).to(X.dtype.element_ty).to(tl.float32)
    cos=tl.load(COS+pos*N+d,mask,0).to(tl.float32)
    sin=tl.load(SIN+pos*N+d,mask,0).to(tl.float32)
    sign=tl.where(d<N//2,-1.,1.)
    tl.store(Y+row*N+d,z*cos+sign*partner*sin,mask)
    tl.store(R+row,r)


@triton.jit
def norm_rotary_backward(X, W, R, COS, SIN, G, DX, PARTIAL_DW,
                         M: tl.constexpr, N: tl.constexpr, GROUPS: tl.constexpr,
                         SEQ: tl.constexpr, HEADS: tl.constexpr, START: tl.constexpr,
                         BLOCK: tl.constexpr):
    group=tl.program_id(0);d=tl.arange(0,BLOCK);mask=d<N
    other_d=(d+N//2)%N;sign=tl.where(d<N//2,1.,-1.)
    w=tl.load(W+d,mask,0).to(tl.float32);dw=tl.full((BLOCK,),0.,tl.float32)
    for row in range(group,M,GROUPS):
        pos=(row//HEADS)%SEQ+START
        if START == -1: pos=row//HEADS
        g=tl.load(G+row*N+d,mask,0).to(tl.float32)
        partner=tl.load(G+row*N+other_d,mask,0).to(tl.float32)
        cos=tl.load(COS+pos*N+d,mask,0).to(tl.float32)
        sin=tl.load(SIN+pos*N+other_d,mask,0).to(tl.float32)
        # Legacy RoPE VJP computed in FP32; RMSNorm then cast its dY.
        dy=(g*cos+sign*partner*sin).to(X.dtype.element_ty).to(tl.float32)
        x=tl.load(X+row*N+d,mask,0).to(tl.float32)
        r=tl.load(R+row);xhat=x*r;dxhat=dy*w
        dot=tl.sum(dxhat*xhat,0)/N
        tl.store(DX+row*N+d,r*(dxhat-xhat*dot),mask)
        dw+=dy*xhat
    tl.store(PARTIAL_DW+group*N+d,dw,mask)


@triton.jit
def rounded_swiglu(G,U,Y,DY,DG,DU,N:tl.constexpr,BACKWARD:tl.constexpr,B:tl.constexpr=256):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N
    g=tl.load(G+i,mask,0).to(tl.float32);u=tl.load(U+i,mask,0).to(tl.float32)
    denominator=1.+tl.exp(-g)
    activated=(g/denominator).to(G.dtype.element_ty).to(tl.float32)
    if BACKWARD:
        dy=tl.load(DY+i,mask,0).to(tl.float32);sig=1./denominator
        derivative=sig*(1.+g*(1.-sig))
        tl.store(DG+i,(dy*u)*derivative,mask)
        tl.store(DU+i,dy*activated,mask)
    else:tl.store(Y+i,activated*u,mask)
