"""Fused Dense kernels preserving the explicit low-precision rounding points."""
import triton
import triton.language as tl


@triton.jit
def residual_norm(X,A,W,Z,Y,R,MAXIMUM,N:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);d=tl.arange(0,BLOCK);mask=d<N
    x=tl.load(X+row*N+d,mask,0).to(tl.float32)
    a=tl.load(A+row*N+d,mask,0).to(tl.float32)
    z=(x+a).to(Z.dtype.element_ty).to(tl.float32)
    w=tl.load(W+d,mask,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(z*z,0)/N+EPS)
    y=(z*inv*w).to(Y.dtype.element_ty)
    tl.store(Z+row*N+d,z,mask);tl.store(Y+row*N+d,y,mask)
    tl.store(R+row,inv)
    tl.store(MAXIMUM+row,tl.max(tl.where(mask,tl.abs(y.to(tl.float32)),0.),0))


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
def norm_rotary_forward_rows(X,W,COS,SIN,Y,R,M:tl.constexpr,N:tl.constexpr,EPS:tl.constexpr,
                             SEQ:tl.constexpr,HEADS:tl.constexpr,START:tl.constexpr,
                             BLOCK:tl.constexpr,ROWS:tl.constexpr):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    d=tl.arange(0,BLOCK);other_d=(d+N//2)%N
    valid=(row[:,None]<M)&(d[None,:]<N)
    pos=(row//HEADS)%SEQ+START
    if START == -1:pos=row//HEADS
    x=tl.load(X+row[:,None]*N+d[None,:],valid,0).to(tl.float32)
    other=tl.load(X+row[:,None]*N+other_d[None,:],valid,0).to(tl.float32)
    w=tl.load(W+d,d<N,0).to(tl.float32);ow=tl.load(W+other_d,d<N,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,1)/N+EPS)
    z=(x*inv[:,None]*w[None,:]).to(X.dtype.element_ty).to(tl.float32)
    partner=(other*inv[:,None]*ow[None,:]).to(X.dtype.element_ty).to(tl.float32)
    cos=tl.load(COS+pos[:,None]*N+d[None,:],valid,0).to(tl.float32)
    sin=tl.load(SIN+pos[:,None]*N+d[None,:],valid,0).to(tl.float32)
    sign=tl.where(d<N//2,-1.,1.)
    tl.store(Y+row[:,None]*N+d[None,:],z*cos+sign[None,:]*partner*sin,valid)
    tl.store(R+row,inv,row<M)


@triton.jit
def norm_rotary_backward_rows(X,W,R,COS,SIN,G,DX,PARTIAL_DW,M:tl.constexpr,N:tl.constexpr,
                              GROUPS:tl.constexpr,SEQ:tl.constexpr,HEADS:tl.constexpr,START:tl.constexpr,
                              BLOCK:tl.constexpr,ROWS:tl.constexpr):
    group=tl.program_id(0);r=tl.arange(0,ROWS);d=tl.arange(0,BLOCK);other_d=(d+N//2)%N
    sign=tl.where(d<N//2,1.,-1.);w=tl.load(W+d,d<N,0).to(tl.float32)
    dw=tl.full((ROWS,BLOCK),0.,tl.float32)
    for first in range(group*ROWS,M,GROUPS*ROWS):
        row=first+r;valid=(row[:,None]<M)&(d[None,:]<N)
        pos=(row//HEADS)%SEQ+START
        if START == -1:pos=row//HEADS
        g=tl.load(G+row[:,None]*N+d[None,:],valid,0).to(tl.float32)
        partner=tl.load(G+row[:,None]*N+other_d[None,:],valid,0).to(tl.float32)
        cos=tl.load(COS+pos[:,None]*N+d[None,:],valid,0).to(tl.float32)
        sin=tl.load(SIN+pos[:,None]*N+d[None,:],valid,0).to(tl.float32)
        dy=(g*cos+sign[None,:]*partner*sin).to(X.dtype.element_ty).to(tl.float32)
        x=tl.load(X+row[:,None]*N+d[None,:],valid,0).to(tl.float32)
        inv=tl.load(R+row,row<M,0);xhat=x*inv[:,None];dxhat=dy*w[None,:]
        dot=tl.sum(dxhat*xhat,1)/N
        tl.store(DX+row[:,None]*N+d[None,:],inv[:,None]*(dxhat-xhat*dot[:,None]),valid)
        dw+=dy*xhat
    tl.store(PARTIAL_DW+group*N+d,tl.sum(dw,0),d<N)


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


@triton.jit
def rounded_swiglu_amax(G,U,Y,DY,DG,DU,MAX_G,MAX_U,N:tl.constexpr,
                        BACKWARD:tl.constexpr,B:tl.constexpr=1024):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N
    g=tl.load(G+i,mask,0).to(tl.float32);u=tl.load(U+i,mask,0).to(tl.float32)
    denominator=1.+tl.exp(-g)
    activated=(g/denominator).to(G.dtype.element_ty).to(tl.float32)
    if BACKWARD:
        dy=tl.load(DY+i,mask,0).to(tl.float32);sig=1./denominator
        derivative=sig*(1.+g*(1.-sig))
        dg=((dy*u)*derivative).to(DG.dtype.element_ty)
        du=(dy*activated).to(DU.dtype.element_ty)
        tl.store(DG+i,dg,mask);tl.store(DU+i,du,mask)
        tl.store(MAX_G+tl.program_id(0),tl.max(tl.where(mask,tl.abs(dg.to(tl.float32)),0.),0))
        tl.store(MAX_U+tl.program_id(0),tl.max(tl.where(mask,tl.abs(du.to(tl.float32)),0.),0))
    else:
        y=(activated*u).to(Y.dtype.element_ty)
        tl.store(Y+i,y,mask)
        tl.store(MAX_G+tl.program_id(0),tl.max(tl.where(mask,tl.abs(y.to(tl.float32)),0.),0))
