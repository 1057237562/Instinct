"""Stable, device-only MoE permutation and BF16 routing operations.

No route is dropped or padded. The stable permutation is the same as argsort.
The gather/scatter backward is specialized to top-1/2 so summation has at most
two operands; BF16 multiplication boundaries match the original eager path.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _counts(Ids, Counts, ROUTES: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    chunk, expert = tl.program_id(0), tl.program_id(1)
    r = chunk*BLOCK+tl.arange(0,BLOCK)
    ids = tl.load(Ids+r,r<ROUTES,-1)
    tl.store(Counts+expert*BLOCKS+chunk,tl.sum((ids==expert).to(tl.int32)))


@triton.jit
def _permutation(Ids, Counts, Order, Inverse, Ends, ROUTES: tl.constexpr,
                 EXPERTS: tl.constexpr, BLOCKS: tl.constexpr,
                 PAD_BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    chunk, expert = tl.program_id(0), tl.program_id(1)
    ee=tl.arange(0,EXPERTS)[:,None]
    cc=tl.arange(0,PAD_BLOCKS)[None,:]
    counts=tl.load(Counts+ee*BLOCKS+cc,cc<BLOCKS,0)
    before=tl.sum(tl.sum(tl.where((ee<expert)|((ee==expert)&(cc<chunk)),counts,0),0),0)
    end=tl.sum(tl.sum(tl.where(ee<=expert,counts,0),0),0)
    if chunk==0: tl.store(Ends+expert,end)
    r=chunk*BLOCK+tl.arange(0,BLOCK)
    ids=tl.load(Ids+r,r<ROUTES,-1)
    selected=(ids==expert)&(r<ROUTES)
    pos=before+tl.cumsum(selected.to(tl.int32))-1
    tl.store(Order+pos,r,selected)
    tl.store(Inverse+r,pos,selected)


def permutation(ids, experts):
    routes=ids.numel(); blocks=triton.cdiv(routes,1024)
    counts=torch.empty((experts,blocks),device=ids.device,dtype=torch.int32)
    order=torch.empty((routes,),device=ids.device,dtype=torch.int32)
    inverse=torch.empty_like(order)
    ends=torch.empty((experts,),device=ids.device,dtype=torch.int32)
    _counts[(blocks,experts)](ids,counts,routes,blocks,1024)
    _permutation[(blocks,experts)](ids,counts,order,inverse,ends,routes,
        experts,blocks,triton.next_power_of_2(blocks),1024)
    return order,inverse,ends


@triton.jit
def _gather(X, Order, Out, H: tl.constexpr, ROUTES: tl.constexpr,
            TOP: tl.constexpr, BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    r=i//H; h=i%H
    original=tl.load(Order+r,r<ROUTES,0)
    val=tl.load(X+(original//TOP)*H+h,r<ROUTES,0)
    tl.store(Out+i,val,r<ROUTES)


@triton.jit
def _ungather(DY, Inv, DX, H: tl.constexpr, TOKENS: tl.constexpr,
              TOP: tl.constexpr, BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    t=i//H; h=i%H
    r0=tl.load(Inv+t*TOP,t<TOKENS,0)
    v0=tl.load(DY+r0*H+h,t<TOKENS,0).to(DX.dtype.element_ty)
    if TOP==2:
        r1=tl.load(Inv+t*TOP+1,t<TOKENS,0)
        v1=tl.load(DY+r1*H+h,t<TOKENS,0).to(DX.dtype.element_ty)
        v0=v0.to(tl.float32)+v1.to(tl.float32)
    tl.store(DX+i,v0,t<TOKENS)


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,order,inverse,top):
        ctx.save_for_backward(inverse); ctx.top=top; ctx.dtype=x.dtype
        out=torch.empty((order.numel(),x.shape[1]),device=x.device,dtype=torch.bfloat16)
        _gather[(triton.cdiv(out.numel(),1024),)](x,order,out,x.shape[1],order.numel(),top,1024)
        return out
    @staticmethod
    def backward(ctx,dy):
        (inv,)=ctx.saved_tensors
        dy=dy.contiguous()
        dx=torch.empty((inv.numel()//ctx.top,dy.shape[1]),device=dy.device,dtype=ctx.dtype)
        _ungather[(triton.cdiv(dx.numel(),1024),)](dy,inv,dx,dy.shape[1],dx.shape[0],ctx.top,1024)
        return dx,None,None,None


@triton.jit
def _combine(Y, W, Inv, Out, H: tl.constexpr, TOKENS: tl.constexpr,
             TOP: tl.constexpr, BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    t=i//H; h=i%H
    r0=tl.load(Inv+t*TOP,t<TOKENS,0)
    w0=tl.load(W+t*TOP,t<TOKENS,0).to(tl.bfloat16).to(tl.float32)
    y0=tl.load(Y+r0*H+h,t<TOKENS,0).to(tl.float32)
    v=(y0*w0).to(tl.bfloat16).to(Out.dtype.element_ty)
    if TOP==2:
        r1=tl.load(Inv+t*TOP+1,t<TOKENS,0)
        w1=tl.load(W+t*TOP+1,t<TOKENS,0).to(tl.bfloat16).to(tl.float32)
        y1=tl.load(Y+r1*H+h,t<TOKENS,0).to(tl.float32)
        v1=(y1*w1).to(tl.bfloat16).to(Out.dtype.element_ty)
        v=v.to(tl.float32)+v1.to(tl.float32)
    tl.store(Out+i,v,t<TOKENS)


@triton.jit
def _combine_backward(G, Y, W, Order, DY, Products, H: tl.constexpr,
                      TOP: tl.constexpr, BH: tl.constexpr):
    r=tl.program_id(0)
    original=tl.load(Order+r)
    h=tl.arange(0,BH)
    g=tl.load(G+(original//TOP)*H+h,h<H,0).to(tl.bfloat16).to(tl.float32)
    w=tl.load(W+original).to(tl.bfloat16).to(tl.float32)
    y=tl.load(Y+r*H+h,h<H,0).to(tl.float32)
    tl.store(DY+r*H+h,g*w,h<H)
    tl.store(Products+r*H+h,g*y,h<H)


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx,y,weights,order,inverse,output_dtype):
        ctx.save_for_backward(y,weights,order,inverse); ctx.top=weights.shape[1]
        out=torch.empty((weights.shape[0],y.shape[1]),device=y.device,dtype=output_dtype)
        _combine[(triton.cdiv(out.numel(),1024),)](y,weights,inverse,out,
            y.shape[1],out.shape[0],ctx.top,1024,enable_fp_fusion=False)
        return out
    @staticmethod
    def backward(ctx,grad):
        y,w,order,inverse=ctx.saved_tensors
        grad=grad.contiguous()
        dy=torch.empty_like(y); products=torch.empty_like(y)
        _combine_backward[(y.shape[0],)](grad,y,w,order,dy,products,y.shape[1],ctx.top,
            triton.next_power_of_2(y.shape[1]),enable_fp_fusion=False)
        # Keep PyTorch's horizontal BF16 reduction tree. A different tree can
        # perturb a router gradient at a rounding midpoint and Muon amplifies it.
        dw=products.sum(dim=1,keepdim=True).index_select(0,inverse).reshape(w.shape).to(w.dtype)
        return dy,dw,None,None,None


gather = _Gather.apply
combine = _Combine.apply
