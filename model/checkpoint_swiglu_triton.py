"""Fused expert gate/up with activation checkpointing inside the operator.

Only the routed input and original FP32 parameters survive forward. Backward
recomputes gate/up and immediately writes SwiGLU derivatives. Every BF16
rounding point and SiLU operation follows the existing expert path.
"""
import os
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from model.swiglu_triton import _div_rn


@triton.jit
def _gate_up(X,G,U,Ends,H,DH,DG,DU,K:tl.constexpr,N:tl.constexpr,E:tl.constexpr,
             X0:tl.constexpr,X1:tl.constexpr,PROGRAMS:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,BACKWARD:tl.constexpr,
             SAVE_H:tl.constexpr):
    tile=tl.program_id(0); problem_start=0; row_start=0
    columns=tl.cdiv(N,BN)
    for expert in range(E):
        gp=G[0];up=U[0]
        for i in tl.static_range(1,len(G)):
            gp=tl.where(expert==i,G[i],gp)
            up=tl.where(expert==i,U[i],up)
        row_end=tl.load(Ends+expert);rows=row_end-row_start
        tiles=tl.cdiv(rows,BM)*columns
        while (tile>=problem_start)&(tile<problem_start+tiles):
            local=tile-problem_start
            rm=(local//columns)*BM+tl.arange(0,BM)
            rn=(local%columns)*BN+tl.arange(0,BN)
            rk=tl.arange(0,BK)
            ag=tl.full((BM,BN),0,tl.float32)
            au=tl.full((BM,BN),0,tl.float32)
            for block in range(tl.cdiv(K,BK)):
                kk=block*BK+rk
                a=tl.load(X+(row_start+rm[:,None])*X0+kk[None,:]*X1,
                          (rm[:,None]<rows)&(kk[None,:]<K),0)
                bg=tl.load(gp+rn[None,:]*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),0).to(tl.bfloat16)
                bu=tl.load(up+rn[None,:]*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),0).to(tl.bfloat16)
                ag=tl.dot(a,bg,ag);au=tl.dot(a,bu,au)
            g=ag.to(tl.bfloat16).to(tl.float32)
            u=au.to(tl.bfloat16).to(tl.float32)
            denominator=1.+libdevice.exp(-g)
            silu=_div_rn(g,denominator).to(tl.bfloat16).to(tl.float32)
            out_index=(row_start+rm[:,None])*N+rn[None,:]
            mask=(rm[:,None]<rows)&(rn[None,:]<N)
            if BACKWARD:
                dh=tl.load(DH+out_index,mask,0).to(tl.float32)
                sigmoid=_div_rn(1.,denominator)
                ds=(dh*u).to(tl.bfloat16).to(tl.float32)
                dg=ds*sigmoid*tl.fma(g,1.-sigmoid,1.)
                tl.store(DG+out_index,dg,mask)
                tl.store(DU+out_index,dh*silu,mask)
            if not BACKWARD:
                tl.store(H+out_index,silu*u,mask)
            elif SAVE_H:
                # This gradient buffer is owned by _CheckpointExperts.backward.
                # Each program consumes its disjoint elements before reusing
                # them for hidden; use the same pointer so aliasing is explicit.
                tl.store(DH+out_index,silu*u,mask)
            tile+=PROGRAMS
        problem_start+=tiles;row_start=row_end


def _launch(x,g,u,ends,dy=None,*,return_hidden=False):
    from model.grouped_mm_triton import _programs
    shape=(x.shape[0],g[0].shape[0])
    out=torch.empty(shape,device=x.device,dtype=torch.bfloat16)
    other=torch.empty_like(out) if dy is not None else out
    # return_hidden consumes dy in place. Only the owned down-projection dX
    # buffer uses this path; caller-owned upstream gradients are never mutated.
    if return_hidden and (dy is None or not dy.is_contiguous() or dy.shape != shape):
        raise ValueError('hidden replay requires an owned contiguous SwiGLU-gradient buffer')
    hidden=dy if return_hidden else out
    programs=_programs(x.device)
    tile = tuple(map(int, os.environ.get('INSTINCT_MOE_CHECKPOINT_TILE', '64,64,32,4,3').split(',')))
    if len(tile) != 5:
        raise ValueError('INSTINCT_MOE_CHECKPOINT_TILE expects BM,BN,BK,warps,stages')
    bm,bn,bk,warps,stages=tile
    _gate_up[(programs,)](x,tuple(g),tuple(u),ends,hidden,
        dy if dy is not None else out,out,other,x.shape[1],shape[1],len(g),*x.stride(),
        programs,bm,bn,bk,dy is not None,return_hidden,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    if return_hidden:
        return out,other,hidden
    return (out,other) if dy is not None else out


class _CheckpointSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,ends,*weights):
        ctx.experts=len(weights)//2
        ctx.save_for_backward(x,ends,*weights)
        return _launch(x,weights[:ctx.experts],weights[ctx.experts:],ends)

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        from model.expert_linear_triton import direct_mm
        from model.grouped_mm_triton import ragged_weight_grad
        x,ends,*weights=ctx.saved_tensors
        g,u=weights[:ctx.experts],weights[ctx.experts:]
        with torch.autocast('cuda',enabled=False):
            dg,du=_launch(x,g,u,ends,dy.contiguous())
            dx=direct_mm(dg,g,ends,True)+direct_mm(du,u,ends,True) if ctx.needs_input_grad[0] else None
            dgw=ragged_weight_grad(x,dg,ends,ctx.experts,transposed_output=True,
                                  output_dtype=torch.float32).transpose(1,2)
            duw=ragged_weight_grad(x,du,ends,ctx.experts,transposed_output=True,
                                  output_dtype=torch.float32).transpose(1,2)
        return dx,None,*dgw.unbind(0),*duw.unbind(0)


def checkpoint_swiglu(x,gate_weights,up_weights,offsets):
    return _CheckpointSwiGLU.apply(x,offsets,*gate_weights,*up_weights)


class _CheckpointExperts(torch.autograd.Function):
    """Recompute gate/up/SwiGLU, without retaining the wide down-projection input.

    Compared with checkpoint_swiglu + expert_linear, the wide intermediate is
    local to each layer's backward. Compared with whole-FFN checkpointing,
    routing, parameter packing and the down projection are not replayed.
    Original FP32 parameters remain explicit inputs and BF16 rounding is kept.
    """
    @staticmethod
    def forward(ctx,x,ends,*weights):
        from model.expert_linear_triton import direct_mm
        ctx.experts=len(weights)//3
        ctx.save_for_backward(x,ends,*weights)
        e=ctx.experts
        hidden=_launch(x,weights[:e],weights[e:2*e],ends)
        return direct_mm(hidden,weights[2*e:],ends)

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        from model.expert_linear_triton import direct_mm
        from model.grouped_mm_triton import ragged_weight_grad
        x,ends,*weights=ctx.saved_tensors
        e=ctx.experts
        g,u,d=weights[:e],weights[e:2*e],weights[2*e:]
        with torch.autocast('cuda',enabled=False):
            dy=dy.contiguous()
            dh=direct_mm(dy,d,ends,True)
            dg,du,hidden=_launch(x,g,u,ends,dh,return_hidden=True)
            del dh
            ddw=ragged_weight_grad(hidden,dy,ends,e,transposed_output=True,
                                  output_dtype=torch.float32).transpose(1,2)
            del hidden
            dx=direct_mm(dg,g,ends,True)+direct_mm(du,u,ends,True) if ctx.needs_input_grad[0] else None
            dgw=ragged_weight_grad(x,dg,ends,e,transposed_output=True,
                                  output_dtype=torch.float32).transpose(1,2)
            duw=ragged_weight_grad(x,du,ends,e,transposed_output=True,
                                  output_dtype=torch.float32).transpose(1,2)
        return dx,None,*dgw.unbind(0),*duw.unbind(0),*ddw.unbind(0)


def checkpoint_experts(x,gate_weights,up_weights,down_weights,offsets):
    return _CheckpointExperts.apply(x,offsets,*gate_weights,*up_weights,*down_weights)
