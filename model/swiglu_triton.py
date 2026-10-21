"""Fused SiLU/multiply preserving the eager BF16 intermediate rounding."""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _div_rn(a,b):
    # libdevice's division is compiled with NVVM reflect-ftz in Triton. The
    # eager SiLU retains subnormals; explicit PTX division does so as well.
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;",constraints="=f,f,f",
        args=[a,b],dtype=tl.float32,is_pure=True,pack=1)


@triton.jit
def _forward(G,U,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    g=tl.load(G+i,i<N,0).to(tl.float32)
    u=tl.load(U+i,i<N,0).to(tl.float32)
    silu=_div_rn(g,1.+libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+i,silu*u,i<N)


@triton.jit
def _backward(G,U,D,DG,DU,N:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    g=tl.load(G+i,i<N,0).to(tl.float32)
    u=tl.load(U+i,i<N,0).to(tl.float32)
    d=tl.load(D+i,i<N,0).to(tl.float32)
    denom=1.+libdevice.exp(-g)
    silu=_div_rn(g,denom).to(tl.bfloat16).to(tl.float32)
    s=_div_rn(1.,denom)
    ds=(d*u).to(tl.bfloat16).to(tl.float32)
    dg=ds*s*tl.fma(g,1.-s,1.)
    tl.store(DG+i,dg,i<N)
    tl.store(DU+i,d*silu,i<N)


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,gate,up):
        ctx.save_for_backward(gate,up)
        out=torch.empty_like(gate)
        _forward[(triton.cdiv(gate.numel(),1024),)](gate,up,out,gate.numel(),1024,enable_fp_fusion=False)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        gate,up=ctx.saved_tensors
        dg,du=torch.empty_like(gate),torch.empty_like(up)
        grad=grad.contiguous()
        _backward[(triton.cdiv(gate.numel(),1024),)](gate,up,grad,dg,du,gate.numel(),1024,enable_fp_fusion=False)
        return dg,du


def swiglu(gate,up):
    return _SwiGLU.apply(gate,up)
