"""Expert GEMMs reading FP32 parameter pointers directly, with BF16 rounding."""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

@triton.jit
def _direct_mm(
    X, W, Ends, Y,
    K: tl.constexpr, N: tl.constexpr, E: tl.constexpr,
    X0: tl.constexpr, X1: tl.constexpr,
    W1: tl.constexpr, W2: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    tile = tl.program_id(0)
    problem_start = 0
    row_start = 0
    columns = tl.cdiv(N, BN)
    for expert in range(E):
        base = W[0]
        for i in tl.static_range(1, len(W)):
            base = tl.where(expert == i, W[i], base)
        row_end = tl.load(Ends + expert)
        rows = row_end - row_start
        tiles = tl.cdiv(rows, BM) * columns
        while (tile >= problem_start) & (tile < problem_start + tiles):
            local = tile - problem_start
            rm = (local // columns) * BM + tl.arange(0, BM)
            rn = (local % columns) * BN + tl.arange(0, BN)
            rk = tl.arange(0, BK)
            acc = tl.full((BM, BN), 0, tl.float32)
            for block in range(tl.cdiv(K, BK)):
                kk = block * BK + rk
                a = tl.load(X + (row_start + rm[:, None]) * X0 + kk[None, :] * X1,
                            (rm[:, None] < rows) & (kk[None, :] < K), other=0)
                b = tl.load(base + kk[:, None] * W1 + rn[None, :] * W2,
                            (kk[:, None] < K) & (rn[None, :] < N), other=0)
                acc = tl.dot(a, b.to(tl.bfloat16), acc)
            tl.store(Y + (row_start + rm[:, None]) * N + rn[None, :],
                     acc.to(Y.dtype.element_ty),
                     (rm[:, None] < rows) & (rn[None, :] < N))
            tile += PROGRAMS
        problem_start += tiles
        row_start = row_end


def direct_mm(x,weights,offsets,backward=False):
    from model.grouped_mm_triton import _programs, _FORWARD_TILE
    inner=x.shape[1]
    columns=weights[0].shape[1 if backward else 0]
    strides=weights[0].stride() if backward else weights[0].T.stride()
    out=torch.empty((x.shape[0],columns),device=x.device,dtype=torch.bfloat16)
    bm,bn,bk,warps,stages=_FORWARD_TILE
    programs=_programs(x.device)
    _direct_mm[(programs,)](x,tuple(weights),offsets,out,inner,columns,len(weights),
        *x.stride(),*strides,programs,bm,bn,bk,num_warps=warps,num_stages=stages)
    return out


@torch.library.custom_op('instinct::checkpoint_direct_mm',mutates_args=())
def _checkpoint_direct_mm(x:torch.Tensor,weights:list[torch.Tensor],offsets:torch.Tensor)->torch.Tensor:
    return direct_mm(x,weights,offsets)


@_checkpoint_direct_mm.register_fake
def _checkpoint_direct_mm_fake(x,weights,offsets):
    return x.new_empty((x.shape[0],weights[0].shape[0]))


class _ExpertLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,offsets,*weights):
        import os
        ctx.save_for_backward(x,offsets,*weights)
        if os.environ.get('INSTINCT_MOE_SELECTIVE_CHECKPOINT','0')!='0':
            return _checkpoint_direct_mm(x,list(weights),offsets)
        return direct_mm(x,weights,offsets)

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        from model.grouped_mm_triton import ragged_weight_grad
        x,offsets,*weights=ctx.saved_tensors
        with torch.autocast('cuda',enabled=False):
            dx=direct_mm(grad,weights,offsets,backward=True) if ctx.needs_input_grad[0] else None
            if any(ctx.needs_input_grad[2:]):
                dw=ragged_weight_grad(x,grad,offsets,len(weights),
                    transposed_output=True,output_dtype=torch.float32).transpose(1,2)
                gradients=tuple(dw.unbind(0))
            else:
                gradients=(None,)*len(weights)
        return dx,None,*gradients


def expert_linear(x,weights,offsets):
    return _ExpertLinear.apply(x,offsets,*weights)
