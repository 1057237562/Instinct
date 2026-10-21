"""One-pass expert FP32 -> BF16 stack, retaining original parameter identities."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(W, O, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    offset=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    expert=tl.program_id(1)
    for i in tl.static_range(len(W)):
        if expert==i:
            value=tl.load(W[i]+offset,offset<ELEMENTS,0)
            tl.store(O+i*ELEMENTS+offset,value,offset<ELEMENTS)


class _Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dtype, *weights):
        ctx.source_dtype=weights[0].dtype
        output=torch.empty((len(weights),*weights[0].shape),device=weights[0].device,dtype=dtype)
        size=weights[0].numel()
        _pack[(triton.cdiv(size,1024),len(weights))](weights,output,size,1024)
        return output

    @staticmethod
    def backward(ctx, grad):
        # Return each Linear's row-major gradient directly. The grouped GEMM
        # supplies a transposed layout; preserving that layout would make
        # AccumulateGrad allocate/copy all expert matrices a second time.
        packed = grad.to(dtype=ctx.source_dtype, memory_format=torch.contiguous_format)
        return (None,*packed.unbind(0))


def pack_weights(weights, dtype):
    return _Pack.apply(dtype,*weights).transpose(1,2)
