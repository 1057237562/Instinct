"""Optional FP16 Gram recurrence for the native V2 rectangular FFN matrices.

Algorithm: Zhang, Amsel, Chen and Dao, Gram Newton-Schulz (2026),
https://github.com/Dao-AILab/gram-newton-schulz . This implementation retains
Instinct's normalization, five quintic steps and coefficients. A restart
after step three bounds accumulated Gram rounding error. No Torch runtime.
"""
import triton
import triton.language as tl
from .muon_gram import gram


@triton.jit
def add_identity(Z,Q,M:tl.constexpr,N:tl.constexpr,B:tl.constexpr=1024):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    z=tl.load(Z+i,valid,0).to(tl.float32)
    tl.store(Q+i,z+tl.where((i//M)%M==i%M,3.4445,0.),valid)


def orthogonalize(plan,source,mark):
    e=plan.opt.e;b=e.backend;batch,m,k=source.shape
    if source.dtype!='bf16' or (m,k)!=(1536,5504) or plan.opt.ns_steps!=5:
        raise ValueError('experimental Gram recurrence requires V2 BF16 FFN and five NS steps')
    x=e.ops.cast(source,'fp16')
    def multiply(a,v,c,rows,cols,inner,*,transpose=False,alpha=1.,beta=0.,symmetric=False):
        out=b.empty((batch,rows,cols),'fp16')
        tiled=False
        if a is v and transpose:
            tiled=gram(b,a,out,tile=64)
        elif a is v and symmetric:
            tiled=gram(b,a,out,tile=64,polynomial=True)
        elif symmetric:
            tiled=gram(b,v,out,tile=64,other=a,addend=c,alpha=alpha,beta=beta)
        if not tiled:
            strides=(inner*cols,1,inner) if transpose else (inner*cols,cols,1)
            b.batched_matmul(a,v,c,out,rows,cols,inner,(rows*inner,inner,1),strides,alpha,beta)
        return out
    r=multiply(x,x,x,m,m,k,transpose=True);q=None
    b.release_since(mark,keep=(x,r))
    for iteration in range(5):
        if iteration==3:
            x=multiply(q,x,x,m,k,m)
            r=multiply(x,x,x,m,m,k,transpose=True);q=None
        z=multiply(r,r,r,m,m,m,alpha=2.0315,beta=-4.775,symmetric=True)
        if q is None:
            q=b.empty(z.shape,'fp16')
            b.launch(add_identity,(triton.cdiv(q.numel,1024),),(z,q),M=m,N=q.numel)
        else:q=multiply(q,z,q,m,m,m,beta=3.4445,symmetric=True)
        if iteration not in (2,4):
            rz=multiply(r,z,r,m,m,m,beta=3.4445,symmetric=True)
            r=multiply(z,rz,rz,m,m,m,beta=3.4445,symmetric=True)
        b.release_since(mark,keep=(x,r,q))
    out=e.ops.cast(multiply(q,x,x,m,k,m),'bf16')
    b.release_since(mark,keep=(out,))
    return out
