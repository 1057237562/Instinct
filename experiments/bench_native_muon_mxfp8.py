"""Screen early MXFP8 Newton-Schulz iterations on saved training momentum.

No optimizer or weight updates. BF16 five-iteration Muon is the reference;
all variants retain five iterations and use the existing BF16 polynomial.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import triton
import triton.language as tl
from experiments.smoke_dense_triton import RejectTorch


@triton.jit
def affine_update(P,X,Y,N:tl.constexpr,B:tl.constexpr=1024):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    p=tl.load(P+i,valid,0).to(tl.float32);x=tl.load(X+i,valid,0).to(tl.float32)
    tl.store(Y+i,tl.fma(3.4445,x,p),valid)


def matrix(t,index):
    from model.dense_triton.runtime import Tensor,ITEMSIZE
    _,m,k=t.shape
    return Tensor((m,k),t.dtype,t.storage+index*m*k*ITEMSIZE[t.dtype],t.owner)


def mx_iteration(plan,x):
    from model.dense_cublas import mxfp8
    e=plan.opt.e;b=e.backend;ops=e.ops;batch,m,k=x.shape
    gram=b.empty((batch,m,m),'bf16');quant=[]
    for i in range(batch):
        q,s=mxfp8.quantize(ops,matrix(x,i),dual_layout=True);quant.append(q)
        b.matmul(q,q,matrix(gram,i),s,s,m,m,k,role='muon_mx_gram')
    poly=plan.gemm(gram,gram,gram,m,m,m,(m*m,m,1),(m*m,m,1),alpha=2.0315,beta=-4.775,symmetric=True)
    product=b.empty((batch,m,k),'fp32');out=b.empty(x.shape,'bf16')
    for i,q in enumerate(quant):
        p,s=mxfp8.quantize(ops,matrix(poly,i));qt=q.transpose_data
        b.matmul(qt,p,matrix(product,i),qt.mx32_scale,s,k,m,m,role='muon_mx_update')
    b.launch(affine_update,(triton.cdiv(x.numel,1024),),(product,x,out),N=x.numel)
    return out


@triton.jit
def add_diagonal(Z,Q,M:tl.constexpr,N:tl.constexpr,B:tl.constexpr=1024):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    z=tl.load(Z+i,valid,0).to(tl.float32)
    tl.store(Q+i,z+tl.where((i//M)%M==i%M,3.4445,0.),valid)


def gram_chain(plan,x,dtype,restart):
    """Native experiment with the Gram-NS recurrence and one stability restart.

    Algorithm reference: https://github.com/Dao-AILab/gram-newton-schulz
    Retain this repository's coefficients, five steps and input normalization;
    do not import the reference's PyTorch backend or Polar Express coefficients.
    """
    from model.dense_cublas.muon import ns_gemm
    e=plan.opt.e;b=e.backend;batch,m,k=x.shape;x=e.ops.cast(x,dtype)
    def mm(a,v,c,rows,cols,inner,transpose=False,alpha=1.,beta=0.,symmetric=False):
        astrides=(rows*inner,inner,1)
        bstrides=(inner*cols,1,inner) if transpose else (inner*cols,cols,1)
        if dtype=='bf16':return plan.gemm(a,v,c,rows,cols,inner,astrides,bstrides,alpha,beta,symmetric)
        out=b.empty((batch,rows,cols),dtype)
        if symmetric and a is not v:
            from model.dense_cublas.muon_gram import gram
            if gram(b,v,out,tile=64,other=a,addend=c,alpha=alpha,beta=beta):return out
        if a is v and ((transpose and beta==0.) or (symmetric and alpha==2.0315 and beta==-4.775)):
            from model.dense_cublas.muon_gram import gram
            if gram(b,a,out,tile=64,polynomial=not transpose):return out
        if symmetric:
            tiles=triton.cdiv(rows,64)
            b.launch(ns_gemm,(tiles*(tiles+1)//2,1,batch),(a,v,c,out),
                M=rows,N=cols,K=inner,A0=astrides[0],A1=astrides[1],A2=astrides[2],
                B0=bstrides[0],B1=bstrides[1],B2=bstrides[2],C0=rows*cols,C1=cols,C2=1,
                ALPHA=alpha,BETA=beta,BM=64,BN=64,BK=128,SYMMETRIC=True,TRANSPOSE_OUTPUT=False)
        else:b.batched_matmul(a,v,c,out,rows,cols,inner,astrides,bstrides,alpha,beta)
        return out
    r=mm(x,x,x,m,m,k,transpose=True,symmetric=False);q=None
    for iteration in range(5):
        if iteration==restart:
            x=mm(q,x,x,m,k,m)
            r=mm(x,x,x,m,m,k,transpose=True,symmetric=False);q=None
        z=mm(r,r,r,m,m,m,alpha=2.0315,beta=-4.775,symmetric=True)
        if q is None:
            q=b.empty(z.shape,dtype)
            b.launch(add_diagonal,(triton.cdiv(q.numel,1024),),(z,q),M=m,N=q.numel)
        else:q=mm(q,z,q,m,m,m,beta=3.4445,symmetric=True)
        if iteration<4 and iteration+1!=restart:
            rz=mm(r,z,r,m,m,m,beta=3.4445,symmetric=True)
            r=mm(z,rz,rz,m,m,m,beta=3.4445,symmetric=True)
    return e.ops.cast(mm(q,x,x,m,k,m),'bf16')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',required=True);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--gram',action='store_true',help='Screen a Gram recurrence with a single restart instead of MX GEMMs')
    p.add_argument('--integrated',action='store_true',help='All FFN roles and tail batch; validate the bounded-memory training implementation')
    p.add_argument('--fp32-reference',action='store_true',help='Also compare NS outputs against FP32 GEMMs on the identical normalized input')
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.muon import MuonPlan,norm_partials,norm_finish,normalize
    from model.dense_cublas.policy import enable_fused_ops
    report=dict(status='running',source=args.resume,iterations=5,cases=[],scope='Frozen saved momentum; NS only, no optimizer updates')
    with np.load(args.resume,allow_pickle=False) as checkpoint, TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=True)) as b:
        enable_fused_ops(b);b.muon_gemm_backend='cublas';e=Engine(b);plan=MuonPlan(SimpleNamespace(e=e,ns_steps=5))
        sources=((f'mlp.{name}_proj.weight',indices) for name in ('gate','up','down') for indices in ([0,15,31],[31])) if args.integrated else (('mlp.gate_proj.weight',[0,15,31]),('self_attn.q_proj.weight',[0,4,8,12,16,20,24,31]))
        for suffix,indices in sources:
            values=[]
            for layer in indices:
                value=checkpoint[f'optimizer/model/model.layers.{layer}.{suffix}/momentum']
                if value.dtype==np.uint16:value=(value.astype(np.uint32)<<16).view(np.float32)
                if value.shape[0]>value.shape[1]:value=value.T
                values.append(value)
            mark=b.mark();u=e.array(np.stack(values),'bf16').data;batch,m,k=u.shape
            count=triton.cdiv(m*k,1024);partial=b.empty((batch,count),'fp32');norm=b.empty((batch,),'bf16');x=b.empty(u.shape,'bf16')
            b.launch(norm_partials,(count,batch),(u,partial),N=m*k,COUNT=count)
            b.launch(norm_finish,(batch,),(partial,norm),COUNT=count,B=triton.next_power_of_2(count))
            b.launch(normalize,(count,batch),(u,norm,x),N=m*k)
            b.synchronize();b.release_since(mark,keep=(x,));reference=None
            oracle=None
            if args.fp32_reference:
                local=b.mark();f=e.ops.cast(x,'fp32')
                for _ in range(5):
                    g=b.empty((batch,m,m),'fp32');p=b.empty(g.shape,'fp32');y=b.empty(f.shape,'fp32')
                    b.batched_matmul(f,f,f,g,m,m,k,(m*k,k,1),(m*k,1,k))
                    b.batched_matmul(g,g,g,p,m,m,m,(m*m,m,1),(m*m,m,1),2.0315,-4.775)
                    b.batched_matmul(p,f,f,y,m,k,m,(m*m,m,1),(m*k,k,1),1.,3.4445)
                    f=y;b.release_since(local,keep=(f,))
                oracle=b.to_numpy(f).astype(np.float64).ravel();b.release_since(local)
            choices=(0,'integrated') if args.integrated else (0,('bf16',2),('fp16',2),('fp16',3)) if args.gram else (0,1,2,5)
            for mx_steps in choices:
                local=b.mark();b._commands=[]
                try:
                    y=x
                    if mx_steps=='integrated':
                        from model.dense_cublas.muon_recurrence import orthogonalize
                        y=orthogonalize(plan,x,local)
                    if isinstance(mx_steps,tuple):y=gram_chain(plan,x,*mx_steps)
                    for iteration in (() if isinstance(mx_steps,(tuple,str)) else range(5)):
                        if iteration<mx_steps:y=mx_iteration(plan,y)
                        else:
                            gram=plan.gemm(y,y,y,m,m,k,(m*k,k,1),(m*k,1,k),symmetric=(m,k)!=(1536,5504))
                            poly=plan.gemm(gram,gram,gram,m,m,m,(m*m,m,1),(m*m,m,1),alpha=2.0315,beta=-4.775,symmetric=True)
                            y=plan.gemm(poly,y,y,m,k,m,(m*m,m,1),(m*k,k,1),beta=3.4445)
                finally:commands=b._commands;b._commands=None
                def run():
                    for command in commands:command.execute()
                run();value=b.to_numpy(y)
                if reference is None:reference=value
                a=value.astype(np.float64).ravel();r=reference.astype(np.float64).ravel()
                metrics=dict(relative_l2=float(np.linalg.norm(a-r)/np.linalg.norm(r)),cosine=float(np.dot(a,r)/(np.linalg.norm(a)*np.linalg.norm(r))),finite=bool(np.isfinite(value).all()))
                if oracle is not None:
                    metrics['to_fp32_ns']=dict(relative_l2=float(np.linalg.norm(a-oracle)/np.linalg.norm(oracle)),
                        cosine=float(np.dot(a,oracle)/(np.linalg.norm(a)*np.linalg.norm(oracle))))
                b.time_call(run,b.main_stream,repeats=8)
                times=[b.time_call(run,b.main_stream,repeats=5) for _ in range(5)]
                row=dict(shape=list(x.shape),parameter=suffix,layers=indices,variant=mx_steps,median_ms=float(np.median(times)),samples_ms=times,**metrics)
                report['cases'].append(row);print(json.dumps(row),flush=True)
                args.output.write_text(json.dumps(report,indent=2));b.release_since(local)
            b.release_since(mark)
        report['algorithms']=b.tuning_report()
    report.update(status='complete',torch_imported='torch' in sys.modules)
    args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
