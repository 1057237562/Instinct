"""Static pointer-table AdamW/norm launches for homogeneous native parameters.

The table references persistent parameters/optimizer state only. Arbitrary
parameter groups, mixed dtypes and reference backends retain the scalar path.
"""
import numpy as np
import triton
import triton.language as tl
from model.dense_triton.runtime import Scalar


@triton.jit
def parameter_pointer(address, DTYPE: tl.constexpr):
    if DTYPE == 'bf16': return address.to(tl.pointer_type(tl.bfloat16))
    elif DTYPE == 'fp16': return address.to(tl.pointer_type(tl.float16))
    else: return address.to(tl.pointer_type(tl.float32))


@triton.jit
def norm_multi(Table, Blocks, Out, scale, B: tl.constexpr=1024):
    block=tl.program_id(0);parameter=tl.load(Blocks+block)
    meta=Table+parameter*7
    n=tl.load(meta+4);start=tl.load(meta+5)
    i=(block-start)*B+tl.arange(0,B)
    grad=tl.load(meta+1).to(tl.pointer_type(tl.float32))
    g=tl.load(grad+i,i<n,0)*scale
    tl.store(Out+block,tl.sum(g*g,0))


@triton.jit
def adamw_multi(Table, Blocks, Steps, Control, Stats, lr, decay, b1, b2, bc1, bc2, eps, scale,
                DTYPE: tl.constexpr, GRAPH: tl.constexpr, B: tl.constexpr=1024):
    block=tl.program_id(0);parameter=tl.load(Blocks+block)
    meta=Table+parameter*7
    n=tl.load(meta+4);start=tl.load(meta+5);i=(block-start)*B+tl.arange(0,B);valid=i<n
    p=parameter_pointer(tl.load(meta),DTYPE)
    gptr=tl.load(meta+1).to(tl.pointer_type(tl.float32))
    mptr=tl.load(meta+2).to(tl.pointer_type(tl.float32))
    vptr=tl.load(meta+3).to(tl.pointer_type(tl.float32))
    update=True;clear=True
    if GRAPH:
        update=tl.load(Stats+3)!=0;clear=tl.load(Stats+4)!=0
    if update:
        if GRAPH:
            index=tl.load(meta+6);step=tl.load(Steps+index)+1
            lr=tl.load(Control);scale=tl.load(Stats+1)
            bc1=1.-tl.exp(tl.log(b1)*step);bc2=1.-tl.exp(tl.log(b2)*step)
        g=tl.load(gptr+i,valid,0)*scale
        m=b1*tl.load(mptr+i,valid,0)+(1.-b1)*g
        v=b2*tl.load(vptr+i,valid,0)+(1.-b2)*g*g
        value=tl.load(p+i,valid,0).to(tl.float32)*(1.-lr*decay)-lr*(m/bc1)/(tl.sqrt(v/bc2)+eps)
        tl.store(p+i,value,valid);tl.store(mptr+i,m,valid);tl.store(vptr+i,v,valid)
    if clear:tl.store(gptr+i,0.,valid)


class MultiTensorPlan:
    block=1024

    @classmethod
    def create(cls,optimizer):
        params=optimizer.parameters
        if optimizer.kind!='adamw' or not params or len({p.dtype for p in params})!=1:return None
        tensors=[t.data for p in params for t in (p,optimizer.gradients[p],optimizer.state[p]['m'],optimizer.state[p]['v'])]
        if not all(isinstance(t.storage,int) for t in tensors):return None
        return cls(optimizer)

    def __init__(self,optimizer):
        self.e=optimizer.e;self.dtype=optimizer.parameters[0].dtype
        rows=[];mapping=[];start=0
        for index,p in enumerate(optimizer.parameters):
            state=optimizer.state[p];count=triton.cdiv(p.size,self.block)
            rows.append([p.data.storage,optimizer.gradients[p].data.storage,state['m'].data.storage,
                         state['v'].data.storage,p.size,start,index])
            mapping.extend([index]*count);start+=count
        self.table=self.e.array(np.asarray(rows,np.int64)).data
        self.blocks=self.e.array(np.asarray(mapping,np.int64)).data
        self.count=start

    def squared_norm(self,scale):
        out=self.e.backend.empty((self.count,),'fp32')
        self.e.backend.launch(norm_multi,(self.count,),(self.table,self.blocks,out,Scalar(scale)),B=self.block)
        return self.e.sum_partials(out)

    def update(self,optimizer,lr,scale,step=None,steps=None,control=None,stats=None):
        graph=steps is not None;b1,b2=optimizer.betas
        placeholders=(self.table,)*3 if not graph else (steps,control,stats)
        scalars=(lr,optimizer.weight_decay,b1,b2,1. if graph else 1.-b1**step,
                 1. if graph else 1.-b2**step,optimizer.eps,scale)
        self.e.backend.launch(adamw_multi,(self.count,),
            (self.table,self.blocks,*placeholders,*(Scalar(x) for x in scalars)),
            DTYPE=self.dtype,GRAPH=graph,B=self.block)
