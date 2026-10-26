"""Shape-batched split-weight Muon, without a Torch runtime dependency."""
import math
import numpy as np
import triton
import triton.language as tl
from model.dense_triton.runtime import Scalar
from .multi_tensor import parameter_pointer


def hidden_matrix(name, shape):
    return len(shape) == 2 and 'embed_tokens' not in (name or '') and 'lm_head' not in (name or '')


@triton.jit
def lerp(a, b, weight):
    return tl.where(tl.abs(weight) < .5, tl.fma(weight, b-a, a),
                    tl.fma(-(1.-weight), b-a, b))


@triton.jit
def prepare(Table, U, scale, ROWS:tl.constexpr, COLS:tl.constexpr, MOM:tl.constexpr,
            MOM_DTYPE:tl.constexpr, B:tl.constexpr=1024):
    matrix=tl.program_id(1);i=tl.program_id(0)*B+tl.arange(0,B);valid=i<ROWS*COLS
    row=Table+matrix*4
    grad=tl.load(row+1).to(tl.pointer_type(tl.bfloat16))
    momentum=parameter_pointer(tl.load(row+2),MOM_DTYPE)
    g=(tl.load(grad+i,valid,0).to(tl.float32)*scale).to(tl.bfloat16).to(tl.float32)
    old=tl.load(momentum+i,valid,0).to(tl.float32)
    updated=lerp(old,g,tl.full((),1.-MOM,tl.float32))
    tl.store(momentum+i,updated,valid)
    value=lerp(g,updated,tl.full((),MOM,tl.float32))
    offset=(i%COLS)*ROWS+i//COLS if ROWS>COLS else i
    tl.store(U+matrix*ROWS*COLS+offset,value,valid)


@triton.jit
def norm_partials(U, P, N:tl.constexpr, COUNT:tl.constexpr, B:tl.constexpr=1024):
    matrix=tl.program_id(1);block=tl.program_id(0);i=block*B+tl.arange(0,B)
    u=tl.load(U+matrix*N+i,i<N,0).to(tl.float32)
    tl.store(P+matrix*COUNT+block,tl.sum(u*u,0))


@triton.jit
def norm_finish(P, NORM, COUNT:tl.constexpr, B:tl.constexpr):
    matrix=tl.program_id(0);i=tl.arange(0,B)
    n=tl.sqrt(tl.sum(tl.load(P+matrix*COUNT+i,i<COUNT,0),0)).to(tl.bfloat16)
    tl.store(NORM+matrix,tl.maximum(n,tl.full((),1.e-7,tl.float32).to(tl.bfloat16)))


@triton.jit
def normalize(U,NORM,O,N:tl.constexpr,B:tl.constexpr=1024):
    matrix=tl.program_id(1);i=tl.program_id(0)*B+tl.arange(0,B)
    value=tl.load(U+matrix*N+i,i<N,0).to(tl.float32)
    norm=tl.load(NORM+matrix).to(tl.float32)
    tl.store(O+matrix*N+i,value/norm,i<N)


@triton.jit
def apply(Table,U,decay,alpha,ROWS:tl.constexpr,COLS:tl.constexpr,B:tl.constexpr=1024):
    matrix=tl.program_id(1);i=tl.program_id(0)*B+tl.arange(0,B);valid=i<ROWS*COLS
    row=Table+matrix*4
    high=tl.load(row).to(tl.pointer_type(tl.bfloat16))
    low=tl.load(row+3).to(tl.pointer_type(tl.uint16))
    hi=tl.load(high+i,valid,0).to(tl.uint16,bitcast=True).to(tl.uint32)
    lo=tl.load(low+i,valid,0).to(tl.uint32)
    weight=((hi<<16)|lo).to(tl.float32,bitcast=True)
    offset=(i%COLS)*ROWS+i//COLS if ROWS>COLS else i
    update=tl.load(U+matrix*ROWS*COLS+offset,valid,0).to(tl.float32)
    precise=tl.fma(update,alpha,weight*decay)
    bits=precise.to(tl.uint32,bitcast=True)
    tl.store(high+i,(bits>>16).to(tl.uint16).to(tl.bfloat16,bitcast=True),valid)
    tl.store(low+i,(bits&0xffff).to(tl.uint16),valid)


class MuonPlan:
    def __init__(self, optimizer):
        self.opt=optimizer
        self.tables={}

    def chunk_size(self, p):
        # Same conservative target as the PyTorch recipe; one matrix minimum.
        per_matrix=18*p.size+8*min(p.shape)**2
        return max(1,min(self.opt.muon_batch_size,int(self.opt.muon_workspace_mb*2**20)//per_matrix))

    def step(self, parameters, lr, scale):
        opt=self.opt;e=opt.e;b=e.backend
        buckets={}
        for p in parameters:buckets.setdefault(p.shape,[]).append(p)
        for shape,params in buckets.items():
            size=self.chunk_size(params[0])
            for begin in range(0,len(params),size):
                chunk=params[begin:begin+size]
                key=tuple(p.data.storage if isinstance(p.data.storage,int) else id(p.data.storage) for p in chunk)
                if key not in self.tables:
                    rows=[[p.data.storage,opt.gradients[p].data.storage,opt.state[p]['momentum'].data.storage,
                           opt.state[p]['mantissa'].data.storage] for p in chunk]
                    self.tables[key]=e.array(np.asarray(rows,np.int64)).data
                    # Pointer tables are optimizer-owned, even when first made inside step().
                    if hasattr(b,'_pinned'):b._pinned.add(self.tables[key].storage)
                table=self.tables[key];mark=b.mark()
                batch=len(chunk);r,c=shape;m,k=min(r,c),max(r,c);n=r*c
                u=b.empty((batch,m,k),'bf16')
                b.launch(prepare,(triton.cdiv(n,1024),batch),(table,u,Scalar(scale)),
                         ROWS=r,COLS=c,MOM=opt.momentum,MOM_DTYPE=opt.muon_momentum_dtype)
                count=triton.cdiv(n,1024);partial=b.empty((batch,count),'fp32');norm=b.empty((batch,),'bf16')
                b.launch(norm_partials,(count,batch),(u,partial),N=n,COUNT=count)
                b.launch(norm_finish,(batch,),(partial,norm),COUNT=count,B=triton.next_power_of_2(count))
                x=b.empty(u.shape,'bf16')
                b.launch(normalize,(count,batch),(u,norm,x),N=n)
                b.release_since(mark,keep=(x,))
                for _ in range(opt.ns_steps):
                    symmetric=(m,k)!=(1536,5504) and (batch,m,k)!=(1,384,1536)
                    gram=self.gemm(x,x,x,m,m,k,(m*k,k,1),(m*k,1,k),beta=0.,symmetric=symmetric)
                    polynomial=self.gemm(gram,gram,gram,m,m,m,(m*m,m,1),(m*m,m,1),alpha=2.0315,beta=-4.775,symmetric=True)
                    next_x=self.gemm(polynomial,x,x,m,k,m,(m*m,m,1),(m*k,k,1),beta=3.4445)
                    x=next_x
                    b.release_since(mark,keep=(x,))
                alpha=-lr*.2*math.sqrt(max(shape))
                b.launch(apply,(count,batch),(table,x,Scalar(1.-lr*opt.weight_decay),Scalar(alpha)),ROWS=r,COLS=c)
                b.release_since(mark)
                for p in chunk:opt.state[p]['step']+=1

    def gemm(self,a,b,c,m,n,k,astrides,bstrides,alpha=1.,beta=0.,symmetric=False):
        backend=self.opt.e.backend;batch=a.shape[0]
        out=backend.empty((batch,m,n),'bf16')
        if getattr(backend,'muon_gemm_backend','triton')=='cublas':
            backend.batched_matmul(a,b,c,out,m,n,k,astrides,bstrides,alpha,beta)
            return out
        tiles=triton.cdiv(m,64)
        grid=(tiles*(tiles+1)//2,1,batch) if symmetric else (tiles,triton.cdiv(n,64),batch)
        backend.launch(ns_gemm,grid,(a,b,c,out),
                       M=m,N=n,K=k,A0=astrides[0],A1=astrides[1],A2=astrides[2],
                       B0=bstrides[0],B1=bstrides[1],B2=bstrides[2],
                       C0=m*n,C1=n,C2=1,ALPHA=alpha,BETA=beta,BM=64,BN=64,BK=32,
                       SYMMETRIC=symmetric,TRANSPOSE_OUTPUT=False)
        return out


@triton.jit
def ns_gemm(A, B, C, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr,
          B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr,
          C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
          ALPHA: tl.constexpr, BETA: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SYMMETRIC: tl.constexpr,
          TRANSPOSE_OUTPUT: tl.constexpr):
    if SYMMETRIC:
        tile = tl.program_id(0)
        row_tile = ((tl.sqrt(8. * tile + 1.) - 1.) * 0.5).to(tl.int32)
        col_tile = tile - row_tile * (row_tile + 1) // 2
    else:
        row_tile, col_tile = tl.program_id(0), tl.program_id(1)
    row = row_tile * BM + tl.arange(0, BM)
    col = col_tile * BN + tl.arange(0, BN)
    batch = tl.program_id(2)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        a = tl.load(A + batch*A0 + row[:, None]*A1 + k[None, :]*A2,
                    (row[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(B + batch*B0 + k[:, None]*B1 + col[None, :]*B2,
                    (k[:, None] < K) & (col[None, :] < N), 0)
        acc = tl.dot(a, b, acc)
    result = ALPHA * acc
    if BETA != 0:
        c = tl.load(C + batch*C0 + row[:, None]*C1 + col[None, :]*C2,
                    (row[:, None] < M) & (col[None, :] < N), 0).to(tl.float32)
        result = tl.fma(ALPHA, acc, BETA * c)
    out_offset = ((batch*N + col[None, :])*M + row[:, None] if TRANSPOSE_OUTPUT
                  else (batch*M + row[:, None])*N + col[None, :])
    tl.store(O + out_offset, result,
             (row[:, None] < M) & (col[None, :] < N))
    if SYMMETRIC and row_tile != col_tile:
        tl.store(O + (batch*M + col[None, :])*N + row[:, None], result,
                 (row[:, None] < M) & (col[None, :] < N))
