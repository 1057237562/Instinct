"""BF16/FP16 Gram via triangular cuBLASLt pointer-array tiles, then mirror."""
import triton
import triton.language as tl
from .bridge import LtError


@triton.jit
def tables(X,Y,A,B,C,D,V,ADD,M:tl.constexpr,K:tl.constexpr,TILE:tl.constexpr,COUNT:tl.constexpr,PAIRS:tl.constexpr,BLOCK:tl.constexpr,POLYNOMIAL:tl.constexpr=False,GENERAL:tl.constexpr=False):
    i=tl.arange(0,BLOCK);valid=i<COUNT;t=i%PAIRS;matrix=i//PAIRS
    row=((tl.sqrt(8.*t+1.)-1.)*.5).to(tl.int32);col=t-row*(row+1)//2
    tl.store(A+i,X.to(tl.uint64)+2*(matrix*M*K+row*TILE*K),valid)
    tl.store(B+i,(V if GENERAL else X).to(tl.uint64)+2*(matrix*M*K+col*TILE*K),valid)
    offset=2*(matrix*M*M+col*TILE*M+row*TILE)
    tl.store(C+i,(ADD if GENERAL else X if POLYNOMIAL else Y).to(tl.uint64)+offset,valid)
    if POLYNOMIAL or GENERAL:tl.store(D+i,Y.to(tl.uint64)+offset,valid)


@triton.jit
def mirror(X,M:tl.constexpr,TILE:tl.constexpr=32):
    row=tl.program_id(0)*TILE+tl.arange(0,TILE);col=tl.program_id(1)*TILE+tl.arange(0,TILE)
    base=tl.program_id(2)*M*M;valid=(row[:,None]<M)&(col[None,:]<M)&(row[:,None]>col[None,:])
    v=tl.load(X+base+col[None,:]*M+row[:,None],valid,0)
    tl.store(X+base+row[:,None]*M+col[None,:],v,valid)


def gram(b,x,out,tile=512,polynomial=False,other=None,addend=None,alpha=None,beta=None):
    batch,m,k=x.shape
    if tile<1 or x.dtype not in ('bf16','fp16') or out.dtype!=x.dtype or out.shape!=(batch,m,m) or m%tile:
        raise ValueError('blocked Gram requires divisible BF16/FP16 square output')
    for tensor in (x,out):
        b._check_tensor(tensor,check_context=False)
        if tensor.storage%256:raise ValueError('blocked Gram requires aligned base allocations')
    if (tile*k*2)%128 or tile*2%128:raise ValueError('blocked Gram tile pointers must be aligned')
    size=b.config.workspace_mib*2**20;stream=b.stream;work=b.workspaces[stream]
    pairs=(m//tile)*(m//tile+1)//2;count=batch*pairs
    if polynomial and m!=k:raise ValueError('polynomial requires a square symmetric input')
    general=other is not None
    if general and (polynomial or m!=k or other.shape!=x.shape or addend is None or addend.shape!=x.shape
                    or other.dtype!=x.dtype or addend.dtype!=x.dtype or alpha is None or beta is None):
        raise ValueError('symmetric product requires matching square inputs and explicit coefficients')
    if general:
        for t in (other,addend):
            b._check_tensor(t,check_context=False)
            if t.storage%256:raise ValueError('symmetric product requires aligned base allocations')
    key=('blocked_symmetric_product' if general else 'blocked_polynomial' if polynomial else 'blocked_gram',batch,m,k,tile,size,stream,x.dtype)
    unsupported=getattr(b,'_unsupported_blocked_grams',set())
    if key in unsupported or getattr(b.native,'_plan_pointer_gram',None) is None:return False
    if x.dtype=='fp16' and getattr(b.native,'_plan_pointer_gram_ex',None) is None:return False
    if not hasattr(b,'_gram_pointer_tables'):b._gram_pointer_tables={}
    if key not in b._gram_pointer_tables:
        mark=b.mark()
        ap,bp,cp=[b.empty((count,),'i64') for _ in range(3)]
        dp=b.empty((count,),'i64') if polynomial or general else cp
        b.pin_since(mark)
        b._gram_pointer_tables[key]=(ap,bp,cp,dp)
    ap,bp,cp,dp=b._gram_pointer_tables[key]
    # General output is other @ x.T. Callers must establish that it is
    # symmetric; the lower triangle is mirrored, not computed independently.
    b.launch(tables,(1,),(x,out,ap,bp,cp,dp,other or x,addend or x),M=m,K=k,TILE=tile,COUNT=count,PAIRS=pairs,BLOCK=triton.next_power_of_2(count),POLYNOMIAL=polynomial,GENERAL=general)
    if not general:alpha,beta=(2.0315,-4.775) if polynomial else (1.,0.)
    pointers=(ap.storage,bp.storage,cp.storage,dp.storage,alpha,beta,None if work is None else work.storage,size,stream)
    if key not in b._plans:
        try:plan,candidates=b.native.create_pointer_gram(tile,k,m,count,size,dtype=x.dtype)
        except LtError:
            unsupported.add(key);b._unsupported_blocked_grams=unsupported
            return False
        if b.config.tune:
            from .tuning import race
            def measure(candidate,repeats):
                try:
                    b.native.run_batched(plan,candidate['index'],*pointers)
                    return b.time_call(lambda:b.native.run_batched(plan,candidate['index'],*pointers),stream,repeats)
                except LtError as error:
                    candidate['rejected']=str(error);return float('inf')
            selected=race(candidates,measure,warmup_ms=100,drift_limit=.05,retries=1)['index']
        else:selected=0
        record=dict(kind=key[0],shape=[batch,m,k],dtype=x.dtype,tile=tile,selected_index=selected,candidates=candidates)
        b._plans[key]=(plan,selected,record)
    plan,index,_=b._plans[key]
    b._submit('cublasLt:muon_'+key[0],lambda:b.native.run_batched(plan,index,*pointers))
    b.launch(mirror,(triton.cdiv(m,32),triton.cdiv(m,32),batch),(out,),M=m)
    return True
