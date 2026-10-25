"""Device-only stable MoE dispatch and ragged grouped GEMM with explicit VJPs."""
import triton
import triton.language as tl


@triton.jit
def route_counts(Selected,Counts,T:tl.constexpr,E:tl.constexpr,B:tl.constexpr=256):
    tile=tl.program_id(0);expert=tl.program_id(1);t=tile*B+tl.arange(0,B)
    chosen=tl.load(Selected+t*E+expert,t<T,0)
    tl.store(Counts+expert*tl.cdiv(T,B)+tile,tl.sum(chosen.to(tl.int32),0))


@triton.jit
def route_offsets(Counts,Prefix,Offsets,TILES:tl.constexpr,E:tl.constexpr,B:tl.constexpr=256):
    expert=tl.program_id(0);i=tl.arange(0,B)
    previous=tl.full((B,),0,tl.int32)
    for block in range(tl.cdiv(expert*TILES,B)):
        index=block*B+i
        previous+=tl.load(Counts+index,index<expert*TILES,0)
    base=tl.sum(previous,0);tl.store(Offsets+expert,base)
    for block in range(tl.cdiv(TILES,B)):
        tile=block*B+i;count=tl.load(Counts+expert*TILES+tile,tile<TILES,0)
        prefix=tl.cumsum(count,0)-count+base
        tl.store(Prefix+expert*TILES+tile,prefix,tile<TILES)
        base+=tl.sum(count,0)
    if expert==E-1:tl.store(Offsets+E,base)


@triton.jit
def route_indices(Selected,Prefix,Ids,Routes,T:tl.constexpr,E:tl.constexpr,B:tl.constexpr=256):
    tile=tl.program_id(0);expert=tl.program_id(1);t=tile*B+tl.arange(0,B)
    chosen=tl.load(Selected+t*E+expert,t<T,0).to(tl.int32)
    slot=tl.load(Prefix+expert*tl.cdiv(T,B)+tile)+tl.cumsum(chosen,0)-1
    tl.store(Ids+slot,t,(t<T)&(chosen!=0))
    tl.store(Routes+slot,t*E+expert,(t<T)&(chosen!=0))


@triton.jit
def expert_scale(X,Offsets,Scale,WIDTH:tl.constexpr,WEIGHTS:tl.constexpr,ROWS:tl.constexpr,
                 LIMIT:tl.constexpr,B:tl.constexpr=256):
    expert=tl.program_id(0)
    if WEIGHTS:begin=expert*ROWS;end=begin+ROWS
    else:begin=tl.load(Offsets+expert);end=tl.load(Offsets+expert+1)
    i=tl.arange(0,B);maximum=tl.full((B,),0.,tl.float32)
    for block in range(tl.cdiv((end-begin)*WIDTH,B)):
        offset=block*B+i
        value=tl.load(X+begin*WIDTH+offset,offset<(end-begin)*WIDTH,0).to(tl.float32)
        maximum=tl.maximum(maximum,tl.abs(value))
    tl.store(Scale+expert,LIMIT/tl.maximum(tl.max(maximum,0),1.e-12))


@triton.jit
def expert_quantize(X,Y,Offsets,Scale,WIDTH:tl.constexpr,WEIGHTS:tl.constexpr,ROWS:tl.constexpr,B:tl.constexpr=256):
    expert=tl.program_id(1)
    if WEIGHTS:begin=expert*ROWS;end=begin+ROWS
    else:begin=tl.load(Offsets+expert);end=tl.load(Offsets+expert+1)
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<(end-begin)*WIDTH
    value=tl.load(X+begin*WIDTH+i,valid,0).to(tl.float32)*tl.load(Scale+expert)
    tl.store(Y+begin*WIDTH+i,value,valid)


@triton.jit
def grouped_mm(X,W,Y,Offsets,XS,WS,FP8:tl.constexpr,N:tl.constexpr,K:tl.constexpr,TRANSPOSE_W:tl.constexpr=False,
               BM:tl.constexpr=16,BN:tl.constexpr=32,BK:tl.constexpr=32):
    expert=tl.program_id(2);begin=tl.load(Offsets+expert);end=tl.load(Offsets+expert+1)
    row=tl.program_id(0)*BM+tl.arange(0,BM)+begin;col=tl.program_id(1)*BN+tl.arange(0,BN)
    if tl.program_id(0)*BM<end-begin:
        acc=tl.full((BM,BN),0,tl.float32)
        for block in range(tl.cdiv(K,BK)):
            k=block*BK+tl.arange(0,BK)
            a=tl.load(X+row[:,None]*K+k[None,:],(row[:,None]<end)&(k[None,:]<K),0.)
            if TRANSPOSE_W:ptr=W+expert*K*N+k[:,None]*N+col[None,:]
            else:ptr=W+expert*N*K+col[None,:]*K+k[:,None]
            b=tl.load(ptr,(k[:,None]<K)&(col[None,:]<N),0.)
            acc+=tl.dot(a,b,input_precision='ieee')
        if FP8:acc=acc/(tl.load(XS+expert)*tl.load(WS+expert))
        tl.store(Y+row[:,None]*N+col[None,:],acc,(row[:,None]<end)&(col[None,:]<N))


@triton.jit
def grouped_dw(X,DY,DW,Offsets,XS,DS,FP8:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr=16,BK:tl.constexpr=32,BT:tl.constexpr=32):
    expert=tl.program_id(2);begin=tl.load(Offsets+expert);end=tl.load(Offsets+expert+1)
    n=tl.program_id(0)*BN+tl.arange(0,BN);k=tl.program_id(1)*BK+tl.arange(0,BK)
    acc=tl.full((BN,BK),0,tl.float32)
    for block in range(tl.cdiv(end-begin,BT)):
        row=begin+block*BT+tl.arange(0,BT)
        a=tl.load(DY+row[None,:]*N+n[:,None],(row[None,:]<end)&(n[:,None]<N),0.)
        b=tl.load(X+row[:,None]*K+k[None,:],(row[:,None]<end)&(k[None,:]<K),0.)
        acc+=tl.dot(a,b,input_precision='ieee')
    if FP8:acc=acc/(tl.load(XS+expert)*tl.load(DS+expert))
    tl.store(DW+expert*N*K+n[:,None]*K+k[None,:],acc,(n[:,None]<N)&(k[None,:]<K))


def dispatch(e, selected, topk):
    t,experts=selected.shape;tiles=triton.cdiv(t,256)
    counts=e.backend.empty((experts*tiles,),'i32');prefix=e.backend.empty(counts.shape,'i32')
    offsets=e.backend.empty((experts+1,),'i32')
    ids=e.backend.empty((t*topk,),'i64');routes=e.backend.empty(ids.shape,'i64')
    e.backend.launch(route_counts,(tiles,experts),(selected.data,counts),T=t,E=experts)
    e.backend.launch(route_offsets,(experts,),(counts,prefix,offsets),TILES=tiles,E=experts,B=256)
    e.backend.launch(route_indices,(tiles,experts),(selected.data,prefix,ids,routes),T=t,E=experts)
    return e.wrap(ids),e.wrap(routes),offsets


def linear(e,x,weight,offsets,max_tokens):
    experts,n,k=weight.shape;fp8=e.ops.fp8
    w=weight if fp8 else e.cast(weight,x.dtype);out=e.backend.empty((x.shape[0],n),x.dtype)
    def quantize(value,width,rows,weights=False,gradient=False):
        if not fp8:return value,value
        scale=e.backend.empty((experts,),'fp32')
        quant=e.backend.empty(value.shape,'fp8e5' if gradient else 'fp8e4nv')
        options=dict(WIDTH=width,ROWS=rows,WEIGHTS=weights)
        e.backend.launch(expert_scale,(experts,),(value,offsets,scale),LIMIT=57344. if gradient else 448.,**options)
        e.backend.launch(expert_quantize,(triton.cdiv(rows*width,256),experts),(value,quant,offsets,scale),**options)
        return quant,scale
    qx,sx=quantize(x.data,k,max_tokens);qw,sw=quantize(w.data,k,n,True)
    e.backend.launch(grouped_mm,(triton.cdiv(max_tokens,16),triton.cdiv(n,32),experts),(qx,qw,out,offsets,sx,sw),N=n,K=k,FP8=fp8)
    def backward(g):
        dy=e.ops.cast(g,x.dtype);qdy,sdy=quantize(dy,n,max_tokens,gradient=True)
        dx=e.backend.empty(x.shape,x.dtype);dw=e.backend.empty(weight.shape,weight.dtype)
        e.backend.launch(grouped_mm,(triton.cdiv(max_tokens,16),triton.cdiv(k,32),experts),(qdy,qw,dx,offsets,sdy,sw),N=k,K=n,TRANSPOSE_W=True,FP8=fp8)
        e.backend.launch(grouped_dw,(triton.cdiv(n,16),triton.cdiv(k,32),experts),(qx,qdy,dw,offsets,sx,sdy),N=n,K=k,FP8=fp8)
        return dx,dw
    return e._node(out,(x,weight),backward)


def feed_forward(model,flat,weights,selected,prefix):
    e,c=model.e,model.config
    ids,routes,offsets=dispatch(e,selected,c['num_experts_per_tok'])
    tokens=e.embedding(flat,ids);scale=e.embedding(weights.reshape(-1,1),routes)
    def project(x,name):
        values=[]
        for expert in range(c['num_experts']):
            key=prefix+f'experts.{expert}.'+name+'.weight';w=model.params[key]
            if key in model.adapters:
                a,b=model.adapters[key];w=w+b@a
            values.append(w.reshape(1,*w.shape))
        return linear(e,x,e.cat(values),offsets,flat.shape[0])
    gate,up=project(tokens,'gate_proj'),project(tokens,'up_proj')
    y=project(e.unary(gate,c['hidden_act'])*up,'down_proj')
    return e.scatter_rows((y*scale).cast('fp32'),ids,flat.shape[0])
