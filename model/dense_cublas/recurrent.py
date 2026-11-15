"""Gated DeltaNet: causal depthwise convolution and chunked recurrent VJP.

FP32 state arithmetic. Chunk boundaries retain only the recurrent state; a
checkpoint recomputes the intra-chunk states for backward (bounded by chunk=16).
"""
import math
import triton
import triton.language as tl


@triton.jit
def delta_forward(Q,K,V,G,B,S0,Y,SAVED,LAST,T:tl.constexpr,H:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,
                  BK:tl.constexpr,BV:tl.constexpr):
    bh=tl.program_id(0); batch,head=bh//H,bh%H
    ki=tl.arange(0,BK);vi=tl.arange(0,BV)
    valid=(ki[:,None]<DK)&(vi[None,:]<DV)
    off=bh*DK*DV+ki[:,None]*DV+vi[None,:]
    state=tl.load(S0+off,valid,0).to(tl.float32)
    for t in range(T):
        row=(batch*T+t)*H+head
        q=tl.load(Q+row*DK+ki,ki<DK,0).to(tl.float32)
        k=tl.load(K+row*DK+ki,ki<DK,0).to(tl.float32)
        v=tl.load(V+row*DV+vi,vi<DV,0).to(tl.float32)
        decay=tl.exp(tl.load(G+row).to(tl.float32));beta=tl.load(B+row).to(tl.float32)
        tl.store(SAVED+(row*DK+ki[:,None])*DV+vi[None,:],state,valid)
        state=state*decay
        predicted=tl.sum(state*k[:,None],0)
        delta=beta*(v-predicted)
        state=state+k[:,None]*delta[None,:]
        out=tl.sum(state*q[:,None],0)*(DK**-0.5)
        tl.store(Y+row*DV+vi,out,vi<DV)
    tl.store(LAST+off,state,valid)


@triton.jit
def delta_backward(Q,K,V,G,B,SAVED,DY,DLAST,DQ,DKOUT,DVOUT,DG,DB,DS0,
                   T:tl.constexpr,H:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    bh=tl.program_id(0);batch,head=bh//H,bh%H
    ki=tl.arange(0,BK);vi=tl.arange(0,BV)
    valid=(ki[:,None]<DK)&(vi[None,:]<DV)
    off=bh*DK*DV+ki[:,None]*DV+vi[None,:]
    carry=tl.load(DLAST+off,valid,0).to(tl.float32)
    for rev in range(T):
        t=T-1-rev;row=(batch*T+t)*H+head
        q=tl.load(Q+row*DK+ki,ki<DK,0).to(tl.float32)
        k=tl.load(K+row*DK+ki,ki<DK,0).to(tl.float32)
        v=tl.load(V+row*DV+vi,vi<DV,0).to(tl.float32)
        prev=tl.load(SAVED+(row*DK+ki[:,None])*DV+vi[None,:],valid,0).to(tl.float32)
        decay=tl.exp(tl.load(G+row).to(tl.float32));beta=tl.load(B+row).to(tl.float32)
        reduced=prev*decay;pred=tl.sum(reduced*k[:,None],0)
        delta=beta*(v-pred);state=reduced+k[:,None]*delta[None,:]
        dy=tl.load(DY+row*DV+vi,vi<DV,0).to(tl.float32)*(DK**-0.5)
        ds=carry+q[:,None]*dy[None,:]
        dq=tl.sum(state*dy[None,:],1)
        dd=tl.sum(ds*k[:,None],0);dv=dd*beta
        dk=tl.sum(ds*delta[None,:]-reduced*dv[None,:],1)
        dr=ds-k[:,None]*dv[None,:]
        dg=tl.sum(tl.sum(dr*reduced,1),0)
        db=tl.sum(dd*(v-pred),0)
        carry=dr*decay
        tl.store(DQ+row*DK+ki,dq,ki<DK);tl.store(DKOUT+row*DK+ki,dk,ki<DK)
        tl.store(DVOUT+row*DV+vi,dv,vi<DV);tl.store(DG+row,dg);tl.store(DB+row,db)
    tl.store(DS0+off,carry,valid)


def recurrence(e,q,k,v,g,beta,state):
    b,t,h,dk=q.shape;dv=v.shape[-1]
    kwargs=dict(T=t,H=h,DK=dk,DV=dv,BK=triton.next_power_of_2(dk),BV=triton.next_power_of_2(dv))
    y=e.backend.empty(v.shape,'fp32');last=e.backend.empty(state.shape,'fp32')
    saved=e.backend.empty((b,t,h,dk,dv),'fp32')
    inputs=(q,k,v,g,beta,state)
    e.backend.launch(delta_forward,(b*h,),tuple(x.data for x in inputs)+(y,saved,last),**kwargs)
    # Concatenate once to expose a single VJP to chunk checkpointing.
    joined=e.cat([e.wrap(y).reshape(-1),e.wrap(last).reshape(-1)])
    def backward(djoined):
        grad=e.wrap(djoined)
        dy,ds=grad[:v.size].reshape(*v.shape),grad[v.size:].reshape(*state.shape)
        gradients=[e.backend.empty(x.shape,'fp32') for x in inputs]
        e.backend.launch(delta_backward,(b*h,),tuple(x.data for x in inputs[:5])+(saved,dy.data,ds.data,*gradients),**kwargs)
        return tuple(gradients)
    return e._node(joined.data,inputs,backward)


def linear_attention(model,x,i,cache,new_cache,key):
    e,c,p=model.e,model.config,f'model.layers.{i}.linear_attn.'
    b,t,_=x.shape;kh,vh,dk,dv=(c[k] for k in ('linear_num_key_heads','linear_num_value_heads','linear_key_head_dim','linear_value_head_dim'))
    width,kernel=2*kh*dk+vh*dv,c['linear_conv_kernel_dim']
    raw=model.linear(x,p+'in_proj_qkv.weight').cast('fp32')
    past=None if cache is None else cache.get(key)
    old=e.full((b,kernel-1,width),0.) if past is None and kernel>1 else (None if kernel==1 else past[0])
    history=raw if old is None else e.cat([old,raw],1)
    weight=model.params[p+'conv1d.weight'].reshape(width,kernel).cast('fp32')
    conv=sum(history[:,j:j+t,:]*weight[:,j].reshape(1,1,width) for j in range(kernel))
    conv=e.unary(conv,'silu')
    q=conv[:,:,:kh*dk].reshape(b,t,kh,dk);k=conv[:,:,kh*dk:2*kh*dk].reshape(b,t,kh,dk)
    q=q*((q*q).sum(-1,True)+1.e-6).rsqrt();k=k*((k*k).sum(-1,True)+1.e-6).rsqrt()
    if vh!=kh:
        q=e.cat([q[:,:,head:head+1] for head in range(kh) for _ in range(vh//kh)],2)
        k=e.cat([k[:,:,head:head+1] for head in range(kh) for _ in range(vh//kh)],2)
    v=conv[:,:,2*kh*dk:].reshape(b,t,vh,dv)
    beta=model.linear(x,p+'in_proj_b.weight').cast('fp32').sigmoid()
    g=-model.params[p+'A_log'].cast('fp32').exp()*(model.linear(x,p+'in_proj_a.weight').cast('fp32')+model.params[p+'dt_bias']).softplus()
    state=e.full((b,vh,dk,dv),0.) if past is None else past[1]
    outputs=[]
    # Recompute local recurrence states independently of the user checkpoint mode.
    # This prevents O(sequence * heads * DK * DV) saved-state growth.
    for begin in range(0,t,16):
        end=min(begin+16,t);length=end-begin
        pieces=[z[:,begin:end] for z in (q,k,v,g,beta)]+[state]
        packed=e.cat([z.reshape(-1) for z in pieces])
        shapes=[z.shape for z in pieces];sizes=[z.size for z in pieces]
        def chunk(z,shapes=shapes,sizes=sizes):
            args=[];offset=0
            for shape,size in zip(shapes,sizes): args.append(z[offset:offset+size].reshape(*shape));offset+=size
            return recurrence(e,*args)
        joint=e.checkpoint(chunk,packed,[]) if model.training else chunk(packed)
        count=b*length*vh*dv
        outputs.append(joint[:count].reshape(b,length,vh,dv));state=joint[count:].reshape(b,vh,dk,dv)
    if new_cache is not None:
        new_cache[key]=(None if kernel==1 else history[:,-(kernel-1):].detach(),state.detach())
    out=e.cat(outputs,1).cast(x.dtype)
    z=model.linear(x,p+'in_proj_z.weight').reshape(b,t,vh,dv)
    out=model.norm(out,p+'norm.weight')*e.unary(z.cast('fp32'),'silu').cast(x.dtype)
    return model.linear(out.reshape(b,t,vh*dv),p+'out_proj.weight')
