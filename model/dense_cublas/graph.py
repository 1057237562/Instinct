"""Fixed-shape supervised forward/backward/AdamW CUDA Graph.

Preparation executes kernels with writes to persistent training state disabled.
Every replay reads LR/force-update controls and optimizer counters from device
buffers. Dynamic loss scaling, accumulation, clipping and overflow stay on GPU.
"""
import numpy as np
import triton
import triton.language as tl


@triton.jit
def accumulate(G,A,Control,N:tl.constexpr,B:tl.constexpr=256):
    if tl.load(Control+1)!=0:
        i=tl.program_id(0)*B+tl.arange(0,B)
        tl.store(A+i,tl.load(A+i,i<N,0)+tl.load(G+i,i<N,0),i<N)


@triton.jit
def statistics(Square,Control,Counts,Stats,Clip:tl.constexpr,Accumulation:tl.constexpr,
               LossScaling:tl.constexpr,Growth:tl.constexpr,Backoff:tl.constexpr,Interval:tl.constexpr):
    tl.store(Stats+3,0.);tl.store(Stats+4,0.)
    if tl.load(Control+1)!=0:
        count=tl.load(Counts)+1;tl.store(Counts,count)
        ready=(count>=Accumulation)|(tl.load(Control+2)!=0)
        if ready:
            norm=tl.sqrt(tl.maximum(tl.load(Square),0.));finite=(norm<float('inf'))&(norm==norm)
            scale=tl.load(Stats)
            factor=1./(scale*count)
            if Clip>0:factor*=tl.minimum(1.,Clip/(norm+1.e-6))
            tl.store(Stats+1,factor);tl.store(Stats+2,norm)
            tl.store(Stats+3,finite.to(tl.float32));tl.store(Stats+4,1.)
            tl.store(Counts,0)
            if finite:
                tl.store(Counts+1,tl.load(Counts+1)+1)
                good=tl.load(Counts+2)+1;tl.store(Counts+2,good)
                if LossScaling and good%Interval==0:tl.store(Stats,scale*Growth)
            elif LossScaling:
                tl.store(Stats,scale*Backoff);tl.store(Counts+2,0)


@triton.jit
def adamw(P,G,M,V,Steps,Control,Stats,INDEX:tl.constexpr,N:tl.constexpr,
          Decay:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,Eps:tl.constexpr,
          Maximize:tl.constexpr=False,B:tl.constexpr=256):
    i=tl.program_id(0)*B+tl.arange(0,B);mask=i<N
    if tl.load(Stats+3)!=0:
        # Step is advanced by a separate kernel after all parameter tiles finish.
        step=tl.load(Steps+INDEX)+1;lr=tl.load(Control)
        g=tl.load(G+i,mask,0)*tl.load(Stats+1)
        if Maximize:g=-g
        m=B1*tl.load(M+i,mask,0)+(1.-B1)*g
        v=B2*tl.load(V+i,mask,0)+(1.-B2)*g*g
        bc1=1.-tl.exp(tl.log(B1)*step);bc2=1.-tl.exp(tl.log(B2)*step)
        p=tl.load(P+i,mask,0).to(tl.float32)*(1.-lr*Decay)-lr*(m/bc1)/(tl.sqrt(v/bc2)+Eps)
        tl.store(P+i,p,mask);tl.store(M+i,m,mask);tl.store(V+i,v,mask)
    if tl.load(Stats+4)!=0:tl.store(G+i,0.,mask)


class StepGraph:
    def __init__(self,model,optimizer,batch,loss_fn,accumulation):
        self.model,self.opt,self.e=model,optimizer,model.e
        self.b=self.e.backend;self.shape=tuple(x.shape for x in batch)
        if optimizer.kind!='adamw' or optimizer.scheduler or optimizer.param_options:
            raise ValueError('CUDA Graph currently requires native AdamW without imported per-group schedules')
        if model.config['dropout'] or model.config['model_architecture']=='looped':
            raise ValueError('CUDA Graph requires dropout=0 and fixed-depth model execution')
        if optimizer.distributed and optimizer.distributed.world>1:raise ValueError('CUDA Graph requires a single GPU')
        if not hasattr(self.b,'build'):raise ValueError('CUDA Graph requires the native CUDA backend')
        mark=self.b.mark()
        self.inputs=tuple(self.e.array(np.asarray(x,np.int64)) for x in batch)
        self.control=self.e.array(np.asarray([optimizer.lr,0.,0.],np.float32))
        self.counts=self.e.array(np.asarray([optimizer.accumulated,optimizer.updates,optimizer.good_steps],np.int64))
        self.stats=self.e.array(np.asarray([optimizer.scale,0.,0.,0.,0.],np.float32))
        self.steps=self.e.array(np.asarray([optimizer.state[p]['step'] for p in optimizer.parameters],np.int64))
        def prepare():
            loss=loss_fn(self.inputs)
            grads=self.e.vjp(loss*self.stats[0])
            self.parameters=[p for p in optimizer.parameters if p in grads]
            with self.e.no_grad():
                divisor=1./(self.stats[0]*(self.counts[0].cast('fp32')+1.))
                squares=self.e.full((1,),0.)
                for p in self.parameters:
                    g=optimizer.gradients[p]
                    self.b.launch(accumulate,(triton.cdiv(p.size,256),),(grads[p],g.data,self.control.data),N=p.size)
                    scaled=g*divisor;squares=squares+(scaled*scaled).sum()
                self.b.launch(statistics,(1,),(squares.data,self.control.data,self.counts.data,self.stats.data),
                    Clip=optimizer.grad_clip,Accumulation=accumulation,LossScaling=optimizer.loss_scaling,
                    Growth=optimizer.growth_factor,Backoff=optimizer.backoff_factor,Interval=optimizer.growth_interval)
                for i,p in enumerate(optimizer.parameters):
                    if p not in grads:continue
                    state=optimizer.state[p]
                    self.b.launch(adamw,(triton.cdiv(p.size,256),),(p.data,optimizer.gradients[p].data,state['m'].data,state['v'].data,self.steps.data,self.control.data,self.stats.data),
                        INDEX=i,N=p.size,Decay=optimizer.weight_decay,B1=optimizer.betas[0],B2=optimizer.betas[1],Eps=optimizer.eps)
                # Only used parameters advance, so preserve the eager optimizer's unused-parameter semantics.
                for i,p in enumerate(optimizer.parameters):
                    if p in grads:
                        self.b.launch(advance_one,(1,),(self.steps.data,self.stats.data),INDEX=i)
            return loss
        self.plan=self.b.build(prepare)
        # Buffers/events referenced by the replay must survive dynamic eager fallbacks.
        self.b.pin_since(mark)
        self.plan.capture()

    def run(self,batch,lr,force=False):
        if tuple(x.shape for x in batch)!=self.shape:raise ValueError('CUDA Graph batch shape changed')
        for tensor,array in zip(self.inputs,batch):self.b.copy_from_numpy(tensor.data,np.asarray(array,np.int64))
        opt=self.opt
        self.b.copy_from_numpy(self.control.data,np.asarray([lr,1.,force],np.float32))
        self.b.copy_from_numpy(self.counts.data,np.asarray([opt.accumulated,opt.updates,opt.good_steps],np.int64))
        self.b.copy_from_numpy(self.stats.data,np.asarray([opt.scale,0.,0.,0.,0.],np.float32))
        self.b.copy_from_numpy(self.steps.data,np.asarray([opt.state[p]['step'] for p in opt.parameters],np.int64))
        loss=self.plan.replay();self.b.synchronize()
        count=self.e.numpy(self.counts);stats=self.e.numpy(self.stats);steps=self.e.numpy(self.steps)
        opt.accumulated,opt.updates,opt.good_steps=map(int,count)
        opt.scale=float(stats[0]);opt.present=set(self.parameters) if opt.accumulated else set()
        for p,step in zip(opt.parameters,steps):opt.state[p]['step']=int(step)
        return self.e.item(loss)


@triton.jit
def advance_one(Steps,Stats,INDEX:tl.constexpr):
    if tl.load(Stats+3)!=0:tl.store(Steps+INDEX,tl.load(Steps+INDEX)+1)
