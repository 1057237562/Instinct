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
def accumulate_norm(G,A,Control,Counts,Stats,Partials,N:tl.constexpr,OFFSET:tl.constexpr,B:tl.constexpr=1024):
    if tl.load(Control+1)!=0:
        i=tl.program_id(0)*B+tl.arange(0,B)
        g=tl.load(A+i,i<N,0)+tl.load(G+i,i<N,0)
        tl.store(A+i,g,i<N)
        scale=1./(tl.load(Stats)*(tl.load(Counts)+1))
        value=g*scale
        tl.store(Partials+OFFSET+tl.program_id(0),tl.sum(value*value,0))


@triton.jit
def accumulate_small(Table,Control,Counts,Stats,Partials,B:tl.constexpr=1024):
    if tl.load(Control+1)!=0:
        row=Table+tl.program_id(0)*4;i=tl.arange(0,B)
        source=tl.load(row).to(tl.pointer_type(tl.float32))
        target=tl.load(row+1).to(tl.pointer_type(tl.float32))
        n=tl.load(row+2);offset=tl.load(row+3)
        g=tl.load(source+i,i<n,0)+tl.load(target+i,i<n,0)
        tl.store(target+i,g,i<n)
        scale=1./(tl.load(Stats)*(tl.load(Counts)+1));value=g*scale
        tl.store(Partials+offset,tl.sum(value*value,0))


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
        offsets={};total=0
        for p in optimizer.parameters:
            offsets[p]=total;total+=triton.cdiv(p.size,1024)
        # Unused parameters keep their zero slots; used slots are overwritten on
        # every replay while accumulating, so there is no second gradient scan.
        self.norm_partials=self.e.full((total,),0.)
        def prepare():
            loss=loss_fn(self.inputs)
            present=set()
            small={p for p in optimizer.parameters if p.size<=1024} if optimizer.multi_tensor is not None else set()
            def consume(p,g):
                if p in small:return False
                if p in optimizer.gradients:
                    self.b.launch(accumulate_norm,(triton.cdiv(p.size,1024),),
                        (g,optimizer.gradients[p].data,self.control.data,self.counts.data,self.stats.data,self.norm_partials.data),
                        N=p.size,OFFSET=offsets[p])
                    present.add(p)
                return True
            retained=self.e.vjp(loss,self.stats[0].data,leaf_callback=consume)
            if retained:
                rows=[]
                for p,g in retained.items():
                    if p not in small:continue
                    rows.append([g.storage,optimizer.gradients[p].data.storage,p.size,offsets[p]])
                    present.add(p)
                if rows:
                    table=self.e.array(np.asarray(rows,np.int64))
                    self.b.launch(accumulate_small,(len(rows),),
                        (table.data,self.control.data,self.counts.data,self.stats.data,self.norm_partials.data))
            self.parameters=[p for p in optimizer.parameters if p in present]
            with self.e.no_grad():
                squares=self.e.sum_partials(self.norm_partials.data)
                self.b.launch(statistics,(1,),(squares.data,self.control.data,self.counts.data,self.stats.data),
                    Clip=optimizer.grad_clip,Accumulation=accumulation,LossScaling=optimizer.loss_scaling,
                    Growth=optimizer.growth_factor,Backoff=optimizer.backoff_factor,Interval=optimizer.growth_interval)
                bulk=optimizer.multi_tensor is not None and len(present)==len(optimizer.parameters)
                if bulk:
                    optimizer.multi_tensor.update(optimizer,0.,0.,steps=self.steps.data,control=self.control.data,stats=self.stats.data)
                for i,p in enumerate(optimizer.parameters):
                    if bulk:break
                    if p not in present:continue
                    state=optimizer.state[p]
                    self.b.launch(adamw,(triton.cdiv(p.size,256),),(p.data,optimizer.gradients[p].data,state['m'].data,state['v'].data,self.steps.data,self.control.data,self.stats.data),
                        INDEX=i,N=p.size,Decay=optimizer.weight_decay,B1=optimizer.betas[0],B2=optimizer.betas[1],Eps=optimizer.eps)
                # Only used parameters advance, so preserve the eager optimizer's unused-parameter semantics.
                used=self.e.array(np.asarray([int(p in present) for p in optimizer.parameters],np.int64))
                self.b.launch(advance_steps,(triton.cdiv(used.size,256),),
                    (self.steps.data,self.stats.data,used.data),N=used.size)
            return loss
        self.plan=self.b.build(prepare)
        # Buffers/events referenced by the replay must survive dynamic eager fallbacks.
        self.b.pin_since(mark)
        self.plan.capture()

    def run(self,batch,lr,force=False):
        if tuple(x.shape for x in batch)!=self.shape:raise ValueError('CUDA Graph batch shape changed')
        opt=self.opt
        self.b.copy_many_from_numpy(
            tuple(t.data for t in self.inputs)+(self.control.data,self.counts.data,self.stats.data,self.steps.data),
            tuple(np.asarray(array,np.int64) for array in batch)+(
                np.asarray([lr,1.,force],np.float32),
                np.asarray([opt.accumulated,opt.updates,opt.good_steps],np.int64),
                np.asarray([opt.scale,0.,0.,0.,0.],np.float32),
                np.asarray([opt.state[p]['step'] for p in opt.parameters],np.int64)))
        loss=self.plan.replay()
        count,stats,steps,loss_value=self.b.to_numpy_many((self.counts.data,self.stats.data,self.steps.data,loss.data))
        opt.accumulated,opt.updates,opt.good_steps=map(int,count)
        opt.scale=float(stats[0]);opt.present=set(self.parameters) if opt.accumulated else set()
        for p,step in zip(opt.parameters,steps):opt.state[p]['step']=int(step)
        return float(loss_value.reshape(-1)[0])


@triton.jit
def advance_steps(Steps,Stats,Used,N:tl.constexpr,B:tl.constexpr=256):
    if tl.load(Stats+3)!=0:
        i=tl.program_id(0)*B+tl.arange(0,B)
        active=(i<N)&(tl.load(Used+i,i<N,0)!=0)
        tl.store(Steps+i,tl.load(Steps+i,active,0)+1,active)
