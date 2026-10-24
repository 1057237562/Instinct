"""Native optimizer state, accumulation, loss scaling and updates."""
import math
import triton
import triton.language as tl
from model.dense_triton.runtime import Scalar
from . import tensor_kernels as tk


@triton.jit
def adamw_update(P,G,M,V,lr,decay,b1,b2,bc1,bc2,epsilon,scale,N:tl.constexpr,BLOCK:tl.constexpr=256):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
    g=tl.load(G+i,valid,0).to(tl.float32)*scale
    m=b1*tl.load(M+i,valid,0)+(1.-b1)*g
    v=b2*tl.load(V+i,valid,0)+(1.-b2)*g*g
    p=tl.load(P+i,valid,0).to(tl.float32)*(1.-lr*decay)-lr*(m/bc1)/(tl.sqrt(v/bc2)+epsilon)
    tl.store(P+i,p,valid);tl.store(M+i,m,valid);tl.store(V+i,v,valid)


class Optimizer:
    def __init__(self,engine,parameters,kind='adamw',lr=5e-4,weight_decay=.01,grad_clip=1.,
                 betas=(.9,.999),eps=1e-8,loss_scaling=False,momentum=.95,ns_steps=5):
        if kind not in ('adamw','adafactor','muon'): raise ValueError('unknown optimizer')
        self.e,self.parameters,self.kind=engine,list(parameters),kind
        self.lr,self.weight_decay,self.grad_clip=lr,weight_decay,grad_clip
        self.betas,self.eps,self.momentum,self.ns_steps=betas,eps,momentum,ns_steps
        self.scale=65536. if loss_scaling else 1.;self.loss_scaling=loss_scaling
        self.good_steps,self.updates,self.accumulated=0,0,0
        self.growth_factor,self.backoff_factor,self.growth_interval=2.,.5,2000
        self.param_options,self.scheduler={},None
        self.present=set();self.state={};self.gradients={}
        for p in self.parameters:
            self.gradients[p]=engine.full(p.shape,0.)
            state={'step':0}
            if kind=='adafactor':
                if len(p.shape)>1:
                    state['row']=engine.full((*p.shape[:-1],1),0.)
                    state['col']=engine.full((*p.shape[:-2],1,p.shape[-1]),0.)
                else: state['variance']=engine.full(p.shape,0.)
            elif kind=='muon' and len(p.shape)==2: state['momentum']=engine.full(p.shape,0.)
            else: state.update(m=engine.full(p.shape,0.),v=engine.full(p.shape,0.))
            self.state[p]=state

    def backward(self,loss):
        gradients=self.e.vjp(loss*self.scale)
        self.accumulate(gradients)

    def accumulate(self,gradients):
        for p,g in gradients.items():
            if p not in self.gradients: continue
            self.e.copy(self.gradients[p],self.e.wrap(g),add=True);self.present.add(p)
        self.accumulated+=1

    def zero_grad(self):
        for p in self.present:
            g=self.gradients[p]
            self.e.backend.launch(tk.fill,(triton.cdiv(g.size,256),),(g.data,Scalar(0.)),N=g.size)
        self.present.clear();self.accumulated=0

    def step(self,lr=None):
        if not self.accumulated:return {'updated':False,'grad_norm':0.}
        e=self.e;requested_lr=lr;lr=self.lr if lr is None else lr
        mark=e.backend.mark()
        with e.no_grad():
            divisor=1./(self.scale*self.accumulated)
            # Scalar transfer for overflow/clip decisions; gradients stay on device.
            squares=sum((self.gradients[p]*divisor*self.gradients[p]*divisor).sum() for p in self.present)
            norm=math.sqrt(max(e.item(squares),0.)) if self.present else 0.
            if not math.isfinite(norm):
                if self.loss_scaling:self.scale=self.scale*self.backoff_factor;self.good_steps=0
                self.zero_grad();e.backend.release_since(mark)
                return {'updated':False,'grad_norm':norm,'loss_scale':self.scale}
            scale=divisor*min(1.,self.grad_clip/(norm+1.e-6)) if self.grad_clip>0 else divisor
            for p in self.parameters:
                if p not in self.present:continue
                local=e.backend.mark();state=self.state[p];state['step']+=1;t=state['step']
                grad=self.gradients[p];w=p.cast('fp32')
                options=self.param_options.get(p.name,{})
                decay=options.get('weight_decay',self.weight_decay)
                rate_lr=options.get('lr',lr) if requested_lr is None else lr
                if self.scheduler:
                    rate_lr=self.scheduler['_last_lr'][options.get('group',0)]
                local_scale=-scale if options.get('maximize',False) else scale
                if 'm' in state:
                    b1,b2=options.get('betas',self.betas)
                    args=(p.data,grad.data,state['m'].data,state['v'].data)+tuple(Scalar(v) for v in (rate_lr,decay,b1,b2,1.-b1**t,1.-b2**t,options.get('eps',self.eps),local_scale))
                    e.backend.launch(adamw_update,(triton.cdiv(p.size,256),),args,N=p.size)
                elif self.kind=='adafactor':
                    g=grad*local_scale;square=g*g;rate=t**options.get('beta2_decay',-.8)
                    eps1,eps2=options.get('eps',(None,1.e-3))
                    if eps1 is None:eps1={'fp32':2**-23,'fp16':2**-10,'bf16':2**-7}[p.dtype]
                    if 'row' in state:
                        row,col=state['row'],state['col']
                        e.copy(row,row*(1.-rate)+square.mean(-1,True)*rate)
                        e.copy(col,col*(1.-rate)+square.mean(-2,True)*rate)
                        variance=row*col/e.binary(row.mean(-2,True),eps1,'max')
                    else:
                        variance=state['variance'];e.copy(variance,variance*(1.-rate)+square*rate)
                    update=g*e.binary(variance,eps1*eps1,'max').rsqrt()
                    denom=e.binary((update*update).mean().sqrt()/options.get('d',1.),1.,'max')
                    alpha=e.binary((w*w).mean().sqrt(),eps2,'max')*min(rate_lr,t**-.5)
                    e.copy(p,w*(1.-rate_lr*decay)-update*alpha/denom)
                else:
                    g=grad*local_scale;buf=state['momentum']
                    momentum=options.get('momentum',self.momentum)
                    nesterov=options.get('nesterov',True)
                    native_muon=options.get('variant')=='torch_muon'
                    if native_muon:
                        e.copy(buf,buf*momentum+g*(1.-momentum))
                        g=g*(1.-momentum)+buf*momentum if nesterov else buf
                        g=g.cast('bf16')
                    elif momentum:
                        e.copy(buf,buf*momentum+g)
                        g=g+buf*momentum if nesterov else buf
                    transpose=g.shape[0]>g.shape[1]
                    z=g.T if transpose else g
                    norm=(z.cast('fp32')*z.cast('fp32')).sum().sqrt()
                    if native_muon:
                        # torch.optim.Muon normalizes and iterates in BF16.
                        norm=e.binary(norm.cast('bf16'),options.get('eps',1.e-7),'max')
                        z=(z/norm).cast('bf16')
                    else:z=z/(norm+1.e-7)
                    ca,cb,cc=options.get('ns_coefficients',(3.4445,-4.775,2.0315))
                    for _ in range(options.get('ns_steps',self.ns_steps)):
                        a=z@z.T
                        if native_muon:
                            # addmm scales/sums in FP32 before the output BF16 cast.
                            b=(a.cast('fp32')*cb+(a.cast('fp32')@a.cast('fp32'))*cc).cast('bf16')
                            z=(z.cast('fp32')*ca+(b.cast('fp32')@z.cast('fp32'))).cast('bf16')
                        else:
                            b=a*cb+a@a*cc;z=z*ca+b@z
                    if transpose:z=z.T
                    if native_muon:
                        adjust=options.get('adjust_lr_fn')
                        factor=.2*math.sqrt(max(p.shape)) if adjust=='match_rms_adamw' else math.sqrt(max(1.,p.shape[0]/p.shape[1]))
                        e.copy(p,w*(1.-rate_lr*decay)-z.cast('fp32')*(rate_lr*factor))
                    else:
                        factor=1. if options.get('orthogonalize_scale','weight_decay')=='lr' else decay
                        e.copy(p,(w-z*(rate_lr*factor))*(1.-rate_lr*decay))
                e.backend.release_since(local)
            self.updates+=1;self.good_steps+=1
            if self.loss_scaling and self.good_steps%self.growth_interval==0:self.scale*=self.growth_factor
            if self.scheduler:
                sched=self.scheduler;sched['last_epoch']+=1
                sched['_step_count']=sched.get('_step_count',0)+1
                sched['_last_lr']=[sched['eta_min']+(base-sched['eta_min'])*(1.+math.cos(math.pi*sched['last_epoch']/sched['T_max']))/2. for base in sched['base_lrs']]
                for options in self.param_options.values():options['lr']=sched['_last_lr'][options.get('group',0)]
            self.zero_grad()
        e.backend.release_since(mark)
        return {'updated':True,'grad_norm':norm,'loss_scale':self.scale}

    def metadata(self):
        return {k:getattr(self,k) for k in ('kind','lr','weight_decay','grad_clip','betas','eps','momentum','ns_steps','scale','loss_scaling','good_steps','updates','accumulated','growth_factor','backoff_factor','growth_interval','param_options','scheduler')}


def cosine_lr(step,total,maximum):
    return maximum*(.1+.45*(1.+math.cos(math.pi*min(step/max(total,1),1.))))
