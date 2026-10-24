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
        e=self.e;lr=self.lr if lr is None else lr
        mark=e.backend.mark()
        with e.no_grad():
            divisor=1./(self.scale*self.accumulated)
            # Scalar transfer for overflow/clip decisions; gradients stay on device.
            squares=sum((self.gradients[p]*divisor*self.gradients[p]*divisor).sum() for p in self.present)
            norm=math.sqrt(max(e.item(squares),0.)) if self.present else 0.
            if not math.isfinite(norm):
                if self.loss_scaling:self.scale=max(self.scale/2.,1.);self.good_steps=0
                self.zero_grad();e.backend.release_since(mark)
                return {'updated':False,'grad_norm':norm,'loss_scale':self.scale}
            scale=divisor*min(1.,self.grad_clip/(norm+1.e-6)) if self.grad_clip>0 else divisor
            for p in self.parameters:
                if p not in self.present:continue
                local=e.backend.mark();state=self.state[p];state['step']+=1;t=state['step']
                grad=self.gradients[p];w=p.cast('fp32')
                if 'm' in state:
                    b1,b2=self.betas
                    args=(p.data,grad.data,state['m'].data,state['v'].data)+tuple(Scalar(v) for v in (lr,self.weight_decay,b1,b2,1.-b1**t,1.-b2**t,self.eps,scale))
                    e.backend.launch(adamw_update,(triton.cdiv(p.size,256),),args,N=p.size)
                elif self.kind=='adafactor':
                    g=grad*scale;square=g*g;rate=t**-.8
                    eps1={'fp32':2**-23,'fp16':2**-10,'bf16':2**-7}[p.dtype]
                    if 'row' in state:
                        row,col=state['row'],state['col']
                        e.copy(row,row*(1.-rate)+square.mean(-1,True)*rate)
                        e.copy(col,col*(1.-rate)+square.mean(-2,True)*rate)
                        variance=row*col/e.binary(row.mean(-2,True),eps1,'max')
                    else:
                        variance=state['variance'];e.copy(variance,variance*(1.-rate)+square*rate)
                    update=g*e.binary(variance,eps1*eps1,'max').rsqrt()
                    denom=e.binary((update*update).mean().sqrt(),1.,'max')
                    alpha=e.binary((w*w).mean().sqrt(),1.e-3,'max')*min(lr,t**-.5)
                    e.copy(p,w*(1.-lr*self.weight_decay)-update*alpha/denom)
                else:
                    g=grad*scale;buf=state['momentum'];e.copy(buf,buf*self.momentum+g)
                    g=g+buf*self.momentum
                    transpose=g.shape[0]>g.shape[1]
                    z=g.T if transpose else g
                    z=z/((z*z).sum().sqrt()+1.e-7)
                    for _ in range(self.ns_steps):
                        a=z@z.T;b=a*(-4.775)+a@a*2.0315;z=z*3.4445+b@z
                    if transpose:z=z.T
                    # Matches trainer_utils.MuonFallback's weight_decay scaling.
                    e.copy(p,(w-z*(lr*self.weight_decay))*(1.-lr*self.weight_decay))
                e.backend.release_since(local)
            self.updates+=1;self.good_steps+=1
            if self.loss_scaling and self.good_steps%2000==0:self.scale=min(self.scale*2.,2.**24)
            self.zero_grad()
        e.backend.release_since(mark)
        return {'updated':True,'grad_norm':norm,'loss_scale':self.scale}

    def metadata(self):
        return {k:getattr(self,k) for k in ('kind','lr','weight_decay','grad_clip','betas','eps','momentum','ns_steps','scale','loss_scaling','good_steps','updates','accumulated')}


def cosine_lr(step,total,maximum):
    return maximum*(.1+.45*(1.+math.cos(math.pi*min(step/max(total,1),1.))))
