"""Actual native CUDA Muon versus an independent NumPy BF16 reference."""
import os
import numpy as np
import pytest
from model.dense_cublas.backend import ScheduleConfig
from model.dense_cublas.memory import TrainingBackend
from model.dense_cublas.autograd import Engine
from model.dense_cublas.optim import Optimizer

pytestmark=[pytest.mark.gpu,pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')=='1',reason='real CUDA only')]


def bf16(x):
    x=np.asarray(x,np.float32)
    bits=x.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def fma(a,b,c):
    return (np.asarray(a,np.float64)*np.asarray(b,np.float64)+np.asarray(c,np.float64)).astype(np.float32)


def lerp(a,b,w):
    w=np.float32(w);delta=(b-a).astype(np.float32)
    return fma(w,delta,a) if abs(w)<.5 else fma(np.float32(-(1.-float(w))),delta,b)


@pytest.mark.parametrize('momentum_dtype',['fp32','bf16'])
@pytest.mark.parametrize('steps',[0,5])
@pytest.mark.parametrize('allocator',['driver','async'])
def test_split_muon_matches_batched_bf16_reference(momentum_dtype,steps,allocator):
    rng=np.random.default_rng(23)
    with TrainingBackend(config=ScheduleConfig(),allocator_backend=allocator) as backend:
        e=Engine(backend);params=[]
        for index,shape in enumerate(((16,32),(16,32),(32,16),(32,32))):
            params.append(e.array(rng.normal(0,.1,shape).astype(np.float32),'bf16',True,f'layer.{index}.weight'))
        opt=Optimizer(e,params,'muon',lr=.003,weight_decay=.1,grad_clip=0.,
                      muon_mantissa=True,muon_momentum_dtype=momentum_dtype,muon_batch_size=2,ns_steps=steps)
        references=[e.numpy(p) for p in params];momenta=[np.zeros(p.shape,np.float32) for p in params]
        for iteration in range(3):
            gradients=[bf16(rng.normal(0,.05,p.shape)) for p in params]
            mark=backend.mark()
            opt.accumulate({p:e.array(g,'bf16').data for p,g in zip(params,gradients)})
            assert opt.step()['updated']
            for index,(p,g) in enumerate(zip(params,gradients)):
                momentum=lerp(momenta[index],g,.05)
                update=bf16(lerp(g,momentum,.95))
                momenta[index]=bf16(momentum) if momentum_dtype=='bf16' else momentum
                transpose=p.shape[0]>p.shape[1]
                x=update.T if transpose else update
                norm=bf16(np.sqrt(np.sum(x*x,dtype=np.float32)))
                x=bf16(x/np.maximum(norm,bf16(1.e-7)))
                for _ in range(steps):
                    gram=bf16(x@x.T)
                    polynomial=bf16(fma(np.float32(2.0315),gram@gram,np.float32(-4.775)*gram))
                    x=bf16(fma(np.float32(1.),polynomial@x,np.float32(3.4445)*x))
                update=x.T if transpose else x
                expected=fma(update,np.float32(-.003*.2*np.sqrt(max(p.shape))),
                             references[index]*np.float32(1.-.003*.1))
                hi=e.numpy(p).view(np.uint32)&np.uint32(0xffff0000)
                low=e.numpy(opt.state[p]['mantissa'])
                actual=(hi|low.astype(np.uint32)).view(np.float32)
                # Tensor Core reduction order may change intermediate BF16 NS
                # rounding; split/update itself is exact with steps=0.
                if steps==0:np.testing.assert_array_equal(actual,expected)
                else:np.testing.assert_allclose(actual,expected,atol=4e-5,rtol=3e-4)
                np.testing.assert_allclose(e.numpy(opt.state[p]['momentum']),momenta[index],atol=2e-8,rtol=2e-6)
                references[index]=actual.copy()
            backend.release_since(mark)
