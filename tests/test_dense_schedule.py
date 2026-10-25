"""Dense fusion, precision transitions and persistent optimizer pointer tables."""
import os
import datasets  # Windows DLL ordering.
import numpy as np
import pytest
import torch
from model.dense_cublas.autograd import Engine
from model.dense_cublas.optim import Optimizer
from tests.test_native_optimization import ReusingBackend
from tests.test_cublas_training import HostBackend

pytestmark=pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1',reason='CPU interpreter suite')


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
@pytest.mark.parametrize('dim',[6,96,160])
def test_norm_rope_fusion_keeps_rounding_and_transpose(dtype,dim):
    rng=np.random.default_rng(55);shape=(2,5,3,dim)
    source=rng.normal(size=shape).astype(np.float32)
    weight=rng.normal(size=(dim,)).astype(np.float32)
    # Nonduplicated tables check true RoPE transpose, not only standard tables.
    cos=rng.normal(size=(9,dim)).astype(np.float32)
    sin=rng.normal(size=(9,dim)).astype(np.float32)
    upstream=rng.normal(size=shape).astype(np.float32)
    results=[]
    for fused in (False,True):
        e=Engine(ReusingBackend())
        x=e.array(source,dtype,True);w=e.array(weight,requires_grad=True)
        c,s=e.array(cos),e.array(sin)
        y=e.norm_rotary(x,w,c,s,start=3) if fused else e.rotary(e.norm(x,w),c,s,start=3)
        grads=e.vjp(y,e.array(upstream).data)
        results.append((e.numpy(y),e.numpy(grads[x]),e.numpy(grads[w])))
    for actual,expected in zip(results[1],results[0]):
        np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-5)


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_multi_tensor_optimizer_matches_scalar_with_unequal_shapes_and_steps(dtype):
    rng=np.random.default_rng(7)
    e,r=Engine(ReusingBackend()),Engine(HostBackend())
    params=[];references=[]
    for i,shape in enumerate(((1,), (3,7), (1031,), (33,65))):
        a=rng.normal(size=shape).astype(np.float32)*.1
        params.append(e.array(a,dtype,True,name=str(i)))
        references.append(r.array(a,dtype,True,name=str(i)))
    actual=Optimizer(e,params,lr=.002,grad_clip=.3,loss_scaling=True)
    expected=Optimizer(r,references,lr=.002,grad_clip=.3,loss_scaling=True)
    assert actual.multi_tensor is not None and expected.multi_tensor is None
    for step in range(4):
        gs=[rng.normal(size=p.shape).astype(np.float32)*actual.scale for p in params]
        actual.accumulate({p:e.array(g).data for p,g in zip(params,gs)})
        expected.accumulate({p:r.array(g).data for p,g in zip(references,gs)})
        if step%2:
            aa,bb=actual.step(),expected.step()
            assert abs(aa['grad_norm']-bb['grad_norm'])<3e-5
            for p,q in zip(params,references):
                np.testing.assert_allclose(e.numpy(p),r.numpy(q),atol=2e-6,rtol=2e-5)
                for key in ('m','v'):
                    np.testing.assert_allclose(e.numpy(actual.state[p][key]),r.numpy(expected.state[q][key]),atol=1e-7,rtol=2e-5)
    # Unequal imported step counters must retain scalar semantics.
    actual.state[params[0]]['step']=7;expected.state[references[0]]['step']=7
    actual.accumulate({p:e.full(p.shape,actual.scale).data for p in params})
    expected.accumulate({p:r.full(p.shape,expected.scale).data for p in references})
    actual.step();expected.step()
    for p,q in zip(params,references):np.testing.assert_allclose(e.numpy(p),r.numpy(q),atol=2e-6,rtol=2e-5)
    assert actual.state[params[0]]['step']==8


def test_precision_aware_vjp_keeps_shared_sums_fp32_and_public_gradients_fp32():
    e=Engine(ReusingBackend())
    x=e.array(np.arange(12,dtype=np.float32).reshape(3,4)*.1,'fp16',True)
    w=e.array(np.eye(4,dtype=np.float32),requires_grad=True)
    a=e.linear(x,w);loss=(a*a+a).sum()
    grads=e.vjp(loss)
    assert grads[x].dtype==grads[w].dtype=='fp32'
    expected=2*e.numpy(x).astype(np.float32)+1.
    np.testing.assert_allclose(e.numpy(grads[x]),expected.astype(np.float16).astype(np.float32),atol=0.,rtol=0.)


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
@pytest.mark.parametrize('count',[2,3])
def test_batched_projection_cast_keeps_forward_and_parameter_gradients(dtype,count):
    rng=np.random.default_rng(23)
    source=rng.normal(size=(2,3,7)).astype(np.float32)
    weights=[rng.normal(size=(n,7)).astype(np.float32) for n in (5,3,9)[:count]]
    results=[]
    for batch in (False,True):
        e=Engine(ReusingBackend());x=e.array(source,dtype,True)
        ps=[e.array(w,requires_grad=True) for w in weights]
        ys=e.linears(x,ps) if batch else [e.linear(x,p) for p in ps]
        loss=sum((y*y).sum() for y in ys)
        gs=e.vjp(loss)
        results.append(([e.numpy(y) for y in ys],[e.numpy(gs[v]) for v in (x,*ps)]))
    for a,b in zip(results[0][0]+results[0][1],results[1][0]+results[1][1]):
        np.testing.assert_allclose(a,b,atol=0.,rtol=0.)


@pytest.mark.parametrize('ignored',[False,True])
def test_shifted_ce_matches_sliced_loss_without_copying_logits(ignored):
    rng=np.random.default_rng(5);values=rng.normal(size=(3,7,11)).astype(np.float32)
    labels=rng.integers(0,11,size=(3,7),dtype=np.int64)
    labels[0,1:]=-100;labels[1,3]=-100
    if ignored:labels.fill(-100)
    results=[]
    for shifted in (False,True):
        e=Engine(ReusingBackend());x=e.array(values,requires_grad=True);y=e.array(labels)
        loss=e.ce(x,y,shift=True) if shifted else e.ce(x[:,:-1],y[:,1:])
        gradient=e.vjp(loss)[x]
        results.append((e.item(loss),e.numpy(gradient)))
    assert abs(results[0][0]-results[1][0])<1e-6
    np.testing.assert_allclose(results[0][1],results[1][1],atol=0.,rtol=0.)
    np.testing.assert_array_equal(results[1][1][:,-1],0.)


@pytest.mark.parametrize('dtype',['fp32','fp16','bf16'])
def test_default_swiglu_fusion_preserves_intermediate_rounding(dtype):
    rng=np.random.default_rng(17);values=[rng.normal(size=(5,97)).astype(np.float32) for _ in range(3)]
    results=[]
    for fused in (False,True):
        e=Engine(ReusingBackend());g=e.array(values[0],dtype,True);u=e.array(values[1],dtype,True)
        out=e.swiglu(g,u,round_gate=True) if fused else e.unary(g,'silu')*u
        grads=e.vjp(out,e.array(values[2]).data)
        results.append([e.numpy(x) for x in (out,grads[g],grads[u])])
    for a,b in zip(*results):np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('unused',[False,True])
def test_graph_small_batch_large_gradient_overflow_and_unused_parameter(monkeypatch,unused):
    from types import SimpleNamespace
    from model.dense_cublas.graph import StepGraph
    from model.dense_cublas.backend import ExecutionPlan
    monkeypatch.setattr(ExecutionPlan,'capture',lambda self:None)
    e,r=Engine(ReusingBackend()),Engine(HostBackend())
    params=[e.array(np.full((n,),.02,np.float32),requires_grad=True,name=str(n)) for n in (1501,17,3)]
    refs=[r.array(e.numpy(p),requires_grad=True,name=p.name) for p in params]
    opt=Optimizer(e,params,grad_clip=.2,lr=.003,loss_scaling=True)
    oracle=Optimizer(r,refs,grad_clip=.2,lr=.003,loss_scaling=True)
    def loss(eng,ps,inputs):
        scale=inputs[0][0,0].cast('fp32')
        return sum((p*p).sum()*scale for p in ps[:2 if unused else 3])
    batch=np.ones((1,2),np.int64)
    model=SimpleNamespace(e=e,config=dict(dropout=0.,model_architecture='standard'))
    graph=StepGraph(model,opt,(batch,batch),lambda xs:loss(e,params,xs),2)
    for step in range(4):
        values=(batch+step%2,batch)
        expected=loss(r,refs,tuple(r.array(a) for a in values));oracle.backward(expected)
        if step%2:oracle.step()
        actual=graph.run(values,.003)
        assert abs(actual-r.item(expected))<2e-5
        for p,q in zip(params,refs):np.testing.assert_allclose(e.numpy(p),r.numpy(q),atol=2e-6,rtol=3e-5)
    assert opt.state[params[-1]]['step']==(0 if unused else 2)
    # Overflow in the small-parameter batch must stop the large-parameter update too.
    old=[e.numpy(p) for p in params];scale=opt.scale
    e.copy(opt.gradients[params[1]],e.full(params[1].shape,float('inf')))
    graph.run((batch,batch),.003,force=True)
    assert opt.scale==scale*.5 and opt.updates==2
    for p,expected in zip(params,old):np.testing.assert_array_equal(e.numpy(p),expected)


def test_pointer_tables_survive_checkpoint_reload(tmp_path,monkeypatch):
    from model.dense_cublas import checkpoint
    from model.dense_cublas.model import Model
    from model.dense_cublas.graph import StepGraph
    from model.dense_cublas.backend import ExecutionPlan
    from tests.test_legacy_resume import CONFIG
    monkeypatch.setattr(ExecutionPlan,'capture',lambda self:None)
    e=Engine(ReusingBackend());m=Model(e,CONFIG,activation_dtype='fp32')
    opt=Optimizer(e,m.parameters(),lr=.002)
    batch=np.array([[1,2,3,4]],np.int64)
    graph=StepGraph(m,opt,(batch,batch),lambda xs:m(*xs,return_logits=False).loss,1)
    graph.run((batch,batch),.002)
    path=tmp_path/'resume.npz';checkpoint.save(path,m,opt,{'steps':1})
    expected_loss=graph.run((batch+1,batch),.002)
    expected=[e.numpy(p) for p in m.parameters()]
    checkpoint.load(path,m,opt)
    actual=graph.run((batch+1,batch),.002)
    assert abs(actual-expected_loss)<1e-6
    for p,value in zip(m.parameters(),expected):np.testing.assert_array_equal(e.numpy(p),value)
