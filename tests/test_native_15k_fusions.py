"""Rounding, sharing and lifetime checks for native throughput candidates."""
import os
import numpy as np
import pytest


@pytest.mark.parametrize('requested,shared',[(False,False),(False,True),(True,False),(True,True)])
def test_residual_norm_backward_matches_explicit_edges(requested,shared):
    if os.environ.get('TRITON_INTERPRET')!='1':pytest.skip('CPU interpreter only')
    from tests.test_native_optimization import ReusingBackend
    from model.dense_cublas.autograd import Engine
    results=[];fused=[]
    for enabled in (False,True):
        b=ReusingBackend();b.compact_gradients=True;b.fused_norm_backward=True
        b.fused_residual_backward=enabled;e=Engine(b);rng=np.random.default_rng(804)
        calls=[];launch=b.launch
        def record(fn,grid,args,**kw):
            if kw.get('ADD_RESIDUAL'):calls.append(fn.__name__)
            return launch(fn,grid,args,**kw)
        b.launch=record
        x=e.array(rng.normal(size=(2,7,32)).astype(np.float32),requires_grad=True)
        a=e.array(rng.normal(size=x.shape).astype(np.float32),'bf16',True)
        w=e.array(rng.normal(size=(32,)).astype(np.float32),requires_grad=True)
        z,y=e.residual_norm(x,a,w,out_dtype='bf16')
        loss=(y*y+z).sum()
        if shared:loss=loss+(z*1.25).sum()
        targets=(x,a,w,z) if requested else (x,a,w)
        grads=e.vjp(loss,inputs=targets) if requested else e.vjp(loss)
        for _ in range(4):e.full(x.shape,-999.)
        results.append([e.numpy(g) for g in grads] if requested else [e.numpy(grads[v]) for v in targets])
        fused.append(len(calls))
    for actual,expected in zip(results[1],results[0]):np.testing.assert_array_equal(actual,expected)
    if not shared:assert fused==[0,1]


def test_cast_amax_uses_rounded_values_and_keeps_partial_alive():
    if os.environ.get('TRITON_INTERPRET')!='1':pytest.skip('CPU interpreter only')
    from tests.test_native_optimization import ReusingBackend
    from model.dense_cublas.autograd import Engine
    b=ReusingBackend();e=Engine(b);b.fused_cast_amax=True
    data=np.random.default_rng(71).normal(size=(33,192)).astype(np.float32)
    x=e.array(data);expected=e.ops.cast(x.data,'bf16');mark=b.mark()
    actual=e.ops.cast_for_quantize(x.data,'bf16').view(33,192)
    b.release_since(mark,keep=(actual,actual.amax_partial))
    b.empty((actual.numel,),'fp32')
    np.testing.assert_array_equal(e.numpy(actual),e.numpy(expected))
    assert np.max(e.numpy(actual.amax_partial))==np.max(np.abs(e.numpy(expected)))


@pytest.mark.parametrize('lengths',[(480,470,460,450,2800,150,160,170),(3200,3400,4000),(11,)*30])
def test_adaptive_groups_preserve_order_and_reduce_cost(lengths):
    from model.dense_cublas.vendor_attention import document_groups
    size,quantum,maximum,penalty=8,512,4096,4096
    result=document_groups(lengths,size,quantum,maximum,penalty)
    assert [i for start,n in result for i in range(start,start+n)]==list(range(len(lengths)))
    assert all(0<n<=size for _,n in result)
    def cost(groups):
        return sum(n*((max(lengths[s:s+n])+quantum-1)//quantum)*quantum+penalty for s,n in groups)
    assert cost(result)<=cost(document_groups(lengths,size,quantum,maximum,0))


def test_weight_scale_cache_refreshes_after_updates_copy_and_capture():
    if os.environ.get('TRITON_INTERPRET')!='1':pytest.skip('CPU interpreter only')
    from tests.test_native_optimization import ReusingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.optim import Optimizer
    from model.dense_cublas.quantization import reserve_weight_scale
    b=ReusingBackend();b.fused_scale_inverse=True;b.cache_weight_scales=True
    e=Engine(b,fp8=True);rng=np.random.default_rng(517)
    w=e.array(rng.normal(size=(32,32)).astype(np.float32),'bf16',True,'matrix')
    reserve_weight_scale(e.ops,w.data);opt=Optimizer(e,[w],lr=.1)
    for step in range(3):
        mark=b.mark()
        quantized,scale=e.ops.quantize(w.data)
        value=e.numpy(quantized);factor=e.numpy(scale);pointer=scale.storage
        b.release_since(mark)
        quantized,scale=e.ops.quantize(w.data)
        assert pointer==scale.storage
        np.testing.assert_array_equal(e.numpy(quantized),value)
        b.cache_weight_scales=False
        reference,expected=e.ops.quantize(w.data);b.cache_weight_scales=True
        np.testing.assert_array_equal(e.numpy(reference),value)
        np.testing.assert_array_equal(e.numpy(expected),factor)
        opt.backward((w*w).sum());opt.step()
        assert not w.data.quant_scale_valid
        b.release_since(mark)
    e.ops.quantize(w.data)
    e.copy(w,e.array(np.full(w.shape,.03125,np.float32)))
    assert not w.data.quant_scale_valid
    b._commands=[]
    _,captured=e.ops.quantize(w.data)
    assert not w.data.quant_scale_valid and captured.storage!=pointer
    b._commands=None
    q,_=e.ops.quantize(w.data)
    np.testing.assert_array_equal(e.numpy(q),np.full(w.shape,448.,np.float32))


def test_captured_optimizer_invalidates_eager_weight_scales():
    from types import SimpleNamespace
    from model.dense_cublas.graph import StepGraph
    parameter=type('Parameter',(),{})();parameter.data=SimpleNamespace(quant_scale_valid=True)
    opt=SimpleNamespace(parameters=[parameter],accumulated=0,updates=0,good_steps=0,scale=1.,
                        state={parameter:{'step':0}})
    graph=StepGraph.__new__(StepGraph);graph.opt=opt;graph.shape=((1,2),)
    graph.parameters=[parameter];graph.inputs=(SimpleNamespace(data=None),)
    graph.control=graph.counts=graph.stats=graph.steps=SimpleNamespace(data=None)
    seen=[]
    def replay():
        seen.append(parameter.data.quant_scale_valid)
        return SimpleNamespace(data=None)
    graph.plan=SimpleNamespace(replay=replay)
    graph.b=SimpleNamespace(copy_many_from_numpy=lambda *args:None,
        to_numpy_many=lambda *args:(np.array([0,1,1]),np.array([1.]),np.array([1]),np.array([2.])))
    assert graph.run((np.zeros((1,2),np.int64),),.001)==2.
    assert seen==[False] and opt.updates==1


def test_cached_parameter_reload_uses_restored_weights(tmp_path):
    if os.environ.get('TRITON_INTERPRET')!='1':pytest.skip('CPU interpreter only')
    from tests.test_native_optimization import ReusingBackend
    from tests.test_native_v2_recipe import CFG
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas import checkpoint
    b=ReusingBackend();b.cache_weight_scales=True;b.fused_scale_inverse=True
    e=Engine(b,fp8=True);m=Model(e,CFG,param_dtype='bf16',muon_mantissa=True)
    weight=m.params['model.layers.0.self_attn.q_proj.weight']
    q,scale=e.ops.quantize(weight.data);expected=e.numpy(q);expected_scale=e.numpy(scale)
    path=tmp_path/'weights.npz';checkpoint.save(path,m)
    e.copy(weight,e.array(np.full(weight.shape,2.,np.float32)))
    e.ops.quantize(weight.data);assert weight.data.quant_scale_valid
    checkpoint.load(path,m);assert not weight.data.quant_scale_valid
    q,scale=e.ops.quantize(weight.data)
    np.testing.assert_array_equal(e.numpy(q),expected)
    np.testing.assert_array_equal(e.numpy(scale),expected_scale)
