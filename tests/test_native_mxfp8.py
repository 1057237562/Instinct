from types import SimpleNamespace
import numpy as np
import pytest


def test_mxfp8_is_not_enabled_by_the_production_policy():
    from model.dense_cublas.policy import enable_fused_ops
    backend=SimpleNamespace();enable_fused_ops(backend)
    assert not getattr(backend,'mxfp8_training',False)
    assert not getattr(backend,'mxfp8_norm_fusion',False)


def test_experimental_resume_cannot_silently_select_tensorwise_compute():
    from model.dense_cublas.model import Model
    engine=SimpleNamespace(backend=SimpleNamespace(),ops=SimpleNamespace(fp8=True))
    with pytest.raises(ValueError,match='explicit experimental MXFP8'):
        Model(engine,dict(native_fp8_recipe='mxfp8_experimental'))


def test_experimental_gram_resume_requires_its_explicit_compute_policy():
    from model.dense_cublas.model import Model
    engine=SimpleNamespace(backend=SimpleNamespace(),ops=SimpleNamespace(fp8=True))
    with pytest.raises(ValueError,match='Muon Gram recurrence'):
        Model(engine,dict(native_muon_gram_recurrence='fp16_restart3'))


def test_fused_quantization_hint_does_not_survive_a_changed_reduction_axis():
    from model.dense_triton.runtime import Tensor
    value=Tensor((2,3,192),'bf16',256,None);value.mx32_quantized=object()
    assert value.view(6,192).mx32_quantized is value.mx32_quantized
    assert not hasattr(value.view(3,384),'mx32_quantized')


def test_transposed_mx_values_require_an_independent_scale_and_rounding():
    from experiments.validate_native_mxfp8 import reference
    x=np.ones((128,128),np.float32);x[::16,::16]=300.
    q,scales,_=reference(x,'fp8e4nv');qt,st,_=reference(x.T,'fp8e4nv')
    assert q.shape==qt.shape and scales.size==st.size
    assert not np.array_equal(q.T,qt)


def test_gradient_auxiliary_buffers_survive_nested_checkpoint_reclamation():
    from model.dense_cublas.mxfp8 import auxiliary_storage
    from model.dense_triton.runtime import Tensor
    dx=Tensor((128,128),'fp32',256,None);bf=Tensor(dx.shape,'bf16',512,None)
    q=Tensor(dx.shape,'fp8e5',768,None);s=Tensor((256,),'u16',1024,None)
    dx.mx32_cast_bf16=bf;bf.mx32_quantized=(q,s);q.mx32_scale=s
    q.mx32_source=bf  # The source edge is not a new ownership relationship.
    assert {t.storage for t in auxiliary_storage((dx,dx,None))}=={512,768,1024}


@pytest.mark.parametrize('addresses',[None,(0,256),(257,512),(256,)])
def test_mx_plan_requires_real_aligned_scale_addresses(addresses):
    from model.dense_cublas.bridge import NativeLt
    bridge=NativeLt.__new__(NativeLt);bridge.abi=2;bridge._plan_ex=None;bridge._plan=lambda *a:None
    bridge._plan_mxfp8=lambda *a:pytest.fail('invalid scale addresses reached native code')
    with pytest.raises(ValueError,match='aligned scale'):
        bridge.create_plan(128,128,128,'fp8e4nv','fp8e4nv','bf16',0,4,0,mxfp8=True,block_scales=addresses)


@pytest.mark.parametrize('selection,expected',[(False,(False,False)),(True,(True,True)),('kv',(True,True)),('k',(True,False)),('v',(False,True))])
def test_small_projection_experiment_preserves_head_and_tensorwise_filters(selection,expected):
    from model.dense_cublas.autograd import Engine
    e=Engine.__new__(Engine);e.backend=SimpleNamespace(mxfp8_training=True,mxfp8_small_projections=selection)
    weights=[SimpleNamespace(shape=(384,1536),name=f'model.layers.0.self_attn.{name}_proj.weight') for name in ('k','v')]
    assert tuple(e._auto_fp8(w) for w in weights)==expected
    assert not e._auto_fp8(SimpleNamespace(shape=(6400,1536),name='lm_head.weight'))
    e.backend.mxfp8_training=False
    assert not any(e._auto_fp8(w) for w in weights)


def test_compressed_ffn_cache_accounts_for_padded_scales_at_the_eighth_layer_boundary():
    from model.dense_cublas.selective import recipe_plan
    config=dict(hidden_size=1536,num_attention_heads=16,num_key_value_heads=4,head_dim=96,
                intermediate_size=5504,num_hidden_layers=32,selective_cache_mib=2816,dropout=0.)
    def plans(compressed):
        return [recipe_plan(config,(4,3072,1536),'bf16',i,attention_min_seq=1,auxiliary_kv=True,compressed_ffn=compressed) for i in range(32)]
    normal,compressed=plans(False),plans(True)
    assert sum(any('gate_proj' in n for n in p.linear_names) for p in normal)==4
    assert sum(any('gate_proj' in n for n in p.linear_names) for p in compressed)==7
    assert sum(p.attention for p in compressed)==32
    assert sum(any('k_proj' in n for n in p.linear_names) for p in compressed)==32
    # Actual storage: 1176 MiB O/LSE, 576 MiB K/V, eight pairs of
    # 129 MiB FP8 values plus 4.03125 MiB UE8M0 scale tiles per pair.
    assert 1176+576+8*(129+4.03125)>2816
    config['selective_cache_mib']=2817
    assert sum(any('gate_proj' in n for n in p.linear_names) for p in plans(True))==8
