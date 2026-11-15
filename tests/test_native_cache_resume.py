"""Changing transient ckpt4 budgets must preserve every resume-state member."""
import hashlib
import json
import os

import numpy as np
import pytest

from model.dense_cublas import checkpoint


def test_cache_only_resume_compatibility_is_narrow():
    cfg=dict(use_grad_checkpoint=4,hidden_size=1536,native_fp8_recipe='mxfp8_experimental',
             selective_cache_mib=2684,selective_cache_by_seq={'3072':2684,'4096':2684})
    newer=cfg|dict(selective_cache_mib=6144,selective_cache_by_seq={'3072':6144,'4096':6144})
    change=checkpoint.validate_resume_config(cfg,newer)
    assert change['previous']['selective_cache_mib']==2684
    assert change['current']['selective_cache_by_seq']=={'3072':6144,'4096':6144}
    assert checkpoint.validate_resume_config(cfg,dict(cfg)) is None
    assert checkpoint.validate_resume_config(newer,cfg)  # Memory-safe fallback also resumes.
    for invalid in (newer|{'hidden_size':2048},newer|{'use_grad_checkpoint':1},
                    newer|{'native_fp8_recipe':'other'},newer|{'unknown_option':True}):
        with pytest.raises(ValueError,match='model configuration'):
            checkpoint.validate_resume_config(cfg,invalid)
    with pytest.raises(ValueError,match='model configuration'):
        checkpoint.validate_resume_config(cfg|{'use_grad_checkpoint':2},newer|{'use_grad_checkpoint':2})


def test_production_override_reaches_both_buckets_and_persists_for_resume():
    from experiments.run_native_1b_acceptance import runtime_cache_options
    from scripts.native_v2_recipe import command,recipe
    from trainer.train_cublas import parser,read_config
    flags,env=runtime_cache_options(6144,24576)
    args=parser().parse_args(command('source.parquet',extra=flags)[3:])
    config=read_config(args)
    assert config['selective_cache_mib']==6144
    assert config['selective_cache_by_seq']=={'3072':6144,'4096':6144}
    assert args.allocator_release_mib==24576
    assert env=={'INSTINCT_NATIVE_SELECTIVE_CACHE_MIB':'6144','INSTINCT_NATIVE_ALLOCATOR_RELEASE_MIB':'24576'}
    assert recipe()['selective_cache_mib']==2684  # Other machines keep the shared baseline.
    assert runtime_cache_options()==([],{})
    with pytest.raises(ValueError):runtime_cache_options(-1)


@pytest.mark.parametrize('budget,first_layer',[(6144,12),(9216,0)])
def test_expanded_cache_keeps_original_fp8_layers_and_adds_bf16_outputs(budget,first_layer):
    from model.dense_cublas.selective import recipe_plan
    from scripts.native_v2_recipe import model_config
    cfg=model_config()|dict(selective_cache_mib=budget)
    reference=dict(selective_cache_mib=2684,selective_cache_by_seq={'3072':2684,'4096':2684})
    for shape in ((4,3072,1536),(3,4096,1536),(1,11264,1536)):
        plans=[recipe_plan(cfg,shape,'bf16',i,attention_min_seq=1,auxiliary_kv=True,
            compressed_ffn=True,storage_tokens=12288,compressed_reference=reference) for i in range(32)]
        gates=[i for i,p in enumerate(plans) if any('gate_proj' in n for n in p.linear_names)]
        compressed=[i for i in gates if plans[i].compress_ffn]
        assert gates==list(range(first_layer,32)) and compressed==list(range(25,32))
        actual_mib=1176+576+7*133.03125+(25-first_layer)*258
        assert actual_mib<=budget
        if first_layer:assert actual_mib+258>budget
    with pytest.raises(ValueError,match='precision reference'):
        recipe_plan(cfg|{'selective_cache_mib':0},(4,3072,1536),'bf16',0,
            compressed_ffn=True,compressed_reference=reference)


INTERPRET=pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1',reason='CPU interpreter')


@INTERPRET
@pytest.mark.parametrize('accumulated',[0,1])
def test_budget_change_restores_all_muon_bits_rng_and_data_cursor(tmp_path,accumulated):
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.optim import Optimizer
    from tests.test_native_optimization import ReusingBackend
    cfg=dict(hidden_size=8,num_hidden_layers=1,intermediate_size=12,vocab_size=13,
             head_dim=4,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=16,
             use_grad_checkpoint=4,selective_cache_mib=0,selective_cache_by_seq={'8':0})
    def create(config,seed):
        e=Engine(ReusingBackend(),seed=seed)
        m=Model(e,config,param_dtype='bf16',muon_mantissa=True)
        opt=Optimizer(e,m.parameters(),'muon',muon_mantissa=True,muon_momentum_dtype='bf16',muon_workspace_mb=1)
        return e,m,opt
    e,m,opt=create(cfg,47)
    # Nonzero raw mantissa, momentum, Adam companions, gradient and counters.
    for p,state in opt.state.items():
        state['step']=17
        for key,t in state.items():
            if key!='step':e.copy(t,e.full(t.shape,13579 if key=='mantissa' else .125,t.dtype))
        e.copy(opt.gradients[p],e.full(p.shape,.25,opt.gradients[p].dtype))
    opt.updates=17;opt.good_steps=42;opt.accumulated=accumulated;opt.present=set(opt.parameters)
    e.counter=139
    rng=np.random.default_rng(21);rng.integers(1000,size=19)
    progress=dict(epoch=0,steps=136,processed_tokens=1665280,
                  parquet_cursor=dict(group=6,batch=91,valid_tokens=12277,useful_targets=12265),
                  dataset_path=r'D:\production\source.parquet',data_options={'parquet_fingerprint':'unchanged'})
    path=tmp_path/'resume.npz';checkpoint.save(path,m,opt,progress,rng)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    before=checkpoint.snapshot(m,opt,progress,rng)
    _,other,other_opt=create(cfg|dict(selective_cache_mib=6144,selective_cache_by_seq={'8':6144}),999)
    restored_rng=np.random.default_rng(999)
    restored=checkpoint.load(path,other,other_opt,restored_rng)
    after=checkpoint.snapshot(other,other_opt,restored,restored_rng)
    for key,value in before.items():
        if key!='__metadata__':np.testing.assert_array_equal(after[key],value,err_msg=key)
    old_meta=json.loads(str(before['__metadata__']));new_meta=json.loads(str(after['__metadata__']))
    assert old_meta.pop('config')==m.config
    assert new_meta.pop('config')==other.config
    for meta in (old_meta,new_meta):
        meta['optimizers']['model']['present'].sort()  # This field serializes a set.
    assert old_meta==new_meta
    np.testing.assert_array_equal(restored_rng.integers(2**31,size=32),rng.integers(2**31,size=32))
    assert restored==progress and other_opt.accumulated==accumulated
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    # A precision/structure mismatch must fail before mutating the destination.
    other.config['native_fp8_recipe']='wrong'
    pristine=[other.e.numpy(p) for p in other.parameters()]
    with pytest.raises(ValueError,match='model configuration'):checkpoint.load(path,other,other_opt)
    for p,expected in zip(other.parameters(),pristine):np.testing.assert_array_equal(other.e.numpy(p),expected)


@INTERPRET
def test_uncompressed_resume_continues_same_dropout_and_optimizer_updates(tmp_path):
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.optim import Optimizer
    from tests.test_native_optimization import ReusingBackend
    cfg=dict(hidden_size=8,num_hidden_layers=1,intermediate_size=12,vocab_size=13,
             head_dim=4,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=16,
             use_grad_checkpoint=4,selective_cache_mib=0,dropout=.2,flash_attn=False)
    e=Engine(ReusingBackend(),seed=47);m=Model(e,cfg,activation_dtype='fp32')
    opt=Optimizer(e,m.parameters(),'adamw');rng=np.random.default_rng(9)
    ids=np.array([[1,3,2,4]],np.int64)
    opt.backward(m(ids,ids).loss);opt.step()
    path=tmp_path/'resume.npz';checkpoint.save(path,m,opt,{'steps':1},rng)
    def continue_with(budget):
        m.config['selective_cache_mib']=budget
        checkpoint.load(path,m,opt,rng)
        for _ in range(2):opt.backward(m(ids,ids).loss);opt.step()
        return checkpoint.snapshot(m,opt,{'steps':3},rng)
    baseline=continue_with(0);expanded=continue_with(1)
    for key in baseline:
        if key!='__metadata__':np.testing.assert_allclose(expanded[key],baseline[key],rtol=1e-6,atol=1e-7,err_msg=key)
    a=json.loads(str(baseline['__metadata__']));b=json.loads(str(expanded['__metadata__']))
    a.pop('config');b.pop('config');assert a==b


@INTERPRET
def test_precision_reference_is_saved_for_later_resumes_and_rejects_unsafe_shrink(tmp_path):
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.optim import Optimizer
    from tests.test_native_optimization import ReusingBackend
    cfg=dict(hidden_size=8,num_hidden_layers=1,intermediate_size=12,vocab_size=13,
             head_dim=4,num_attention_heads=2,num_key_value_heads=1,use_grad_checkpoint=4,
             selective_cache_mib=2684,selective_cache_by_seq={'3072':2684,'4096':2684})
    e=Engine(ReusingBackend());model=Model(e,cfg,activation_dtype='fp32');opt=Optimizer(e,model.parameters())
    # Test metadata migration on the CPU; actual compressed kernels are audited
    # separately by experiments.validate_native_cache_resume on CUDA.
    model.config['native_mxfp8_cache_ffn']=True
    original=tmp_path/'original.npz';checkpoint.save(original,model,opt,{'steps':7})
    model.config.update(selective_cache_mib=6144,selective_cache_by_seq={'3072':6144,'4096':6144})
    checkpoint.load(original,model,opt)
    reference=dict(selective_cache_mib=2684,selective_cache_by_seq={'3072':2684,'4096':2684})
    assert model.cache_precision_reference==reference
    expanded=tmp_path/'expanded.npz';checkpoint.save(expanded,model,opt,{'steps':7})
    assert checkpoint.metadata(expanded)['cache_precision_reference']==reference
    del model.cache_precision_reference
    assert checkpoint.load(expanded,model,opt)=={'steps':7}
    assert model.cache_precision_reference==reference
    model.config.update(selective_cache_mib=1024,selective_cache_by_seq={'3072':1024,'4096':1024})
    with pytest.raises(ValueError,match='precision reference'):checkpoint.load(expanded,model,opt)
