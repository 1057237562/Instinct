"""Production profile selection, hardware guards and data cursor conservation."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from model.dense_cublas import training_profile as profile


def production_args():
    from scripts.native_v2_recipe import command
    from trainer.train_cublas import parser
    return parser().parse_args(command('production.parquet')[3:])


def test_shared_recipe_selects_the_production_profile_and_single_catalog():
    from trainer.train_cublas import read_config
    from experiments.native_candidate import read
    args=production_args();config=read_config(args)
    assert args.compute_profile=='v2_mxfp8' and args.selective_cache_mib==2684
    policy=profile.validate(args.compute_profile,args,config)
    assert policy['compact_long_multiple']==512
    assert read('experiments/native_mxfp8_candidate.json')==policy['backend']
    assert policy['backend']['mxfp8_gradient_dtype']=='fp8e4nv'
    assert policy['backend']['muon_gram_recurrence'] and policy['backend']['frost_native_head96']


@pytest.mark.parametrize('key,value',[('fp8','0'),('fp8_filter','eligible'),('dtype','float16'),
    ('optimizer','adamw'),('fused_ops',False),('packed_attention',False),('max_seq_len',3072),
    ('overlap_dw',True),('sequence_buckets','4096'),('use_grad_checkpoint',2)])
def test_incompatible_profile_is_rejected_before_cuda(key,value):
    from trainer.train_cublas import read_config
    args=production_args();setattr(args,key,value)
    with pytest.raises(ValueError):read_config(args)


def test_tensorwise_remains_the_general_cli_default_and_does_not_override_probes():
    from trainer.train_cublas import parser,read_config
    args=parser().parse_args([]);assert args.compute_profile=='tensorwise'
    read_config(args)
    b=SimpleNamespace(mxfp8_training=True)
    profile.apply(b,'tensorwise',profile.settings('tensorwise'))
    assert b.mxfp8_training


def test_hardware_and_bridge_checks_fail_closed():
    policy=profile.settings('v2_mxfp8')
    with pytest.raises(ValueError,match='SM120'):profile.apply(SimpleNamespace(arch=89),'v2_mxfp8',policy)
    with pytest.raises(RuntimeError,match='bridge'):profile.apply(SimpleNamespace(arch=120,native=None),'v2_mxfp8',policy)


def test_profile_application_leaves_attention_initialization_to_normal_warmup():
    b=SimpleNamespace(arch=120,native=SimpleNamespace(_plan_mxfp8=object(),_plan_pointer_gram_ex=object()),kernel_options={})
    profile.apply(b,'v2_mxfp8',profile.settings('v2_mxfp8'))
    assert not hasattr(b,'_profile_frost')
    assert b.attention_tile==32 and b.attention_forward_tile==128 and b.attention_forward_cols==64
    assert b.kernel_options['model.dense_triton.attention.forward']==dict(num_warps=8,num_stages=2)


def test_production_loader_preserves_targets_and_resume_cursor_and_keeps_tail_batches():
    from model.dense_cublas.packing import document_max_length
    def batch(rows):
        ids=np.ones((rows,4096),np.int64);segments=np.zeros_like(ids);segments[:,3400:]=-1
        labels=ids.copy();labels[:,0]=-100;labels[segments<0]=-100
        return ids,labels,segments
    items=[(dict(group=1,batch=i+1),batch(rows)) for i,rows in enumerate((3,2,1))]
    loader=SimpleNamespace(batches=lambda *args:iter(items))
    profile.prepare_data(loader,0,profile.settings('v2_mxfp8'));actual=list(loader.batches(0))
    assert actual[0][1][0].shape==(1,10240)
    assert actual[1][1] is items[1][1] and actual[2][1] is items[2][1]
    for (a,x),(b,y) in zip(actual,items):
        assert a is b
        assert np.count_nonzero(x[1][:,1:]!=-100)==np.count_nonzero(y[1][:,1:]!=-100)
        assert document_max_length(x[2])==3400


def test_experimental_checkpoint_is_recognized_by_its_saved_precision_contract():
    config=dict(native_fp8_recipe='mxfp8_experimental',native_mxfp8_gradient_dtype='fp8e4nv',
                native_mxfp8_small_projections=True,native_mxfp8_cache_ffn=True,native_muon_gram_recurrence='fp16_restart3')
    assert profile.checkpoint_profile(config)=='v2_mxfp8'
    assert profile.checkpoint_profile({})=='tensorwise'
    assert profile.checkpoint_profile(config|{'native_mxfp8_gradient_dtype':'fp8e5'})!='v2_mxfp8'

def test_prewarm_skips_resume_with_pending_gradients():
    from types import SimpleNamespace
    from model.dense_cublas.training_profile import prewarm

    optimizer = SimpleNamespace(accumulated=3)
    assert prewarm(None, optimizer, {'compact_long_multiple': 512}) is None
    assert optimizer.accumulated == 3
