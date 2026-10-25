"""Native V2 recipe contracts and numerical tests using recycled CPU storage."""
import os
import datasets  # DLL import order on Windows.
import numpy as np
import pytest
import torch
from model.dense_cublas.autograd import Engine
from model.dense_cublas.model import Model, configuration
from model.dense_cublas.selective import recipe_plan
from tests.test_native_optimization import ReusingBackend

INTERPRET = pytest.mark.skipif(os.environ.get('TRITON_INTERPRET') != '1', reason='CPU interpreter')
CFG = dict(hidden_size=32, num_hidden_layers=2, vocab_size=67, intermediate_size=48,
           num_attention_heads=2, num_key_value_heads=1, head_dim=16,
           max_position_embeddings=4096, use_grad_checkpoint=0)


def test_mode4_v2_plans_match_candidate_budget():
    cfg = configuration(dict(hidden_size=1536,num_hidden_layers=32,intermediate_size=5504,
                             num_attention_heads=16,num_key_value_heads=4,head_dim=96))
    for shape, attention, gates in [((4,3072,1536),False,4),((3,4096,1536),True,2)]:
        plans = [recipe_plan(cfg,shape,'bf16',i) for i in range(32)]
        assert sum(p.attention for p in plans) == (32 if attention else 0)
        assert sum(any('gate_proj' in n for n in p.linear_names) for p in plans) == gates
        assert sum(any('k_proj' in n for n in p.linear_names) for p in plans) == (0 if attention else 32)
    cfg['selective_cache_mib']=768
    short=[recipe_plan(cfg,(4,3072,1536),'bf16',i) for i in range(32)]
    long=[recipe_plan(cfg,(3,4096,1536),'bf16',i) for i in range(32)]
    assert sum(any('k_proj' in n for n in p.linear_names) for p in short)==32
    assert sum(p.attention for p in long)==20
    assert all(not any('gate_proj' in n for n in p.linear_names) for p in short+long)


@INTERPRET
@pytest.mark.parametrize('mode',[2,3,4])
def test_selective_replay_loss_and_every_gradient(mode):
    ids=np.arange(14,dtype=np.int64).reshape(2,7)+1
    results=[]
    for checkpoint in (0,mode):
        e=Engine(ReusingBackend());m=Model(e,CFG|{'use_grad_checkpoint':checkpoint},activation_dtype='fp32')
        out=m(ids,ids);grads=e.vjp(out.loss)
        results.append((e.numpy(out.loss),{p.name:e.numpy(g) for p,g in grads.items()}))
    np.testing.assert_allclose(results[1][0],results[0][0],atol=1e-6,rtol=1e-6)
    assert results[1][1].keys()==results[0][1].keys()
    for name,g in results[0][1].items():
        np.testing.assert_allclose(results[1][1][name],g,atol=1e-5,rtol=1e-4,err_msg=name)


@INTERPRET
def test_packed_matches_independent_documents_loss_and_gradients():
    ids=np.asarray([[1,2,3,4,5,6]],np.int64)
    e=Engine(ReusingBackend());m=Model(e,CFG|{'use_grad_checkpoint':4},activation_dtype='fp32')
    out=m(ids,ids,sequence_ids=np.asarray([[7,7,7,9,9,9]],np.int64))
    packed_loss=e.numpy(out.loss);packed_logits=e.numpy(out.logits)
    grads={p.name:e.numpy(g) for p,g in e.vjp(out.loss).items()}
    losses=[];references={}
    for doc in (ids[:,:3],ids[:,3:]):
        r=Engine(ReusingBackend());model=Model(r,CFG,activation_dtype='fp32')
        result=model(doc,doc);losses.append(r.numpy(result.loss))
        expected=r.numpy(result.logits)
        np.testing.assert_allclose(packed_logits[:,len(losses)*3-3:len(losses)*3],expected,atol=2e-6,rtol=1e-5)
        for p,g in r.vjp(result.loss).items():
            references[p.name]=references.get(p.name,0)+r.numpy(g)/2
    np.testing.assert_allclose(packed_loss,np.mean(losses,axis=0),atol=1e-6)
    for name,g in references.items():
        np.testing.assert_allclose(grads[name],g,atol=1e-5,rtol=1e-4,err_msg=name)


@INTERPRET
def test_packed_range_skip_matches_mask_only_with_unaligned_documents_and_padding():
    ids=np.arange(67,dtype=np.int64)[None]%64+1
    segments=np.asarray([[0]*19+[1]*22+[2]*20+[-1]*6],np.int64)
    results=[]
    for skip in (False,True):
        b=ReusingBackend();b.packed_skip=skip;e=Engine(b)
        m=Model(e,CFG|{'use_grad_checkpoint':4},activation_dtype='fp32')
        out=m(ids,ids,sequence_ids=segments)
        results.append((e.numpy(out.loss),{p.name:e.numpy(g) for p,g in e.vjp(out.loss).items()}))
    np.testing.assert_array_equal(results[1][0],results[0][0])
    for name,g in results[0][1].items():
        np.testing.assert_array_equal(results[1][1][name],g,err_msg=name)


@INTERPRET
def test_packing_metadata_repeated_ids_padding_and_boundaries():
    from model.dense_cublas.packing import metadata
    e=Engine(ReusingBackend())
    ids=e.array(np.asarray([[5,5,8,5,5,-1]],np.int64))
    labels=e.array(np.arange(6,dtype=np.int64)[None])
    segments,positions,targets=metadata(e,ids,labels)
    np.testing.assert_array_equal(e.numpy(segments),[[0,0,2,3,3,-1]])
    np.testing.assert_array_equal(e.numpy(positions),[[0,1,0,0,1,0]])
    np.testing.assert_array_equal(e.numpy(targets),[[-100,1,-100,-100,4,-100]])


@INTERPRET
@pytest.mark.parametrize('momentum_dtype',['fp32','bf16'])
def test_muon_split_state_and_exact_resume(tmp_path,momentum_dtype):
    from model.dense_cublas.optim import Optimizer
    from model.dense_cublas import checkpoint
    e=Engine(ReusingBackend())
    m=Model(e,CFG,param_dtype='bf16',muon_mantissa=True)
    opt=Optimizer(e,m.parameters(),'muon',muon_mantissa=True,muon_momentum_dtype=momentum_dtype,
                  muon_workspace_mb=1,muon_batch_size=2,ns_steps=2,grad_clip=0.)
    # The interpreter cannot dereference BF16 pointer tables correctly; actual
    # updates are checked in the Torch-free CUDA acceptance executable.
    for state in opt.state.values():
        if 'mantissa' in state:e.copy(state['mantissa'],e.full(state['mantissa'].shape,13579.,'u16'))
    path=tmp_path/'state.npz';checkpoint.save(path,m,opt,{'step':1})
    before=checkpoint.snapshot(m,opt,{'step':1})
    for state in opt.state.values():
        if 'mantissa' in state:e.copy(state['mantissa'],e.full(state['mantissa'].shape,23456.,'u16'))
    checkpoint.load(path,m,opt)
    after=checkpoint.snapshot(m,opt,{'step':1})
    for key,value in before.items():np.testing.assert_array_equal(after[key],value,err_msg=key)
    lows=[value for key,value in before.items() if key.endswith('/mantissa')]
    assert lows and all(a.dtype==np.uint16 for a in lows) and any(np.any(a) for a in lows)
    assert opt.muon_workspace_mb==1
    malformed=dict(before)
    key=next(k for k in malformed if k.endswith('/mantissa'))
    malformed[key]=malformed[key].astype(np.float32)
    bad=tmp_path/'wrong_bits.npz';checkpoint.write(bad,malformed)
    with pytest.raises(ValueError,match='raw uint16'):checkpoint.load(bad,m,opt)
    opt.muon_momentum_dtype='bf16' if momentum_dtype=='fp32' else 'fp32'
    with pytest.raises(ValueError,match='precision'):checkpoint.load(path,m,opt)


@INTERPRET
def test_native_pause_saves_at_accumulation_boundary(tmp_path):
    import json
    from trainer.train_cublas import parser,train
    from tests.test_cublas_training import HostBackend,ToyTokenizer
    from model.dense_cublas import checkpoint
    config=tmp_path/'model.json'
    config.write_text(json.dumps(CFG|dict(hidden_size=8,intermediate_size=12,head_dim=4,
                                       num_hidden_layers=1,vocab_size=13,flash_attn=False)))
    data=tmp_path/'data.jsonl';data.write_text('{"text":"abc"}\n'*4)
    directory=tmp_path/'checkpoints';directory.mkdir()
    request=directory/'.pause_request';request.touch()
    args=parser().parse_args(['--config_path',str(config),'--data_path',str(data),
        '--from_weight','none','--dtype','fp32','--fp8','0','--batch_size','1',
        '--max_seq_len','5','--accumulation_steps','2','--num_workers','0',
        '--checkpoint_dir',str(directory),'--save_dir',str(tmp_path/'out')])
    result=train(args,HostBackend(),ToyTokenizer())
    assert result['paused'] and result['steps']==2 and not request.exists()
    meta=checkpoint.metadata(result['resume'])
    assert meta['optimizers']['model']['accumulated']==0
    assert checkpoint.metadata(result['weights'])['progress']==meta['progress']
