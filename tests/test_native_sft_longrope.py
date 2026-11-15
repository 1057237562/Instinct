import json
import os
from types import SimpleNamespace
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from model.dense_cublas.rope import longrope_tables, validate_longrope
from model.dense_cublas.config import WebUIRequest
from model.dense_cublas.sft_data import PackedSFT


def scaling():
    return dict(type='longrope', factor=32., original_max_position_embeddings=4,
                short_factor=[1.] * 4, long_factor=[1., 2., 8., 32.],
                short_attention_factor=1., long_retained_start_tokens=2)


def test_native_longrope_frequencies_retained_positions_and_short_table():
    config=dict(head_dim=8,max_position_embeddings=128,rope_theta=10000.,rope_scaling=scaling())
    short=longrope_tables(config,False)
    long=longrope_tables(config,True)
    assert short[0].shape == (4,8)
    np.testing.assert_allclose(long[0][:2],short[0][:2])
    frequencies=1./10000.**(np.arange(0,8,2,dtype=np.float32)/8)
    amplitude=np.sqrt(1+np.log(32)/np.log(4))
    expected=np.cos(127*frequencies/np.array([1.,2.,8.,32.]))*amplitude
    np.testing.assert_allclose(long[0][127,:4],expected,rtol=2e-6,atol=2e-6)
    request=WebUIRequest.from_dicts(dict(hidden_size=32,num_attention_heads=4,
        num_key_value_heads=2,head_dim=8,max_position_embeddings=128,
        freeze_embeddings=True,rope_scaling=scaling()))
    assert request.model['freeze_embeddings']
    with pytest.raises(ValueError):
        validate_longrope(scaling()|{'long_factor':[2.]},8,128)


def test_packed_sft_cache_preserves_labels_boundaries_and_padding(tmp_path):
    row=dict(input_ids=[1,7,2,1,8,2,0,0],loss_mask=[False,True,True,False,True,True,False,False],
             segment_lengths=[3,3],valid_tokens=6,block_length=8)
    table=pa.Table.from_pylist([row,row])
    shard=tmp_path/'part-00000.arrow'
    with pa.OSFile(str(shard),'wb') as sink, ipc.new_stream(sink,table.schema) as writer:
        writer.write_table(table)
    manifest=dict(identity=dict(kind='sft',format=2),files=[dict(name=shard.name,bytes=shard.stat().st_size)],
                  bucket_ranges=[dict(max_length=8,blocks=2)])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    data=PackedSFT(tmp_path,SimpleNamespace(pad=0),12)
    ids,labels,segments=data.batch([1,0])
    np.testing.assert_array_equal(ids[0,:8],row['input_ids'])
    np.testing.assert_array_equal(labels[0,:6],[-100,7,2,-100,8,2])
    np.testing.assert_array_equal(segments[0],[0,0,0,1,1,1,-1,-1,-1,-1,-1,-1])
    assert np.count_nonzero(labels!=-100)==8
    assert data.fingerprint==PackedSFT(tmp_path,SimpleNamespace(pad=0),12).fingerprint


def test_sft_profile_uses_existing_mxfp8_catalog_with_16k_capacity():
    from model.dense_cublas.training_profile import settings
    pretrain=settings('v2_mxfp8')
    sft=settings('v2_mxfp8_sft')
    assert sft['backend']['activation_token_capacities']==[16384]
    assert sft['backend']['mxfp8_training']
    assert pretrain['backend']['activation_token_capacities'][-1]==12288
    assert sft['compact_long_multiple']==0


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1', reason='CPU interpreter')
def test_frozen_linear_ce_preserves_input_gradient():
    import torch
    from tests.test_cublas_training import HostBackend
    from model.dense_cublas.autograd import Engine
    e=Engine(HostBackend())
    rng=np.random.default_rng(21)
    x=rng.normal(size=(1,7,8)).astype(np.float32)
    w=rng.normal(size=(13,8)).astype(np.float32)
    labels=np.array([[-100,-100,1,3,5,-100,2]])
    tx=torch.tensor(x,requires_grad=True)
    expected=torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(tx,torch.tensor(w)).reshape(-1,13),
        torch.tensor(labels).reshape(-1))
    expected.backward()
    nx=e.array(x,requires_grad=True)
    nw=e.array(w,requires_grad=False)
    loss=e.linear_ce(nx,nw,e.array(labels),3)
    gradients=e.vjp(loss)
    assert nw not in gradients
    np.testing.assert_allclose(e.numpy(gradients[nx]),tx.grad.numpy(),atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(e.item(loss),expected.item(),atol=2e-5)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1', reason='CPU interpreter')
def test_native_model_freezes_shared_embedding_and_runs_longrope_ckpt4():
    from tests.test_cublas_training import HostBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.checkpoint import validate_resume_config
    e=Engine(HostBackend())
    config=dict(hidden_size=8,num_hidden_layers=2,vocab_size=13,
                num_attention_heads=1,num_key_value_heads=1,head_dim=8,
                intermediate_size=12,max_position_embeddings=128,flash_attn=False,
                freeze_embeddings=True,use_grad_checkpoint=4,rope_scaling=scaling())
    model=Model(e,config,activation_dtype='fp32')
    embedding=model.params['model.embed_tokens.weight']
    assert model.params['lm_head.weight'] is embedding
    assert embedding not in model.parameters()
    model.loss_chunk_size=3
    ids=np.array([[1,3,4,5,6,7]])
    output=model(ids,ids,return_logits=False)
    gradients=e.vjp(output.loss)
    assert embedding not in gradients
    assert np.any(e.numpy(gradients[model.params['model.layers.0.mlp.down_proj.weight']])!=0)
    assert validate_resume_config({}, {'freeze_embeddings':False}) is None
    with pytest.raises(ValueError):validate_resume_config(config,config|{'freeze_embeddings':False})
