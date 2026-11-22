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


def test_native_longrope_position_mode_tables_are_position_unique():
    from model.dense_cublas.rope import longrope_tables, mixed_longrope_tables
    config=dict(head_dim=8,max_position_embeddings=128,rope_theta=10000.,rope_scaling=scaling())
    long_tables=longrope_tables(config,True)
    short_tables=longrope_tables(config,False)
    mixed=mixed_longrope_tables(config)
    np.testing.assert_allclose(mixed[0][:4],short_tables[0][:4])
    np.testing.assert_allclose(mixed[1][:4],short_tables[1][:4])
    np.testing.assert_allclose(mixed[0][4:],long_tables[0][4:])
    np.testing.assert_allclose(mixed[1][4:],long_tables[1][4:])
    validate_longrope(scaling()|{'switch_mode':'batch'},8,128)
    validate_longrope(scaling()|{'switch_mode':'position'},8,128)
    with pytest.raises(ValueError):
        validate_longrope(scaling()|{'switch_mode':'token'},8,128)


def test_native_smooth_longrope_is_continuous_and_keeps_prefix():
    from model.dense_cublas.rope import longrope_tables, mixed_longrope_tables, smooth_longrope_tables
    base = scaling()
    cfg = dict(head_dim=8, max_position_embeddings=128, rope_theta=10000.,
               rope_scaling=base | dict(switch_mode='smooth', smooth_start=4, smooth_end=16))
    smooth = smooth_longrope_tables(cfg)
    short_cfg = dict(cfg, rope_scaling=base | dict(switch_mode='batch'))
    short = longrope_tables(short_cfg, False)
    long_ = longrope_tables(short_cfg, True)
    np.testing.assert_allclose(smooth[0][:4], short[0][:4], rtol=1e-6)
    np.testing.assert_allclose(smooth[1][:4], short[1][:4], rtol=1e-6)
    # Beyond smooth_end the phase runs parallel to the long table (a constant offset that
    # RoPE never sees), so compare the relative rotation between positions, not the table.
    for q, k in ((16, 16), (17, 16), (40, 20), (127, 16)):
        np.testing.assert_allclose(_relative(smooth, q, k), _relative(long_, q, k), atol=5e-5)
    frequencies = 1. / (10000. ** (np.arange(0, 8, 2, dtype=np.float32) / 8))
    np.testing.assert_allclose(smooth[0][4], np.tile(np.cos(4 * frequencies), (1, 2))[0],
                               rtol=2e-6, atol=2e-6)
    assert not np.allclose(smooth[0][4], long_[0][4])
    smooth_steps = np.abs(np.diff(smooth[0], axis=0)).max(axis=1)
    assert smooth_steps[3] <= 3.0 * max(smooth_steps[2], smooth_steps[4])
    with pytest.raises(ValueError):
        validate_longrope(base | {'switch_mode': 'smooth', 'smooth_start': 4,
                                  'smooth_end': 3}, 8, 128)


def _relative(tables, q, k):
    cos, sin = tables
    return np.stack([cos[q] * cos[k] + sin[q] * sin[k], sin[q] * cos[k] - cos[q] * sin[k]])


def test_native_smooth_phase_is_monotone_and_parallel_to_the_long_table():
    """Regression: interpolating the factor and multiplying by position (p * w / F(p)) makes the
    phase of every pair with long factor > e run backwards inside the ramp."""
    from model.dense_cublas.rope import smooth_longrope_phase
    head_dim, original, maximum, theta = 96, 4096, 65536, 1e6
    long_factor = np.concatenate([np.ones(11), np.geomspace(1.08, 7.43, 12), np.full(25, 16.)])
    cfg = dict(head_dim=head_dim, max_position_embeddings=maximum, rope_theta=theta,
               rope_scaling=dict(type='longrope', factor=16., original_max_position_embeddings=original,
                                 short_factor=[1.] * 48, long_factor=long_factor.tolist(),
                                 short_attention_factor=1., switch_mode='smooth'))
    phase, amplitude = smooth_longrope_phase(cfg)
    w = 1. / theta ** (np.arange(0, head_dim, 2) / head_dim)
    start, stop = original, 2 * original
    assert phase.shape == (maximum, 48)
    assert (np.diff(phase, axis=0) > 0).all()                       # strictly increasing for every pair
    np.testing.assert_allclose(phase[:start + 1], np.arange(start + 1)[:, None] * w, rtol=1e-12)
    for q, k in ((20000, 9000), (60000, 30000), (stop, stop + 1)):  # parallel to the long table beyond the ramp
        np.testing.assert_allclose(phase[q] - phase[k], (q - k) * w / long_factor, rtol=1e-9, atol=1e-12)
    step = np.diff(phase, axis=0)
    assert np.abs(step[start] / step[start - 1] - 1).max() < 1e-3   # continuous slope at the start of the ramp
    assert amplitude[start] == 1. and abs(amplitude[stop] - np.sqrt(1 + np.log(16.) / np.log(original))) < 1e-12
    # the formula this replaces, for the record: tail pair 40 ends the ramp far behind where it started
    t = np.clip((np.arange(maximum) - start) / (stop - start), 0, 1)
    old = np.arange(maximum) * w[40] / long_factor[40] ** t
    assert old[stop] < old[start]
    assert phase[stop, 40] > phase[start, 40]


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
