import os
import numpy as np
import pytest
from model.dense_cublas.packing import compact_packed_batch,document_max_length


def sample():
    ids=np.arange(24,dtype=np.int64).reshape(3,8)%61+1
    segments=np.asarray([[0,0,0,1,1,-1,-1,-1],[0,0,0,0,0,0,-1,-1],[0,0,0,0,-1,-1,-1,-1]],np.int64)
    labels=ids.copy();labels[segments<0]=-100
    for i,row in enumerate(segments):
        labels[i,np.r_[True,row[1:]!=row[:-1]]]=-100
    return ids,labels,segments


def test_compaction_preserves_tokens_targets_and_document_lengths():
    original=sample();compact=compact_packed_batch(original,0,4)
    assert compact[0].shape==(1,16)
    np.testing.assert_array_equal(compact[0][compact[2]>=0],original[0][original[2]>=0])
    np.testing.assert_array_equal(compact[1][compact[2]>=0],original[1][original[2]>=0])
    assert np.count_nonzero(compact[1][:,1:]!=-100)==np.count_nonzero(original[1][:,1:]!=-100)
    assert document_max_length(compact[2])==document_max_length(original[2])==6
    assert tuple(compact[2][0,[0,3,5,11]])==(0,1,2,3)
    assert compact_packed_batch(original,0,32) is original


def test_compaction_rejects_interior_holes_and_active_document_starts():
    ids,labels,segments=sample();segments[0,1]=-1
    with pytest.raises(ValueError,match='interior holes'):compact_packed_batch((ids,labels,segments),0,4)
    ids,labels,segments=sample();labels[1,0]=1
    with pytest.raises(ValueError,match='starts'):compact_packed_batch((ids,labels,segments),0,4)


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1',reason='CPU kernel interpreter')
@pytest.mark.parametrize('shared_capacity',[False,True])
def test_compacted_physical_rows_preserve_model_loss_and_all_gradients(shared_capacity):
    from tests.test_native_optimization import ReusingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    config=dict(hidden_size=32,num_hidden_layers=2,vocab_size=67,intermediate_size=48,
                num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=8,use_grad_checkpoint=4)
    batches=(sample(),compact_packed_batch(sample(),0,4));results=[]
    for batch in batches:
        b=ReusingBackend()
        if shared_capacity:b.activation_token_capacities=[16,24]
        e=Engine(b);model=Model(e,config,activation_dtype='fp32',seed=149)
        out=model(batch[0],batch[1],sequence_ids=batch[2])
        results.append((e.numpy(out.loss),{p.name:e.numpy(g) for p,g in e.vjp(out.loss).items()}))
    np.testing.assert_allclose(results[1][0],results[0][0],atol=1e-6,rtol=1e-6)
    assert results[0][1].keys()==results[1][1].keys()
    for name,value in results[0][1].items():np.testing.assert_allclose(results[1][1][name],value,atol=2e-6,rtol=2e-5,err_msg=name)
    with pytest.raises(ValueError,match='packed document'):
        model(np.ones((1,16),np.int64),sequence_ids=np.zeros((1,16),np.int64))


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1',reason='CPU kernel interpreter')
def test_offset_metadata_matches_scan_for_compacted_long_rows():
    from tests.test_native_optimization import ReusingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.packing import metadata
    b=ReusingBackend();e=Engine(b);ids,labels,segments=compact_packed_batch(sample(),0,4)
    seq=e.array(segments);target=e.array(labels);heads,dim=2,16
    starts=np.asarray([0,3,5,11,16],np.int64);lengths=np.asarray([3,2,6,4],np.int64)
    offsets=e.array(starts*heads*dim).data;lens=e.ops.cast(e.array(lengths).data,'i32')
    context=(4,8,heads,1,dim,16,[offsets,offsets,offsets,lens])
    expected=metadata(e,seq,target)
    actual=metadata(e,seq,target,native_context=context)
    for x,y in zip(actual,expected):np.testing.assert_array_equal(e.numpy(x),e.numpy(y))


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1',reason='CPU kernel interpreter')
def test_boundary_norm_reuses_a_logical_prefix_across_token_counts():
    from tests.test_native_optimization import ReusingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    config=dict(hidden_size=32,num_hidden_layers=2,vocab_size=67,intermediate_size=64,
                num_attention_heads=2,num_key_value_heads=1,head_dim=16,max_position_embeddings=8,use_grad_checkpoint=4)
    results=[]
    for enabled in (False,True):
        b=ReusingBackend();b.fused_boundary_norm=enabled;b.fused_norm_cast=True
        e=Engine(b,fp8=False);model=Model(e,config,param_dtype='bf16',muon_mantissa=True,seed=149)
        runs=[];pointer=None
        for batch in (2,1,2):
            mark=b.mark();ids=np.arange(batch*8,dtype=np.int64).reshape(batch,8)%60+1
            out=model(ids,ids)
            runs.append((e.numpy(out.loss),{p.name:e.numpy(g) for p,g in e.vjp(out.loss).items()}))
            if enabled:
                assert model._boundary_active
                storage=model._boundary_scratch[0].storage
                assert pointer is None or pointer==storage
                pointer=storage
            b.release_since(mark)
        results.append(runs)
    for actual,expected in zip(results[1],results[0]):
        np.testing.assert_array_equal(actual[0],expected[0])
        for name,value in actual[1].items():np.testing.assert_array_equal(value,expected[1][name],err_msg=name)
