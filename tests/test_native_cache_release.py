"""Ownership and allocator accounting for consumed private replay buffers."""
from types import SimpleNamespace
import pytest


def backend():
    from model.dense_cublas.memory import TrainingBackend
    b=TrainingBackend.__new__(TrainingBackend)
    b.stream=b.main_stream=b.weight_stream=13;b._pinned={1000}
    b._live={1000:(1,64),2000:(2,128),3000:(3,256)}
    b._live_bytes=448;b._pool_bytes=0;b._pool={};b._commands=None
    b.pool_limit=256;b._allocations={1000,2000,3000};b.freed=[];b._free=b.freed.append
    b.allocator_stats={'release_syncs':0}
    return b


def test_release_owned_deduplicates_and_keeps_parameters():
    b=backend();x=SimpleNamespace(storage=2000,owner=b)
    b.release_owned((x,x))
    assert b._live_bytes==320 and b._pool_bytes==128 and b._pool[128]==[2000]
    assert 1000 in b._live and b.allocator_stats['release_syncs']==0
    with pytest.raises(ValueError):b.release_owned((x,))


@pytest.mark.parametrize('pointer,foreign',[(1000,False),(2128,False),(2000,True)])
def test_release_owned_rejects_parameters_views_and_foreign_buffers(pointer,foreign):
    b=backend()
    with pytest.raises(ValueError):b.release_owned((SimpleNamespace(storage=pointer,owner=None if foreign else b),))
    assert b._live_bytes==448 and b._pool_bytes==0


def test_release_owned_respects_pool_limit_and_capture_lifetime():
    b=backend();b.pool_limit=0;b._commands=[]
    b.release_owned((SimpleNamespace(storage=3000,owner=b),))
    assert b._pool_bytes==256 and not b.freed
    b._commands=None;b._trim_pool()
    assert b._pool_bytes==0 and b.freed==[3000] and 3000 not in b._allocations


def test_pool_reuse_needs_no_cuda_call_but_rejects_inactive_backend():
    b=backend();b._context=object();b._serial=3;b._uploading_constant=False;b._peak_live_bytes=448
    b.allocator_stats.update(requests=0,pool_hits=0)
    b.release_owned((SimpleNamespace(storage=2000,owner=b),))
    def forbidden():raise AssertionError('pool metadata reuse must not call CUDA')
    b._check=forbidden
    value=b.empty((32,),'fp32')
    assert value.storage==2000 and value.owner is b and b._live_bytes==448
    b.release_owned((value,));b._context=None
    from model.dense_cublas.memory import TrainingBackend
    b._check=TrainingBackend._check.__get__(b)
    with pytest.raises(RuntimeError,match='not active'):b.empty((32,),'fp32')
    assert b._pool[128]==[2000]
