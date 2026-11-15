"""Logical tensor sizes remain independent of shared allocation capacity."""
import ctypes as ct
from types import SimpleNamespace
import pytest
from model.dense_cublas.memory import TrainingBackend


class Allocator(TrainingBackend):
    def __init__(self):
        super().__init__(pool_limit_mib=64)
        self.arch=120;self.pointer=256;self.sizes=[];self.freed=[]
        def allocate(out,size):
            self.pointer+=256;ct.cast(out,ct.POINTER(ct.c_uint64))[0]=self.pointer
            self.sizes.append(size)
        self._malloc=allocate;self._free=self.freed.append
    def _check(self):pass
    def synchronize(self):pass


def test_capacity_reuses_storage_across_shapes_and_accounts_allocated_bytes():
    b=Allocator();b.activation_token_capacities=[9,13,17]
    x=b.empty((9,32),'bf16')
    assert x.shape==(9,32) and x.numel==288
    assert b.sizes==[17*32*2] and b._live_bytes==17*32*2
    assert next(iter(b._allocations))[1]==17*32*2
    b.release_owned((x,))
    y=b.empty((32,13),'bf16')
    assert y.storage==x.storage and y.shape==(32,13) and len(b.sizes)==1
    assert b._live_bytes==b._peak_live_bytes==17*32*2 and b._pool_bytes==0
    b.pool_limit=0;b.release_owned((y,))
    assert b._live_bytes==b._pool_bytes==0 and b.freed==[x.storage]


def test_capacity_is_explicit_and_rejects_invalid_counts():
    b=Allocator()
    assert b.capacity_shape((13,32))==(13,32)
    for invalid in ([0],[-1],[1.5],[True]):
        with pytest.raises(ValueError,match='positive integers'):b.activation_token_capacities=invalid
    b.activation_token_capacities=[9728,12288]
    assert b.capacity_shape((5504,1536))==(5504,1536)
    assert b.capacity_shape((1,9728,1536))==(1,12288,1536)


def test_mx_scale_capacity_is_rounded_in_both_orientations_without_changing_layout_metadata():
    from model.dense_cublas.mxfp8 import new_scales
    b=Allocator();b.activation_token_capacities=[9728,12288]
    ops=SimpleNamespace(backend=b,empty=b.empty)
    for shape in ((9728,5504),(5504,9728)):
        s=new_scales(ops,shape)
        assert s.matrix_shape==shape
        assert s.numel*2==(12288//128)*(5504//128)*512


def test_selective_plan_charges_allocated_capacity_not_compact_token_count():
    from model.dense_cublas.selective import recipe_plan
    config=dict(hidden_size=1536,num_attention_heads=16,num_key_value_heads=4,head_dim=96,
                intermediate_size=5504,num_hidden_layers=32,selective_cache_mib=2817,dropout=0.)
    def plans(shape,capacity):
        return [recipe_plan(config,shape,'bf16',i,attention_min_seq=1,auxiliary_kv=True,
                            compressed_ffn=True,storage_tokens=capacity) for i in range(32)]
    original=plans((3,4096,1536),12288)
    compact=plans((1,9728,1536),12288)
    assert [(p.linear_names,p.attention) for p in compact]==[(p.linear_names,p.attention) for p in original]
    assert sum(any('gate_proj' in n for n in p.linear_names) for p in compact)==8
    with pytest.raises(ValueError,match='capacity'):
        plans((1,9728,1536),9000)
