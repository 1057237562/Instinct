"""Native execution controls and cross-stream input-lifetime contracts."""
import ctypes as ct
from contextlib import contextmanager
import numpy as np
import pytest
from model.dense_cublas.selective import parse_cache_budgets, recipe_plan
from model.dense_cublas.model import configuration


def test_per_bucket_budget_changes_only_the_selected_length():
    cfg=configuration(dict(hidden_size=1536,num_hidden_layers=32,intermediate_size=5504,
        num_attention_heads=16,num_key_value_heads=4,head_dim=96,selective_cache_mib=768,
        selective_cache_by_seq=parse_cache_budgets('4096:1024')))
    long=[recipe_plan(cfg,(3,4096,1536),'bf16',i) for i in range(32)]
    short=[recipe_plan(cfg,(4,3072,1536),'bf16',i) for i in range(32)]
    assert sum(p.attention for p in long)==27
    assert all(not p.attention and len(p.linear_names)==2 for p in short)


@pytest.mark.parametrize('text',['0:768','4096:-1','4096:nan','4096:inf','4096:768,4096:1024','4096'])
def test_invalid_bucket_budgets_are_rejected(text):
    with pytest.raises(ValueError):parse_cache_budgets(text)


def test_bucket_budgets_survive_webui_json_export():
    from tests.test_cublas_webui_contract import exported_config
    from model.dense_cublas.config import WebUIRequest
    exported=exported_config(dict(runtime_backend='cublas',native_selective_cache_by_seq='3072:768,4096:1024'))
    assert exported['selective_cache_by_seq']=={'3072':768,'4096':1024}
    assert WebUIRequest.from_dicts(exported).model['selective_cache_by_seq']==exported['selective_cache_by_seq']


def test_prefetch_does_not_overwrite_pending_device_readers():
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.prefetch import DevicePrefetch
    from model.dense_triton.runtime import Tensor
    class Backend:
        def __init__(self):
            self.storage=[];self.pending=[];self.events={};self.freed=[];self.stream=1;self.syncs=0
        def _bind(self,name,types):
            if name=='cuMemHostAlloc':
                def alloc(output,size,flags):
                    block=ct.create_string_buffer(size);self.storage.append(block)
                    ct.cast(output,ct.POINTER(ct.c_void_p))[0]=ct.addressof(block)
                return alloc
            if name=='cuMemFreeHost':return lambda pointer:self.freed.append(pointer.value)
            if name=='cuMemcpyHtoDAsync_v2':return lambda dst,src,size,stream:self.pending.append((dst,src,size))
            raise AssertionError(name)
        def empty(self,shape,dtype):
            block=ct.create_string_buffer(int(np.prod(shape))*8);self.storage.append(block)
            return Tensor(shape,dtype,ct.addressof(block),self)
        def _make_stream(self,priority):return 2
        def create_event(self):
            event=len(self.events)+1;self.events[event]=[];return event
        def record_event(self,event):self.events[event]=self.pending;self.pending=[]
        def wait_event(self,event):
            for dst,src,size in self.events[event]:ct.memmove(dst,src,size)
            self.events[event]=[]
        def _event_sync(self,event):self.wait_event(event)
        def synchronize(self):
            self.syncs+=1
            for event in self.events:self.wait_event(event)
        @contextmanager
        def on_stream(self,stream):
            old,self.stream=self.stream,stream
            try:yield
            finally:self.stream=old
    b=Backend();e=Engine(b);captured=[]
    source=[(i,tuple(np.full((1,3+i%2),10*i+j,np.int64) for j in range(3))) for i in range(7)]
    with DevicePrefetch(e,1,4,True,fields=3) as prefetch:
        for _,cpu,device in prefetch.batches(source):
            for expected,value in zip(cpu,device):
                target=ct.create_string_buffer(expected.nbytes);b.storage.append(target)
                # Simulate a queued compute read that must finish before the
                # copy stream replaces this device slot's contents.
                b.pending.append((ct.addressof(target),value.data.storage,expected.nbytes))
                captured.append((target,expected.copy()))
            assert b.syncs==0
    for target,expected in captured:
        np.testing.assert_array_equal(np.frombuffer(target,dtype=np.int64),expected.ravel())
    assert b.syncs==1 and len(b.freed)==6
