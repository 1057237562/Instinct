"""Native execution controls and cross-stream input-lifetime contracts."""
import ctypes as ct
from contextlib import contextmanager
import numpy as np
import pytest
from model.dense_cublas.selective import parse_cache_budgets, recipe_plan
from model.dense_cublas.model import configuration


def test_async_pool_retention_cli_preserves_default_and_rejects_negative_budget():
    from trainer.train_cublas import parser
    from model.dense_cublas.memory import TrainingBackend
    assert parser().parse_args([]).allocator_release_mib is None
    args=parser().parse_args(['--allocator_backend','async','--allocator_release_mib','13312'])
    backend=TrainingBackend(allocator_backend=args.allocator_backend,async_release_mib=args.allocator_release_mib)
    assert backend.async_release_mib==13312
    with pytest.raises(ValueError,match='nonnegative'):TrainingBackend(async_release_mib=-1)


def test_production_fusion_policy_preserves_explicit_probe_overrides():
    from types import SimpleNamespace
    from model.dense_cublas.policy import enable_fused_ops
    backend=SimpleNamespace(aligned_pointers=False)
    enable_fused_ops(backend)
    assert backend.fused_norm_backward and backend.fused_residual_norm
    assert backend.fused_scale_inverse and backend.attention_local_maxima
    assert backend.fused_residual_backward and backend.fused_cast_amax and backend.cache_weight_scales
    assert backend.attention_group_penalty==4096
    assert backend.ffn_flat_forward and backend.ffn_quant_tile==64
    assert backend.norm_rope_rows==8 and backend.attention_docs_per_plan==4
    assert backend.attention_plan_quantum==256 and backend.attention_plan_limit==128 and backend.attention_shared_handle
    assert backend.muon_tiled_io and backend.muon_symmetric and backend.quant_tile==64
    assert backend.selective_aux_kv and not backend.attention_tune and not backend.attention_prewarm
    assert backend.aligned_pointers and backend.aligned_pointwise_only
    override=SimpleNamespace(fused_norm_backward=False,ffn_quant_tile=32,
                             fused_residual_backward=False,fused_cast_amax=False,cache_weight_scales=False,
                             attention_group_penalty=0,
                             attention_plan_quantum=512,attention_plan_limit=64,attention_shared_handle=False,
                             fused_scale_inverse=False,attention_local_maxima=False,
                             aligned_pointers=False,aligned_pointwise_only=False)
    enable_fused_ops(override)
    assert not override.fused_norm_backward and override.ffn_quant_tile==32
    assert not override.fused_scale_inverse and not override.attention_local_maxima
    assert not override.fused_residual_backward and not override.fused_cast_amax and not override.cache_weight_scales
    assert override.attention_group_penalty==0
    assert override.attention_plan_quantum==512 and override.attention_plan_limit==64 and not override.attention_shared_handle
    assert not override.aligned_pointers and not override.aligned_pointwise_only


def test_attention_local_maxima_preserve_ragged_offsets_and_total_address_space():
    from types import SimpleNamespace
    from model.dense_cublas.vendor_attention import NativeAttention
    from model.dense_triton.runtime import Tensor
    native=NativeAttention.__new__(NativeAttention)
    native.b=SimpleNamespace(attention_docs_per_plan=2,attention_plan_quantum=1024,attention_local_maxima=True)
    metadata=[Tensor((6,),dtype,1000+i*100,native.b) for i,dtype in enumerate(('i64','i64','i64','i32'))]
    context=(5,4096,16,4,96,12288,metadata,(3900,25,600,900,15))
    chunks=list(native.chunks(context))
    assert [(x[0],x[1]) for x in chunks]==[(2,4096),(2,1024),(1,1024)]
    assert all(x[5]==12288 for x in chunks)
    assert chunks[1][6][0].storage==1016 and chunks[1][6][3].storage==1308
    assert chunks[2][6][0].shape==(2,) and chunks[2][6][3].shape==(1,)
    native.b.attention_local_maxima=False
    assert all(x[1]==4096 for x in native.chunks(context))


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
