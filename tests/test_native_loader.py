"""Spawn workers, deterministic augmentation, bounded prefetch and pin staging."""
import ctypes as ct
import json
from pathlib import Path
import numpy as np
import pytest
from model.dense_cublas.data import JsonlDataset,Tokenizer
from model.dense_cublas.loader import BatchLoader


def _worker_has_torch():
    import sys
    return 'torch' in sys.modules or 'torchao' in sys.modules


def test_spawn_prefetch_matches_serial_and_resume(tmp_path):
    path=tmp_path/'data.jsonl'
    rows=[{'conversations':[{'role':'user','content':f'问题 {i}'},{'role':'assistant','content':'<think>\n\n</think>\n\n回答'}]} for i in range(11)]
    path.write_text('\n'.join(json.dumps(x) for x in rows),encoding='utf-8')
    data=JsonlDataset(path,Tokenizer('model'),'full_sft',64)
    order=[3,6,2,9,4,1,0,5,7,8,10]
    state=json.dumps(data.rng.bit_generator.state)
    with BatchLoader(data,0,seed=11) as serial:
        expected=list(serial.batches(order,3,epoch=2))
    with BatchLoader(data,2,prefetch_factor=1,seed=11) as parallel:
        actual=list(parallel.batches(order,3,epoch=2))
        assert not parallel.executor.submit(_worker_has_torch).result()
        resumed=list(parallel.batches(order,3,epoch=2,cursor=6))
    assert json.dumps(data.rng.bit_generator.state)==state
    for left,right in ((expected,actual),(expected[2:],resumed)):
        for (a,x),(b,y) in zip(left,right):
            assert a==b
            for aa,bb in zip(x,y):np.testing.assert_array_equal(aa,bb)
    assert actual[-1][1][0].shape[0]==2


def test_worker_exception_propagates_and_closes(tmp_path):
    path=tmp_path/'bad.jsonl';path.write_text('{"wrong":"field"}\n')
    data=JsonlDataset(path,Tokenizer('model'),'pretrain',8)
    loader=BatchLoader(data,1)
    with pytest.raises(KeyError,match='text'):
        with loader:list(loader.batches([0],1))
    assert loader.executor is None


def test_pin_staging_reused_without_touching_worker_arrays():
    from model.dense_triton.runtime import CudaBackend,Tensor
    class Driver(CudaBackend):
        def __init__(self):
            super().__init__();self.pin_memory=True;self.allocations=[];self.freed=[];self.uploads=[]
            self._htod=lambda dst,src,n:self.uploads.append(ct.string_at(src,n))
        def empty(self,shape,dtype):return Tensor(shape,dtype,123,self)
        def _bind(self,name,types):
            if name=='cuMemFreeHost':return lambda ptr:self.freed.append(ptr.value)
            def allocate(out,size,flags):
                buf=ct.create_string_buffer(size);self.allocations.append(buf)
                ct.cast(out,ct.POINTER(ct.c_void_p))[0]=ct.addressof(buf)
            return allocate
    d=Driver()
    for n in (12,4,20):
        a=np.arange(n,dtype=np.int64);d.from_numpy(a)
        assert d.uploads[-1]==a.tobytes()
    assert len(d.allocations)==2 and len(d.freed)==1 and d._host_capacity==160


def test_isolated_packing_keeps_all_within_document_targets_and_resume(tmp_path):
    from model.dense_cublas.data import PackedPretrain,trim_batch
    from tests.test_cublas_training import ToyTokenizer
    tokenizer=ToyTokenizer();tokenizer.cache_namespace='test-isolated-v2'
    path=tmp_path/'packed.jsonl'
    texts=['abc','de','fghijkl','mn']
    path.write_text('\n'.join(json.dumps({'text':t}) for t in texts),encoding='utf-8')
    source=JsonlDataset(path,tokenizer,'pretrain',6)
    packed=PackedPretrain(source,tmp_path/'cache',isolate_documents=True)
    order=list(range(len(packed)))
    with BatchLoader(packed,0) as loader:
        all_batches=list(loader.batches(order,2))
        resumed=list(loader.batches(order,2,cursor=2))
    targets=sum(np.count_nonzero(batch[1][:,1:]!=-100) for _,batch in all_batches)
    expected=sum(len(tokenizer.encode(text))+1 for text in texts)
    assert targets==expected
    for (_,a),(_,b) in zip(all_batches[1:],resumed):
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    for _,batch in all_batches:
        ids,labels,segments=batch
        assert np.all(labels[:,1:][segments[:,1:]!=segments[:,:-1]]==-100)
        trimmed=trim_batch(batch,tokenizer.pad,[4,6])
        assert len(trimmed)==3 and all(x.shape==trimmed[0].shape for x in trimmed)
    again=PackedPretrain(source,tmp_path/'cache',isolate_documents=True)
    for x,y in zip(again.batch(order),packed.batch(order)):np.testing.assert_array_equal(x,y)
    path2=tmp_path/'one_doc.jsonl';path2.write_text(json.dumps({'text':'abcdefghijklmnop'}))
    single=PackedPretrain(JsonlDataset(path2,tokenizer,'pretrain',6),tmp_path/'cache',isolate_documents=True)
    assert len(single.batch([0,1]))==2
