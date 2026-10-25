"""Original-model oracles and full replay tests for the remaining native migration."""
import json
import multiprocessing as mp
import os
import socket
from pathlib import Path
import datasets  # Windows DLL ordering.
import numpy as np
import pytest
import torch
from model.dense_cublas.autograd import Engine
from model.dense_cublas.model import Model
from model.dense_cublas.optim import Optimizer
from model.dense_cublas import checkpoint
from tests.test_cublas_training import HostBackend
from tests.test_dense_triton import kernel_test,close
from tests.test_legacy_resume import CONFIG,Data,arguments


@kernel_test
@pytest.mark.parametrize('injection,moe',[(True,False),(False,False),(True,True)])
def test_original_looped_logits_gradient_cache_and_resume(tmp_path,injection,moe):
    from model.model_instinct_loop import InstinctConfig,InstinctForCausalLM
    from model.dense_cublas.legacy_checkpoint import import_resume
    cfg=CONFIG|dict(model_architecture='looped_legacy',prelude_layers=1,loop_iters=2,coda_layers=1,
        use_input_injection=injection,use_moe=moe,num_experts=2,num_experts_per_tok=1,moe_intermediate_size=12)
    ref=InstinctForCausalLM(InstinctConfig(**cfg)).float()
    e=Engine(HostBackend());m=Model(e,cfg,activation_dtype='fp32')
    with torch.no_grad():ref.model.loop_pos_embed.weight.normal_(0,.01)
    for name,p in m.params.items():e.copy(p,e.array(ref.state_dict()[name].detach().numpy()))
    ids=np.asarray([[1,2,3,4],[2,3,4,5]],np.int64);labels=ids.copy();labels[0,2]=-100
    expected=ref(torch.tensor(ids),labels=torch.tensor(labels));loss=expected.loss+expected.aux_loss;loss.backward()
    actual=m(ids,labels);grads=e.vjp(actual.loss+actual.aux_loss)
    close(actual.logits.data,expected.logits,atol=1e-5,rtol=1e-4)
    close(actual.loss.data,expected.loss.reshape(1),atol=1e-5)
    for name,p in ref.named_parameters():
        if p.grad is not None:close(grads[m.params[name]],p.grad,atol=3e-5,rtol=2e-3)
    m.training=False
    with e.no_grad():
        full=m(ids);prefix=m(ids[:,:-1],use_cache=True);last=m(ids[:,-1:],cache=prefix.cache,use_cache=True)
    np.testing.assert_allclose(e.numpy(last.logits),e.numpy(full.logits)[:,-1:],atol=2e-5,rtol=1e-4)
    opt=torch.optim.AdamW(ref.parameters(),lr=.002);opt.step()
    source=tmp_path/'loop.pth';torch.save(dict(model=ref.state_dict(),optimizer=opt.state_dict(),config=ref.config.to_dict(),epoch=0,step=2),source)
    native=Optimizer(e,m.parameters())
    with pytest.warns(RuntimeWarning):import_resume(source,tmp_path/'loop.npz',m,native,Data(tmp_path/'data'),arguments())
    checkpoint.load(tmp_path/'loop.npz',m,native)
    assert all(state['step']==1 for state in native.state.values())


@kernel_test
@pytest.mark.parametrize('topk',[1,2,3])
def test_grouped_moe_dispatch_is_device_only_and_matches_dense_oracle(topk):
    from model.dense_cublas.grouped import dispatch,linear
    e=Engine(HostBackend());rng=np.random.default_rng(3)
    scores=e.array(rng.normal(size=(19,3)).astype(np.float32));selected=e.topk_mask(scores,topk)
    ids,routes,offsets=dispatch(e,selected,topk)
    selected_np=e.numpy(selected);id_np=e.numpy(ids).astype(int);route_np=e.numpy(routes).astype(int)
    expected=np.concatenate([np.flatnonzero(selected_np[:,expert]) for expert in range(3)])
    np.testing.assert_array_equal(id_np,expected)
    assert len(set(route_np))==19*topk
    x=e.array(rng.normal(size=(19*topk,7)).astype(np.float32),requires_grad=True)
    w=e.array(rng.normal(size=(3,5,7)).astype(np.float32),requires_grad=True)
    xp=torch.tensor(e.numpy(x),requires_grad=True);wp=torch.tensor(e.numpy(w),requires_grad=True)
    bounds=e.numpy(offsets).astype(int)
    ref=torch.cat([xp[bounds[i]:bounds[i+1]]@wp[i].T for i in range(3)]);(ref*ref).sum().backward()
    e.numpy=lambda *_:(_ for _ in ()).throw(AssertionError('routing/compute must not read host tensors'))
    actual=linear(e,x,w,offsets,19);grad=e.vjp((actual*actual).sum())
    close(actual.data,ref,atol=2e-5);close(grad[x],xp.grad,atol=5e-5);close(grad[w],wp.grad,atol=5e-5)


@kernel_test
@pytest.mark.parametrize('chunk',[1,3,20])
def test_fused_projection_linear_ce_matches_loss_and_all_gradients(chunk):
    outputs=[]
    for fused in (False,True):
        e=Engine(HostBackend());m=Model(e,CONFIG,activation_dtype='fp32',seed=17)
        m.fuse_projections=fused;m.loss_chunk_size=chunk if fused else 0
        ids=np.asarray([[1,2,3,4],[2,3,4,5]]);labels=ids.copy();labels[0,2]=-100
        out=m(ids,labels,return_logits=False);grads=e.vjp(out.loss)
        outputs.append((e.item(out.loss),{p.name:e.numpy(g) for p,g in grads.items()}))
    assert abs(outputs[0][0]-outputs[1][0])<1e-5
    assert outputs[0][1].keys()==outputs[1][1].keys()
    for key in outputs[0][1]:np.testing.assert_allclose(outputs[0][1][key],outputs[1][1][key],atol=2e-5,rtol=1e-3)


class ReplayBackend(HostBackend):
    def copy_from_numpy(self,tensor,array):tensor.storage.copy_(torch.from_numpy(np.asarray(array).copy()))
    def copy_many_from_numpy(self,tensors,arrays):
        for tensor,array in zip(tensors,arrays):self.copy_from_numpy(tensor,array)
    def to_numpy_many(self,tensors):return tuple(self.to_numpy(tensor) for tensor in tensors)
    def pin_since(self,mark):pass
    def release_since(self,mark,keep=()):
        if self._commands is None and self.execution_plan is None:super().release_since(mark,keep)


@kernel_test
def test_full_training_replay_accumulation_updates_overflow_and_resume(monkeypatch,tmp_path):
    from model.dense_cublas.backend import ExecutionPlan
    from model.dense_cublas.graph import StepGraph
    monkeypatch.setattr(ExecutionPlan,'capture',lambda self:None)  # Replay the exact kernel command tape on CPU.
    e=Engine(ReplayBackend());m=Model(e,CONFIG,activation_dtype='fp32',seed=13)
    m.fuse_projections=True;m.loss_chunk_size=2
    opt=Optimizer(e,m.parameters(),lr=.003,grad_clip=.2,loss_scaling=True)
    e2=Engine(HostBackend());ref=Model(e2,CONFIG,activation_dtype='fp32',seed=13);ref.fuse_projections=True;ref.loss_chunk_size=2
    oracle=Optimizer(e2,ref.parameters(),lr=.003,grad_clip=.2,loss_scaling=True)
    ids=np.asarray([[1,2,3,4]],np.int64)
    fn=lambda values:m(*values,return_logits=False).loss
    graph=StepGraph(m,opt,(ids,ids),fn,2)
    assert opt.updates==0 and all(s['step']==0 for s in opt.state.values())
    for step in range(4):
        batch=(ids+step%2,ids+step%2);lr=.003/(step+1)
        expected=ref(*batch,return_logits=False).loss;oracle.backward(expected)
        if step%2:oracle.step(lr)
        actual=graph.run(batch,lr)
        assert abs(actual-e2.item(expected))<1e-4
        assert opt.accumulated==oracle.accumulated and opt.updates==oracle.updates
        for p in m.parameters():close(p.data,ref.params[p.name].data.storage,atol=2e-5,rtol=2e-3)
    checkpoint.save(tmp_path/'graph.npz',m,opt,{'steps':4})
    checkpoint.load(tmp_path/'graph.npz',m,opt)
    # An overflow in an accumulated gradient must skip every parameter update.
    p=opt.parameters[0];e.copy(opt.gradients[p],e.full(p.shape,float('inf')))
    before=e.numpy(p);scale=opt.scale
    graph.run((ids,ids),.001,force=True)
    assert opt.updates==2 and opt.accumulated==0 and opt.scale==scale*.5
    np.testing.assert_array_equal(e.numpy(p),before)


def _distributed_worker(rank,port,output):
    from model.dense_cublas.distributed import ProcessGroup
    with ProcessGroup(rank=rank,world=2,port=port,transport='tcp',token='test',timeout=20) as group:
        result=group.sum(np.asarray([rank+1,rank*3],np.float32))
        assert group.any(rank==1)
        e=Engine(HostBackend());params=[e.array(np.zeros((2,),np.float32),requires_grad=True,name=str(i)) for i in range(2)]
        optimizer=Optimizer(e,params,lr=.1,weight_decay=0,grad_clip=0,loss_scaling=True);optimizer.distributed=group
        optimizer.backward(params[rank].sum());optimizer.step()
        for parameter in params:np.testing.assert_allclose(e.numpy(parameter),-.1,atol=1e-6)
        before=[e.numpy(p) for p in params]
        optimizer.backward(params[rank].sum()*(float('inf') if rank else 1.))
        assert not optimizer.step()['updated']
        for parameter,expected in zip(params,before):np.testing.assert_array_equal(e.numpy(parameter),expected)
        output.put((rank,result.tolist()))


def test_two_process_collectives_and_sampler():
    from model.dense_cublas.distributed import shard_order
    context=mp.get_context('spawn');queue=context.Queue()
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    children=[context.Process(target=_distributed_worker,args=(rank,port,queue)) for rank in range(2)]
    try:
        for child in children:child.start()
        results=[queue.get(timeout=30) for _ in children]
        for child in children:child.join(30);assert child.exitcode==0
        assert sorted(results)==[(0,[3.,3.]),(1,[3.,3.])]
        assert shard_order([0,1,2,3,4],2,0)==[0,2,4]
        assert shard_order([0,1,2,3,4],2,1)==[1,3,0]
    finally:
        for child in children:
            if child.is_alive():child.terminate();child.join()


@kernel_test
def test_async_checkpoint_snapshot_is_immutable_and_errors_surface(tmp_path,monkeypatch):
    import threading
    e=Engine(HostBackend());m=Model(e,CONFIG,activation_dtype='fp32');before=e.numpy(m.parameters()[0])
    gate=threading.Event();write=checkpoint.write
    def delayed(path,arrays):gate.wait(10);write(path,arrays)
    monkeypatch.setattr(checkpoint,'write',delayed)
    with checkpoint.AsyncCheckpointer() as saver:
        saver.save(tmp_path/'snapshot.npz',m)
        e.copy(m.parameters()[0],e.full(m.parameters()[0].shape,7.));gate.set()
    checkpoint.load(tmp_path/'snapshot.npz',m);np.testing.assert_array_equal(e.numpy(m.parameters()[0]),before)
    def fail(*args):raise OSError('disk failure')
    monkeypatch.setattr(checkpoint,'write',fail)
    with pytest.raises(OSError,match='disk failure'):
        with checkpoint.AsyncCheckpointer() as saver:saver.save(tmp_path/'broken.npz',m)


def test_token_cache_packing_and_buckets(tmp_path):
    from model.dense_cublas.data import Tokenizer,JsonlDataset,PackedPretrain,trim_batch
    tokenizer=Tokenizer('model',str(tmp_path/'tokens.sqlite'))
    assert tokenizer.encode('hello 世界')==Tokenizer('model').encode('hello 世界')
    data=tmp_path/'data.jsonl';data.write_text('\n'.join(json.dumps({'text':s}) for s in ('hello world','second document','last')))
    source=JsonlDataset(data,tokenizer,'pretrain',5);packed=PackedPretrain(source,tmp_path/'packed')
    expected=[]
    for i in range(len(source)):expected.extend([tokenizer.bos]+tokenizer.encode(source.row(i)['text'])+[tokenizer.eos])
    reconstructed=[]
    for i in range(len(packed)):
        ids,labels=packed.batch([i]);valid=ids[0][labels[0]!=-100].tolist()
        reconstructed.extend(valid if i==0 else valid[1:])
    assert reconstructed==expected
    ids=np.asarray([[1,2,0,0,0,0],[2,3,4,0,0,0]]);labels=np.where(ids,ids,-100)
    assert trim_batch((ids,labels),0,[2,4,6])[0].shape==(2,4)


@kernel_test
def test_elastic_resume_preserves_partial_accumulation(tmp_path):
    from model.dense_cublas.distributed import restore_pending
    e=Engine(HostBackend());m=Model(e,CONFIG,activation_dtype='fp32');opt=Optimizer(e,m.parameters())
    p=m.parameters()[0];path=tmp_path/'resume.npz'
    for rank in range(2):
        opt.zero_grad();opt.accumulate({p:e.full(p.shape,float(rank+1)).data})
        checkpoint.save(path if rank==0 else tmp_path/'resume.rank1.npz',m,opt,dict(steps=7))
    checkpoint.load(path,m,opt)
    restore_pending(path,{'model':opt},2,7)
    np.testing.assert_allclose(e.numpy(opt.gradients[p]),1.5)
    assert opt.accumulated==1 and p in opt.present
    with pytest.raises(ValueError,match='same step'):restore_pending(path,{'model':opt},2,8)


@kernel_test
def test_pending_weight_gradients_join_after_cross_layer_dx():
    from tests.test_dense_cublas import ScheduleOracle
    b=ScheduleOracle(overlap=True)
    # Memory bookkeeping is irrelevant here; the real operation/event command order is inspected.
    b.mark=lambda:0;b.release_since=lambda *a,**k:None
    b.from_numpy=lambda x:b.wrap(torch.from_numpy(np.asarray(x).copy()))
    e=Engine(b)
    x=e.array(np.ones((2,4),np.float32),requires_grad=True)
    a=e.array(np.eye(4,dtype=np.float32),requires_grad=True)
    w=e.array(np.eye(4,dtype=np.float32),requires_grad=True)
    plan=b.build(lambda:e.vjp(e.linear(e.linear(x,a),w).sum()))
    trace=plan.describe();dx=[r['index'] for r in trace if r['operation']=='cublasLt:dX']
    # Waiting on each parameter dW immediately would put a main wait between the two dX GEMMs.
    waits=[r['index'] for r in trace if r['operation']=='event:wait' and r['lane']=='main']
    assert len(dx)==2 and min(waits)>max(dx)


def test_copy_prefetch_double_buffer_dependency_and_lifetime():
    import ctypes as ct
    from contextlib import contextmanager
    from model.dense_cublas.prefetch import DevicePrefetch
    from model.dense_triton.runtime import Tensor
    class Backend:
        def __init__(self):self.storage=[];self.pending=[];self.events={};self.freed=[];self.stream=1
        def _bind(self,name,types):
            if name=='cuMemHostAlloc':
                def allocate(output,size,flags):
                    block=ct.create_string_buffer(size);self.storage.append(block)
                    ct.cast(output,ct.POINTER(ct.c_void_p))[0]=ct.addressof(block)
                return allocate
            if name=='cuMemFreeHost':return lambda pointer:self.freed.append(pointer.value)
            if name=='cuMemcpyHtoDAsync_v2':return lambda target,source,size,stream:self.pending.append((target,source,size))
            raise AssertionError(name)
        def empty(self,shape,dtype):
            block=ct.create_string_buffer(int(np.prod(shape))*8);self.storage.append(block)
            return Tensor(shape,dtype,ct.addressof(block),self)
        def _make_stream(self,priority):return 2
        def create_event(self):key=len(self.events)+1;self.events[key]=[];return key
        def record_event(self,event):self.events[event]=self.pending;self.pending=[]
        def wait_event(self,event):
            for target,source,size in self.events[event]:ct.memmove(target,source,size)
            self.events[event]=[]
        def _event_sync(self,event):self.wait_event(event)
        def synchronize(self):
            for event in self.events:self.wait_event(event)
        @contextmanager
        def on_stream(self,stream):yield
    backend=Backend();e=Engine(backend)
    source=[(i,(np.full((1,3+i%2),i,np.int64),np.full((1,3+i%2),i+1,np.int64))) for i in range(5)]
    with DevicePrefetch(e,1,4,True) as prefetch:
        for begin,cpu,device in prefetch.batches(source):
            for expected,value in zip(cpu,device):
                actual=np.ctypeslib.as_array((ct.c_int64*expected.size).from_address(value.data.storage)).reshape(expected.shape)
                np.testing.assert_array_equal(actual,expected)
    assert len(backend.freed)==4
