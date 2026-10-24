"""Execute native training primitives in Triton's CPU interpreter; Torch is oracle-only."""
import os
import datasets
import numpy as np
import pytest
import torch

from tests.test_dense_cublas import ScheduleOracle
from tests.test_dense_triton import kernel_test, close, DEVICE, random
from model.dense_cublas.autograd import Engine

pytestmark = pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="CPU interpreter/oracle suite; native GPU acceptance is experiments.validate_cublas_training")


class HostBackend(ScheduleOracle):
    def __init__(self):
        super().__init__()
        self.serial, self.live, self.peak = 0, {}, 0

    def _track(self, tensor):
        self.serial += 1
        self.live[id(tensor.storage)] = (self.serial, tensor)
        self.peak = max(self.peak, sum(t.numel for _, t in self.live.values()))
        return tensor

    def empty(self, *args, **kwargs):
        return self._track(super().empty(*args, **kwargs))

    def from_numpy(self, array):
        return self._track(self.wrap(torch.from_numpy(np.array(array, copy=True))))

    def to_numpy(self, tensor):
        return tensor.storage.view(tensor.shape).detach().float().cpu().numpy().copy()

    def mark(self): return self.serial

    def release_since(self, mark, keep=()):
        retained = {id(t.storage) for t in keep if t is not None}
        for key, (serial, tensor) in list(self.live.items()):
            if serial > mark and key not in retained: del self.live[key]
        self._allocations[:] = [t for t in self._allocations if id(t.storage) in self.live]


@kernel_test
@pytest.mark.parametrize('act', ['silu', 'gelu', 'relu', 'softplus', 'sigmoid'])
def test_engine_broadcast_reduction_activation_and_shared_gradient(act):
    e = Engine(HostBackend())
    x = random((2, 3, 7)).requires_grad_()
    w = random((7,), seed=43).requires_grad_()
    f = getattr(torch.nn.functional, act, None) or getattr(torch, act)
    expected = (f(x * w + w) / 3).square().mean()
    expected.backward()
    a, b = e.array(x.detach().cpu().numpy(), requires_grad=True), e.array(w.detach().cpu().numpy(), requires_grad=True)
    y = e.unary(a * b + b, act) / 3
    loss = (y * y).mean()
    grads = e.vjp(loss)
    close(loss.data, expected.reshape(1))
    close(grads[a], x.grad)
    close(grads[b], w.grad)


@kernel_test
def test_engine_indexing_softmax_ce_and_embedding_tie():
    e = Engine(HostBackend())
    w = random((17, 8), scale=0.2).requires_grad_()
    ids = torch.tensor([[1, 2, 1], [3, 4, 3]], device=DEVICE)
    labels = torch.tensor([[2, -100, 5], [6, 1, 2]], device=DEVICE)
    x = torch.nn.functional.embedding(ids, w)
    logits = x @ w.T
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 17), labels.reshape(-1))
    expected.backward()
    p = e.array(w.detach().cpu().numpy(), requires_grad=True)
    x = e.embedding(p, e.array(ids.cpu().numpy()))
    logit = e.linear(x, p)
    loss = e.ce(logit, e.array(labels.cpu().numpy()))
    grads = e.vjp(loss)
    close(loss.data, expected.reshape(1))
    close(grads[p], w.grad, atol=1e-5)
    a = e.array(np.arange(24, dtype=np.float32).reshape(2, 3, 4), requires_grad=True)
    out = e.cat([a[:, :1], a[:, 2:]], 1).permute(1, 0, 2)
    g = e.vjp(out.sum())[a]
    target = torch.ones((2, 3, 4)); target[:, 1] = 0
    close(g, target)


@kernel_test
def test_checkpoint_repeats_rng_and_releases_forward_buffers():
    def execute(checkpoint):
        backend, seed = HostBackend(), 123
        e = Engine(backend, seed=seed)
        x = e.array(np.arange(12, dtype=np.float32).reshape(3, 4) / 10, requires_grad=True)
        w = e.array(np.eye(4, dtype=np.float32), requires_grad=True)
        fn = lambda z: e.dropout(e.linear(z, w), .3) * 2
        out = e.checkpoint(fn, x, [w]) if checkpoint else fn(x)
        live_before_backward = sum(t.numel for _, t in backend.live.values())
        counter = e.counter
        grads = e.vjp((out * out).sum())
        assert counter == e.counter
        return [e.numpy(t) for t in (out, grads[x], grads[w])], live_before_backward
    baseline, saved = execute(False)
    recompute, recomputed_saved = execute(True)
    for a, b in zip(baseline, recompute): np.testing.assert_allclose(a, b)
    assert recomputed_saved < saved

@kernel_test
def test_delta_recurrent_kernel_and_vjp():
    from model.dense_cublas.recurrent import recurrence
    e=Engine(HostBackend())
    shapes=[(1,3,2,3)]*2+[(1,3,2,4),(1,3,2),(1,3,2),(1,2,3,4)]
    tensors=[random(s,seed=30+i,scale=.15).requires_grad_() for i,s in enumerate(shapes)]
    q,k,v,g,beta,state=tensors
    ys=[];s=state
    for t in range(3):
        r=s*g[:,t].exp()[...,None,None]
        delta=(v[:,t]-(r*k[:,t,...,None]).sum(-2))*beta[:,t,...,None]
        s=r+k[:,t,...,None]*delta[...,None,:]
        ys.append((s*q[:,t,...,None]).sum(-2)/3**.5)
    expected=torch.cat([torch.stack(ys,1).reshape(-1),s.reshape(-1)])
    expected.square().sum().backward()
    values=[e.array(x.detach().numpy(),requires_grad=True) for x in tensors]
    actual=recurrence(e,*values);grads=e.vjp((actual*actual).sum())
    close(actual.data,expected,atol=2e-5)
    for x,z in zip(tensors,values):close(grads[z],x.grad,atol=2e-5)


@kernel_test
@pytest.mark.parametrize('checkpoint',[0,1,2])
@pytest.mark.parametrize('moe',[False,True])
def test_full_model_matches_torch(checkpoint,moe):
    from model.dense_cublas.model import Model
    from model.model_instinct import InstinctConfig,InstinctForCausalLM
    cfg=dict(hidden_size=16,num_hidden_layers=2,vocab_size=23,num_attention_heads=2,num_key_value_heads=1,
        head_dim=8,intermediate_size=32,max_position_embeddings=16,dropout=0.,flash_attn=False,
        use_moe=moe,num_experts=2,num_experts_per_tok=2,moe_intermediate_size=24,use_grad_checkpoint=checkpoint,
        early_exit_layers=[1],early_exit_loss_weight=.3)
    e=Engine(HostBackend());m=Model(e,cfg,activation_dtype='fp32')
    ref=InstinctForCausalLM(InstinctConfig(**cfg)).float()
    state={k:torch.from_numpy(e.numpy(v)) for k,v in m.params.items()}
    ref.load_state_dict(state)
    ids=np.array([[1,3,5,4]],np.int64);labels=ids.copy();labels[0,2]=-100
    actual=m(ids,labels);expected=ref(torch.tensor(ids),labels=torch.tensor(labels),early_exit=True)
    grads=e.vjp(actual.loss+actual.aux_loss)
    (expected.loss+expected.aux_loss).backward()
    close(actual.logits.data,expected.logits,atol=2e-5)
    close(actual.loss.data,expected.loss.reshape(1),atol=2e-5)
    for name,p in ref.named_parameters():
        if p.grad is not None:close(grads[m.params[name]],p.grad,atol=2e-4,rtol=2e-3)

@kernel_test
@pytest.mark.parametrize('kind',['adamw','adafactor','muon'])
def test_optimizer_updates_and_accumulation(kind):
    from model.dense_cublas.optim import Optimizer
    from trainer.trainer_utils import MuonOptimizer as MuonFallback
    e=Engine(HostBackend());initial=random((5,3),scale=.3)
    p=e.array(initial.numpy(),requires_grad=True,name='weight');ref=torch.nn.Parameter(initial.clone())
    opt=Optimizer(e,[p],kind,lr=.003,grad_clip=0.)
    cls={'adamw':torch.optim.AdamW,'adafactor':torch.optim.Adafactor,'muon':MuonFallback}[kind]
    reference=cls([ref],lr=.003,weight_decay=.01)
    for step in range(3):
        g=random((5,3),seed=step,scale=.2)
        ref.grad=g;reference.step();reference.zero_grad()
        opt.accumulate({p:e.array(g.numpy()).data});opt.accumulate({p:e.array(g.numpy()).data});opt.step()
        close(p.data,ref,atol=3e-6,rtol=1e-4)


@kernel_test
@pytest.mark.parametrize('arch',['standard','linear','looped'])
def test_cache_decode_and_training_gradient(arch):
    from model.dense_cublas.model import Model
    cfg=dict(hidden_size=8,num_hidden_layers=2,vocab_size=13,num_attention_heads=2,num_key_value_heads=1,
        head_dim=4,intermediate_size=12,max_position_embeddings=16,flash_attn=False,model_architecture=arch,
        linear_key_head_dim=3,linear_value_head_dim=4,linear_num_key_heads=1,linear_num_value_heads=2,
        linear_conv_kernel_dim=3,full_attention_interval=2,loop_max_steps=2,q_threshold=1.)
    e=Engine(HostBackend());m=Model(e,cfg,activation_dtype='fp32');ids=np.array([[1,2,3]],np.int64)
    result=m(ids,ids);grads=e.vjp(result.loss+result.aux_loss)
    assert all(np.isfinite(e.numpy(g)).all() for g in grads.values())
    m.training=False
    with e.no_grad():
        full=m(ids);prefill=m(ids[:,:2],use_cache=True);last=m(ids[:,2:],cache=prefill.cache,use_cache=True)
    close(last.logits.data,full.logits.data.storage.view(full.logits.shape)[:,2:],atol=3e-5)


@kernel_test
def test_checkpoint_resumes_optimizer_rng_and_partial_accumulation(tmp_path):
    from model.dense_cublas.model import Model
    from model.dense_cublas.optim import Optimizer
    from model.dense_cublas import checkpoint
    c=dict(hidden_size=8,num_hidden_layers=1,vocab_size=11,num_attention_heads=2,num_key_value_heads=1,
        head_dim=4,intermediate_size=12,max_position_embeddings=8,flash_attn=False,dropout=.2)
    e=Engine(HostBackend(),seed=47);m=Model(e,c,activation_dtype='fp32');opt=Optimizer(e,m.parameters())
    ids=np.array([[1,3,2]])
    opt.backward(m(ids,ids).loss)
    path=tmp_path/'resume.npz';rng=np.random.default_rng(21);checkpoint.save(path,m,opt,{'epoch':1,'cursor':7},rng)
    counter=e.counter;opt.backward(m(ids,ids).loss);opt.step();expected=[e.numpy(p) for p in m.parameters()]
    progress=checkpoint.load(path,m,opt,rng)
    assert progress=={'epoch':1,'cursor':7} and e.counter==counter and opt.accumulated==1
    opt.backward(m(ids,ids).loss);opt.step()
    for p,w in zip(m.parameters(),expected):np.testing.assert_allclose(e.numpy(p),w,atol=1e-6)


class ToyTokenizer:
    bos,eos,pad=1,2,0
    bos_token,eos_token,pad_token='<B>','<E>','<P>'
    def encode(self,text):return [3+ord(c)%10 for c in text]
    def decode(self,ids,**kw):return ''.join(chr(97+int(i)) for i in ids)
    def chat(self,messages,tools=None,generation=False,thinking=False):return ' '.join(m.get('content','') for m in messages)+('a' if generation else '')
    def labels(self,ids):return np.where(np.asarray(ids)!=0,ids,-100)


@kernel_test
@pytest.mark.parametrize('mode',['pretrain','full_sft','lora','dpo','distillation','grpo','agent','ppo'])
def test_end_to_end_native_trainer_modes(tmp_path,mode):
    import json
    from trainer.train_cublas import parser,train
    from model.dense_cublas.model import Model
    from model.dense_cublas import checkpoint
    backend=HostBackend();config=dict(hidden_size=8,num_hidden_layers=1,vocab_size=13,num_attention_heads=2,num_key_value_heads=1,
        head_dim=4,intermediate_size=12,max_position_embeddings=16,flash_attn=False)
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    e=Engine(backend);base=Model(e,config,activation_dtype='fp32');base_path=tmp_path/'base.npz';checkpoint.save(base_path,base)
    messages=[{'role':'user','content':'a'},{'role':'assistant','content':'bc'}]
    row={'text':'abc','conversations':messages,'chosen':messages,'rejected':[messages[0],{'role':'assistant','content':'de'}],'gt':[]}
    data=tmp_path/'data.jsonl';data.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    argv=['--train_type',mode,'--config_path',str(config_path),'--data_path',str(data),'--from_weight',str(base_path),
        '--dtype','fp32','--fp8','0','--batch_size','1','--max_seq_len','5','--accumulation_steps','2',
        '--max_gen_len','2','--max_total_len','8','--num_generations','2','--ppo_update_iters','1','--mini_batch_size','1',
        '--save_dir',str(tmp_path/'out'),'--checkpoint_dir',str(tmp_path/'checkpoints'),'--max_steps','1']
    if mode=='distillation':argv+=['--from_teacher_weight',str(base_path)]
    class Reward:
        def get_score(self,messages,text):return len(text)*.1
    result=train(parser().parse_args(argv),backend,ToyTokenizer(),Reward())
    assert np.isfinite(result['loss']) and result['steps']==1
    meta=checkpoint.metadata(result['resume'])
    assert meta['progress']['cursor']==1
    # Resume the remaining partial batch/accumulation, using a new optimizer.
    args=parser().parse_args(argv+['--from_resume','1','--max_steps','2'])
    result=train(args,backend,ToyTokenizer(),Reward())
    assert result['steps']==2 and np.isfinite(result['loss'])


@kernel_test
@pytest.mark.parametrize('mode',['dpo','kd','grpo','cispo','ppo'])
def test_objectives_value_and_gradient_against_torch(mode):
    from model.dense_cublas import objectives as o
    e=Engine(HostBackend());x=random((4,3),scale=.2).requires_grad_();a=e.array(x.detach().numpy(),requires_grad=True)
    old=random((4,3),seed=2,scale=.1);ref=random((4,3),seed=3,scale=.1);mask=torch.tensor([[1,1,0],[1,1,1],[1,0,0],[1,1,0.]],device=DEVICE)
    adv=random((4,),seed=4,scale=.4);oldv=random((4,3),seed=5,scale=.1);returns=random((4,3),seed=6,scale=.3)
    arr=lambda z:e.array(z.numpy())
    if mode=='dpo':
        expected=torch.nn.functional.softplus(-.2*((x.sum(-1)[:2]-x.sum(-1)[2:])-(ref.sum(-1)[:2]-ref.sum(-1)[2:]))).mean()
        actual=o.dpo(e,a.sum(-1),arr(ref.sum(-1)),.2)
    elif mode=='kd':
        target=ref.softmax(-1);expected=(target*(ref.log_softmax(-1)-x.log_softmax(-1))).sum(-1).mean()
        actual=o.distillation(e,a,arr(ref),arr(torch.ones(4,dtype=torch.long)))
    elif mode in ('grpo','cispo'):
        ratio=(x-old).exp();delta=ref-x;kl=delta.exp()-delta-1
        surrogate=ratio.clamp(0,5).detach()*adv[:,None]*x if mode=='cispo' else torch.minimum(ratio*adv[:,None],ratio.clamp(.8,1.2)*adv[:,None])
        expected=((-(surrogate-.1*kl)*mask).sum(-1)/mask.sum(-1)).mean()
        actual=o.grpo(e,a,arr(old),arr(ref),arr(adv),arr(mask),kind=mode)
    else:
        ratio=(x-old).exp();delta=ref-x;kl=delta.exp()-delta-1
        policy=(torch.maximum(-adv[:,None]*ratio,-adv[:,None]*ratio.clamp(.8,1.2))+.02*kl)*mask
        clipped=oldv+(x-oldv).clamp(-.2,.2)
        expected=(policy.sum()+.25*(torch.maximum((x-returns).square(),(clipped-returns).square())*mask).sum())/mask.sum()
        actual=o.ppo(e,a,arr(old),arr(ref),a,arr(oldv),arr(adv[:,None]*torch.ones(4,3)),arr(returns),arr(mask))[0]
    expected.backward();grads=e.vjp(actual)
    close(actual.data,expected.reshape(1),atol=2e-5);close(grads[a],x.grad,atol=2e-5)


@kernel_test
def test_linear_model_against_original():
    from model.dense_cublas.model import Model
    from model.model_instinct_linear import InstinctConfig,InstinctForCausalLM
    cfg=dict(hidden_size=8,num_hidden_layers=2,vocab_size=13,num_attention_heads=2,num_key_value_heads=1,
        head_dim=4,intermediate_size=12,max_position_embeddings=16,flash_attn=False,model_architecture='linear',
        linear_key_head_dim=4,linear_value_head_dim=4,linear_num_key_heads=1,linear_num_value_heads=2,linear_conv_kernel_dim=3,full_attention_interval=2)
    e=Engine(HostBackend());m=Model(e,cfg,activation_dtype='fp32');ref=InstinctForCausalLM(InstinctConfig(**cfg)).float()
    ref.load_state_dict({k:torch.from_numpy(e.numpy(v)) for k,v in m.params.items()})
    ids=np.array([[1,2,4,5]]);out=m(ids,ids);expected=ref(torch.tensor(ids),labels=torch.tensor(ids))
    (expected.loss+expected.aux_loss).backward();grads=e.vjp(out.loss+out.aux_loss)
    close(out.logits.data,expected.logits,atol=5e-5)
    for name,p in ref.named_parameters():
        if p.grad is not None:close(grads[m.params[name]],p.grad,atol=3e-4,rtol=3e-3)


def test_native_import_guard_and_tokenizer_parity():
    import subprocess,sys
    subprocess.run([sys.executable,'-c',"from experiments.smoke_dense_triton import RejectTorch; import sys;sys.meta_path.insert(0,RejectTorch());import trainer.train_cublas; import model.dense_cublas.reward; assert 'torch' not in sys.modules"],check=True)
    from model.dense_cublas.data import Tokenizer
    from transformers import AutoTokenizer
    native=Tokenizer('model');reference=AutoTokenizer.from_pretrained('model')
    messages=[{'role':'user','content':'你好'}, {'role':'assistant','content':'Hello!'}]
    expected=reference.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    assert native.chat(messages)==expected
    assert native.encode(expected)==reference.encode(expected,add_special_tokens=False)


def test_gae_places_terminal_reward_after_prompt():
    from model.dense_cublas.objectives import gae
    mask=np.array([[0,0,1,1,0],[0,1,1,1,1]],np.float32);values=np.zeros_like(mask)
    _,returns=gae([2,4],values,mask,gamma=1.,lam=1.)
    np.testing.assert_allclose(returns,[[0,0,2,2,0],[0,4,4,4,4]])


def test_dynamic_allocator_reuse_bound_and_event_recycling():
    import ctypes as ct
    from model.dense_cublas.memory import TrainingBackend
    class Allocator(TrainingBackend):
        def __init__(self):
            super().__init__(pool_limit_mib=1);self.arch=89;self.next_pointer=0;self.freed=[];self.event_count=0
            self._free=lambda p:self.freed.append(p)
            def malloc(out,size):
                self.next_pointer+=1;ct.cast(out,ct.POINTER(ct.c_uint64))[0]=self.next_pointer
            def event(out,flags):
                self.event_count+=1;ct.cast(out,ct.POINTER(ct.c_void_p))[0]=self.event_count+10000
            self._malloc=malloc;self._event_create=event
        def _check(self):pass
        def synchronize(self):pass
    backend=Allocator();root=backend.empty((7,),'fp32');mark=backend.mark()
    a=backend.empty((1024,),'fp32');ev=backend.create_event();backend.release_since(mark)
    b=backend.empty((1024,),'fp32');ev2=backend.create_event()
    assert a.storage==b.storage and ev==ev2 and backend.event_count==1
    large=backend.empty((400000,),'fp32');backend.release_since(mark)
    assert large.storage in backend.freed and root.storage in backend._live and backend._pool_bytes<=backend.pool_limit
    assert all(p.value!=large.storage for p,n in backend._allocations)


@kernel_test
def test_overflow_clears_nan_gradients_and_recovers():
    from model.dense_cublas.optim import Optimizer
    e=Engine(HostBackend());p=e.array([1.,2.],requires_grad=True,name='p');opt=Optimizer(e,[p],loss_scaling=True)
    opt.accumulate({p:e.array([np.nan,1.]).data});result=opt.step()
    assert not result['updated'] and opt.scale==32768
    np.testing.assert_array_equal(e.numpy(opt.gradients[p]),[0.,0.])
    opt.accumulate({p:e.array([32768.,32768.]).data});assert opt.step()['updated']
    assert np.isfinite(e.numpy(p)).all()


@kernel_test
def test_native_reward_grouped_qkv_and_last_token_score():
    from model.dense_cublas.reward import InternLMReward
    from model.dense_cublas.model import rope_tables
    e=Engine(HostBackend());rm=InternLMReward.__new__(InternLMReward);rm.e=e
    rm.config=dict(hidden_size=8,num_attention_heads=2,num_key_value_heads=1,num_hidden_layers=1,max_position_embeddings=16,
        head_dim=4,rope_theta=10000.,rms_norm_eps=1e-5)
    shapes={'model.tok_embeddings.weight':(13,8),'model.norm.weight':(8,),'v_head.weight':(1,8),
        'model.layers.0.attention_norm.weight':(8,),'model.layers.0.ffn_norm.weight':(8,),
        'model.layers.0.attention.wqkv.weight':(16,8),'model.layers.0.attention.wo.weight':(8,8),
        'model.layers.0.feed_forward.w1.weight':(12,8),'model.layers.0.feed_forward.w3.weight':(12,8),'model.layers.0.feed_forward.w2.weight':(8,12)}
    w={name:(torch.ones(shape) if len(shape)==1 else random(shape,seed=idx,scale=.1)) for idx,(name,shape) in enumerate(shapes.items())}
    rm.weights={name:e.array(t.numpy()) for name,t in w.items()};tables=rope_tables(rm.config);rm.cos,rm.sin=(e.array(a) for a in tables)
    ids=[1,3,5];x=w['model.tok_embeddings.weight'][torch.tensor([ids])]
    norm=lambda x,p:x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*w[p+'.weight']
    lin=lambda x,p:x@w[p+'.weight'].T
    z=norm(x,'model.layers.0.attention_norm');qkv=lin(z,'model.layers.0.attention.wqkv').reshape(1,3,1,4,4)
    q=qkv[:,:,:,:2,:].reshape(1,3,2,4);k=qkv[:,:,:,2,:];v=qkv[:,:,:,3,:]
    cos,sin=(torch.from_numpy(a[:3])[None,:,None,:] for a in tables)
    rope=lambda x:x*cos+torch.cat([-x[...,2:],x[...,:2]],-1)*sin
    q,k=rope(q),rope(k)
    attn=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2).repeat_interleave(2,1),v.transpose(1,2).repeat_interleave(2,1),is_causal=True)
    x=x+lin(attn.transpose(1,2).reshape(1,3,8),'model.layers.0.attention.wo');z=norm(x,'model.layers.0.ffn_norm')
    x=x+lin(torch.nn.functional.silu(lin(z,'model.layers.0.feed_forward.w1'))*lin(z,'model.layers.0.feed_forward.w3'),'model.layers.0.feed_forward.w2')
    expected=lin(norm(x,'model.norm')[:,-1:],'v_head').item()
    assert rm.score_ids(ids)==pytest.approx(expected,abs=2e-5)


def test_agent_tool_observation_is_not_a_policy_target(monkeypatch):
    from model.dense_cublas import rollout
    tk=ToyTokenizer();rows={'messages':[{'role':'user','content':'a'}],'tools':[{'function':{'name':'calculate_math'}}],'gt':['2']}
    responses=['<tool_call>{"name":"calculate_math","arguments":{"expression":"1+1"}}</tool_call>','2']
    monkeypatch.setattr(tk,'decode',lambda ids,**kw:responses.pop(0))
    monkeypatch.setattr(rollout,'generate',lambda *args,**kwargs:([3,2],[-.5,-.8]))
    from types import SimpleNamespace
    model=SimpleNamespace(config={'max_position_embeddings':2048})
    out=rollout.trajectory(model,tk,rows,np.random.default_rng(2),max_total_len=1000,agent=True)
    assert len(out['outputs'])==2 and sum(out['mask'])==4
    assert len(out['tokens'])>sum(out['mask']) and not out['unfinished']
    assert rollout.reward(out,None,agent=True)==pytest.approx(3.)
    from model.dense_cublas.agent_tools import execute_tool
    assert execute_tool('calculate_math',{'expression':'1+1'})=={'result':'2'}
    assert execute_tool('calculate_math',{'expression':'().__class__'}) is None


@kernel_test
def test_lora_merge_and_cache_reorder():
    from model.dense_cublas.model import Model
    e=Engine(HostBackend());cfg=dict(hidden_size=8,num_hidden_layers=1,vocab_size=13,num_attention_heads=2,num_key_value_heads=1,
        head_dim=4,intermediate_size=12,max_position_embeddings=8,flash_attn=False)
    model=Model(e,cfg,activation_dtype='fp32');model.add_lora(2)
    for a,b in model.adapters.values():e.copy(b,e.full(b.shape,.03))
    ids=np.array([[1,3,4],[2,5,6]]);model.training=False
    with e.no_grad():
        before=e.numpy(model(ids).logits);cache=model(ids[:,:2],use_cache=True).cache
        reordered=model.reorder_cache(cache,[1,1,0]);actual=model(ids[[1,1,0],2:],cache=reordered,use_cache=True)
        np.testing.assert_allclose(e.numpy(actual.logits),before[[1,1,0],2:],atol=3e-5)
        model.merge_lora();after=e.numpy(model(ids).logits)
    np.testing.assert_allclose(after,before,atol=3e-5)
    assert not model.adapters and not any('.lora.' in k for k in model.params)


@kernel_test
def test_separate_cpu_oracle_cli(tmp_path):
    import json,subprocess,sys
    from model.dense_cublas.model import Model
    e=Engine(HostBackend());cfg=dict(hidden_size=8,num_hidden_layers=1,vocab_size=13,num_attention_heads=2,num_key_value_heads=1,
        head_dim=4,intermediate_size=12,max_position_embeddings=8,flash_attn=False)
    model=Model(e,cfg,activation_dtype='fp32');ids=np.array([[1,2,5]])
    source,target=tmp_path/'input.npz',tmp_path/'result.npz'
    arrays={'config':np.asarray(json.dumps(model.config)),'ids':ids}
    arrays.update({'parameter/'+name:e.numpy(p) for name,p in model.params.items()});np.savez(source,**arrays)
    subprocess.run([sys.executable,'-m','experiments.reference_cublas_training','--input',str(source),'--output',str(target)],check=True)
    actual=model(ids,ids);grads=e.vjp(actual.loss+actual.aux_loss)
    with np.load(target) as oracle:
        np.testing.assert_allclose(e.numpy(actual.logits),oracle['logits'],atol=3e-5)
        for name,p in model.params.items():
            if 'gradient/'+name in oracle:np.testing.assert_allclose(e.numpy(grads[p]),oracle['gradient/'+name],atol=2e-4)
