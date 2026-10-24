"""Import actual source optimizer layouts, then compare the next parameter update."""
import json
from pathlib import Path
from types import SimpleNamespace
import datasets
import numpy as np
import pytest
import torch
from model.dense_cublas.autograd import Engine
from model.dense_cublas.model import Model
from model.dense_cublas.optim import Optimizer
from model.dense_cublas import checkpoint
from model.dense_cublas.legacy_checkpoint import import_resume
from tests.test_cublas_training import HostBackend
from tests.test_dense_triton import kernel_test,close


CONFIG=dict(hidden_size=8,num_hidden_layers=1,vocab_size=13,num_attention_heads=2,num_key_value_heads=1,
    head_dim=4,intermediate_size=12,max_position_embeddings=16,flash_attn=False)


class Data:
    def __init__(self,path):self.path=path
    def __len__(self):return 12


def arguments(**overrides):
    values=dict(seed=42,train_type='pretrain',batch_size=2,accumulation_steps=2,epochs=3,
        legacy_batch_size=None,legacy_accumulation_steps=None,legacy_epochs=None,learning_rate=None)
    return SimpleNamespace(**(values|overrides))


@kernel_test
@pytest.mark.parametrize('kind,lora,tie',[('adamw',False,True),('adamw',False,False),('adafactor',False,True),('muon',False,True),('adamw',True,True)])
def test_legacy_full_state_next_update(tmp_path,kind,lora,tie):
    from model.model_instinct import InstinctConfig,InstinctForCausalLM
    from model.model_lora import apply_lora
    from trainer.trainer_utils import build_optimizer,lm_checkpoint
    cfg=CONFIG|{'tie_word_embeddings':tie}
    ref=InstinctForCausalLM(InstinctConfig(**cfg)).float()
    if lora:
        apply_lora(ref,rank=2)
        for name,p in ref.named_parameters():p.requires_grad_('.lora.' in name)
    params=[p for p in ref.parameters() if p.requires_grad]
    original=build_optimizer(params,lr=.002,optimizer=kind)
    for p in params:p.grad=torch.randn_like(p)*.1
    original.step();original.zero_grad()
    lm_checkpoint(ref.config,weight='test',model=ref,optimizer=original,epoch=0,step=2,save_dir=tmp_path)
    source=tmp_path/'test_8_resume.pth'
    e=Engine(HostBackend());m=Model(e,cfg,activation_dtype='fp32')
    if lora:m.add_lora(2)
    opt=Optimizer(e,m.parameters(),kind=kind)
    path=tmp_path/'native.npz'
    with pytest.warns(RuntimeWarning,match='Legacy'):
        import_resume(source,path,m,opt,Data(tmp_path/'data.jsonl'),arguments(train_type='lora' if lora else 'pretrain'))
    metadata=checkpoint.metadata(path)
    assert metadata['progress']['cursor']==4 and metadata['progress']['steps']==2
    assert metadata['optimizers']['model']['accumulated']==0
    checkpoint.load(path,m,opt,np.random.default_rng(42))
    # The source checkpoint intentionally saves FP16 weights. Reload those in
    # the oracle so this tests state migration, not irrecoverable old FP32 bits.
    saved=torch.load(source,weights_only=True,map_location='cpu');ref.load_state_dict(saved['model'])
    for name,p in ref.named_parameters():
        if not p.requires_grad:continue
        grad=torch.randn_like(p)*.05;p.grad=grad
        opt.accumulate({m.params[name]:e.array(grad.numpy()).data})
    # Native accumulate counts microbatches, not individual parameters.
    opt.accumulated=1;opt.grad_clip=0
    original.step();opt.step()
    for name,p in ref.named_parameters():
        close(m.params[name].data,p,atol=4e-6,rtol=4e-4)
    # Re-export and reload to ensure recipe-specific group options persist.
    checkpoint.save(tmp_path/'roundtrip.npz',m,opt,metadata['progress'])
    checkpoint.load(tmp_path/'roundtrip.npz',m,opt)
    assert opt.param_options and all(x['lr']==.002 for x in opt.param_options.values())


@kernel_test
def test_scaler_and_cosine_scheduler_import(tmp_path):
    from model.model_instinct import InstinctConfig,InstinctForCausalLM
    ref=InstinctForCausalLM(InstinctConfig(**CONFIG));opt=torch.optim.AdamW(ref.parameters(),lr=.003,betas=(.8,.95),eps=1e-6)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=.0001)
    for p in ref.parameters():p.grad=torch.ones_like(p)*.02
    opt.step();sched.step()
    source=tmp_path/'old.pth';torch.save(dict(model=ref.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
        epoch=0,step=2,config=ref.config.to_dict(),scaler={'scale':1024.,'_growth_tracker':17,'growth_factor':3.,'backoff_factor':.25,'growth_interval':19}),source)
    e=Engine(HostBackend());m=Model(e,CONFIG,activation_dtype='fp16');native=Optimizer(e,m.parameters(),loss_scaling=True)
    with pytest.warns(RuntimeWarning):import_resume(source,tmp_path/'new.npz',m,native,Data(tmp_path/'data'),arguments())
    checkpoint.load(tmp_path/'new.npz',m,native)
    assert native.scale==1024 and native.good_steps==17 and native.growth_interval==19
    assert native.growth_factor==3 and native.backoff_factor==.25
    assert native.scheduler['last_epoch']==sched.last_epoch
    p=m.parameters()[0];native.accumulate({p:e.full(p.shape,0.).data});native.step(999.)
    opt.step();sched.step()
    np.testing.assert_allclose(native.scheduler['_last_lr'],sched.get_last_lr())


@kernel_test
def test_trainer_autodetects_legacy_resume(tmp_path):
    from model.model_instinct import InstinctConfig,InstinctForCausalLM
    from trainer.train_cublas import train,parser
    from tests.test_cublas_training import ToyTokenizer
    ref=InstinctForCausalLM(InstinctConfig(**CONFIG));opt=torch.optim.AdamW(ref.parameters())
    for p in ref.parameters():p.grad=torch.ones_like(p)*.01
    opt.step()
    directory=tmp_path/'checkpoints';directory.mkdir()
    torch.save(dict(model=ref.state_dict(),optimizer=opt.state_dict(),epoch=0,step=1,config=ref.config.to_dict()),directory/'pretrain_8_resume.pth')
    config=tmp_path/'config.json';config.write_text(json.dumps(CONFIG))
    data=tmp_path/'data.jsonl';data.write_text('\n'.join(json.dumps({'text':'abc'}) for _ in range(3)))
    args=parser().parse_args(['--config_path',str(config),'--data_path',str(data),'--from_resume','1','--dtype','fp32','--fp8','0',
        '--batch_size','1','--accumulation_steps','1','--max_seq_len','5','--max_steps','2','--num_workers','0',
        '--save_dir',str(tmp_path/'out'),'--checkpoint_dir',str(directory)])
    with pytest.warns(RuntimeWarning):result=train(args,HostBackend(),ToyTokenizer())
    assert result['steps']==2 and Path(result['resume']).suffix=='.npz'
    assert checkpoint.metadata(result['resume'])['progress']['cursor']==2


@kernel_test
def test_legacy_rl_extra_models_and_critic_scheduler(tmp_path):
    from model.model_instinct import InstinctConfig,InstinctForCausalLM
    cfg=InstinctConfig(**CONFIG)
    actor=InstinctForCausalLM(cfg);reference=InstinctForCausalLM(cfg);teacher=InstinctForCausalLM(cfg)
    critic=InstinctForCausalLM(cfg);critic.value_head=torch.nn.Linear(8,1)
    ao=torch.optim.AdamW(actor.parameters(),lr=.002);co=torch.optim.AdamW(critic.parameters(),lr=.004)
    schedules=[torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=7,eta_min=.0001) for o in (ao,co)]
    for module,opt,sched in zip((actor,critic),(ao,co),schedules):
        for p in module.parameters():p.grad=torch.randn_like(p)*.1
        opt.step();sched.step()
    source=tmp_path/'rl.pth';torch.save(dict(model=actor.state_dict(),optimizer=ao.state_dict(),config=cfg.to_dict(),epoch=0,step=2,
        ref_model=reference.state_dict(),teacher_model=teacher.state_dict(),critic_model=critic.state_dict(),
        critic_optimizer=co.state_dict(),scheduler=schedules[0].state_dict(),critic_scheduler=schedules[1].state_dict()),source)
    e=Engine(HostBackend());model=Model(e,CONFIG,activation_dtype='fp32')
    extra={k:Model(e,CONFIG,activation_dtype='fp32') for k in ('reference','teacher','critic')}
    extra['critic']._parameter('value_head.weight',(1,8));extra['critic']._parameter('value_head.bias',(1,))
    opt=Optimizer(e,model.parameters());critic_opt=Optimizer(e,extra['critic'].parameters())
    target=tmp_path/'rl.npz'
    with pytest.warns(RuntimeWarning):import_resume(source,target,model,opt,Data(tmp_path/'data'),arguments(train_type='ppo'),extra,{'critic':critic_opt})
    checkpoint.load(target,model,opt,extra_models=extra,extra_optimizers={'critic':critic_opt})
    for key,ref in [('reference',reference),('teacher',teacher),('critic',critic)]:
        for name,p in ref.named_parameters():close(extra[key].params[name].data,p)
    assert critic_opt.scheduler==schedules[1].state_dict()
    assert opt.scheduler==schedules[0].state_dict()


@kernel_test
def test_failed_legacy_mapping_does_not_replace_existing_destination(tmp_path):
    from model.model_instinct import InstinctConfig,InstinctForCausalLM
    ref=InstinctForCausalLM(InstinctConfig(**CONFIG));old=torch.optim.AdamW(ref.parameters())
    state=old.state_dict();state['param_groups'][0]['params'].pop()
    source=tmp_path/'bad.pth';torch.save(dict(model=ref.state_dict(),optimizer=state,epoch=0,step=0),source)
    e=Engine(HostBackend());m=Model(e,CONFIG,activation_dtype='fp32');opt=Optimizer(e,m.parameters())
    path=tmp_path/'unchanged.npz';path.write_bytes(b'original contents')
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):import_resume(source,path,m,opt,Data(tmp_path/'data'),arguments())
    assert path.read_bytes()==b'original contents'


@kernel_test
def test_legacy_linear_parameter_order(tmp_path):
    from model.model_instinct_linear import InstinctConfig,InstinctForCausalLM
    cfg=CONFIG|dict(num_hidden_layers=2,model_architecture='linear',linear_key_head_dim=4,linear_value_head_dim=4,
        linear_num_key_heads=1,linear_num_value_heads=2,linear_conv_kernel_dim=3,full_attention_interval=2)
    ref=InstinctForCausalLM(InstinctConfig(**cfg));opt=torch.optim.AdamW(ref.parameters(),lr=.001)
    for index,p in enumerate(ref.parameters()):p.grad=torch.full_like(p,index*.001)
    opt.step()
    source=tmp_path/'linear.pth';torch.save(dict(model=ref.state_dict(),optimizer=opt.state_dict(),config=ref.config.to_dict(),epoch=0,step=2),source)
    e=Engine(HostBackend());m=Model(e,cfg,activation_dtype='fp32');native=Optimizer(e,m.parameters())
    with pytest.warns(RuntimeWarning):import_resume(source,tmp_path/'linear.npz',m,native,Data(tmp_path/'data'),arguments())
    checkpoint.load(tmp_path/'linear.npz',m,native)
    for name,p in ref.named_parameters():
        close(native.state[m.params[name]]['m'].data,opt.state[p]['exp_avg'])


@kernel_test
def test_builtin_muon_recipe_restores_distinct_averaged_momentum(tmp_path):
    # A torch.optim.Muon-style group differs from the repository fallback:
    # EMA momentum, shape-adjusted LR, decay before update, BF16 NS iterations.
    from model.model_instinct import InstinctConfig,InstinctForCausalLM
    ref=InstinctForCausalLM(InstinctConfig(**CONFIG))
    matrices=[p for p in ref.parameters() if p.ndim==2];vectors=[p for p in ref.parameters() if p.ndim!=2]
    matrix=dict(state={i:{'momentum_buffer':torch.ones_like(p)*.02} for i,p in enumerate(matrices)},
        param_groups=[dict(params=list(range(len(matrices))),lr=.001,weight_decay=.1,momentum=.95,nesterov=True,
            ns_coefficients=(3.4445,-4.775,2.0315),ns_steps=0,eps=1e-7,adjust_lr_fn='match_rms_adamw')])
    # Zero NS iterations isolates momentum/normalization/LR/decay semantics;
    # full BF16 Tensor Core NS parity is a separate GPU acceptance check.
    other=torch.optim.AdamW(vectors,lr=.001).state_dict()
    source=tmp_path/'muon.pth';torch.save(dict(model=ref.state_dict(),optimizer={'optimizers':[matrix,other]},epoch=0,step=2,config=ref.config.to_dict()),source)
    e=Engine(HostBackend());m=Model(e,CONFIG,activation_dtype='fp32');opt=Optimizer(e,m.parameters(),kind='muon',grad_clip=0)
    with pytest.warns(RuntimeWarning):import_resume(source,tmp_path/'muon.npz',m,opt,Data(tmp_path/'data'),arguments())
    checkpoint.load(tmp_path/'muon.npz',m,opt)
    p=m.parameters()[0];before=e.numpy(p).copy();g=np.ones(p.shape,np.float32)*.1
    opt.accumulate({p:e.array(g).data});opt.step()
    np.testing.assert_allclose(e.numpy(opt.state[p]['momentum']),.02*.95+.1*.05,atol=1e-7)
    # BF16 conversion is only approximated by Triton's CPU interpreter.
    expected=before*(1.-.001*.1)-(.001*.2*max(p.shape)**.5)/np.sqrt(p.size)
    np.testing.assert_allclose(e.numpy(p),expected,atol=4e-6)


@pytest.mark.parametrize('mode,prefix',[('pretrain','pretrain'),('lora','lora_medical'),('ppo','ppo_actor'),('distillation','full_dist')])
def test_legacy_default_filename_detection(tmp_path,mode,prefix):
    from trainer.train_cublas import parser,resolve_resume
    args=parser().parse_args(['--train_type',mode,'--from_resume','1','--checkpoint_dir',str(tmp_path)])
    source=tmp_path/f'{prefix}_8_resume.pth';source.touch()
    output,found=resolve_resume(args,mode+'_8')
    assert found==source and output==source.with_suffix('.npz')
    own=tmp_path/f'{mode}_8_resume.npz';own.touch()
    assert resolve_resume(args,mode+'_8')==(own,own)
