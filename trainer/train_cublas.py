"""Native cuBLASLt/Triton training entry point. Does not import Torch/TorchAO.

Run from the repository root. Original .pth weights can be converted by the
separate CPU-only conversion utility; native resume checkpoints use .npz.
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import threading
import sys
import time

__package__='trainer'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from model.dense_cublas.autograd import Engine
from model.dense_cublas.model import Model
from model.dense_cublas.optim import Optimizer,cosine_lr
from model.dense_cublas.data import Tokenizer,JsonlDataset
from model.dense_cublas import checkpoint,objectives
from model.dense_cublas.rollout import trajectory,reward,collate
from model.dense_cublas.config import WebUIRequest,TRAIN_TYPES


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train_type',choices=TRAIN_TYPES,default='pretrain')
    p.add_argument('--config_path');p.add_argument('--teacher_config_path');p.add_argument('--tokenizer_path',default='./model')
    for name in ('hidden_size','num_hidden_layers','use_moe','use_looped','use_grad_checkpoint','early_exit'):
        p.add_argument('--'+name,type=int,default=None)
    p.add_argument('--dtype',choices=['fp32','bfloat16','float16'],default='bfloat16')
    p.add_argument('--param_dtype',choices=['fp32','bf16','fp16'],default='fp32')
    p.add_argument('--kv_cache_dtype',choices=['fp32','bf16','fp16','fp8_e4m3','fp8_e5m2'],default='fp32')
    p.add_argument('--fp8',choices=['auto','0','1'],default='auto')
    p.add_argument('--optimizer',choices=['adamw','adafactor','muon'],default='adamw')
    p.add_argument('--learning_rate',type=float,default=None);p.add_argument('--weight_decay',type=float,default=.01)
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch_size',type=int,default=32)
    p.add_argument('--accumulation_steps',type=int,default=8);p.add_argument('--max_seq_len',type=int,default=768)
    p.add_argument('--grad_clip',type=float,default=1.);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='0');p.add_argument('--data_path');p.add_argument('--dataset_size',choices=['mini','normal'],default='mini')
    p.add_argument('--from_weight',default='auto');p.add_argument('--from_student_weight',default=None);p.add_argument('--from_teacher_weight',default=None)
    p.add_argument('--from_resume',type=int,default=0);p.add_argument('--resume_path')
    p.add_argument('--save_weight');p.add_argument('--lora_name');p.add_argument('--lora_rank',type=int,default=16)
    p.add_argument('--save_dir',default='./out');p.add_argument('--checkpoint_dir',default='./checkpoints')
    p.add_argument('--log_interval',type=int,default=10);p.add_argument('--save_interval',type=int,default=100)
    p.add_argument('--max_steps',type=int,default=0);p.add_argument('--use_wandb',action='store_true');p.add_argument('--wandb_project',default='Instinct-cuBLAS')
    p.add_argument('--peak_tflops',type=float,default=0.)
    p.add_argument('--allocator_cache_mib',type=int,default=256)
    p.add_argument('--workspace_mib',type=int,default=32);p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--tune',action='store_true');p.add_argument('--overlap_dw',action='store_true');p.add_argument('--dw_sm_target',type=int,default=0)
    p.add_argument('--beta',type=float,default=.1);p.add_argument('--alpha',type=float,default=.5);p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--max_gen_len',type=int,default=256);p.add_argument('--max_total_len',type=int,default=2500);p.add_argument('--max_turns',type=int,default=3)
    p.add_argument('--num_generations',type=int,default=4);p.add_argument('--thinking_ratio',type=float,default=.5)
    p.add_argument('--loss_type',choices=['grpo','cispo'],default='cispo');p.add_argument('--epsilon',type=float,default=.2);p.add_argument('--epsilon_high',type=float,default=5.)
    p.add_argument('--reward_model_path',default='../internlm2-1_8b-reward')
    p.add_argument('--critic_learning_rate',type=float,default=5e-7);p.add_argument('--clip_epsilon',type=float,default=.2)
    p.add_argument('--cliprange_value',type=float,default=.2);p.add_argument('--vf_coef',type=float,default=.5);p.add_argument('--kl_coef',type=float,default=.02)
    p.add_argument('--gamma',type=float,default=1.);p.add_argument('--lam',type=float,default=.95)
    p.add_argument('--ppo_update_iters',type=int,default=2);p.add_argument('--mini_batch_size',type=int,default=2);p.add_argument('--early_stop_kl',type=float,default=.25)
    return p


def read_config(args):
    raw=json.loads(Path(args.config_path).read_text(encoding='utf-8')) if args.config_path else {}
    for key in ('hidden_size','num_hidden_layers','use_moe','use_grad_checkpoint'):
        if getattr(args,key) is not None:raw[key]=getattr(args,key)
    if args.use_looped:raw['model_architecture']='looped'
    if args.early_exit==0:raw['early_exit_layers']=[]
    request=WebUIRequest.from_dicts(raw,dict(train_type=args.train_type,batch_size=args.batch_size,max_seq_len=args.max_seq_len,
        optimizer=args.optimizer,param_dtype=args.param_dtype,activation_dtype=args.dtype,kv_cache_dtype=args.kv_cache_dtype))
    for key in ('epochs','batch_size','max_seq_len','accumulation_steps','log_interval','save_interval','max_gen_len','max_total_len','max_turns','num_generations','ppo_update_iters','mini_batch_size'):
        if getattr(args,key)<=0:raise ValueError(key+' must be positive')
    if args.max_seq_len<2:raise ValueError('max_seq_len must allow at least one prediction target')
    if args.train_type in ('grpo','agent') and args.num_generations<2:raise ValueError('group RL requires num_generations >= 2')
    if args.temperature<=0 or not 0<=args.alpha<=1:raise ValueError('temperature must be positive and alpha in [0,1]')
    return request.model


def resolve_weight(value,args,config,kind=None):
    if value in (None,'none','none (from scratch)'):return None
    if value=='auto':
        kind=kind or ('pretrain' if args.train_type=='full_sft' else 'full_sft')
        candidates=[]
        for root in (Path(args.save_dir),Path(args.checkpoint_dir)):
            for path in root.glob(kind+'*'):
                if path.suffix in ('.npz','.pth') and '_resume' not in path.stem and path.stem.endswith(f"_{config['hidden_size']}{'_moe' if config.get('use_moe') else ''}"):
                    candidates.append(path)
        if not candidates:
            if args.train_type=='pretrain':return None
            raise FileNotFoundError(f'No {kind} base weights found; select weights or explicitly use --from_weight none')
        path=max(candidates,key=lambda p:p.stat().st_mtime)
    else:
        path=Path(value)
        if not path.exists():
            path=Path(args.save_dir)/f"{value}_{config['hidden_size']}{'_moe' if config.get('use_moe') else ''}.npz"
    if not path.exists():raise FileNotFoundError(path)
    if path.suffix=='.pth':
        converted=path.with_suffix('.npz')
        if not converted.exists() or converted.stat().st_mtime<path.stat().st_mtime:
            print(f'Converting {path} in a separate CPU-only process.',flush=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1')
            subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/convert_cublas_checkpoint.py'),'--input',str(path),'--output',str(converted)],check=True,env=env)
        path=converted
    return path


def copy_parameters(source,destination):
    e=source.e
    with e.no_grad():
        for name,p in destination.params.items():
            if name not in source.params:continue
            if source.params[name].shape!=p.shape:raise ValueError('reference shape mismatch: '+name)
            e.copy(p,source.params[name])


def critic_values(model,ids,mask):
    out=model(ids,attention_mask=mask,return_logits=False)
    # Match the project's CriticModel, including its extra final RMSNorm.
    hidden=model.norm(out.hidden,'model.norm.weight')
    return (model.linear(hidden,'value_head.weight')+model.params['value_head.bias']).reshape(*ids.shape)


def supervised_step(args,model,batch,reference=None,teacher=None):
    e=model.e;ids,labels=batch;out=model(ids,labels)
    if args.train_type=='dpo':
        policy=objectives.sequence_logps(e,out.logits[:,:-1],e.array(labels[:,1:]))
        with e.no_grad():ref=reference(ids)
        base=objectives.sequence_logps(e,ref.logits[:,:-1],e.array(labels[:,1:]))
        return objectives.dpo(e,policy,base,args.beta)+out.aux_loss
    if args.train_type=='distillation':
        with e.no_grad():teach=teacher(ids)
        vocab=out.logits.shape[-1]
        if teach.logits.shape[-1]<vocab:raise ValueError('teacher vocabulary must cover student token IDs')
        kd=objectives.distillation(e,out.logits[:,:-1],teach.logits[:,:-1,:vocab],e.array(labels[:,1:]),args.temperature)
        return args.alpha*(out.loss+out.aux_loss)+(1.-args.alpha)*kd
    return out.loss+out.aux_loss


def rl_step(args,model,reference,critic,optimizer,critic_optimizer,rows,tokenizer,rng,reward_model,lr):
    e=model.e;group=1 if args.train_type=='ppo' else args.num_generations
    trajectories=[trajectory(model,tokenizer,row,rng,args.max_gen_len,args.max_total_len,args.max_turns,args.thinking_ratio,args.train_type=='agent') for row in rows for _ in range(group)]
    rewards=np.asarray([reward(t,reward_model,args.train_type=='agent') for t in trajectories],np.float32)
    ids,attention,mask,old=collate(trajectories,tokenizer.pad)
    if not mask.any():raise ValueError('rollout contains no generated tokens; increase max_total_len or shorten prompts')
    # Evaluation mode makes old/current policy probabilities independent of dropout.
    prior=model.training;model.training=False
    try:
        with e.no_grad():
            ref=reference(ids,attention_mask=attention)
            reference_logps=e.numpy(e.log_probs(ref.logits[:,:-1],e.array(ids[:,1:])))
            if critic is not None:
                critic.training=False
                old_values=e.numpy(critic_values(critic,ids,attention)[:,:-1])
        if critic is None:
            advantage=objectives.group_advantages(rewards,group)
            out=model(ids,attention_mask=attention)
            current=e.log_probs(out.logits[:,:-1],e.array(ids[:,1:]))
            loss=objectives.grpo(e,current,e.array(old),e.array(reference_logps),e.array(advantage),e.array(mask),args.beta,args.epsilon,args.epsilon_high,args.loss_type)+out.aux_loss
            optimizer.backward(loss)
            return e.item(loss),float(rewards.mean()),ids.size
        advantage,returns=objectives.gae(rewards,old_values,mask,args.gamma,args.lam)
        losses=[];stop=False
        for repeat in range(args.ppo_update_iters):
            order=rng.permutation(len(ids))
            for begin in range(0,len(ids),args.mini_batch_size):
                mark=e.backend.mark();idx=order[begin:begin+args.mini_batch_size]
                out=model(ids[idx],attention_mask=attention[idx]);current=e.log_probs(out.logits[:,:-1],e.array(ids[idx,1:]))
                values=critic_values(critic,ids[idx],attention[idx])[:,:-1]
                total,policy,value,kl=objectives.ppo(e,current,e.array(old[idx]),e.array(reference_logps[idx]),values,e.array(old_values[idx]),
                    e.array(advantage[idx]),e.array(returns[idx]),e.array(mask[idx]),args.clip_epsilon,args.cliprange_value,args.kl_coef,args.vf_coef)
                total=total+out.aux_loss;losses.append(e.item(total))
                gradients=e.vjp(total*optimizer.scale);optimizer.accumulate(gradients)
                critic_gradients={p:(e.wrap(g)*(critic_optimizer.scale/optimizer.scale)).data for p,g in gradients.items() if p in critic_optimizer.gradients}
                critic_optimizer.accumulate(critic_gradients)
                stop=e.item(kl)>args.early_stop_kl
                if optimizer.accumulated>=args.accumulation_steps or begin+args.mini_batch_size>=len(ids) or stop:
                    optimizer.step(lr);critic_optimizer.step(args.critic_learning_rate)
                e.backend.release_since(mark)
                if stop:break
            if stop:break
        return float(np.mean(losses)),float(rewards.mean()),ids.size
    finally:model.training=prior


def train(args,backend=None,tokenizer=None,reward_model=None):
    config=read_config(args)
    dtype={'bfloat16':'bf16','float16':'fp16','fp32':'fp32'}[args.dtype]
    lr=args.learning_rate if args.learning_rate is not None else (5e-4 if args.train_type=='pretrain' else 1e-6 if args.train_type in ('ppo','grpo','agent') else 5e-5)
    if backend is None:
        from model.dense_cublas.memory import TrainingBackend
        from model.dense_cublas.backend import ScheduleConfig
        backend_context=TrainingBackend(int(args.device.removeprefix('cuda:')),ScheduleConfig(workspace_mib=args.workspace_mib,candidates=args.candidates,tune=args.tune,overlap_dw=args.overlap_dw,dw_sm_target=args.dw_sm_target),pool_limit_mib=args.allocator_cache_mib)
    else:backend_context=nullcontext(backend)
    rng=np.random.default_rng(args.seed);tokenizer=tokenizer or Tokenizer(args.tokenizer_path)
    suffix='_mini' if args.dataset_size=='mini' else ''
    default_data={'pretrain':f'pretrain_t2t{suffix}','full_sft':f'sft_t2t{suffix}','distillation':f'sft_t2t{suffix}',
        'lora':'lora_medical','dpo':'dpo','ppo':'rlaif','grpo':'rlaif','agent':'agent_rl'}
    data=JsonlDataset(args.data_path or './dataset/'+default_data[args.train_type]+'.jsonl',tokenizer,args.train_type,args.max_seq_len,rng)
    with backend_context as backend:
        fp8=args.fp8=='1' or (args.fp8=='auto' and getattr(backend,'arch',0)>=89)
        e=Engine(backend,fp8=fp8,seed=args.seed)
        model=Model(e,config,args.param_dtype,dtype,args.kv_cache_dtype,args.seed)
        if args.train_type=='lora':model.add_lora(args.lora_rank)
        name=args.lora_name or args.save_weight or args.train_type
        stem=f"{name}_{config['hidden_size']}{'_moe' if config.get('use_moe') else ''}"
        weights=Path(args.save_dir)/(stem+'.npz');resume=Path(args.resume_path) if args.resume_path else Path(args.checkpoint_dir)/(stem+'_resume.npz')
        if args.from_resume and not resume.exists():raise FileNotFoundError('native resume checkpoint not found: '+str(resume))
        if not args.from_resume:
            base=resolve_weight(args.from_student_weight or args.from_weight,args,config)
            if base:checkpoint.load(base,model)
        reference=teacher=critic=critic_optimizer=None
        if args.train_type in ('dpo','ppo','grpo','agent'):
            reference=Model(e,config,args.param_dtype,dtype,args.kv_cache_dtype,args.seed).freeze();copy_parameters(model,reference)
        if args.train_type=='distillation':
            teacher_config=json.loads(Path(args.teacher_config_path).read_text(encoding='utf-8')) if args.teacher_config_path else config
            teacher=Model(e,teacher_config,args.param_dtype,dtype,args.kv_cache_dtype,args.seed).freeze()
            teacher_path=resolve_weight(args.from_teacher_weight or args.from_student_weight or args.from_weight,args,teacher.config)
            if teacher_path is None:raise ValueError('distillation requires --from_teacher_weight')
            checkpoint.load(teacher_path,teacher)
        if args.train_type=='ppo':
            critic=Model(e,config,args.param_dtype,dtype,args.kv_cache_dtype,args.seed);copy_parameters(model,critic)
            critic._parameter('value_head.weight',(1,config['hidden_size']));critic._parameter('value_head.bias',(1,),zeros=True)
        optimizer=Optimizer(e,model.parameters(),args.optimizer,lr,args.weight_decay,args.grad_clip,loss_scaling=dtype=='fp16')
        if critic:critic_optimizer=Optimizer(e,critic.parameters(),args.optimizer,args.critic_learning_rate,args.weight_decay,args.grad_clip,loss_scaling=dtype=='fp16')
        extra_models={k:v for k,v in {'reference':reference,'teacher':teacher,'critic':critic}.items() if v is not None}
        extra_optimizers={'critic':critic_optimizer} if critic_optimizer else {}
        progress={'epoch':0,'cursor':0,'order':None,'steps':0,'dataset_size':len(data),'dataset_path':str(data.path.resolve()),'batch_size':args.batch_size,'accumulation_steps':args.accumulation_steps,'train_type':args.train_type}
        if args.from_resume:
            progress=checkpoint.load(resume,model,optimizer,rng,extra_models,extra_optimizers)
            expected={'dataset_size':len(data),'dataset_path':str(data.path.resolve()),'batch_size':args.batch_size,'accumulation_steps':args.accumulation_steps,'train_type':args.train_type}
            for key,value in expected.items():
                if progress.get(key)!=value:raise ValueError(key+' changed since native resume checkpoint')
        if args.train_type in ('ppo','grpo','agent') and reward_model is None:
            from model.dense_cublas.reward import InternLMReward
            reward_model=InternLMReward(backend,args.reward_model_path,dtype='fp16' if dtype=='fp16' else 'bf16')
        logger=None
        if args.use_wandb:
            import swanlab
            logger=swanlab.init(project=args.wandb_project,config=vars(args))
        total_batches=math.ceil(len(data)/args.batch_size);total_steps=args.epochs*total_batches
        print(f'Backend=cuBLASLt/Triton | mode={args.train_type} | FP8={fp8} | parameters={sum(p.size for p in model.parameters(False)):,}',flush=True)
        started=time.perf_counter();last_loss=None;stop_requested=False
        previous_sigint=None
        if threading.current_thread() is threading.main_thread():
            previous_sigint=signal.getsignal(signal.SIGINT)
            def request_stop(signum,frame):
                nonlocal stop_requested
                stop_requested=True
                print('Stop requested; finishing this batch before saving native resume state.',flush=True)
            signal.signal(signal.SIGINT,request_stop)
        def save_all():
            checkpoint.save(weights,model)
            if args.train_type=='lora':checkpoint.save(weights.with_name(weights.stem+'_adapter.npz'),model,adapters_only=True)
            checkpoint.save(resume,model,optimizer,progress,rng,extra_models,extra_optimizers)
        try:
            for epoch in range(progress['epoch'],args.epochs):
                order=progress['order'] if progress['order'] is not None else rng.permutation(len(data)).tolist()
                progress.update(epoch=epoch,order=order)
                cursor=progress['cursor']
                for begin in range(cursor,len(order),args.batch_size):
                    mark=backend.mark();step_start=time.perf_counter();batch=data.batch(order[begin:begin+args.batch_size])
                    current_lr=cosine_lr(progress['steps'],total_steps,lr)
                    if args.train_type in ('ppo','grpo','agent'):
                        last_loss,mean_reward,tokens=rl_step(args,model,reference,critic,optimizer,critic_optimizer,batch,tokenizer,rng,reward_model,current_lr)
                    else:
                        if not np.any(batch[1][:,1:]!=-100):
                            print('Skipping batch with no supervised tokens.',flush=True)
                            progress['cursor']=min(begin+args.batch_size,len(order));backend.release_since(mark);continue
                        loss=supervised_step(args,model,batch,reference,teacher);last_loss=e.item(loss);optimizer.backward(loss);tokens=batch[0].size
                    if optimizer.accumulated>=args.accumulation_steps or begin+args.batch_size>=len(order):optimizer.step(current_lr)
                    progress['steps']+=1;progress['cursor']=min(begin+args.batch_size,len(order))
                    backend.synchronize();elapsed=time.perf_counter()-step_start;rate=tokens/max(elapsed,1e-9)
                    backend.release_since(mark)
                    if progress['steps']%args.log_interval==0 or progress['cursor']==len(order):
                        eta=(time.perf_counter()-started)/max(progress['steps'],1)*(total_steps-progress['steps'])/60.
                        print(f'Epoch:[{epoch+1}/{args.epochs}]({math.ceil(progress["cursor"]/args.batch_size)}/{total_batches}), loss: {last_loss:.6f}, learning_rate: {current_lr:.8f}, epoch_time: {eta:.3f}min, tokens/s: {rate:.1f}',flush=True)
                        metrics={'loss':last_loss,'learning_rate':current_lr,'tokens_per_second':rate,'step':progress['steps']}
                        if args.peak_tflops:
                            metrics['estimated_mfu']=6*sum(p.size for p in model.parameters(False))*rate/(args.peak_tflops*1e12)
                        if logger:logger.log(metrics)
                    if progress['steps']%args.save_interval==0:
                        save_all()
                    if stop_requested or (args.max_steps and progress['steps']>=args.max_steps):
                        save_all();return {'weights':str(weights),'resume':str(resume),'loss':last_loss,'steps':progress['steps']}
                progress.update(epoch=epoch+1,cursor=0,order=None)
                save_all()
        finally:
            if previous_sigint is not None:signal.signal(signal.SIGINT,previous_sigint)
            if logger:logger.finish()
        if 'torch' in sys.modules and type(backend).__module__.startswith('model.'):
            raise RuntimeError('Torch was imported into the native training process')
        return {'weights':str(weights),'resume':str(resume),'loss':last_loss,'steps':progress['steps']}


if __name__=='__main__':
    from experiments.smoke_dense_triton import RejectTorch
    if 'torch' in sys.modules or 'torchao' in sys.modules:raise RuntimeError('start native training in a fresh process')
    sys.meta_path.insert(0,RejectTorch())
    print(train(parser().parse_args()),flush=True)
