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
from model.dense_cublas.data import Tokenizer,JsonlDataset,PackedPretrain,bucket_order,trim_batch
from model.dense_cublas.prefetch import DevicePrefetch
from model.dense_cublas.loader import BatchLoader
from model.dense_cublas.distributed import ProcessGroup,shard_order
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
    p.add_argument('--fp8_filter',choices=['auto','eligible'],default='eligible')
    p.add_argument('--muon_mantissa',type=int,choices=[0,1],default=0)
    p.add_argument('--muon_momentum_dtype',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--muon_batch_size',type=int,default=16)
    p.add_argument('--muon_workspace_mb',type=float,default=512)
    p.add_argument('--muon_gemm_backend',choices=['triton','cublas'],default='triton')
    p.add_argument('--optimizer',choices=['adamw','adafactor','muon'],default='adamw')
    p.add_argument('--learning_rate',type=float,default=None);p.add_argument('--weight_decay',type=float,default=.01)
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch_size',type=int,default=32)
    p.add_argument('--accumulation_steps',type=int,default=8);p.add_argument('--max_seq_len',type=int,default=768)
    p.add_argument('--num_workers',type=int,default=8);p.add_argument('--prefetch_factor',type=int,default=2)
    p.add_argument('--token_cache',help='SQLite tokenization cache path')
    p.add_argument('--packing',action='store_true',help='EOS-delimited pretraining token packing')
    p.add_argument('--packed_attention',action='store_true',help='Isolate packed documents, reset RoPE and mask cross-document labels')
    p.add_argument('--packing_cache',default='./cache/native_tokens')
    p.add_argument('--length_bucket',type=int,default=0,help='Sort lengths within shuffled windows of this size')
    p.add_argument('--sequence_buckets',default='',help='Comma separated sequence lengths')
    p.add_argument('--copy_prefetch',action='store_true')
    p.add_argument('--async_checkpoint',action='store_true')
    p.add_argument('--cuda_graph',action='store_true')
    p.add_argument('--fuse_projections',action='store_true')
    p.add_argument('--fused_ops',action='store_true',help='Native Dense norm/cast and SwiGLU/amax fusion with compact gradients')
    p.add_argument('--attention_backend',choices=['triton','cudnn','hybrid'],default='triton')
    p.add_argument('--attention_cudnn_min_seq',type=int,default=4096)
    p.add_argument('--loss_chunk_size',type=int,default=0)
    p.add_argument('--distributed_backend',choices=['auto','nccl','tcp'],default='auto')
    p.add_argument('--pin_memory',type=int,choices=[0,1],default=1)
    p.add_argument('--grad_clip',type=float,default=1.);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default=os.environ.get('LOCAL_RANK','0'));p.add_argument('--data_path');p.add_argument('--dataset_size',choices=['mini','normal'],default='mini')
    p.add_argument('--from_weight',default='auto');p.add_argument('--from_student_weight',default=None);p.add_argument('--from_teacher_weight',default=None)
    p.add_argument('--from_resume',type=int,default=0);p.add_argument('--resume_path')
    p.add_argument('--legacy_batch_size',type=int);p.add_argument('--legacy_accumulation_steps',type=int)
    p.add_argument('--legacy_epochs',type=int)
    p.add_argument('--save_weight');p.add_argument('--lora_name');p.add_argument('--lora_rank',type=int,default=16)
    p.add_argument('--save_dir',default='./out');p.add_argument('--checkpoint_dir',default='./checkpoints')
    p.add_argument('--log_interval',type=int,default=10);p.add_argument('--save_interval',type=int,default=100)
    p.add_argument('--max_steps',type=int,default=0);p.add_argument('--use_wandb',action='store_true');p.add_argument('--wandb_project',default='Instinct-cuBLAS')
    p.add_argument('--peak_tflops',type=float,default=0.)
    p.add_argument('--allocator_cache_mib',type=int,default=256)
    p.add_argument('--allocator_backend',choices=['driver','async'],default='driver')
    p.add_argument('--allocator_release_mib',type=int,default=None,
                   help='Async CUDA pool retention threshold in MiB, not a memory cap; omitted preserves the driver setting')
    p.add_argument('--selective_cache_mib',type=int)
    p.add_argument('--selective_cache_by_seq',default='',help='Optional per-bucket saved-output MiB, e.g. 3072:768,4096:1024')
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
    if args.use_looped and raw.get('model_architecture') not in ('looped','looped_legacy'):raw['model_architecture']='looped_legacy'
    if any(k in raw for k in ('prelude_layers','loop_iters','coda_layers')) and raw.get('model_architecture','looped')=='looped':raw['model_architecture']='looped_legacy'
    if args.early_exit==0:raw['early_exit_layers']=[]
    if args.selective_cache_mib is not None:
        if args.selective_cache_mib<0:raise ValueError('selective cache budget must be nonnegative')
        raw['selective_cache_mib']=args.selective_cache_mib
    if args.selective_cache_by_seq:
        from model.dense_cublas.selective import parse_cache_budgets
        raw['selective_cache_by_seq']=parse_cache_budgets(args.selective_cache_by_seq)
    request=WebUIRequest.from_dicts(raw,dict(train_type=args.train_type,batch_size=args.batch_size,max_seq_len=args.max_seq_len,
        optimizer=args.optimizer,param_dtype=args.param_dtype,activation_dtype=args.dtype,kv_cache_dtype=args.kv_cache_dtype,num_workers=args.num_workers,prefetch_factor=args.prefetch_factor,pin_memory=bool(args.pin_memory)))
    for key in ('epochs','batch_size','max_seq_len','accumulation_steps','log_interval','save_interval','max_gen_len','max_total_len','max_turns','num_generations','ppo_update_iters','mini_batch_size'):
        if getattr(args,key)<=0:raise ValueError(key+' must be positive')
    for key in ('legacy_batch_size','legacy_accumulation_steps','legacy_epochs'):
        if getattr(args,key) is not None and getattr(args,key)<1:raise ValueError(key+' must be positive')
    if args.num_workers < 0 or args.prefetch_factor < 1:raise ValueError('invalid data worker/prefetch settings')
    if args.packing and args.train_type!='pretrain':raise ValueError('packing is defined for pretraining only')
    if args.packed_attention and not args.packing:raise ValueError('packed_attention requires packing')
    if args.muon_mantissa and (args.optimizer!='muon' or args.param_dtype!='bf16'):
        raise ValueError('Muon mantissa requires optimizer=muon and param_dtype=bf16')
    if args.muon_momentum_dtype=='bf16' and not args.muon_mantissa:
        raise ValueError('BF16 Muon momentum requires mantissa')
    if args.muon_workspace_mb<=0 or args.muon_batch_size<1:raise ValueError('invalid Muon chunk limits')
    if request.model['use_grad_checkpoint'] in (3,4) and (args.fuse_projections or args.train_type=='lora'):
        raise ValueError('native ckpt3/4 require separate Dense projections without LoRA')
    if args.length_bucket<0:raise ValueError('length_bucket must be nonnegative')
    if args.copy_prefetch and args.train_type in ('ppo','grpo','agent'):raise ValueError('copy_prefetch expects supervised token batches')
    if args.cuda_graph and (args.train_type not in ('pretrain','full_sft','lora') or args.copy_prefetch):raise ValueError('CUDA Graph supports supervised CE; use its fixed input uploads without copy_prefetch')
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


def resolve_resume(args,stem):
    destination=Path(args.resume_path) if args.resume_path else Path(args.checkpoint_dir)/(stem+'_resume.npz')
    source=destination
    if not args.from_resume:return destination,source
    if not source.exists() and source.suffix=='.npz':source=source.with_suffix('.pth')
    if not source.exists() and not args.resume_path and not args.save_weight and not args.lora_name:
        old_prefix={'lora':'lora_medical','ppo':'ppo_actor','distillation':'full_dist'}.get(args.train_type,args.train_type)
        legacy_stem=old_prefix+stem[len(args.train_type):]
        source=Path(args.checkpoint_dir)/(legacy_stem+'_resume.npz')
        if not source.exists():source=source.with_suffix('.pth')
    if not source.exists():raise FileNotFoundError('resume checkpoint not found: '+str(destination))
    return source.with_suffix('.npz'),source


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
    e=model.e;ids,labels=batch[:2]
    out=model(ids,labels,return_logits=args.train_type in ('dpo','distillation'),
              sequence_ids=batch[2] if len(batch)==3 else None)
    if args.train_type=='dpo':
        policy=objectives.sequence_logps(e,out.logits[:,:-1],labels[:,1:] if hasattr(labels,'data') and not isinstance(labels,np.ndarray) else e.array(labels[:,1:]))
        with e.no_grad():ref=reference(ids)
        base=objectives.sequence_logps(e,ref.logits[:,:-1],labels[:,1:] if hasattr(labels,'data') and not isinstance(labels,np.ndarray) else e.array(labels[:,1:]))
        return objectives.dpo(e,policy,base,args.beta)+out.aux_loss
    if args.train_type=='distillation':
        with e.no_grad():teach=teacher(ids)
        vocab=out.logits.shape[-1]
        if teach.logits.shape[-1]<vocab:raise ValueError('teacher vocabulary must cover student token IDs')
        kd=objectives.distillation(e,out.logits[:,:-1],teach.logits[:,:-1,:vocab],labels[:,1:] if hasattr(labels,'data') and not isinstance(labels,np.ndarray) else e.array(labels[:,1:]),args.temperature)
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
                if optimizer.distributed:stop=optimizer.distributed.any(stop)
                if optimizer.accumulated>=args.accumulation_steps or begin+args.mini_batch_size>=len(ids) or stop:
                    optimizer.step(lr);critic_optimizer.step(args.critic_learning_rate)
                e.backend.release_since(mark)
                if stop:break
            if stop:break
        return float(np.mean(losses)),float(rewards.mean()),ids.size
    finally:model.training=prior


def train(args,backend=None,tokenizer=None,reward_model=None,observer=None):
    config=read_config(args)
    dtype={'bfloat16':'bf16','float16':'fp16','fp32':'fp32'}[args.dtype]
    lr=args.learning_rate if args.learning_rate is not None else (5e-4 if args.train_type=='pretrain' else 1e-6 if args.train_type in ('ppo','grpo','agent') else 5e-5)
    if backend is None:
        from model.dense_cublas.memory import TrainingBackend
        from model.dense_cublas.backend import ScheduleConfig
        backend_context=TrainingBackend(int(args.device.removeprefix('cuda:')),ScheduleConfig(workspace_mib=args.workspace_mib,candidates=args.candidates,tune=args.tune,overlap_dw=args.overlap_dw,dw_sm_target=args.dw_sm_target),pool_limit_mib=args.allocator_cache_mib, pin_memory=bool(args.pin_memory), allocator_backend=args.allocator_backend,async_release_mib=args.allocator_release_mib)
    else:backend_context=nullcontext(backend)
    rng=np.random.default_rng(args.seed);tokenizer=tokenizer or Tokenizer(args.tokenizer_path,args.token_cache)
    suffix='_mini' if args.dataset_size=='mini' else ''
    default_data={'pretrain':f'pretrain_t2t{suffix}','full_sft':f'sft_t2t{suffix}','distillation':f'sft_t2t{suffix}',
        'lora':'lora_medical','dpo':'dpo','ppo':'rlaif','grpo':'rlaif','agent':'agent_rl'}
    buckets=sorted(set(int(x) for x in args.sequence_buckets.split(',') if x))
    if any(x<2 or x>args.max_seq_len for x in buckets):raise ValueError('sequence_buckets must be between 2 and max_seq_len')
    path=args.data_path or './dataset/'+default_data[args.train_type]+'.jsonl'
    streaming=Path(path).suffix.lower() in ('.parquet','.pq')
    if streaming:
        if args.train_type!='pretrain' or not args.packing or not args.packed_attention:
            raise ValueError('native Parquet currently requires packed, document-isolated pretraining')
        from model.dense_cublas.parquet_data import ParquetPretrain
        data=ParquetPretrain(path,tokenizer,args.packing_cache,lengths=buckets or [args.max_seq_len],
            token_budget=args.batch_size*args.max_seq_len,threads=max(1,args.num_workers),seed=args.seed)
        loader_context=data
    else:
        data=JsonlDataset(path,tokenizer,args.train_type,args.max_seq_len,rng)
        if args.packing:data=PackedPretrain(data,args.packing_cache,isolate_documents=args.packed_attention)
        loader_context=BatchLoader(data,args.num_workers,args.prefetch_factor,args.seed)
    with backend_context as backend, ProcessGroup(backend,args.distributed_backend) as group, loader_context as loader, checkpoint.AsyncCheckpointer(args.async_checkpoint) as saver:
        if streaming and group.world!=1:raise ValueError('bounded native Parquet currently supports one GPU')
        if args.fused_ops:
            if config.get('use_moe') or config.get('model_architecture','standard')!='standard':
                raise ValueError('fused_ops currently requires the standard Dense backbone')
            from model.dense_cublas.policy import enable_fused_ops
            enable_fused_ops(backend)
        backend.muon_gemm_backend=args.muon_gemm_backend
        if args.muon_gemm_backend=='cublas':
            if not args.muon_mantissa or args.optimizer!='muon':
                raise ValueError('Muon cuBLASLt GEMM requires split-weight Muon')
            native=getattr(backend,'native',None)
            if native is None or native._plan_batched is None or native._run_batched is None:
                raise RuntimeError('rebuild the native bridge before selecting Muon cuBLASLt GEMM')
        if args.attention_cudnn_min_seq<1:raise ValueError('attention threshold must be positive')
        if args.attention_backend!='triton':
            if not args.packed_attention or dtype!='bf16' or args.cuda_graph:
                raise ValueError('native cuDNN attention currently requires BF16 packed attention without CUDA Graph')
            from model.dense_cublas.vendor_attention import NativeAttention
            backend.native_attention=NativeAttention(backend)
            backend.native_attention.minimum_seq=args.attention_cudnn_min_seq if args.attention_backend=='hybrid' else 0
            if args.attention_backend=='hybrid':
                backend.attention_forward_tile=128;backend.attention_forward_cols=64
                backend.kernel_options['model.dense_triton.attention.forward']=dict(num_warps=8,num_stages=2)
        fp8=args.fp8=='1' or (args.fp8=='auto' and getattr(backend,'arch',0)>=89)
        e=Engine(backend,fp8=fp8,seed=args.seed+group.rank,fp8_filter=args.fp8_filter)
        model=Model(e,config,args.param_dtype,dtype,args.kv_cache_dtype,args.seed,muon_mantissa=args.muon_mantissa)
        model.fuse_projections=args.fuse_projections;model.loss_chunk_size=args.loss_chunk_size
        if args.loss_chunk_size<0:raise ValueError('loss_chunk_size must be nonnegative')
        if args.train_type=='lora':model.add_lora(args.lora_rank)
        name=args.lora_name or args.save_weight or args.train_type
        stem=f"{name}_{config['hidden_size']}{'_moe' if config.get('use_moe') else ''}"
        weights=Path(args.save_dir)/(stem+'.npz')
        resume,resume_source=resolve_resume(args,stem)
        base_resume=resume
        if group.rank:
            resume=resume.with_name(resume.stem+f'.rank{group.rank}'+resume.suffix)
            if args.from_resume and resume.exists():resume_source=resume
        if not args.from_resume:
            base=resolve_weight(args.from_student_weight or args.from_weight,args,config)
            if base:checkpoint.load(base,model)
        reference=teacher=critic=critic_optimizer=None
        if args.train_type in ('dpo','ppo','grpo','agent'):
            reference=Model(e,config,args.param_dtype,dtype,args.kv_cache_dtype,args.seed).freeze();copy_parameters(model,reference)
        if args.train_type=='distillation':
            teacher_config=json.loads(Path(args.teacher_config_path).read_text(encoding='utf-8')) if args.teacher_config_path else config
            teacher=Model(e,teacher_config,args.param_dtype,dtype,args.kv_cache_dtype,args.seed).freeze()
            if not args.from_resume:
                teacher_path=resolve_weight(args.from_teacher_weight or args.from_student_weight or args.from_weight,args,teacher.config)
                if teacher_path is None:raise ValueError('distillation requires --from_teacher_weight')
                checkpoint.load(teacher_path,teacher)
        if args.train_type=='ppo':
            critic=Model(e,config,args.param_dtype,dtype,args.kv_cache_dtype,args.seed);copy_parameters(model,critic)
            critic._parameter('value_head.weight',(1,config['hidden_size']));critic._parameter('value_head.bias',(1,),zeros=True)
        optimizer=Optimizer(e,model.parameters(),args.optimizer,lr,args.weight_decay,args.grad_clip,loss_scaling=dtype=='fp16',
                            muon_mantissa=args.muon_mantissa,muon_momentum_dtype=args.muon_momentum_dtype,
                            muon_batch_size=args.muon_batch_size,muon_workspace_mb=args.muon_workspace_mb)
        if critic:critic_optimizer=Optimizer(e,critic.parameters(),args.optimizer,args.critic_learning_rate,args.weight_decay,args.grad_clip,loss_scaling=dtype=='fp16')
        optimizer.distributed=group
        if critic_optimizer:critic_optimizer.distributed=group
        extra_models={k:v for k,v in {'reference':reference,'teacher':teacher,'critic':critic}.items() if v is not None}
        extra_optimizers={'critic':critic_optimizer} if critic_optimizer else {}
        progress={'epoch':0,'cursor':0,'order':None,'steps':0,'dataset_size':len(data),'dataset_path':str(data.path.resolve()),'batch_size':args.batch_size,'accumulation_steps':args.accumulation_steps,'train_type':args.train_type,'data_seed':args.seed,'loader_version':1,'sampler_version':2,'world_size':group.world,'data_options':dict(packing=args.packing,length_bucket=args.length_bucket,sequence_buckets=buckets,packed_attention=args.packed_attention),'compute_options':dict(fuse_projections=args.fuse_projections,loss_chunk_size=args.loss_chunk_size)}
        progress['compute_options']['fused_ops']=args.fused_ops
        progress['compute_options']['muon_gemm_backend']=args.muon_gemm_backend
        progress['compute_options'].update(attention_backend=args.attention_backend,attention_cudnn_min_seq=args.attention_cudnn_min_seq)
        data_options=dict(progress['data_options'])
        if streaming:
            data_options['parquet_fingerprint']=data.fingerprint
            progress.update(data_options=data_options,parquet_cursor=dict(group=0,batch=0),processed_tokens=0)
        if args.from_resume:
            if resume_source.suffix=='.pth':
                from model.dense_cublas.legacy_checkpoint import import_resume
                import_resume(resume_source,resume,model,optimizer,data,args,extra_models,extra_optimizers)
            progress=checkpoint.load(resume,model,optimizer,rng,extra_models,extra_optimizers)
            imported=checkpoint.metadata(resume).get('legacy_import')
            if imported:progress['legacy_import']=imported
            progress.get('data_options',{}).setdefault('packed_attention',False)
            if progress.get('data_options',dict(packing=False,length_bucket=0,sequence_buckets=[],packed_attention=False))!=data_options:raise ValueError('data packing/buckets changed since resume')
            compute_options=dict(fuse_projections=args.fuse_projections,loss_chunk_size=args.loss_chunk_size,
                fused_ops=args.fused_ops,muon_gemm_backend=args.muon_gemm_backend,
                attention_backend=args.attention_backend,attention_cudnn_min_seq=args.attention_cudnn_min_seq)
            if imported:progress['compute_options']=compute_options
            previous_compute=progress.get('compute_options',dict(fuse_projections=False,loss_chunk_size=0))
            previous_compute.setdefault('fused_ops',False)
            previous_compute.setdefault('muon_gemm_backend','triton')
            previous_compute.setdefault('attention_backend','triton')
            previous_compute.setdefault('attention_cudnn_min_seq',4096)
            if previous_compute!=compute_options:raise ValueError('native precision-affecting execution options changed since resume')
            old_world=progress.get('world_size',1)
            if old_world!=group.world:
                if optimizer.accumulated or (critic_optimizer and critic_optimizer.accumulated):
                    from model.dense_cublas.distributed import restore_pending
                    restore_pending(base_resume,{'model':optimizer,**extra_optimizers},old_world,progress['steps'])
                if progress.get('order') is not None:
                    progress['order']=progress['order'][min(progress['cursor']*old_world,len(progress['order'])):]
                progress['cursor']=0;progress['world_size']=group.world;progress['sampler_version']=2
                e.counter+=group.rank*1000003
                rng=np.random.default_rng(np.random.SeedSequence((args.seed,group.rank,progress['steps'])))
                data.rng=rng
            progress.setdefault('loader_version',0);progress.setdefault('data_seed',args.seed)
            if progress['loader_version']==0:
                loader.serial_rng=True
                print('Older native checkpoint: preserving serial augmentation RNG; CPU prefetch is disabled for this run.',flush=True)
            expected={'dataset_size':len(data),'dataset_path':str(data.path.resolve()),'batch_size':args.batch_size,'accumulation_steps':args.accumulation_steps,'train_type':args.train_type,'data_seed':args.seed}
            for key,value in expected.items():
                if progress.get(key)!=value:raise ValueError(key+' changed since native resume checkpoint')
        if group.world>1:
            steps=group.sum(np.eye(group.world,dtype=np.float32)[group.rank]*progress['steps'])
            if not np.all(steps==steps[0]):raise ValueError('rank checkpoints are from different steps; restore a complete checkpoint set')
        if args.train_type in ('ppo','grpo','agent') and reward_model is None:
            from model.dense_cublas.reward import InternLMReward
            reward_model=InternLMReward(backend,args.reward_model_path,dtype='fp16' if dtype=='fp16' else 'bf16')
        logger=None
        if args.use_wandb and group.rank==0:
            import swanlab
            logger=swanlab.init(project=args.wandb_project,config=vars(args))
        total_batches=math.ceil(math.ceil(len(data)/group.world)/args.batch_size);total_steps=args.epochs*total_batches
        if streaming:
            total_batches=math.ceil(data.plan['tokens']/(data.token_budget*.97))
            total_steps=args.epochs*total_batches
        if observer:observer.start(args,backend,model,optimizer,data,progress,rng=rng)
        print(f'Backend=cuBLASLt/Triton | mode={args.train_type} | FP8={fp8} | parameters={sum(p.size for p in model.parameters(False)):,}',flush=True)
        started=time.perf_counter();last_loss=None;stop_requested=False
        report_started=started;report_tokens=0
        previous_sigint=None
        if threading.current_thread() is threading.main_thread():
            previous_sigint=signal.getsignal(signal.SIGINT)
            def request_stop(signum,frame):
                nonlocal stop_requested
                stop_requested=True
                print('Pause requested; finishing the accumulated optimizer update before saving native resume state.',flush=True)
            signal.signal(signal.SIGINT,request_stop)
        def save_all():
            if group.rank==0:saver.save(weights,model,progress=progress)
            if group.rank==0 and args.train_type=='lora':saver.save(weights.with_name(weights.stem+'_adapter.npz'),model,adapters_only=True)
            saver.save(resume,model,optimizer,progress,rng,extra_models,extra_optimizers)
        prefetch=DevicePrefetch(e,args.batch_size*(2 if args.train_type=='dpo' else 1),args.max_seq_len,args.copy_prefetch,fields=3 if args.packed_attention else 2)
        step_graph=None
        try:
            for epoch in range(progress['epoch'],args.epochs):
                if streaming:
                    progress.update(epoch=epoch,order=None)
                    batches=loader.batches(epoch,progress.get('parquet_cursor'))
                    order=None
                else:
                    sampler_rng=rng if progress.get('sampler_version',1)==1 else np.random.default_rng(np.random.SeedSequence((args.seed,epoch)))
                    global_order=progress['order'] if progress['order'] is not None else sampler_rng.permutation(len(data)).tolist()
                    if progress['order'] is None and args.length_bucket:global_order=bucket_order(data,global_order,args.length_bucket)
                    progress.update(epoch=epoch,order=global_order)
                    order=shard_order(global_order,group.world,group.rank)
                    cursor=progress['cursor']
                    batches=((begin,trim_batch(batch,tokenizer.pad,buckets)) for begin,batch in loader.batches(order,args.batch_size,epoch,cursor))
                wall_start=time.perf_counter()
                for begin,batch,device_batch in prefetch.batches(batches):
                    mark=backend.mark();step_start=time.perf_counter()
                    last_batch=begin['group']>=len(data.plan['groups']) if streaming else begin+args.batch_size>=len(order)
                    next_cursor=progress['cursor']+len(batch[0]) if streaming else min(begin+args.batch_size,len(order))
                    schedule=progress.get('legacy_schedule')
                    current_lr=cosine_lr(progress['steps']+1,schedule['total'],schedule['maximum']) if schedule else cosine_lr(progress['steps'],total_steps,lr)
                    if streaming:current_lr=cosine_lr(progress['processed_tokens']+begin['valid_tokens'],data.plan['tokens']*args.epochs,lr)
                    if optimizer.scheduler:current_lr=optimizer.scheduler['_last_lr'][0]
                    graph_used=False
                    if args.train_type in ('ppo','grpo','agent'):
                        last_loss,mean_reward,tokens=rl_step(args,model,reference,critic,optimizer,critic_optimizer,batch,tokenizer,rng,reward_model,current_lr)
                    else:
                        if not group.any(np.any(batch[1][:,1:]!=-100)):
                            print('Skipping batch with no supervised tokens.',flush=True)
                            progress['cursor']=next_cursor
                            if streaming:
                                progress['parquet_cursor']=dict(begin)
                                progress['processed_tokens']+=begin['valid_tokens']
                            backend.release_since(mark);continue
                        if args.cuda_graph and step_graph is None:
                            from model.dense_cublas.graph import StepGraph
                            step_graph=StepGraph(model,optimizer,batch,lambda values:supervised_step(args,model,values),args.accumulation_steps)
                        graph_used=step_graph is not None and tuple(x.shape for x in batch)==step_graph.shape
                        if graph_used:last_loss=step_graph.run(batch,current_lr,last_batch)
                        else:
                            loss=supervised_step(args,model,device_batch,reference,teacher)
                            optimizer.backward(loss)
                            if args.train_type!='pretrain':last_loss=e.item(loss)
                        tokens=int(np.count_nonzero(batch[1][:,1:]!=-100)) if args.train_type=='pretrain' else batch[0].size
                    if not graph_used and args.train_type=='pretrain':
                        # Gradients have reached persistent optimizer buffers.
                        # Release checkpoint activations before allocating NS
                        # workspaces; only the scalar loss is still consumed.
                        backend.release_since(mark,keep=(loss.data,))
                    if not graph_used and (optimizer.accumulated>=args.accumulation_steps or last_batch):optimizer.step(current_lr)
                    progress['steps']+=1;progress['cursor']=next_cursor
                    if streaming:
                        progress['parquet_cursor']=dict(begin)
                        progress['processed_tokens']+=begin['valid_tokens']
                    stop_file=os.environ.get('INSTINCT_STOP_FILE')
                    pause_requested=stop_requested or bool(stop_file and Path(stop_file).exists()) or Path(args.checkpoint_dir,'.pause_request').exists()
                    pause_requested=group.any(pause_requested)
                    limit_reached=group.any(bool(args.max_steps and progress['steps']>=args.max_steps))
                    stopping=(pause_requested or limit_reached) and optimizer.accumulated==0
                    reporting=group.rank==0 and (progress['steps']%args.log_interval==0 or last_batch)
                    # Pretraining only downloads the loss when it is consumed.
                    # This also completes queued work before reporting/stopping.
                    if not graph_used:
                        if args.train_type=='pretrain':
                            if reporting or last_batch or stopping:last_loss=e.item(loss)
                        else:backend.synchronize()
                    now=time.perf_counter();elapsed=now-wall_start;wall_start=now
                    rate=tokens/max(elapsed,1e-9)
                    report_tokens+=tokens
                    if reporting and args.train_type=='pretrain':
                        rate=report_tokens/max(now-report_started,1e-9)
                        report_tokens=0;report_started=now
                    backend.release_since(mark)
                    if observer:observer.step(backend,model,optimizer,progress,batch,tokens,elapsed)
                    if reporting:
                        eta=(time.perf_counter()-started)/max(progress['steps'],1)*(total_steps-progress['steps'])/60.
                        print(f'Epoch:[{epoch+1}/{args.epochs}]({math.ceil(progress["cursor"]/args.batch_size)}/{total_batches}), loss: {last_loss:.6f}, learning_rate: {current_lr:.8f}, epoch_time: {eta:.3f}min, tokens/s: {rate:.1f}',flush=True)
                        metrics={'loss':last_loss,'learning_rate':current_lr,'tokens_per_second':rate,'step':progress['steps']}
                        if step_graph is not None:
                            metrics.update(cuda_graph_captures=step_graph.plan.capture_count,
                                           cuda_graph_replays=step_graph.plan.replay_count,
                                           cuda_graph_commands=len(step_graph.plan.commands))
                            print(f'CUDA Graph: captures={step_graph.plan.capture_count}, replays={step_graph.plan.replay_count}, commands={len(step_graph.plan.commands)}',flush=True)
                        if args.peak_tflops:
                            metrics['estimated_mfu']=6*sum(p.size for p in model.parameters(False))*rate/(args.peak_tflops*1e12)
                        if logger:logger.log(metrics)
                    if progress['steps']%args.save_interval==0:
                        save_all()
                    if stopping:
                        save_all();saver.flush()
                        assert checkpoint.metadata(resume)['progress']['steps']==progress['steps']
                        if group.rank==0:assert checkpoint.metadata(weights)['progress']['steps']==progress['steps']
                        if pause_requested:
                            request=Path(args.checkpoint_dir,'.pause_request')
                            if group.rank==0 and request.exists():request.unlink()
                            print(f'Native training paused: verified full state at step {progress["steps"]}.',flush=True)
                        return {'weights':str(weights),'resume':str(resume),'loss':last_loss,'steps':progress['steps'],'paused':bool(pause_requested)}
                progress.update(epoch=epoch+1,cursor=0,order=None)
                if streaming:progress['parquet_cursor']=dict(group=0,batch=0)
                save_all()
        finally:
            prefetch.close()
            saver.flush()
            if previous_sigint is not None:signal.signal(signal.SIGINT,previous_sigint)
            if logger:logger.finish()
        if 'torch' in sys.modules and type(backend).__module__.startswith('model.'):
            raise RuntimeError('Torch was imported into the native training process')
        return {'weights':str(weights),'resume':str(resume),'loss':last_loss,'steps':progress['steps']}


if __name__=='__main__':
    from experiments.smoke_dense_triton import RejectTorch
    if 'torch' in sys.modules or 'torchao' in sys.modules:raise RuntimeError('start native training in a fresh process')
    sys.meta_path.insert(0,RejectTorch())
    result=train(parser().parse_args())
    print(result,flush=True)
    if result.get('paused'):raise SystemExit(42)
