"""Offline CPU-only .pth <-> native .npz conversion; --resume-spec imports state."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
import math
from pathlib import Path
import tempfile
import numpy as np


def clean_name(name):
    while name.startswith(('module.','_orig_mod.')):name=name.split('.',1)[1]
    return name


def _scalar(value):
    return value.item() if hasattr(value,'item') else value


def _plain(value):
    if isinstance(value,dict):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    return _scalar(value)


def _atomic_npz(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def convert_resume(loaded,spec,source):
    """Translate the repository's AdamW/Adafactor/CombinedMuon checkpoints.

    PyTorch parameter IDs are resolved in source state_dict registration order,
    with tied aliases deduplicated. Explicit param_names take precedence. No
    positional mapping against the native model's different registration order.
    """
    import torch
    if not isinstance(loaded,dict) or not {'model','optimizer','epoch','step'} <= loaded.keys():
        raise ValueError('full resume requires model, optimizer, epoch and step; this appears to be weights-only')
    cfg=loaded.get('config',{})
    structural=('hidden_size','num_hidden_layers','vocab_size','intermediate_size','num_attention_heads',
        'num_key_value_heads','head_dim','tie_word_embeddings','use_moe','num_experts','num_experts_per_tok',
        'moe_intermediate_size','hidden_act','rms_norm_eps','rope_theta','rope_scaling','dropout',
        'full_attention_interval','linear_conv_kernel_dim','linear_key_head_dim','linear_value_head_dim',
        'linear_num_key_heads','linear_num_value_heads')
    for key in structural:
        if key in cfg and key in spec['config'] and cfg[key]!=spec['config'][key]:
            raise ValueError('legacy checkpoint configuration mismatch: '+key)
    source_keys={'model':'model','reference':'ref_model','teacher':'teacher_model','critic':'critic_model'}
    arrays={};source_parameters={}
    for group,expected in spec['models'].items():
        key=source_keys[group]
        if key not in loaded:raise ValueError('legacy checkpoint is missing '+key+'; cannot reconstruct its saved state')
        weights={clean_name(k):v for k,v in loaded[key].items()}
        unknown=[k for k in weights if k not in expected and not k.endswith(('freqs_cos','freqs_sin'))]
        if unknown:raise ValueError('unmapped legacy model parameters (old Looped structures require a separate migration): '+', '.join(unknown[:5]))
        order=[];seen=set()
        for name,info in expected.items():
            if name not in weights:raise ValueError('missing legacy parameter: '+name)
            value=weights[name]
            if list(value.shape)!=info['shape']:raise ValueError('legacy parameter shape mismatch: '+name)
            arrays[group+'/'+name]=value.detach().float().cpu().numpy()
            canonical=info['canonical']
            if canonical in weights and name!=canonical and not torch.equal(value,weights[canonical]):
                raise ValueError('legacy tied aliases have different weights: '+name)
        for name in weights:
            if name in expected:
                canonical=expected[name]['canonical']
                if canonical not in seen:order.append(canonical);seen.add(canonical)
        source_parameters[group]=order

    epoch,step,world=int(loaded['epoch']),int(loaded['step']),int(loaded.get('world_size',1))
    n=spec['dataset_size'];source_batch=spec['source_batch_size'];accum=spec['source_accumulation_steps']
    if min(n,source_batch,accum,world,spec['source_epochs'])<1 or min(epoch,step)<0:raise ValueError('invalid legacy progress/settings')
    source_iters=math.ceil(math.ceil(n/world)/source_batch)
    if step>source_iters:raise ValueError('legacy step exceeds dataset epoch; provide the original --legacy_batch_size/dataset')
    cursor=min(step*source_batch*world,n)
    # Original trainers reset the torch sampler seed at each epoch. Reconstruct
    # the saved epoch order in the CPU converter, never importing torch in CUDA.
    sampler_seed=epoch if world>1 else 42+epoch
    order=torch.randperm(n,generator=torch.Generator().manual_seed(sampler_seed)).tolist()
    global_steps=epoch*source_iters+step
    estimated_updates=epoch*math.ceil(source_iters/accum)+step//accum
    notes=['Legacy checkpoint did not save RNG/worker augmentation state or pending gradients; future random masks/rollouts are reseeded, not bitwise replayed.',
           'Legacy model weights were saved as FP16; original higher-precision master-weight bits cannot be recovered.',
           'Legacy batch size, accumulation and epoch count were not stored; using the supplied --legacy_* values (or current settings).']
    if step%accum:notes.append('Checkpoint was inside an accumulation group: unsaved partial gradients are reset, matching the available saved state.')
    if world>1:notes.append('Imported multi-GPU progress into one GPU; global consumed-token count is preserved, but batch grouping and random history change.')
    progress=dict(epoch=epoch,cursor=cursor,order=order,steps=global_steps,dataset_size=n,dataset_path=spec['dataset_path'],
        batch_size=spec['batch_size'],accumulation_steps=spec['accumulation_steps'],train_type=spec['train_type'],
        data_seed=spec['seed'],loader_version=1)
    metadata=dict(format='instinct-cublas-v1',config=spec['config'],precision=spec['precision'],fp8=spec['fp8'],
        progress=progress,engine_rng={'seed':spec['seed'],'counter':0},rng=np.random.default_rng(spec['seed']).bit_generator.state,
        optimizers={},legacy_import=dict(source=str(source),world_size=world,warnings=notes))

    for group,definition in spec['optimizers'].items():
        raw=loaded['optimizer' if group=='model' else 'critic_optimizer']
        kind=definition['metadata']['kind'];info=dict(definition['metadata'])
        info.update(steps={},present=[],accumulated=0,param_options={},scheduler=None)
        candidates=[name for name in source_parameters[group] if name in definition['parameters']]
        if set(candidates)!=set(definition['parameters']):raise ValueError('legacy trainable parameter names do not match '+group)
        combined='optimizers' in raw
        if combined and (kind!='muon' or len(raw['optimizers'])!=2):raise ValueError('unsupported CombinedOptimizer layout')
        components=raw['optimizers'] if combined else [raw]
        assigned=set();group_index=0
        for component in components:
            groups=component['param_groups']
            is_muon=any('momentum' in g for g in groups)
            is_ada=any('beta2_decay' in g for g in groups)
            actual='muon' if is_muon else 'adafactor' if is_ada else 'adamw'
            if actual!=kind and not (kind=='muon' and actual=='adamw' and combined):
                raise ValueError(f'legacy optimizer is {actual}, requested {kind}')
            selected=candidates
            if combined:selected=[p for p in candidates if (len(spec['models'][group][p]['shape'])==2)==is_muon]
            ids=[pid for g in groups for pid in g['params']]
            if len(ids)!=len(selected) or len(ids)!=len(set(ids)):raise ValueError('legacy optimizer parameter count/order is incompatible')
            unnamed=dict(zip(ids,selected))
            for pg in groups:
                names=pg.get('param_names')
                if names is not None and len(names)!=len(pg['params']):raise ValueError('invalid param_names')
                options=_plain({k:v for k,v in pg.items() if k not in ('params','param_names')})
                if options.get('amsgrad'):raise ValueError('AMSGrad was not exposed by the original trainers and is not supported')
                if options.get('differentiable'):raise ValueError('differentiable optimizer steps cannot be imported')
                if actual=='adafactor' and ('relative_step' in options or 'beta2_decay' not in options):
                    raise ValueError('unsupported pre-2.8 Adafactor recipe; use the matching original runtime')
                if is_muon:
                    options['variant']='fallback_muon' if 'orthogonalize_scale' in options else 'torch_muon'
                options['group']=group_index;group_index+=1
                for j,pid in enumerate(pg['params']):
                    name=clean_name(names[j]) if names is not None else unnamed[pid]
                    if name in spec['models'][group]:name=spec['models'][group][name]['canonical']
                    if name not in definition['states'] or name in assigned:raise ValueError('ambiguous optimizer parameter mapping: '+name)
                    assigned.add(name);info['param_options'][name]=dict(options)
                    state=component['state'].get(pid,component['state'].get(str(pid),{}))
                    info['steps'][name]=int(_scalar(state.get('step',estimated_updates if state and is_muon else 0)))
                    mapping={'m':'exp_avg','v':'exp_avg_sq','row':'row_var','col':'col_var','variance':'variance','momentum':'momentum_buffer'}
                    consumed={'step'}
                    for target,shape in definition['states'][name].items():
                        source_name=mapping[target];consumed.add(source_name)
                        value=state.get(source_name)
                        if state and value is None:raise ValueError('missing optimizer state '+name+'/'+source_name)
                        array=np.zeros(shape,np.float32) if value is None else value.detach().float().cpu().numpy()
                        if list(array.shape)!=shape:raise ValueError('optimizer state shape mismatch: '+name+'/'+source_name)
                        arrays[f'optimizer/{group}/{name}/{target}']=array
                    if set(state)-consumed:raise ValueError('unrecognized optimizer state for '+name+': '+str(set(state)-consumed))
                    arrays[f'gradient/{group}/{name}']=np.zeros(spec['models'][group][name]['shape'],np.float32)
        if assigned!=set(candidates):raise ValueError('not all optimizer parameters were mapped')
        info['updates']=max(info['steps'].values(),default=0)
        first=next(iter(info['param_options'].values()))
        info['lr']=first['lr']
        scaler=loaded.get('scaler',{}) if group=='model' else {}
        if scaler:
            if not info['loss_scaling']:raise ValueError('legacy GradScaler requires --dtype float16 to resume its behavior')
            info.update(scale=float(scaler['scale']),good_steps=int(scaler.get('_growth_tracker',0)),
                growth_factor=float(scaler.get('growth_factor',2.)),backoff_factor=float(scaler.get('backoff_factor',.5)),
                growth_interval=int(scaler.get('growth_interval',2000)))
        scheduler=loaded.get('scheduler' if group=='model' else 'critic_scheduler')
        if scheduler:
            if not {'T_max','eta_min','base_lrs','last_epoch','_last_lr'}<=scheduler.keys():raise ValueError('only the original CosineAnnealingLR scheduler is supported')
            if len(scheduler['base_lrs'])!=group_index or scheduler['T_max']<=0:raise ValueError('invalid scheduler groups/T_max')
            info['scheduler']=_plain(scheduler)
        metadata['optimizers'][group]=info

    if not metadata['optimizers']['model']['scheduler']:
        # Supervised legacy schedules set LR using a 1-based microbatch index.
        total=spec['source_epochs']*source_iters
        factor=.1+.45*(1.+math.cos(math.pi*min(global_steps/total,1.)))
        maximum=spec['learning_rate']
        if maximum is None:
            maximum=metadata['optimizers']['model']['lr']/factor
            notes.append('Original maximum LR was not stored; inferred it from saved LR and --legacy_epochs. Set --learning_rate to override.')
        progress['legacy_schedule']=dict(maximum=maximum,total=total)
    arrays['__metadata__']=np.asarray(json.dumps(metadata,ensure_ascii=False))
    return arrays


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume-spec',help='Native trainer manifest for importing complete legacy training state')
    args=p.parse_args()
    if Path(args.input).resolve()==Path(args.output).resolve():raise ValueError('conversion output must not overwrite its source')
    # Windows pyarrow DLL ordering; imports remain confined to this CPU process.
    import datasets
    import torch
    if args.input.endswith('.pth'):
        loaded=torch.load(args.input,map_location='cpu',weights_only=True)
        if args.resume_spec:
            spec=json.loads(Path(args.resume_spec).read_text(encoding='utf-8'))
            arrays=convert_resume(loaded,spec,args.input)
        else:
            if 'model' in loaded and isinstance(loaded['model'],dict):loaded=loaded['model']
            if 'state_dict' in loaded:loaded=loaded['state_dict']
            arrays={'model/'+clean_name(name):value.detach().float().numpy() for name,value in loaded.items() if isinstance(value,torch.Tensor)}
            arrays['__metadata__']=np.asarray(json.dumps({'format':'instinct-cublas-v1','source':args.input,'weights_only':True}))
        _atomic_npz(args.output,arrays)
    else:
        if args.resume_spec:raise ValueError('--resume-spec requires a legacy .pth source')
        with np.load(args.input,allow_pickle=False) as z:
            state={name.removeprefix('model/'):torch.from_numpy(z[name].copy()) for name in z if name.startswith('model/')}
        torch.save(state,args.output)
    print(args.output)


if __name__=='__main__':main()
