"""Atomic, pickle-free native weights and complete training checkpoints."""
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
import numpy as np


def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def snapshot(model,optimizer=None,progress=None,rng=None,extra_models=None,extra_optimizers=None,adapters_only=False):
    arrays={};models={'model':model,**(extra_models or {})};optimizers={'model':optimizer,**(extra_optimizers or {})}
    metadata=dict(format='instinct-cublas-v1',config=model.config,progress=progress or {},
        engine_rng={'seed':model.e.seed,'counter':model.e.counter},rng=None if rng is None else rng.bit_generator.state,
        precision=dict(param_dtype=model.param_dtype,activation_dtype=model.activation_dtype,kv_cache_dtype=model.kv_cache_dtype),
        fp8=model.e.ops.fp8,fp8_filter=model.e.fp8_filter,optimizers={})
    if hasattr(model,'cache_precision_reference'):
        metadata['cache_precision_reference']=model.cache_precision_reference
    for group,m in models.items():
        for name,p in m.params.items():
            if not adapters_only or '.lora.' in name:arrays[f'{group}/{name}']=m.e.numpy(p).astype(np.float32, copy=False)
    for group,opt in optimizers.items():
        if opt is None:continue
        info=opt.metadata();info['steps']={};info['present']=[p.name for p in opt.present]
        for p,state in opt.state.items():
            info['steps'][p.name]=state['step']
            for name,value in state.items():
                if name!='step':
                    arrays[f'optimizer/{group}/{p.name}/{name}']=opt.e.numpy(value).astype(np.uint16 if value.dtype=='u16' else np.float32, copy=False)
            arrays[f'gradient/{group}/{p.name}']=opt.e.numpy(opt.gradients[p]).astype(np.float32, copy=False)
        metadata['optimizers'][group]=info
    arrays['__metadata__']=np.asarray(json.dumps(metadata,default=_json,ensure_ascii=False))
    return arrays


def write(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def save(path,*args,**kwargs):
    write(path,snapshot(*args,**kwargs))


def _write_training(resume, weights, adapter, arrays, report):
    started = perf_counter()
    if weights is not None:
        info = json.loads(str(arrays['__metadata__']))
        info.update(optimizers={}, rng=None)
        model = {key: array for key, array in arrays.items() if key.startswith('model/')}
        write(weights, {**model, '__metadata__': np.asarray(json.dumps(info, ensure_ascii=False))})
        if adapter is not None:
            info['progress'] = {}
            adapters = {key: array for key, array in model.items() if '.lora.' in key}
            write(adapter, {**adapters, '__metadata__': np.asarray(json.dumps(info, ensure_ascii=False))})
    # Publish resume last. Each archive is written and replaced individually,
    # so the temporary files never require room for two checkpoint archives.
    write(resume, arrays)
    if report:
        print(f'[Native Checkpoint] step={report} background write completed in '
              f'{perf_counter() - started:.2f}s', flush=True)


class AsyncCheckpointer:
    """One immutable CPU snapshot in flight; IO errors surface at next save/close."""
    def __init__(self,enabled=True):
        from concurrent.futures import ThreadPoolExecutor
        self.executor=ThreadPoolExecutor(max_workers=1) if enabled else None
        self.pending=None
    def __enter__(self):return self
    def flush(self):
        if self.pending is not None:
            pending,self.pending=self.pending,None;pending.result()
    def save(self,path,*args,**kwargs):
        self.flush()  # Bound host memory and preserve overwrite ordering.
        arrays=snapshot(*args,**kwargs)
        if self.executor:self.pending=self.executor.submit(write,path,arrays)
        else:write(path,arrays)
    def save_training(self, resume, model, optimizer, progress, rng,
                      extra_models=None, extra_optimizers=None, *, weights=None, adapter=None):
        self.flush()
        started = perf_counter()
        arrays = snapshot(model, optimizer, progress, rng, extra_models, extra_optimizers)
        if self.executor:
            step = progress['steps']
            size = sum(array.nbytes for array in arrays.values()) / 2**30
            print(f'[Native Checkpoint] step={step} staged {size:.2f} GiB in '
                  f'{perf_counter() - started:.2f}s; writing in background', flush=True)
            self.pending = self.executor.submit(_write_training, resume, weights, adapter, arrays, step)
        else:
            _write_training(resume, weights, adapter, arrays, None)

    def __exit__(self,*exc):
        try:self.flush()
        finally:
            if self.executor:self.executor.shutdown(wait=True)


def metadata(path):
    with np.load(path,allow_pickle=False) as z:return json.loads(str(z['__metadata__'])) if '__metadata__' in z else {}


def validate_resume_config(saved,current):
    """Allow only ckpt4 saved-activation budget changes during full resume.

    These fields select transient forward caches, not persistent training state.
    In particular, changing the checkpoint mode or MXFP8 precision is still an
    error. MXFP8 resumes pin the original compressed layer selection separately;
    additional cached FFN outputs stay BF16 to avoid new quantization error.
    """
    if isinstance(saved,dict) and isinstance(current,dict):
        saved={'freeze_embeddings':False,**saved}
        current={'freeze_embeddings':False,**current}
    if saved==current:return None
    budgets=('selective_cache_mib','selective_cache_by_seq')
    if (not isinstance(saved,dict) or not isinstance(current,dict)
            or saved.get('use_grad_checkpoint')!=4 or current.get('use_grad_checkpoint')!=4
            or {k:v for k,v in saved.items() if k not in budgets}
                != {k:v for k,v in current.items() if k not in budgets}):
        raise ValueError('model configuration differs from the native resume checkpoint')
    return dict(previous={k:saved[k] for k in budgets if k in saved},
                current={k:current[k] for k in budgets if k in current})


def load(path,model,optimizer=None,rng=None,extra_models=None,extra_optimizers=None,adapters_only=False):
    if str(path).endswith('.pth'):
        raise ValueError('Convert .pth weights first: python scripts/convert_cublas_checkpoint.py --input PATH --output PATH.npz')
    e=model.e;models={'model':model,**(extra_models or {})};optimizers={'model':optimizer,**(extra_optimizers or {})}
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['__metadata__'])) if '__metadata__' in z else {}
        cache_change=None
        cache_reference=None
        if optimizer is not None:
            cache_change=validate_resume_config(meta.get('config'),model.config)
            cache_reference=meta.get('cache_precision_reference')
            if model.config.get('native_mxfp8_cache_ffn') and cache_change and cache_reference is None:
                cache_reference={k:meta['config'][k] for k in ('selective_cache_mib','selective_cache_by_seq') if k in meta['config']}
            if cache_reference is not None:
                from .selective import CACHE_BYTES
                current=model.config
                lengths=set(cache_reference.get('selective_cache_by_seq',{}))|set(current.get('selective_cache_by_seq',{}))
                for length in (None,*lengths):
                    def budget(values):
                        return values.get('selective_cache_by_seq',{}).get(length,values.get('selective_cache_mib',CACHE_BYTES/2**20))
                    if budget(current)<budget(cache_reference):
                        raise ValueError('cache budget is below the saved compressed-cache precision reference')
            precision=dict(param_dtype=model.param_dtype,activation_dtype=model.activation_dtype,kv_cache_dtype=model.kv_cache_dtype)
            if meta.get('precision') != precision or meta.get('fp8') != model.e.ops.fp8:
                raise ValueError('precision/FP8 recipe differs from the native resume checkpoint')
            if meta.get('fp8_filter','eligible')!=model.e.fp8_filter:
                raise ValueError('FP8 filter differs from the native resume checkpoint')
        # Validate all names/shapes before mutating any device buffers.
        assignments=[]
        for group,m in models.items():
            for name,p in m.params.items():
                if adapters_only and '.lora.' not in name:continue
                key=f'{group}/{name}'
                if key not in z:
                    if '.lora.' in name and not adapters_only:continue
                    if optimizer is None and name.startswith(('model.exit_head.','model.loop_gate.')):continue
                    raise ValueError('missing checkpoint parameter: '+key)
                if tuple(z[key].shape)!=p.shape:raise ValueError('checkpoint shape mismatch: '+key)
                assignments.append((p,key))
        for group,opt in optimizers.items():
            if opt is None:continue
            info=meta.get('optimizers',{}).get(group)
            if not info or info['kind']!=opt.kind:raise ValueError('optimizer type/state does not match resume checkpoint')
            if (info.get('muon_mantissa',False)!=opt.muon_mantissa or
                    info.get('muon_momentum_dtype','fp32')!=opt.muon_momentum_dtype):
                raise ValueError('Muon mantissa/momentum precision differs from resume checkpoint')
            for p,state in opt.state.items():
                for name,value in state.items():
                    if name=='step':continue
                    key=f'optimizer/{group}/{p.name}/{name}'
                    if key not in z or tuple(z[key].shape)!=value.shape:raise ValueError('missing or incompatible optimizer state: '+key)
                    if value.dtype=='u16' and z[key].dtype!=np.uint16:
                        raise ValueError('mantissa must contain raw uint16 state: '+key)
                    assignments.append((value,key))
                assignments.append((opt.gradients[p],f'gradient/{group}/{p.name}'))
        for target,key in assignments:
            if key not in z or tuple(z[key].shape)!=target.shape:
                raise ValueError('missing or incompatible checkpoint tensor: '+key)
        for target,key in assignments:
            mark=e.backend.mark()
            with e.no_grad():
                source=e.wrap(e.backend.from_numpy(z[key])) if target.dtype=='u16' else e.array(z[key])
                e.copy(target,source)
            e.backend.release_since(mark)
        for group,opt in optimizers.items():
            if opt is None:continue
            info=meta['optimizers'][group]
            for name in opt.metadata():
                if name in info:setattr(opt,name,info[name])
            opt.present={p for p in opt.parameters if p.name in info['present']}
            for p,state in opt.state.items():state['step']=info['steps'][p.name]
        if optimizer is not None:
            e.seed,e.counter=meta['engine_rng']['seed'],meta['engine_rng']['counter']
            if rng is not None and meta.get('rng') is not None:rng.bit_generator.state=meta['rng']
            if cache_reference is not None:model.cache_precision_reference=cache_reference
            elif hasattr(model,'cache_precision_reference'):del model.cache_precision_reference
        if cache_change:
            print('Native ckpt4 cache budget changed on full resume: '+json.dumps(cache_change,sort_keys=True)
                  +'; weights, optimizer, RNG and data progress restored unchanged; compressed-cache precision reference='
                  +json.dumps(cache_reference,sort_keys=True)+'.',flush=True)
        return meta.get('progress',{})
