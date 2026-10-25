"""Atomic, pickle-free native weights and complete training checkpoints."""
import json
import os
from pathlib import Path
import tempfile
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
        fp8=model.e.ops.fp8,optimizers={})
    for group,m in models.items():
        for name,p in m.params.items():
            if not adapters_only or '.lora.' in name:arrays[f'{group}/{name}']=m.e.numpy(p).astype(np.float32)
    for group,opt in optimizers.items():
        if opt is None:continue
        info=opt.metadata();info['steps']={};info['present']=[p.name for p in opt.present]
        for p,state in opt.state.items():
            info['steps'][p.name]=state['step']
            for name,value in state.items():
                if name!='step':arrays[f'optimizer/{group}/{p.name}/{name}']=opt.e.numpy(value).astype(np.float32)
            arrays[f'gradient/{group}/{p.name}']=opt.e.numpy(opt.gradients[p]).astype(np.float32)
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
    def __exit__(self,*exc):
        try:self.flush()
        finally:
            if self.executor:self.executor.shutdown(wait=True)


def metadata(path):
    with np.load(path,allow_pickle=False) as z:return json.loads(str(z['__metadata__'])) if '__metadata__' in z else {}


def load(path,model,optimizer=None,rng=None,extra_models=None,extra_optimizers=None,adapters_only=False):
    if str(path).endswith('.pth'):
        raise ValueError('Convert .pth weights first: python scripts/convert_cublas_checkpoint.py --input PATH --output PATH.npz')
    e=model.e;models={'model':model,**(extra_models or {})};optimizers={'model':optimizer,**(extra_optimizers or {})}
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['__metadata__'])) if '__metadata__' in z else {}
        if optimizer is not None:
            if meta.get('config') != model.config:
                raise ValueError('model configuration differs from the native resume checkpoint')
            precision=dict(param_dtype=model.param_dtype,activation_dtype=model.activation_dtype,kv_cache_dtype=model.kv_cache_dtype)
            if meta.get('precision') != precision or meta.get('fp8') != model.e.ops.fp8:
                raise ValueError('precision/FP8 recipe differs from the native resume checkpoint')
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
            for p,state in opt.state.items():
                for name,value in state.items():
                    if name=='step':continue
                    key=f'optimizer/{group}/{p.name}/{name}'
                    if key not in z or tuple(z[key].shape)!=value.shape:raise ValueError('missing or incompatible optimizer state: '+key)
                    assignments.append((value,key))
                assignments.append((opt.gradients[p],f'gradient/{group}/{p.name}'))
        for target,key in assignments:
            if key not in z or tuple(z[key].shape)!=target.shape:
                raise ValueError('missing or incompatible checkpoint tensor: '+key)
        for target,key in assignments:
            mark=e.backend.mark()
            with e.no_grad():e.copy(target,e.array(z[key]))
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
        return meta.get('progress',{})
