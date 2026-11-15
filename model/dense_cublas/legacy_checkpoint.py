"""Import saved PyTorch training state through an isolated CPU conversion process.

The native training process never imports torch. Names, aliases, tensor shapes
and optimizer layouts are verified before producing a normal native checkpoint.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings


def import_resume(source, destination, model, optimizer, data, args, extra_models=None, extra_optimizers=None):
    models={'model':model,**(extra_models or {})}
    optimizers={'model':optimizer,**(extra_optimizers or {})}
    spec=dict(models={},optimizers={},config=model.config,
        precision=dict(param_dtype=model.param_dtype,activation_dtype=model.activation_dtype,kv_cache_dtype=model.kv_cache_dtype),
        fp8=model.e.ops.fp8,seed=args.seed,train_type=args.train_type,
        batch_size=args.batch_size,accumulation_steps=args.accumulation_steps,epochs=args.epochs,
        source_batch_size=args.legacy_batch_size or args.batch_size,
        source_accumulation_steps=args.legacy_accumulation_steps or args.accumulation_steps,
        source_epochs=args.legacy_epochs or args.epochs,
        learning_rate=args.learning_rate,dataset_size=len(data),dataset_path=str(data.path.resolve()))
    for name,m in models.items():
        spec['models'][name]={k:dict(shape=p.shape,canonical=p.name) for k,p in m.params.items()}
    for name,opt in optimizers.items():
        spec['optimizers'][name]=dict(metadata=opt.metadata(),parameters=[p.name for p in opt.parameters],
            states={p.name:{k:v.shape for k,v in state.items() if k!='step'} for p,state in opt.state.items()})
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='instinct-resume-') as temp:
        manifest=Path(temp)/'manifest.json';manifest.write_text(json.dumps(spec),encoding='utf-8')
        script=Path(__file__).resolve().parents[2]/'scripts/convert_cublas_checkpoint.py'
        subprocess.run([sys.executable,str(script),'--input',str(source),'--output',str(destination),
                        '--resume-spec',str(manifest)],check=True,
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1'))
    from .checkpoint import metadata
    report=metadata(destination)['legacy_import']
    for message in report['warnings']:warnings.warn(message,RuntimeWarning)
    return destination
