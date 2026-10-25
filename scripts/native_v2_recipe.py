"""CPU-only shared native V2 recipe loader and reproducible CLI builder."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import os

ROOT=Path(__file__).resolve().parents[1]


def recipe():
    return json.loads((ROOT/'trainer/recipes/instinct_v2_native.json').read_text(encoding='utf-8'))


def model_config():
    return json.loads((ROOT/recipe()['config_path']).read_text(encoding='utf-8'))


def ui_values():
    cfg=recipe()
    common={'optimizer','param_dtype','use_grad_checkpoint','batch_size','max_seq_len',
            'accumulation_steps','learning_rate','weight_decay','train_type','from_weight'}
    native={'fp8','fp8_filter','muon_mantissa','muon_momentum_dtype','muon_batch_size','muon_workspace_mb',
            'selective_cache_mib','allocator_backend','allocator_cache_mib','packing','packed_attention'}
    result={k:v for k,v in cfg.items() if k in common}
    result.update({'native_'+k:v for k,v in cfg.items() if k in native})
    result.update(runtime_backend='cublas',activation_dtype=cfg['dtype'],use_compile=False,
                  intermediate_override=True,checkpoint_profile='legacy')
    result['native_muon_mantissa']=bool(result['native_muon_mantissa'])
    return result


def command(data_path,extra=()):
    cmd=[sys.executable,'-u',str(ROOT/'trainer/train_cublas.py')]
    for key,value in recipe().items():
        if isinstance(value,bool):
            if value:cmd.append('--'+key)
        else:
            cmd.extend(['--'+key,str(ROOT/value) if key=='config_path' else str(value)])
    cmd.extend(['--data_path',str(data_path)])
    if '--from_resume' not in extra and '--save_weight' not in extra:
        cmd.extend(['--save_weight',f'native_v2_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'])
    return cmd+list(extra)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path',required=True)
    p.add_argument('--run',action='store_true',help='Otherwise print the command only')
    args,extra=p.parse_known_args()
    cmd=command(args.data_path,extra)
    if args.run:raise SystemExit(subprocess.call(cmd,cwd=ROOT))
    print(subprocess.list2cmdline(cmd))
