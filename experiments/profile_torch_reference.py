"""Read-only Torch comparison on the exact native probe batches; no updates."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--torch-root',type=Path,default=Path('D:/AI/Instinct'))
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-by-seq',default='',help='Optional equal-budget control, e.g. 3072:1024,4096:1280')
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    target=args.output.resolve();samples=args.samples.resolve();root=args.torch_root.resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    os.chdir(root);sys.path.insert(0,str(root))
    os.environ.update(INSTINCT_MUON_MANTISSA='1',INSTINCT_MUON_MOMENTUM_DTYPE='bf16',
        INSTINCT_MUON_BATCH_SIZE='16',INSTINCT_MUON_WORKSPACE_MB='512',INSTINCT_V2_FLEX_TMA='1',
        INSTINCT_V2_GEMM_BACKEND='auto')
    import trainer.compile_cache
    import datasets
    import torch
    from experiments.profile_checkpoint_policy import reserve_training_state
    from model.model_instinct import InstinctConfig,InstinctForCausalLM
    import model.checkpointing as checkpointing
    from trainer.trainer_utils import cast_training_parameters,apply_torchao_fp8_training
    from model.compile_policy import training_compile_kwargs
    torch.set_num_threads(1);torch.manual_seed(307)
    torch.cuda.set_per_process_memory_fraction(min(1.,14.37*2**30/torch.cuda.get_device_properties(0).total_memory))
    config=InstinctConfig(**(json.loads((root/'trainer/config_instinct_v2.json').read_text())|dict(use_grad_checkpoint=4,num_hidden_layers=32)))
    raw=cast_training_parameters(InstinctForCausalLM(config),'bf16').cuda().train()
    optimizer=reserve_training_state(raw)
    apply_torchao_fp8_training(raw,SimpleNamespace(fp8_training='tensorwise',fp8_filter='auto',dtype='bfloat16',device='cuda',use_compile=1))
    compile_options=training_compile_kwargs(raw,mode='max-autotune-no-cudagraphs')
    model=torch.compile(raw,**compile_options,dynamic=False)
    budgets={int(k):int(v) for k,v in (item.split(':') for item in args.cache_by_seq.split(','))} if args.cache_by_seq else {}
    default_budget=checkpointing.DENSE_RECIPE_CACHE_BYTES
    batches=[];counts=[]
    with np.load(samples,allow_pickle=False) as z:
        for i in range(2):
            arrays=[z[f'{key}{i}'] for key in ('ids','labels','segments')]
            batches.append(tuple(torch.from_numpy(x.copy()).cuda() for x in arrays))
            counts.append(int(np.count_nonzero(arrays[1][:,1:]!=-100)))
    report=dict(status='running',scope='fixed weights; exact shared production batches; resident optimizer/gradients; no updates or loader',
        samples=str(samples),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        model_config=config.to_dict(),compile_options=compile_options,cache_mib_by_seq=budgets or {'default':default_budget/2**20},
        useful_targets=counts,shapes=[list(x[0].shape) for x in batches],bucket_ms=[[],[]],trace_files=[])
    def write():target.write_text(json.dumps(report,indent=2,default=str))
    write()
    def step(index):
        ids,labels,segments=batches[index]
        checkpointing.DENSE_RECIPE_CACHE_BYTES=budgets.get(ids.shape[1],default_budget//2**20)*2**20
        model.zero_grad(set_to_none=False)
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(ids,labels=labels,sequence_ids=segments);loss=out.loss
        del out
        loss.backward()
        return loss.detach()
    try:
        for _ in range(args.warmup):
            for i in range(2):
                loss=step(i);torch.cuda.synchronize();print(f'warmup bucket={i} loss={float(loss):.6f}',flush=True)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.steps):
            for i in range(2):
                start=time.perf_counter();step(i);torch.cuda.synchronize()
                report['bucket_ms'][i].append((time.perf_counter()-start)*1000)
        report.update(median_ms=[statistics.median(x) for x in report['bucket_ms']],allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        write();print(json.dumps({k:report[k] for k in ('median_ms','allocated_gib','reserved_gib')}),flush=True)
        if args.profile:
            for i in range(2):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    step(i);torch.cuda.synchronize()
                trace=target.with_suffix(f'.bucket{i}.trace.json');prof.export_chrome_trace(str(trace))
                report['trace_files'].append(str(trace))
        report['status']='complete';write()
    except BaseException as error:
        report.update(status='failed',error=repr(error));write();raise


if __name__=='__main__':main()
