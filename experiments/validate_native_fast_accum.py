"""Isolate cuBLAS forward fast accumulation from every other policy change."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--production-shapes',action='store_true')
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--sample-file')
    p.add_argument('--gated',action='store_true')
    p.add_argument('--repeat-control',action='store_true')
    p.add_argument('--no-tune',action='store_true')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas.vendor_attention import NativeAttention
    report=dict(status='running',cases=[])
    cfg=json.loads((Path(__file__).resolve().parents[1]/'trainer/config_instinct_v2.json').read_text())
    cfg.update(num_hidden_layers=a.layers,use_grad_checkpoint=4,selective_cache_mib=1280)
    cases=((307,4,3072),(817,3,4096)) if a.production_shapes else ((307,1,137),(817,1,1024),(127,1,3072))
    for index,(seed,batch,seq) in enumerate(cases):
        reference=None
        for enabled in ((False,False,True) if a.repeat_control else (False,True)):
            with TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=not a.no_tune,fp8_fast_accum_output=enabled and not a.gated)) as b:
                b.fp8_fast_accum_large=enabled and a.gated
                enable_fused_ops(b);b.native_attention=NativeAttention(b)
                e=Engine(b,fp8=True,fp8_filter='auto');m=Model(e,cfg,param_dtype='bf16',muon_mantissa=True,seed=seed)
                ids=np.random.default_rng(seed).integers(3,6400,(batch,seq),dtype=np.int64)
                segments=np.broadcast_to((np.arange(seq)*3//seq)[None],ids.shape).copy()
                labels=ids
                if a.sample_file:
                    with np.load(a.sample_file,allow_pickle=False) as saved:
                        ids,labels,segments=[saved[f'{key}{index}'] for key in ('ids','labels','segments')]
                out=m(ids,labels,sequence_ids=segments);loss=e.item(out.loss)
                gradients={p.name:e.numpy(g) for p,g in e.vjp(out.loss).items()}
                if reference is None:reference=loss,gradients;continue
                errors={k:float(np.linalg.norm(v-reference[1][k])/max(np.linalg.norm(reference[1][k]),1e-20)) for k,v in gradients.items()}
                row=dict(seed=seed,batch=batch,seq=seq,layers=a.layers,gated=a.gated,enabled=enabled,tune=not a.no_tune,loss=loss,loss_difference=loss-reference[0],worst_relative_l2=max(errors.values()),
                         gradient_relative_l2=errors,within_1e_3=max(errors.values())<.001)
                report['cases'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='gradient_relative_l2'}),flush=True)
                a.output.write_text(json.dumps(report,indent=2))
    report.update(status='complete',passed=all(x['within_1e_3'] and abs(x['loss_difference'])<1e-6 for x in report['cases']),torch_imported='torch' in sys.modules)
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
