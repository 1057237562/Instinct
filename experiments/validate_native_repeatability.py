"""Same-context full-model repeats with global and per-parameter error norms.

Fixed weights only. This distinguishes baseline variability from the measured
candidate delta without averaging away a failing parameter or changing limits.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--candidate-config');p.add_argument('--fp8',type=int,choices=(0,1),default=1)
    p.add_argument('--layers',type=int,default=32);p.add_argument('--bucket',type=int,default=0)
    p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas.vendor_attention import NativeAttention
    from experiments.native_candidate import apply
    cfg=json.loads(Path('trainer/config_instinct_v2.json').read_text())
    cfg.update(num_hidden_layers=a.layers,use_grad_checkpoint=4,selective_cache_mib=1792)
    with np.load('experiments/results/cublas/comparison_20261027/identical_batches.npz',allow_pickle=False) as sample:
        ids,labels,segments=[sample[f'{key}{a.bucket}'] for key in ('ids','labels','segments')]
    report=dict(status='running',args=vars(a)|{'output':str(a.output)},shape=list(ids.shape),cases=[])
    def write():a.output.write_text(json.dumps(report,indent=2))
    write()
    with TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=False)) as b:
        enable_fused_ops(b);b.native_attention=NativeAttention(b)
        # Freeze the pre-round-2 baseline even after shipped defaults advance.
        b.fused_activation_layouts=False;b.fused_boundary_norm=False;b.release_checkpoint_cache=False
        b.attention_cost_model=False;b.muon_blocked_gram=False;b.muon_blocked_polynomial=False
        e=Engine(b,fp8=bool(a.fp8),fp8_filter='auto');m=Model(e,cfg,param_dtype='bf16',muon_mantissa=True,seed=307)
        reference=None;reference_loss=None
        for run in range(1+a.repeats+bool(a.candidate_config)):
            candidate=run>a.repeats
            if candidate:report['candidate']=apply(b,a.candidate_config)
            mark=b.mark();out=m(ids,labels,sequence_ids=segments);loss=e.item(out.loss)
            grads=e.vjp(out.loss);values={};errors={};delta2=norm2=dot=actual2=0.
            for param,grad in grads.items():
                actual=e.numpy(grad)
                if reference is None:values[param.name]=actual;continue
                expected=reference[param.name];delta=actual-expected
                # FP64 accumulation on CPU; no extra GPU buffers or timing claims.
                n=float(np.einsum('i,i->',expected.ravel(),expected.ravel(),dtype=np.float64))
                d=float(np.einsum('i,i->',delta.ravel(),delta.ravel(),dtype=np.float64))
                delta2+=d;norm2+=n
                dot+=float(np.einsum('i,i->',actual.ravel(),expected.ravel(),dtype=np.float64))
                actual2+=float(np.einsum('i,i->',actual.ravel(),actual.ravel(),dtype=np.float64))
                errors[param.name]=dict(relative_l2=float(np.sqrt(d/max(n,1e-40))),reference_l2=float(np.sqrt(n)),max_abs=float(np.max(np.abs(delta))))
            if reference is None:reference=values;reference_loss=loss
            else:
                row=dict(run=run,candidate=candidate,loss=loss,loss_difference=loss-reference_loss,
                    global_relative_l2=float(np.sqrt(delta2/max(norm2,1e-40))),cosine=dot/max(np.sqrt(norm2*actual2),1e-40),
                    worst_relative_l2=max(v['relative_l2'] for v in errors.values()),parameters=errors)
                report['cases'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='parameters'}),flush=True);write()
            b.release_since(mark)
    report.update(status='complete',torch_imported='torch' in sys.modules);write()


if __name__=='__main__':main()
