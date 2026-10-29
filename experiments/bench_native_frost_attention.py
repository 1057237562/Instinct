"""Paired packed backward check, including SM120 staging and scatter costs."""
import argparse
import os
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--samples',default='experiments/results/cublas/comparison_20261027/identical_batches.npz')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);sys.meta_path.insert(0,RejectTorch())
    # Kernel-only source loading has no Python graph-binding device registry.
    # Native plan objects below provide the per-context in-process cache.
    os.environ['CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO']='0'
    os.environ['CUDNN_FRONTEND_DISABLE_COMPILED_CACHE']='1'
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.vendor_attention import NativeAttention
    from model.dense_cublas.policy import enable_fused_ops
    report=dict(status='running',cases=[])
    def write():a.output.write_text(json.dumps(report,indent=2))
    write()
    with TrainingBackend(allocator_backend='async') as b:
        enable_fused_ops(b);e=Engine(b);native=NativeAttention(b);b.native_attention=native
        with np.load(a.samples,allow_pickle=False) as sample:
            for index in (0,1):
                mark=b.mark();segments=sample[f'segments{index}'];shape=segments.shape;rng=np.random.default_rng(1801+index)
                q,k,v,dy=[e.array(rng.standard_normal((*shape,h,96),dtype=np.float32)*.2,'bf16').data for h in (16,4,4,16)]
                native.prepare(shape,segments,16,4,96);mask=b.empty((1,),'i32');mask.native_attention_context=native.context
                _,saved=native.attention(q,k,v,mask,0.,0,0,None,True,None)
                reference=None
                for enabled in (False,True):
                    local=b.mark();b.attention_frost_backward=enabled
                    grads=native.backward(saved,dy,None);values=[b.to_numpy(x) for x in grads]
                    if not enabled:reference=values
                    errors=[float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)) for x,y in zip(values,reference)]
                    def run():
                        step=b.mark();native.backward(saved,dy,None);b.release_since(step)
                    for _ in range(3):run()
                    times=[b.time_call(run,b.main_stream,repeats=3) for _ in range(3)]
                    row=dict(shape=list(shape),frost=enabled,median_ms=float(np.median(times)),relative_l2=errors,within_attention_limit=max(errors)<.0011)
                    report['cases'].append(row);print(json.dumps(row),flush=True);write();b.release_since(local)
                b.release_since(mark)
        report.update(status='complete',torch_imported='torch' in sys.modules,frost_metrics=native.frost.metrics)
        write()


if __name__=='__main__':main()
