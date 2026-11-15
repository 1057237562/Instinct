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
    p.add_argument('--tile-shortlist',action='store_true')
    p.add_argument('--alias-inputs',action='store_true')
    p.add_argument('--refined',action='store_true',help='Follow the winning q32 tile with one wider KV neighbor')
    p.add_argument('--split-docs',action='store_true',help='Use the q32 winner with bounded 4/2/1-document backward groups')
    p.add_argument('--buckets',nargs='+',type=int,default=[0,1])
    p.add_argument('--documentwise',action='store_true',help='Paired cuDNN versus one FROST plan per document')
    p.add_argument('--head96',action='store_true',help='Isolated no-head-padding feasibility probe; no production default changes')
    p.add_argument('--tune',action='store_true',help='Measure the bounded cuDNN plan shortlist before comparing')
    p.add_argument('--head96-wide',action='store_true',help='One KV128 neighbor, feasible after removing the 128-column head padding')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);sys.meta_path.insert(0,RejectTorch())
    if a.head96_wide:a.head96=True
    # Kernel-only source loading has no Python graph-binding device registry.
    # Native plan objects below provide the per-context in-process cache.
    os.environ['CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO']='0'
    os.environ['CUDNN_FRONTEND_DISABLE_COMPILED_CACHE']='1'
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.vendor_attention import NativeAttention
    from model.dense_cublas.policy import enable_fused_ops
    class SplitAttention(NativeAttention):
        def chunks(self,context,backward=False):
            old=self.b.attention_docs_per_plan
            try:
                if backward and self.b.attention_frost_backward:
                    self.b.attention_docs_per_plan=getattr(self.b,'frost_documents',old)
                return tuple(super().chunks(context,backward))
            finally:self.b.attention_docs_per_plan=old
    report=dict(status='running',cases=[])
    def write():a.output.write_text(json.dumps(report,indent=2))
    write()
    with TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=a.tune)) as b:
        enable_fused_ops(b);e=Engine(b);b.attention_frost_backward=False
        native=SplitAttention(b) if a.split_docs else NativeAttention(b);b.native_attention=native
        with np.load(a.samples,allow_pickle=False) as sample:
            for index in a.buckets:
                mark=b.mark();segments=sample[f'segments{index}'];shape=segments.shape;rng=np.random.default_rng(1801+index)
                q,k,v,dy=[e.array(rng.standard_normal((*shape,h,96),dtype=np.float32)*.2,'bf16').data for h in (16,4,4,16)]
                native.prepare(shape,segments,16,4,96);mask=b.empty((1,),'i32');mask.native_attention_context=native.context
                _,saved=native.attention(q,k,v,mask,0.,0,0,None,True,None)
                reference=None;frost_reference=None
                trials=((False,0,0),(True,0,0))
                if a.documentwise:trials=((False,0,0),(True,32,64))
                if a.tile_shortlist:trials=((False,0,0),(True,64,64),(True,32,64),(True,64,128))
                if a.alias_inputs:trials=((False,0,0),(True,32,64),(True,32,64))
                if a.refined:trials=((False,0,0),(True,32,64),(True,32,128))
                if a.split_docs:trials=((False,0,0),(True,32,64),(True,32,64),(True,32,64))
                if a.head96:trials=((False,0,0),(True,32,64),(True,64,64))
                if a.head96_wide:trials=((False,0,0),(True,64,64),(True,64,128))
                for trial,(enabled,q_tile,kv_tile) in enumerate(trials):
                    native96=a.head96 and (trial==2 or a.head96_wide and trial==1)
                    b.frost_alias_inputs=a.documentwise or a.alias_inputs and (a.refined or a.split_docs or trial==2)
                    b.attention_frost_documentwise=a.documentwise
                    if a.documentwise:
                        b.attention_frost_min_seq=0;b.attention_frost_min_plan_seq=0;b.attention_frost_max_docs=128
                    b.frost_documents=(4,4,2,1)[trial] if a.split_docs else 4
                    local=b.mark();b.attention_frost_backward=enabled
                    if enabled:
                        b.frost_q_tile=q_tile;b.frost_kv_tile=kv_tile;native.frost=None
                        if native96:
                            from model.dense_cublas.frost_attention import FrostBackward
                            from experiments.native_frost_head96 import specialize
                            native.frost=FrostBackward(b)
                            native.frost.template=specialize(native.frost.template)
                    print(json.dumps(dict(phase='prepare',shape=list(shape),frost=enabled,q_tile=q_tile,kv_tile=kv_tile)),flush=True)
                    grads=native.backward(saved,dy,None);values=[b.to_numpy(x) for x in grads]
                    if not enabled:reference=values
                    if enabled and frost_reference is None:frost_reference=values
                    errors=[float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)) for x,y in zip(values,reference)]
                    def run():
                        step=b.mark();native.backward(saved,dy,None);b.release_since(step)
                    for _ in range(3):run()
                    b.time_call(run,b.main_stream,repeats=100)
                    times=[b.time_call(run,b.main_stream,repeats=5) for _ in range(5)]
                    repeated_mark=b.mark();again=native.backward(saved,dy,None)
                    repeat_errors=[float(np.linalg.norm(b.to_numpy(x)-y)/max(np.linalg.norm(y),1e-20)) for x,y in zip(again,values)]
                    assert max(repeat_errors)<.0011,('repeated scratch initialization',repeat_errors)
                    b.release_since(repeated_mark)
                    row=dict(bucket=index,shape=list(shape),documents=native.context[0],frost=enabled,native_head96=native96,q_tile=q_tile,kv_tile=kv_tile,backward_documents=b.frost_documents,alias_inputs=b.frost_alias_inputs,
                        aliased_calls=0 if not enabled else native.frost.metrics.get('aliased_calls',0),median_ms=float(np.median(times)),samples_ms=times,
                        relative_l2=errors,repeat_relative_l2=repeat_errors,within_attention_limit=max(errors)<.0011)
                    if enabled:
                        row['relative_l2_to_frost']=[float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)) for x,y in zip(values,frost_reference)]
                        if row['native_head96']:assert max(row['relative_l2_to_frost'])<.0011,row
                    report['cases'].append(row);print(json.dumps(row),flush=True);write();b.release_since(local)
                b.release_since(mark)
        report.update(status='complete',torch_imported='torch' in sys.modules,frost_metrics=native.frost.metrics)
        write()


if __name__=='__main__':main()
