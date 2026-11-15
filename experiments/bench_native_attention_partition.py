"""Three backward grouping candidates with unchanged forward and Q/K/V/dO."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--samples',default='experiments/results/cublas/comparison_20261027/identical_batches.npz')
    parser.add_argument('--paired',action='store_true',help='Alternate timings of fixed-four, fixed-two and separately fitted backward grouping')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas.vendor_attention import NativeAttention
    class PartitionAttention(NativeAttention):
        backward_documents=4
        def chunks(self,context,backward=False):
            old=self.b.attention_docs_per_plan
            try:
                if backward and self.backward_documents:self.b.attention_docs_per_plan=self.backward_documents
                return tuple(super().chunks(context,backward))
            finally:self.b.attention_docs_per_plan=old
    report=dict(status='running',scope='fixed-input attention only; no loader or updates',cases=[])
    with np.load(args.samples) as samples, TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=True)) as b:
        enable_fused_ops(b);b.attention_cost_model=False
        attention=PartitionAttention(b);b.native_attention=attention;e=Engine(b);rng=np.random.default_rng(821)
        if args.paired:
            b.attention_cost_model='separate';attention.warm_catalog(12288,4096,16,4,96)
        for index in range(2):
            segments=samples[f'segments{index}'];batch,seq=segments.shape;mark=b.mark()
            q,k,v,dy=[e.array(rng.standard_normal((batch,seq,heads,96),dtype=np.float32),'bf16').data for heads in (16,4,4,16)]
            reference=None;jobs=[]
            for docs in ((4,2,0) if args.paired else (4,4,2,1)):
                if args.paired:b.attention_cost_model='separate' if docs==0 else 'forward'
                attention.backward_documents=docs;attention.prepare((batch,seq),segments,16,4,96)
                local=b.mark();b._commands=[]
                try:
                    out,saved=attention.attention(q,k,v,None,0.,0,0,None,True,None)
                    forward_count=len(b._commands)
                    grads=attention.backward(saved,dy,None)
                finally:commands=b._commands;b._commands=None
                def run():
                    for command in commands:command.execute()
                for _ in range(3):run()
                arrays=[b.to_numpy(t) for t in (out,*grads)]
                if reference is None:reference=arrays
                errors=[float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)) for x,y in zip(arrays,reference)]
                if args.paired:
                    jobs.append(dict(docs=docs,commands=commands,errors=errors,forward_count=forward_count,
                        groups=[dict(docs=c[0],length=c[1]) for c in attention.chunks(attention.context,True)],samples=[]))
                    continue
                b.time_call(run,b.main_stream,repeats=25)
                times=[b.time_call(run,b.main_stream,repeats=10) for _ in range(5)]
                row=dict(shape=[batch,seq],backward_docs=docs,relative_l2=errors,median_ms=float(np.median(times)),samples_ms=times,
                    forward_commands=forward_count,backward_commands=len(commands)-forward_count,
                    backward_groups=[dict(docs=c[0],length=c[1]) for c in attention.chunks(attention.context,True)])
                report['cases'].append(row);args.output.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
                assert max(errors)<.0011,errors
                b.release_since(local)
            if args.paired:
                def execute(job):
                    for command in job['commands']:command.execute()
                b.time_call(lambda:execute(jobs[0]),b.main_stream,repeats=150)
                for repeat in range(5):
                    for job in jobs[::(-1 if repeat%2 else 1)]:
                        job['samples'].append(b.time_call(lambda:execute(job),b.main_stream,repeats=10))
                for job in jobs:
                    row=dict(shape=[batch,seq],backward_docs=job['docs'],relative_l2=job['errors'],
                        median_ms=float(np.median(job['samples'])),samples_ms=job['samples'],backward_groups=job['groups'])
                    report['cases'].append(row);print(json.dumps(row),flush=True)
                    assert max(job['errors'])<.0011,job['errors']
                report['cost_model']=attention.metrics.get('group_cost_model')
                args.output.write_text(json.dumps(report,indent=2))
            b.release_since(mark)
        report.update(status='complete',torch_imported='torch' in sys.modules)
    args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
