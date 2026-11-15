"""Bounded runtime races from a larger CPU-only cuBLAS heuristic catalog."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas.tuning import race,gemm_shortlist
    report=dict(status='running',cases=[],scope='GEMM only, <=4 timed candidates per CPU heuristic catalog')
    with TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=True)) as b:
        enable_fused_ops(b);ops=CublasDenseOps(b);rng=np.random.default_rng(395)
        for m,n,k,grad in ((5504,12288,1536,False),(1536,12288,5504,True),(1536,5504,12288,True),(5504,1536,12288,True)):
            mark=b.mark();operands=[];scales=[]
            for shape,dtype in (((m,k),'fp8e4nv'),((n,k),'fp8e5' if grad else 'fp8e4nv')):
                x=b.from_numpy(rng.standard_normal(shape,dtype=np.float32));q,s=ops.quantize(x,dtype)
                operands.append(q);scales.append(ops._inverse(s))
            output=b.empty((n,m),'bf16');workspace=b.workspaces[b.main_stream];size=b.config.workspace_mib*2**20
            pointers=(*[x.storage for x in operands],output.storage,*[s.storage for s in scales],workspace.storage,size,b.main_stream)
            reference=None
            for count in (8,64):
                plan,candidates=b.native.create_plan(m,n,k,operands[0].dtype,operands[1].dtype,'bf16',size,count,0)
                def measure(candidate,repeats):
                    return b.time_call(lambda:b.native.run(plan,candidate['index'],*pointers),b.main_stream,repeats)
                shortlist=gemm_shortlist(candidates)
                winner=race(shortlist,measure,warmup_ms=100,drift_limit=.05,retries=1)
                b.native.run(plan,winner['index'],*pointers);value=b.to_numpy(output)
                if reference is None:reference=value
                times=[measure(winner,10) for _ in range(5)]
                row=dict(shape_mnk=[m,n,k],types=[x.dtype for x in operands],catalog_limit=count,catalog_count=len(candidates),
                    selected=winner['index'],median_ms=float(np.median(times)),samples_ms=times,
                    bitwise_equal=bool(np.array_equal(value,reference)),relative_l2=float(np.linalg.norm(value-reference)/max(np.linalg.norm(reference),1e-20)),
                    candidates=candidates)
                report['cases'].append(row);args.output.write_text(json.dumps(report,indent=2))
                print(json.dumps({key:value for key,value in row.items() if key!='candidates'}),flush=True)
            b.release_since(mark)
    report.update(status='complete',torch_imported='torch' in sys.modules);args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
