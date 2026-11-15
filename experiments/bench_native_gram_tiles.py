"""Two triangular cuBLAS tile candidates against the existing full Gram."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import triton
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--batch',type=int,default=2);p.add_argument('--tiles',type=int,nargs='+',default=[0,512,768]);a=p.parse_args()
    if len(a.tiles)>4:p.error('at most four shortlisted tiles')
    a.output.parent.mkdir(parents=True,exist_ok=True);sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_cublas.bridge import LtError
    from model.dense_cublas.tuning import race
    from model.dense_cublas.muon_gram import tables as pointer_tables,mirror
    report=dict(status='running',cases=[])
    with TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=True)) as b:
        ops=CublasDenseOps(b);rng=np.random.default_rng(729);batch,m,k=a.batch,1536,5504
        x=ops.cast(b.from_numpy((rng.standard_normal((batch,m,k),dtype=np.float32)/np.sqrt(k)).astype(np.float32)),'bf16')
        ref=b.empty((batch,m,m),'bf16');out=b.empty(ref.shape,'bf16');work=b.empty((32*2**20//4,),'fp32')
        def baseline():b.batched_matmul(x,x,x,ref,m,m,k,(m*k,k,1),(m*k,1,k))
        baseline();expected=b.to_numpy(ref)
        for tile in a.tiles:
            if tile:
                nt=m//tile;pairs=nt*(nt+1)//2;count=batch*pairs
                ap,bp,cp=[b.empty((count,),'i64') for _ in range(3)]
                b.launch(pointer_tables,(1,),(x,out,ap,bp,cp,cp),M=m,K=k,TILE=tile,COUNT=count,PAIRS=pairs,BLOCK=triton.next_power_of_2(count))
                try:plan,candidates=b.native.create_pointer_gram(tile,k,m,count,32*2**20)
                except LtError as error:
                    report['cases'].append(dict(tile=tile,unsupported=str(error)));continue
                def execute(index):b.native.run_batched(plan,index,ap.storage,bp.storage,cp.storage,cp.storage,1.,0.,work.storage,32*2**20,b.main_stream)
                def measure(c,repeats):execute(c['index']);return b.time_call(lambda:execute(c['index']),b.main_stream,repeats)
                selected=race(candidates,measure)['index']
            def run():
                if not tile:return baseline()
                b.launch(pointer_tables,(1,),(x,out,ap,bp,cp,cp),M=m,K=k,TILE=tile,COUNT=count,PAIRS=pairs,BLOCK=triton.next_power_of_2(count))
                execute(selected);b.launch(mirror,(triton.cdiv(m,32),triton.cdiv(m,32),batch),(out,),M=m)
            run();actual=b.to_numpy(out if tile else ref)
            error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            times=[b.time_call(run,b.main_stream,repeats=5) for _ in range(3)]
            row=dict(batch=batch,tile=tile,median_ms=float(np.median(times)),relative_l2=error,bitwise_equal=bool(np.array_equal(actual,expected)))
            report['cases'].append(row);print(json.dumps(row),flush=True)
    report.update(status='complete',torch_imported='torch' in sys.modules);a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
