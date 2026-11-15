"""Compare dW + BF16 accumulation copy with a cuBLASLt beta=1 epilogue."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import triton
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas.mxfp8 import quantize
    from model.dense_cublas.tuning import race,gemm_shortlist
    from model.dense_cublas import tensor_kernels as tk
    from model.dense_triton.runtime import Scalar
    report=dict(status='running',cases=[],scope='Synthetic dW operands, eight accumulation calls; no model/optimizer updates')
    with TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=True)) as b:
        enable_fused_ops(b);ops=CublasDenseOps(b);rng=np.random.default_rng(173)
        for m,n,k in ((1536,5504,12288),(5504,1536,12288),(1536,1536,12288),(1536,384,12288)):
            mark=b.mark();operands=[];scales=[]
            for shape in ((m,k),(n,k)):
                tmp=b.mark();x=ops.cast(b.from_numpy(rng.standard_normal(shape,dtype=np.float32)*.1),'bf16')
                q,s=quantize(ops,x,'fp8e4nv');b.release_since(tmp,keep=(q,s));operands.append(q);scales.append(s)
            out=b.empty((n,m),'bf16');grad=b.empty(out.shape,'bf16');size=b.config.workspace_mib*2**20;work=b.workspaces[b.main_stream]
            plan,candidates=b.native.create_plan(m,n,k,'fp8e4nv','fp8e4nv','bf16',size,8,0,mxfp8=True,block_scales=tuple(s.storage for s in scales))
            def pointers(target):return (*[x.storage for x in operands],target.storage,*[s.storage for s in scales],work.storage,size,b.main_stream)
            def measure(c,repeats):return b.time_call(lambda:b.native.run(plan,c['index'],*pointers(out)),b.main_stream,repeats)
            winner=race(gemm_shortlist(candidates),measure,warmup_ms=100,drift_limit=.05,retries=1)
            def clear():b.launch(tk.fill,(triton.cdiv(grad.numel,1024),),(grad,Scalar(0.)),N=grad.numel,BLOCK=1024)
            def baseline():
                b.native.run(plan,winner['index'],*pointers(out))
                b.launch(tk.scaled_copy,(triton.cdiv(grad.numel,256),),(out,grad,Scalar(1.)),N=grad.numel,ADD=True)
            def fused():b.native.run_accumulate(plan,winner['index'],*pointers(grad))
            reference=None
            for name,run in (('copy',baseline),('epilogue',fused)):
                clear()
                for _ in range(8):run()
                actual=b.to_numpy(grad)
                if reference is None:reference=actual
                relative=float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-20))
                assert np.isfinite(actual).all() and relative<.01,relative
                clear();b.time_call(run,b.main_stream,repeats=100)
                samples=[]
                for _ in range(5):
                    clear();samples.append(b.time_call(run,b.main_stream,repeats=10))
                row=dict(shape_mnk=[m,n,k],mode=name,median_ms=float(np.median(samples)),samples_ms=samples,
                         accumulated_gradient_relative_l2=relative,bitwise_equal=bool(np.array_equal(actual,reference)))
                report['cases'].append(row);print(json.dumps(row),flush=True);a.output.write_text(json.dumps(report,indent=2))
            b.release_since(mark)
    report.update(status='complete',torch_imported='torch' in sys.modules);a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
