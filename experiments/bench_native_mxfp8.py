"""MXFP8 vs tensorwise FP8: quantization and cuBLAS GEMM separately/together."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--small-kv',action='store_true')
    parser.add_argument('--fast-accum',action='store_true')
    parser.add_argument('--gate-up',action='store_true',help='Screen joined gate/up GEMM dimensions before implementing layouts')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas import mxfp8
    from model.dense_cublas.tuning import race,gemm_shortlist
    report=dict(status='running',cases=[],scope='Fixed contiguous BF16 operands; no model/updates; full chain includes fresh scales for both operands')
    with TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=True)) as b:
        enable_fused_ops(b);ops=CublasDenseOps(b);rng=np.random.default_rng(395)
        shapes=((5504,12288,1536,False),(1536,12288,5504,True),(1536,5504,12288,True),(5504,1536,12288,True))
        if args.small_kv:shapes=((384,12288,1536,False),(1536,12288,384,True),(1536,384,12288,True))
        if args.gate_up:shapes=((5504,12288,1536,False),(11008,12288,1536,False),(1536,12288,5504,True),(1536,12288,11008,True))
        for m,n,k,grad in shapes:
            mark=b.mark();inputs=[]
            for shape in ((m,k),(n,k)):
                local=b.mark();x=ops.cast(b.from_numpy(rng.standard_normal(shape,dtype=np.float32)),'bf16')
                b.release_since(local,keep=(x,));inputs.append(x)
            reference=None
            modes=('bf16','mx','mx_fast') if args.small_kv else (('mx','mx_fast') if args.fast_accum else ('tensorwise','mx'))
            if args.gate_up:modes=('mx',)
            for mode in modes:
                enabled=mode.startswith('mx')
                local=b.mark();b._commands=[]
                try:
                    operands=[];scales=[]
                    for x,dtype in zip(inputs,('fp8e4nv','fp8e5' if grad else 'fp8e4nv')):
                        if mode=='bf16':operands.append(x);scales.append(None);continue
                        q,s=mxfp8.quantize(ops,x,dtype) if enabled else ops.quantize(x,dtype)
                        operands.append(q);scales.append(s if enabled else ops._inverse(s))
                finally:commands=b._commands;b._commands=None
                def quantize():
                    for command in commands:command.execute()
                quantize();out=b.empty((n,m),'bf16');size=b.config.workspace_mib*2**20;work=b.workspaces[b.main_stream]
                pointers=(*[x.storage for x in operands],out.storage,*[None if s is None else s.storage for s in scales],work.storage,size,b.main_stream)
                plan,candidates=b.native.create_plan(m,n,k,operands[0].dtype,operands[1].dtype,'bf16',size,8,0,
                    mxfp8=enabled,block_scales=tuple(s.storage for s in scales) if enabled else None,fast_accum=mode=='mx_fast')
                def measure(candidate,repeats):
                    return b.time_call(lambda:b.native.run(plan,candidate['index'],*pointers),b.main_stream,repeats)
                winner=race(gemm_shortlist(candidates),measure,warmup_ms=100,drift_limit=.05,retries=1)
                def gemm():b.native.run(plan,winner['index'],*pointers)
                def combined():quantize();gemm()
                gemm();value=b.to_numpy(out)
                if reference is None:reference=value
                timings={}
                for name,fn in (('quantization',quantize),('gemm',gemm),('combined',combined)):
                    b.time_call(fn,b.main_stream,repeats=30)
                    times=[b.time_call(fn,b.main_stream,repeats=10) for _ in range(5)]
                    timings[name]=dict(median_ms=float(np.median(times)),samples_ms=times)
                error=float(np.linalg.norm(value-reference)/max(np.linalg.norm(reference),1e-20))
                row=dict(shape_mnk=[m,n,k],mxfp8=enabled,mode=mode,reference_mode=modes[0],selected=winner['index'],timings=timings,
                    relative_l2_to_reference=error,bitwise_equal=bool(np.array_equal(value,reference)),
                    scale_bytes=sum(s.numel*(2 if enabled else 4) for s in scales if s is not None),candidates=candidates)
                report['cases'].append(row);args.output.write_text(json.dumps(report,indent=2))
                print(json.dumps({k:v for k,v in row.items() if k!='candidates'}),flush=True);b.release_since(local)
            b.release_since(mark)
    report.update(status='complete',torch_imported='torch' in sys.modules);args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
