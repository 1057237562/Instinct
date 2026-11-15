"""BF16 score reconstruction plus native FP8 gradient GEMMs; no updates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--bf16-tiles',action='store_true');p.add_argument('--all-fp8',action='store_true')
    p.add_argument('--pretransposed',action='store_true');a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas.vendor_attention import NativeAttention
    from model.dense_cublas.fp8_attention import backward
    report=dict(status='running',cases=[],scope='Attention backward only, including scratch initialization/reduction; BF16 forward')
    with TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=True)) as b,np.load('experiments/results/cublas/comparison_20261027/identical_batches.npz') as sample:
        enable_fused_ops(b);b.aligned_pointwise_only=False;b.attention_cost_model=False
        e=Engine(b);native=NativeAttention(b);b.native_attention=native;rng=np.random.default_rng(913)
        for bucket in range(2):
            mark=b.mark();segments=sample['segments'+str(bucket)];batch,seq=segments.shape
            q,k,v,dy=[e.array(rng.standard_normal((batch,seq,heads,96),dtype=np.float32),'bf16').data for heads in (16,4,4,16)]
            native.prepare((batch,seq),segments,16,4,96);mask=e.array(segments).data;mask.native_attention_context=native.context
            _,saved=native.attention(q,k,v,mask,0.,0,0,None,True,None)
            ref=None
            trials=(('cudnn',64,8),('bf16_fused',64,8),('fp8',64,8),('fp8',128,8))
            if a.bf16_tiles:trials=(('cudnn',64,8),('bf16_fused',128,8),('bf16_fused',128,16),('bf16_fused',64,4))
            if a.all_fp8:trials=(('cudnn',64,8),('fp8',64,8),('fp8_score',64,8),('fp8_score',128,8))
            if a.pretransposed:trials=(('cudnn',64,8),('fp8_transposed',64,8),('fp8_transposed',128,8),('fp8_transposed',64,4))
            for mode,bn,warps in trials:
                local=b.mark()
                def invoke():return native.backward(saved,dy,None) if mode=='cudnn' else backward(e.ops,saved,dy,fp8=mode.startswith('fp8'),bn=bn,warps=warps,score_fp8=mode=='fp8_score',pretransposed=mode=='fp8_transposed')
                try:result=invoke()
                except RuntimeError as error:
                    if 'cuFuncSetAttribute' not in str(error):raise
                    row=dict(bucket=bucket,mode=mode,bn=bn,warps=warps,unsupported=str(error))
                    report['cases'].append(row);print(json.dumps(row),flush=True);a.output.write_text(json.dumps(report,indent=2))
                    b.release_since(local);continue
                value=[b.to_numpy(t) for t in result]
                if ref is None:ref=value
                errors=[float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)) for x,y in zip(value,ref)]
                assert all(np.isfinite(x).all() for x in value),errors
                if mode=='bf16_fused':assert max(errors)<.01,errors
                def run():
                    inner=b.mark();invoke();b.release_since(inner)
                for _ in range(2):run()
                b.time_call(run,b.main_stream,repeats=20)
                times=[b.time_call(run,b.main_stream,repeats=3) for _ in range(3)]
                row=dict(bucket=bucket,shape=[batch,seq],mode=mode,bn=bn,warps=warps,relative_l2=errors,
                         median_ms=float(np.median(times)),samples_ms=times)
                report['cases'].append(row);print(json.dumps(row),flush=True);a.output.write_text(json.dumps(report,indent=2));b.release_since(local)
            b.release_since(mark)
    report.update(status='complete',torch_imported='torch' in sys.modules);a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
