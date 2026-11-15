"""Tensorwise two-pass vs MXFP8 one-pass SwiGLU; bounded tile shortlist."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rectangular',action='store_true');a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True);sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas import ffn,mxfp8
    report=dict(status='running',shape=[12288,5504],cases=[],scope='Quantization chain only; MXFP8 changes quantization semantics')
    with TrainingBackend(allocator_backend='async') as b:
        enable_fused_ops(b);ops=CublasDenseOps(b);rng=np.random.default_rng(169);inputs=[]
        for _ in range(3):
            mark=b.mark();x=ops.cast(b.from_numpy(rng.standard_normal((12288,5504),dtype=np.float32)),'bf16')
            b.release_since(mark,keep=(x,));inputs.append(x)
        for backward in (False,True):
            reference=None
            choices=((False,64,64,4),(True,32,32,4),(True,64,64,4),(True,64,64,8))
            if a.rectangular:choices=((True,64,64,8),(True,128,32,4),(True,128,32,8),(True,128,64,8))
            for mx,tile,width,warps in choices:
                local=b.mark();b.mxfp8_ffn_tile=tile
                b.mxfp8_ffn_col_tile=width
                b.kernel_options['model.dense_cublas.mxfp8.mxfp8_swiglu']={'num_warps':warps}
                b._commands=[]
                try:out=(mxfp8 if mx else ffn).quantize_swiglu(ops,*inputs[:2],inputs[2] if backward else None)
                finally:commands=b._commands;b._commands=None
                def run():
                    for c in commands:c.execute()
                for _ in range(3):run()
                if a.rectangular:
                    from experiments.bench_native_ffn import compare
                    import triton
                    tensors=list(out)+([out[2].transpose_data,out[4].transpose_data,
                        out[2].transpose_data.mx32_scale,out[4].transpose_data.mx32_scale] if backward else [])
                    if reference is None:reference=tensors
                    for current,expected in zip(tensors,reference):
                        counts=b.empty((triton.cdiv(current.numel,1024),),'i32')
                        b.launch(compare,(counts.numel,),(current,expected,counts),N=current.numel)
                        assert not b.to_numpy(counts).sum(),(backward,tile,width,warps)
                b.time_call(run,b.main_stream,repeats=50)
                times=[b.time_call(run,b.main_stream,repeats=10) for _ in range(5)]
                row=dict(backward=backward,mxfp8=mx,tile=tile,col_tile=width,warps=warps,median_ms=float(np.median(times)),samples_ms=times)
                report['cases'].append(row);print(json.dumps(row),flush=True);a.output.write_text(json.dumps(report,indent=2));b.release_since(local,keep=reference or ())
            if reference:b.release_owned(reference)
    report.update(status='complete',torch_imported='torch' in sys.modules);a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
