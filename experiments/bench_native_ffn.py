"""Fixed-input FFN layout/launch screening; no model or optimizer updates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import triton
import triton.language as tl
from experiments.smoke_dense_triton import RejectTorch


@triton.jit
def compare(A,B,O,N:tl.constexpr,BLOCK:tl.constexpr=1024):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(A+i,i<N,0.).to(tl.float32);b=tl.load(B+i,i<N,0.).to(tl.float32)
    tl.store(O+tl.program_id(0),tl.sum((a!=b).to(tl.int32),0))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_cublas.ffn import quantize_swiglu
    from model.dense_cublas.policy import enable_fused_ops
    report=dict(status='running',rows=12288,cols=5504,cases=[])
    with TrainingBackend(allocator_backend='async') as b:
        enable_fused_ops(b)
        ops=CublasDenseOps(b);b.fused_scale_inverse=True;b.ffn_flat_forward=True
        rng=np.random.default_rng(169)
        inputs=[]
        for _ in range(3):
            mark=b.mark()
            x=ops.cast(b.from_numpy(rng.standard_normal((12288,5504),dtype=np.float32)),'bf16')
            b.release_since(mark,keep=(x,));inputs.append(x)
        configs=[(64,64,4,1024,False)]
        configs += [(r,c,w,block,major) for r,c in ((32,64),(32,128),(64,64),(64,128),(128,64))
                    for w in (4,8) for block in (2048,) for major in (False,True)]
        configs += [(64,64,w,block,False) for w in (4,8) for block in (512,1024,4096)]
        for backward in (False,True):
            reference=None
            for row,col,warps,block,major in configs:
                mark=b.mark();b.ffn_quant_tile=row;b.ffn_quant_col_tile=col
                b.ffn_col_major_grid=major;b.ffn_statistics_block=block
                b.kernel_options['model.dense_cublas.ffn.quantized_values']={'num_warps':warps}
                b.kernel_options['model.dense_cublas.ffn.statistics']={'num_warps':warps}
                b._commands=[]
                try:out=quantize_swiglu(ops,*inputs[:2],inputs[2] if backward else None)
                finally:commands=b._commands;b._commands=None
                def run():
                    for command in commands:command.execute()
                for _ in range(3):run()
                times=[b.time_call(run,b.main_stream,repeats=5) for _ in range(3)]
                tensors=list(out)+([out[2].transpose_data,out[4].transpose_data] if backward else [])
                if reference is None:reference=tensors
                mismatches=0
                for x,y in zip(tensors,reference):
                    counts=b.empty((triton.cdiv(x.numel,1024),),'i32')
                    b.launch(compare,(counts.numel,),(x,y,counts),N=x.numel)
                    mismatches+=int(b.to_numpy(counts).sum())
                assert mismatches==0,(backward,row,col,warps,block,major,mismatches)
                item=dict(backward=backward,row_tile=row,col_tile=col,warps=warps,statistics_block=block,
                          col_major=major,median_ms=float(np.median(times)),mismatches=mismatches)
                report['cases'].append(item);print(json.dumps(item),flush=True)
                a.output.write_text(json.dumps(report,indent=2))
                b.release_since(mark,keep=reference)
        report.update(status='complete',torch_imported='torch' in sys.modules)
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
