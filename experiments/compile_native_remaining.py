"""Compile grouped FP8 MoE and complete-step optimizer kernels without a GPU."""
import argparse
import json
import sys
from experiments.smoke_dense_triton import RejectTorch


def run(arch):
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_triton.runtime import compile_kernel
    from model.dense_cublas import grouped as g,graph as s
    results=[]
    def compile(fn,types,**constants):
        k=compile_kernel(fn,types,constants,arch);m=k.metadata
        if m.global_scratch_size or m.num_ctas!=1 or m.shared>99*1024:raise RuntimeError('unsupported direct launch ABI or shared memory')
        results.append(dict(kernel=fn.__name__,shared=m.shared))
    compile(g.route_counts,['fp32','i32'],T=513,E=4)
    compile(g.route_offsets,['i32']*3,TILES=3,E=4,B=16)
    compile(g.route_indices,['fp32','i32','i64','i64'],T=513,E=4)
    for dtype in ('fp32','bf16','fp16','fp8e4nv'):
        fp8=dtype.startswith('fp8');scale='fp32' if fp8 else dtype
        for transpose in (False,True):
            for gradient in (False,True):
                a='fp8e5' if fp8 and gradient else dtype
                compile(g.grouped_mm,[a,dtype,'fp32','i32',scale,scale],N=96,K=160,TRANSPOSE_W=transpose,FP8=fp8)
        compile(g.grouped_dw,[dtype,'fp8e5' if fp8 else dtype,'fp32','i32',scale,scale],N=96,K=160,FP8=fp8)
    for dtype in ('bf16','fp16'):
        compile(g.grouped_mm,['fp8e4nv','fp8e4nv',dtype,'i32','fp32','fp32'],N=96,K=160,FP8=True)
        compile(g.grouped_mm,['fp8e5','fp8e4nv',dtype,'i32','fp32','fp32'],N=160,K=96,TRANSPOSE_W=True,FP8=True)
        compile(g.grouped_dw,['fp8e4nv','fp8e5',dtype,'i32','fp32','fp32'],N=96,K=160,FP8=True)
    for dtype in ('fp32','bf16','fp16'):
        for weights in (False,True):
            for gradient in (False,True):
                compile(g.expert_scale,[dtype,'i32','fp32'],WIDTH=96,ROWS=160,WEIGHTS=weights,LIMIT=57344. if gradient else 448.)
                compile(g.expert_quantize,[dtype,'fp8e5' if gradient else 'fp8e4nv','i32','fp32'],WIDTH=96,ROWS=160,WEIGHTS=weights)
        compile(s.adamw,[dtype,'fp32','fp32','fp32','i64','fp32','fp32'],INDEX=0,N=513,Decay=.01,B1=.9,B2=.999,Eps=1e-8)
    compile(s.accumulate,['fp32']*3,N=513)
    compile(s.statistics,['fp32','fp32','i64','fp32'],Clip=1.,Accumulation=8,LossScaling=True,Growth=2.,Backoff=.5,Interval=2000)
    compile(s.advance_one,['i64','fp32'],INDEX=0)
    assert 'torch' not in sys.modules
    return dict(arch=arch,specializations=len(results),max_shared_bytes=max(x['shared'] for x in results),torch_imported=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arch',type=int,default=89)
    print(json.dumps(run(p.parse_args().arch)))
