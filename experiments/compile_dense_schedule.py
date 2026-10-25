"""Offline compile Dense fusion and pointer-table optimizer kernels; no GPU/Torch."""
import argparse
import json
import sys
from experiments.smoke_dense_triton import RejectTorch


def run(arch):
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_triton.runtime import compile_kernel
    from model.dense_cublas import fused,multi_tensor,graph,tensor_kernels
    results=[]
    def compile(fn,types,**constants):
        k=compile_kernel(fn,types,constants,arch);m=k.metadata
        if m.global_scratch_size or m.num_ctas!=1 or m.shared>99*1024:
            raise RuntimeError('unsupported direct-launch resources')
        results.append(dict(kernel=fn.__name__,shared_bytes=m.shared,cubin_bytes=len(k.asm['cubin'])))
    for dtype in ('fp32','bf16','fp16'):
        for backward in (False,True):
            compile(fused.rounded_swiglu,[dtype,dtype,dtype,'fp32',dtype,'fp32'],N=4099,BACKWARD=backward)
        for weights in (2,3):
            compile(fused.projection_cast,['fp32']*3+[dtype]*3,N0=2432*768,N1=2432*768,N2=0 if weights==2 else 384*768)
        for backward in (False,True):
            compile(tensor_kernels.cross_entropy_recompute,[dtype,'i64','fp32','fp32','fp32','fp32','fp32'],
                    VOCAB=6400,BLOCK=8192,BACKWARD=backward,SHIFT=True,SEQ=4096)
        for dim in (96,160):
            opts=dict(N=dim,SEQ=4096,HEADS=8,START=0,BLOCK=128 if dim==96 else 256)
            compile(fused.norm_rotary_forward,[dtype,'fp32','fp32','fp32',dtype,'fp32'],EPS=1.e-6,**opts)
            for grad in dict.fromkeys((dtype,'fp32')):
                compile(fused.norm_rotary_backward,[dtype,'fp32','fp32','fp32','fp32',grad,dtype,'fp32'],M=32768,GROUPS=128,**opts)
        for capture in (False,True):
            compile(multi_tensor.adamw_multi,['i64','i64','i64','fp32','fp32']+['scalar:fp32']*8,DTYPE=dtype,GRAPH=capture)
    compile(multi_tensor.norm_multi,['i64','i64','fp32','scalar:fp32'])
    compile(graph.accumulate_norm,['fp32','fp32','fp32','i64','fp32','fp32'],N=1048579,OFFSET=16)
    compile(graph.accumulate_small,['i64','fp32','i64','fp32','fp32'])
    assert 'torch' not in sys.modules
    return dict(arch=arch,specializations=len(results),max_shared_bytes=max(x['shared_bytes'] for x in results),kernels=results,torch_imported=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arch',type=int,default=120)
    print(json.dumps(run(p.parse_args().arch),indent=2))
