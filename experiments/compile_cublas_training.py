"""Offline compile the native training primitives, without Torch or a GPU."""
import argparse
import json
import sys
from experiments.smoke_dense_triton import RejectTorch


def run(arch):
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_triton.runtime import compile_kernel
    from model.dense_cublas import tensor_kernels as k
    from model.dense_cublas.recurrent import delta_forward,delta_backward
    from model.dense_cublas.optim import adamw_update
    results=[]
    def compile(fn,types,**constants):
        kernel=compile_kernel(fn,types,constants,arch);meta=kernel.metadata
        if meta.global_scratch_size or meta.num_ctas!=1:raise RuntimeError('unsupported direct-launch ABI')
        results.append(dict(name=fn.__name__,shared=meta.shared,cubin_bytes=len(kernel.asm['cubin'])))
    compile(k.fill,['fp32','scalar:fp32'],N=513)
    for op in ('add','mul','div','max','min','gt','ge','eq'):
        compile(k.binary,['bf16','fp32','fp32'],SHAPE=(2,3,7),AS=(21,7,1),BS=(0,0,1),N=42,OP=op)
        compile(k.binary_scalar,['bf16','bf16','scalar:fp32'],N=513,OP=op)
    for maximum in (False,True):
        compile(k.scalar_extreme_backward,['bf16','fp32','fp32','scalar:fp32'],N=513,MAXIMUM=maximum)
    for device_scale in (False,True):
        compile(k.squared_norm_partials,['fp32','fp32','fp32','scalar:fp32'],N=4099,OFFSET=3,DEVICE_SCALE=device_scale)
    compile(k.sum_partials,['fp32','fp32'],N=193001)
    for backward in (False,True):
        compile(k.cross_entropy_recompute,['bf16','i64','fp32','fp32','fp32','fp32','fp32'],VOCAB=6400,BLOCK=8192,BACKWARD=backward)
    for op in ('neg','exp','log','sqrt','rsqrt','sigmoid','silu','relu','gelu','softplus','abs'):
        compile(k.unary,['bf16','bf16'],N=513,OP=op)
        compile(k.unary_backward,['bf16','bf16','fp32','bf16'],N=513,OP=op)
    for op in ('sum','max'):compile(k.reduce_axis,['fp32','fp32'],M=2,N=513,K=3,OP=op)
    for reverse in (False,True):compile(k.strided_copy,['fp32','fp32'],SHAPE=(2,3,7),STRIDES=(7,14,1),OFFSET=0,N=42,REVERSE=reverse)
    for log in (False,True):
        compile(k.softmax,['bf16','fp32'],N=92544,LOG=log)
        compile(k.softmax_backward,['fp32','fp32','fp32'],N=92544,LOG=log)
    for backward in (False,True):compile(k.gather_last,['fp32','i64','fp32'],ROWS=37,WIDTH=6400,BACKWARD=backward)
    compile(k.dropout,['bf16','bf16','scalar:i32','scalar:i64','scalar:fp32'],N=513)
    for add in (False,True):compile(k.scaled_copy,['fp32','bf16','scalar:fp32'],N=513,ADD=add)
    compile(k.topk_mask,['fp32','fp32'],WIDTH=8,TOPK=2,BLOCK=8)
    compile(k.masked_fill,['fp32','i64','fp32','scalar:fp32'],SHAPE=(3,7),XS=(7,1),MS=(0,1),N=21)
    compile(adamw_update,['bf16','fp32','fp32','fp32']+['scalar:fp32']*8,N=513)
    for dim in (32,128):
        options=dict(T=16,H=8,DK=dim,DV=dim,BK=dim,BV=dim)
        compile(delta_forward,['fp32']*9,**options)
        compile(delta_backward,['fp32']*14,**options)
    assert 'torch' not in sys.modules and 'torchao' not in sys.modules
    return dict(arch=arch,specializations=len(results),kernels=results,torch_imported=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arch',type=int,default=89)
    print(json.dumps(run(parser.parse_args().arch),indent=2))
