"""Offline compile fused attention/dropout; GPU execution still requires acceptance."""
import argparse
import json
import sys
from experiments.smoke_dense_triton import RejectTorch


def run(arch):
    sys.meta_path.insert(0,RejectTorch())
    from experiments.compile_dense_triton import CompileBackend
    from model.dense_triton.ops import DenseOps
    backend=CompileBackend(arch);ops=DenseOps(backend,fp8=False)
    for dtype in ('fp32','fp16','bf16'):
        for dim in (96,160,192,256):
            for dropout in (0.,.2):
                q=backend.empty((2,35,4,dim),dtype)
                k=backend.empty((2,35,2,dim),dtype)
                mask=backend.empty((2,35),'i64')
                out,saved=ops.attention(q,k,k,mask,dropout=dropout,seed=42,counter=999)
                ops.attention_backward(saved,out)
    for kernel in backend.kernels.values():
        if kernel['shared_bytes'] > 99 * 1024:
            raise RuntimeError('kernel exceeds the 99 KiB shared-memory budget: '+str(kernel))
    assert 'torch' not in sys.modules
    return dict(arch=arch,specializations=len(backend.kernels),
        max_shared_bytes=max(x['shared_bytes'] for x in backend.kernels.values()),torch_imported=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arch',type=int,default=89)
    print(json.dumps(run(parser.parse_args().arch)))
