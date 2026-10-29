"""Pool metadata reuse across two live CUDA contexts; no training updates."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_triton.runtime import Scalar
    from model.dense_cublas import tensor_kernels as tk
    with TrainingBackend(allocator_backend='async') as outer:
        e=Engine(outer);x=e.full((1024,),3.);pointer=x.data.storage
        outer.release_owned((x.data,))
        current=outer._bind('cuCtxGetCurrent',[ct.POINTER(ct.c_void_p)])
        with TrainingBackend(allocator_backend='async') as inner:
            before=ct.c_void_p();current(ct.byref(before))
            reused=outer.empty((1024,),'fp32')
            after=ct.c_void_p();current(ct.byref(after))
            assert reused.storage==pointer and before.value==after.value
            outer.launch(tk.fill,(4,),(reused,Scalar(7.)),N=1024)
            np.testing.assert_array_equal(outer.to_numpy(reused),7.)
            other=Engine(inner).full((1024,),9.)
            np.testing.assert_array_equal(inner.to_numpy(other.data),9.)
        np.testing.assert_array_equal(outer.to_numpy(reused),7.)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(status='complete',pool_reuse_no_context_change=True,
        consumer_context_binding=True,torch_imported='torch' in sys.modules),indent=2))


if __name__=='__main__':main()
