"""Isolated cuDNN SDPA experiment using CUDA addresses, never framework tensors."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def load_cudnn():
    root = Path(__file__).resolve().parents[1] / '.cache/native-libs'
    dlls = root / 'nvidia/cudnn/bin'
    sys.path.insert(0, str(root))
    cuda = Path(os.environ['CUDA_PATH']) / 'bin/x64'
    os.environ['PATH'] = os.pathsep.join((str(dlls), str(cuda), os.environ['PATH']))
    global _cudart
    _cudart = ct.WinDLL(str(cuda / 'cudart64_13.dll'))
    os.environ['CUDNN_FRONTEND_CUDART_LIB_NAME'] = str(cuda / 'cudart64_13.dll')
    # The cuDNN dispatcher itself uses LoadLibrary, not Python's DLL loader.
    ct.windll.kernel32.SetDllDirectoryW(str(dlls))
    import cudnn
    return cudnn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='experiments/results/cublas/cudnn_causal.json')
    a = p.parse_args()
    sys.meta_path.insert(0, RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.ops import CublasDenseOps
    cudnn = load_cudnn()
    result = dict(status='running', version=cudnn.backend_version(), cases=[])
    target = Path(a.output); target.parent.mkdir(parents=True, exist_ok=True)
    with TrainingBackend(allocator_backend='async') as backend:
        ops = CublasDenseOps(backend)
        handle = cudnn._pybind_module.create_handle()
        cudnn._pybind_module._raw_set_stream(handle, backend.main_stream)
        try:
            for b, s in ((2, 128), (4, 3072), (3, 4096)):
                mark = backend.mark()
                rng = np.random.default_rng(13)
                arrays = [ops.cast(backend.from_numpy(rng.standard_normal((b,s,h,96),dtype=np.float32)), 'bf16')
                          for h in (16,4,4,16)]
                q,k,v,dy = arrays
                ref, saved = ops.attention(q,k,v)
                grads = ops.attention_backward(saved,dy)
                refs = [backend.to_numpy(x) for x in (ref,*grads)]
                graphs=[]
                out=backend.empty(q.shape,'bf16'); stats=backend.empty((b,16,s),'fp32')
                dqs=[backend.empty(x.shape,x.dtype) for x in (q,k,v)]
                for backward in (False,True):
                    g=cudnn._pybind_module.backend_graph(io_data_type=cudnn.data_type.BFLOAT16,
                        intermediate_data_type=cudnn.data_type.FLOAT,compute_data_type=cudnn.data_type.FLOAT,handle=handle)
                    def tensor(x):
                        bb,ss,hh,dd=x.shape
                        return g.tensor(dim=[bb,hh,ss,dd],stride=[ss*hh*dd,dd,hh*dd,1])
                    Q,K,V=[tensor(x) for x in (q,k,v)]
                    bindings=[(Q,q),(K,k),(V,v)]
                    if backward:
                        O,DY=tensor(out),tensor(dy)
                        L=g.tensor(dim=[b,16,s,1],stride=[16*s,s,1,1],data_type=cudnn.data_type.FLOAT)
                        outputs=g.sdpa_backward(Q,K,V,O,DY,L,attn_scale=96**-.5,use_causal_mask=True)
                        bindings.extend(((O,out),(DY,dy),(L,stats)))
                        for desc,x in zip(outputs,dqs):
                            bb,ss,hh,dd=x.shape
                            desc.set_output(True).set_dim([bb,hh,ss,dd]).set_stride([ss*hh*dd,dd,hh*dd,1])
                            bindings.append((desc,x))
                    else:
                        O,L=g.sdpa(Q,K,V,generate_stats=True,attn_scale=96**-.5,use_causal_mask=True)
                        O.set_output(True).set_dim([b,16,s,96]).set_stride([s*16*96,96,16*96,1])
                        L.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b,16,s,1]).set_stride([16*s,s,1,1])
                        bindings.extend(((O,out),(L,stats)))
                    g.build([cudnn.heur_mode.A])
                    bind={desc.get_uid():x.storage for desc,x in bindings}
                    n=g.get_workspace_size()
                    work=backend.empty((max(1,(n+3)//4),),'fp32')
                    def run(g=g,bind=bind,work=work):
                        ct.windll.kernel32.SetLastError(0)
                        g._execute(bind,work.storage,handle)
                    for _ in range(3):run()
                    ms=backend.time_call(run,backend.main_stream,repeats=20)
                    graphs.append(dict(backward=backward,ms=ms,workspace_bytes=n))
                values=[backend.to_numpy(x) for x in (out,*dqs)]
                errors=[float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-20)) for x,y in zip(values,refs)]
                assert max(errors)<.01,errors
                row=dict(batch=b,seq=s,relative_l2=errors,timings=graphs)
                result['cases'].append(row);print(json.dumps(row),flush=True)
                target.write_text(json.dumps(result,indent=2))
                backend.release_since(mark)
        finally:
            if 'run' in locals():del run
            if 'g' in locals():del g
            cudnn._pybind_module._raw_destroy_handle(handle)
    result.update(status='complete',torch_imported='torch' in sys.modules)
    target.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
