"""Split-weight Muon transpose I/O parity and timing; synthetic tensors only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import triton
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--momentum-dtype',choices=['bf16','fp32'],default='bf16')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas import muon
    from model.dense_triton.runtime import Scalar
    report=dict(status='running',cases=[])
    with TrainingBackend(allocator_backend='async') as b:
        enable_fused_ops(b);e=Engine(b);rng=np.random.default_rng(238)
        for rows,cols in ((5504,1536),(253,96)):
            mark=b.mark();n=rows*cols
            grad=e.array(rng.normal(size=(rows,cols)).astype(np.float32),'bf16')
            initial_m=rng.normal(size=(rows,cols)).astype(np.float32)
            initial_p=rng.normal(size=(rows,cols)).astype(np.float32)
            reference=None
            for tile in (0,32,64):
                local=b.mark();mom=e.array(initial_m,a.momentum_dtype);weight=e.array(initial_p,'bf16')
                low=e.full(weight.shape,0.,'u16');u=b.empty((1,cols,rows),'bf16')
                table=e.array(np.asarray([[weight.data.storage,grad.data.storage,mom.data.storage,low.data.storage]],np.int64))
                grid=(triton.cdiv(rows,tile),triton.cdiv(cols,tile),1) if tile else (triton.cdiv(n,1024),1)
                def prepare():
                    b.launch(muon.prepare_tiled if tile else muon.prepare,grid,(table.data,u,Scalar(.037)),
                        ROWS=rows,COLS=cols,MOM=.95,MOM_DTYPE=a.momentum_dtype,**({'TILE':tile} if tile else {}))
                def update():
                    b.launch(muon.apply_tiled if tile else muon.apply,grid,(table.data,u,Scalar(.9999),Scalar(-.001)),
                        ROWS=rows,COLS=cols,**({'TILE':tile} if tile else {}))
                prepare();update()
                values=[b.to_numpy(x) for x in (u,mom.data,weight.data,low.data)]
                if reference is None:reference=values
                else:
                    for x,y in zip(values,reference):np.testing.assert_array_equal(x,y)
                times={name:float(np.median([b.time_call(fn,b.main_stream,repeats=10) for _ in range(3)]))
                       for name,fn in (('prepare',prepare),('apply',update))}
                row=dict(rows=rows,cols=cols,tile=tile,momentum_dtype=a.momentum_dtype,gpu_ms=times,bitwise_equal=True)
                report['cases'].append(row);print(json.dumps(row),flush=True)
                a.output.write_text(json.dumps(report,indent=2));b.release_since(local)
            b.release_since(mark)
    report.update(status='complete',torch_imported='torch' in sys.modules)
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
