"""Production-shape NS GEMM parity and timing; random operands, no updates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import triton
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--shapes',default='2:1536:5504,8:1536:1536,16:384:1536',help='batch:rows:columns tuples')
    p.add_argument('--resident-v2',action='store_true')
    a=p.parse_args();sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_cublas.muon import ns_gemm
    report=dict(status='running',cases=[]);a.output.parent.mkdir(parents=True,exist_ok=True)
    with TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=True)) as b:
        ops=CublasDenseOps(b);rng=np.random.default_rng(19)
        if a.resident_v2:
            from model.dense_cublas.autograd import Engine
            from model.dense_cublas.model import Model
            from model.dense_cublas.optim import Optimizer
            cfg=json.loads((Path(__file__).resolve().parents[1]/'trainer/config_instinct_v2.json').read_text())
            e=Engine(b,fp8=True,fp8_filter='auto')
            model=Model(e,cfg,param_dtype='bf16',muon_mantissa=True)
            optimizer=Optimizer(e,model.parameters(),'muon',muon_mantissa=True,muon_momentum_dtype='bf16')
            report['resident_bytes']=b._live_bytes
        shapes=[tuple(map(int,item.split(':'))) for item in a.shapes.split(',')]
        for batch,m,k in shapes:
            mark=b.mark()
            x=ops.cast(b.from_numpy((rng.standard_normal((batch,m,k),dtype=np.float32)/np.sqrt(k)).astype(np.float32)),'bf16')
            inputs={}
            for role,n,kk,alpha,beta in (('gram',m,k,1.,0.),('polynomial',m,m,2.0315,-4.775),('update',k,m,1.,3.4445)):
                if role=='gram':left,right,c=x,x,x;astrides=(m*k,k,1);bstrides=(m*k,1,k)
                elif role=='polynomial':
                    left=right=c=inputs['gram'];astrides=bstrides=(m*m,m,1)
                else:left,right,c=inputs['polynomial'],x,x;astrides=(m*m,m,1);bstrides=(m*k,k,1)
                expected,actual=[b.empty((batch,m,n),'bf16') for _ in range(2)]
                symmetric=role=='polynomial' or (role=='gram' and (m,k)!=(1536,5504))
                tiles=triton.cdiv(m,64)
                grid=(tiles*(tiles+1)//2,1,batch) if symmetric else (tiles,triton.cdiv(n,64),batch)
                def reference():
                    b.launch(ns_gemm,grid,(left,right,c,expected),M=m,N=n,K=kk,
                        A0=astrides[0],A1=astrides[1],A2=astrides[2],B0=bstrides[0],B1=bstrides[1],B2=bstrides[2],
                        C0=m*n,C1=n,C2=1,ALPHA=alpha,BETA=beta,BM=64,BN=64,BK=32,SYMMETRIC=symmetric,TRANSPOSE_OUTPUT=False)
                def candidate():b.batched_matmul(left,right,c,actual,m,n,kk,astrides,bstrides,alpha,beta)
                reference();candidate()
                original,new=b.to_numpy(expected),b.to_numpy(actual)
                relative=float(np.linalg.norm(new-original)/max(np.linalg.norm(original),1e-20))
                assert np.isfinite(new).all() and relative<.005, (role,relative)
                timings={name:float(np.median([b.time_call(fn,b.main_stream,repeats=3) for _ in range(3)]))
                         for name,fn in (('triton',reference),('cublas',candidate))}
                row=dict(batch=batch,m=m,n=n,k=kk,role=role,relative_l2=relative,gpu_ms=timings)
                report['cases'].append(row);print(json.dumps(row),flush=True)
                a.output.write_text(json.dumps(report,indent=2))
                inputs[role]=expected
            b.release_since(mark)
        report.update(status='complete',torch_imported='torch' in sys.modules,algorithms=b.tuning_report())
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
