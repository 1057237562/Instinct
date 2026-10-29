"""Heuristic three-candidate symmetric NS GEMM screen; no updates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import triton
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--chain-only',action='store_true')
    p.add_argument('--blocked-gram',action='store_true')
    p.add_argument('--gram-tile',type=int,default=512)
    p.add_argument('--blocked-polynomial',action='store_true')
    p.add_argument('--ffn-batch',type=int,default=2)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.ops import CublasDenseOps
    from model.dense_cublas.muon import ns_gemm
    report=dict(status='running',cases=[])
    with TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=True)) as b:
        ops=CublasDenseOps(b);rng=np.random.default_rng(729)
        shapes=[(a.ffn_batch,1536,5504),(8,1536,1536),(16,384,1536)]
        if a.chain_only:shapes.extend([(3,1536,5504),(1,1536,5504)])
        for batch,m,k in shapes:
            mark=b.mark()
            x=ops.cast(b.from_numpy((rng.standard_normal((batch,m,k),dtype=np.float32)/np.sqrt(k)).astype(np.float32)),'bf16')
            gram=b.empty((batch,m,m),'bf16')
            b.batched_matmul(x,x,x,gram,m,m,k,(m*k,k,1),(m*k,1,k),1.,0.)
            for role in (() if a.chain_only else ('gram','polynomial')):
                left,right,c=(x,x,x) if role=='gram' else (gram,gram,gram)
                kk=k if role=='gram' else m
                astrides=(m*kk,kk,1);bstrides=(m*k,1,k) if role=='gram' else astrides
                alpha,beta=(1.,0.) if role=='gram' else (2.0315,-4.775)
                expected=b.empty((batch,m,m),'bf16');actual=b.empty((batch,m,m),'bf16')
                def baseline():b.batched_matmul(left,right,c,expected,m,m,kk,astrides,bstrides,alpha,beta)
                baseline();ref=b.to_numpy(expected)
                # Square 64 tiles bound accumulator registers; choose only the
                # two K tiles that halve/quarter the old BK32 loop trip count.
                choices=(0,128,-1) if (a.blocked_gram or a.blocked_polynomial) and m==1536 else (0,64,128)
                for bk in choices:
                    def run():
                        if not bk:return baseline()
                        if bk==-1:
                            from model.dense_cublas.muon_gram import gram as blocked
                            assert blocked(b,left,actual,tile=a.gram_tile,polynomial=role=='polynomial')
                            return
                        tiles=triton.cdiv(m,64)
                        b.launch(ns_gemm,(tiles*(tiles+1)//2,1,batch),(left,right,c,actual),
                            M=m,N=m,K=kk,A0=astrides[0],A1=astrides[1],A2=astrides[2],
                            B0=bstrides[0],B1=bstrides[1],B2=bstrides[2],C0=m*m,C1=m,C2=1,
                            ALPHA=alpha,BETA=beta,BM=64,BN=64,BK=bk,SYMMETRIC=True,TRANSPOSE_OUTPUT=False)
                    run();value=b.to_numpy(actual if bk else expected)
                    relative=float(np.linalg.norm(value-ref)/max(np.linalg.norm(ref),1e-20))
                    assert np.isfinite(value).all() and relative<.005,(role,bk,relative)
                    for _ in range(2):run()
                    ms=float(np.median([b.time_call(run,b.main_stream,repeats=3) for _ in range(3)]))
                    row=dict(batch=batch,m=m,k=kk,role=role,bk=bk,median_ms=ms,relative_l2=relative)
                    report['cases'].append(row);print(json.dumps(row),flush=True)
                    a.output.write_text(json.dumps(report,indent=2))
            if a.chain_only:
                from types import SimpleNamespace
                from model.dense_cublas.muon import MuonPlan
                plan=MuonPlan(SimpleNamespace(e=SimpleNamespace(backend=b)))
                b.muon_gemm_backend='cublas'
                start=ops.cast(b.from_numpy((rng.standard_normal((batch,m,k),dtype=np.float32)/np.sqrt(m*k)).astype(np.float32)),'bf16')
                reference=None
                for enabled in (False,True):
                    local=b.mark();b.muon_symmetric=enabled;y=start
                    b.muon_blocked_gram=enabled and a.blocked_gram
                    b.muon_gram_tile=a.gram_tile
                    b.muon_blocked_polynomial=enabled and a.blocked_polynomial
                    for _ in range(5):
                        gram=plan.gemm(y,y,y,m,m,k,(m*k,k,1),(m*k,1,k),symmetric=(m,k)!=(1536,5504))
                        poly=plan.gemm(gram,gram,gram,m,m,m,(m*m,m,1),(m*m,m,1),alpha=2.0315,beta=-4.775,symmetric=True)
                        y=plan.gemm(poly,y,y,m,k,m,(m*m,m,1),(m*k,k,1),beta=3.4445)
                        b.release_since(local,keep=(y,))
                    value=b.to_numpy(y)
                    if reference is None:reference=value
                    else:
                        relative=float(np.linalg.norm(value-reference)/max(np.linalg.norm(reference),1e-20))
                        assert np.isfinite(value).all() and relative<.005,(batch,m,k,relative)
                        row=dict(batch=batch,m=m,k=k,ns_steps=5,relative_l2=relative,bitwise_equal=bool(np.array_equal(value,reference)))
                        report['cases'].append(row);print(json.dumps(row),flush=True)
                        a.output.write_text(json.dumps(report,indent=2))
                    b.release_since(local)
            b.release_since(mark)
    report.update(status='complete',torch_imported='torch' in sys.modules)
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
