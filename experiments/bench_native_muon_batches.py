"""Three Muon batch geometries, complete five-step NS chain, no weight updates."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--catalog-limit',type=int,choices=(8,64),default=8)
    p.add_argument('--baseline-only',action='store_true')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.muon import MuonPlan
    from model.dense_cublas.policy import enable_fused_ops
    report=dict(status='running',cases=[],scope='Five NS iterations only; no momentum, clipping, updates or training')
    with TrainingBackend(allocator_backend='async',config=ScheduleConfig(tune=True,candidates=a.catalog_limit)) as b:
        enable_fused_ops(b);b.muon_gemm_backend='cublas';e=Engine(b)
        plan=MuonPlan(SimpleNamespace(e=e));rng=np.random.default_rng(311)
        for m,k,batches in ((1536,5504,(3,8,16)),(1536,1536,(8,16))):
            source=rng.standard_normal((m,k),dtype=np.float32)/np.sqrt(m*k);reference=None
            for batch in (batches[:1] if a.baseline_only else batches):
                mark=b.mark();x=e.array(np.broadcast_to(source,(batch,m,k)).copy(),'bf16').data
                b.muon_expanded_batches=True;b._commands=[]
                try:
                    y=x
                    for _ in range(5):
                        gram=plan.gemm(y,y,y,m,m,k,(m*k,k,1),(m*k,1,k),symmetric=(m,k)!=(1536,5504))
                        poly=plan.gemm(gram,gram,gram,m,m,m,(m*m,m,1),(m*m,m,1),alpha=2.0315,beta=-4.775,symmetric=True)
                        y=plan.gemm(poly,y,y,m,k,m,(m*m,m,1),(m*k,k,1),beta=3.4445)
                finally:commands=b._commands;b._commands=None
                def run():
                    for command in commands:command.execute()
                run();value=b.to_numpy(y)
                if reference is None:reference=value[0].copy()
                errors=[float(np.linalg.norm(v-reference)/max(np.linalg.norm(reference),1e-20)) for v in value]
                assert np.isfinite(value).all() and max(errors)<.005,errors
                b.time_call(run,b.main_stream,repeats=10)
                times=[b.time_call(run,b.main_stream,repeats=5) for _ in range(5)]
                row=dict(shape=[m,k],batch=batch,median_ms=float(np.median(times)),per_matrix_ms=float(np.median(times))/batch,
                         samples_ms=times,max_relative_l2=max(errors),bitwise_equal=bool(np.all(value==reference)),
                         temporary_live_bytes=sum(size for serial,size in b._live.values() if serial>=mark))
                report['cases'].append(row);print(json.dumps(row),flush=True)
                a.output.write_text(json.dumps(report,indent=2));b.release_since(mark)
        report['algorithms']=b.tuning_report()
    report.update(status='complete',torch_imported='torch' in sys.modules)
    a.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
