"""Torch-free GPU correctness acceptance for selective replay and packed V2.

Only the tiny optimizer check changes weights; its final full resume state is
saved and checked before the context closes. No corpus or real trainer is used.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def run(target,allocator='driver',tile=None):
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.optim import Optimizer
    from model.dense_cublas import checkpoint
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'running','cases':[],'allocator':allocator,'attention_tile':tile}
    def write():target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    cfg=dict(hidden_size=192,num_hidden_layers=2,intermediate_size=512,
             num_attention_heads=4,num_key_value_heads=1,head_dim=96,
             vocab_size=256,max_position_embeddings=4096)
    for fp8,seq,packed in ((False,67,False),(False,67,True),(True,67,True),(True,4096,False)):
        results=[]
        for mode in (2,4):
            with TrainingBackend(config=ScheduleConfig(),allocator_backend=allocator) as b:
                b.attention_tile=tile
                e=Engine(b,fp8=fp8)
                m=Model(e,cfg|{'use_grad_checkpoint':mode},param_dtype='bf16',muon_mantissa=True)
                ids=np.arange(seq,dtype=np.int64)[None]%253+3
                segments=np.arange(seq,dtype=np.int64)[None]*3//seq if packed else None
                out=m(ids,ids,sequence_ids=segments)
                loss=e.item(out.loss)
                grads={p.name:e.numpy(g) for p,g in e.vjp(out.loss).items()}
                assert np.isfinite(loss) and all(np.isfinite(g).all() for g in grads.values())
                results.append((loss,grads))
        np.testing.assert_allclose(results[1][0],results[0][0],atol=1e-6,rtol=1e-6)
        worst=0.
        for name,g in results[0][1].items():
            actual=results[1][1][name]
            np.testing.assert_allclose(actual,g,atol=2e-5,rtol=2e-3,err_msg=name)
            worst=max(worst,float(np.max(np.abs(actual-g))))
        row=dict(fp8=fp8,seq=seq,packed=packed,loss=results[0][0],max_gradient_difference=worst)
        report['cases'].append(row);write();print(json.dumps(row),flush=True)
    # Verify skipping masked tiles against the original GPU mask-only path.
    skip_results=[]
    for skip in (False,True):
        with TrainingBackend(config=ScheduleConfig(),allocator_backend=allocator) as b:
            b.packed_skip=skip
            e=Engine(b,fp8=True);m=Model(e,cfg|{'use_grad_checkpoint':4},param_dtype='bf16',muon_mantissa=True)
            ids=np.arange(137,dtype=np.int64)[None]%253+3
            segments=np.asarray([[3]*19+[5]*22+[3]*81+[-1]*15],np.int64)
            out=m(ids,ids,sequence_ids=segments)
            loss=e.item(out.loss);grads={p.name:e.numpy(g) for p,g in e.vjp(out.loss).items()}
            skip_results.append((loss,grads))
    assert skip_results[0][0]==skip_results[1][0]
    skip_error=0.
    for name,expected in skip_results[0][1].items():
        actual=skip_results[1][1][name]
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-3,err_msg=name)
        skip_error=max(skip_error,float(np.max(np.abs(actual-expected))))
    report['packed_skip_gradient_max_error']=skip_error;write()
    # Isolated documents must have the same logits as two independent sequences.
    with TrainingBackend(config=ScheduleConfig(),allocator_backend=allocator) as b:
        b.attention_tile=tile
        e=Engine(b);m=Model(e,cfg|{'use_grad_checkpoint':4},activation_dtype='bf16')
        ids=np.arange(70,dtype=np.int64)[None]+1
        packed=m(ids,ids,sequence_ids=np.repeat([[0,1]],35,axis=1))
        logits=e.numpy(packed.logits)
        for start in (0,35):
            independent=m(ids[:,start:start+35])
            np.testing.assert_allclose(logits[:,start:start+35],e.numpy(independent.logits),atol=.004,rtol=.01)
        report['packed_isolation']=True;write()
    for storage in ('fp32','bf16'):
        with TrainingBackend(config=ScheduleConfig(),allocator_backend=allocator) as b:
            b.attention_tile=tile
            e=Engine(b);m=Model(e,cfg|{'num_hidden_layers':1,'use_grad_checkpoint':4},
                               param_dtype='bf16',muon_mantissa=True)
            opt=Optimizer(e,m.parameters(),'muon',muon_mantissa=True,muon_momentum_dtype=storage,
                          muon_workspace_mb=8,muon_batch_size=2)
            ids=np.arange(34,dtype=np.int64).reshape(2,17)+1
            for step in range(2):
                mark=b.mark()
                opt.backward(m(ids,ids,sequence_ids=np.zeros_like(ids)).loss)
                assert opt.step()['updated']
                b.release_since(mark)
            progress={'synthetic':True,'seed':42,'steps':2,'optimizer_updates':2,'next_microstep':2}
            resume=target.parent/f'{target.stem}_muon_{storage}_{allocator}_resume.npz'
            arrays=checkpoint.snapshot(m,opt,progress)
            checkpoint.write(resume,arrays)
            with np.load(resume,allow_pickle=False) as saved:
                for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value,err_msg=key)
            restored=checkpoint.load(resume,m,opt)
            assert restored==progress
            for key,value in checkpoint.snapshot(m,opt,progress).items():
                np.testing.assert_array_equal(value,arrays[key],err_msg=key)
            low=[a for key,a in arrays.items() if key.endswith('/mantissa')]
            assert all(a.dtype==np.uint16 for a in low) and any(np.any(a) for a in low)
            row={'momentum_dtype':storage,'resume':str(resume),'verified_current_state':True}
            report.setdefault('optimizer',[]).append(row);write();print(json.dumps(row),flush=True)
    report.update(status='ok',torch_imported='torch' in sys.modules)
    write()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='experiments/results/cublas/native_v2_validation.json')
    parser.add_argument('--allocator',choices=['driver','async'],default='driver')
    parser.add_argument('--attention-tile',type=int,choices=[16,32,64])
    args=parser.parse_args()
    run(args.output,args.allocator,args.attention_tile)
