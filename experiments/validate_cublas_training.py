"""Real-GPU acceptance: complete model/optimizer, resume parity and bounded memory.

python -m experiments.validate_cublas_training --fp8 0 --dtype fp32
python -m experiments.validate_cublas_training --fp8 1 --dtype bf16 --overlap_dw
A successful run is not a convergence or throughput claim.
"""
import argparse
import json
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import time
from experiments.smoke_dense_triton import RejectTorch


def run(args):
    if 'torch' in sys.modules:raise RuntimeError('run the native validation in a fresh process')
    sys.meta_path.insert(0,RejectTorch())
    import numpy as np
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.optim import Optimizer
    from model.dense_cublas import checkpoint
    config=dict(hidden_size=32,num_hidden_layers=2,vocab_size=67,num_attention_heads=2,num_key_value_heads=1,
        head_dim=args.head_dim,intermediate_size=48,max_position_embeddings=128,flash_attn=bool(args.flash_attn),
        model_architecture=args.architecture,use_grad_checkpoint=args.checkpoint,dropout=args.dropout,
        num_experts=3,num_experts_per_tok=2,moe_intermediate_size=48,full_attention_interval=2,
        linear_key_head_dim=16,linear_value_head_dim=16,linear_num_key_heads=1,linear_num_value_heads=2,
        linear_conv_kernel_dim=4,loop_max_steps=3,q_threshold=.8,n_supervision=2)
    schedule=ScheduleConfig(workspace_mib=args.workspace_mib,overlap_dw=args.overlap_dw,tune=args.tune)
    with TrainingBackend(args.device,schedule,pool_limit_mib=args.allocator_cache_mib) as backend:
        baseline=backend.memory();e=Engine(backend,fp8=bool(args.fp8),seed=123)
        model=Model(e,config,args.param_dtype,args.dtype,args.kv_cache_dtype)
        optimizer=Optimizer(e,model.parameters(),args.optimizer,lr=1.e-3,loss_scaling=args.dtype=='fp16')
        ids=(np.arange(34).reshape(2,17)%66+1).astype(np.int64);losses=[];memories=[];times=[]
        oracle_checked=False
        if args.reference:
            if args.architecture=='looped' or args.dropout:
                raise ValueError('--reference requires Dense/MoE/Linear and dropout=0; use deterministic resume tests for native Looped/dropout')
            with tempfile.TemporaryDirectory() as tmp:
                source,target=Path(tmp)/'input.npz',Path(tmp)/'reference.npz'
                arrays={'config':np.asarray(json.dumps(model.config)),'ids':ids}
                arrays.update({'parameter/'+name:e.numpy(value) for name,value in model.params.items()})
                np.savez(source,**arrays)
                subprocess.run([sys.executable,'-m','experiments.reference_cublas_training','--input',str(source),'--output',str(target)],
                    check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1'))
                mark=backend.mark();output=model(ids,ids);gradients=e.vjp(output.loss+output.aux_loss)
                atol,rtol=(args.atol,args.rtol)
                with np.load(target,allow_pickle=False) as expected:
                    np.testing.assert_allclose(e.numpy(output.logits),expected['logits'],atol=atol,rtol=rtol)
                    np.testing.assert_allclose(e.numpy(output.loss+output.aux_loss),expected['loss'],atol=atol,rtol=rtol)
                    for name,p in model.params.items():
                        key='gradient/'+name
                        if key in expected:np.testing.assert_allclose(e.numpy(gradients[p]),expected[key],atol=atol,rtol=rtol,err_msg=name)
                backend.release_since(mark);oracle_checked=True
        def step():
            mark=backend.mark();started=time.perf_counter()
            out=model(ids,ids);loss=e.item(out.loss+out.aux_loss)
            if not np.isfinite(loss):raise FloatingPointError('nonfinite loss')
            optimizer.backward(out.loss+out.aux_loss);result=optimizer.step()
            backend.release_since(mark)
            if not result['updated']:raise FloatingPointError('nonfinite gradient/overflow; lower initial loss scale for diagnostics')
            return loss,time.perf_counter()-started
        for _ in range(args.steps):
            loss,elapsed=step();losses.append(loss);times.append(elapsed);memories.append(backend.memory())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'resume.npz';checkpoint.save(path,model,optimizer,{'step':args.steps})
            expected_loss,_=step();expected=[e.numpy(p) for p in model.parameters()]
            checkpoint.load(path,model,optimizer)
            actual_loss,_=step()
            np.testing.assert_allclose(actual_loss,expected_loss,atol=3e-5,rtol=3e-5)
            for p,reference in zip(model.parameters(),expected):np.testing.assert_allclose(e.numpy(p),reference,atol=3e-5,rtol=3e-5)
        mark=backend.mark();model.training=False
        with e.no_grad():
            full=model(ids[:1,:4]);prefill=model(ids[:1,:3],use_cache=True);decoded=model(ids[:1,3:4],cache=prefill.cache,use_cache=True)
            # FP8 KV is lossy; both cache layouts still need to stay numerically bounded.
            tolerance=.08 if args.kv_cache_dtype.startswith('fp8') or args.fp8 else .015 if args.dtype!='fp32' else 3e-4
            np.testing.assert_allclose(e.numpy(decoded.logits),e.numpy(full.logits)[:,3:4],atol=tolerance,rtol=tolerance)
        backend.release_since(mark)
        if memories[-1]['live_bytes']!=memories[0]['live_bytes']:raise RuntimeError('live training buffers grew across steps')
        if memories[-1]['cached_bytes']>backend.pool_limit:raise RuntimeError('allocator cache exceeded configured limit')
        return dict(status='ok',arch=backend.arch,configuration=config,fp8=bool(args.fp8),dtype=args.dtype,
            torch_imported='torch' in sys.modules,losses=losses,step_seconds=times,
            baseline_memory=baseline,step_memory=memories,final_memory=backend.memory(),
            scheduler=backend.tuning_report(),cpu_oracle_checked=oracle_checked,resume_parity=True,decode_parity=True,
            note='Tiny-model correctness smoke; first steps include JIT/tuning. Measure steady-state MFU separately.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',type=int,default=0);p.add_argument('--fp8',type=int,choices=[0,1],default=0)
    p.add_argument('--dtype',choices=['fp32','bf16','fp16'],default='fp32');p.add_argument('--param_dtype',choices=['fp32','bf16','fp16'],default='fp32')
    p.add_argument('--architecture',choices=['standard','moe','linear','looped'],default='standard')
    p.add_argument('--optimizer',choices=['adamw','adafactor','muon'],default='adamw')
    p.add_argument('--checkpoint',type=int,choices=[0,1,2],default=0);p.add_argument('--dropout',type=float,default=0.)
    p.add_argument('--kv_cache_dtype',choices=['fp32','bf16','fp16','fp8_e4m3','fp8_e5m2'],default='fp32')
    p.add_argument('--head_dim',type=int,default=16)
    p.add_argument('--flash_attn',type=int,choices=[0,1],default=1);p.add_argument('--steps',type=int,default=5)
    p.add_argument('--workspace_mib',type=int,default=32);p.add_argument('--allocator_cache_mib',type=int,default=256)
    p.add_argument('--reference',action='store_true',help='compare initial logits/loss/gradients with an isolated CPU PyTorch process')
    p.add_argument('--atol',type=float,default=3e-4);p.add_argument('--rtol',type=float,default=3e-3)
    p.add_argument('--overlap_dw',action='store_true');p.add_argument('--tune',action='store_true')
    args=p.parse_args()
    if args.head_dim<2 or args.head_dim%2:p.error('head_dim must be positive and even')
    if args.steps<2:p.error('at least two steps are needed for memory checks')
    print(json.dumps(run(args),indent=2))
