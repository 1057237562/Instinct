"""Finite native V2 recipe probe. Updates always finish with a verified save.

Default: fixed-weight forward/backward with real resident optimizer state.
--updates enables complete accumulated Muon/Adam steps and saves the final
model, optimizer, scaling counters, gradients, RNG and synthetic batch cursor.
"""
import argparse
import json
import math
from pathlib import Path
import signal
import statistics
import sys
import time
import os
import ctypes as ct
import numpy as np
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',type=int,choices=[2,4],default=4)
    p.add_argument('--layers',type=int,default=32)
    p.add_argument('--shapes',nargs='+',default=['4x3072x3','3x4096x1'],help='batch x sequence x documents')
    p.add_argument('--fp8',type=int,choices=[0,1],default=1)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--updates',action='store_true')
    p.add_argument('--pool-mib',type=int,default=256)
    p.add_argument('--allocator-backend',choices=['driver','async'],default='driver')
    p.add_argument('--muon-workspace-mib',type=int,default=512)
    p.add_argument('--tune',action='store_true')
    p.add_argument('--cache-mib',type=int,default=1792)
    p.add_argument('--profile-driver',action='store_true')
    p.add_argument('--profile-kernels',action='store_true',help='CUDA-event breakdown after warmup (profiling overhead is included)')
    p.add_argument('--attention-tile',type=int,choices=[16,32,64])
    p.add_argument('--no-packed-skip',action='store_true')
    p.add_argument('--arena-limit-gib',type=float,default=14.37)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if min(args.layers,args.steps,args.accumulation)<1 or args.warmup<0:p.error('invalid counts')
    sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.optim import Optimizer
    from model.dense_cublas import checkpoint
    from model.dense_triton.runtime import ITEMSIZE
    class BudgetBackend(TrainingBackend):
        peak_arena=0
        def __init__(self,*a,**kw):
            self.driver_profile={}
            self.kernel_events=[];self.profile_active=False
            super().__init__(*a,**kw)
        def timed(self,label,call):
            if not self.profile_active:return call()
            start,end=self.create_event(timing=True),self.create_event(timing=True)
            self.record_event(start);begin=time.perf_counter()
            result=call()
            host_ms=(time.perf_counter()-begin)*1000
            self.record_event(end)
            self.kernel_events.append((label,start,end,host_ms))
            return result
        def launch(self,fn,grid,values,**constants):
            call=super().launch
            return self.timed('kernel/'+getattr(fn,'__name__',str(fn)),lambda:call(fn,grid,values,**constants))
        def matmul(self,*values,**kwargs):
            call=super().matmul
            role=kwargs.get('role',values[8] if len(values)>8 else 'gemm')
            label=f'gemm/{role}/{values[0].dtype}/{values[5:8]}'
            return self.timed(label,lambda:call(*values,**kwargs))
        def kernel_report(self):
            self.synchronize();totals={}
            for label,start,end,host in self.kernel_events:
                elapsed=ct.c_float();self._elapsed(ct.byref(elapsed),start,end)
                entry=totals.setdefault(label,dict(calls=0,gpu_ms=0.,host_submit_ms=0.))
                entry['calls']+=1;entry['gpu_ms']+=elapsed.value;entry['host_submit_ms']+=host
            return dict(sorted(totals.items(),key=lambda item:-item[1]['gpu_ms']))
        def _bind(self,name,types):
            call=super()._bind(name,types)
            if not args.profile_driver:return call
            def measured(*values):
                start=time.perf_counter()
                result=call(*values)
                count,seconds=self.driver_profile.get(name,(0,0.))
                self.driver_profile[name]=(count+1,seconds+time.perf_counter()-start)
                return result
            return measured
        def empty(self,shape,dtype='fp32',zero=False):
            needed=math.prod(shape)*ITEMSIZE[dtype]
            pool=getattr(self,'_pool',{}).get(needed)
            additional=0 if pool and not self._uploading_constant else needed
            arena=sum(n for _,n in self._allocations)
            if arena+additional>args.arena_limit_gib*2**30:
                raise MemoryError(f'native arena guard: {(arena+additional)/2**30:.3f} GiB > {args.arena_limit_gib}')
            result=super().empty(shape,dtype,zero)
            self.peak_arena=max(self.peak_arena,sum(n for _,n in self._allocations))
            return result
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'trainer/config_instinct_v2.json').read_text())
    cfg.update(num_hidden_layers=args.layers,use_grad_checkpoint=args.mode,selective_cache_mib=args.cache_mib)
    report={'status':'running','args':vars(args),'config':cfg,'windows':[],
            'scope':'complete accumulated updates; excludes loader and saves' if args.updates else 'fixed weights; forward/backward only, real optimizer/gradient residency'}
    def write():target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    stop=False
    def request_pause(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,request_pause)
    prefix=f'native_v2_ckpt{args.mode}_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    pause=root/'checkpoints'/(prefix+'.pause_request')
    report['pause_file']=str(pause)
    micro=updates=0
    write()
    with BudgetBackend(config=ScheduleConfig(tune=args.tune),pool_limit_mib=args.pool_mib,allocator_backend=args.allocator_backend) as b:
        b.attention_tile=args.attention_tile
        b.packed_skip=not args.no_packed_skip
        e=Engine(b,fp8=bool(args.fp8),fp8_filter='auto',seed=307)
        model=Model(e,cfg,param_dtype='bf16',muon_mantissa=True,seed=307)
        opt=Optimizer(e,model.parameters(),'muon',lr=3e-4,weight_decay=.1,
                      muon_mantissa=True,muon_momentum_dtype='bf16',
                      muon_batch_size=16,muon_workspace_mb=args.muon_workspace_mib)
        batches=[];rng=np.random.default_rng(307)
        for shape in args.shapes:
            batch,seq,documents=map(int,shape.split('x'))
            if not 1<=documents<=seq:p.error('invalid document count')
            ids=rng.integers(3,cfg['vocab_size'],(batch,seq),dtype=np.int64)
            segments=np.broadcast_to(np.arange(seq)*documents//seq,(batch,seq)).copy()
            labels=ids.copy()
            labels[:,1:][segments[:,1:]!=segments[:,:-1]]=-100
            batches.append((e.array(ids),e.array(labels),e.array(segments) if documents>1 else None,
                            int(np.count_nonzero(labels[:,1:]!=-100))))
        report.update(initial_memory=b.memory(),arch=b.arch,parameter_count=sum(p.size for p in model.parameters(False)))
        write()
        def save_current():
            cursor=dict(synthetic=True,seed=307,next_microstep=micro,optimizer_updates=updates,
                        shapes=args.shapes,accumulation=args.accumulation)
            resume=root/'checkpoints'/(prefix+'_resume.npz')
            weights=root/'out'/(prefix+'.npz')
            snapshot=checkpoint.snapshot(model,opt,cursor,rng)
            checkpoint.write(resume,snapshot)
            with np.load(resume,allow_pickle=False) as stored:
                for key,value in snapshot.items():np.testing.assert_array_equal(stored[key],value,err_msg=key)
            weight_snapshot={key:value for key,value in snapshot.items() if key.startswith('model/') or key=='__metadata__'}
            checkpoint.write(weights,weight_snapshot)
            with np.load(weights,allow_pickle=False) as stored:
                for key,value in weight_snapshot.items():np.testing.assert_array_equal(stored[key],value,err_msg=key)
            report.update(saved_resume=str(resume),saved_weights=str(weights),verified_current_state=True)
            print(json.dumps({'saved_resume':str(resume),'verified_current_state':True}),flush=True)
        try:
            for window in range(args.warmup+args.steps):
                b.profile_active=args.profile_kernels and window>=args.warmup
                b.synchronize();start=time.perf_counter();valid=0
                b._peak_live_bytes=b._live_bytes;b.peak_arena=sum(n for _,n in b._allocations)
                for _ in range(args.accumulation if args.updates else 1):
                    ids,labels,segments,count=batches[micro%len(batches)]
                    mark=b.mark()
                    loss=model(ids,labels,sequence_ids=segments,return_logits=False).loss
                    if args.updates:opt.backward(loss)
                    else:
                        def consume(param,grad):
                            e.copy(opt.gradients[param],e.wrap(grad))
                            return True
                        e.vjp(loss,leaf_callback=consume)
                    last_loss=e.item(loss);micro+=1;valid+=count
                    b.release_since(mark)
                    if args.updates:
                        print(json.dumps({'phase':'backward_done','microstep':micro,'loss':last_loss}),flush=True)
                b.synchronize();backward_end=time.perf_counter()
                if args.updates:
                    print(json.dumps({'phase':'optimizer_begin','microsteps':micro}),flush=True)
                    result=opt.step()
                    if not result['updated']:raise FloatingPointError('optimizer skipped a nonfinite update')
                    updates+=1
                b.synchronize();end=time.perf_counter()
                row=dict(window=window,warmup=window<args.warmup,total_ms=(end-start)*1000,
                         forward_backward_ms=(backward_end-start)*1000,optimizer_ms=(end-backward_end)*1000,
                         valid_tokens=valid,loss=last_loss,peak_live_gib=b._peak_live_bytes/2**30,
                         peak_arena_gib=b.peak_arena/2**30,memory=b.memory())
                report['windows'].append(row);write();print(json.dumps(row),flush=True)
                if stop or pause.exists():break
            measured=[r for r in report['windows'] if not r['warmup']]
            report['summary']=dict(median_ms=statistics.median(r['total_ms'] for r in measured),
                useful_tokens_per_second=sum(r['valid_tokens'] for r in measured)*1000/sum(r['total_ms'] for r in measured),
                peak_live_gib=max(r['peak_live_gib'] for r in measured),peak_arena_gib=max(r['peak_arena_gib'] for r in measured)) if measured else {}
            report['status']='paused' if stop or pause.exists() else 'ok'
        except Exception as exc:
            report.update(status='error',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if args.updates and (updates or opt.accumulated):save_current()
            report.update(microsteps=micro,updates=updates,algorithms=b.tuning_report(),torch_imported='torch' in sys.modules)
            report['driver_profile']=b.driver_profile
            if args.profile_kernels:report['kernel_profile']=b.kernel_report()
            write()


if __name__=='__main__':main()
