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
    p.add_argument('--async-release-mib',type=int)
    p.add_argument('--muon-workspace-mib',type=int,default=512)
    p.add_argument('--tune',action='store_true')
    p.add_argument('--workspace-mib',type=int,default=32)
    p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--fp8-fast-accum-output',action='store_true')
    p.add_argument('--cache-mib',type=int,default=1792)
    p.add_argument('--profile-driver',action='store_true')
    p.add_argument('--profile-memory',action='store_true',help='Synchronize at new phase allocation peaks; diagnostic windows are excluded from throughput')
    p.add_argument('--profile-kernels',action='store_true',help='CUDA-event breakdown after warmup (profiling overhead is included)')
    p.add_argument('--profile-cupti',action='store_true',help='Actual CUPTI kernel timestamps, without importing Torch')
    p.add_argument('--cupti-last-steps',type=int,default=2)
    p.add_argument('--micro-graph',action='store_true',help='Fixed-shape native forward/backward graph; Muon remains eager')
    p.add_argument('--attention-tile',type=int,choices=[16,32,64])
    p.add_argument('--pack-tile',type=int,choices=[32,64,128])
    p.add_argument('--attention-forward-tile',type=int,choices=[32,64,128])
    p.add_argument('--attention-forward-cols',type=int,choices=[32,64,128])
    p.add_argument('--attention-warps',type=int,choices=[4,8])
    p.add_argument('--attention-stages',type=int,choices=[1,2,3,4])
    p.add_argument('--fast-attention',action='store_true')
    p.add_argument('--compact-gradients',action='store_true')
    p.add_argument('--aligned-pointers',action='store_true')
    p.add_argument('--aligned-pointwise',action='store_true')
    p.add_argument('--async-attention-metadata',action='store_true')
    p.add_argument('--attention-plan-quantum',type=int,default=128)
    p.add_argument('--attention-plan-limit',type=int,default=32)
    p.add_argument('--attention-docs-per-plan',type=int,default=0)
    p.add_argument('--pad-attention',action='store_true')
    p.add_argument('--native-sdpa',action='store_true')
    p.add_argument('--native-sdpa-min-seq',type=int,default=0)
    p.add_argument('--cache-by-seq',default='')
    p.add_argument('--elementwise-block',type=int,choices=[256,512,1024,2048,4096])
    p.add_argument('--split-kv-gradients',action='store_true')
    p.add_argument('--production-samples',action='store_true',help='Resident samples from the production Parquet cache; loader time is excluded')
    p.add_argument('--export-samples')
    p.add_argument('--sample-file')
    p.add_argument('--norm-groups',type=int,choices=[128,256,512,1024,2048])
    p.add_argument('--norm-rope-warps',type=int,choices=[1,2,4])
    p.add_argument('--norm-rope-rows',type=int,default=1)
    p.add_argument('--fused-norm-cast',action='store_true')
    p.add_argument('--fused-swiglu-amax',action='store_true')
    p.add_argument('--fused-quant-pack',action='store_true')
    p.add_argument('--terminal-replay',action='store_true')
    p.add_argument('--fused-ffn',action='store_true')
    p.add_argument('--ffn-quant-tile',type=int,choices=[16,32,64],default=32)
    p.add_argument('--ffn-flat-forward',action='store_true')
    p.add_argument('--fused-residual-norm',action='store_true')
    p.add_argument('--fused-norm-backward',action='store_true')
    p.add_argument('--fused-scale-inverse',action='store_true')
    p.add_argument('--fused-residual-backward',action='store_true')
    p.add_argument('--fused-cast-amax',action='store_true')
    p.add_argument('--cache-weight-scales',action='store_true')
    p.add_argument('--attention-group-penalty',type=int,default=0)
    p.add_argument('--attention-local-maxima',action='store_true')
    p.add_argument('--selective-attention-min-seq',type=int,default=4096)
    p.add_argument('--no-packed-skip',action='store_true')
    p.add_argument('--arena-limit-gib',type=float,default=14.37)
    p.add_argument('--output',required=True)
    p.add_argument('--candidate-config')
    args=p.parse_args()
    if min(args.layers,args.steps,args.accumulation)<1 or args.warmup<0:p.error('invalid counts')
    if args.micro_graph and len(args.shapes)!=1:p.error('micro-graph screening currently requires one shape')
    if args.production_samples and args.updates:p.error('use bench_native_production for actual updates on production data')
    if args.profile_cupti and (args.profile_kernels or args.updates):p.error('CUPTI probe uses fixed weights and no event profiler')
    if args.profile_memory and (args.updates or args.micro_graph or args.profile_cupti or args.profile_kernels):
        p.error('memory diagnostics require a separate fixed-weight eager probe')
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
            self.cupti=None;self.gemm_label=None
            self.memory_active=False;self.memory_phase='setup';self.memory_peaks={};self.memory_samples=[]
            super().__init__(*a,**kw)
        def _submit(self,label,fn,stream=None):
            if self.memory_active:
                peak=self.memory_peaks.get(self.memory_phase)
                if peak and peak['live_bytes']>peak.get('sampled_live_bytes',0):
                    # Deliberate diagnostic barrier: resolve stream-ordered
                    # allocations before reading physical free memory. Never
                    # mix these windows into a throughput result.
                    snapshot=self.memory()
                    self.memory_samples.append(dict(phase=self.memory_phase,operation=label,
                        allocation=peak['allocation'],memory=snapshot,
                        device_used_bytes=snapshot['device_total_bytes']-snapshot['device_free_bytes']))
                    peak['sampled_live_bytes']=peak['live_bytes']
            profiler=self.cupti
            if profiler is None or not profiler.active:return super()._submit(label,fn,stream)
            selected=self.gemm_label if label.startswith('cublasLt:') and self.gemm_label else label
            key=profiler.push(selected);begin=time.perf_counter()
            try:return super()._submit(label,fn,stream)
            finally:
                elapsed=(time.perf_counter()-begin)*1000
                entry=profiler.host.setdefault(key,dict(calls=0,ms=0.));entry['calls']+=1;entry['ms']+=elapsed
                profiler.pop()
        def timed(self,label,call):
            if not self.profile_active:return call()
            start,end=self.create_event(timing=True),self.create_event(timing=True)
            self.record_event(start);begin=time.perf_counter()
            result=call()
            host_ms=(time.perf_counter()-begin)*1000
            self.record_event(end)
            self.kernel_events.append((label,start,end,host_ms))
            return result
        def control(self,label,call):
            if self.cupti is None or not self.cupti.active:return call()
            self.cupti.push('control:'+label)
            try:return call()
            finally:self.cupti.pop()
        def launch(self,fn,grid,values,**constants):
            call=super().launch
            return self.timed('kernel/'+getattr(fn,'__name__',str(fn)),lambda:call(fn,grid,values,**constants))
        def matmul(self,*values,**kwargs):
            call=super().matmul
            role=kwargs.get('role',values[8] if len(values)>8 else 'gemm')
            label=f'gemm/{role}/{values[0].dtype}/{values[5:8]}'
            old,self.gemm_label=self.gemm_label,label
            try:return self.timed(label,lambda:call(*values,**kwargs))
            finally:self.gemm_label=old
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
            arena=self._live_bytes+self._pool_bytes
            if arena+additional>args.arena_limit_gib*2**30:
                raise MemoryError(f'native arena guard: {(arena+additional)/2**30:.3f} GiB > {args.arena_limit_gib}')
            result=super().empty(shape,dtype,zero)
            self.peak_arena=max(self.peak_arena,self._live_bytes+self._pool_bytes)
            if self.memory_active:
                peak=self.memory_peaks.setdefault(self.memory_phase,dict(live_bytes=0))
                if self._live_bytes>peak['live_bytes']:
                    peak.update(live_bytes=self._live_bytes,arena_bytes=self._live_bytes+self._pool_bytes,
                                allocation=dict(shape=list(shape),dtype=dtype,bytes=needed))
            return result
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'trainer/config_instinct_v2.json').read_text())
    cfg.update(num_hidden_layers=args.layers,use_grad_checkpoint=args.mode,selective_cache_mib=args.cache_mib)
    if args.cache_by_seq:
        from model.dense_cublas.selective import parse_cache_budgets
        cfg['selective_cache_by_seq']=parse_cache_budgets(args.cache_by_seq)
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
    with BudgetBackend(config=ScheduleConfig(tune=args.tune,workspace_mib=args.workspace_mib,candidates=args.candidates,fp8_fast_accum_output=args.fp8_fast_accum_output),pool_limit_mib=args.pool_mib,allocator_backend=args.allocator_backend,async_release_mib=args.async_release_mib) as b:
        b.attention_tile=args.attention_tile
        b.pack_tile=args.pack_tile
        b.elementwise_block=args.elementwise_block
        b.split_kv_gradients=args.split_kv_gradients
        b.norm_groups=args.norm_groups
        b.norm_rope_rows=args.norm_rope_rows
        b.fused_norm_cast=args.fused_norm_cast
        b.fused_swiglu_amax=args.fused_swiglu_amax
        b.fused_quant_pack=args.fused_quant_pack
        b.terminal_replay=args.terminal_replay
        b.fused_ffn=args.fused_ffn
        b.ffn_quant_tile=args.ffn_quant_tile
        b.ffn_flat_forward=args.ffn_flat_forward
        b.fused_residual_norm=args.fused_residual_norm
        b.fused_norm_backward=args.fused_norm_backward
        b.fused_scale_inverse=args.fused_scale_inverse
        b.fused_residual_backward=args.fused_residual_backward
        b.fused_cast_amax=args.fused_cast_amax
        b.cache_weight_scales=args.cache_weight_scales
        b.attention_group_penalty=args.attention_group_penalty
        b.attention_local_maxima=args.attention_local_maxima
        b.selective_attention_min_seq=args.selective_attention_min_seq
        if args.norm_rope_warps:
            b.kernel_options['model.dense_cublas.fused.norm_rotary_forward']=dict(num_warps=args.norm_rope_warps)
        b.attention_forward_tile=args.attention_forward_tile
        b.attention_forward_cols=args.attention_forward_cols
        b.fast_attention=args.fast_attention
        b.compact_gradients=args.compact_gradients
        b.aligned_pointers=args.aligned_pointers or args.aligned_pointwise
        b.aligned_pointwise_only=args.aligned_pointwise
        b.async_attention_metadata=args.async_attention_metadata
        b.attention_plan_quantum=args.attention_plan_quantum
        b.attention_plan_limit=args.attention_plan_limit
        b.attention_docs_per_plan=args.attention_docs_per_plan
        b.pad_attention=args.pad_attention
        b.kernel_options['model.dense_triton.attention.forward']={
            **({'num_warps':args.attention_warps} if args.attention_warps else {}),
            **({'num_stages':args.attention_stages} if args.attention_stages else {})}
        b.kernel_options['model.dense_triton.attention.forward_fast']=b.kernel_options['model.dense_triton.attention.forward']
        b.packed_skip=not args.no_packed_skip
        from experiments.native_candidate import apply
        report['candidate_config']=apply(b,args.candidate_config)
        if args.native_sdpa:
            from experiments.native_sdpa.adapter import NativeAttention
            b.native_attention=NativeAttention(b)
            b.native_attention.minimum_seq=args.native_sdpa_min_seq
        e=Engine(b,fp8=bool(args.fp8),fp8_filter='auto',seed=307)
        model=Model(e,cfg,param_dtype='bf16',muon_mantissa=True,seed=307)
        opt=Optimizer(e,model.parameters(),'muon',lr=3e-4,weight_decay=.1,
                      muon_mantissa=True,muon_momentum_dtype='bf16',
                      muon_batch_size=16,muon_workspace_mb=args.muon_workspace_mib)
        batches=[];host_batches=[];rng=np.random.default_rng(307)
        real_chunk=None
        if args.production_samples:
            from model.dense_cublas.data import Tokenizer
            from model.dense_cublas.parquet_data import ParquetPretrain,PackedChunk
            tok=Tokenizer('D:/AI/Instinct/model')
            data=ParquetPretrain('D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet',tok,
                'cache/native_tokens',lengths=[3072,4096],token_budget=12288,threads=4)
            manifest=data.build(0);real_chunk=PackedChunk(data.cache_path(0),manifest,tok.pad)
            report['data_source']=data.plan['source']
        loaded=np.load(args.sample_file,allow_pickle=False) if args.sample_file else None
        for shape_index,shape in enumerate(args.shapes):
            batch,seq,documents=map(int,shape.split('x'))
            if not 1<=documents<=seq:p.error('invalid document count')
            ids=rng.integers(3,cfg['vocab_size'],(batch,seq),dtype=np.int64)
            segments=np.broadcast_to(np.arange(seq)*documents//seq,(batch,seq)).copy()
            labels=ids.copy()
            labels[:,1:][segments[:,1:]!=segments[:,:-1]]=-100
            if real_chunk is not None:
                bucket=next(i for i,v in enumerate(manifest['buckets']) if v['length']==seq)
                indices=np.random.default_rng(73).choice(manifest['buckets'][bucket]['blocks'],batch,replace=False)
                (ids,labels,segments),_,_=real_chunk.batch(bucket,indices)
                documents=2
            if loaded is not None:
                ids,labels,segments=[loaded[f'{key}{shape_index}'] for key in ('ids','labels','segments')]
                if ids.shape!=(batch,seq):raise ValueError('sample file shape does not match requested bucket')
                documents=2
            host_batches.append((ids,labels,segments) if documents>1 else (ids,labels))
            device_segments=e.array(segments) if documents>1 or args.native_sdpa else None
            if device_segments is not None:device_segments.host_array=segments
            batches.append((e.array(ids),e.array(labels),device_segments,
                            int(np.count_nonzero(labels[:,1:]!=-100))))
        if real_chunk is not None:real_chunk.close()
        if loaded is not None:loaded.close()
        if args.native_sdpa and getattr(b,'attention_cost_model',False):
            totals={values[0].shape[0]*values[0].shape[1] for values in host_batches}
            if len(totals)==1:b.native_attention.warm_catalog(totals.pop(),max(values[0].shape[1] for values in host_batches),
                cfg['num_attention_heads'],cfg['num_key_value_heads'],cfg['head_dim'])
        if args.export_samples:
            np.savez(args.export_samples,**{f'{key}{i}':array for i,values in enumerate(host_batches)
                for key,array in zip(('ids','labels','segments'),values)})
        if args.profile_cupti:
            from experiments.cupti_trace.collector import Collector
            b.cupti=Collector()
        report.update(initial_memory=b.memory(),arch=b.arch,parameter_count=sum(p.size for p in model.parameters(False)))
        graph=None
        if args.micro_graph:
            from model.dense_cublas.graph import MicrostepGraph
            graph=MicrostepGraph(model,opt,host_batches[0],lambda values:
                model(values[0],values[1],sequence_ids=values[2] if len(values)==3 else None,return_logits=False).loss)
            report['graph_memory']=b.memory()
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
                b.memory_active=args.profile_memory and window>=args.warmup
                b.memory_peaks={};b.memory_samples=[]
                b.profile_active=args.profile_kernels and window>=args.warmup
                collecting=args.profile_cupti and window>=args.warmup+max(0,args.steps-args.cupti_last_steps)
                if collecting:b.synchronize();b.cupti.start()
                b.synchronize();start=time.perf_counter();valid=0
                b._peak_live_bytes=b._live_bytes;b.peak_arena=sum(n for _,n in b._allocations)
                for _ in range(args.accumulation if args.updates else 1):
                    ids,labels,segments,count=batches[micro%len(batches)]
                    mark=b.mark()
                    if graph:
                        last_loss=graph.run(host_batches[0])
                        if not args.updates:opt.zero_grad()
                    else:
                        b.memory_phase='forward'
                        if b.cupti:b.cupti.phase='forward'
                        loss=model(ids,labels,sequence_ids=segments,return_logits=False).loss
                    if args.updates and not graph:opt.backward(loss)
                    elif not graph:
                        b.memory_phase='backward_including_replay'
                        if b.cupti:b.cupti.phase='backward'
                        def consume(param,grad):
                            e.copy(opt.gradients[param],e.wrap(grad))
                            return True
                        prior_consume=getattr(e,'consume_checkpoint_cache',False)
                        e.consume_checkpoint_cache=getattr(b,'release_checkpoint_cache',False)
                        try:e.vjp(loss,leaf_callback=consume)
                        finally:e.consume_checkpoint_cache=prior_consume
                    if not graph:last_loss=b.control('read_loss',lambda:e.item(loss))
                    micro+=1;valid+=count
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
                b.memory_active=False
                if collecting:
                    trace=target.with_suffix(f'.window{window}.cupti.jsonl')
                    report.setdefault('cupti_traces',[]).append(dict(window=window,bucket=(micro-1)%len(batches),path=b.cupti.stop(trace)))
                row=dict(window=window,warmup=window<args.warmup,total_ms=(end-start)*1000,
                         cupti_profiled=collecting,
                         memory_profiled=args.profile_memory and window>=args.warmup,
                         forward_backward_ms=(backward_end-start)*1000,optimizer_ms=(end-backward_end)*1000,
                         valid_tokens=valid,loss=last_loss,peak_live_gib=b._peak_live_bytes/2**30,
                         peak_arena_gib=b.peak_arena/2**30,memory=b.memory())
                if row['memory_profiled']:
                    row.update(memory_phase_peaks=b.memory_peaks,memory_samples=b.memory_samples,
                        memory_scope='Synchronized samples at new logical phase peaks; sampled device usage is a lower bound on the physical peak. CUDA pool high-water counters include warmup.')
                report['windows'].append(row);write();print(json.dumps(row),flush=True)
                if stop or pause.exists():break
            measured=[r for r in report['windows'] if not r['warmup'] and not r['cupti_profiled'] and not r['memory_profiled']]
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
            if args.native_sdpa:
                report['native_sdpa_calls']={'forward':b.native_attention.forward_calls,'backward':b.native_attention.backward_calls}
                report['attention_runtime']=dict(b.native_attention.metrics)
            report['driver_profile']=b.driver_profile
            if args.profile_kernels:report['kernel_profile']=b.kernel_report()
            write()


if __name__=='__main__':main()
