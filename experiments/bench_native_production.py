"""Native production-data throughput, normal trainer save/stop, exact state audit.

No Torch/TorchAO imports. The window includes Rust preprocessing overlap,
mmap batch decoding, H2D, accumulated Muon updates, logs and periodic saves.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from experiments.smoke_dense_triton import RejectTorch

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--tokenizer', default='D:/AI/Instinct/model')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--start-group', type=int, default=0)
    parser.add_argument('--tail-batches', type=int, default=0)
    parser.add_argument('--warmup-updates', type=int, default=4)
    parser.add_argument('--updates', type=int, default=128)
    parser.add_argument('--pack-tile', type=int, default=32)
    parser.add_argument('--attention-forward-tile', type=int, default=128)
    parser.add_argument('--attention-forward-cols', type=int, default=64)
    parser.add_argument('--attention-warps', type=int, default=8)
    parser.add_argument('--compact-gradients', action='store_true')
    parser.add_argument('--native-sdpa-min-seq', type=int, default=0)
    parser.add_argument('--attention-backend',choices=['triton','cudnn','hybrid'])
    parser.add_argument('--muon-gemm-backend',choices=['triton','cublas'])
    parser.add_argument('--selective-cache-by-seq',default='')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--cache-mib', type=int, default=768)
    parser.add_argument('--async-release-mib', type=int)
    parser.add_argument('--allocator-cache-mib',type=int,default=256)
    parser.add_argument('--profile-stages',action='store_true')
    parser.add_argument('--aligned-pointers',action='store_true')
    parser.add_argument('--aligned-pointwise',action='store_true')
    parser.add_argument('--async-attention-metadata',action='store_true')
    parser.add_argument('--attention-plan-quantum',type=int,default=128)
    parser.add_argument('--attention-plan-limit',type=int,default=32)
    parser.add_argument('--attention-docs-per-plan',type=int,default=0)
    parser.add_argument('--fused-residual-norm',action='store_true')
    opts = parser.parse_args()
    if min(opts.start_group, opts.tail_batches) < 0 or min(opts.updates, opts.warmup_updates) < 1:
        parser.error('invalid sample or update counts')
    os.chdir(ROOT)
    sys.meta_path.insert(0, RejectTorch())
    target = opts.output.resolve(); target.parent.mkdir(parents=True, exist_ok=True)
    from model.dense_cublas.data import Tokenizer
    from model.dense_cublas.parquet_data import ParquetPretrain, PackedChunk
    if opts.prepare_only:
        data = ParquetPretrain(opts.data, Tokenizer(opts.tokenizer), 'cache/native_tokens',
                              lengths=[3072,4096], token_budget=12288, threads=4)
        manifest = data.build(opts.start_group)
        write(target, dict(status='prepared', manifest=manifest, torch_imported='torch' in sys.modules))
        return
    from scripts.native_v2_recipe import command
    from trainer.train_cublas import parser as trainer_parser, train
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas import checkpoint
    prefix = 'native_production_' + time.strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid())
    microsteps = (opts.warmup_updates + opts.updates) * 8
    cmd = command(opts.data, extra=['--tokenizer_path', opts.tokenizer, '--sequence_buckets', '3072,4096',
        '--num_workers', '4', '--copy_prefetch', '--learning_rate', '0.0005', '--epochs', '1',
        '--save_interval', str(opts.save_interval), '--log_interval', '100',
        '--save_dir', 'D:/AI/Instinct/out/native_cublas', '--checkpoint_dir', 'D:/AI/Instinct/checkpoints/native_cublas',
        '--max_steps', str(microsteps), '--save_weight', prefix, '--selective_cache_mib', str(opts.cache_mib),
        '--allocator_cache_mib',str(opts.allocator_cache_mib)])
    args = trainer_parser().parse_args(cmd[3:])
    if opts.native_sdpa_min_seq and opts.attention_backend:
        parser.error('use either legacy SDPA threshold or the public attention backend option')
    extra=[]
    for key,value in (('attention_backend',opts.attention_backend),('muon_gemm_backend',opts.muon_gemm_backend),
                      ('selective_cache_by_seq',opts.selective_cache_by_seq)):
        if value is not None and value!='':extra.extend(['--'+key,str(value)])
    if extra:
        cmd.extend(extra);args=trainer_parser().parse_args(cmd[3:])
    os.environ['INSTINCT_STOP_FILE'] = str(target.with_suffix('.pause'))
    if Path(os.environ['INSTINCT_STOP_FILE']).exists(): parser.error('pause marker already exists')
    report = dict(status='running', command=cmd, benchmark_options=vars(opts)|{'output':str(target)},
                  windows=[], prefix=prefix, source_path=str(Path(opts.data).resolve()))
    launch = time.perf_counter()
    write(target, report)

    class Observer:
        def start(self, args, backend, model, optimizer, data, progress, *, rng):
            self.args,self.model,self.optimizer,self.data,self.progress = args,model,optimizer,data,progress
            self.rng=rng
            self.window_start=None;self.pending=[];self.micro=0
            self.last_optimizer_events=None
            if opts.profile_stages:
                original_step=optimizer.step
                def measured_step(*values,**keywords):
                    start,end=backend.create_event(timing=True),backend.create_event(timing=True)
                    backend.record_event(start)
                    result=original_step(*values,**keywords)
                    backend.record_event(end)
                    self.last_optimizer_events=start,end
                    return result
                optimizer.step=measured_step
            report.update(source=data.plan, model_config=model.config,
                          parameters=sum(p.size for p in model.parameters(False)),
                          sampling=dict(start_group=opts.start_group,tail_batches=opts.tail_batches,
                                        omitted_prefix_trained=False))
            if opts.start_group or opts.tail_batches:
                manifest=data.build(opts.start_group)
                chunk=PackedChunk(data.cache_path(opts.start_group),manifest,data.tokenizer.pad)
                try:
                    count=len(chunk.order(data.token_budget,data.seed,0,opts.start_group))
                finally:chunk.close()
                skip=max(0,count-opts.tail_batches) if opts.tail_batches else 0
                progress['parquet_cursor']=dict(group=opts.start_group,batch=skip)
                report['sampling']['skipped_batches_in_first_group']=skip
            report['setup_seconds']=time.perf_counter()-launch
            write(target, report)

        def step(self, backend, model, optimizer, progress, batch, tokens, elapsed):
            self.micro+=1
            self.pending.append(dict(shape=list(batch[0].shape), useful_tokens=tokens,
                                     cursor=dict(progress.get('parquet_cursor',{}))))
            if optimizer.accumulated: return
            backend.synchronize()
            now=time.perf_counter()
            if self.micro < opts.warmup_updates*args.accumulation_steps:
                self.pending=[];return
            if self.window_start is None:
                report['warmup_seconds']=now-launch
                self.window_start=now;self.pending=[]
                backend._peak_live_bytes=backend._live_bytes
                return
            useful=sum(b['useful_tokens'] for b in self.pending)
            row=dict(step=progress['steps'],seconds=now-self.window_start,useful_tokens=useful,
                     tokens_per_second=useful/(now-self.window_start),
                     shapes=dict(Counter(str(b['shape']) for b in self.pending)),
                     cursor=dict(progress.get('parquet_cursor',{})),
                     end_seconds=now-launch,memory=backend.memory())
            self.pending=[];self.window_start=now
            if self.last_optimizer_events is not None:
                import ctypes as ct
                elapsed=ct.c_float()
                backend._elapsed(ct.byref(elapsed),*self.last_optimizer_events)
                row['optimizer_ms']=elapsed.value
            report['windows'].append(row)
            print('[Native production] '+json.dumps(row),flush=True)
            write(target,report)

    observer=Observer()
    with TrainingBackend(config=ScheduleConfig(tune=opts.tune),pool_limit_mib=args.allocator_cache_mib,
                         allocator_backend=args.allocator_backend,async_release_mib=opts.async_release_mib) as backend:
        backend.pack_tile=opts.pack_tile
        backend.aligned_pointers=opts.aligned_pointers or opts.aligned_pointwise
        backend.aligned_pointwise_only=opts.aligned_pointwise
        backend.async_attention_metadata=opts.async_attention_metadata
        backend.attention_plan_quantum=opts.attention_plan_quantum
        backend.attention_plan_limit=opts.attention_plan_limit
        backend.attention_docs_per_plan=opts.attention_docs_per_plan
        backend.fused_residual_norm=opts.fused_residual_norm
        backend.attention_tile=32
        backend.attention_forward_tile=opts.attention_forward_tile
        backend.attention_forward_cols=opts.attention_forward_cols
        backend.compact_gradients=opts.compact_gradients
        backend.kernel_options['model.dense_triton.attention.forward']=dict(num_warps=opts.attention_warps,num_stages=2)
        if opts.native_sdpa_min_seq:
            from experiments.native_sdpa.adapter import NativeAttention
            backend.native_attention=NativeAttention(backend)
            backend.native_attention.minimum_seq=opts.native_sdpa_min_seq
        try:
            result=train(args,backend=backend,observer=observer)
            report['save_and_loader_close_seconds']=time.perf_counter()-launch
            if getattr(backend,'native_attention',None) is not None:
                report['native_sdpa_calls']=dict(forward=backend.native_attention.forward_calls,backward=backend.native_attention.backward_calls)
                report['attention_runtime']=backend.native_attention.metrics
            report['algorithms']=backend.tuning_report()
            actual=checkpoint.snapshot(observer.model,observer.optimizer,observer.progress,observer.rng)
            with np.load(result['resume'],allow_pickle=False) as saved:
                metadata=json.loads(str(saved['__metadata__']))
                current=json.loads(str(actual['__metadata__']))
                assert current==metadata
                for key,value in actual.items():
                    if key!='__metadata__':np.testing.assert_array_equal(value,saved[key],err_msg=key)
            with np.load(result['weights'],allow_pickle=False) as weights:
                for key,value in actual.items():
                    if key.startswith('model/'):np.testing.assert_array_equal(value,weights[key],err_msg=key)
            assert observer.optimizer.accumulated==0
            elapsed=sum(w['seconds'] for w in report['windows'])
            useful=sum(w['useful_tokens'] for w in report['windows'])
            report.update(status='paused_verified' if result.get('paused') else 'complete',
                result=result,verified_current_state=True,completed_requested_windows=len(report['windows'])==opts.updates,
                measured=dict(seconds=elapsed,useful_tokens=useful,tokens_per_second=useful/elapsed if elapsed else None),
                preprocessing=observer.data.events,torch_imported='torch' in sys.modules,
                total_seconds=time.perf_counter()-launch)
            write(target,report)
        except BaseException as error:
            report.update(status='failed',error=repr(error))
            if hasattr(observer,'model'):
                # Preserve current state even on unexpected diagnostic failure.
                # A partial failing backward/update is forensic, not an accepted resume.
                failure=Path(args.checkpoint_dir)/(prefix+'_failure.npz')
                try:
                    state=checkpoint.snapshot(observer.model,observer.optimizer,
                        observer.progress|{'diagnostic_failure':repr(error),'resume_valid':False},observer.rng)
                    checkpoint.write(failure,state)
                    with np.load(failure,allow_pickle=False) as saved:
                        for key,value in state.items():np.testing.assert_array_equal(value,saved[key])
                    report.update(failure_state=str(failure),failure_state_verified=True)
                except BaseException as save_error:
                    report['save_error']=repr(save_error);write(target,report)
                    print('Failed to preserve native state; process kept alive for recovery.',flush=True)
                    while True:time.sleep(30)
            write(target,report)
            raise


if __name__=='__main__':main()
