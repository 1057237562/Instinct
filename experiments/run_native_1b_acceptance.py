"""Full-corpus production training with safe pause and exact resume.

The file and output prefix retain their original names to preserve checkpoint
and monitor paths. The target is one full pass over the production corpus.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def runtime_cache_options(cache_mib=None,allocator_release_mib=None):
    """Execution overrides for this machine; leave the shared 16 GiB recipe intact."""
    flags=[];environment={}
    for value in (cache_mib,allocator_release_mib):
        if value is not None and value<0:raise ValueError('cache budgets must be nonnegative')
    if cache_mib is not None:
        flags += ['--selective_cache_mib',str(cache_mib),'--selective_cache_by_seq',
                  f'3072:{cache_mib},4096:{cache_mib}']
        environment['INSTINCT_NATIVE_SELECTIVE_CACHE_MIB']=str(cache_mib)
    if allocator_release_mib is not None:
        flags += ['--allocator_release_mib',str(allocator_release_mib)]
        environment['INSTINCT_NATIVE_ALLOCATOR_RELEASE_MIB']=str(allocator_release_mib)
    return flags,environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume-path', type=Path)
    p.add_argument('--data-path', type=Path)
    p.add_argument('--num-workers', type=int, default=int(os.environ.get('INSTINCT_NATIVE_PREPROCESS_THREADS', '4')))
    p.add_argument('--cache-mib',type=int,default=os.environ.get('INSTINCT_NATIVE_SELECTIVE_CACHE_MIB'))
    p.add_argument('--allocator-release-mib',type=int,default=os.environ.get('INSTINCT_NATIVE_ALLOCATOR_RELEASE_MIB'))
    p.add_argument('--merge-packed-pairs',action='store_true',
                   default=os.environ.get('INSTINCT_NATIVE_MERGE_PACKED_PAIRS')=='1')
    args = p.parse_args()
    cache_flags,cache_environment=runtime_cache_options(args.cache_mib,args.allocator_release_mib)
    if args.merge_packed_pairs:
        cache_environment['INSTINCT_NATIVE_MERGE_PACKED_PAIRS']='1'
    os.chdir(ROOT)
    from experiments.smoke_dense_triton import RejectTorch
    sys.meta_path.insert(0, RejectTorch())
    from scripts.native_v2_recipe import command
    from trainer.train_cublas import parser, train
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas import checkpoint
    from scripts.native_training_monitor import register, write_json
    import psutil

    name = 'native_v2_1b_acceptance'
    artifact_root = Path(os.environ.get('INSTINCT_ARTIFACT_ROOT', 'D:/AI/Instinct'))
    checkpoint_dir = artifact_root / 'checkpoints/native_cublas'
    weight_dir = artifact_root / 'out/native_cublas'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    resume = (args.resume_path or checkpoint_dir / (name + '_1536_resume.npz')).resolve()
    if not args.resume_path and resume.exists():
        raise FileExistsError('Existing acceptance checkpoint: use --resume-path to continue: ' + str(resume))
    stop = (ROOT / 'experiments/results/cublas/target18500/native_v2_1b.pause').resolve()
    status = stop.with_suffix('.status.json')
    manifest = checkpoint_dir / 'training_run.json'
    saved_run = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
    data_path = args.data_path or saved_run.get('data_path') or artifact_root / 'dataset/pretrain_coder_l2l3_24b.parquet'
    def publish_run(planned_tokens):
        register(manifest, runtime_root=ROOT, python=sys.executable,
                 status_path=status, stdout_path=stop.with_suffix('.stdout.log'),
                 stderr_path=stop.with_suffix('.stderr.log'), pause_path=stop,
                 resume_path=resume, planned_tokens=planned_tokens, data_path=data_path,
                 environment={key: os.environ[key] for key in ('CUDA_PATH', 'CUDA_HOME',
                     'INSTINCT_ARTIFACT_ROOT', 'INSTINCT_NATIVE_PARQUET_CACHE_GB',
                     'INSTINCT_CUDNN_FRONTEND_SOURCE', 'INSTINCT_CUTE_DSL_PATH',
                     'LD_LIBRARY_PATH', 'TRITON_CACHE_DIR', 'MALLOC_ARENA_MAX') if key in os.environ}
                     | {'INSTINCT_NATIVE_PREPROCESS_THREADS': str(args.num_workers)} | cache_environment)
    stop.parent.mkdir(parents=True, exist_ok=True)
    if stop.exists():raise FileExistsError('Pause marker exists: ' + str(stop))
    os.environ['INSTINCT_STOP_FILE'] = str(stop)
    extra = ['--tokenizer_path', str(artifact_root/'model'), '--num_workers', str(args.num_workers), '--save_dir', str(weight_dir),
             '--checkpoint_dir', str(checkpoint_dir), '--save_weight', name,
             '--resume_path', str(resume), '--save_interval', '2000', '--log_interval', '100', '--epochs', '1', '--async_checkpoint']
    if args.resume_path:extra += ['--from_resume', '1']
    extra += cache_flags
    cmd = command(data_path, extra=extra)
    opts = parser().parse_args(cmd[3:])

    def write(value):
        write_json(status, dict(value, resume_path=str(resume), updated_at=time.time(),
                               process_created=psutil.Process().create_time()))

    class Observer:
        def start(self, args, backend, model, optimizer, data, progress, *, rng):
            self.model, self.optimizer, self.progress, self.rng = model, optimizer, progress, rng
            self.planned_tokens = int(data.plan['tokens'])
            publish_run(self.planned_tokens)
            self.step(backend, model, optimizer, progress, None, 0, 0)

        def step(self, backend, model, optimizer, progress, batch, tokens, elapsed):
            seen = progress['processed_tokens']
            if batch is None or progress['steps'] // 80 > self.report_block:
                self.report_block = progress['steps'] // 80
                write(dict(status='running', pid=os.getpid(), target='full_corpus_one_epoch',
                           planned_source_tokens=self.planned_tokens,
                           processed_valid_tokens=seen, steps=progress['steps'],
                           updates=optimizer.updates, accumulated=optimizer.accumulated,
                           parquet_cursor=progress['parquet_cursor'], resume_path=str(resume),
                           paired_forwards=progress.get('paired_forwards',0),
                           batch_shape=list(batch[0].shape) if batch is not None else None,
                           batch_useful_tokens=tokens))

    observer = Observer()
    publish_run(23_999_989_035)
    write(dict(status='starting', pid=os.getpid(), target='full_corpus_one_epoch', resume_path=str(resume), command=cmd))
    with TrainingBackend(config=ScheduleConfig(workspace_mib=opts.workspace_mib, candidates=opts.candidates,
                         tune=opts.tune, overlap_dw=opts.overlap_dw, dw_sm_target=opts.dw_sm_target),
                         pool_limit_mib=opts.allocator_cache_mib, pin_memory=bool(opts.pin_memory),
                         allocator_backend=opts.allocator_backend, async_release_mib=opts.allocator_release_mib) as backend:
        try:
            result = train(opts, backend=backend, observer=observer, merge_packed_pairs=args.merge_packed_pairs)
            meta = checkpoint.metadata(resume)
            progress = meta['progress']
            assert progress['steps'] == result['steps'] and observer.optimizer.accumulated == 0
            assert checkpoint.metadata(result['weights'])['progress']['steps'] == progress['steps']
            if result.get('next_data_path'):
                data_path = result['next_data_path']
                publish_run(observer.planned_tokens)
            if stop.exists():stop.unlink()
            write(dict(status='paused' if result.get('paused') else 'completed', pid=os.getpid(),
                       target='full_corpus_one_epoch', planned_source_tokens=observer.planned_tokens,
                       processed_valid_tokens=progress['processed_tokens'],
                       steps=progress['steps'], updates=observer.optimizer.updates, accumulated=0,
                       parquet_cursor=progress['parquet_cursor'], result=result, resume_path=str(resume)))
            print('Saved and verified: ' + json.dumps(result, ensure_ascii=False), flush=True)
        except BaseException as error:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            if hasattr(observer, 'model'):
                try:
                    failure = checkpoint_dir / (name + '_failure.npz')
                    checkpoint.write(failure, checkpoint.snapshot(observer.model, observer.optimizer,
                        observer.progress | {'diagnostic_failure': repr(error), 'resume_valid': False}, observer.rng))
                    write(dict(status='failed_held', pid=os.getpid(), error=repr(error), failure_state=str(failure)))
                except BaseException as save_error:
                    write(dict(status='failed_held', pid=os.getpid(), error=repr(error), save_error=repr(save_error)))
                print('Training failed; process remains alive to preserve current progress.', flush=True)
                while True:time.sleep(30)
            write(dict(status='failed_before_training', pid=os.getpid(), error=repr(error)))
            raise


if __name__ == '__main__':main()
