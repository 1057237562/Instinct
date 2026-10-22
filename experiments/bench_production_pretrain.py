"""Bounded run of the real pretrain entry point, with streaming and Arrow intact.

Only instrumentation and the stopping boundary differ from the listed CLI.
No synthetic/replayed tensors, changed samplers or model substitutions.
Optional --resident-data stages the exact bounded batch sequence before training
to isolate loader and host-to-device overhead; it is a diagnostic upper bound.
Optional --threads is an explicitly recorded CPU scheduling experiment.
Finish after warmup + measured updates, before the trainer's
epoch-final checkpoint. Ordinary periodic logging/saving remains enabled.

Launch-record runs block checkpoint writes. --production-timing keeps the
trainer profiler off and excludes pre-run evaluation. The explicit budget-test
mode restarts only the in-memory data cursor with checkpoint weights/optimizer;
it is a feasibility experiment, not an exact resume at a different budget.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache  # configure before torch
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1]
                                     / '.cache' / 'huggingface'))
import datasets  # Windows DLL order
import torch
import trainer.training_profiler as profiler_module
import trainer.trainer_utils as trainer_utils
from trainer.streaming_pretrain import ChunkedPackedEpochLoader


class BenchmarkComplete(BaseException):
    pass


def describe(values):
    ordered = sorted(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                p95=ordered[min(len(values) - 1, int(.95 * len(values)))],
                maximum=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--launch-record', type=Path,
                        help='Captured production command/environment JSON, preserving its model and recipe')
    parser.add_argument('--resume-checkpoint', type=Path,
                        help='Read an immutable full checkpoint; all checkpoint writes are blocked')
    parser.add_argument('--cold-attention-probe', action='store_true',
                        help='Reference only: reproduce optional FA4 import inside first compiled forward')
    parser.add_argument('--validation-cache', type=Path,
                        help='Packed source-chunk-2 cache for held-out validation')
    parser.add_argument('--validation-count', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--data-path', default='dataset/pretrain_coder_12b.parquet')
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/results/top2_production.json'))
    parser.add_argument('--memory-gb', type=float, default=15.5)
    parser.add_argument('--token-budget', type=int, default=0,
                        help='Explicit production bucket token budget; changes batch sizes and training trajectory')
    parser.add_argument('--restart-data-for-budget-test', action='store_true',
                        help='Isolated budget feasibility test: keep checkpoint weights/optimizer, restart the data cursor; never writes a training checkpoint')
    parser.add_argument('--check-finite-loss', action='store_true',
                        help='Retain detached scalar losses and check them after the timed window')
    parser.add_argument('--checkpoint-mode', type=int, choices=[1,2], default=None,
                        help='Keep checkpointing enabled: FFN selective (1) or whole block (2)')
    parser.add_argument('--selective-save-attention', action='store_true',
                        help='Experimental: cache FlexAttention results inside full-layer checkpoints')
    parser.add_argument('--prefetch', type=int, choices=[0, 1], default=1)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--trace-last-step', action='store_true',
                        help='Diagnostic only: profiler overhead affects the reported window')
    parser.add_argument('--production-timing', action='store_true',
                        help='Keep profile=off; synchronize only at warm-window boundaries, without evaluation or per-phase CUDA events')
    parser.add_argument('--resident-data', choices=('off', 'cpu', 'cuda'), default='off',
                        help='Diagnostic only: stage the exact bounded batches in CPU/GPU memory before warmup, preserving their order and CPU bookkeeping')
    parser.add_argument('--quality', action='store_true',
                        help='Record training losses and evaluate 32 fixed second-chunk sequences before/after')
    parser.add_argument('--quality-batch-change', action='store_true',
                        help='Use equal-update non-regression gates for an authorized batch/packing change')
    options = parser.parse_args()
    if options.production_timing and (options.quality or options.trace_last_step):
        parser.error('production-timing excludes quality evaluation and operator tracing; run those separately')
    if options.resident_data != 'off' and not options.production_timing:
        parser.error('resident-data requires production-timing for an unprofiled comparison')
    launch = json.loads(options.launch_record.read_text(encoding='utf-8')) if options.launch_record else None
    if not launch and options.checkpoint_mode is None:
        options.checkpoint_mode = 1
    if options.selective_save_attention and (not launch or options.checkpoint_mode == 1):
        parser.error('selective-save-attention requires a full-layer-checkpoint launch record')
    if options.resume_checkpoint and not launch:
        parser.error('resume-checkpoint requires launch-record')
    if options.restart_data_for_budget_test and not (launch and options.resume_checkpoint and options.token_budget > 0):
        parser.error('restart-data-for-budget-test requires launch-record, resume-checkpoint and a positive token-budget')
    if launch and options.resume_checkpoint and options.token_budget > 0 and not options.restart_data_for_budget_test:
        parser.error('a different resume budget requires the explicit isolated restart-data-for-budget-test mode')
    if launch and options.quality and not options.validation_cache:
        parser.error('launch-record quality validation requires its held-out validation-cache')
    if options.validation_count < 1:
        parser.error('validation-count must be positive')
    if launch:
        for key, value in launch.get('environment', {}).items():
            if key.startswith(('INSTINCT_', 'TORCH', 'PYTORCH_', 'TRITON_')) or key in ('PYTHONUTF8', 'HF_HOME'):
                os.environ.setdefault(key, value)
        options.data_path = launch['command'][launch['command'].index('--data_path') + 1]
    if options.steps < 1 or options.warmup < 1:
        parser.error('positive steps and warmup required')
    if not torch.cuda.is_available():
        raise RuntimeError('Local CUDA required')
    options.output.parent.mkdir(parents=True, exist_ok=True)
    source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in map(Path,[
        'model/moe_dispatch.py','model/moe_routing_triton.py','model/grouped_mm.py',
        'model/grouped_mm_triton.py','model/checkpointing.py','model/packed_attention.py',
        'model/flash_attn_4.py','trainer/trainer_utils.py','trainer/train_pretrain.py',
        'trainer/batched_muon.py','trainer/muon_kernels.py','model/model_instinct.py',
        'model/checkpoint_swiglu_triton.py','model/expert_linear_triton.py',
        'scripts/data_loader/sequence_bucket.py','trainer/packing_transition.py','trainer/trainer_cli.py',
        'trainer/windows_gpu_memory.py'])}
    corpus = Path(options.data_path).resolve()
    # Bind the chosen container to its compiler audit and original corpus.
    compiler_audit = json.loads(corpus.with_suffix('.parquet.report.json').read_text())
    corpus_audit = json.loads(corpus.with_suffix('.report.json').read_text())
    cached_source = {}
    if launch and '--autotune_profile' in launch['command']:
        profile_path = launch['command'][launch['command'].index('--autotune_profile') + 1]
        cached_source = json.loads(Path(profile_path).read_text(encoding='utf-8')).get('source', {})
    source_stat = corpus.stat()
    if (cached_source.get('size') == source_stat.st_size
            and cached_source.get('mtime_ns') == source_stat.st_mtime_ns):
        source_hash = cached_source['sha256']
        source_verification = 'reuse verified launch-profile SHA-256 with unchanged size/mtime and compiler audit'
    else:
        with corpus.open('rb') as handle:
            source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        source_verification = 'fresh SHA-256 and compiler audit'
    if source_hash != compiler_audit['output']['sha256']:
        raise RuntimeError('Parquet audit SHA-256 mismatch')
    print(f'[BENCH] audited container={corpus}, sha256={source_hash}', flush=True)
    started = time.perf_counter()
    wall_clock_offset = time.time() - started
    instances = []
    observed = []
    samples = []
    losses = []
    quality = {}
    batch_digest = hashlib.sha256()
    clock = {}
    windows_memory = {}
    checkpoint_origin = {}
    resident_batches = []
    resident_prepared = {}
    resident_report = dict(mode=options.resident_data)

    def record_windows_memory(label):
        if os.name == 'nt':
            from trainer.windows_gpu_memory import query_budgets
            windows_memory[label] = query_budgets()
    original_profiler = profiler_module.TrainingProfiler
    original_iterator = ChunkedPackedEpochLoader.__iter__
    original_checkpoint = trainer_utils.lm_checkpoint
    original_prepare_batch = trainer_utils.prepare_lm_batch

    def prepare_resident_batch(batch, device):
        if options.resident_data == 'cuda':
            return resident_prepared[id(batch)]
        return original_prepare_batch(batch, device)
    if options.selective_save_attention:
        os.environ['INSTINCT_DENSE_SELECTIVE_CHECKPOINT'] = 'attention'
    import importlib
    flash_module = importlib.import_module('model.flash_attn_4')
    original_resolve = flash_module.resolve_flash_attn
    if options.cold_attention_probe:
        flash_module.resolve_flash_attn = lambda: None
        flash_module._tried_import = False
        flash_module._flash_attn_4_func = None

    def isolated_checkpoint(*args, **kwargs):
        if kwargs.get('model') is not None or len(args) > 2:
            raise RuntimeError('Benchmark checkpoint writes are forbidden')
        state = torch.load(options.resume_checkpoint, map_location='cpu', weights_only=False)
        if state.get('world_size', 1) != 1:
            raise ValueError('Benchmark resume requires a single-GPU checkpoint')
        if options.restart_data_for_budget_test:
            checkpoint_origin.update(epoch=state.get('epoch', 0), step=state.get('step', 0),
                                     data_config=dict(state.get('data_config') or {}))
            state['epoch'] = state['step'] = 0
            state['data_config'] = dict(state.get('data_config') or {})
            state['data_config'].update(bucket_token_budget=options.token_budget,
                                       streaming_chunk_index=0, streaming_chunk_step=0,
                                       streaming_processed_tokens=0)
        return state

    class RecordingProfiler(original_profiler):
        def __init__(self, args, **kwargs):
            if options.threads:
                torch.set_num_threads(options.threads)
            self.windows = []
            self.production_globals = sys._getframe(1).f_globals
            super().__init__(args, **kwargs)
            instances.append(self)

        @contextmanager
        def phase(self, name):
            if options.trace_last_step:
                with torch.profiler.record_function(f'train::{name}'):
                    with super().phase(name):
                        yield
            else:
                with super().phase(name):
                    yield

        def end_step(self):
            if options.quality:
                losses.append((sys._getframe(1).f_locals['loss'].detach()
                               * self.production_globals['args'].accumulation_steps).clone())
            elif options.check_finite_loss:
                # Scalar storage only: no CUDA event, clone, item or reduction
                # enters the measured production loop.
                losses.append(sys._getframe(1).f_locals['loss'].detach())
            sample = self._current
            result = super().end_step()
            if sample is not None:
                samples.append(sample)
            return result

        def _report(self):
            result = super()._report()
            if result:
                self.windows.append(result)
            return result

    def validate_model(label):
        from scripts.data_loader.lm_dataset import PretrainDataset
        from trainer.trainer_utils import prepare_lm_batch
        namespace = instances[0].production_globals
        raw = getattr(namespace['model'], '_orig_mod', namespace['model'])
        if 'validation_batches' not in quality:
            cache = options.validation_cache or Path('.cache/huggingface/datasets/instinct-packed/'
                         'aabc28cfe809fb49f5620398e5d3cd8b30df144ec4e90b4502cdcd3cb3bf35a6')
            manifest = json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
            assert manifest['identity']['source'] == source_hash
            packed = datasets.concatenate_datasets([
                datasets.Dataset.from_file(str(cache/entry['name'])) for entry in manifest['files']])
            dataset = PretrainDataset.__new__(PretrainDataset)
            dataset.samples = packed
            dataset.tokenizer = namespace['tokenizer']
            indices = []
            small = max(1, options.validation_count * 3 // 4)
            for bucket, count in zip(manifest['bucket_ranges'], (small, options.validation_count-small)):
                indices.extend(int(bucket['start']) + 97 + 997*i for i in range(count))
            quality['validation_batches'] = [tuple(t.unsqueeze(0) for t in dataset[i]) for i in indices]
            quality['validation_indices'] = indices
            quality['validation_cache'] = str(cache.resolve())
            quality['validation_scope'] = f'{len(indices)} fixed sequences from source chunk 2; training remains in source chunk 1'
            h=hashlib.sha256()
            for name, parameter in raw.named_parameters():
                h.update(name.encode()); h.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            quality['initial_parameter_sha256'] = h.hexdigest()
        raw.eval()
        metrics=[]
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            for batch in quality['validation_batches']:
                inputs, targets, ids = prepare_lm_batch(batch,'cuda')
                result=raw(inputs,labels=targets,sequence_ids=ids)
                metrics.append(dict(loss=float(result.loss),tokens=int((targets[:,1:]!=-100).sum())))
        raw.train()
        quality[label]=dict(examples=metrics,
            token_weighted_loss=sum(r['loss']*r['tokens'] for r in metrics)/sum(r['tokens'] for r in metrics))
        print(f'[QUALITY] {label} validation NLL={quality[label]["token_weighted_loss"]:.8f}',flush=True)

    def measured_iterator(loader):
        iterator = original_iterator(loader)
        if options.resident_data != 'off':
            staging_start = time.perf_counter()
            staging_digest = hashlib.sha256()
            for _ in range(options.warmup + options.steps):
                batch = next(iterator)
                resident_batches.append(batch)
                for tensor in batch:
                    staging_digest.update(str(tuple(tensor.shape)).encode())
                    staging_digest.update(tensor.numpy().tobytes())
                if options.resident_data == 'cuda':
                    resident_prepared[id(batch)] = original_prepare_batch(
                        batch, instances[0].production_globals['args'].device)
            torch.cuda.synchronize()
            resident_report.update(
                staging_seconds=time.perf_counter() - staging_start,
                initial_batch_sha256=staging_digest.hexdigest(),
                batches=len(resident_batches),
                cpu_bytes=sum(t.numel() * t.element_size()
                              for batch in resident_batches for t in batch),
                cuda_bytes=sum(t.numel() * t.element_size()
                               for batch in resident_prepared.values()
                               for t in batch if t is not None))
            print('[BENCH RESIDENT] ' + json.dumps(resident_report), flush=True)
        for index in range(options.warmup + options.steps):
            if index == options.warmup:
                torch.cuda.synchronize()
                if options.production_timing:
                    torch.cuda.reset_peak_memory_stats()
                record_windows_memory('warmup_end')
                print(f'[BENCH] warmup complete; measuring {options.steps} steps; '
                      f'resident_data={options.resident_data}', flush=True)
                clock['steady_start'] = time.perf_counter()
            fetch_start = time.perf_counter()
            batch = resident_batches[index] if resident_batches else next(iterator)
            fetch_end = time.perf_counter()
            if options.quality:
                for tensor in batch:
                    batch_digest.update(str(tuple(tensor.shape)).encode())
                    batch_digest.update(tensor.numpy().tobytes())
            if not observed:
                clock['first_batch_ready'] = fetch_end
                namespace = instances[0].production_globals
                config = namespace['lm_config']
                assert config.num_hidden_layers == 32
                if not launch:
                    assert config.num_experts_per_tok == 2 and config.norm_topk_prob
                    assert config.use_grad_checkpoint == options.checkpoint_mode
                if options.quality:
                    validate_model('initial')
                    if options.cold_attention_probe:
                        flash_module._tried_import = False
                        flash_module._flash_attn_4_func = None
            record = dict(step=index + 1, batch_size=int(batch[0].shape[0]),
                          sequence_length=int(batch[0].shape[1]),
                          loader_next_ms=1000 * (fetch_end - fetch_start))
            observed.append(record)
            # Counts/hashes are collected after the timed run, outside this
            # iterator, to avoid adding CPU work to the production hot path.
            if options.trace_last_step and index == options.warmup + options.steps - 1:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True) as trace:
                    yield batch
                    torch.cuda.synchronize()
                trace.export_chrome_trace(str(options.output.with_suffix('.trace.json')))
                options.output.with_suffix('.operators.txt').write_text(
                    trace.key_averages().table(sort_by='self_device_time_total', row_limit=60),
                    encoding='utf-8')
            else:
                yield batch
            record['cpu_loop_body_ms'] = 1000 * (time.perf_counter() - fetch_end)
            if index == 0:
                if not options.production_timing:
                    torch.cuda.synchronize()
                clock['first_step_done'] = time.perf_counter()
            if index % 10 == 9 and not options.production_timing:
                print(f'[BENCH] completed={index + 1}', flush=True)
        torch.cuda.synchronize()
        clock['steady_end'] = time.perf_counter()
        record_windows_memory('timed_window_end')
        if resident_batches:
            digest = hashlib.sha256()
            for batch in resident_batches:
                for tensor in batch:
                    digest.update(str(tuple(tensor.shape)).encode())
                    digest.update(tensor.numpy().tobytes())
                if options.resident_data == 'cuda':
                    expected_ids = batch[2] if len(batch) == 3 and torch.any(batch[2] > 0).item() else None
                    for expected, actual in zip((*batch[:2], expected_ids), resident_prepared[id(batch)]):
                        if expected is None:
                            assert actual is None
                        else:
                            assert torch.equal(expected, actual.cpu()), 'Resident input changed during training'
            assert digest.hexdigest() == resident_report['initial_batch_sha256']
            resident_report.update(batch_sha256=digest.hexdigest(), inputs_unchanged=True)
        if options.quality:
            validate_model('final')
            del quality['validation_batches']
            quality['training_losses']=[float(loss) for loss in losses]
            quality['training_batch_sha256']=batch_digest.hexdigest()
            quality['predeclared_acceptance']={
                'maximum_20_step_mean_loss_relative':0.02,
                'final_validation_nll_relative':0.01,
                'same_initial_weights_and_training_batches':True,
                'all_losses_finite':True,
            }
            if options.quality_batch_change:
                quality['predeclared_acceptance']={
                    'maximum_20_step_mean_loss_increase_relative':0.02,
                    'final_validation_nll_increase_relative':0.01,
                    'same_initial_weights_and_validation_examples':True,
                    'same_training_batches_required':False,
                    'equal_update_count_not_equal_tokens':True,
                    'all_losses_finite':True,
                }
        # Prevent production epoch-end saves; no partial formal training output.
        raise BenchmarkComplete()

    argv = [
        'trainer/train_pretrain.py', '--config_path', 'trainer/config_pretrain.json',
        '--hidden_size', '512', '--num_hidden_layers', '32', '--use_moe', '1',
        '--from_weight', 'none', '--from_resume', '0',
        '--data_path', str(corpus), '--batch_size', '1', '--epochs', '1',
        '--max_seq_len', '4096', '--sequence_packing', '1',
        '--sequence_packing_mode', 'bucket', '--seq_bucket', '2',
        '--bucket_gpu_memory_gb', str(options.memory_gb),
        '--bucket_token_budget', str(options.token_budget),
        '--bucket_max_seq_len', '4096', '--bucket_large_threshold', '4096',
        '--packing_batch_size', '1000', '--packing_num_proc', '4',
        '--bucket_loader_workers', '0', '--num_workers', '0',
        '--data_cache_max_gb', '9', '--dataset_streaming', 'auto',
        '--streaming_chunk_mb', '1024', '--streaming_prefetch_chunks', str(options.prefetch),
        '--cache_build_mode', 'inline', '--optimizer', 'muon',
        '--dtype', 'bfloat16', '--param_dtype', 'fp32', '--kv_cache_dtype', 'fp8_e5m2',
        '--fp8_training', 'tensorwise', '--fp8_filter', 'auto',
        '--use_grad_checkpoint', str(options.checkpoint_mode), '--use_compile', '1',
        '--compile_mode', 'max-autotune-no-cudagraphs', '--accumulation_steps', '1',
        '--learning_rate', '0.0005', '--warmup_ratio', '0', '--min_lr_ratio', '0.1',
        '--log_interval', '100', '--save_interval', '1000',
        '--profile', 'timing', '--profile_warmup', str(options.warmup),
        '--profile_interval', str(options.steps),
        '--save_weight', 'bench_top2_production',
        '--save_dir', str(options.output.parent / 'isolated_weights'),
        '--pause_file', str(options.output.parent / 'nonexistent_benchmark_pause'),
    ]
    if launch:
        command = launch['command']
        script = next(i for i, value in enumerate(command) if Path(value).name == 'train_pretrain.py')
        argv = command[script:]

        def option(name, value=None):
            while name in argv:
                index = argv.index(name)
                del argv[index:index + 2]
            if value is not None:
                argv.extend([name, str(value)])

        if '--autotune_profile' in argv:
            profile_path = argv[argv.index('--autotune_profile') + 1]
            profile = json.loads(Path(profile_path).read_text(encoding='utf-8'))
            option('--autotune_profile')
            for key, value in profile['settings']['args'].items():
                option('--' + key, value)
            for key, value in profile['settings']['env'].items():
                os.environ.setdefault(key, value)
        for key, value in {
            '--from_resume': int(options.resume_checkpoint is not None),
            '--profile': 'off' if options.production_timing else 'timing', '--profile_warmup': options.warmup,
            '--profile_interval': options.steps, '--save_interval': 2147483647,
            '--save_dir': options.output.parent / 'isolated_weights',
            '--pause_file': options.output.parent / 'nonexistent_benchmark_pause',
        }.items():
            option(key, value)
        if options.checkpoint_mode is not None:
            option('--use_grad_checkpoint', options.checkpoint_mode)
        if options.token_budget > 0:
            option('--bucket_token_budget', options.token_budget)
    elif options.production_timing:
        argv[argv.index('--profile') + 1] = 'off'
    previous_argv = sys.argv
    profiler_module.TrainingProfiler = RecordingProfiler
    ChunkedPackedEpochLoader.__iter__ = measured_iterator
    if options.resume_checkpoint:
        trainer_utils.lm_checkpoint = isolated_checkpoint
    if options.resident_data == 'cuda':
        trainer_utils.prepare_lm_batch = prepare_resident_batch
    sys.argv = argv
    try:
        runpy.run_path('trainer/train_pretrain.py', run_name='__main__')
        raise RuntimeError('Production trainer ended before the benchmark boundary')
    except BenchmarkComplete:
        pass
    finally:
        sys.argv = previous_argv
        profiler_module.TrainingProfiler = original_profiler
        ChunkedPackedEpochLoader.__iter__ = original_iterator
        trainer_utils.lm_checkpoint = original_checkpoint
        trainer_utils.prepare_lm_batch = original_prepare_batch
        flash_module.resolve_flash_attn = original_resolve
    instance = instances[0]
    instance.finish()
    namespace = instance.production_globals
    measured = observed[options.warmup:]
    measured_samples = samples[options.warmup:]
    seconds = clock['steady_end'] - clock['steady_start']
    loss_check = None
    if options.check_finite_loss:
        factor = 1 if options.quality else namespace['args'].accumulation_steps
        checked_losses = [float(value) * factor for value in losses]
        loss_check = dict(count=len(checked_losses), all_finite=all(map(math.isfinite, checked_losses)),
                          last_loss=checked_losses[-1], minimum_loss=min(checked_losses),
                          maximum_loss=max(checked_losses))
    phase_ms = {}
    for record, sample in zip(measured, measured_samples):
        record['gpu_step_ms'] = sample.total_start.elapsed_time(sample.total_end)
        record['gpu_phase_ms'] = {}
        for name, markers in sample.phases.items():
            elapsed = sum(
                start.elapsed_time(end) for start, end in markers)
            record['gpu_phase_ms'][name] = elapsed
            phase_ms[name] = phase_ms.get(name, 0) + elapsed
    by_shape = []
    for shape, count in Counter((item['batch_size'], item['sequence_length'])
                                for item in measured).items():
        subset = [(item, sample) for item, sample in zip(measured, measured_samples)
                  if (item['batch_size'], item['sequence_length']) == shape]
        shape_observed = [item for item in measured
                          if (item['batch_size'], item['sequence_length']) == shape]
        entry = dict(batch_size=shape[0], sequence_length=shape[1], steps=count,
                     loader_next_ms=describe([item['loader_next_ms'] for item in shape_observed]))
        if subset:
            entry['gpu_step_ms'] = describe([sample.total_start.elapsed_time(sample.total_end)
                                            for _, sample in subset])
        by_shape.append(entry)
    # Detect finite losses without an extra synchronization in the hot path.
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  steps=options.steps, warmup=options.warmup,
                  wall_seconds=seconds, wall_ms_per_step=1000 * seconds / options.steps,
                  measured_start_unix=clock['steady_start'] + wall_clock_offset,
                  measured_end_unix=clock['steady_end'] + wall_clock_offset,
                  startup_to_first_batch_seconds=clock['first_batch_ready'] - started,
                  first_step_seconds=clock['first_step_done'] - clock['first_batch_ready'],
                  start_to_measured_window_end_seconds=clock['steady_end'] - started,
                  phase_ms_per_step={key: value / options.steps for key, value in phase_ms.items()},
                  loader_next_ms=describe([item['loader_next_ms'] for item in measured]),
                  by_shape=by_shape, observations=observed, profiler_windows=instance.windows,
                  config=namespace['lm_config'].to_dict(), args=vars(namespace['args']),
                  cli=argv, corpus_sha256=source_hash,
                  corpus_verification=source_verification,
                  corpus_jsonl_sha256=corpus_audit['output_sha256'],
                  data_config=namespace['data_config'],
                  cpu_threads=torch.get_num_threads(),
                  trace_diagnostic=options.trace_last_step,
                  production_timing=options.production_timing,
                  resident_data=resident_report,
                  first_step_device_synchronized=not options.production_timing,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  cold_attention_probe=options.cold_attention_probe,
                  selective_save_attention=options.selective_save_attention,
                  quality=quality,
                  loss_check=loss_check,
                  restart_data_for_budget_test=options.restart_data_for_budget_test,
                  checkpoint_origin=checkpoint_origin,
                  windows_gpu_memory=windows_memory,
                  source_sha256=source_sha256,
                  cuda_allocator=dict(
                      backend=torch.cuda.memory.get_allocator_backend(),
                      **{key:torch.cuda.memory_stats().get(key) for key in (
                          'inactive_split_bytes.all.peak','num_alloc_retries','num_ooms')}),
                  environment={key: value for key, value in os.environ.items()
                               if key.startswith(('INSTINCT_MUON_', 'INSTINCT_MOE_', 'INSTINCT_DENSE_',
                                                  'INSTINCT_PACKED_ATTENTION_',
                                                  'INSTINCT_CAUSAL_ATTENTION_',
                                                  'INSTINCT_TRAINING_',
                                                  'PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF'))},
                  runtime_settings={key:os.environ.get(key,'auto') for key in (
                      'INSTINCT_MOE_FUSED_ROUTING','INSTINCT_CAUSAL_ATTENTION_BACKEND',
                      'INSTINCT_FLEX_AUTOTUNE','INSTINCT_MOE_TRANSPOSED_DW',
                      'INSTINCT_MUON_WORKSPACE_MB','INSTINCT_TRAINING_CPU_THREADS')},
                  scope='real production CLI, real chunked Arrow DataLoader, automatic '
                  'packing batches, prefetch, LR/clip/optimizer/zero_grad/log/pause; '
                  f'{options.steps} measured steps after {options.warmup} warmup steps; '
                  'epoch-final checkpoint omitted',
                  weights_saved=False)
    if options.resume_checkpoint:
        report['resume_checkpoint'] = str(options.resume_checkpoint.resolve())
        report['resume_step'] = namespace['ckp_data']['step']
    options.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[BENCH RESULT] ' + json.dumps({key: value for key, value in report.items()
                                        if key not in ('config', 'args', 'cli', 'data_config',
                                                       'observations')}, indent=2), flush=True)
    if loss_check is not None and not loss_check['all_finite']:
        raise RuntimeError('Non-finite training loss in the isolated benchmark')


if __name__ == '__main__':
    main()
