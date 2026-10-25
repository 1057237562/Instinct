"""Measure the real pretrainer, including its loader, then save and verify a pause.

Uses the saved panel's actual command builder without executing Streamlit. No
resident/replayed batches or benchmark exceptions: the trainer's normal pause
path applies pending gradients and saves the current full resume state.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
import math
import os
from pathlib import Path
import runpy
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__mp_main__':
    # Spawn unpickles the tokenizer before entering the cache worker function.
    # Match train_pretrain.py's DLL order before that can import torch.
    import trainer.compile_cache  # noqa: F401
    import datasets  # noqa: F401


def panel_command(state, prefix, config_path):
    """Execute only the pure WebUI builders (same isolation as relay parity tests)."""
    source = ROOT / 'scripts/config_webui.py'
    names = {'compute_intermediate_size', 'compute_head_dim', '_parse_longrope_factors',
             'build_config_dict', '_build_training_command', '_packing_preprocess_workers',
             '_default_epochs', '_default_learning_rate', '_trainer_name'}
    constants = {'_DEFAULT_EPOCHS', '_DEFAULT_LEARNING_RATES', '_TRAINER_NAME'}
    nodes = [node for node in ast.parse(source.read_text(encoding='utf-8')).body
             if isinstance(node, ast.FunctionDef) and node.name in names
             or isinstance(node, ast.Assign) and any(
                 isinstance(t, ast.Name) and t.id in constants for t in node.targets)]
    from scripts.moe_router_controls import migration_args
    scope = dict(st=SimpleNamespace(session_state=state), math=math, json=json,
                 os=os, sys=sys, __file__=str(source),
                 # This probe intentionally starts a new Dense pretraining run.
                 _resolve_cli_from_weight=lambda *a: None,
                 moe_router_migration_args=migration_args)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    config = scope['build_config_dict']()
    command = scope['_build_training_command'](
        'pretrain', config, prefix, False, apply_autotune=False)['cmd']
    command[command.index('--config_path') + 1] = str(config_path)
    return config, command


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=ROOT / 'trainer/webui_state.json')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup-updates', type=int, default=4)
    parser.add_argument('--updates', type=int, default=24)
    parser.add_argument('--checkpoint-mode', type=int, default=4)
    parser.add_argument('--start-chunk', type=int, default=0,
                        help='Zero-based real source chunk for a diagnostic sample; no prefix training is claimed')
    parser.add_argument('--tail-batches', type=int, default=0,
                        help='Start at the last N real batches of that chunk to exercise a transition')
    parser.add_argument('--cache-build-mode', choices=['inline', 'spawn'])
    parser.add_argument('--flex-tma', choices=['auto', '0', '1'], default='auto')
    parser.add_argument('--prepare-only', action='store_true')
    opts = parser.parse_args()
    if min(opts.warmup_updates, opts.updates) < 1:
        parser.error('update counts must be positive')
    if min(opts.start_chunk, opts.tail_batches) < 0:
        parser.error('sample positions must be nonnegative')
    os.chdir(ROOT)
    os.environ['INSTINCT_V2_FLEX_TMA'] = opts.flex_tma
    output = opts.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = 'production_probe_' + time.strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid())
    pause = output.with_suffix('.pause')
    if pause.exists():
        parser.error('pause file already exists; select a new output')
    state = json.loads(opts.panel.read_text(encoding='utf-8'))
    if state.get('use_moe') or state.get('model_architecture', 'standard') != 'standard':
        parser.error('this probe currently supports standard Dense pretraining')
    state.update(data_path_pretrain=str(opts.data.resolve()), use_grad_checkpoint=opts.checkpoint_mode,
                 profile='off', autotune_use_profile=False)
    if opts.cache_build_mode:
        state['cache_build_mode'] = opts.cache_build_mode
    config_path = output.with_suffix('.config.json')
    config, command = panel_command(state, prefix, config_path)
    command += ['--pause_file', str(pause)]
    write_json(config_path, config)
    report = dict(status='prepared', command=command, model_config=config, panel=state,
                  scope='real production loader; new model; sampled finite prefix',
                  source=dict(path=str(opts.data.resolve()), bytes=opts.data.stat().st_size,
                              mtime_ns=opts.data.stat().st_mtime_ns),
                  requested_warmup_updates=opts.warmup_updates, requested_updates=opts.updates,
                  prefix=prefix, windows=[], batches=[], prefetch=[])
    report['sample'] = dict(start_chunk=opts.start_chunk, tail_batches=opts.tail_batches,
                            omitted_prefix_was_trained=False)
    write_json(output, report)
    if opts.prepare_only:
        return

    launch_time = time.perf_counter()
    # Match the real entrypoint before datasets freezes its cache configuration.
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
    import trainer.compile_cache  # noqa: F401 -- must precede torch
    import datasets  # noqa: F401 -- Windows DLL load order
    import torch
    import trainer.training_profiler as profiler_module
    import trainer.streaming_pretrain as streaming
    import trainer.trainer_utils as utils
    from experiments.profile_checkpoint_recipe import verify_state

    original_profiler = profiler_module.TrainingProfiler
    original_iter = streaming.ChunkedPackedEpochLoader.__iter__
    original_prefetch_run = streaming._ThreadPrefetch._run
    original_pause_save = utils.pause_save_checkpoint
    context = {}
    meter = dict(micro=0, warmup_done=False, window_start=None, window_batch_start=0)

    def request_pause(signum, frame):
        pause.touch()
        print('[PRODUCTION PROBE] pause armed; waiting for accumulated update and save', flush=True)

    signal.signal(signal.SIGINT, request_pause)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, request_pause)

    def measured_prefetch(task):
        event = dict(start_seconds=time.perf_counter() - launch_time)
        report['prefetch'].append(event)
        try:
            return original_prefetch_run(task)
        finally:
            event.update(end_seconds=time.perf_counter() - launch_time,
                         success=task.error is None)

    def measured_iter(loader):
        context['loader'] = loader
        report['source'].update(rows=loader.plan['rows'], tokens=loader.plan['tokens'],
                                sha256=loader.plan['source_sha256'],
                                chunks=len(loader.plan['chunks']))
        if opts.start_chunk or opts.tail_batches:
            if not 0 <= opts.start_chunk < len(loader.plan['chunks']):
                raise ValueError('sample starts outside the source plan')
            loader.resume_chunk = opts.start_chunk
            if opts.tail_batches:
                chunk = loader.plan['chunks'][opts.start_chunk]
                dataset = loader.dataset_factory(True, None, (int(chunk['start']), int(chunk['end'])))
                batches = loader.packing_plan.batch_sampler(
                    dataset, active_packing=True, epoch=loader.epoch,
                    batch_size=loader.args.batch_size)
                loader.resume_chunk_step = max(0, len(batches) - opts.tail_batches)
                report['sample']['skipped_batches_in_start_chunk'] = loader.resume_chunk_step
                del dataset, batches
            report['scope'] = 'real production loader; contiguous boundary sample; omitted prefix not trained'
        iterator = original_iter(loader)
        try:
            while True:
                begin = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                # This CPU-only accounting is also inside the timed window.
                labels = batch[1]
                report['batches'].append(dict(
                    shape=list(labels.shape), slots=labels.numel(),
                    packed_mask=bool((batch[2] > 0).any()) if len(batch) == 3 else False,
                    useful=int((labels[:, 1:] != -100).sum()),
                    valid=int((batch[2] >= 0).sum()) if len(batch) == 3 else int((labels != -100).sum()),
                    loader_ms=(time.perf_counter() - begin) * 1000,
                    chunk=loader.data_config['streaming_chunk_index'],
                    chunk_step=loader.data_config['streaming_chunk_step']))
                yield batch
        finally:
            iterator.close()

    class Recorder(original_profiler):
        def __init__(self, args, **kwargs):
            super().__init__(args, **kwargs)
            context['trainer'] = sys._getframe(1).f_globals
            context['args'] = args
            report.update(status='running', resolved_args=vars(args),
                          device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                          hf_datasets_cache=str(datasets.config.HF_DATASETS_CACHE),
                          parameters=sum(p.numel() for p in context['trainer']['model'].parameters()),
                          backend_environment={k: os.environ.get(k, 'auto') for k in (
                              'INSTINCT_V2_GEMM_BACKEND', 'INSTINCT_V2_FLEX_TMA')},
                          setup_seconds=time.perf_counter() - launch_time)
            write_json(output, report)

        def end_step(self):
            meter['micro'] += 1
            accumulation = context['args'].accumulation_steps
            if meter['micro'] % accumulation:
                return None
            torch.cuda.synchronize()
            now = time.perf_counter()
            updates = meter['micro'] // accumulation
            if not meter['warmup_done']:
                if updates < opts.warmup_updates:
                    return None
                meter.update(warmup_done=True, window_start=now,
                             window_batch_start=len(report['batches']))
                report['warmup_seconds'] = now - launch_time
                report['warmup_microsteps'] = meter['micro']
                from trainer.windows_gpu_memory import local_memory_budget
                report['warmup_driver_memory'] = local_memory_budget(torch.cuda.get_device_name())
                torch.cuda.reset_peak_memory_stats()
                print('[PRODUCTION PROBE] warmup complete; timing includes next(loader), H2D, backward and update', flush=True)
                return None
            samples = report['batches'][meter['window_batch_start']:]
            seconds = now - meter['window_start']
            useful = sum(b['useful'] for b in samples)
            window = dict(update=updates, seconds=seconds, useful_tokens=useful,
                          tokens_per_second=useful / seconds,
                          slots=sum(b['slots'] for b in samples),
                          loader_ms=sum(b['loader_ms'] for b in samples),
                          shapes=dict(Counter(str(b['shape']) for b in samples)),
                          compiled_graphs=torch._dynamo.utils.counters['stats']['unique_graphs'],
                          allocated_gib=torch.cuda.memory_allocated() / 2**30,
                          reserved_gib=torch.cuda.memory_reserved() / 2**30,
                          end_seconds=now - launch_time)
            loss = sys._getframe(1).f_locals.get('loss')
            if loss is not None:
                window['last_micro_loss'] = float(loss.detach()) * accumulation
            report['windows'].append(window)
            meter.update(window_start=now, window_batch_start=len(report['batches']))
            print('[PRODUCTION PROBE] ' + json.dumps(window), flush=True)
            write_json(output, report)
            if len(report['windows']) >= opts.updates:
                pause.touch()
            return None

    def verified_pause_save(*args, **kwargs):
        torch.cuda.synchronize()
        report['training_end_seconds'] = time.perf_counter() - launch_time
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        report['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
        kwargs['rng_state'] = rng
        original_pause_save(*args, **kwargs)
        report['saved_seconds'] = time.perf_counter() - launch_time
        ns = context['trainer']
        checkpoint = ROOT / 'checkpoints' / f'{prefix}_{ns["lm_config"].hidden_size}_resume.pth'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        raw = getattr(ns['model'], '_orig_mod', ns['model'])
        verify_state(raw.state_dict(), saved['model'])
        verify_state(ns['optimizer'].state_dict(), saved['optimizer'])
        verify_state(ns['scaler'].state_dict(), saved['scaler'])
        verify_state(ns['data_config'], saved['data_config'])
        verify_state(rng, saved['rng_state'])
        verify_state(ns['lm_config'].to_dict(), saved['config'])
        sidecar = checkpoint.with_name(checkpoint.name.replace('_resume.pth', '.json'))
        # JSON object keys (e.g. Transformers id2label) are always strings.
        assert json.loads(sidecar.read_text(encoding='utf-8')) == json.loads(json.dumps(saved['config']))
        assert saved['step'] == kwargs['step']
        assert saved['step'] % ns['args'].accumulation_steps == 0
        assert saved['data_config']['streaming_chunk_index'] == report['batches'][-1]['chunk']
        assert saved['data_config']['streaming_chunk_step'] == report['batches'][-1]['chunk_step']
        weights = ROOT / ns['args'].save_dir / f'{prefix}_{ns["lm_config"].hidden_size}.pth'
        export = torch.load(weights, map_location='cpu', weights_only=False)
        verify_state({k: v.half() for k, v in saved['model'].items()}, export)
        report.update(checkpoint=str(checkpoint), weights=str(weights),
                      checkpoint_verified=True, checkpoint_step=saved['step'],
                      data_cursor=saved['data_config'])
        windows = report['windows']
        elapsed = sum(w['seconds'] for w in windows)
        useful = sum(w['useful_tokens'] for w in windows)
        report['measured'] = dict(seconds=elapsed, useful_tokens=useful,
                                 tokens_per_second=useful / elapsed if elapsed else None,
                                 loader_ms=sum(w['loader_ms'] for w in windows))
        report['launch_through_save_tokens_per_second'] = (
            sum(b['useful'] for b in report['batches']) / report['saved_seconds'])
        report['status'] = 'saved_verified'
        report['completed_requested_windows'] = len(report['windows']) == opts.updates
        write_json(output, report)
        print('[PRODUCTION PROBE] current model, optimizer, scaler, RNG, data cursor and matching weights verified', flush=True)

    profiler_module.TrainingProfiler = Recorder
    streaming.ChunkedPackedEpochLoader.__iter__ = measured_iter
    streaming._ThreadPrefetch._run = measured_prefetch
    utils.pause_save_checkpoint = verified_pause_save
    sys.argv = command[2:]
    try:
        runpy.run_path(str(ROOT / 'trainer/train_pretrain.py'), run_name='__main__')
    except SystemExit as exc:
        if exc.code != 42 or not report.get('checkpoint_verified'):
            raise
        report.update(status='complete', total_seconds=time.perf_counter() - launch_time)
        write_json(output, report)
    finally:
        profiler_module.TrainingProfiler = original_profiler
        streaming.ChunkedPackedEpochLoader.__iter__ = original_iter
        streaming._ThreadPrefetch._run = original_prefetch_run
        utils.pause_save_checkpoint = original_pause_save


if __name__ == '__main__':
    main()
