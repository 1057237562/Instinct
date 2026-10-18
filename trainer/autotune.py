"""Offline, bounded joint autotuning of the real single-GPU pretraining pipeline.

Run from the repository root. Each candidate uses a fresh process and scratch
outputs. This parent process never initializes CUDA or owns model parameters.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.autotune_config import (
    ROOT, SCHEMA, ARG_CHOICES, ARG_POSITIVE, PERF_ENV, ENV_CHOICES, ENV_POSITIVE, ENV_TILES,
    audit_source, changed, digest, quality_gate, read_json, validate_settings, write_json,
)


def arguments(values):
    result = []
    for key, value in values.items():
        if not key.replace('_', '').isalnum() or key.startswith('_'):
            raise ValueError(f'invalid CLI argument name: {key}')
        if key == 'use_wandb':
            if value:
                result.append('--use_wandb')
        elif value is not None:
            result.extend(['--' + key, str(int(value) if isinstance(value, bool) else value)])
    return result


def search_space(spec, baseline):
    """Default ranges are relative to the measured baseline, not a GPU model."""
    budget = baseline['resolved_token_budget']
    grid = spec.get('search')
    if grid is None:
        grid = {
            'args.bucket_token_budget': sorted({max(16, round(budget*x/16)*16) for x in (.5, .75, 1., 1.25, 1.5, 2.)}),
            'args.use_grad_checkpoint': [1, 2],
            'env.INSTINCT_MOE_INNER_CHECKPOINT': ['0', '1', '2'],
            'env.INSTINCT_MUON_WORKSPACE_MB': ['64', '128', '256', '512'],
            'env.INSTINCT_MUON_BATCH_SIZE': ['8', '16', '32'],
            'env.INSTINCT_MUON_PREALLOCATE': ['0', '1'],
            'env.INSTINCT_TRAINING_CPU_THREADS': [str(x) for x in (1, 2, 4) if x <= (os.cpu_count() or 1)],
            'args.compile_mode': ['default', 'max-autotune-no-cudagraphs'],
            'env.INSTINCT_GROUPED_MM_FWD_TILE': ['64,64,32,4,3', '64,128,32,4,3'],
            'env.INSTINCT_GROUPED_MM_DW_TILE': ['64,64,32,4,3', '64,128,32,4,3'],
            'env.INSTINCT_GROUPED_MM_PROGRAMS_PER_SM': ['2', '4'],
        }
    for name, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f'empty search axis: {name}')
        section, key = name.split('.', 1)
        if section not in ('args', 'env'):
            raise ValueError(f'invalid search axis: {name}')
        for value in values:
            validate_settings({section: {key: value}})
    return grid


def neighbors(settings, grid):
    """One-coordinate trials plus coupled batch/checkpoint candidates."""
    for axis, values in grid.items():
        section, key = axis.split('.', 1)
        for value in values:
            candidate = changed(settings, section, key, value)
            if compatible(candidate):
                yield candidate
    budgets = grid.get('args.bucket_token_budget', [settings['args']['bucket_token_budget']])
    checkpoints = grid.get('args.use_grad_checkpoint', [settings['args']['use_grad_checkpoint']])
    inner = grid.get('env.INSTINCT_MOE_INNER_CHECKPOINT', [settings['env'].get('INSTINCT_MOE_INNER_CHECKPOINT', '0')])
    for budget, checkpoint, mode in itertools.product(budgets, checkpoints, inner):
        candidate = copy.deepcopy(settings)
        candidate['args'].update(bucket_token_budget=budget, use_grad_checkpoint=checkpoint)
        candidate['env']['INSTINCT_MOE_INNER_CHECKPOINT'] = mode
        if compatible(candidate):
            yield candidate


def compatible(settings):
    env = settings['env']
    return not (env.get('INSTINCT_MOE_INNER_CHECKPOINT', '0') != '0' and (
        settings['args'].get('use_grad_checkpoint', 1) != 1
        or env.get('INSTINCT_MOE_SELECTIVE_CHECKPOINT', '0') != '0'))


class BudgetBisection:
    """Locate an empirical VRAM boundary, then sample nearby throughput points.

    Only OOM/driver-budget failures prune the upper half. Compile/quality errors
    must not be interpreted as memory exhaustion. Contradictory measurements
    disable monotonic pruning (packing shapes and free VRAM can change).
    """
    def __init__(self, values, anchor):
        self.values = sorted({int(x) for x in values} | {int(anchor)})
        self.anchor = self.values.index(int(anchor))
        self.low, self.high = -1, len(self.values)
        self.tested = {}
        self.nonmonotonic = False

    def observe(self, value, status):
        index = self.values.index(int(value))
        self.tested[index] = status
        if status == 'ok':
            self.nonmonotonic |= index >= self.high
            self.low = max(self.low, index)
        elif status in ('oom', 'driver_budget_exceeded'):
            self.nonmonotonic |= index <= self.low
            self.high = min(self.high, index)

    def next(self):
        indices = range(len(self.values)) if self.nonmonotonic else range(self.low+1, self.high)
        remaining = [i for i in indices if i not in self.tested]
        if remaining:
            return self.values[remaining[len(remaining)//2]]
        # Throughput is NOT monotone in batch size. Compare smaller neighbors
        # rather than blindly publishing the largest allocation that fits.
        for index in sorted({max(0, self.anchor-1), max(0, self.low-1)}):
            if index not in self.tested and index < self.high:
                return self.values[index]
        return None


def kill_trial(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=20)


def gpu_snapshot():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,power.draw,clocks.sm',
            '--format=csv'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def subprocess_trial(command, environment, log, timeout):
    """Only the child we created (and its descendants) is stopped on timeout."""
    options = dict(cwd=None if Path.cwd() == ROOT else ROOT, env=environment)
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options['start_new_session'] = True
    with Path(log).open('w', encoding='utf-8') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, **options)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            kill_trial(process)
            return -1, True
        except BaseException:
            kill_trial(process)
            raise


class Search:
    def __init__(self, spec, options, directory, source):
        self.spec, self.options, self.directory, self.source = spec, options, directory, source
        self.hardware = self.contract = None
        self.started = time.monotonic()
        self.rows = []
        self.seen = set()
        self.memory_searches = {}
        self.memory_recovery = set()

    def memory_state(self, settings, grid):
        context = copy.deepcopy(settings)
        anchor = context['args'].pop('bucket_token_budget')
        key = digest(context)
        if key not in self.memory_searches:
            self.memory_searches[key] = (copy.deepcopy(settings), BudgetBisection(
                grid['args.bucket_token_budget'], anchor))
        return key, self.memory_searches[key][1]

    def observe_memory(self, result, grid):
        if 'args.bucket_token_budget' not in grid:
            return
        key, state = self.memory_state(result['settings'], grid)
        state.observe(result['settings']['args']['bucket_token_budget'], result['status'])
        if result['status'] in ('oom', 'driver_budget_exceeded'):
            self.memory_recovery.add(key)

    def next_candidate(self, best, grid):
        settings = best['settings']
        # Cheap same-shape kernel probes reuse packed data. Avoid spending the
        # entire budget rebuilding datasets before trying the major GEMMs.
        kernels = {key:value for key,value in grid.items() if key in (
            'env.INSTINCT_GROUPED_MM_FWD_TILE', 'env.INSTINCT_GROUPED_MM_DW_TILE')}
        for candidate in neighbors(settings, kernels):
            if digest(candidate) not in self.seen:
                return candidate
        if 'args.bucket_token_budget' in grid:
            self.observe_memory(best, grid)
            best_key, _ = self.memory_state(settings, grid)
            for key in [best_key, *sorted(self.memory_recovery - {best_key})]:
                context, state = self.memory_searches[key]
                value = state.next()
                while value is not None:
                    candidate = changed(context, 'args', 'bucket_token_budget', value)
                    if digest(candidate) not in self.seen:
                        return candidate
                    # The existing row owns the actual feasibility result.
                    previous = next(row for row in self.rows if digest(row['settings']) == digest(candidate))
                    state.observe(value, previous['status'])
                    value = state.next()
        remaining = {key:value for key,value in grid.items()
                     if key != 'args.bucket_token_budget' and key not in kernels}
        for candidate in neighbors(settings, remaining):
            if digest(candidate) not in self.seen:
                return candidate
        return None

    def trial(self, settings, *, probe=False, confirm=False):
        validate_settings(settings)
        identifier = digest(settings)[:20] + ('-probe' if probe else '-confirm' if confirm else '')
        directory = self.directory / identifier
        request = dict(schema=SCHEMA, settings=settings, source=self.source,
            hardware=self.hardware, contract=self.contract, probe=probe,
            warmup=self.options.warmup, steps=self.options.steps,
            validation_examples=self.options.validation_examples, peak_tflops=self.options.peak_tflops)
        request_file = directory / 'request.json'
        key = digest(request)
        previous = directory / 'result.json'
        if not probe and not self.options.refresh and previous.exists():
            result = read_json(previous)
            if result.get('request_key') == key and result.get('status') == 'ok':
                print(f'[Autotune] reuse {identifier}: {result["status"]}', flush=True)
                return result
        write_json(request_file, request)
        write_json(previous, dict(status='running'))  # Never read a stale success after a crashed retry.
        values = dict(self.spec['args'])
        values.update(settings['args'])
        values.update(from_resume=0, use_wandb=False, save_dir=str(directory / 'weights'),
            save_weight='autotune_disposable', save_interval=2**60, pause_file=str(directory / 'pause'),
            profile='timing', profile_warmup=self.options.warmup, profile_interval=self.options.steps,
            autotune_trial=str(request_file), dataset_streaming='on', cache_build_mode='inline')
        environment = dict(os.environ)
        environment.update(PYTHONUTF8='1', TOKENIZERS_PARALLELISM='true')
        environment.update(settings['env'])
        command = [sys.executable]
        if self.options.worker_launcher:
            command.append(str(Path(self.options.worker_launcher).resolve()))
        command += [str(ROOT/'trainer/train_pretrain.py'), *arguments(values)]
        remaining = self.options.time_limit - (time.monotonic()-self.started)
        if remaining <= 0:
            raise TimeoutError('autotuning time budget exhausted')
        print(f'[Autotune] {identifier}: {json.dumps(settings)}', flush=True)
        telemetry_before = gpu_snapshot()
        trial_started = time.monotonic()
        code, timed_out = subprocess_trial(command, environment, directory/'trial.log',
            min(self.options.trial_timeout, remaining))
        result = read_json(previous) if previous.exists() else dict(status='process_error', exit_code=code)
        if timed_out:
            result = dict(status='timeout', exit_code=code)
        elif result.get('status') == 'running' or code != 0 and result.get('status') in ('ok', 'probe'):
            result = dict(status='process_error', exit_code=code)
        result.update(settings=settings, request_key=key, directory=str(directory),
                      startup_and_training_seconds=time.monotonic()-trial_started,
                      gpu_snapshot_before=telemetry_before, gpu_snapshot_after=gpu_snapshot())
        write_json(previous, result)
        print(f'[Autotune] {result["status"]}, useful tokens/s={result.get("useful_tokens_per_second")}', flush=True)
        return result

    def record(self, row):
        self.rows.append(row)
        self.seen.add(digest(row['settings']))
        write_json(self.directory/'search.json', dict(schema=SCHEMA, trials=self.rows, hardware=self.hardware,
                                                     contract=self.contract, source=self.source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('.cache/autotune'))
    parser.add_argument('--max-trials', type=int, default=24, help='Includes baseline and final confirmation')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--validation-examples', type=int, default=16)
    parser.add_argument('--time-limit', type=float, default=3600, help='Total wall-clock seconds, including compilation')
    parser.add_argument('--trial-timeout', type=float, default=900)
    parser.add_argument('--peak-tflops', type=float, help='Optional explicit dense peak for the actual compute precision')
    parser.add_argument('--target-tokens-per-second', type=float,
                        help='Require this measured useful-token throughput before publishing a profile')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--plan-only', action='store_true', help='Validate search config and print scope; no training')
    parser.add_argument('--worker-launcher', type=Path, help='Optional diagnostic Python bootstrap, normally omitted')
    options = parser.parse_args()
    if (options.max_trials < 2 or min(options.warmup, options.steps, options.validation_examples) < 1
            or any(not math.isfinite(x) or x <= 0 for x in (options.time_limit, options.trial_timeout))
            or options.peak_tflops is not None and (not math.isfinite(options.peak_tflops) or options.peak_tflops <= 0)):
        parser.error('positive limits and at least two trials required')
    if options.target_tokens_per_second is not None and (
            not math.isfinite(options.target_tokens_per_second) or options.target_tokens_per_second <= 0):
        parser.error('target throughput must be positive and finite')
    spec = read_json(options.config)
    base = spec['args']
    if any(key in base for key in ('autotune_profile', 'autotune_trial')) or base.get('from_resume', 0):
        raise ValueError('calibrate fresh scratch state, not a resume cursor or another profile')
    if base.get('use_wandb'):
        raise ValueError('autotuning does not publish external training logs')
    if base.get('from_weight') != 'none':
        raise ValueError('set from_weight=none for isolated from-scratch calibration')
    if not base.get('sequence_packing') or base.get('sequence_packing_mode') != 'bucket':
        raise ValueError('configuration must enable bucket packing')
    validate_settings({'env': spec.get('environment', {})})
    if options.plan_only:
        grid = search_space(spec, {'resolved_token_budget': int(base.get('bucket_token_budget', 0) or 16384)})
        print(json.dumps(dict(scope='single GPU, standard Dense/MoE pretrain, real audited streaming source',
            search=grid, default_budget_note='default batch candidates scale from measured baseline',
            max_trials=options.max_trials, maximum_updates=options.max_trials*(options.warmup+options.steps),
            time_limit_seconds=options.time_limit, saves_formal_weights=False), indent=2))
        return
    source = audit_source(base['data_path'], spec['audit'], spec.get('provenance'))
    settings = dict(args={key: base[key] for key in ARG_CHOICES if key in base},
                    env={key: os.environ[key] for key in PERF_ENV if key in os.environ})
    settings['env'].update(spec.get('environment', {}))
    settings['args'].setdefault('use_grad_checkpoint', 1)
    # A probe gets the physical budget without leaving a parent CUDA context.
    memory = base.get('bucket_gpu_memory_gb', 'auto')
    settings['args']['bucket_gpu_memory_gb'] = 1. if memory == 'auto' else float(memory)
    if base.get('bucket_token_budget', 0) > 0:
        settings['args']['bucket_token_budget'] = int(base['bucket_token_budget'])
    validate_settings(settings)
    from filelock import FileLock
    options.output = options.output.resolve()
    options.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(options.output/'autotune.lock'), timeout=0):
        search = Search(spec, options, options.output/'probe', source)
        probe = search.trial(settings, probe=True)
        if probe['status'] != 'probe':
            raise RuntimeError(f'preflight failed; see {probe["directory"]}/trial.log')
        search.hardware, search.contract = probe['hardware'], probe['contract']
        if memory == 'auto':
            settings['args']['bucket_gpu_memory_gb'] = search.hardware['vram_bytes']/2**30
        cache_key = digest(dict(hardware=search.hardware, contract=search.contract, source=source,
                               settings=settings, search=spec.get('search'), warmup=options.warmup,
                               steps=options.steps, validation_examples=options.validation_examples,
                               target=options.target_tokens_per_second))[:24]
        search.directory = options.output/cache_key
        try:
            baseline = search.trial(settings)
            search.record(baseline)
            while baseline['status'] == 'oom' and len(search.rows) < options.max_trials-1:
                previous_budget = baseline['resolved_token_budget']
                reduced = max(16, previous_budget//32*16)
                if reduced >= previous_budget:
                    break
                settings = changed(settings, 'args', 'bucket_token_budget', reduced)
                baseline = search.trial(settings)
                search.record(baseline)
            if baseline['status'] != 'ok':
                raise RuntimeError(f'baseline failed; lower initial bucket_token_budget; see {baseline["directory"]}/trial.log')
            settings = copy.deepcopy(baseline['resolved_settings'])
            baseline['settings'] = settings
            search.seen.add(digest(settings))
            grid = search_space(spec, baseline)
            best = baseline
            # Same-shape kernels first, binary memory boundaries per strategy,
            # then other coordinate neighbors. No monotone-throughput assumption.
            while (len(search.rows) < options.max_trials-1
                   and options.time_limit-(time.monotonic()-search.started) > options.trial_timeout):
                settings_to_try = search.next_candidate(best, grid)
                if settings_to_try is None:
                    break
                candidate = search.trial(settings_to_try)
                passed, reason = quality_gate(baseline, candidate)
                candidate.update(quality_passed=passed, quality_reason=reason)
                search.record(candidate)
                search.observe_memory(candidate, grid)
                if passed and candidate['useful_tokens_per_second'] > best['useful_tokens_per_second']:
                    best = candidate
            confirm = search.trial(best['settings'], confirm=True)
            passed, reason = quality_gate(baseline, confirm)
            confirm.update(quality_passed=passed, quality_reason=reason)
            search.record(confirm)
            if not passed:
                raise RuntimeError('winner confirmation failed; no production profile was published')
            # Requested convention: best valid observed throughput is primary;
            # retain confirmation and variation rather than silently discarding them.
            primary = max((best, confirm), key=lambda x:x['useful_tokens_per_second'])
            if (options.target_tokens_per_second is not None
                    and primary['useful_tokens_per_second'] < options.target_tokens_per_second):
                write_json(search.directory/'target_unmet.json', dict(status='target_unmet',
                    target_tokens_per_second=options.target_tokens_per_second, best=primary, confirmation=confirm))
                raise RuntimeError('quality passed but throughput target was not reached; no new profile published')
            profile = dict(schema=SCHEMA, status='validated', trainer='pretrain', hardware=search.hardware,
                contract=search.contract, source=source, settings=best['settings'], primary=primary,
                confirmation=confirm, baseline_directory=baseline['directory'],
                target_tokens_per_second=options.target_tokens_per_second,
                limitation='short equal-update validation; batch changes are not equal-token convergence proof')
            profile_path = search.directory/'profile.json'
            write_json(profile_path, profile)
            launch_args = {k:v for k,v in base.items() if k not in set(ARG_CHOICES)|ARG_POSITIVE}
            launch_args.update(autotune_profile=str(profile_path), dataset_streaming='on', cache_build_mode='inline')
            launch = [sys.executable, 'trainer/train_pretrain.py', *arguments(launch_args)]
            write_json(search.directory/'launch.json', dict(cwd=str(ROOT), argv=launch))
            print('[Autotune] validated profile: ' + str(profile_path), flush=True)
            print(subprocess.list2cmdline(launch) if os.name == 'nt' else shlex.join(launch), flush=True)
        except (TimeoutError, KeyboardInterrupt):
            print('[Autotune] stopped at budget/interruption; completed trials remain cached, no unconfirmed profile published', flush=True)
            raise


if __name__ == '__main__':
    main()
