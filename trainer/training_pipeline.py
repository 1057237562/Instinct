"""Sequential training with per-stage caches and process-owned CUDA lifetimes."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
TRAINERS = {'pretrain', 'full_sft', 'lora', 'dpo', 'ppo', 'grpo', 'agent', 'distillation'}


def validate(plan):
    stages = plan.get('stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('stages must be a non-empty list')
    seen = set()
    for stage in stages:
        name = stage.get('name', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in seen:
            raise ValueError('Stage names must be unique letters, digits, underscores or hyphens')
        if stage.get('trainer') not in TRAINERS:
            raise ValueError(f'{name}: unknown trainer')
        if not isinstance(stage.get('args', {}), dict) or not isinstance(stage.get('model', {}), dict):
            raise ValueError(f'{name}: args and model must be objects')
        source = stage.get('input_stage')
        if source and source not in seen:
            raise ValueError(f'{name}: input_stage must refer to an earlier stage')
        if source and next(s for s in stages if s['name'] == source)['trainer'] == 'lora':
            raise ValueError('LoRA adapters must be merged before use as base weights')
        if stage['trainer'] == 'pretrain' and source:
            raise ValueError('Pretrain must start from scratch')
        if stage['trainer'] != 'pretrain' and not source and not stage.get('args', {}).get('from_weight'):
            raise ValueError(f'{name}: select input_stage or explicit from_weight')
        forbidden = {'save_dir', 'save_weight', 'lora_name', 'pause_file', 'config_path', 'from_resume'}
        if forbidden.intersection(stage.get('args', {})):
            raise ValueError(f'{name}: output, config, pause and resume arguments are managed by the pipeline')
        for key, value in stage.get('args', {}).items():
            if not re.fullmatch(r'[a-z][a-z0-9_]*', key) or not isinstance(value, (str, int, float, bool)):
                raise ValueError(f'{name}: invalid argument {key}')
        seen.add(name)
    return plan


def cache_environment(cache):
    env = os.environ.copy()
    for key, folder in {
        'HF_DATASETS_CACHE': 'datasets',
        'TEMP': 'tmp', 'TMP': 'tmp', 'TMPDIR': 'tmp',
    }.items():
        target = cache / folder
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    env['PYTHONUTF8'] = '1'
    return env


def cleanup_cache(cache, parent):
    cache, parent = Path(cache), Path(parent).resolve()
    if cache.is_symlink() or cache.resolve().parent != parent or not cache.name.startswith('stage-'):
        raise ValueError(f'Refusing to clean unowned cache: {cache}')
    shutil.rmtree(cache)


def run(plan, run_dir):
    validate(plan)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    cache_parent = ROOT / '.cache' / 'pipeline'
    cache_parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    state = {'status': 'running', 'stages': []}

    def save():
        temp = run_dir / 'status.tmp'
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(run_dir / 'status.json')

    save()
    try:
        for index, stage in enumerate(plan['stages']):
            if (run_dir / 'stop').exists():
                state['status'] = 'stopped'
                break
            name, trainer = stage['name'], stage['trainer']
            record = {'name': name, 'status': 'running'}
            state['stages'].append(record)
            save()
            args = dict(stage.get('args', {}))
            model = stage.get('model', {})
            config = run_dir / f'{name}.json'
            config.write_text(json.dumps(model), encoding='utf-8')
            prefix = f'pipeline_{run_dir.name}_{index}_{name}'
            args.update(config_path=str(config), save_dir=str(ROOT / 'out'),
                        pause_file=str(run_dir / 'stop'), from_resume=0)
            args['lora_name' if trainer == 'lora' else 'save_weight'] = prefix
            if stage.get('input_stage'):
                args['from_weight'] = outputs[stage['input_stage']]
            if trainer == 'pretrain':
                args['from_weight'] = 'none'
            if trainer == 'distillation':
                args.setdefault('from_student_weight', args['from_weight'])
                args.setdefault('from_teacher_weight', args['from_weight'])
                args.pop('from_weight')
            command = [sys.executable, '-u']
            if model.get('model_architecture') == 'linear':
                command.append(str(ROOT / 'run_linear.py'))
            command.append(str(ROOT / 'trainer' / f'train_{trainer}.py'))
            for key, value in args.items():
                if key == 'use_wandb':
                    if value:
                        command.append('--use_wandb')
                else:
                    command.extend([f'--{key}', str(int(value) if isinstance(value, bool) else value)])
            cache = Path(tempfile.mkdtemp(prefix='stage-', dir=cache_parent))
            print(f'[Pipeline] Starting {name}', flush=True)
            try:
                with (run_dir / f'{name}.log').open('w', encoding='utf-8') as log:
                    with subprocess.Popen(command, cwd=ROOT, env=cache_environment(cache),
                                          stdout=log, stderr=subprocess.STDOUT) as process:
                        rc = process.wait()
            finally:
                # The trainer has exited: its CUDA context and DataLoader workers
                # are gone before deleting memory-mapped dataset Arrow files.
                cleanup_cache(cache, cache_parent)
                print(f'[Pipeline] {name}: stage cache removed', flush=True)
            record['exit_code'] = rc
            if rc:
                record['status'] = state['status'] = 'paused' if rc == 42 else 'failed'
                break
            matches = list((ROOT / 'out').glob(f'{prefix}_*.pth'))
            if len(matches) != 1:
                raise RuntimeError(f'{name}: expected exactly one output weight, found {len(matches)}')
            outputs[name] = str(matches[0])
            record.update(status='success', output=outputs[name])
            save()
        else:
            state['status'] = 'success'
    except Exception as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        save()
    return 0 if state['status'] == 'success' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run-dir')
    parser.add_argument('--validate-only', action='store_true')
    options = parser.parse_args()
    plan = validate(json.loads(Path(options.plan).read_text(encoding='utf-8')))
    if options.validate_only:
        print('Pipeline configuration valid')
    else:
        sys.exit(run(plan, options.run_dir or ROOT / 'trainer' / 'pipeline_runs' / uuid.uuid4().hex[:12]))
