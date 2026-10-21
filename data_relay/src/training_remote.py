"""Persistent, process-owned training control. Invoked by data_relay over SSH."""
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

TRAINERS = {name: 'trainer/train_' + name + '.py' for name in
            ('pretrain', 'full_sft', 'lora', 'dpo', 'ppo', 'grpo', 'agent', 'distillation')}
TRAINERS['cpt'] = TRAINERS['pretrain']
ACTIVE = {'starting', 'running', 'pausing', 'stopping', 'orphaned'}
RESERVED = {'save_dir', 'save_weight', 'lora_name', 'pause_file', 'config_path', 'from_resume'}
FLAGS = {'use_wandb', 'debug_mode'}


def now():
    return int(time.time())


def inside(root, relative, exists=False):
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute():
        if not path.is_relative_to(root):
            raise ValueError('Path is outside the training repository')
        path = path.relative_to(root)
    if not path.parts or '..' in path.parts:
        raise ValueError('Invalid managed path')
    result = root / path
    for ancestor in [result, *result.parents]:
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError('Managed paths cannot contain symlinks')
    if not result.resolve().is_relative_to(root):
        raise ValueError('Managed path escapes repository')
    if exists and not result.exists():
        raise ValueError('File does not exist: ' + str(result))
    return result


def repository(opts, require_code=True):
    path = Path(opts['repo_root'])
    if not path.is_absolute() or str(path) == '/':
        raise ValueError('Training requires an absolute repository root')
    root = path.resolve()
    if require_code and not (root / 'model' / 'model_instinct.py').is_file():
        raise ValueError('Instinct repository not found at ' + str(root))
    return root


def manager(root):
    return inside(root, '.relay_training')


def run_dir(root, identifier):
    if not re.fullmatch(r'r_[a-f0-9]{32}', identifier):
        raise ValueError('Invalid training job ID')
    return inside(root, '.relay_training/' + identifier)


def save(path, data):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


@contextlib.contextmanager
def control_lock(root):
    base = manager(root)
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'control.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def identity(pid):
    try:
        raw = Path('/proc', str(pid), 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return boot + ':' + fields[19]
    except (OSError, IndexError):
        return None


def alive(pid, token):
    return bool(pid and token and identity(pid) == token)


def unmanaged_trainers():
    result = []
    proc = Path('/proc')
    if not proc.exists():
        return result
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            scripts = [a for a in args if a.endswith('.py') and (Path(a).name.startswith('train_')
                       or Path(a).name in {'training_pipeline.py', 'autotune.py', 'autotune_trial.py', 'autotune_webui_worker.py'})]
            if scripts and identity(int(entry.name)):
                result.append({'pid': int(entry.name), 'scripts': scripts})
        except OSError:
            pass
    return result


def checkpoints(root, job):
    prefix = job['save_weight'] + '_'
    directory = root / 'checkpoints'
    if not directory.is_dir():
        return []
    return [str(p.relative_to(root)) for p in sorted(directory.glob(prefix + '*_resume.pth'))
            if p.is_file() and not p.is_symlink()]


def refresh(root, job):
    if job['status'] in ACTIVE:
        child = alive(job.get('pid'), job.get('process_identity'))
        supervisor = alive(job.get('supervisor_pid'), job.get('supervisor_identity'))
        if child and not supervisor:
            job['status'] = 'orphaned'
        elif not child and not supervisor:
            job['status'] = 'interrupted'
            job['finished_at'] = now()
            job['error'] = 'Supervisor/process disappeared; exit code is unknown'
        save(run_dir(root, job['id']) / 'job.json', job)
    job['resume_checkpoints'] = checkpoints(root, job)
    job['can_resume'] = job['status'] in {'paused', 'failed', 'cancelled', 'interrupted'} and bool(job['resume_checkpoints'])
    return job


def read_job(root, identifier):
    path = run_dir(root, identifier) / 'job.json'
    if not path.is_file():
        raise ValueError('Training job does not exist')
    return refresh(root, json.loads(path.read_text(encoding='utf-8')))


def all_jobs(root):
    base = manager(root)
    if not base.exists():
        return []
    jobs = []
    for path in base.glob('r_*/job.json'):
        job = read_job(root, path.parent.name)
        jobs.append(job)
    return sorted(jobs, key=lambda job: job['created_at'], reverse=True)


def lookup_operation(root, operation_id):
    if not operation_id:
        return None
    if not re.fullmatch(r'op_[a-f0-9]{32}', operation_id):
        raise ValueError('Invalid training operation ID')
    return next((job for job in all_jobs(root) if operation_id in
                 (job.get('operation_id'), job.get('resume_operation_id'))), None)


def idle(root, allow_code_sync=False):
    if not allow_code_sync and inside(root, '.relay_code_sync/ACTIVE').exists():
        raise ValueError('训练代码同步未完成，请先在中继回滚或恢复同步')
    jobs = all_jobs(root)
    active = [job for job in jobs if job['status'] in ACTIVE]
    managed_pids = {job.get('pid') for job in active}
    external = [process for process in unmanaged_trainers() if process['pid'] not in managed_pids]
    if active or external:
        raise ValueError('Training is active; pause/stop it before starting another job or powering off')


def environment(root, opts, identifier=None):
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    devices = opts.get('gpu_devices', '')
    if devices:
        if not re.fullmatch(r'\d+(,\d+)*', devices):
            raise ValueError('GPU devices must be comma-separated indices')
        env['CUDA_VISIBLE_DEVICES'] = devices
    if identifier:
        cache = inside(root, '.cache/relay_training/' + identifier)
        for key, folder in {'HF_DATASETS_CACHE': 'datasets', 'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp'}.items():
            location = cache / folder
            location.mkdir(parents=True, exist_ok=True)
            env[key] = str(location)
        env.update(INSTINCT_MANAGED_DATA_CACHE='1', INSTINCT_DATA_CACHE_MAX_GB=str(opts['cache_max_gb']))
        env['HF_HOME'] = str(inside(root, '.cache/huggingface'))
    return env


def gpu_check(root, opts, count):
    probe = subprocess.run([opts['python'], '-c', 'import torch,json;print(json.dumps({"count":torch.cuda.device_count()}))'],
                           cwd=root, env=environment(root, opts), capture_output=True, text=True, timeout=60)
    if probe.returncode or json.loads(probe.stdout.strip())['count'] < count:
        raise ValueError('CUDA devices are unavailable; use GPU mode and check the Python environment')


def audit_data(root, request, args):
    data = inside(root, args.get('data_path', ''), exists=True)
    if not data.is_file() or not data.name.endswith(('.jsonl', '.jsonl.gz', '.parquet')):
        raise ValueError('Select a JSONL or Parquet training artifact')
    args['data_path'] = str(data)
    if not request.get('audit_path'):
        return {'path': None, 'data_path': str(data), 'checked_at': now()}
    report_path = inside(root, request['audit_path'], exists=True)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    expected = report.get('output_sha256')
    if not isinstance(expected, str):
        expected = (report.get('output') or {}).get('sha256')
    if not isinstance(expected, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', expected):
        raise ValueError('Cleaning audit must contain the artifact output_sha256')
    stat = data.stat()
    digest = hashlib.sha256()
    with data.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != expected.lower() or data.stat().st_mtime_ns != stat.st_mtime_ns or data.stat().st_size != stat.st_size:
        raise ValueError('Training artifact changed or does not match the reviewed audit')
    args['data_path'] = str(data)
    return {'path': str(report_path), 'data_sha256': digest.hexdigest(), 'reviewed_at': now()}


def prepare(root, opts, request):
    trainer = request.get('trainer', 'pretrain')
    if trainer not in TRAINERS or not (root / TRAINERS[trainer]).is_file():
        raise ValueError('Unsupported or missing trainer')
    args = dict(request.get('args', {}))
    for key, value in args.items():
        if not re.fullmatch(r'[a-z][a-z0-9_]*', key) or key in RESERVED:
            raise ValueError('Invalid or manager-owned argument: ' + key)
        if not isinstance(value, (str, int, float, bool)) or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError('Training arguments must be finite scalars')
    audit = audit_data(root, request, args)
    inline = request.get('model_config')
    config = None
    imported = request.get('existing_resume', '')
    prefix = None
    if imported:
        checkpoint = inside(root, imported, exists=True)
        match = re.fullmatch(r'([A-Za-z0-9_-]+)_(\d+)(_moe)?_resume\.pth', checkpoint.name)
        if checkpoint.parent != root / 'checkpoints' or not match:
            raise ValueError('Select a full resume checkpoint from checkpoints/')
        prefix = match[1]
        for known in sorted(TRAINERS, key=len, reverse=True):
            if prefix.startswith(known + '_') and TRAINERS[known] != TRAINERS[trainer]:
                raise ValueError('Selected trainer does not match this resume checkpoint')
        config = inside(root, 'checkpoints/' + checkpoint.name.removesuffix('_resume.pth') + '.json', exists=True)
    if imported:
        model = json.loads(config.read_text(encoding='utf-8'))
    elif inline is not None:
        if not isinstance(inline, dict):
            raise ValueError('Cached model configuration must be a JSON object')
        model = dict(inline)
    else:
        config = inside(root, request.get('config_path', 'trainer/config_instinct_v1_moe.json'), exists=True)
        model = json.loads(config.read_text(encoding='utf-8'))
    if not isinstance(model, dict) or any(not isinstance(model.get(key), int) or isinstance(model.get(key), bool) or model[key] <= 0 for key in ['hidden_size','num_hidden_layers']):
        raise ValueError('Select a model architecture JSON with hidden_size and num_hidden_layers')
    model.setdefault('use_moe', False)
    if imported and (model.get('hidden_size') != int(match[2]) or bool(model.get('use_moe')) != bool(match[3])):
        raise ValueError('Resume filename and saved model configuration disagree')
    for key in ['hidden_size', 'num_hidden_layers', 'use_moe', 'use_grad_checkpoint', 'param_dtype', 'kv_cache_dtype']:
        if key in model:
            if imported: args[key] = model[key]
            else: args.setdefault(key, model[key])
    if imported and 'model_architecture' in args:
        args['model_architecture'] = model.get('model_architecture', 'standard')
    if 'use_recurrsive' in args or model.get('model_architecture') == 'recurrsive_instinct':
        args['use_recurrsive'] = int(model.get('model_architecture') == 'recurrsive_instinct')
    if 'early_exit' in args or model.get('early_exit_layers'):
        args['early_exit'] = int(bool(model.get('early_exit_layers')))
    if trainer == 'distillation':
        for role in ['student', 'teacher']:
            for suffix, source in [('hidden_size','hidden_size'),('num_layers','num_hidden_layers'),('use_moe','use_moe')]:
                key = role + '_' + suffix
                if imported and role == 'student': args[key] = model[source]
                else: args.setdefault(key, model[source])
        args.setdefault('from_student_weight', 'none')
    if not imported:
        if trainer == 'pretrain':
            if args.get('from_weight', 'none') != 'none':
                raise ValueError('Use CPT for continued pretraining; pretrain starts from scratch')
            args['from_weight'] = 'none'
        else:
            base = args.get('from_teacher_weight', args.get('from_weight', '')) if trainer == 'distillation' else args.get('from_weight', '')
            if not re.fullmatch(r'[A-Za-z0-9_-]+', base):
                raise ValueError('Select a completed base weight for CPT/SFT/post-training')
            dimension = args['teacher_hidden_size'] if trainer == 'distillation' else args.get('hidden_size', 768)
            moe = args['teacher_use_moe'] if trainer == 'distillation' else args.get('use_moe')
            suffix = '_moe' if moe else ''
            inside(root, 'out/' + base + '_' + str(dimension) + suffix + '.pth', exists=True)
            if trainer == 'distillation':
                args['from_teacher_weight'] = base
                args.pop('from_weight', None)
    args.setdefault('data_cache_max_gb', opts['cache_max_gb'])
    if trainer in {'pretrain', 'cpt'}:
        args.setdefault('streaming_chunk_mb', 128)
        args.setdefault('streaming_prefetch_chunks', 0)
    count = request.get('nproc', 1)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 8:
        raise ValueError('nproc must be between 1 and 8')
    return trainer, args, config, model, audit, prefix, count


def command(root, job, resume=False):
    args = dict(job['args'])
    args.update(save_dir=str(root / 'out'), pause_file=str(run_dir(root, job['id']) / 'pause.request'),
                config_path=job['config_path'], from_resume=int(resume))
    args['lora_name' if job['trainer'] == 'lora' else 'save_weight'] = job['save_weight']
    script = TRAINERS[job['trainer']]
    architecture = job['model_config'].get('model_architecture', 'standard')
    files = [script]
    if architecture in {'recurrsive_instinct', 'linear'}:
        files.insert(0, 'run_recurrsive.py' if architecture == 'recurrsive_instinct' else 'run_linear.py')
    argv = [job['settings']['python']]
    if job['nproc'] > 1:
        argv += ['-m', 'torch.distributed.run', '--nproc_per_node', str(job['nproc']), '--master_port', str(job['master_port'])]
    argv += files
    for key, value in sorted(args.items()):
        if key in FLAGS:
            if value: argv.append('--' + key)
        else:
            argv += ['--' + key, str(int(value)) if isinstance(value, bool) else str(value)]
    return argv


def detach_outputs(root, job):
    # Historical storage maintenance may have hard-linked exported weights.
    for path in (root / 'out').glob(job['save_weight'] + '_*.pth'):
        path = inside(root, str(path), exists=True)
        if path.stat().st_nlink > 1:
            temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                shutil.copy2(path, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


def spawn_supervisor(root, job, resume):
    directory = run_dir(root, job['id'])
    for name in ['pause.request', 'pause.intent', 'stop.request']:
        (directory / name).unlink(missing_ok=True)
    job.update(status='starting', started_at=now(), finished_at=None, exit_code=None, error=None,
               pid=None, process_identity=None, supervisor_pid=None, supervisor_identity=None)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0)); job['master_port'] = probe.getsockname()[1]
    job['command'] = command(root, job, resume)
    save(directory / 'job.json', job)
    with (directory / 'supervisor.log').open('ab', buffering=0) as log:
        worker = subprocess.Popen([job['settings']['python'], str(manager(root) / 'manager.py'),
                                   '--worker', str(root), job['id'], str(job['attempt'])],
                                  cwd=root, stdout=log, stderr=log, start_new_session=True,
                                  env=environment(root, job['settings']))
    job['supervisor_pid'], job['supervisor_identity'] = worker.pid, identity(worker.pid)
    threading.Thread(target=worker.wait, daemon=True).start()
    save(directory / 'job.json', job)
    return job


def start(root, opts, request):
    with control_lock(root):
        existing = lookup_operation(root, request.get('operation_id'))
        if existing: return existing
    trainer, args, config, model, audit, prefix, count = prepare(root, opts, request)
    gpu_check(root, opts, count)
    with control_lock(root):
        existing = lookup_operation(root, request.get('operation_id'))
        if existing: return existing
        idle(root)
        if shutil.disk_usage(root).free < opts['min_free_gb'] * 1024**3:
            raise ValueError('Insufficient free disk space for configured training reserve')
        identifier = 'r_' + uuid.uuid4().hex
        directory = run_dir(root, identifier)
        directory.mkdir()
        # Snapshot both editable-panel and selected-file configurations per run.
        config = directory / 'model_config.json'
        save(config, model)
        prefix = prefix or trainer + '_relay_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + identifier[2:10]
        job = {'id': identifier, 'name': str(request.get('name') or prefix)[:120], 'trainer': trainer,
               'instance_uuid': opts['instance_uuid'], 'created_at': now(), 'attempt': 0, 'attempts': [],
               'save_weight': prefix, 'args': args, 'model_config': model, 'config_path': str(config),
               'settings': {key: opts[key] for key in ['python', 'min_free_gb', 'cache_max_gb', 'gpu_devices']},
               'nproc': count, 'master_port': 0, 'audit': audit,
               'operation_id': request.get('operation_id'),
               'panel_cache_source': request.get('panel_cache_source')}
        job['settings'].update(shutdown_after_train=bool(request.get('shutdown_after_train', True)),
                               shutdown_delay_seconds=max(0, min(3600, int(request.get('shutdown_delay_seconds', 60)))))
        detach_outputs(root, job)
        if shutil.disk_usage(root).free < opts['min_free_gb'] * 1024**3:
            raise ValueError('Insufficient space after separating shared output weights')
        return spawn_supervisor(root, job, bool(request.get('existing_resume')))


def resume(root, opts, identifier):
    with control_lock(root):
        job = read_job(root, identifier)
        if opts.get('operation_id') and job.get('resume_operation_id') == opts['operation_id']:
            return job
        if not job['can_resume']:
            raise ValueError('Job cannot resume: no saved full checkpoint or process is still active')
        if job['instance_uuid'] != opts['instance_uuid']:
            raise ValueError('Job belongs to a different instance')
        # Resume keeps the saved corpus; no identity review or full-file hash pass.
        audit_data(root, {}, dict(job['args']))
        idle(root)
        gpu_check(root, job['settings'], job['nproc'])
        if shutil.disk_usage(root).free < job['settings']['min_free_gb'] * 1024**3:
            raise ValueError('Insufficient free disk space to resume')
        detach_outputs(root, job)
        if shutil.disk_usage(root).free < job['settings']['min_free_gb'] * 1024**3:
            raise ValueError('Insufficient space after separating shared output weights')
        job['attempt'] += 1
        job['resume_operation_id'] = opts.get('operation_id')
        job['settings'].update(shutdown_after_train=bool(opts.get('shutdown_after_train', True)),
                               shutdown_delay_seconds=max(0, min(3600, int(opts.get('shutdown_delay_seconds', 60)))))
        return spawn_supervisor(root, job, True)


def action(root, identifier, operation):
    with control_lock(root):
        job = read_job(root, identifier)
        if operation == 'pause':
            if job['status'] not in {'running', 'pausing', 'orphaned'}:
                raise ValueError('Wait for the training process to start before pausing')
            (run_dir(root, identifier) / 'pause.request').touch()
            (run_dir(root, identifier) / 'pause.intent').touch()
            job['status'] = 'pausing' if job['status'] != 'orphaned' else 'orphaned'
        elif operation == 'stop':
            if job['status'] not in ACTIVE:
                raise ValueError('Training job is not active')
            (run_dir(root, identifier) / 'stop.request').touch()
            if job['status'] == 'orphaned' and alive(job.get('pid'), job.get('process_identity')):
                os.killpg(job['pid'], signal.SIGTERM)
            job['status'] = 'stopping'
        save(run_dir(root, identifier) / 'job.json', job)
        return job


def shutdown_after_success(root, identifier, attempt):
    """Remote supervisor owns shutdown; no browser/relay connection or API secret required.

    AutoDL documents /usr/bin/shutdown as the in-instance power-off command.
    Keep the guard and command in one control lock so a managed trainer cannot
    start between the last idle check and power-off.
    """
    directory = run_dir(root, identifier)
    with control_lock(root):
        job = read_job(root, identifier)
        if job['attempt'] != attempt or job['status'] != 'completed' or job.get('exit_code') != 0 or not job['settings'].get('shutdown_after_train', False):
            return
        delay = job['settings'].get('shutdown_delay_seconds', 60)
        job.update(shutdown_status='scheduled', shutdown_at=now()+delay)
        save(directory/'job.json', job)
    time.sleep(delay)
    with control_lock(root):
        job = read_job(root, identifier)
        if job['attempt'] != attempt or job['status'] != 'completed' or not job['settings'].get('shutdown_after_train', False):
            return
        try:
            idle(root)
        except ValueError as error:
            job.update(shutdown_status='skipped', shutdown_error=str(error))
            save(directory/'job.json', job)
            return
        job.update(shutdown_status='requested', shutdown_requested_at=now())
        save(directory/'job.json', job)
        try:
            result = subprocess.run(['/usr/bin/shutdown'], cwd=root, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError((result.stderr or result.stdout or f'exit {result.returncode}')[-2000:])
        except Exception as error:
            job.update(shutdown_status='failed', shutdown_error=str(error))
            save(directory/'job.json', job)
        # Never rewrite the trainer's successful exit status after a shutdown error.


def worker(root, identifier, attempt):
    directory = run_dir(root, identifier)
    child = None
    job = None
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    try:
        with control_lock(root):
            job = read_job(root, identifier)
            if job['attempt'] != attempt or job['status'] not in {'starting', 'stopping'}:
                return
            if (directory / 'stop.request').exists():
                job.update(status='cancelled', exit_code=None, finished_at=now())
                save(directory / 'job.json', job); return
            logfile = directory / ('attempt-' + str(attempt) + '.log')
            with logfile.open('ab', buffering=0) as output:
                output.write(('[relay_training] ' + shlex.join(job['command']) + '\n').encode())
                child = subprocess.Popen(job['command'], cwd=root, stdout=output, stderr=output,
                                         start_new_session=True, env=environment(root, job['settings'], identifier))
            job.update(status='running', pid=child.pid, process_identity=identity(child.pid))
            job['attempts'].append({'attempt': attempt, 'started_at': now(), 'log': str(logfile)})
            save(directory / 'job.json', job)
        while child.poll() is None:
            # Trainers clear stale pause markers during startup; retain operator intent.
            if (directory / 'pause.intent').exists():
                (directory / 'pause.request').touch()
            if stopping[0] or (directory / 'stop.request').exists():
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                break
            time.sleep(0.3)
        code = child.returncode
        status = 'cancelled' if stopping[0] or (directory / 'stop.request').exists() else 'paused' if code == 42 else 'completed' if code == 0 else 'failed'
        with control_lock(root):
            current = read_job(root, identifier)
            if current['attempt'] != attempt: return
            current.update(status=status, exit_code=code, finished_at=now(),
                           error=None if status in {'completed', 'paused', 'cancelled'} else 'Trainer exited with code ' + str(code))
            current['attempts'][-1].update(status=status, exit_code=code, finished_at=now())
            save(directory / 'job.json', current)
    except Exception as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        with control_lock(root):
            path = directory / 'job.json'
            if path.is_file():
                current = json.loads(path.read_text())
                if current['attempt'] == attempt:
                    current.update(status='failed', finished_at=now(), error=str(error))
                    save(path, current)
        raise
    else:
        try:
            shutdown_after_success(root, identifier, attempt)
        except Exception as error:
            print('Automatic shutdown failed: ' + str(error), file=sys.stderr, flush=True)


def inventory(root, opts):
    def files(folder, suffixes, include_reports=False):
        path = root / folder
        return [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size} for p in sorted(path.iterdir())
                if p.is_file() and not p.is_symlink() and p.name.endswith(suffixes)
                and (include_reports or not p.name.endswith('.report.json'))] if path.is_dir() else []
    gpu = subprocess.run(['sh', '-c', 'command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    return {'repo_root': str(root), 'python': opts['python'], 'free_bytes': shutil.disk_usage(root).free,
            'min_free_gb': opts['min_free_gb'], 'cache_max_gb': opts['cache_max_gb'],
            'gpu': gpu.stdout.strip().splitlines() if gpu.returncode == 0 else [],
            'datasets': files('dataset', ('.jsonl', '.jsonl.gz', '.parquet')),
            'reports': files('dataset', ('.report.json',), include_reports=True),
            'configs': files('trainer', ('.json',)), 'weights': files('out', ('.pth',)),
            'resume_checkpoints': files('checkpoints', ('_resume.pth',)),
            'unmanaged_trainers': unmanaged_trainers()}


def dispatch(opts):
    operation = opts['action']
    root = repository(opts, operation != 'assert_idle')
    if operation == 'check_repo': return {'repo_root': str(root)}
    if operation == 'assert_idle':
        if manager(root).exists():
            with control_lock(root): idle(root, allow_code_sync=opts.get('allow_code_sync', False))
        elif unmanaged_trainers(): raise ValueError('Training is active on this instance')
        return {'idle': True}
    if operation == 'inventory': return inventory(root, opts)
    if operation == 'lookup_operation':
        with control_lock(root): return {'job': lookup_operation(root, opts.get('operation_id'))}
    if operation == 'gpu_ready':
        probe = subprocess.run([opts['python'], '-c', 'import torch,json;print(json.dumps({"count":torch.cuda.device_count()}))'],
                               cwd=root, env=environment(root, opts), capture_output=True, text=True, timeout=60)
        if probe.returncode:
            raise ValueError('GPU Python environment probe failed: ' + probe.stderr[-1000:])
        return {'ready': json.loads(probe.stdout.strip())['count'] > 0}
    if operation == 'preflight':
        with control_lock(root):
            idle(root)
            if opts.get('id'):
                job = read_job(root, opts['id'])
                if not job['can_resume'] or job['instance_uuid'] != opts['instance_uuid']:
                    raise ValueError('Job cannot resume on this instance')
                audit_data(root, {}, dict(job['args']))
                minimum = job['settings']['min_free_gb']
            else:
                prepare(root, opts, opts['request'])
                environment(root, opts)
                minimum = opts['min_free_gb']
            if shutil.disk_usage(root).free < minimum * 1024**3:
                raise ValueError('Insufficient free disk space for configured training reserve')
        return {'ready': True}
    if operation == 'start': return start(root, opts, opts['request'])
    if operation == 'resume': return resume(root, opts, opts['id'])
    if operation in {'pause', 'stop'}: return action(root, opts['id'], operation)
    if operation == 'list':
        if not manager(root).exists(): return {'jobs': []}
        with control_lock(root): return {'jobs': all_jobs(root)}
    if operation in {'get', 'log'}:
        with control_lock(root): job = read_job(root, opts['id'])
        if operation == 'get': return job
        attempt = int(opts.get('attempt', job['attempt']))
        if not 0 <= attempt <= job['attempt']: raise ValueError('Invalid log attempt')
        path = run_dir(root, job['id']) / ('attempt-' + str(attempt) + '.log')
        if not path.exists(): path = run_dir(root, job['id']) / 'supervisor.log'
        limit = max(1024, min(int(opts.get('bytes', 131072)), 1048576))
        text = ''
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - limit)); text = stream.read(limit).decode('utf-8', errors='replace')
        return {'id': job['id'], 'attempt': attempt, 'status': job['status'], 'text': text}
    raise ValueError('Unknown training operation')


if __name__ == '__main__':
    if sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]).resolve(), sys.argv[3], int(sys.argv[4]))
    else:
        print(json.dumps(dispatch(json.loads(sys.argv[1])), ensure_ascii=False))
