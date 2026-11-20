"""CPU-only bridge between a native training worktree and Config WebUI.

The manifest lives beside shared checkpoints; panel settings remain independent.
Only the registered Python process (including its creation time) owns the run.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from zipfile import BadZipFile
from functools import lru_cache


@lru_cache(maxsize=8)
def shared_manifest(repo_root):
    """Worktrees share the primary checkout's artifact directory."""
    result = subprocess.run(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                            cwd=repo_root, capture_output=True, text=True, timeout=5)
    root = Path(result.stdout.strip()).parent if result.returncode == 0 else Path(repo_root)
    return root / 'checkpoints' / 'native_cublas' / 'training_run.json'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def register(path, *, runtime_root, python, status_path, stdout_path, stderr_path,
             pause_path, resume_path, planned_tokens, environment=None, data_path=None):
    record = dict(schema=1, backend='cublas', name='Instinct V2 Dense · MXFP8',
                  runtime_root=str(Path(runtime_root).resolve()), python=str(Path(python).absolute()),
                  planned_tokens=int(planned_tokens), environment=environment or {})
    for key, value in dict(status_path=status_path, stdout_path=stdout_path,
                           stderr_path=stderr_path, pause_path=pause_path,
                           resume_path=resume_path).items():
        record[key] = str(Path(value).resolve())
    if data_path is not None:
        record['data_path'] = str(Path(data_path).resolve())
    write_json(path, record)


def _read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('训练记录格式无效')
    return value


def process_matches(record, status):
    import psutil
    try:
        process = psutil.Process(int(status['pid']))
        if abs(process.create_time() - float(status['process_created'])) > 0.1:
            return False
        command = process.cmdline()
        return (Path(process.exe()).resolve() == Path(record['python']).resolve()
                and Path(process.cwd()).resolve() == Path(record['runtime_root']).resolve()
                and any(command[i:i+2] == ['-m', 'experiments.run_native_1b_acceptance']
                        for i in range(len(command) - 1)))
    except (psutil.Error, KeyError, TypeError, ValueError, OSError):
        return False


def verify_resume(record, status):
    # NPZ loads only the metadata member, never the model/optimizer tensors.
    import numpy as np
    with np.load(record['resume_path'], allow_pickle=False) as archive:
        metadata = json.loads(str(archive['__metadata__']))
    if metadata.get('format') != 'instinct-cublas-v1':
        raise ValueError('不是原生 cuBLAS 完整检查点')
    progress = metadata['progress']
    optimizer = metadata['optimizers']['model']
    if (optimizer['accumulated'] != 0 or status.get('accumulated') != 0
            or progress.get('resume_valid') is False
            or progress['steps'] != status['steps']
            or progress['processed_tokens'] != status['processed_valid_tokens']
            or optimizer['updates'] != status['updates']
            or progress['parquet_cursor'] != status['parquet_cursor']):
        raise ValueError('检查点与已保存的训练进度不一致，不能续训')
    return progress


def snapshot(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        record = _read(path)
        if record.get('schema') != 1 or record.get('backend') != 'cublas':
            raise ValueError('不支持的原生训练记录')
        status = _read(record['status_path'])
        launch_path = path.with_suffix('.launch.json')
        if launch_path.exists():
            launch = _read(launch_path)
            if (launch['pid'] != status.get('pid')
                    and launch['process_created'] > status.get('updated_at', 0)
                    and process_matches(record, launch)):
                status = dict(status, **launch, status='starting')
        if Path(status['resume_path']).resolve() != Path(record['resume_path']).resolve():
            raise ValueError('训练记录的检查点路径不一致')
        alive = process_matches(record, status)
        phase = status.get('status', 'unknown')
        if phase in ('starting', 'running', 'failed_held') and not alive:
            phase = 'interrupted'
        elif phase in ('paused', 'completed') and alive:
            phase = 'saving'
        return dict(record=record, status=status, phase=phase, alive=alive,
                    pause_requested=Path(record['pause_path']).exists())
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(error=str(error), alive=False, phase='invalid')


def request_pause(path):
    state = snapshot(path)
    if not state or not state['alive'] or state['phase'] not in ('starting', 'running'):
        raise ValueError('没有可暂停的原生训练进程')
    Path(state['record']['pause_path']).touch()


def resume_run(path):
    """Continue this exact checkpoint through its original native entry point."""
    import psutil
    path = Path(path)
    lock = path.with_suffix('.launch.lock')
    # Serialize clicks from independent browser sessions. Never remove a lock
    # belonging to another launch, including after a failed exclusive open.
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.close(fd)
        state = snapshot(path)
        if not state or state['phase'] != 'paused' or state['alive']:
            raise ValueError('当前原生训练尚未保存并暂停')
        record, status = state['record'], state['status']
        verify_resume(record, status)
        for process in psutil.process_iter(['name', 'cmdline']):
            if (process.info.get('name') or '').lower() not in ('python', 'python.exe'):
                continue
            arguments = process.info.get('cmdline') or []
            if any(Path(arg).name.startswith('train_') and arg.endswith('.py')
                   or arg == 'experiments.run_native_1b_acceptance' for arg in arguments):
                raise ValueError('已有训练进程，未重复启动')
        marker = Path(record['pause_path'])
        if marker.exists():
            raise ValueError('暂停请求仍未清除，请检查训练保存状态')
        command = [record['python'], '-u', '-m', 'experiments.run_native_1b_acceptance',
                   '--resume-path', record['resume_path']]
        if record.get('data_path'):
            command.extend(['--data-path', record['data_path']])
        environment = dict(os.environ, PYTHONUTF8='1')
        environment.update(record.get('environment', {}))
        with open(record['stdout_path'], 'a', encoding='utf-8') as output, \
                open(record['stderr_path'], 'a', encoding='utf-8') as errors:
            process = subprocess.Popen(command, cwd=record['runtime_root'],
                env=environment, stdout=output, stderr=errors,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        # Keep launch ownership separate: the parent must never overwrite a
        # newer status already published by a quickly initialized child.
        created = psutil.Process(process.pid).create_time()
        write_json(path.with_suffix('.launch.json'), dict(pid=process.pid,
                   process_created=created, updated_at=time.time()))
        return process.pid
    finally:
        lock.unlink()


def _render_recent_rate(st, status, tokens, total):
    from trainer.training_eta import RecentThroughputETA

    # Fragment refreshes can read the same trainer report several times.
    # Count it once and reset the window when a different trainer takes over.
    key = '_native_training_eta'
    identity = (status['pid'], status['process_created'])
    timestamp = status['updated_at']
    sample = st.session_state.get(key)
    if sample is None or sample['identity'] != identity:
        sample = dict(identity=identity, timestamp=None, estimator=RecentThroughputETA(), eta=None)
    if timestamp != sample['timestamp']:
        sample['eta'] = sample['estimator'].update(tokens, timestamp, total)
        sample['timestamp'] = timestamp
        st.session_state[key] = sample

    rate, eta = sample['estimator'].rate, sample['eta']
    if rate is None or eta is None:
        st.caption('等待下一次 token 游标更新以估算速度和剩余时间。')
        return
    st.caption(f'最近 {sample["estimator"].window_seconds:.0f} 秒平均 {rate:,.0f} token/s · '
               f'预计剩余 {eta / 60:,.1f} 小时')


def render(st, path, *, read_log, render_charts):
    """Render without changing the editable model preset or Torch run state."""
    state = snapshot(path)
    if state is None:
        return
    st.subheader('原生 cuBLAS 生产训练')
    if state.get('error'):
        st.error(state['error'])
        return
    record, status, phase = state['record'], state['status'], state['phase']
    captions = dict(starting='正在初始化 / 续训', running='训练中', saving='正在完成保存',
                    paused='已保存并暂停', completed='已完成', interrupted='进程已退出，待检查',
                    failed_held='训练异常，进程保留以保护当前进度')
    st.info(captions.get(phase, phase) + ' · ' + record['name'])
    steps = int(status.get('steps', 0))
    tokens = int(status.get('processed_valid_tokens', 0))
    total = int(record['planned_tokens'])
    st.progress(min(tokens / max(total, 1), 1.0),
                text=f'{tokens:,} / {total:,} tokens · {steps:,} 微步 · {status.get("updates", 0):,} 次更新')
    if phase == 'running':
        _render_recent_rate(st, status, tokens, total)
    loss = status.get('result', {}).get('loss')
    if loss is not None:
        st.caption(f'保存时 loss：{loss:.6f}')
    if phase in ('starting', 'running') and state['alive']:
        if st.button('暂停并保存原生训练', key='native_pause', disabled=state['pause_requested']):
            try:
                request_pause(path)
                st.success('已请求暂停；等待累计更新完成并保存完整续训状态。')
            except (OSError, ValueError) as error:
                st.error(str(error))
        if state['pause_requested']:
            st.caption('暂停请求已发送，正在等待保存。')
    if phase == 'paused':
        problem = None
        try:
            verify_resume(record, status)
        except (OSError, ValueError, KeyError, BadZipFile) as error:
            problem = str(error)
            st.error('续训状态校验失败：' + problem)
        if st.button('继续原生生产训练', key='native_resume', disabled=problem is not None):
            try:
                resume_run(path)
                st.rerun(scope='app')
            except (OSError, ValueError) as error:
                st.error(str(error))
    st.caption('完整续训检查点：' + record['resume_path'])
    if record.get('data_path'):
        st.caption('本次续训数据：' + record['data_path'])
    log, metrics = read_log(record['stdout_path'])
    with st.expander('原生训练指标与日志', expanded=True):
        if metrics:
            # Native logs estimate time from the step counter, which spans
            # launches. The live token estimate above is valid after resuming.
            render_charts([{key: value for key, value in row.items() if key != 'epoch_time'}
                           for row in metrics])
        st.code(log[-12000:] or '(等待日志)', language='text', line_numbers=False)
        errors, _ = read_log(record['stderr_path'])
        if errors.strip():
            st.code(errors[-6000:], language='text', line_numbers=False)
