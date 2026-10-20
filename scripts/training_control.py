"""CPU-only control of a trainer in the selected local repository."""

import os
from pathlib import Path
import re
import subprocess

import psutil


TRAINERS = {
    'train_pretrain.py', 'train_full_sft.py', 'train_lora.py', 'train_dpo.py',
    'train_ppo.py', 'train_grpo.py', 'train_agent.py', 'train_distillation.py',
}


def local_training_process(pid, repo_root):
    """Validate the executable script, including topology wrappers, before control."""
    process = psutil.Process(pid)
    command = process.cmdline()
    if not command or not re.fullmatch(r'(?:python(?:\d+(?:\.\d+)*)?w?|pypy\d*)(?:\.exe)?',
                                      Path(command[0]).name, re.IGNORECASE):
        raise ValueError('该进程不是 Python 训练进程，未执行终止。')
    arguments = command[1:]
    if any(str(arg).split('=', 1)[0] == '--autotune_trial' for arg in arguments):
        raise ValueError('自动调优进程请通过自动调优面板停止。')
    # Only interpreter options may precede the actual script. In particular,
    # never treat a train_*.py string inside python -c/-m or a flag as a script.
    while arguments and arguments[0] in ('-u', '-B', '-O', '-OO', '-E', '-s', '-S', '-I'):
        arguments = arguments[1:]
    root = Path(repo_root).resolve()

    def script_path(argument):
        path = Path(argument)
        return (path if path.is_absolute() else Path(process.cwd()) / path).resolve()

    if arguments and script_path(arguments[0]) in (
        root / 'run_linear.py', root / 'run_recurrsive.py',
    ):
        arguments = arguments[1:]
    if not arguments or script_path(arguments[0]) not in {
        root / 'trainer' / name for name in TRAINERS
    }:
        raise ValueError('该进程不是当前仓库的训练进程，未执行终止。')
    # Cache the process identity; psutil's control methods also check PID reuse.
    process.create_time()
    return process


def stop_training_process(pid, repo_root):
    """Stop only this validated trainer and its workers; never save a checkpoint."""
    try:
        return _stop_training_process(pid, repo_root)
    except (psutil.Error, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'无法终止训练进程：{exc}') from exc


def _stop_training_process(pid, repo_root):
    try:
        process = local_training_process(pid, repo_root)
    except psutil.NoSuchProcess:
        return False
    # Freeze the trainer so it cannot spawn workers or finish normally between
    # validation and tree termination (which must have a nonzero exit status).
    process.suspend()
    try:
        children = process.children(recursive=True)
        if os.name == 'nt':
            result = subprocess.run(
                ['taskkill', '/PID', str(process.pid), '/T', '/F'],
                capture_output=True, timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode:
                raise RuntimeError(
                    '终止训练失败：' + (result.stderr or result.stdout).decode(errors='replace')
                )
        else:
            process.kill()
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        _, alive = psutil.wait_procs([process, *children], timeout=5)
        if alive:
            raise RuntimeError('训练子进程仍未退出，请重试终止。')
    finally:
        try:
            if process.is_running():
                process.resume()
        except psutil.NoSuchProcess:
            pass
    return True
