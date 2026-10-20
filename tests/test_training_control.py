"""Local stop controls must leave unrelated Python processes and checkpoints alone."""

import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from scripts import training_control


@pytest.mark.parametrize('arguments,accepted', [
    (['-u', 'trainer/train_pretrain.py'], True),
    (['trainer/train_full_sft.py', '--from_resume', '1'], True),
    (['run_linear.py', 'trainer/train_pretrain.py'], True),
    (['run_recurrsive.py', 'trainer/train_pretrain.py'], True),
    (['-c', 'trainer/train_pretrain.py'], False),
    (['-m', 'trainer.train_pretrain'], False),
    (['other.py', '--data', 'trainer/train_pretrain.py'], False),
    (['trainer/train_pretrain.py', '--autotune_trial', '1'], False),
    (['trainer/train_pretrain.py', '--autotune_trial=1'], False),
    (['../other/trainer/train_pretrain.py'], False),
])
def test_process_validation_uses_actual_repo_script(tmp_path, monkeypatch, arguments, accepted):
    class Process:
        def cmdline(self):
            return [sys.executable, *arguments]

        def cwd(self):
            return str(tmp_path)

        def create_time(self):
            return 123.0

    process = Process()
    monkeypatch.setattr(training_control.psutil, 'Process', lambda _: process)
    if accepted:
        assert training_control.local_training_process(123, tmp_path) is process
    else:
        with pytest.raises(ValueError):
            training_control.stop_training_process(123, tmp_path)


def test_stop_rejects_the_webui_or_an_unrelated_python_process(tmp_path):
    with pytest.raises(ValueError):
        training_control.stop_training_process(os.getpid(), tmp_path)


def test_stop_terminates_only_trainer_tree(tmp_path):
    trainer = tmp_path / 'trainer' / 'train_pretrain.py'
    trainer.parent.mkdir()
    trainer.write_text(
        'import pathlib, subprocess, sys, time\n'
        'worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])\n'
        'pathlib.Path("worker.pid").write_text(str(worker.pid))\n'
        'time.sleep(120)\n', encoding='utf-8',
    )
    checkpoint = tmp_path / 'existing_resume.pth'
    checkpoint.write_bytes(b'keep this saved checkpoint')
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], **options)
    process = subprocess.Popen([sys.executable, '-u', str(trainer)], cwd=tmp_path, **options)
    worker = None
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / 'worker.pid').exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.02)
        worker = psutil.Process(int((tmp_path / 'worker.pid').read_text()))
        assert training_control.stop_training_process(process.pid, tmp_path)
        assert process.wait(timeout=5) != 0
        assert not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
        assert checkpoint.read_bytes() == b'keep this saved checkpoint'
    finally:
        for owned in (process, unrelated):
            if owned.poll() is None:
                owned.kill()
            owned.wait(timeout=5)
        if worker is not None and worker.is_running():
            try:
                worker.kill()
            except psutil.NoSuchProcess:
                pass
