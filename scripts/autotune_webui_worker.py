"""Detached CPU-only supervisor for WebUI autotuning and owned-tree cleanup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autotune_panel import ROOT, read_json, write_json


def stop_owned(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    else:
        import signal
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    request = read_json(run_dir / 'request.json')
    identity = dict(pid=os.getpid(), create_time=psutil.Process().create_time())
    write_json(run_dir / 'run.json', dict(identity, status='running'))
    process = None
    result = dict(identity, status='failed')
    try:
        options = dict(cwd=ROOT, env=dict(os.environ, PYTHONUTF8='1', INSTINCT_MANAGED_DATA_CACHE='1'))
        if os.name == 'nt':
            options['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options['start_new_session'] = True
        with (run_dir / 'controller.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(request['command'], stdout=log, stderr=subprocess.STDOUT, **options)
            while process.poll() is None:
                if (run_dir / 'stop').exists():
                    stop_owned(process)
                    result['status'] = 'cancelled'
                    break
                time.sleep(.5)
            else:
                result.update(status='success' if process.returncode == 0 else 'failed', exit_code=process.returncode)
    except BaseException as exc:
        result['error'] = str(exc)
        raise
    finally:
        if process is not None:
            stop_owned(process)
        write_json(run_dir / 'run.json', result)


if __name__ == '__main__':
    main()
