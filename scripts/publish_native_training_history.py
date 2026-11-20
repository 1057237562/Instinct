"""Join the AutoDL and local 24B logs, and publish read-only monitoring data over SSH."""
import argparse
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import zipfile

import psutil

from scripts.native_training_monitor import snapshot
from trainer.training_eta import RecentThroughputETA


ROOT = Path(__file__).resolve().parents[1]
MONITOR = 'root@81.68.133.63'
PUBLISH_DIR = '/var/lib/ssh-log-monitor/instinct-24b'
FILES = ('training.log', 'status.json')
NUMBER = r'[+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf)'
METRIC = re.compile(r'Epoch:\[\d+/\d+\]\((\d+)/\d+\), loss: (' + NUMBER +
                    r'), learning_rate: (' + NUMBER + r'), epoch_time: ' + NUMBER +
                    r'min, tokens/s: (' + NUMBER + r')')
CHECKPOINT = re.compile(r'\[Native Checkpoint\] step=\d+ (?:staged [\d.]+ GiB in [\d.]+s; writing in background|'
                        r'background write completed in [\d.]+s)')

# The upload can only replace these two generated files in the dedicated directory.
# Keep the serving process, application source, and its configuration out of the update.
INSTALL = r'''
import io, os, sys, tempfile, zipfile
from pathlib import Path
root = Path(sys.argv[1])
root.mkdir(mode=0o750, parents=True, exist_ok=True)
if root.resolve() != root:
    raise ValueError('Unexpected publication path')
files = ('training.log', 'status.json')
with zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())) as archive:
    if tuple(archive.namelist()) != files:
        raise ValueError('Unexpected snapshot members')
    for name in files:
        data = archive.read(name)
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as file:
            try:
                file.write(data)
                file.close()
                os.chmod(file.name, 0o640)
                os.replace(file.name, root / name)
            finally:
                Path(file.name).unlink(missing_ok=True)
'''


def split_cloud(raw, step):
    marker = re.compile(rb'^\[Native Checkpoint\] step=' + str(step).encode() +
                        rb' background write completed in [\d.]+s\r?\n', re.MULTILINE)
    matches = list(marker.finditer(raw))
    if len(matches) != 1:
        raise ValueError(f'Expected one verified cloud checkpoint boundary at {step}')
    end = matches[0].end()
    return raw[:end], raw[end:]


def public_log(raw):
    """Publish only known metric/checkpoint records, without paths or tracebacks."""
    lines, points = [], []
    for line in raw.decode('utf-8', errors='replace').splitlines():
        match = METRIC.fullmatch(line)
        if match:
            cursor, loss, _, rate = match.groups()
            lines.append(line)
            if 'nan' not in loss and 'inf' not in loss:
                points.append((int(cursor), float(loss), float(rate)))
        elif CHECKPOINT.fullmatch(line):
            lines.append(line)
    return '\n'.join(lines), points


def telemetry(state, handoff, rate):
    status = {key: state['status'].get(key) for key in (
        'status', 'pid', 'steps', 'updates', 'accumulated', 'parquet_cursor',
        'processed_valid_tokens', 'updated_at', 'process_created',
    )}
    fields = ('index,name,utilization.gpu,utilization.memory,memory.used,memory.total,'
              'power.draw,power.limit,temperature.gpu,clocks.current.graphics,clocks.max.graphics')
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
                                      text=True, timeout=5, stderr=subprocess.DEVNULL,
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        gpu = ''
    mem, disk = psutil.virtual_memory(), psutil.disk_usage(str(ROOT.anchor))
    system = dict(cores=psutil.cpu_count(), uptimeSeconds=time.time() - psutil.boot_time(),
                  memory=dict(totalBytes=mem.total, usedBytes=mem.total - mem.available),
                  disk=dict(totalBytes=disk.total, usedBytes=disk.used))
    return json.dumps(dict(status=status, alive=state['alive'], phase=state['phase'],
                           plannedTokens=state['record']['planned_tokens'],
                           handoff=handoff, tokensPerSec=rate, gpu=gpu, system=system)).encode()


def publish(data):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.writestr(name, data[name])
    command = shlex.join(['python3', '-c', INSTALL, PUBLISH_DIR])
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                    '-o', 'ConnectTimeout=8', MONITOR, command], input=bundle.getvalue(),
                   check=True, timeout=45)


def write_file(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cloud_stdout', type=Path)
    parser.add_argument('launch_report', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    launch = json.loads(args.launch_report.read_text(encoding='utf-8'))
    step, start_tokens = launch['start_step'], launch['start_tokens']
    cloud, replayed = split_cloud(args.cloud_stdout.read_bytes(), step)
    cloud_log, cloud_points = public_log(cloud)
    divider = f'\n--- Local continuation at checkpoint step {step:,} ({start_tokens:,} tokens) ---\n'
    args.output.mkdir(parents=True, exist_ok=True)
    write_file(args.output / 'cloud.replayed.stdout.log', replayed)
    manifest = ROOT / 'checkpoints/native_cublas/training_run.json'
    last_status, published = None, None
    estimator = RecentThroughputETA()
    while True:
        try:
            state = snapshot(manifest)
            if not state or state.get('error'):
                raise ValueError('Local training status is unavailable')
            status, record = state['status'], state['record']
            if Path(record['status_path']).resolve() != Path(launch['status_path']).resolve():
                raise ValueError('The registered training job changed')
            if status['processed_valid_tokens'] < start_tokens:
                raise ValueError('Local status is earlier than the cloud handoff')
            with Path(record['stdout_path']).open('rb') as handle:
                handle.seek(launch['offsets']['stdout'])
                local = handle.read()
            if local and not local.endswith(b'\n'):
                local = local[:local.rfind(b'\n') + 1]
            current = (status['process_created'], status['updated_at'])
            if last_status != current:
                if last_status and current[0] != last_status[0]:
                    estimator = RecentThroughputETA()
                estimator.update(status['processed_valid_tokens'], current[1], record['planned_tokens'])
                last_status = current
            revision = (current, len(local), state['phase'])
            if revision != published:
                local_log, local_points = public_log(local)
                merged = '--- AutoDL training ---\n' + cloud_log + '\n' + divider + local_log + '\n'
                write_file(args.output / 'merged.stdout.log', cloud + divider.encode() + local)
                handoff = dict(step=step, tokens=start_tokens, cloudPoints=len(cloud_points),
                               localPoints=len(local_points))
                rate = estimator.rate if state['phase'] == 'running' else None
                publish({'training.log': merged.encode(), 'status.json': telemetry(state, handoff, rate)})
                published = revision
                print(f"Published step {status['steps']:,}: {len(cloud_points):,} cloud, {len(local_points):,} local loss points",
                      flush=True)
            if args.once or state['phase'] == 'completed':
                return
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            if args.once:
                raise
            print(f'Publication will retry: {error}', flush=True)
        time.sleep(20)


if __name__ == '__main__':
    main()
