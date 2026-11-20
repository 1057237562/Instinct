"""Relay health checks and a guarded, user-authorized local shutdown."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'data_relay/relay_data/native5090/watchdog.json'
BASELINE = 11216
REQUIRED_STEPS = BASELINE + 1000


def qualifies(remote):
    status = remote.get('status') or {}
    baseline = remote.get('baseline_steps', BASELINE)
    return (remote.get('healthy') is True and status.get('status') == 'running'
            and baseline is not None and baseline >= BASELINE
            and status.get('steps', 0) >= baseline+1000
            and status.get('processed_valid_tokens', 0) > 134347008
            and remote.get('pid') and remote.get('process_created'))


def stable_again(remote, previous):
    first = previous.get('qualified')
    return bool(qualifies(remote) and first
        and remote['pid'] == first['pid']
        and remote['process_created'] == first['process_created']
        and remote['status']['steps'] > first['steps']
        and remote['time']-first['time'] >= 60)


def estimate(remote, samples):
    files = remote['files']
    remaining = sum(p['total_bytes']-p['transferred_bytes'] for p in files)
    rate = None
    if samples:
        first = samples[0]
        elapsed = remote['time']-first['time']
        transferred = sum(p['transferred_bytes'] for p in files)
        if elapsed > 0:
            rate = max(0, transferred-first['transferred'])/elapsed
    status = remote.get('status') or {}
    steps = status.get('steps', BASELINE)
    step_seconds = .75
    old = next((s for s in samples if s.get('steps', BASELINE) > BASELINE), None)
    if old and steps > old['steps']:
        step_seconds = max(.01, (remote['time']-old['time'])/(steps-old['steps']))
    transfer_seconds = remaining/rate if rate and rate > 0 else remaining/(6*1024*1024)
    launch = remote.get('launch') or {}
    cold_start_age = max(0, remote['time']-launch.get('updated_at', remote['time']))
    warmup_seconds = 0 if steps > BASELINE else max(120, 12*60-cold_start_age)
    total = transfer_seconds+warmup_seconds+max(0, REQUIRED_STEPS-steps)*step_seconds+10*60
    eta = datetime.fromtimestamp(remote['time'], timezone(timedelta(hours=8)))+timedelta(seconds=total)
    return dict(remaining_bytes=remaining, upload_mib_s=rate/2**20 if rate else None,
                estimated_minutes=round(total/60, 1), estimated_time=eta.isoformat())


def local_training_processes():
    import psutil
    found = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info.get('cmdline') or []
        if any(arg == 'experiments.run_native_1b_acceptance'
               or Path(arg).name.startswith('train_') and arg.endswith('.py') for arg in args):
            found.append(process.info['pid'])
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shutdown-if-ready', action='store_true')
    args = parser.parse_args()
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    if previous.get('shutdown_scheduled'):
        print(json.dumps(previous, ensure_ascii=False)); return
    command = [str(ROOT/'data_relay/bin/data_relay_manager.exe'), '--config',
        str(ROOT/'data_relay/relay_data/native5090/config.toml'), 'remote-script',
        '--remote-script', str(ROOT/'data_relay/src/native_training_watch_remote.py')]
    result = subprocess.run(command, cwd=ROOT/'data_relay', capture_output=True,
                            text=True, encoding='utf-8', timeout=90)
    if result.returncode:
        raise RuntimeError('Relay watchdog probe failed: '+result.stderr[-2000:]+result.stdout[-2000:])
    start = result.stdout.find('\n{')
    remote = json.loads(result.stdout[start+1:] if start >= 0 else result.stdout)
    samples = previous.get('samples', [])[-2:]
    state = dict(remote=remote, estimate=estimate(remote, samples), samples=samples,
                 checked_at=time.time(), shutdown_scheduled=False)
    sample = dict(time=remote['time'], transferred=sum(f['transferred_bytes'] for f in remote['files']),
                  steps=(remote.get('status') or {}).get('steps', BASELINE))
    state['samples'].append(sample)
    if qualifies(remote):
        state['qualified'] = previous.get('qualified') or dict(time=remote['time'], pid=remote['pid'],
            process_created=remote['process_created'], steps=remote['status']['steps'])
        if previous.get('qualified') and (remote['pid'], remote['process_created']) != (
                previous['qualified']['pid'], previous['qualified']['process_created']):
            state['qualified'] = dict(time=remote['time'], pid=remote['pid'],
                process_created=remote['process_created'], steps=remote['status']['steps'])
    ready = stable_again(remote, previous)
    state['ready_for_local_shutdown'] = ready
    if ready and args.shutdown_if_ready:
        active = local_training_processes()
        if active:
            state['shutdown_blocked_by_local_training_pids'] = active
        else:
            state['shutdown_scheduled'] = True
            state['shutdown_proof'] = dict(pid=remote['pid'], process_created=remote['process_created'],
                steps=remote['status']['steps'], new_microsteps=remote['status']['steps']-remote.get('baseline_steps', BASELINE),
                tokens=remote['status']['processed_valid_tokens'], loss=remote['last_loss'])
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(STATE)
    if state['shutdown_scheduled']:
        result = subprocess.run(['shutdown.exe', '/s', '/t', '60', '/c',
            'Remote 5090 training passed 1000 new microsteps and repeated health verification.'])
        if result.returncode:
            state['shutdown_scheduled'] = False
            state['shutdown_error_code'] = result.returncode
            STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in state.items() if k not in ('samples', 'remote')}, ensure_ascii=False))
    print(json.dumps(dict(remote_phase=(remote.get('status') or {}).get('status'),
        remote_steps=(remote.get('status') or {}).get('steps'), healthy=remote['healthy'], reasons=remote['reasons']), ensure_ascii=False))


if __name__ == '__main__':
    main()
