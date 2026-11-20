"""Read-only native production health probe, invoked by relay remote-script."""
import json
import math
import re
from pathlib import Path
import time

import psutil


def main():
    root = Path('/autodl-tmp')
    system = Path('/root/instinct-runtime')
    now = time.time()
    files = []
    inputs = [
        (root/'dataset/pretrain_coder_l2l3_24b_shards_5gib/part-00001-of-00005.parquet', 5368243357),
        (root/'checkpoints/native_cublas/native_v2_1b_acceptance_1536_resume.npz', 14203664750),
    ]
    for target, total in inputs:
        candidates = [target, Path(str(target)+'.incoming')]
        present = next((p for p in candidates if p.is_file()), None)
        if present:
            done = present.stat().st_size
        else:
            state = Path(str(target)+'.incoming.ranges.json')
            if state.is_file():
                done = json.loads(state.read_text())['prefix']
                for part in target.parent.glob(target.name+'.incoming.segment-*'):
                    try: done += part.stat().st_size
                    except FileNotFoundError: pass
            else:
                part = Path(str(target)+'.incoming.rpart')
                done = part.stat().st_size if part.is_file() else 0
        files.append(dict(file=target.name, total_bytes=total,
                          transferred_bytes=min(done, total), verified_final=target.is_file()))
    launch = system/'launch_status.json'
    record_path = root/'checkpoints/native_cublas/training_run.json'
    result = dict(time=now, files=files,
                  launch=json.loads(launch.read_text()) if launch.is_file() else None,
                  healthy=False, reasons=['training_not_registered'])
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        status_path = Path(record['status_path'])
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        alive = False
        try:
            process = psutil.Process(int(status['pid']))
            arguments = process.cmdline()
            alive = (abs(process.create_time()-status['process_created']) < .1
                     and Path(process.exe()).resolve() == Path(record['python']).resolve()
                     and Path(process.cwd()).resolve() == Path(record['runtime_root']).resolve()
                     and 'experiments.run_native_1b_acceptance' in arguments
                     and process.status() != psutil.STATUS_ZOMBIE)
        except (psutil.Error, KeyError, TypeError, ValueError): pass
        output = Path(record['stdout_path'])
        errors = Path(record['stderr_path'])
        text = output.read_text(errors='replace') if output.is_file() else ''
        error_text = errors.read_text(errors='replace') if errors.is_file() else ''
        losses = re.findall(r'loss:\s*([^,\s]+)', text)
        loss = None
        try: loss = float(losses[-1]) if losses else None
        except ValueError: pass
        fatal = re.findall(r'Traceback|CUDA error|illegal memory access|out of memory|FloatingPointError|nonfinite|nan loss',
                           text+'\n'+error_text, flags=re.I)
        reasons = []
        launch = result.get('launch') or {}
        same_run = (launch.get('status') == 'launched' and launch.get('pid') == status.get('pid')
                    and abs(launch.get('process_created', 0)-status.get('process_created', 0)) < .1)
        baseline = launch.get('steps_before') if same_run else None
        if baseline is None: reasons.append('current_run_baseline_not_verified')
        if not alive: reasons.append('process_identity_not_alive')
        if status.get('status') != 'running': reasons.append('phase_'+str(status.get('status')))
        if not 0 <= now-status.get('updated_at', 0) <= 180: reasons.append('stale_status')
        if baseline is None or status.get('steps', 0) < baseline+1000: reasons.append('fewer_than_1000_new_microsteps')
        if status.get('processed_valid_tokens', 0) <= 134347008: reasons.append('tokens_not_advanced')
        if loss is None or not math.isfinite(loss): reasons.append('no_finite_training_loss')
        if fatal: reasons.append('fatal_log_markers')
        result.update(healthy=not reasons, reasons=reasons, status=status, baseline_steps=baseline,
                      pid=status.get('pid'), process_created=status.get('process_created'),
                      last_loss=loss, fatal_markers=fatal, log_tail=text[-2000:], error_tail=error_text[-3000:])
        result['warmup_complete'] = 'Backend=cuBLASLt/Triton' in text
        result['stage'] = ('initial_data_preparation' if result['warmup_complete'] and status.get('status') == 'starting'
                           else status.get('status'))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
