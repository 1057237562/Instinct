"""Command construction and background jobs for the evaluation UI (no torch)."""
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_config(checkpoint, root=ROOT):
    """Match trainer_utils' exact sidecar convention; never use latest unrelated config."""
    path = Path(checkpoint)
    if not path.is_absolute():
        path = Path(root) / path
    for candidate in (path.with_suffix('.json'), path.parent.parent / 'checkpoints' / (path.stem + '.json'),
                      Path(root) / 'checkpoints' / (path.stem + '.json')):
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding='utf-8'))
            if not isinstance(payload, dict) or not {'hidden_size', 'num_hidden_layers'} <= payload.keys():
                raise ValueError(f'模型配置缺少 hidden_size / num_hidden_layers：{candidate}')
            return str(candidate), payload
    return None, None


def build_command(config):
    c = config
    kind = c['benchmark']
    mode = c.get('mode', 'generate')
    if kind not in ('HumanEval', 'LiveCodeBench', 'GSM8K', '推理自动测试', 'ToolCall'):
        raise ValueError('未知评测类型')
    if kind in ('HumanEval', 'LiveCodeBench', 'GSM8K') and c.get('k'):
        ks = [int(k) for k in c['k'].replace(',', ' ').split()]
        if not ks or min(ks) < 1:
            raise ValueError('K 必须是正整数列表')
        if mode == 'all' and c.get('num_samples', 1) < max(ks):
            raise ValueError(f'计算 pass@{max(ks)} 需要每题至少 {max(ks)} 个样本，请增加每题样本数')
    command = [sys.executable, '-u', str(ROOT / (
        'eval_humaneval.py' if kind == 'HumanEval' else
        'eval_gsm8k.py' if kind == 'GSM8K' else
        'scripts/eval_toolcall.py' if kind == 'ToolCall' else 'eval_llm.py'))]

    def add(flag, value):
        if value is not None and value != '':
            command.extend(['--' + flag, str(int(value) if isinstance(value, bool) else value)])

    if mode != 'evaluate' or kind in ('ToolCall', '推理自动测试'):
        if 'checkpoint_path' in c:
            checkpoint = Path(c['checkpoint_path'])
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
            if not c['checkpoint_path'] or not checkpoint.is_file():
                raise ValueError('请选择存在的模型权重文件')
            if c.get('require_model_config') and not c.get('config_path'):
                raise ValueError('该权重没有同名模型配置，请在模型高级设置中指定训练时的 config JSON')
        if 'load_from' in c and not c['load_from']:
            raise ValueError('请选择模型目录')
        for key in ('load_from', 'save_dir', 'weight', 'checkpoint_path', 'config_path', 'hidden_size', 'num_hidden_layers',
                    'use_moe', 'device', 'max_new_tokens', 'temperature', 'top_p'):
            add(key, c.get(key))
        if kind != 'ToolCall':
            for key in ('model_architecture', 'residual_type', 'lora_weight',
                        'open_thinking', 'hc_mult', 'hc_sinkhorn_iters',
                        'attnres_variant', 'attnres_block_size', 'eval_kv_cache_dtype'):
                add(key, c.get(key))
        if c.get('num_samples', 1) > 1 and c.get('temperature', 0) <= 0:
            raise ValueError('多样本生成需要 temperature > 0')
        if kind == 'ToolCall' and c.get('temperature', 0) <= 0:
            raise ValueError('ToolCall 当前采样接口需要 temperature > 0')
    if kind in ('HumanEval', 'GSM8K'):
        for key in ('mode', 'output', 'problem_file', 'limit', 'num_samples', 'batch_size', 'prompt_style',
                    'seed', 'workers', 'timeout'):
            if kind != 'GSM8K' or key not in ('workers', 'timeout'):
                add(key, c.get(key))
        ks = [int(k) for k in c.get('k', '1,10,100').replace(',', ' ').split()]
        if not ks or min(ks) < 1:
            raise ValueError('k 必须是正整数列表')
        command.extend(['--k', *map(str, ks)])
        if c.get('resume'):
            command.append('--resume')
    elif kind == 'LiveCodeBench':
        if mode != 'generate' and c.get('limit', 0):
            raise ValueError('LiveCodeBench 官方评分要求 limit = 0')
        if mode != 'generate' and c.get('problem_file'):
            raise ValueError('本地 LiveCodeBench 数据仅用于生成；官方评分请使用匹配的数据版本')
        add('benchmark', 'livecodebench')
        if c.get('k'):
            command.extend(['--lcb_k', *map(str, ks)])
        for key, flag in {'output': 'output', 'problem_file': 'dataset_path', 'limit': 'limit',
                          'num_samples': 'num_samples', 'batch_size': 'batch_size', 'prompt_style': 'prompt_style',
                          'seed': 'seed', 'resume': 'resume', 'release_version': 'release_version',
                          'start_date': 'start_date', 'end_date': 'end_date',
                          'runner_path': 'runner_path', 'workers': 'num_process_evaluate',
                          'timeout': 'timeout'}.items():
            add('lcb_' + flag, c.get(key))
        if mode != 'generate':
            command.append('--lcb_evaluate_only' if mode == 'evaluate' else '--lcb_evaluate')
    if kind in ('HumanEval', 'LiveCodeBench', 'GSM8K'):
        output = Path(c['output'])
        if not output.is_absolute():
            output = ROOT / output
        if mode == 'evaluate' and not output.is_file():
            raise ValueError('待评分的答案文件不存在')
        if mode != 'evaluate' and output.exists() and not c.get('resume'):
            raise ValueError('输出文件已存在，请启用续跑或更换输出路径')
    return command


def tail_log(path, size=64000):
    with open(path, 'rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - size))
        return stream.read().decode('utf-8', errors='replace')


class JobManager:
    """One shared process per UI server, surviving browser reruns/reconnections."""
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.process = None
        self.current = None

    def _save(self):
        path = Path(self.current['directory']) / 'run.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.current, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)

    def status(self):
        with self.lock:
            if self.process is not None and self.current['status'] == 'running':
                code = self.process.poll()
                if code is not None:
                    self.current.update(status='completed' if code == 0 else 'failed', exit_code=code)
                    self._save()
            return dict(self.current) if self.current else None

    def start(self, command, config):
        with self.lock:
            if self.status() and self.current['status'] == 'running':
                raise ValueError('已有评测任务运行中')
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
            directory = self.root / 'eval' / 'runs' / run_id
            directory.mkdir(parents=True)
            self.current = dict(id=run_id, directory=str(directory), command=command,
                                config=dict(config), status='running', exit_code=None)
            kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
            try:
                with (directory / 'run.log').open('wb') as log:
                    self.process = subprocess.Popen(
                        command, cwd=self.root, stdin=subprocess.PIPE,
                        stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1'}, **kwargs)
                if config['benchmark'] in ('ToolCall', '推理自动测试'):
                    try:
                        self.process.stdin.write(b'0\n')
                        self.process.stdin.flush()
                    except BrokenPipeError:
                        pass
                self.process.stdin.close()
            except Exception:
                self.current['status'] = 'failed'
                self._save()
                raise
            self._save()
            process = self.process

            def record_exit():
                process.wait()
                with self.lock:
                    if self.process is process:
                        self.status()

            threading.Thread(target=record_exit, daemon=True).start()
            return self.status()

    def stop(self):
        with self.lock:
            if not self.status() or self.current['status'] != 'running':
                return
            try:
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                if os.name != 'nt':
                    os.killpg(self.process.pid, signal.SIGKILL)
                else:
                    # Suspend parent first so it cannot launch another evaluation worker.
                    parent.suspend()
                    for process in reversed(children):
                        try:
                            process.kill()
                        except psutil.NoSuchProcess:
                            pass
                    parent.kill()
                psutil.wait_procs(children, timeout=3)
                self.process.wait(timeout=5)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
            self.current.update(status='stopped', exit_code=self.process.poll())
            self._save()


def history(root=ROOT):
    rows = []
    paths = list((Path(root) / 'eval' / 'runs').glob('*/run.json'))
    paths.extend((Path(root) / 'out' / 'eval_runs').glob('*/run.json'))
    for path in sorted(paths, key=lambda p: p.parent.name, reverse=True):
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
            rows.append(row)
        except (ValueError, OSError):
            continue
    return rows
