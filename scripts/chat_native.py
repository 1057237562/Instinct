"""CPU checkpoint discovery and a session-owned native cuBLAS process adapter."""
from collections import deque
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import weakref
from zipfile import BadZipFile

import numpy as np

NATIVE_CHAT_RUNTIME_VERSION = 2


def read_native_metadata(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['__metadata__']))
    except (BadZipFile, EOFError, KeyError, TypeError) as exc:
        raise ValueError(f'Invalid native NPZ metadata: {exc}') from exc
    if (not isinstance(metadata, dict) or metadata.get('format') != 'instinct-cublas-v1'
            or not isinstance(metadata.get('config'), dict)):
        raise ValueError('Unsupported NPZ checkpoint; expected instinct-cublas-v1 metadata')
    return metadata


def native_runtime(repo_root):
    root = Path(repo_root)
    from scripts.native_training_monitor import shared_manifest
    manifest = shared_manifest(str(root))
    record = json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else {}
    runtime = Path(os.environ.get('INSTINCT_NATIVE_RUNTIME_ROOT') or
                   (root if (root / 'model/dense_cublas').is_dir() else record.get('runtime_root', root)))
    if not (runtime / 'model/dense_cublas/model.py').is_file():
        raise FileNotFoundError('Native cuBLAS runtime missing; set INSTINCT_NATIVE_RUNTIME_ROOT to its checkout')
    python = os.environ.get('INSTINCT_NATIVE_PYTHON') or record.get('python') or sys.executable
    env = dict(os.environ, **record.get('environment', {}), PYTHONUTF8='1')
    return runtime, python, env


def discover_chat_weights(repo_root):
    """Include native resume archives, omit archives/temp files; newest first."""
    root = Path(repo_root)
    paths = []
    for folder in ('out', 'checkpoints'):
        directory = root / folder
        if not directory.is_dir():
            continue
        paths.extend(p for p in directory.glob('*.pth') if '_resume' not in p.name)
        paths.extend(p for p in directory.rglob('*.npz')
                     if not any(part.startswith(('_', '.')) for part in p.relative_to(directory).parts[:-1]))
    entries = []
    for path in paths:
        try:
            stat = path.stat()
            if path.is_file():
                entries.append((stat.st_mtime_ns, path, stat.st_size))
        except OSError:
            continue  # A checkpoint may be atomically replaced during discovery.
    entries.sort(key=lambda entry: (entry[0], str(entry[1])), reverse=True)
    return {f'{p.relative_to(root).as_posix()}  ({size / 2**20:.0f} MB)': str(p)
            for _, p, size in entries}


def _close_worker(process, readers):
    if process.poll() is None:
        try:
            process.stdin.write('{"op":"close"}\n')
            process.stdin.flush()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            # This process is exclusively an inference worker, never a trainer.
            process.kill()
            process.wait(timeout=10)
    for reader in readers:
        reader.join(timeout=5)
    for pipe in (process.stdin, process.stdout, process.stderr):
        pipe.close()


class NativeChatModel:
    backend = 'cuBLASLt · BF16'
    supports_logit_lens = False
    input_device = 'cpu'
    runtime_version = NATIVE_CHAT_RUNTIME_VERSION

    def __init__(self, checkpoint, repo_root, timeout=600):
        from model.model_instinct import InstinctConfig
        self.checkpoint_metadata = read_native_metadata(checkpoint)
        runtime, python, env = native_runtime(repo_root)
        self._events, self._errors = queue.Queue(), deque(maxlen=40)
        self._lock = threading.Lock()
        self.process = subprocess.Popen(
            [str(python), '-u', str(Path(__file__).with_name('chat_native_worker.py')),
             str(runtime), str(Path(checkpoint).resolve())],
            cwd=runtime, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        readers = (
            threading.Thread(target=self._read, args=(self.process.stdout, self._events, self._errors), daemon=True),
            threading.Thread(target=self._read_errors, args=(self.process.stderr, self._errors), daemon=True),
        )
        for reader in readers:
            reader.start()
        # Reader targets must not retain self: an abandoned session can then
        # collect its adapter and close its private CUDA process as well.
        self._cleanup = weakref.finalize(self, _close_worker, self.process, readers)
        try:
            event = self._next(timeout)
            if event['event'] != 'ready':
                raise RuntimeError(event.get('message', str(event)))
            self.config = InstinctConfig(**event['config'])
            self.fast_decode = event.get('fast_decode', False)
            if event.get('cuda_graph'):
                self.backend = 'cuBLASLt · BF16 · CUDA Graph'
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _read(pipe, events, errors):
        try:
            for line in pipe:
                try:
                    events.put(json.loads(line))
                except ValueError:
                    errors.append(line.rstrip())
        finally:
            events.put({'event': 'exit'})

    @staticmethod
    def _read_errors(pipe, errors):
        for line in pipe:
            errors.append(line.rstrip())

    def _next(self, timeout=600):
        try:
            event = self._events.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError('Native inference timed out: ' + '\n'.join(self._errors)) from None
        if event['event'] == 'exit':
            raise RuntimeError('Native inference process exited: ' + '\n'.join(self._errors))
        return event

    def _send(self, command):
        self.process.stdin.write(json.dumps(command) + '\n')
        self.process.stdin.flush()

    def generate(self, input_ids, streamer, attention_mask=None, **kwargs):
        import torch
        if kwargs.get('layer_callback') is not None:
            raise ValueError('Logit lens is unavailable for native cuBLAS inference')
        if len(input_ids) != 1 or kwargs.get('num_return_sequences', 1) != 1:
            raise ValueError('Native chat supports one sequence per request')
        tokens = input_ids[0].tolist()
        if attention_mask is not None:
            tokens = [token for token, keep in zip(tokens, attention_mask[0].tolist()) if keep]
        with self._lock:
            timing = getattr(streamer, 'enable_token_timing', None)
            if timing is not None:
                timing()
            streamer.put(torch.tensor([tokens], dtype=torch.long))
            self._send(dict(op='generate', tokens=tokens,
                            seed=int(np.random.randint(0, 2**32, dtype=np.int64)),
                            max_new_tokens=int(kwargs.get('max_new_tokens', 256)),
                            temperature=float(kwargs.get('temperature', .9)) if kwargs.get('do_sample', True) else 0.,
                            top_k=int(kwargs.get('top_k', 0)), top_p=float(kwargs.get('top_p', .9)),
                            repetition_penalty=float(kwargs.get('repetition_penalty', 1.)),
                            eos_token_id=kwargs.get('eos_token_id')))
            error = None
            while True:
                event = self._next()
                if event['event'] == 'done':
                    break
                if event['event'] == 'error':
                    error = RuntimeError(event['message'])
                elif event['event'] == 'token' and error is None:
                    try:
                        streamer.put(torch.tensor([event['token']], dtype=torch.long))
                    except BaseException as exc:
                        # Drain through done before releasing the model to another request.
                        error = exc
                        self._send({'op': 'cancel'})
            if error is not None:
                raise error
            streamer.end()

    def close(self):
        self._cleanup()

    def to(self, device):
        if str(device) == 'cpu':
            self.close()
        return self
