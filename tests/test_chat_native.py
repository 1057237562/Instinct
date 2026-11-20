import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from scripts import chat_native
from scripts.chat_native_worker import inference_config, stream_tokens
from scripts.web_demo_utils import resolve_model_config_path


def checkpoint(path, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(format='instinct-cublas-v1', config=dict(hidden_size=16), **extra)
    np.savez(path, __metadata__=json.dumps(metadata), **{'model/x': np.ones(2)})
    return path


def test_metadata_and_config_do_not_fall_back(tmp_path):
    path = checkpoint(tmp_path / 'resume.npz', progress={'steps': 123})
    assert chat_native.read_native_metadata(path)['progress']['steps'] == 123
    assert resolve_model_config_path(str(path), str(tmp_path)) == str(path)
    assert resolve_model_config_path(str(tmp_path / 'missing.npz'), str(tmp_path)) == ''
    np.savez(path, __metadata__=json.dumps({'format': 'other', 'config': {}}))
    with pytest.raises(ValueError, match='Unsupported'):
        chat_native.read_native_metadata(path)


def test_discovery_nested_resume_newest_and_archives(tmp_path):
    import os
    old = checkpoint(tmp_path / 'out/native/old.npz')
    new = checkpoint(tmp_path / 'checkpoints/native/run/model_resume.npz')
    checkpoint(tmp_path / 'checkpoints/_archive/ignored.npz')
    (tmp_path / 'out/legacy.pth').touch()
    (tmp_path / 'checkpoints/legacy_resume.pth').touch()
    (tmp_path / 'out/incomplete.npz.tmp').touch()
    os.utime(old, (1, 1))
    os.utime(new, (2_000_000_000, 2_000_000_000))
    paths = list(chat_native.discover_chat_weights(tmp_path).values())
    assert paths[0] == str(new)
    assert set(paths) == {str(new), str(old), str(tmp_path / 'out/legacy.pth')}


def test_inference_config_preserves_topology_and_rope():
    source = dict(hidden_size=1536, rope_scaling={'rope_type': 'longrope'},
                  native_fp8_recipe='mxfp8_experimental', native_muon_gram_recurrence='fp16_restart3',
                  native_mxfp8_cache_ffn=True, use_grad_checkpoint=4)
    result = inference_config(source)
    assert result == dict(hidden_size=1536, rope_scaling=source['rope_scaling'], use_grad_checkpoint=0)
    assert source['use_grad_checkpoint'] == 4


class FakeModel:
    def __init__(self):
        self.config = dict(max_position_embeddings=6, vocab_size=5)
        self.calls, self.released = [], []
        self.e = SimpleNamespace(
            backend=SimpleNamespace(mark=lambda: 1, release_since=lambda *a, **k: self.released.append(k)),
            no_grad=contextlib.nullcontext, numpy=np.asarray)

    def __call__(self, ids, cache, use_cache):
        self.calls.append(ids.tolist())
        position = (cache['position'] if cache else 0) + ids.shape[1]
        return SimpleNamespace(logits=np.ones((1, ids.shape[1], 5)), cache={'position': position})

    def cache_tensors(self, cache):
        return []


def request(**extra):
    return dict(tokens=[1, 2], seed=42, max_new_tokens=10, eos_token_id=[4],
                temperature=.8, top_k=2, top_p=.9, repetition_penalty=1.1, **extra)


def test_cached_stream_eos_and_cleanup():
    model = FakeModel()
    choices = iter([3, 4, 0])
    histories = []
    def sample(logits, rng, history, *args):
        histories.append(history)
        assert args == (.8, 2, .9, 1.1)
        return next(choices), 0
    assert list(stream_tokens(model, request(), threading.Event(), sample)) == [3, 4]
    assert model.calls == [[[1, 2]], [[3]]]
    assert histories == [[1, 2], [1, 2, 3]]
    assert model.released[-1] == {}


def test_context_budget_cancel_and_failure_cleanup():
    model = FakeModel()
    assert len(list(stream_tokens(model, request(), threading.Event(), lambda *a: (3, 0)))) == 4
    cancelled = threading.Event()
    cancelled.set()
    assert list(stream_tokens(model, request(), cancelled, lambda *a: (3, 0))) == []
    with pytest.raises(ValueError, match='Prompt'):
        list(stream_tokens(model, dict(request(), tokens=[1] * 6), cancelled, None))
    def fail(*args):
        raise RuntimeError('sampling failed')
    with pytest.raises(RuntimeError, match='sampling failed'):
        list(stream_tokens(model, request(), threading.Event(), fail))
    assert model.released[-1] == {}


def test_longrope_rebuilds_prefix():
    model = FakeModel()
    model.short_rope = True
    model.config['rope_scaling'] = {'original_max_position_embeddings': 2}
    list(stream_tokens(model, dict(request(), max_new_tokens=2), threading.Event(), lambda *a: (3, 0)))
    assert model.calls == [[[1, 2]], [[1, 2, 3]]]


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'weights.npz')
    worker = tmp_path / 'fake_worker.py'
    worker.write_text('''import json,sys
def send(**event):
 print(json.dumps(event),flush=True)
send(event='ready',config={'hidden_size':16})
for line in sys.stdin:
 command=json.loads(line)
 if command['op']=='close': break
 if command['op']=='generate':
  if command['tokens']==[9]: send(event='error',message='bad prompt')
  else:
   send(event='token',token=3)
   send(event='token',token=4)
  send(event='done')
''', encoding='utf-8')
    monkeypatch.setattr(chat_native, 'native_runtime', lambda _: (tmp_path, __import__('sys').executable, dict(__import__('os').environ)))
    popen = chat_native.subprocess.Popen
    monkeypatch.setattr(chat_native.subprocess, 'Popen', lambda args, **kw: popen([args[0], '-u', str(worker)], **kw))
    model = chat_native.NativeChatModel(path, tmp_path, timeout=10)
    yield model
    model.close()
    assert model.process.poll() == 0


class Streamer:
    def __init__(self, cancel=False):
        self.values, self.ended, self.cancel = [], False, cancel
    def put(self, value):
        self.values.append(value.tolist())
        if self.cancel and len(self.values) == 2:
            raise RuntimeError('cancelled')
    def end(self):
        self.ended = True


def test_process_protocol_cancellation_then_second_turn(adapter):
    with pytest.raises(RuntimeError, match='cancelled'):
        adapter.generate(torch.tensor([[1, 2]]), Streamer(cancel=True))
    stream = Streamer()
    adapter.generate(torch.tensor([[0, 1, 2]]), stream, attention_mask=torch.tensor([[0, 1, 1]]))
    assert stream.values == [[[1, 2]], [3], [4]]
    assert stream.ended


def test_process_error_does_not_poison_next_turn(adapter):
    with pytest.raises(RuntimeError, match='bad prompt'):
        adapter.generate(torch.tensor([[9]]), Streamer())
    stream = Streamer()
    adapter.generate(torch.tensor([[1, 2]]), stream)
    assert stream.ended


def test_webui_npz_load_unload(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from transformers import AutoTokenizer
    from model.model_instinct import InstinctConfig
    path = checkpoint(tmp_path / 'weights.npz', progress={'steps': 123})
    monkeypatch.setattr(chat_native, 'discover_chat_weights', lambda _: {'latest.npz': str(path)})
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: object())
    instances = []
    class Model:
        backend = 'cuBLASLt · BF16'
        supports_logit_lens = False
        runtime_version = chat_native.NATIVE_CHAT_RUNTIME_VERSION
        def __init__(self, weight, root):
            assert weight == str(path)
            self.config = InstinctConfig(hidden_size=16)
            self.closed = False
            instances.append(self)
        def to(self, device):
            assert device == 'cpu'
            self.closed = True
    monkeypatch.setattr(chat_native, 'NativeChatModel', Model)
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'scripts/web_demo.py')).run(timeout=60)
    assert not app.exception
    assert app.session_state.weight_path == str(path)
    load = next(b for b in app.sidebar.button if 'Load Model' in b.label)
    load.click().run(timeout=60)
    assert not app.exception
    assert app.session_state.model_loaded
    assert next(c for c in app.sidebar.checkbox if c.label == 'Logit Lens').disabled
    next(b for b in app.sidebar.button if 'Unload Model' in b.label).click().run(timeout=60)
    assert not app.exception
    assert instances[0].closed
