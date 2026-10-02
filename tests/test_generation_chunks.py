from datasets import load_dataset  # noqa: F401
import torch

from model.generation_stream import TokenChunkBuffer
from scripts.web_demo_utils import animated_stream_html


class Streamer:
    def __init__(self):
        self.blocks = []

    def put(self, tokens):
        self.blocks.append(tokens.clone())


def test_transfers_only_at_16_token_boundaries_and_flushes_tail(monkeypatch):
    streamer = Streamer()
    chunks = TokenChunkBuffer(streamer, 16, 2)
    calls = []
    original = torch.Tensor.cpu
    def cpu(tensor, *args, **kwargs):
        calls.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    for i in range(35):
        assert not chunks.push(torch.tensor([[i + 3]]), torch.tensor([False]), final=i == 34)
    assert calls == [(1, 17), (1, 17), (1, 4)]
    assert [block.shape[1] for block in streamer.blocks] == [16, 16, 3]
    assert torch.cat(streamer.blocks, dim=1).tolist() == [list(range(3, 38))]


def test_eos_at_different_positions_trims_only_unused_suffix():
    streamer = Streamer()
    chunks = TokenChunkBuffer(streamer, 16, 2)
    for step in range(32):
        finished = torch.tensor([step >= 2, step >= 19])
        token = torch.where(finished[:, None], 2, 7)
        done = chunks.push(token, finished)
        assert done == (step == 31)
    assert chunks.overshoot == 12
    assert [block.shape[1] for block in streamer.blocks] == [16, 4]
    rows = torch.cat(streamer.blocks, dim=1)
    assert rows[0, :3].tolist() == [7, 7, 2]
    assert rows[1, -1].item() == 2


def test_animation_escapes_content_and_preserves_reduced_motion():
    markup = animated_stream_html('<old>', '<script>x</script>')
    assert '<script>' not in markup
    assert '&lt;old&gt;' in markup
    assert 'animation-delay:' in markup
    assert 'prefers-reduced-motion' in markup


def test_code_fences_are_rendered_as_html_not_markdown():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
from scripts.web_demo_utils import render_animated_stream
render_animated_stream(st.empty(), "### 示例：\\n\\n```python\\ndef gcd(a, b):\\n    return a\\n", "```\\n\\n注意")
''').run()
    assert not app.exception
    assert len(app.markdown) == 0
    elements = app.get('html')
    assert len(elements) == 1
    assert 'def gcd(a, b):' in elements[0].proto.body
    assert '<span style=' in elements[0].proto.body
