"""Generation failures and reruns must not leave a live CUDA producer behind."""

from queue import Queue
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from scripts.chat_generation import GenerationTask, stop_generation


class QueueStreamer:
    def __init__(self):
        self.queue = Queue()
        self.ends = 0

    def put(self, value):
        self.queue.put(value)

    def end(self):
        self.ends += 1
        self.queue.put(None)

    def __iter__(self):
        while True:
            value = self.queue.get(timeout=5)
            if value is None:
                return
            yield value


def test_success_ends_stream_once_after_generate_returns():
    streamer = QueueStreamer()
    returned = Event()

    def generate(streamer):
        streamer.put('answer')
        streamer.end()
        returned.set()

    task = GenerationTask(SimpleNamespace(generate=generate), {'streamer': streamer})
    task.start()
    assert list(streamer) == ['answer']
    assert returned.is_set()
    task.raise_if_failed()
    assert task.stop()
    assert streamer.ends == 1


@pytest.mark.parametrize('partial', [False, True])
def test_failure_ends_stream_and_reaches_consumer(partial):
    streamer = QueueStreamer()

    def generate(streamer):
        if partial:
            streamer.put('partial answer')
        raise RuntimeError('mat2 is on cpu')

    task = GenerationTask(SimpleNamespace(generate=generate), {'streamer': streamer})
    task.start()
    assert list(streamer) == (['partial answer'] if partial else [])
    with pytest.raises(RuntimeError, match='mat2 is on cpu'):
        task.raise_if_failed()
    assert task.stop()
    assert streamer.ends == 1


def test_busy_producer_stays_attached_until_it_can_stop():
    entered, release = Event(), Event()
    streamer = QueueStreamer()
    forwarded = []

    def generate(streamer):
        entered.set()
        assert release.wait(5)
        streamer.put('should be cancelled')
        forwarded.append(True)

    task = GenerationTask(SimpleNamespace(generate=generate), {'streamer': streamer})
    state = {'generation_task': task, 'model_loaded': True}
    task.start()
    try:
        assert entered.wait(5)
        assert not stop_generation(state, timeout=0.01)
        assert state['generation_task'] is task
    finally:
        release.set()
        assert stop_generation(state)
    assert list(streamer) == []
    assert not forwarded
    assert state == {'model_loaded': True}
    task.raise_if_failed()  # cancellation is a normal rerun, not an error


@pytest.mark.gpu
@pytest.mark.parametrize('static_cache', [False, True])
def test_cuda_generation_stops_before_model_is_moved_to_cpu(static_cache):
    from model.model_instinct import InstinctConfig, InstinctForCausalLM

    config = InstinctConfig(hidden_size=32, num_hidden_layers=1, vocab_size=64,
                            num_attention_heads=4, num_key_value_heads=2,
                            intermediate_size=64)
    model = InstinctForCausalLM(config).cuda().eval()
    model._static_cache_ok = static_cache
    entered, release = Event(), Event()
    streamer = QueueStreamer()

    def pause_before_head(module, inputs):
        entered.set()
        assert release.wait(5)

    handle = model.lm_head.register_forward_pre_hook(pause_before_head)
    task = GenerationTask(model, {
        'input_ids': torch.tensor([[3, 4]], device='cuda'),
        'streamer': streamer, 'stream_chunk_size': 1,
        'max_new_tokens': 8, 'do_sample': False, 'eos_token_id': None,
    })
    state = {'generation_task': task}
    task.start()
    try:
        assert entered.wait(5)
        assert not stop_generation(state, timeout=0.01)
        assert model.lm_head.weight.device.type == 'cuda'
    finally:
        release.set()
        assert stop_generation(state)
        handle.remove()
    task.raise_if_failed()
    model.to('cpu')
    assert model.lm_head.weight.device.type == 'cpu'
    assert not task.thread.is_alive()
    assert streamer.ends == 1
