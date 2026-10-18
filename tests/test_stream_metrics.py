from types import SimpleNamespace

from scripts.stream_metrics import TokenRateStreamer, speed_caption, render_updates


def test_counts_blocks_excludes_prompt_and_freezes_elapsed():
    times = iter([10.0, 12.0, 13.0, 14.0])
    forwarded = []
    delegate = SimpleNamespace(put=forwarded.append, end=lambda: None)
    streamer = TokenRateStreamer(delegate, clock=lambda: next(times))
    def tokens(n):
        return SimpleNamespace(numel=lambda: n)
    streamer.put(tokens(100))
    streamer.put(tokens(16))
    assert streamer.snapshot()['tokens_per_second'] == 8
    streamer.put(tokens(4))
    streamer.end()
    assert streamer.snapshot() == {'tokens': 20, 'seconds': 4.0, 'tokens_per_second': 5.0}
    assert streamer.snapshot()['seconds'] == 4
    assert len(forwarded) == 3
    assert '5.0 tokens/s' in speed_caption(streamer.snapshot())


def test_decode_marker_excludes_prefill_time():
    times = iter([10.0, 20.0, 21.0, 22.0])
    delegate = SimpleNamespace(put=lambda value: None, end=lambda: None)
    streamer = TokenRateStreamer(delegate, clock=lambda: next(times))
    token_block = lambda n: SimpleNamespace(numel=lambda: n)

    streamer.put(token_block(100))  # prompt submitted at t=10
    streamer.start_decode()         # prefill finishes at t=20
    streamer.put(token_block(16))   # first decoded block at t=21
    streamer.end()                  # generation finishes at t=22

    assert streamer.snapshot() == {
        'tokens': 16,
        'seconds': 2.0,
        'tokens_per_second': 8.0,
    }


def test_ui_updates_are_throttled_without_losing_text():
    times = iter(i * .01 for i in range(100))
    updates = list(render_updates(iter(['x'] * 100), clock=lambda: next(times)))
    assert len(updates) <= 12
    assert updates[-1] == ('x' * 100, True)


def test_ready_blocks_coalesce_and_end_marker_is_preserved():
    from queue import Queue
    class Delegate:
        stop_signal = None
        def __init__(self):
            self.text_queue = Queue()
            for item in ('a', 'b', 'c', None):
                self.text_queue.put(item)
        def __iter__(self):
            return self
        def __next__(self):
            item = self.text_queue.get_nowait()
            if item is None:
                raise StopIteration
            return item
    updates = list(render_updates(TokenRateStreamer(Delegate())))
    assert updates == [('abc', False), ('abc', True)]


def test_api_metrics_use_token_ids_and_exclude_prefill_from_tpot():
    from scripts.stream_metrics import GenerationMetricsStreamer
    times = iter([1.0, 2.0, 5.0, 6.0, 7.0])
    forwarded = []
    delegate = SimpleNamespace(put=forwarded.append, end=lambda: None)
    metrics = GenerationMetricsStreamer(delegate, request_started=0.0, clock=lambda: next(times))
    metrics.start_generation()  # t=1; request preprocessing took 1s.
    metrics.put(SimpleNamespace(numel=lambda: 100))  # Prompt consumes no timing samples.
    metrics.put(SimpleNamespace(numel=lambda: 1))    # First output at t=2.
    metrics.put(SimpleNamespace(numel=lambda: 3))    # Final block at t=5.
    metrics.finish_generation()                    # Generation returned at t=6.
    stats = metrics.snapshot(4)                    # Parsing completed at t=7.
    assert stats == {
        'output_tokens': 4, 'tokens_per_second': 1.0, 'tpot_ms': 1000.0,
        'ttft_ms': 2000.0, 'generation_ttft_ms': 1000.0, 'decode_ms': 3000.0,
        'generation_ms': 5000.0, 'total_ms': 7000.0,
        'generation_tokens_per_second': 0.8, 'end_to_end_tokens_per_second': 4 / 7,
        'timing_source': 'token_delivery',
    }
    assert len(forwarded) == 3


def test_api_metrics_single_token_has_ttft_but_no_tpot():
    from scripts.stream_metrics import GenerationMetricsStreamer
    times = iter([1.0, 2.0, 3.0, 4.0])
    metrics = GenerationMetricsStreamer(request_started=0.0, clock=lambda: next(times))
    metrics.start_generation()
    metrics.put(SimpleNamespace(numel=lambda: 100))
    metrics.put(SimpleNamespace(numel=lambda: 1))
    metrics.finish_generation()
    stats = metrics.snapshot(1)
    assert stats['ttft_ms'] == 2000
    assert stats['tpot_ms'] is None
    assert stats['tokens_per_second'] is None


def test_api_metrics_missing_token_callbacks_does_not_invent_ttft():
    from scripts.stream_metrics import GenerationMetricsStreamer
    times = iter([1.0, 3.0, 4.0])
    metrics = GenerationMetricsStreamer(request_started=0.0, clock=lambda: next(times))
    metrics.start_generation()
    metrics.finish_generation()
    stats = metrics.snapshot(4)
    assert stats['timing_source'] == 'generation_only'
    assert stats['ttft_ms'] is None
    assert stats['tpot_ms'] is None
    assert stats['generation_tokens_per_second'] == 2.0
