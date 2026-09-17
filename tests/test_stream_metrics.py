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
