"""Count actual streamed token IDs without additional GPU synchronization."""
import time
from queue import Empty


class TokenRateStreamer:
    def __init__(self, streamer, clock=time.perf_counter):
        self.streamer = streamer
        self.clock = clock
        self.started = None
        self.state = (0, None)
        self.first = True

    def put(self, tokens):
        now = self.clock()
        if self.first:
            self.first = False
            self.started = now
        else:
            count, _ = self.state
            self.state = (count + tokens.numel(), now)
        self.streamer.put(tokens)

    def end(self):
        count, _ = self.state
        self.state = (count, self.clock())
        self.streamer.end()

    def snapshot(self):
        count, end = self.state
        elapsed = max(0, end - self.started) if end is not None and self.started is not None else 0
        return {'tokens': count, 'seconds': elapsed, 'tokens_per_second': count / elapsed if elapsed else 0}

    def __iter__(self):
        return iter(self.streamer)


def speed_caption(stats):
    return f"{stats['tokens_per_second']:.1f} tokens/s · {stats['tokens']} tokens · {stats['seconds']:.2f}s（含首轮处理，不含前端动画）"


def render_updates(streamer, interval=0.1, clock=time.perf_counter):
    """Coalesce ready chunks, cap UI updates at 10 Hz, always flush the final text."""
    previous_time = None
    text, pending = '', []
    delegate = getattr(streamer, 'streamer', None)
    queue = getattr(delegate, 'text_queue', None)
    for chunk in streamer:
        pending.append(chunk)
        if queue is not None:
            for _ in range(256):
                try:
                    extra = queue.get_nowait()
                except Empty:
                    break
                if extra == delegate.stop_signal:
                    queue.put_nowait(extra)
                    break
                pending.append(extra)
        now = clock()
        effective_interval = max(interval, min(0.3, len(text) / 200000))
        if previous_time is not None and now - previous_time < effective_interval:
            continue
        addition = ''.join(pending)
        pending.clear()
        if addition:
            text += addition
            previous_time = now
            yield text, False
    text += ''.join(pending)
    yield text, True
