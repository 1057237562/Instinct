"""Count actual streamed token IDs without additional GPU synchronization."""
import time
from queue import Empty


class GenerationMetricsStreamer:
    """Per-request API timings measured at CPU token delivery boundaries.

    The first put is the prompt, as required by native/HF generate(). Token
    shapes/counts require no device synchronization. Native chunking opts into
    a single-token first transfer so TTFT is not a 16-token chunk latency.
    """
    flush_first_token = True

    def __init__(self, streamer=None, *, request_started=None, clock=None):
        self.streamer = streamer
        self.clock = clock or time.perf_counter
        self.request_started = self.clock() if request_started is None else request_started
        self.generation_started = None
        self.generation_ended = None
        self.first_token_at = self.last_token_at = None
        self.prompt_pending = True
        self.observed_tokens = 0

    def start_generation(self):
        self.generation_started = self.clock()

    def put(self, tokens):
        if self.prompt_pending:
            self.prompt_pending = False
        else:
            count = tokens.numel()
            if count:
                now = self.clock()
                if self.first_token_at is None:
                    self.first_token_at = now
                self.last_token_at = now
                self.observed_tokens += count
        if self.streamer is not None:
            self.streamer.put(tokens)

    def end(self):
        if self.streamer is not None:
            self.streamer.end()

    def on_finalized_text(self, text, stream_end=False):
        # Delegation for text-only producers; text is never counted as tokens.
        if self.streamer is not None:
            self.streamer.on_finalized_text(text, stream_end=stream_end)

    def finish_generation(self):
        self.generation_ended = self.clock()

    def snapshot(self, output_tokens):
        now = self.clock()
        total = max(0, now - self.request_started)
        generation = max(0, self.generation_ended - self.generation_started)
        measured = output_tokens > 0 and output_tokens == self.observed_tokens and self.first_token_at is not None
        decode = max(0, self.last_token_at - self.first_token_at) if measured else None
        intervals = output_tokens - 1
        tpot = decode / intervals if measured and intervals > 0 and decode > 0 else None
        return {
            "output_tokens": output_tokens,
            "tokens_per_second": intervals / decode if tpot is not None and decode > 0 else None,
            "tpot_ms": tpot * 1000 if tpot is not None else None,
            "ttft_ms": max(0, self.first_token_at - self.request_started) * 1000 if measured else None,
            "generation_ttft_ms": max(0, self.first_token_at - self.generation_started) * 1000 if measured else None,
            "decode_ms": decode * 1000 if decode is not None else None,
            "generation_ms": generation * 1000,
            "total_ms": total * 1000,
            "generation_tokens_per_second": output_tokens / generation if generation > 0 else None,
            "end_to_end_tokens_per_second": output_tokens / total if total > 0 else None,
            "timing_source": "token_delivery" if measured else "generation_only",
        }


class TokenRateStreamer:
    def __init__(self, streamer, clock=time.perf_counter):
        self.streamer = streamer
        self.clock = clock
        self.started = None
        self.state = (0, None)
        self.first = True

    def start_decode(self):
        """Exclude prompt prefill/compilation from the conventional decode rate."""
        self.started = self.clock()
        self.state = (0, None)

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
    return f"{stats['tokens_per_second']:.1f} tokens/s · {stats['tokens']} tokens · {stats['seconds']:.2f}s"


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
