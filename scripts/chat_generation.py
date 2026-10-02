"""Own a chat generation thread until it has stopped using the model."""

from threading import Event, Thread


class _GenerationCancelled(Exception):
    pass


class _GuardedStreamer:
    def __init__(self, task):
        self.task = task

    def put(self, tokens):
        if self.task.cancelled.is_set():
            raise _GenerationCancelled()
        self.task.streamer.put(tokens)

    def end(self):
        # Publish the end marker only after generate has returned, including
        # when it raises before its normal streamer.end() call.
        pass

    def __getattr__(self, name):
        return getattr(self.task.streamer, name)


class GenerationTask:
    def __init__(self, model, kwargs):
        self.model = model
        self.kwargs = dict(kwargs)
        self.streamer = self.kwargs['streamer']
        self.cancelled = Event()
        self.error = None
        self.thread = Thread(target=self._run, name='chat-generate', daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            if not self.cancelled.is_set():
                kwargs = dict(self.kwargs, streamer=_GuardedStreamer(self))
                self.model.generate(**kwargs)
        except _GenerationCancelled:
            pass
        except Exception as exc:
            # Retain the message, without a traceback retaining GPU tensors.
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.streamer.end()

    def stop(self, timeout=5):
        """Cancel at the next streamed block; never move an active model."""
        self.cancelled.set()
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def raise_if_failed(self):
        if self.error is not None:
            raise RuntimeError(self.error)


def stop_generation(state, timeout=5):
    task = state.get('generation_task')
    if task is None:
        return True
    if not task.stop(timeout):
        return False
    if state.get('generation_task') is task:
        state.pop('generation_task', None)
    return True
