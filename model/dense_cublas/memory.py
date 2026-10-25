"""Reusable temporary allocations for dynamic training and recomputation."""
from contextlib import contextmanager
from .backend import CublasBackend
from model.dense_triton.runtime import Tensor, ITEMSIZE
from . import tensor_kernels as tk
from model.dense_triton.runtime import Scalar
import math
import triton


class TrainingBackend(CublasBackend):
    def __init__(self, *args, pool_limit_mib=256, pin_memory=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.pin_memory = pin_memory
        self._pool, self._live, self._serial = {}, {}, 0
        self._pool_bytes = 0
        self.pool_limit = int(pool_limit_mib * 1024 * 1024)
        if self.pool_limit < 0: raise ValueError('pool_limit_mib must be nonnegative')
        self._event_pool, self._live_events = [], {}
        self._pinned=set();self._pinned_events=set()

    def empty(self, shape, dtype='fp32', zero=False):
        self._check()
        shape = tuple(shape)
        if not shape or any(type(d) is not int or d <= 0 for d in shape) or dtype not in ITEMSIZE:
            raise ValueError('expected a positive shape and supported dtype')
        if dtype.startswith('fp8') and self.arch < 89:
            raise RuntimeError('Native FP8 requires compute capability >= 8.9')
        size = math.prod(shape) * ITEMSIZE[dtype]
        pool = self._pool.get(size)
        if pool:
            tensor = Tensor(shape, dtype, pool.pop(), self)
            self._pool_bytes -= size
        else:
            tensor = super().empty(shape, dtype, zero=False)
        self._serial += 1
        self._live[tensor.storage] = (self._serial, size)
        if zero:
            self.launch(tk.fill, (triton.cdiv(tensor.numel, 256),), (tensor, Scalar(0.)), N=tensor.numel)
        return tensor

    def mark(self):
        return self._serial

    def create_event(self, timing=False):
        if timing: return super().create_event(timing=True)
        event = self._event_pool.pop() if self._event_pool else super().create_event()
        self._serial += 1
        self._live_events[event] = self._serial
        return event

    def pin_since(self,mark):
        self._pinned.update(p for p,(serial,size) in self._live.items() if serial>mark)
        self._pinned_events.update(event for event,serial in self._live_events.items() if serial>mark)

    def release_since(self, mark, keep=()):
        if self._commands is not None:return  # Preparation retains a fixed-address arena until context exit.
        self.synchronize()  # Includes optional dW stream and all prior consumers.
        retained = {t.storage for t in keep if t is not None}|self._pinned
        for pointer, (serial, size) in list(self._live.items()):
            if serial > mark and pointer not in retained:
                self._pool.setdefault(size, []).append(pointer)
                self._pool_bytes += size
                del self._live[pointer]
        for event, serial in list(self._live_events.items()):
            if serial > mark and event not in self._pinned_events:
                self._event_pool.append(event)
                del self._live_events[event]
        # Dynamic MoE token counts and decode lengths must not grow the cache
        # without a bound. Release unused buffers, never active tensors.
        freed = set()
        for size in sorted(self._pool, reverse=True):
            pool = self._pool[size]
            while pool and self._pool_bytes > self.pool_limit:
                pointer = pool.pop()
                self._free(pointer)
                freed.add(pointer)
                self._pool_bytes -= size
        if freed:
            self._allocations[:] = [(p, n) for p, n in self._allocations if p.value not in freed]

    def memory(self):
        return super().memory() | {'live_bytes': sum(n for _, n in self._live.values()),
                                  'cached_bytes': self._pool_bytes,
                                  'cached_limit_bytes': self.pool_limit}

    @contextmanager
    def temporary(self):
        mark = self.mark()
        try:
            yield
        finally:
            self.release_since(mark)
