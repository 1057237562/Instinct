"""Bounded, ordered CPU prefetch with spawn workers and reproducible augmentation.

Workers never receive CUDA objects. Each batch owns an epoch/position seed, so
worker count, completion order and unconsumed prefetch do not alter resume.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import signal
import numpy as np

_worker_dataset = None


def _initialize(dataset):
    global _worker_dataset
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # The parent saves at a completed batch.
    _worker_dataset = dataset


def _batch(dataset, indices, seed):
    previous = dataset.rng
    dataset.rng = np.random.default_rng(np.random.SeedSequence(seed))
    try:
        return dataset.batch(indices)
    finally:
        dataset.rng = previous


def _worker_batch(indices, seed):
    return _batch(_worker_dataset, indices, seed)


class BatchLoader:
    def __init__(self, dataset, num_workers=8, prefetch_factor=2, seed=42):
        if num_workers < 0 or prefetch_factor < 1:
            raise ValueError('num_workers must be >= 0 and prefetch_factor >= 1')
        self.dataset, self.workers, self.seed = dataset, num_workers, seed
        self.capacity = max(1, num_workers * prefetch_factor)
        self.executor = None
        self.serial_rng = False  # Compatibility with native checkpoints before batch-local seeds.

    def __enter__(self):
        if self.workers:
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize, initargs=(self.dataset,))
        return self

    def batches(self, order, batch_size, epoch=0, cursor=0):
        if batch_size < 1 or not 0 <= cursor <= len(order):
            raise ValueError('invalid batch size or resume cursor')
        positions = iter(range(cursor, len(order), batch_size))
        pending = deque()

        def submit():
            begin = next(positions, None)
            if begin is None:
                return False
            indices = order[begin:begin + batch_size]
            seed = (self.seed, epoch, begin)
            future = self.executor.submit(_worker_batch, indices, seed)
            pending.append((begin, future))
            return True

        if self.executor is None or self.serial_rng:
            for begin in positions:
                indices=order[begin:begin + batch_size]
                yield begin, self.dataset.batch(indices) if self.serial_rng else _batch(self.dataset, indices, (self.seed, epoch, begin))
            return
        try:
            for _ in range(self.capacity):
                if not submit(): break
            while pending:
                begin, future = pending.popleft()
                batch = future.result()  # Propagate worker errors; preserve sampler order.
                submit()
                yield begin, batch
        finally:
            for _, future in pending:
                future.cancel()

    def __exit__(self, *exc):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
