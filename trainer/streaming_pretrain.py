"""Chunk-at-a-time packed pretraining under a bounded Arrow cache budget."""
from __future__ import annotations

import gc
import math
import os
from pathlib import Path
import threading
import traceback

import torch.distributed as dist
from torch.utils.data import DataLoader

from dataset.cache_budget import (
    GIB,
    cache_files,
    enforce_cache_budget,
    release_cache_use,
)
from dataset.sequence_bucket import bucket_token_budget
from trainer.packing_transition import build_dataset_with_cache_barrier
from trainer.trainer_utils import Logger


def should_stream_pretrain(args) -> bool:
    """Resolve auto/on/off without ever treating cache GB as GPU memory."""
    mode = str(getattr(args, "dataset_streaming", "auto"))
    if mode == "on":
        return True
    if mode == "off":
        return False
    source = os.path.abspath(getattr(args, "data_path", ""))
    if not os.path.isfile(source):
        return False
    budget_gb = float(getattr(args, "data_cache_max_gb", 5.0))
    if budget_gb == 0:
        return False
    # The ordinary loader reserves 1.25x source bytes before creating Arrow.
    return os.path.getsize(source) * 1.25 > budget_gb * GIB


def validate_streaming_budget(args) -> int:
    chunk_bytes = int(getattr(args, "streaming_chunk_mb", 1024)) * 1024 ** 2
    if chunk_bytes <= 0:
        raise ValueError("--streaming_chunk_mb must be positive")
    budget_gb = float(getattr(args, "data_cache_max_gb", 5.0))
    prefetch = int(getattr(args, "streaming_prefetch_chunks", 1))
    if prefetch not in (0, 1):
        raise ValueError("--streaming_prefetch_chunks must be 0 or 1")
    # A build can briefly contain source JSON, parsed Arrow, tokenized Arrow,
    # final packed Arrow, plus the current mmap dataset while prefetching.
    factor = 5 if prefetch else 4
    if budget_gb > 0 and chunk_bytes * factor > budget_gb * GIB:
        safe_mb = max(1, int(budget_gb * 1024 // factor))
        raise ValueError(
            f"streaming chunk {chunk_bytes / 1024 ** 2:.0f} MiB is too large for "
            f"the {budget_gb:g} GiB cache budget; use --streaming_chunk_mb "
            f"{safe_mb} or less"
        )
    return chunk_bytes


class _ThreadPrefetch:
    """One daemon-thread result without an executor's shutdown-on-exit wait."""

    def __init__(self, function):
        self._function = function
        self._done = threading.Event()
        self.value = None
        self.error = None
        self.thread = threading.Thread(
            target=self._run,
            name="instinct-packing-prefetch",
            daemon=True,
        )
        self.thread.start()

    def _run(self):
        try:
            self.value = self._function()
        except BaseException as exc:  # propagated on the training thread
            self.error = (exc, traceback.format_exc())
        finally:
            self._done.set()

    def result(self):
        self._done.wait()
        if self.error is not None:
            exc, formatted = self.error
            raise RuntimeError(
                "streaming packing prefetch failed:\n" + formatted
            ) from exc
        return self.value


class ChunkedPackedEpochLoader:
    """Build, train, and discard one packed Arrow chunk at a time."""

    def __init__(self, *, plan, dataset_factory, packing_plan, args, epoch,
                 data_config, resume_config=None):
        self.plan = plan
        self.dataset_factory = dataset_factory
        self.packing_plan = packing_plan
        self.args = args
        self.epoch = int(epoch)
        self.data_config = data_config
        self.prefetch = bool(int(getattr(args, "streaming_prefetch_chunks", 1)))
        resume_config = resume_config or {}
        self.resume_chunk = int(resume_config.get("streaming_chunk_index", 0))
        self.resume_chunk_step = int(resume_config.get("streaming_chunk_step", 0))
        if not 0 <= self.resume_chunk <= len(plan["chunks"]):
            raise ValueError("streaming checkpoint chunk cursor is outside the source plan")

    def __len__(self):
        """Conservative progress estimate; LR uses exact consumed tokens."""
        world = dist.get_world_size() if dist.is_initialized() else 1
        max_length = int(self.plan["identity"]["max_length"])
        if self.packing_plan.packing_mode == "bucket":
            token_budget = int(getattr(self.args, 'bucket_token_budget', 0))
            if token_budget <= 0:
                token_budget = bucket_token_budget(
                    float(self.args.bucket_gpu_memory_gb)
                )
            batch = max(
                1,
                token_budget // max_length,
            )
        else:
            batch = int(self.args.batch_size)
        steps = 0
        for chunk in self.plan["chunks"]:
            # Local best-fit packing normally exceeds 90% fill. Slightly
            # overestimating steps is preferable for UI ETA and final logging.
            blocks = max(1, math.ceil(chunk["tokens"] / (max_length * 0.90)))
            per_rank = math.ceil(blocks / world)
            steps += math.ceil(per_rank / batch)
        return steps

    def __iter__(self):
        chunks = self.plan["chunks"]
        remaining = chunks[self.resume_chunk:]
        prefetched_dataset = None
        prefetched_ready = False
        for position, chunk in enumerate(remaining):
            index = int(chunk["index"])
            local_skip = self.resume_chunk_step if index == self.resume_chunk else 0
            byte_range = (int(chunk["start"]), int(chunk["end"]))
            Logger(
                f"[Streaming Chunk] epoch={self.epoch + 1}, "
                f"chunk={index + 1}/{len(chunks)}, rows={chunk['rows']:,}, "
                f"tokens={chunk['tokens']:,}, source="
                f"{(byte_range[1] - byte_range[0]) / GIB:.2f}GiB"
            )
            if prefetched_ready:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    dataset = prefetched_dataset
                else:
                    dataset = self.dataset_factory(True, None, byte_range)
                prefetched_dataset = None
                prefetched_ready = False
                Logger(
                    f"[Streaming Prefetch] chunk={index + 1}/{len(chunks)} ready; "
                    "switching without synchronous packing"
                )
            else:
                dataset = build_dataset_with_cache_barrier(
                    lambda: self.dataset_factory(True, None, byte_range),
                    packing=True,
                )
            files = cache_files(dataset.samples)
            leases = files + sorted({str(Path(value).resolve().parent) for value in files})
            next_chunk = remaining[position + 1] if position + 1 < len(remaining) else None
            prefetch_task = None
            loader = None
            prefetch_error = None
            try:
                batches = self.packing_plan.batch_sampler(
                    dataset, active_packing=True, epoch=self.epoch,
                    batch_size=self.args.batch_size, skip_batches=local_skip,
                )
                loader = DataLoader(
                    dataset, batch_sampler=batches,
                    num_workers=self.packing_plan.loader_num_workers(),
                    pin_memory=True,
                )
                local_step = local_skip
                for batch in loader:
                    local_step += 1
                    # The checkpoint written while this batch is executing must
                    # resume *after* it, not replay it.
                    self.data_config.update(
                        streaming_chunk_index=index,
                        streaming_chunk_step=local_step,
                    )
                    yield batch
                    # The generator resumes here only after the caller has
                    # finished this batch. Delaying prefetch until then keeps
                    # first-step torch.compile/Triton work free from a Python
                    # tokenizer thread that would otherwise starve the main
                    # thread and leave the GPU idle for several minutes.
                    if (
                        prefetch_task is None and self.prefetch
                        and next_chunk is not None
                        and (not dist.is_initialized() or dist.get_rank() == 0)
                    ):
                        next_range = (
                            int(next_chunk["start"]), int(next_chunk["end"]),
                        )
                        Logger(
                            f"[Streaming Prefetch] first training batch completed; "
                            f"packing chunk={int(next_chunk['index']) + 1}/"
                            f"{len(chunks)} in background thread"
                        )
                        prefetch_task = _ThreadPrefetch(
                            lambda byte_range=next_range: self.dataset_factory(
                                True, None, byte_range
                            )
                        )
                self.data_config.update(
                    streaming_chunk_index=index + 1,
                    streaming_chunk_step=0,
                )
            finally:
                if self.prefetch and next_chunk is not None and prefetch_task is not None:
                    error_text = None
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        try:
                            prefetched_dataset = prefetch_task.result()
                            prefetched_ready = True
                        except BaseException:
                            error_text = traceback.format_exc()
                    if dist.is_initialized():
                        payload = [error_text]
                        dist.broadcast_object_list(payload, src=0)
                        error_text = payload[0]
                    if error_text:
                        prefetch_error = RuntimeError(error_text)
                del loader, dataset
                gc.collect()
                release_cache_use(leases)
                if dist.is_initialized():
                    dist.barrier()
                if not dist.is_initialized() or dist.get_rank() == 0:
                    # Keep completed packed shards as an LRU cache so a restart
                    # or a later epoch can reuse tokenization/packing work.  The
                    # global disk budget still makes peak usage independent of
                    # corpus size: old shards are evicted only when space is
                    # actually needed instead of unconditionally at chunk end.
                    report = enforce_cache_budget()
                    if report['removed']:
                        Logger(
                            '[Streaming Cache] retained completed chunk when '
                            'space allowed; LRU eviction reclaimed '
                            f"{report['removed_bytes'] / GIB:.2f}GiB"
                        )
                if dist.is_initialized():
                    dist.barrier()
                if prefetch_error is not None:
                    raise prefetch_error
            self.resume_chunk_step = 0
