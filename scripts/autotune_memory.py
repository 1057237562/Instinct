"""Phase-boundary CUDA diagnostics; no CUDA imports in the WebUI process."""
from __future__ import annotations

import gc
from pathlib import Path
import time

from trainer.autotune_config import write_json


class CudaMemoryTrace:
    """Keep startup peaks across the existing timed-window counter reset.

    Call only outside timed steps. Reserved minus allocated is a diagnostic
    difference, not a promise that all those bytes can be returned to CUDA.
    """
    def __init__(self, directory, device, *, torch_module=None, driver_query=None):
        if torch_module is None:
            import torch as torch_module
        if driver_query is None:
            from trainer.windows_gpu_memory import local_memory_budget
            driver_query = local_memory_budget
        self.torch = torch_module
        self.device = device
        self.driver_query = driver_query
        self.path = Path(directory) / 'memory.json'
        self.samples = []
        self.started = time.perf_counter()

    def capture(self, phase):
        cuda = self.torch.cuda
        gib = 2**30
        stats = cuda.memory_stats(self.device)
        free, total = cuda.mem_get_info(self.device)
        allocated = cuda.memory_allocated(self.device)
        reserved = cuda.memory_reserved(self.device)
        row = dict(phase=phase, elapsed_seconds=time.perf_counter() - self.started,
            allocated_gib=allocated / gib, reserved_gib=reserved / gib,
            unallocated_reserved_gib=max(0, reserved - allocated) / gib,
            inactive_split_gib=stats.get('inactive_split_bytes.all.current', 0) / gib,
            peak_allocated_gib=cuda.max_memory_allocated(self.device) / gib,
            peak_reserved_gib=cuda.max_memory_reserved(self.device) / gib,
            device_free_gib=free / gib, device_total_gib=total / gib,
            allocation_retries=stats.get('num_alloc_retries', 0),
            oom_count=stats.get('num_ooms', 0))
        try:
            row['driver'] = self.driver_query(cuda.get_device_name(self.device))
        except (OSError, RuntimeError):
            row['driver'] = None
        self.samples.append(row)
        write_json(self.path, self.document())
        print(f'[Autotune memory] {phase}: allocated={row["allocated_gib"]:.3f}GiB, '
              f'reserved={row["reserved_gib"]:.3f}GiB, '
              f'inactive_split={row["inactive_split_gib"]:.3f}GiB, '
              f'peak_allocated={row["peak_allocated_gib"]:.3f}GiB, '
              f'peak_reserved={row["peak_reserved_gib"]:.3f}GiB, '
              f'device_free={row["device_free_gib"]:.3f}GiB', flush=True)
        return row

    def release_cache(self, phase):
        """Release only idle allocator blocks; keep graphs, gradients and state."""
        before = self.capture(phase + '_before_cache_release')
        gc.collect()
        self.torch.cuda.empty_cache()
        after = self.capture(phase + '_after_cache_release')
        after['released_reserved_gib'] = max(0, before['reserved_gib'] - after['reserved_gib'])
        write_json(self.path, self.document())
        return after

    def document(self):
        return dict(schema=1, samples=self.samples,
            trial_peak_allocated_gib=max((row['peak_allocated_gib'] for row in self.samples), default=0),
            trial_peak_reserved_gib=max((row['peak_reserved_gib'] for row in self.samples), default=0))
