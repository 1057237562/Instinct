"""Wall-time-aware contiguous sequence bucket optimization.

The optimizer estimates the number of blocks produced by well-mixed
best-fit-decreasing packing, converts each bucket length to the batch size
allowed by the configured VRAM token budget, and minimizes estimated epoch
wall time. Lengths are aggregated at the requested alignment before the DP.
"""

from array import array
from bisect import bisect_right
from dataclasses import dataclass
import math
import os


BUCKET_CALIBRATION_MEMORY_GB = 16.0
BUCKET_CALIBRATION_SEQ_LEN = 2048
BUCKET_CALIBRATION_BATCH_SIZE = 12
BUCKET_CALIBRATION_TOKEN_BUDGET = (
    BUCKET_CALIBRATION_SEQ_LEN * BUCKET_CALIBRATION_BATCH_SIZE
)

# Initial one-point wall-time calibration from the measured fixed-packing run.
_TIME_REFERENCE_BATCH_SIZE = 28
_TIME_REFERENCE_SEQ_LEN = 1024
_TIME_REFERENCE_STEP_SECONDS = 0.56
_TIME_FIXED_STEP_SECONDS = 0.03
_TIME_ATTENTION_FRACTION = 0.20
_EXACT_BFD_MAX_GROUPS = 256


def packing_preprocess_workers(requested=0, sample_count=None, *,
                               platform_name=None, cpu_count=None):
    """Resolve a memory-safe process count for packing preprocessing.

    ``datasets.map`` uses spawned processes on Windows. Every child imports
    PyTorch, Transformers, and optional TorchAO modules, so large CPU-count
    defaults can exhaust committed memory before long-context buckets begin.
    Keep Windows at four processes while retaining the existing Linux limits.
    """
    platform_name = os.name if platform_name is None else str(platform_name)
    cpu_count = (os.cpu_count() or 1) if cpu_count is None else int(cpu_count)
    cpu_count = max(1, cpu_count)
    maximum = min(4 if platform_name == 'nt' else 32, cpu_count)
    automatic = min(4 if platform_name == 'nt' else 8, cpu_count)
    workers = int(requested)
    workers = automatic if workers <= 0 else min(workers, maximum)
    if sample_count is not None:
        workers = min(workers, max(1, int(sample_count)))
    return max(1, workers)


@dataclass(frozen=True)
class SequenceBucket:
    """One contiguous range in the length-sorted source examples."""

    start: int
    end: int
    max_length: int
    estimated_cost: float
    estimated_blocks: int = 0
    batch_size: int = 1


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def bucket_token_budget(gpu_memory_gb: float) -> int:
    """Scale the measured 16GB B*L budget to requested per-card VRAM."""
    gpu_memory_gb = float(gpu_memory_gb)
    if not math.isfinite(gpu_memory_gb) or gpu_memory_gb <= 0:
        raise ValueError('bucket_gpu_memory_gb must be a positive finite number')
    return max(1, math.floor(
        BUCKET_CALIBRATION_TOKEN_BUDGET
        * gpu_memory_gb / BUCKET_CALIBRATION_MEMORY_GB
    ))


def _estimate_step_seconds(batch_size: int, seq_len: int) -> float:
    scalable = _TIME_REFERENCE_STEP_SECONDS - _TIME_FIXED_STEP_SECONDS
    linear = scalable * (1.0 - _TIME_ATTENTION_FRACTION)
    attention = scalable * _TIME_ATTENTION_FRACTION
    linear_ratio = (
        batch_size * seq_len
        / (_TIME_REFERENCE_BATCH_SIZE * _TIME_REFERENCE_SEQ_LEN)
    )
    attention_ratio = (
        batch_size * seq_len * seq_len
        / (
            _TIME_REFERENCE_BATCH_SIZE
            * _TIME_REFERENCE_SEQ_LEN
            * _TIME_REFERENCE_SEQ_LEN
        )
    )
    return _TIME_FIXED_STEP_SECONDS + linear * linear_ratio + attention * attention_ratio


def _estimate_bucket_seconds(blocks: int, max_length: int,
                             batch_size: int) -> float:
    full_batches, remainder = divmod(int(blocks), int(batch_size))
    seconds = full_batches * _estimate_step_seconds(batch_size, max_length)
    if remainder:
        seconds += _estimate_step_seconds(remainder, max_length)
    return seconds


def _grouped_bfd_blocks(group_max_lengths, group_counts, start, end,
                        alignment):
    """Count aligned BFD blocks without expanding repeated sample lengths."""
    capacity = group_max_lengths[end - 1] // alignment
    residual_counts = [0] * (capacity + 1)
    blocks = 0
    for index in range(end - 1, start - 1, -1):
        item_size = group_max_lengths[index] // alignment
        remaining_items = group_counts[index]
        while remaining_items:
            residual = next((
                value for value in range(item_size, capacity + 1)
                if residual_counts[value]
            ), None)
            if residual is None:
                break
            placed = min(remaining_items, residual_counts[residual])
            residual_counts[residual] -= placed
            residual_counts[residual - item_size] += placed
            remaining_items -= placed
        if remaining_items:
            items_per_new_block = capacity // item_size
            full_blocks, partial_items = divmod(
                remaining_items, items_per_new_block,
            )
            blocks += full_blocks + int(partial_items > 0)
            residual_counts[
                capacity - items_per_new_block * item_size
            ] += full_blocks
            if partial_items:
                residual_counts[
                    capacity - partial_items * item_size
                ] += 1
    return blocks


def optimal_sequence_buckets(lengths, bucket_count=2, alignment=1,
                             token_budget=BUCKET_CALIBRATION_TOKEN_BUDGET):
    """Return stable sorted indices and minimum estimated-time buckets.

    For an interval with capacity ``L``, the post-packing block estimate is the
    stronger of the token-volume lower bound and the number of items longer
    than ``L / 2`` (which necessarily occupy separate blocks). The DP then
    minimizes the calibrated O(BL) + O(BL^2) wall-time model.

    Complexity is ``O(bucket_count * group_count^2)`` after sorting. With
    16-token alignment the current 905k-row SFT corpus has only 153 groups.
    """
    if int(bucket_count) < 1:
        raise ValueError('seq_bucket must be at least 1')
    if int(alignment) < 1:
        raise ValueError('sequence bucket alignment must be at least 1')
    if int(token_budget) < 1:
        raise ValueError('token_budget must be at least 1')
    alignment = int(alignment)
    token_budget = int(token_budget)
    normalized_lengths = [int(length) for length in lengths]
    if any(length <= 0 for length in normalized_lengths):
        raise ValueError('sequence lengths must all be positive')
    if not normalized_lengths:
        return [], []

    sorted_indices = sorted(
        range(len(normalized_lengths)), key=normalized_lengths.__getitem__,
    )
    values = [normalized_lengths[index] for index in sorted_indices]

    group_starts = []
    group_ends = []
    group_max_lengths = []
    group_counts = []
    group_sums = []
    for position, length in enumerate(values):
        aligned_length = _align_up(length, alignment)
        if not group_max_lengths or group_max_lengths[-1] != aligned_length:
            group_starts.append(position)
            group_ends.append(position + 1)
            group_max_lengths.append(aligned_length)
            group_counts.append(1)
            group_sums.append(length)
        else:
            group_ends[-1] = position + 1
            group_counts[-1] += 1
            group_sums[-1] += length

    group_count = len(group_max_lengths)
    bucket_count = min(int(bucket_count), group_count)
    count_prefix = [0]
    token_prefix = [0]
    for count, length_sum in zip(group_counts, group_sums):
        count_prefix.append(count_prefix[-1] + count)
        token_prefix.append(token_prefix[-1] + length_sum)

    exact_bfd = group_count <= _EXACT_BFD_MAX_GROUPS
    interval_cache = {}

    def interval_estimate(start: int, end: int):
        """Estimate the half-open group interval [start, end)."""
        cached = interval_cache.get((start, end))
        if cached is not None:
            return cached
        max_length = group_max_lengths[end - 1]
        token_sum = token_prefix[end] - token_prefix[start]
        if exact_bfd:
            blocks = _grouped_bfd_blocks(
                group_max_lengths, group_counts, start, end, alignment,
            )
        else:
            # Bound memory and runtime for unusually dense length histograms.
            # This fallback stays O(1) per transition; ordinary 16-aligned LM
            # corpora use the exact grouped BFD path above.
            volume_blocks = (token_sum + max_length - 1) // max_length
            first_large = bisect_right(
                group_max_lengths, max_length // 2, start, end,
            )
            large_blocks = count_prefix[end] - count_prefix[first_large]
            blocks = max(1, int(volume_blocks), int(large_blocks))
        batch_size = max(1, token_budget // max_length)
        seconds = _estimate_bucket_seconds(blocks, max_length, batch_size)
        result = seconds, blocks, batch_size
        interval_cache[start, end] = result
        return result

    infinity = float('inf')
    previous = [infinity] * (group_count + 1)
    previous[0] = 0.0
    choices = [
        array('q', [-1]) * (group_count + 1)
        for _ in range(bucket_count + 1)
    ]

    for used in range(1, bucket_count + 1):
        current = [infinity] * (group_count + 1)
        for end in range(used, group_count + 1):
            best_cost = infinity
            best_split = -1
            for split in range(used - 1, end):
                if previous[split] == infinity:
                    continue
                interval_seconds, _, _ = interval_estimate(split, end)
                candidate = previous[split] + interval_seconds
                if candidate < best_cost:
                    best_cost = candidate
                    best_split = split
            current[end] = best_cost
            choices[used][end] = best_split
        previous = current

    ranges = []
    end = group_count
    for used in range(bucket_count, 0, -1):
        start_group = choices[used][end]
        seconds, blocks, batch_size = interval_estimate(start_group, end)
        ranges.append(SequenceBucket(
            start=group_starts[start_group],
            end=group_ends[end - 1],
            max_length=group_max_lengths[end - 1],
            estimated_cost=seconds,
            estimated_blocks=blocks,
            batch_size=batch_size,
        ))
        end = start_group
    ranges.reverse()
    return sorted_indices, ranges
