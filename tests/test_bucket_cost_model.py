"""Small, hardware-calibrated wall-time model for sequence bucket analysis.

This is intentionally an analysis test instead of production scheduling code.
It lets us validate the cost formula against known runs before using it to
choose bucket boundaries.
"""

import bisect
from collections import Counter
from dataclasses import dataclass
import math
import os
import random

import pytest


@dataclass(frozen=True)
class BucketWorkload:
    packed_blocks: int
    max_seq_len: int
    batch_size: int


@dataclass(frozen=True)
class BucketEstimate:
    min_seq_len: int
    max_seq_len: int
    raw_samples: int
    packed_blocks: int
    batch_size: int
    micro_steps: int
    estimated_fill: float
    estimated_seconds: float


@dataclass(frozen=True)
class ThreeBucketPlan:
    buckets: tuple[BucketEstimate, BucketEstimate, BucketEstimate]
    epoch_seconds: float
    micro_steps: int


@dataclass(frozen=True)
class StepTimeCalibration:
    """One-point calibration for the same model/training configuration.

    ``attention_fraction`` is the fraction of the scalable reference-step time
    attributed to dense attention.  The remaining scalable work represents
    projections, MLP, LM head, backward, and other approximately token-linear
    work.  It must eventually be fitted from several measured shapes; 0.20 is
    only a reasonable first estimate for the current 16-layer model.
    """

    reference_batch_size: int
    reference_seq_len: int
    reference_step_seconds: float
    fixed_step_seconds: float = 0.03
    attention_fraction: float = 0.20

    def __post_init__(self):
        if self.reference_batch_size <= 0 or self.reference_seq_len <= 0:
            raise ValueError('reference shape must be positive')
        if self.reference_step_seconds <= 0:
            raise ValueError('reference_step_seconds must be positive')
        if not 0 <= self.attention_fraction <= 1:
            raise ValueError('attention_fraction must be in [0, 1]')
        if not 0 <= self.fixed_step_seconds < self.reference_step_seconds:
            raise ValueError(
                'fixed_step_seconds must be non-negative and smaller than '
                'reference_step_seconds'
            )

    def estimate_step_seconds(self, batch_size: int, seq_len: int) -> float:
        """Estimate one physical batch using O(BL) + O(BL^2) components."""
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError('batch_size and seq_len must be positive')
        scalable_reference = self.reference_step_seconds - self.fixed_step_seconds
        linear_reference = scalable_reference * (1 - self.attention_fraction)
        attention_reference = scalable_reference * self.attention_fraction
        linear_ratio = (
            batch_size * seq_len
            / (self.reference_batch_size * self.reference_seq_len)
        )
        attention_ratio = (
            batch_size * seq_len * seq_len
            / (
                self.reference_batch_size
                * self.reference_seq_len
                * self.reference_seq_len
            )
        )
        return (
            self.fixed_step_seconds
            + linear_reference * linear_ratio
            + attention_reference * attention_ratio
        )


def estimate_bucket_seconds(
        workload: BucketWorkload, calibration: StepTimeCalibration) -> float:
    """Include the smaller final physical batch instead of rounding it up."""
    full_batches, remainder = divmod(
        workload.packed_blocks, workload.batch_size,
    )
    seconds = full_batches * calibration.estimate_step_seconds(
        workload.batch_size, workload.max_seq_len,
    )
    if remainder:
        seconds += calibration.estimate_step_seconds(
            remainder, workload.max_seq_len,
        )
    return seconds


def estimate_total_seconds(
        workloads, calibration, *, epochs=1, preprocessing_seconds=0.0,
        cold_compile_seconds=0.0, checkpoint_count=0,
        checkpoint_seconds=0.0):
    """Estimate wall time, keeping one-off and recurring costs explicit."""
    epoch_seconds = sum(
        estimate_bucket_seconds(workload, calibration)
        for workload in workloads
    )
    total_seconds = (
        preprocessing_seconds
        + cold_compile_seconds
        + epochs * epoch_seconds
        + checkpoint_count * checkpoint_seconds
    )
    return {
        'epoch_seconds': epoch_seconds,
        'training_seconds': epochs * epoch_seconds,
        'total_seconds': total_seconds,
        'micro_steps': sum(
            math.ceil(workload.packed_blocks / workload.batch_size)
            for workload in workloads
        ),
    }


def _grouped_bfd_blocks(sizes, counts, start, end, alignment):
    """Exact BFD block count for the aligned length histogram interval.

    Repeated equal-sized items are moved in groups between residual-capacity
    slots, so the runtime depends on the number of length groups rather than
    the number of training samples.
    """
    capacity = sizes[end] // alignment
    residual_counts = [0] * (capacity + 1)
    blocks = 0
    for index in range(end, start - 1, -1):
        item_size = sizes[index] // alignment
        remaining_items = counts[index]
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


def search_three_bucket_plans(
        lengths, calibration, *, token_budget=24_576, alignment=16,
        top_n=10):
    """Exhaustively minimize the calibrated epoch-time formula for 3 buckets.

    This first search assumes globally mixed BFD packing inside each bucket.
    It intentionally gives an optimistic block count; production integration
    must either implement the same global/mixed packing or replace this count
    with the exact chunked packer result before adopting the selected plan.
    """
    if token_budget <= 0 or alignment <= 0 or top_n <= 0:
        raise ValueError('token_budget, alignment, and top_n must be positive')
    histogram = Counter(
        math.ceil(int(length) / alignment) * alignment
        for length in lengths if int(length) > 0
    )
    sizes = sorted(histogram)
    if len(sizes) < 3:
        raise ValueError('at least three distinct aligned lengths are required')
    counts = [histogram[size] for size in sizes]

    count_prefix = []
    token_prefix = []
    running_count = 0
    running_tokens = 0
    for size, count in zip(sizes, counts):
        running_count += count
        running_tokens += size * count
        count_prefix.append(running_count)
        token_prefix.append(running_tokens)

    def interval_sum(prefix, start, end):
        return prefix[end] - (prefix[start - 1] if start else 0)

    intervals = {}
    for start in range(len(sizes)):
        for end in range(start, len(sizes)):
            max_seq_len = sizes[end]
            packed_blocks = _grouped_bfd_blocks(
                sizes, counts, start, end, alignment,
            )
            batch_size = max(1, token_budget // max_seq_len)
            workload = BucketWorkload(
                packed_blocks, max_seq_len, batch_size,
            )
            raw_samples = interval_sum(count_prefix, start, end)
            aligned_tokens = interval_sum(token_prefix, start, end)
            intervals[start, end] = BucketEstimate(
                min_seq_len=sizes[start],
                max_seq_len=max_seq_len,
                raw_samples=raw_samples,
                packed_blocks=packed_blocks,
                batch_size=batch_size,
                micro_steps=math.ceil(packed_blocks / batch_size),
                estimated_fill=aligned_tokens / (packed_blocks * max_seq_len),
                estimated_seconds=estimate_bucket_seconds(
                    workload, calibration,
                ),
            )

    plans = []
    last = len(sizes) - 1
    for first_end in range(0, last - 1):
        for second_end in range(first_end + 1, last):
            buckets = (
                intervals[0, first_end],
                intervals[first_end + 1, second_end],
                intervals[second_end + 1, last],
            )
            plans.append(ThreeBucketPlan(
                buckets=buckets,
                epoch_seconds=sum(bucket.estimated_seconds for bucket in buckets),
                micro_steps=sum(bucket.micro_steps for bucket in buckets),
            ))
    plans.sort(key=lambda plan: (plan.epoch_seconds, plan.micro_steps))
    return plans[:top_n]


def _read_arrow_lengths(path):
    """Read only the scalar length column from a HF Arrow stream cache."""
    import pyarrow as pa

    lengths = []
    with pa.memory_map(path, 'r') as source:
        for batch in pa.ipc.open_stream(source):
            column_index = batch.schema.get_field_index('length')
            if column_index < 0:
                raise ValueError(f'Arrow cache has no length column: {path}')
            lengths.extend(batch.column(column_index).to_pylist())
    return lengths


def simulate_chunked_bfd(
        lengths, capacities, *, packing_batch_size=1000,
        deterministic_mix=False, seed=42):
    """Simulate the real per-map-batch BFD and report exact raw-token fill.

    ``deterministic_mix=False`` reproduces the current bucket implementation's
    length-sorted chunks.  ``True`` shuffles each bucket deterministically
    before creating packing chunks, which is the proposed low-risk fix.
    """
    if packing_batch_size <= 0:
        raise ValueError('packing_batch_size must be positive')
    sorted_lengths = sorted(int(length) for length in lengths if int(length) > 0)
    previous = 0
    rows = []
    for capacity in capacities:
        end = bisect.bisect_right(sorted_lengths, int(capacity))
        bucket_lengths = list(sorted_lengths[previous:end])
        if deterministic_mix:
            random.Random(seed + int(capacity)).shuffle(bucket_lengths)
        packed_blocks = 0
        for offset in range(0, len(bucket_lengths), packing_batch_size):
            residuals = []
            for length in sorted(
                    bucket_lengths[offset:offset + packing_batch_size],
                    reverse=True):
                position = bisect.bisect_left(residuals, length)
                if position == len(residuals):
                    packed_blocks += 1
                    bisect.insort(residuals, int(capacity) - length)
                else:
                    residual = residuals.pop(position)
                    bisect.insort(residuals, residual - length)
        valid_tokens = sum(bucket_lengths)
        rows.append({
            'max_seq_len': int(capacity),
            'raw_samples': len(bucket_lengths),
            'valid_tokens': valid_tokens,
            'packed_blocks': packed_blocks,
            'fill': (
                valid_tokens / (packed_blocks * int(capacity))
                if packed_blocks else 1.0
            ),
        })
        previous = end
    if previous != len(sorted_lengths):
        raise ValueError('largest capacity does not cover every sequence')
    return rows


@pytest.fixture
def measured_fixed_calibration():
    # Same-day measured baseline in PLAN.md.  This is suitable for validating
    # the formula, but a final bucket chooser must recalibrate when model,
    # precision, checkpointing, or compile mode changes.
    return StepTimeCalibration(
        reference_batch_size=28,
        reference_seq_len=1024,
        reference_step_seconds=0.56,
        fixed_step_seconds=0.03,
        attention_fraction=0.20,
    )


def test_reference_shape_reproduces_measured_step_time(
        measured_fixed_calibration):
    assert measured_fixed_calibration.estimate_step_seconds(28, 1024) == pytest.approx(0.56)


def test_same_token_budget_penalizes_longer_attention_shape(
        measured_fixed_calibration):
    short = measured_fixed_calibration.estimate_step_seconds(24, 1024)
    long = measured_fixed_calibration.estimate_step_seconds(12, 2048)
    assert long > short


def test_partial_final_batch_is_not_charged_as_a_full_batch(
        measured_fixed_calibration):
    workload = BucketWorkload(
        packed_blocks=29, max_seq_len=1024, batch_size=28,
    )
    actual = estimate_bucket_seconds(workload, measured_fixed_calibration)
    two_full_batches = 2 * measured_fixed_calibration.estimate_step_seconds(28, 1024)
    assert actual < two_full_batches
    assert actual > measured_fixed_calibration.estimate_step_seconds(28, 1024)


def test_known_runs_show_why_current_three_buckets_take_longer(
        measured_fixed_calibration, capsys):
    fixed = [BucketWorkload(472_755, 1024, 28)]
    current_buckets = [
        BucketWorkload(639_497, 624, 39),
        BucketWorkload(182_227, 944, 26),
        BucketWorkload(2_646, 3216, 7),
    ]

    fixed_estimate = estimate_total_seconds(
        fixed, measured_fixed_calibration,
    )
    bucket_estimate = estimate_total_seconds(
        current_buckets, measured_fixed_calibration,
    )
    print(
        f"fixed: {fixed_estimate['micro_steps']} steps, "
        f"{fixed_estimate['epoch_seconds'] / 60:.1f} min"
    )
    print(
        f"bucket: {bucket_estimate['micro_steps']} steps, "
        f"{bucket_estimate['epoch_seconds'] / 60:.1f} min"
    )

    assert fixed_estimate['micro_steps'] == 16_885
    assert bucket_estimate['micro_steps'] == 23_785
    assert fixed_estimate['epoch_seconds'] / 60 == pytest.approx(157.6, abs=0.1)
    assert bucket_estimate['epoch_seconds'] / 60 == pytest.approx(181.1, abs=0.1)
    assert bucket_estimate['epoch_seconds'] > fixed_estimate['epoch_seconds'] * 1.14
    assert 'fixed: 16885 steps' in capsys.readouterr().out


def test_total_time_keeps_compile_and_checkpoint_costs_separate(
        measured_fixed_calibration):
    result = estimate_total_seconds(
        [BucketWorkload(280, 1024, 28)], measured_fixed_calibration,
        epochs=2,
        preprocessing_seconds=30,
        cold_compile_seconds=120,
        checkpoint_count=2,
        checkpoint_seconds=10,
    )
    assert result['epoch_seconds'] == pytest.approx(10 * 0.56)
    assert result['training_seconds'] == pytest.approx(20 * 0.56)
    assert result['total_seconds'] == pytest.approx(30 + 120 + 20 + 20 * 0.56)


def test_grouped_histogram_bfd_matches_individual_reference():
    lengths = [32, 48, 48, 64, 80, 96, 112, 144, 160, 176]
    capacity = 192
    remaining = []
    reference_blocks = 0
    for length in sorted(lengths, reverse=True):
        position = bisect.bisect_left(remaining, length)
        if position == len(remaining):
            reference_blocks += 1
            bisect.insort(remaining, capacity - length)
        else:
            residual = remaining.pop(position)
            bisect.insort(remaining, residual - length)

    histogram = Counter(lengths)
    sizes = sorted(histogram)
    counts = [histogram[size] for size in sizes]
    assert _grouped_bfd_blocks(
        sizes, counts, 0, len(sizes) - 1, alignment=16,
    ) == reference_blocks


def test_three_bucket_search_covers_every_sample_once(
        measured_fixed_calibration):
    lengths = (
        [80] * 80 + [128] * 40 + [192] * 30
        + [320] * 20 + [512] * 10 + [768] * 4
    )
    plans = search_three_bucket_plans(
        lengths, measured_fixed_calibration,
        token_budget=2048, alignment=16, top_n=5,
    )
    assert len(plans) == 5
    for plan in plans:
        assert sum(bucket.raw_samples for bucket in plan.buckets) == len(lengths)
        assert [bucket.max_seq_len for bucket in plan.buckets] == sorted(
            bucket.max_seq_len for bucket in plan.buckets
        )
    assert plans == sorted(plans, key=lambda plan: (plan.epoch_seconds, plan.micro_steps))


def test_deterministic_mix_recovers_cross_chunk_complementarity():
    lengths = [40] * 1000 + [60] * 1000
    sorted_rows = simulate_chunked_bfd(
        lengths, [100], packing_batch_size=1000,
        deterministic_mix=False,
    )
    mixed_rows = simulate_chunked_bfd(
        lengths, [100], packing_batch_size=1000,
        deterministic_mix=True,
    )
    assert sorted_rows[0]['packed_blocks'] == 1500
    assert mixed_rows[0]['packed_blocks'] < sorted_rows[0]['packed_blocks']
    assert mixed_rows[0]['fill'] > 0.98


def test_current_sft_three_bucket_search(measured_fixed_calibration):
    """Manual real-data analysis: set INSTINCT_TOKEN_LENGTH_ARROW and use -s."""
    arrow_path = os.environ.get('INSTINCT_TOKEN_LENGTH_ARROW')
    if not arrow_path:
        pytest.skip('set INSTINCT_TOKEN_LENGTH_ARROW to a tokenized SFT Arrow cache')
    lengths = _read_arrow_lengths(arrow_path)
    plans = search_three_bucket_plans(
        lengths, measured_fixed_calibration,
        token_budget=24_576, alignment=16, top_n=10,
    )
    print(f'\nSearched {len(lengths):,} samples; best 3-bucket plans:')
    for number, plan in enumerate(plans, start=1):
        bucket_text = ', '.join(
            f'{bucket.max_seq_len}x{bucket.batch_size}: '
            f'{bucket.packed_blocks:,} blocks/{bucket.micro_steps:,} steps/'
            f'{bucket.estimated_fill:.2%} fill'
            for bucket in plan.buckets
        )
        print(
            f'{number:2d}. {plan.epoch_seconds / 60:.2f} min, '
            f'{plan.micro_steps:,} steps | {bucket_text}'
        )
    selected_capacities = [
        bucket.max_seq_len for bucket in plans[0].buckets
    ]
    for label, mixed in (('current sorted chunks', False),
                         ('deterministically mixed chunks', True)):
        rows = simulate_chunked_bfd(
            lengths, selected_capacities, packing_batch_size=1000,
            deterministic_mix=mixed,
        )
        total_valid = sum(row['valid_tokens'] for row in rows)
        total_capacity = sum(
            row['packed_blocks'] * row['max_seq_len'] for row in rows
        )
        print(f'\n{label}:')
        for row in rows:
            print(
                f"  {row['max_seq_len']}: {row['packed_blocks']:,} blocks, "
                f"{row['fill']:.2%} fill"
            )
        print(f'  total: {total_valid / total_capacity:.2%} fill')
    assert len(plans) == 10
    assert sum(bucket.raw_samples for bucket in plans[0].buckets) == len(lengths)
