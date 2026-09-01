"""Length-aware sequence buckets optimized for packed attention.

After sorting examples by token length, a contiguous bucket ``(j, i]`` uses
the longest example as its packed block length.  Ignoring the small integer
rounding error in bin packing, its attention work is proportional to::

    sum(length[j:i]) * max(length[j:i])

The resulting partition DP has a linear form and can therefore be solved with
a monotone convex hull instead of the quadratic transition loop.
"""

from collections import deque
from dataclasses import dataclass
from array import array


@dataclass(frozen=True)
class SequenceBucket:
    """One contiguous range in the length-sorted source examples."""

    start: int
    end: int
    max_length: int
    estimated_cost: int


@dataclass(frozen=True)
class _Line:
    slope: int
    intercept: int
    split: int

    def value(self, x: int) -> int:
        return self.slope * x + self.intercept


def _redundant(first: _Line, second: _Line, third: _Line) -> bool:
    """Whether ``second`` can never win for monotonically increasing queries."""
    return (
        (second.intercept - first.intercept) * (second.slope - third.slope)
        >= (third.intercept - second.intercept) * (first.slope - second.slope)
    )


def _add_line(hull: deque, line: _Line) -> None:
    # Positive sequence lengths make prefix sums strictly increasing, hence
    # slopes strictly decreasing.  Keep the duplicate guard for direct reuse.
    if hull and hull[-1].slope == line.slope:
        if hull[-1].intercept <= line.intercept:
            return
        hull.pop()
    while len(hull) >= 2 and _redundant(hull[-2], hull[-1], line):
        hull.pop()
    hull.append(line)


def _query(hull: deque, x: int) -> _Line:
    while len(hull) >= 2 and hull[1].value(x) <= hull[0].value(x):
        hull.popleft()
    return hull[0]


def optimal_sequence_buckets(lengths, bucket_count=2):
    """Return stable sorted indices and the minimum-cost contiguous buckets.

    Complexity is ``O(bucket_count * sample_count)`` after the initial sort.
    Empty inputs return no buckets; asking for more buckets than examples is
    equivalent to one bucket per example.
    """
    if bucket_count < 1:
        raise ValueError("seq_bucket must be at least 1")
    if any(int(length) <= 0 for length in lengths):
        raise ValueError("sequence lengths must all be positive")
    if not lengths:
        return [], []

    sorted_indices = sorted(range(len(lengths)), key=lambda index: (int(lengths[index]), index))
    values = [int(lengths[index]) for index in sorted_indices]
    count = len(values)
    bucket_count = min(int(bucket_count), count)

    prefix = [0]
    for length in values:
        prefix.append(prefix[-1] + length)

    infinity = float("inf")
    previous = [infinity] * (count + 1)
    previous[0] = 0
    choices = [array('q', [-1]) * (count + 1) for _ in range(bucket_count + 1)]

    for used in range(1, bucket_count + 1):
        current = [infinity] * (count + 1)
        hull = deque()
        split = used - 1
        _add_line(hull, _Line(-prefix[split], int(previous[split]), split))
        for end in range(used, count + 1):
            max_length = values[end - 1]
            best = _query(hull, max_length)
            current[end] = prefix[end] * max_length + best.value(max_length)
            choices[used][end] = best.split
            if previous[end] != infinity:
                _add_line(hull, _Line(-prefix[end], int(previous[end]), end))
        previous = current

    ranges = []
    end = count
    for used in range(bucket_count, 0, -1):
        start = choices[used][end]
        max_length = values[end - 1]
        ranges.append(SequenceBucket(
            start=start,
            end=end,
            max_length=max_length,
            estimated_cost=(prefix[end] - prefix[start]) * max_length,
        ))
        end = start
    ranges.reverse()

    # Splitting equal-length rows creates identical block sizes without any
    # computational benefit.  Merge them so the batch sampler sees one group.
    merged = []
    for bucket in ranges:
        if merged and merged[-1].max_length == bucket.max_length:
            prior = merged[-1]
            merged[-1] = SequenceBucket(
                start=prior.start,
                end=bucket.end,
                max_length=bucket.max_length,
                estimated_cost=prior.estimated_cost + bucket.estimated_cost,
            )
        else:
            merged.append(bucket)
    return sorted_indices, merged
