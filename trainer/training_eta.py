"""CPU-only remaining-time estimates from recent completed work."""
from collections import deque
import math


class RecentThroughputETA:
    """Use progress/time deltas across at most three reporting intervals.

    The first observation is an anchor, so launch/compile time before it is
    excluded. Later data waits and checkpoint writes remain in the wall time.
    Call after synchronizing completed work, with a monotonic elapsed clock.
    """

    def __init__(self, window_intervals=3):
        if window_intervals < 1:
            raise ValueError("window_intervals must be positive")
        self._points = deque(maxlen=window_intervals + 1)
        self.window_progress = 0
        self.window_seconds = 0.0
        self.rate = None

    def update(self, completed, elapsed_seconds, total):
        completed, total = int(completed), int(total)
        elapsed_seconds = float(elapsed_seconds)
        if completed < 0 or total < 0 or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("progress and elapsed time must be finite and nonnegative")
        if self._points:
            previous_progress, previous_time = self._points[-1]
            if completed < previous_progress or elapsed_seconds < previous_time:
                self._points.clear()  # A new launch/epoch must not inherit a rate.
            elif elapsed_seconds == previous_time:
                # Duplicate reports cannot consume the bounded history; work
                # with no measurable elapsed time cannot establish a rate.
                return 0.0 if completed >= total else None
        self._points.append((completed, elapsed_seconds))
        self.window_progress = completed - self._points[0][0]
        self.window_seconds = elapsed_seconds - self._points[0][1]
        self.rate = (
            self.window_progress / self.window_seconds
            if self.window_progress > 0 and self.window_seconds > 0 else None
        )
        remaining = max(0, total - completed)
        if remaining == 0:
            return 0.0
        return remaining / self.rate / 60 if self.rate is not None else None
