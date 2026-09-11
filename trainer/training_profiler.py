"""Low-overhead training timing and optional PyTorch operator traces."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import torch

from trainer.trainer_utils import Logger, is_main_process


@dataclass
class _StepSample:
    step: int
    tokens: int
    useful_tokens: int
    host_gap_ms: float
    total_start: object
    total_end: object | None = None
    phases: dict[str, list[tuple[object, object]]] = field(default_factory=dict)


class TrainingProfiler:
    """Measure real training throughput and optionally export a Kineto trace.

    ``timing`` uses CUDA events, so asynchronous kernels are measured without a
    synchronize on every step.  Synchronization only happens once per report
    interval.  ``torch`` enables the same timings and records one short
    PyTorch-profiler window for operator/kernel inspection.
    """

    def __init__(self, args, *, name: str = "train"):
        self.mode = str(getattr(args, "profile", "off"))
        self.enabled = self.mode != "off" and is_main_process()
        self.name = name
        self.interval = max(1, int(getattr(args, "profile_interval", 100)))
        self.warmup = max(0, int(getattr(args, "profile_warmup", 10)))
        self.active_steps = max(1, int(getattr(args, "profile_active_steps", 5)))
        self.trace_dir = os.path.abspath(getattr(args, "profile_dir", "./profiler_traces"))
        self.device = torch.device(getattr(args, "device", "cpu"))
        self.cuda = self.enabled and self.device.type == "cuda" and torch.cuda.is_available()
        self.local_step = 0
        self._samples: list[_StepSample] = []
        self._current: _StepSample | None = None
        self._last_step_end = time.perf_counter()
        self._torch_profiler = None
        self._trace_index = 0

        if not self.enabled:
            return
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        if self.mode == "torch":
            self._start_torch_profiler()
        Logger(
            f"[PROFILE] enabled: mode={self.mode}, warmup={self.warmup}, "
            f"interval={self.interval}, trace_active_steps={self.active_steps}"
        )

    def _new_marker(self):
        if self.cuda:
            return torch.cuda.Event(enable_timing=True)
        return time.perf_counter()

    def _record_marker(self, marker) -> None:
        if self.cuda:
            marker.record()

    def _elapsed_ms(self, start, end) -> float:
        if self.cuda:
            return float(start.elapsed_time(end))
        return max(0.0, float(end - start) * 1000.0)

    def _start_torch_profiler(self) -> None:
        os.makedirs(self.trace_dir, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        try:
            self._torch_profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=max(self.warmup - 1, 0), warmup=1,
                    active=self.active_steps, repeat=1,
                ),
                on_trace_ready=self._export_trace,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            self._torch_profiler.start()
        except Exception as exc:
            self._torch_profiler = None
            self.mode = "timing"
            Logger(f"[PROFILE] PyTorch trace unavailable ({exc}); falling back to timing")

    def _export_trace(self, prof) -> None:
        self._trace_index += 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        path = os.path.join(
            self.trace_dir, f"{self.name}_rank{rank}_{self._trace_index}.pt.trace.json"
        )
        prof.export_chrome_trace(path)
        Logger(f"[PROFILE] trace saved: {path}")

    def begin_step(self, *, tokens: int, useful_tokens: int | None = None) -> None:
        """Start one measured training step after the DataLoader yielded a batch."""
        if not self.enabled:
            return
        self.local_step += 1
        now = time.perf_counter()
        # This includes DataLoader wait plus occasional logging/checkpoint work
        # between measured compute steps, so do not mislabel it as pure I/O.
        host_gap_ms = 0.0 if self.local_step == 1 else (now - self._last_step_end) * 1000.0
        marker = self._new_marker()
        self._record_marker(marker)
        self._current = _StepSample(
            step=self.local_step,
            tokens=int(tokens),
            useful_tokens=int(tokens if useful_tokens is None else useful_tokens),
            host_gap_ms=host_gap_ms,
            total_start=marker,
        )
        if self.local_step == self.warmup + 1 and self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def set_tokens(self, tokens: int, useful_tokens: int | None = None) -> None:
        """Update counts when dynamic-length rollout shapes are known later."""
        if not self.enabled or self._current is None:
            return
        self._current.tokens = int(tokens)
        self._current.useful_tokens = int(tokens if useful_tokens is None else useful_tokens)

    @contextmanager
    def phase(self, name: str):
        """Label and time a phase such as forward, backward, or optimizer."""
        if not self.enabled or self._current is None:
            yield
            return
        record_ctx = torch.profiler.record_function(f"train::{name}") if self.mode == "torch" else nullcontext()
        start = self._new_marker()
        self._record_marker(start)
        with record_ctx:
            yield
        end = self._new_marker()
        self._record_marker(end)
        self._current.phases.setdefault(name, []).append((start, end))

    def end_step(self) -> dict[str, float] | None:
        """Finish a step and occasionally return metrics suitable for SwanLab."""
        if not self.enabled or self._current is None:
            return None
        end = self._new_marker()
        self._record_marker(end)
        self._current.total_end = end
        if self.local_step > self.warmup:
            self._samples.append(self._current)
        self._current = None

        if self._torch_profiler is not None:
            self._torch_profiler.step()

        metrics = self._report() if len(self._samples) >= self.interval else None
        self._last_step_end = time.perf_counter()
        return metrics

    def _report(self) -> dict[str, float] | None:
        if not self._samples:
            return None
        if self.cuda:
            torch.cuda.synchronize(self.device)

        total_ms = sum(self._elapsed_ms(s.total_start, s.total_end) for s in self._samples)
        count = len(self._samples)
        tokens = sum(s.tokens for s in self._samples)
        useful_tokens = sum(s.useful_tokens for s in self._samples)
        phase_totals: dict[str, float] = {}
        for sample in self._samples:
            for name, events in sample.phases.items():
                phase_totals[name] = phase_totals.get(name, 0.0) + sum(
                    self._elapsed_ms(start, end) for start, end in events
                )

        seconds = max(total_ms / 1000.0, 1e-9)
        metrics = {
            "profile/step_time_ms": total_ms / count,
            "profile/tokens_per_second": tokens / seconds,
            "profile/useful_tokens_per_second": useful_tokens / seconds,
            "profile/host_gap_ms": sum(s.host_gap_ms for s in self._samples) / count,
        }
        for name, phase_ms in phase_totals.items():
            metrics[f"profile/{name}_ms"] = phase_ms / count
            metrics[f"profile/{name}_pct"] = 100.0 * phase_ms / max(total_ms, 1e-9)
        if self.cuda:
            metrics["profile/peak_allocated_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
            metrics["profile/peak_reserved_gb"] = torch.cuda.max_memory_reserved(self.device) / 2**30

        phase_text = ", ".join(
            f"{name}={phase_totals[name] / count:.1f}ms/{100.0 * phase_totals[name] / max(total_ms, 1e-9):.0f}%"
            for name in sorted(phase_totals)
        )
        memory_text = (
            f", peak={metrics['profile/peak_allocated_gb']:.2f}GB "
            f"(reserved {metrics['profile/peak_reserved_gb']:.2f}GB)"
            if self.cuda else ""
        )
        Logger(
            f"[PROFILE] steps={self._samples[0].step}-{self._samples[-1].step}, "
            f"step={metrics['profile/step_time_ms']:.2f}ms, "
            f"tokens/s={metrics['profile/tokens_per_second']:.0f}, "
            f"useful_tokens/s={metrics['profile/useful_tokens_per_second']:.0f}, "
            f"host_gap={metrics['profile/host_gap_ms']:.2f}ms"
            + (f", {phase_text}" if phase_text else "") + memory_text
        )
        self._samples.clear()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        return metrics

    def finish(self) -> dict[str, float] | None:
        """Flush remaining samples and close a possible Kineto trace."""
        if not self.enabled:
            return None
        metrics = self._report()
        if self._torch_profiler is not None:
            self._torch_profiler.stop()
            self._torch_profiler = None
        return metrics
