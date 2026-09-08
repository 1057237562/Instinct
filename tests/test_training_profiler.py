import json
import time
from types import SimpleNamespace

from trainer.trainer_cli import build_trainer_parser
from trainer.training_profiler import TrainingProfiler


def _args(tmp_path, mode="timing", **overrides):
    values = {
        "profile": mode,
        "profile_interval": 2,
        "profile_warmup": 0,
        "profile_active_steps": 1,
        "profile_dir": str(tmp_path),
        "device": "cpu",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_profile_cli_flags():
    args = build_trainer_parser("profile-test").parse_args([
        "--profile", "torch",
        "--profile_interval", "7",
        "--profile_warmup", "3",
        "--profile_active_steps", "2",
    ])
    assert args.profile == "torch"
    assert args.profile_interval == 7
    assert args.profile_warmup == 3
    assert args.profile_active_steps == 2


def test_timing_profiler_reports_throughput_and_phases(tmp_path):
    profiler = TrainingProfiler(_args(tmp_path), name="unit")
    result = None
    for _ in range(2):
        profiler.begin_step(tokens=100, useful_tokens=75)
        with profiler.phase("forward"):
            time.sleep(0.001)
        result = profiler.end_step()

    assert result is not None
    assert result["profile/step_time_ms"] > 0
    assert result["profile/tokens_per_second"] > 0
    assert result["profile/useful_tokens_per_second"] > 0
    assert result["profile/forward_ms"] > 0
    assert 0 < result["profile/forward_pct"] <= 100
    profiler.finish()


def test_off_profiler_is_noop(tmp_path):
    profiler = TrainingProfiler(_args(tmp_path, mode="off"), name="off")
    profiler.begin_step(tokens=10)
    with profiler.phase("forward"):
        pass
    assert profiler.end_step() is None
    assert profiler.finish() is None


def test_torch_profiler_exports_official_chrome_trace(tmp_path):
    profiler = TrainingProfiler(
        _args(tmp_path, mode="torch", profile_interval=10), name="trace"
    )
    # The official schedule has one warmup step followed by one active step.
    for _ in range(2):
        profiler.begin_step(tokens=10)
        with profiler.phase("forward"):
            sum(i * i for i in range(100))
        profiler.end_step()
    profiler.finish()

    traces = list(tmp_path.glob("trace_rank0_*.pt.trace.json"))
    assert traces
    payload = json.loads(traces[0].read_text(encoding="utf-8"))
    assert "traceEvents" in payload

