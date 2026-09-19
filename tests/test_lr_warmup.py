from types import SimpleNamespace

import pytest

from trainer.trainer_cli import (
    build_trainer_parser,
    resolve_warmup_steps,
    set_cosine_lr,
    set_cosine_lr_progress,
)
from trainer.trainer_utils import get_lr


def test_legacy_cosine_schedule_is_preserved_without_warmup():
    assert get_lr(0, 100, 1.0) == pytest.approx(1.0)
    assert get_lr(50, 100, 1.0) == pytest.approx(0.55)
    assert get_lr(100, 100, 1.0) == pytest.approx(0.1)


def test_linear_rewarm_then_cosine_redecay():
    values = [
        get_lr(step, 100, 1.0, warmup_steps=10, min_lr_ratio=0.1)
        for step in range(101)
    ]
    assert values[0] == 0.0
    assert values[5] == pytest.approx(0.5)
    assert values[10] == pytest.approx(1.0)
    assert values[55] == pytest.approx(0.55)
    assert values[100] == pytest.approx(0.1)
    assert values[:11] == sorted(values[:11])
    assert values[10:] == sorted(values[10:], reverse=True)


def test_warmup_ratio_resolution_and_explicit_step_precedence():
    ratio_args = SimpleNamespace(warmup_steps=0, warmup_ratio=0.01)
    explicit_args = SimpleNamespace(warmup_steps=7, warmup_ratio=0.5)
    assert resolve_warmup_steps(1000, ratio_args) == 10
    assert resolve_warmup_steps(1000, explicit_args) == 7
    with pytest.raises(ValueError):
        resolve_warmup_steps(100, SimpleNamespace(warmup_steps=0, warmup_ratio=1.0))


def test_set_cosine_lr_uses_global_step_across_epochs():
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
    args = SimpleNamespace(
        epochs=2, learning_rate=5e-5, warmup_steps=0,
        warmup_ratio=0.1, min_lr_ratio=0.1,
    )
    set_cosine_lr(optimizer, epoch=0, step=10, iters=100, args=args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-5)
    set_cosine_lr(optimizer, epoch=0, step=20, iters=100, args=args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    set_cosine_lr(optimizer, epoch=1, step=100, iters=100, args=args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)


def test_common_cli_exposes_warmup_controls():
    parser = build_trainer_parser("test")
    args = parser.parse_args([
        "--warmup_ratio", "0.01", "--warmup_steps", "12",
        "--min_lr_ratio", "0.05",
    ])
    assert args.warmup_ratio == pytest.approx(0.01)
    assert args.warmup_steps == 12
    assert args.min_lr_ratio == pytest.approx(0.05)


def test_token_progress_schedule_does_not_restart_between_chunks():
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
    args = SimpleNamespace(
        learning_rate=1.0, warmup_steps=0,
        warmup_ratio=0.1, min_lr_ratio=0.1,
    )
    set_cosine_lr_progress(optimizer, 10, 100, args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    set_cosine_lr_progress(optimizer, 55, 100, args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.55)
    set_cosine_lr_progress(optimizer, 100, 100, args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
