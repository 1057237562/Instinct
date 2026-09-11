"""Task T4: use_grad_checkpoint flag threading through 3 configs + config_from_args.

Pure wiring test — asserts the flag exists (default 0), that kwargs overrides
(1/2) land, and that ``config_from_args`` threads the argparse value into the
config. No recompute logic is asserted here (that is T2/T3/T7-T9).
"""

from types import SimpleNamespace

import pytest

from model.model_instinct import InstinctConfig
from model.model_instinct_loop import InstinctConfig as LoopedInstinctConfig
from model.model_instinct_linear import InstinctConfig as LinearInstinctConfig
from trainer.trainer_utils import config_from_args

ALL_CONFIGS = [InstinctConfig, LoopedInstinctConfig, LinearInstinctConfig]


def make_args(**kw):
    base = dict(
        param_dtype="fp32",
        kv_cache_dtype="fp32",
        hidden_size=64,
        num_hidden_layers=2,
        use_moe=0,
        use_looped=0,
        config_path="",
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("cfg_cls", ALL_CONFIGS, ids=["dense", "loop", "linear"])
def test_default_zero(cfg_cls):
    assert cfg_cls().use_grad_checkpoint == 0


@pytest.mark.parametrize("cfg_cls", ALL_CONFIGS, ids=["dense", "loop", "linear"])
@pytest.mark.parametrize("value", [1, 2])
def test_kwargs_override(cfg_cls, value):
    assert cfg_cls(use_grad_checkpoint=value).use_grad_checkpoint == value


@pytest.mark.parametrize("value", [0, 1, 2])
def test_config_from_args_passes_flag(value):
    cfg = config_from_args(make_args(use_grad_checkpoint=value))
    assert cfg.use_grad_checkpoint == value


def test_config_from_args_missing_attr_defaults_zero():
    cfg = config_from_args(make_args())
    assert cfg.use_grad_checkpoint == 0


def test_config_from_args_explicit_override_wins():
    cfg = config_from_args(make_args(use_grad_checkpoint=2), use_grad_checkpoint=1)
    assert cfg.use_grad_checkpoint == 1


def test_config_from_args_looped_variant():
    cfg = config_from_args(make_args(use_looped=1, use_grad_checkpoint=2))
    assert isinstance(cfg, LoopedInstinctConfig)
    assert cfg.use_grad_checkpoint == 2
