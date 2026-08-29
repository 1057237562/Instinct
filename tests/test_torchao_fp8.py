from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch

from trainer.trainer_cli import build_trainer_parser
from trainer.trainer_utils import _fp8_linear_is_eligible, apply_torchao_fp8_training


def test_fp8_cli_defaults_to_off():
    args = build_trainer_parser("test").parse_args([])
    assert args.fp8_training == "off"
    assert args.fp8_filter == "auto"


def test_fp8_cli_accepts_all_recipes():
    for recipe in ("tensorwise", "rowwise", "rowwise_with_gw_hp"):
        args = build_trainer_parser("test").parse_args(["--fp8_training", recipe])
        assert args.fp8_training == recipe


def test_fp8_filter_preserves_lm_head_lora_and_unaligned_layers():
    assert _fp8_linear_is_eligible(torch.nn.Linear(64, 128), "model.layers.0.mlp.up_proj")
    assert not _fp8_linear_is_eligible(torch.nn.Linear(64, 128), "lm_head")
    assert not _fp8_linear_is_eligible(torch.nn.Linear(64, 128), "block.lora.A")
    assert not _fp8_linear_is_eligible(torch.nn.Linear(63, 128), "block.proj")


def test_fp8_off_is_identity():
    model = torch.nn.Sequential(torch.nn.Linear(16, 16))
    args = SimpleNamespace(fp8_training="off")
    assert apply_torchao_fp8_training(model, args) is model


@pytest.mark.skipif(
    not torch.cuda.is_available() or find_spec("torchao") is None,
    reason="requires CUDA and optional torchao",
)
def test_tensorwise_forward_backward_and_state_dict_compatibility():
    from torchao.float8.float8_linear import Float8Linear

    baseline = torch.nn.Sequential(
        torch.nn.Linear(64, 128, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 64, bias=False),
    ).cuda().bfloat16()
    original_state = baseline.state_dict()
    args = SimpleNamespace(
        fp8_training="tensorwise",
        fp8_filter="eligible",
        dtype="bfloat16",
        device="cuda:0",
        use_compile=0,
    )

    converted = apply_torchao_fp8_training(baseline, args, label="test")
    assert sum(isinstance(module, Float8Linear) for module in converted.modules()) == 2
    assert set(converted.state_dict()) == set(original_state)

    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    converted(x).float().square().mean().backward()
    assert all(param.grad is not None for param in converted.parameters())

    plain = torch.nn.Sequential(
        torch.nn.Linear(64, 128, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 64, bias=False),
    ).cuda().bfloat16()
    plain.load_state_dict(converted.state_dict(), strict=True)
