"""Numerical, compatibility and capacity checks for the new PyTorch tiers."""
from contextlib import nullcontext
import gc
import json
from unittest.mock import patch

import pytest
import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from tests.helpers import make_tiny_config
from experiments.profile_checkpoint_profiles import (
    StorageTracker, add_ratios, choose_profile, simulate,
)

PROFILES = ["ckpt1", "ckpt2", "ckpt3", "ckpt4"]


def run(profile, flash, dropout, bf16, *, device="cpu", masked=False, early_exit=False):
    config = make_tiny_config()
    config.checkpoint_profile = profile
    config.flash_attn = flash
    config.dropout = dropout
    config.early_exit_layers = [1]
    torch.manual_seed(42)
    model = InstinctForCausalLM(config).to(device).train()
    ids = torch.randint(0, config.vocab_size, (2, 12), device=device)
    mask = torch.ones_like(ids) if masked else None
    if masked:
        mask[:, -2:] = 0
    context = torch.autocast(device, dtype=torch.bfloat16) if bf16 else nullcontext()
    with context:
        out = model(ids, labels=ids, attention_mask=mask, early_exit=early_exit)
    out.loss.backward()
    rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    return out.logits.detach(), {n: p.grad for n, p in model.named_parameters()}, rng


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("flash,dropout,bf16,masked,early_exit", [
    (False, 0., False, False, False), (True, 0., True, False, False),
    (True, .1, False, False, False), (True, .1, True, True, True),
])
def test_outputs_all_gradients_and_rng(profile, flash, dropout, bf16, masked, early_exit):
    reference = run("off", flash, dropout, bf16, masked=masked, early_exit=early_exit)
    actual = run(profile, flash, dropout, bf16, masked=masked, early_exit=early_exit)
    for left, right in [(reference[0], actual[0]), (reference[2], actual[2])]:
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for name, grad in reference[1].items():
        assert grad is not None and actual[1][name] is not None, name
        torch.testing.assert_close(grad, actual[1][name], rtol=0, atol=0, msg=name)


@pytest.mark.gpu
@pytest.mark.parametrize("profile", PROFILES)
def test_cuda_bf16_gradients(profile):
    with patch("model.flash_attn_4._get_fa4", return_value=None):
        reference = run("off", True, 0., True, device="cuda")
        actual = run(profile, True, 0., True, device="cuda")
    torch.testing.assert_close(reference[0], actual[0], rtol=.02, atol=.002)
    for name, grad in reference[1].items():
        torch.testing.assert_close(grad, actual[1][name], rtol=.02, atol=.002, msg=name)
    assert torch.equal(reference[2], actual[2])


@pytest.mark.parametrize("profile", PROFILES)
def test_eval_and_kv_cache_do_not_checkpoint(profile):
    config = make_tiny_config()
    config.checkpoint_profile = profile
    model = InstinctForCausalLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 6))
    with patch("model.model_instinct.checkpoint_layer", side_effect=AssertionError("must bypass")):
        model.eval()
        with torch.no_grad():
            cached = model(ids, use_cache=True)
            continued = model(ids[:, :1], past_key_values=cached.past_key_values, use_cache=True)
        assert continued.logits.shape == (2, 1, config.vocab_size)
        model.train()
        model(ids, use_cache=True).logits.sum().backward()


def test_legacy_and_config_roundtrip():
    for old in (0, 1, 2):
        config = InstinctConfig(use_grad_checkpoint=old)
        assert config.checkpoint_profile == "legacy" and config.use_grad_checkpoint == old
    for profile in ["off", *PROFILES]:
        config = InstinctConfig(use_grad_checkpoint=1, checkpoint_profile=profile)
        restored = InstinctConfig.from_dict(config.to_dict())
        assert restored.checkpoint_profile == profile and restored.use_grad_checkpoint == 0
    with pytest.raises(ValueError, match="Unknown checkpoint_profile"):
        InstinctConfig(checkpoint_profile="ckpt5")


def test_cli_and_json_precedence(tmp_path):
    from trainer.trainer_cli import build_trainer_parser
    from trainer.trainer_utils import config_from_args
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"checkpoint_profile": "ckpt3"}))
    parser = build_trainer_parser("test")
    args = parser.parse_args(["--config_path", str(path)])
    assert config_from_args(args).checkpoint_profile == "ckpt3"
    args = parser.parse_args(["--config_path", str(path), "--checkpoint_profile", "off"])
    assert config_from_args(args).checkpoint_profile == "off"
    assert config_from_args(args, checkpoint_profile="ckpt2").checkpoint_profile == "ckpt2"


def test_unsupported_architectures_fail_explicitly():
    from model.model_instinct_loop import InstinctConfig as LoopConfig
    from model.model_instinct_linear import InstinctConfig as LinearConfig
    for cls, kwargs in [(InstinctConfig, {"use_moe": True}), (LoopConfig, {}), (LinearConfig, {})]:
        with pytest.raises(ValueError, match="Dense only"):
            cls(checkpoint_profile="ckpt1", **kwargs)


def test_webui_exports_profiles_only_for_pytorch_dense():
    from tests.test_cublas_webui_contract import exported_config
    state = {"checkpoint_profile": "ckpt3", "use_grad_checkpoint": 1}
    exported = exported_config(state)
    assert exported["checkpoint_profile"] == "ckpt3"
    assert InstinctConfig(**exported).use_grad_checkpoint == 0
    for other in [{"runtime_backend": "cublas"}, {"model_architecture": "looped"},
                  {"model_architecture": "linear"}, {"use_moe": True}]:
        exported = exported_config(dict(state, **other))
        assert "checkpoint_profile" not in exported
        assert exported["use_grad_checkpoint"] == 1


def test_storage_tracker_alias_lifetime_and_temporary_peak():
    tracker = StorageTracker()
    with tracker:
        tensor = torch.zeros(1024)
        alias = tensor.view(256, 4)
        assert tracker.current == 4096
        del tensor
        assert tracker.current == 4096
        temporary = alias + 1
        assert tracker.peak >= 8192
        del temporary, alias
    gc.collect()
    assert tracker.current == 0


def test_simulated_profile_order_and_budget():
    config = dict(hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
                  num_key_value_heads=2, vocab_size=256, max_position_embeddings=128,
                  dropout=0., flash_attn=True)
    rows = [simulate(profile, config, 4, 128) for profile in ["off", *PROFILES]]
    assert rows[1]["step_flops"] == rows[0]["step_flops"]
    for high, low in zip(rows[1:-1], rows[2:]):
        assert high["peak_bytes"] > low["peak_bytes"]
        assert high["step_flops"] < low["step_flops"]
    add_ratios(rows, 2., .1)
    for row in rows[1:]:
        threshold = row["required_capacity_gib"]
        assert choose_profile(rows, threshold) == row["profile"]
        assert row["pareto_efficient"]
    assert choose_profile(rows, rows[-1]["required_capacity_gib"] - 1e-6) is None
    assert rows[-1]["tflops_saved_per_extra_gib_vs_ckpt4"] is None


@pytest.mark.parametrize("profile", PROFILES)
def test_dynamo_checkpoint_context(profile):
    config = make_tiny_config()
    config.checkpoint_profile = profile
    model = InstinctForCausalLM(config).train()
    compiled = torch.compile(model, backend="eager")
    ids = torch.randint(0, config.vocab_size, (2, 8))
    # This validates Dynamo integration; GPU Inductor/CUDA Graph memory is not
    # covered by the eager simulator or by this CPU test.
    compiled(ids, labels=ids).loss.backward()
    assert all(p.grad is not None for p in model.parameters())
