"""Correctness tests for mHC and Attention Residuals across every backbone."""

import ast
import io
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from model.model_instinct import (
    AttentionResidual,
    InstinctConfig,
    InstinctForCausalLM,
    ManifoldHyperConnection,
)
from model.model_instinct_linear import (
    InstinctConfig as LinearInstinctConfig,
    InstinctForCausalLM as LinearInstinctForCausalLM,
)
from model.model_instinct_loop import (
    InstinctConfig as LoopedInstinctConfig,
    InstinctForCausalLM as LoopedInstinctForCausalLM,
)
from trainer.trainer_cli import build_trainer_parser
from trainer.trainer_utils import config_from_args, restore_config_from_checkpoint
import trainer.trainer_utils as trainer_utils


def load_webui_weight_helpers(session_state):
    source_path = Path(__file__).parents[1] / "scripts" / "config_webui.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    functions = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_arch_tag", "_matches_arch_tag"}
    ]
    namespace = {"re": re, "st": SimpleNamespace(session_state=session_state)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


def tiny_config(residual_type="standard", **kwargs):
    return InstinctConfig(
        hidden_size=32,
        num_hidden_layers=2,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=32,
        flash_attn=False,
        residual_type=residual_type,
        **kwargs,
    )


def tiny_model(architecture, residual_type, **extra):
    common = dict(
        hidden_size=32,
        num_hidden_layers=2,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=32,
        flash_attn=False,
        residual_type=residual_type,
    )
    if architecture == "dense":
        return InstinctForCausalLM(InstinctConfig(**common, **extra))
    if architecture == "linear":
        config = LinearInstinctConfig(
            **common, full_attention_interval=2, linear_conv_kernel_dim=2, **extra
        )
        return LinearInstinctForCausalLM(config)
    config = LoopedInstinctConfig(
        **common, prelude_layers=1, loop_iters=2, coda_layers=1, **extra
    )
    return LoopedInstinctForCausalLM(config)


def test_standard_residual_keeps_legacy_parameter_graph():
    model = InstinctForCausalLM(tiny_config())
    names = set(dict(model.named_parameters()))
    assert not any("attn_hc" in name or "output_residual" in name for name in names)


def test_attnres_zero_query_is_uniform_depth_average():
    operator = AttentionResidual(tiny_config("attnres"))
    sources = torch.randn(5, 2, 3, 32)
    torch.testing.assert_close(operator(sources), sources.mean(dim=0))


def test_mhc_sinkhorn_mixer_is_doubly_stochastic():
    config = tiny_config("mhc", hc_mult=3, hc_sinkhorn_iters=20)
    connector = ManifoldHyperConnection(config)
    streams = torch.randn(2, 4, 3, 32)
    post, comb, collapsed = connector(streams)
    assert post.shape == (2, 4, 3)
    assert collapsed.shape == (2, 4, 32)
    torch.testing.assert_close(comb.sum(dim=-1), torch.ones_like(comb.sum(dim=-1)), atol=2e-5, rtol=0)
    torch.testing.assert_close(comb.sum(dim=-2), torch.ones_like(comb.sum(dim=-2)), atol=2e-5, rtol=0)


@pytest.mark.parametrize(
    ("residual_type", "extra"),
    [
        ("mhc", {"hc_mult": 2, "hc_sinkhorn_iters": 3}),
        ("attnres", {"attnres_variant": "full"}),
        ("attnres", {"attnres_variant": "block", "attnres_block_size": 3}),
    ],
)
@pytest.mark.parametrize("checkpoint_mode", [0, 1, 2])
def test_residual_topology_forward_backward(residual_type, extra, checkpoint_mode):
    torch.manual_seed(7)
    model = InstinctForCausalLM(
        tiny_config(residual_type, use_grad_checkpoint=checkpoint_mode, **extra)
    ).train()
    input_ids = torch.randint(0, 64, (2, 6))
    output = model(input_ids, labels=input_ids)
    assert output.logits.shape == (2, 6, 64)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


@pytest.mark.parametrize(
    ("residual_type", "extra"),
    [
        ("mhc", {"hc_mult": 2, "hc_sinkhorn_iters": 3}),
        ("attnres", {"attnres_variant": "full"}),
        ("attnres", {"attnres_variant": "block", "attnres_block_size": 3}),
    ],
)
def test_residual_topology_kv_cache_matches_full_decode(residual_type, extra):
    torch.manual_seed(11)
    model = InstinctForCausalLM(tiny_config(residual_type, **extra)).eval()
    input_ids = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        expected = model(input_ids).logits[:, -1]
        past = None
        for index in range(input_ids.shape[1]):
            output = model(input_ids[:, index:index + 1], past_key_values=past, use_cache=True)
            past = output.past_key_values
    torch.testing.assert_close(output.logits[:, -1], expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("residual_type", "unknown"),
        ("hc_mult", 0),
        ("hc_sinkhorn_iters", 0),
        ("attnres_variant", "unknown"),
        ("attnres_block_size", 0),
    ],
)
def test_invalid_residual_config_is_rejected(field, value):
    with pytest.raises(ValueError):
        InstinctConfig(**{field: value})


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--residual_type", "mhc", "--hc_mult", "3", "--hc_sinkhorn_iters", "7"],
         {"residual_type": "mhc", "hc_mult": 3, "hc_sinkhorn_iters": 7}),
        (["--residual_type", "attnres", "--attnres_variant", "block", "--attnres_block_size", "4"],
         {"residual_type": "attnres", "attnres_variant": "block", "attnres_block_size": 4}),
    ],
)
def test_shared_trainer_cli_threads_residual_options(arguments, expected):
    args = build_trainer_parser("residual-test").parse_args(arguments)
    config = config_from_args(args)
    for field, value in expected.items():
        assert getattr(config, field) == value


@pytest.mark.parametrize(
    ("architecture", "config_cls", "model_cls"),
    [
        ("standard", InstinctConfig, InstinctForCausalLM),
        ("linear", LinearInstinctConfig, LinearInstinctForCausalLM),
        ("looped", LoopedInstinctConfig, LoopedInstinctForCausalLM),
    ],
)
def test_trainer_dispatches_every_backbone_with_residuals(
    monkeypatch, architecture, config_cls, model_cls
):
    args = build_trainer_parser("architecture-test").parse_args([
        "--model_architecture", architecture,
        "--hidden_size", "32",
        "--num_hidden_layers", "2",
        "--residual_type", "mhc",
        "--hc_mult", "2",
        "--hc_sinkhorn_iters", "3",
    ])
    config = config_from_args(
        args,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=32,
        flash_attn=False,
    )
    assert type(config) is config_cls
    assert config.residual_type == "mhc"

    monkeypatch.setattr(
        trainer_utils.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object()
    )
    model, _ = trainer_utils.init_model(config, from_weight="none", device="cpu")
    assert type(model) is model_cls


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"model_architecture": "standard", "residual_type": "standard"}, ""),
        ({"model_architecture": "linear", "residual_type": "standard"}, "_linear"),
        ({"model_architecture": "looped", "residual_type": "mhc"}, "_looped_mhc"),
        ({"model_architecture": "standard", "residual_type": "attnres", "attnres_variant": "full"}, "_attnres_full"),
        ({"model_architecture": "linear", "residual_type": "attnres", "attnres_variant": "block", "attnres_block_size": 3}, "_linear_attnres_block3"),
    ],
)
def test_webui_weight_prefix_identifies_backbone_and_residual(state, expected):
    helpers = load_webui_weight_helpers(state)
    assert helpers["_arch_tag"]() == expected


def test_standard_weight_discovery_rejects_other_topologies():
    matches = load_webui_weight_helpers({})["_matches_arch_tag"]
    assert matches("pretrain_20260828_120000", "")
    for prefix in (
        "pretrain_20260828_120000_linear",
        "pretrain_20260828_120000_looped",
        "pretrain_20260828_120000_mhc",
        "pretrain_20260828_120000_looped_attnres_block2",
    ):
        assert not matches(prefix, "")
    assert matches("pretrain_20260828_120000_looped_mhc", "_looped_mhc")


def test_dense_trainer_import_does_not_load_linear_backbone():
    repo_root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import trainer.trainer_utils; "
            "print('model.model_instinct_linear' in sys.modules)",
        ],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"
    assert "causal-conv1d" not in result.stderr


def test_block_attnres_compile_offset_reuses_graph_patterns():
    config = InstinctConfig(
        hidden_size=32,
        num_hidden_layers=16,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=32,
        flash_attn=False,
        residual_type="attnres",
        attnres_variant="block",
        attnres_block_size=4,
    )
    model = InstinctForCausalLM(config)
    offsets = [layer.attnres_partial_count for layer in model.model.layers]
    assert set(offsets) == {0, 2}
    assert len(set(offsets)) < 8


@pytest.mark.parametrize(
    "script_name",
    [
        "train_pretrain.py",
        "train_full_sft.py",
        "train_dpo.py",
        "train_ppo.py",
        "train_grpo.py",
        "train_agent.py",
        "train_distillation.py",
    ],
)
def test_attnres_training_compile_uses_dynamic_shapes(script_name):
    source = (Path(__file__).parents[1] / "trainer" / script_name).read_text(
        encoding="utf-8"
    )
    assert "dynamic=getattr(" in source
    assert "'residual_type', 'standard') == 'attnres'" in source


@pytest.mark.parametrize("architecture", ["dense", "linear", "looped"])
@pytest.mark.parametrize(
    ("residual_type", "extra"),
    [
        ("mhc", {"hc_mult": 2, "hc_sinkhorn_iters": 3}),
        ("attnres", {"attnres_variant": "block", "attnres_block_size": 3}),
    ],
)
def test_resume_rebuilds_checkpoint_topology_before_strict_load(
    architecture, residual_type, extra
):
    saved_model = tiny_model(architecture, residual_type, **extra)
    current = tiny_config("standard")
    restored_config = restore_config_from_checkpoint(
        current, {"config": saved_model.config.to_dict()}
    )
    model_cls = {
        "standard": InstinctForCausalLM,
        "linear": LinearInstinctForCausalLM,
        "looped": LoopedInstinctForCausalLM,
    }[restored_config.model_architecture]
    restored_model = model_cls(restored_config)
    restored_model.load_state_dict(saved_model.state_dict(), strict=True)
    assert type(restored_model.config) is type(saved_model.config)
    assert restored_config.residual_type == residual_type


def test_legacy_distillation_teacher_inherits_saved_topology_only():
    current_teacher = InstinctConfig(hidden_size=48, num_hidden_layers=3)
    saved_student = tiny_config("mhc", hc_mult=3).to_dict()
    restored = restore_config_from_checkpoint(
        current_teacher,
        {"config": saved_student},
        config_key="teacher_config",
        fallback_topology_key="config",
    )
    assert restored.hidden_size == 48
    assert restored.num_hidden_layers == 3
    assert restored.residual_type == "mhc"
    assert restored.hc_mult == 3


@pytest.mark.parametrize("architecture", ["dense", "linear", "looped"])
@pytest.mark.parametrize(
    ("residual_type", "extra"),
    [
        ("mhc", {"hc_mult": 2, "hc_sinkhorn_iters": 3}),
        ("attnres", {"attnres_variant": "full"}),
        ("attnres", {"attnres_variant": "block", "attnres_block_size": 3}),
    ],
)
def test_every_backbone_weight_roundtrip(architecture, residual_type, extra):
    """Every full-model weight format must reconstruct the selected topology strictly."""
    torch.manual_seed(23)
    model = tiny_model(architecture, residual_type, **extra).eval()
    input_ids = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        expected = model(input_ids).logits

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = tiny_model(architecture, residual_type, **extra).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        actual = restored(input_ids).logits
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("architecture", ["linear", "looped"])
@pytest.mark.parametrize(
    ("residual_type", "extra"),
    [
        ("mhc", {"hc_mult": 2, "hc_sinkhorn_iters": 3}),
        ("attnres", {"attnres_variant": "full"}),
        ("attnres", {"attnres_variant": "block", "attnres_block_size": 3}),
    ],
)
def test_additional_backbones_checkpoint_and_kv_cache(architecture, residual_type, extra):
    torch.manual_seed(29)
    model = tiny_model(
        architecture, residual_type, use_grad_checkpoint=2, **extra
    ).train()
    input_ids = torch.randint(0, 64, (2, 5))
    output = model(input_ids, labels=input_ids)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert any("attn_hc" in name or "residual" in name for name, _ in model.named_parameters())

    model.eval()
    with torch.no_grad():
        expected = model(input_ids[:1]).logits[:, -1]
        past = None
        for index in range(input_ids.shape[1]):
            output = model(
                input_ids[:1, index:index + 1], past_key_values=past, use_cache=True
            )
            past = output.past_key_values
    torch.testing.assert_close(output.logits[:, -1], expected, atol=2e-5, rtol=2e-5)
