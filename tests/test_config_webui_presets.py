import ast
import hashlib
import json
import math
import re
from pathlib import Path

import pytest


SOURCE = Path(__file__).parents[1] / "scripts" / "config_webui.py"


def _preset_helpers():
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    selected = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & {
                "PRESETS", "_ROPE_PRESET_KEYS", "_OPTIONAL_PRESET_DEFAULTS",
                "_LEGACY_PRESET_NAMES",
            }:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "compute_intermediate_size", "compute_head_dim", "calc_params",
            "_config_preset_value", "arch_diagram", "estimate_bucket_vram_plan",
            "_preset_fingerprint",
        }:
            selected.append(node)
    namespace = {"math": math, "json": json, "hashlib": hashlib}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace


def test_instinct_v1_replaces_instinct2_with_current_architecture():
    helpers = _preset_helpers()
    presets = helpers["PRESETS"]
    assert "instinct2" not in presets
    config = presets["Instinct V1"]
    assert config["hidden_size"] == 768
    assert config["num_hidden_layers"] == 20
    assert config["num_attention_heads"] == 8
    assert config["num_key_value_heads"] == 4
    assert config["inference_rope_scaling"] is True
    assert config["factor"] == 4.0
    assert config["original_max_position_embeddings"] == 4096.0
    assert config["attention_factor"] == 1.1385
    assert helpers["calc_params"](config)["Total Params"]["value"] == 152_406_528


def test_new_moe_presets_keep_top1_router_task_gradient():
    for config in _preset_helpers()["PRESETS"].values():
        if config["use_moe"] and config["num_experts_per_tok"] == 1:
            assert config["norm_topk_prob"] is False


def test_instinct_v2_uses_latent_recurrent_depth_architecture():
    helpers = _preset_helpers()
    presets = helpers["PRESETS"]
    assert "instinct-3" not in presets
    assert "instinct-3-moe" not in presets
    assert "instinct2-small" not in presets
    assert presets["Instinct V2"]["use_moe"] is False
    assert presets["Instinct V2 MoE"]["use_moe"] is True
    assert presets["Instinct V2 Small"]["hidden_size"] == 512
    for name in ("Instinct V2", "Instinct V2 MoE", "Instinct V2 Small"):
        config = presets[name]
        assert config["model_architecture"] == "looped"
        assert (config["prelude_layers"], config["recurrent_layers"],
                config["coda_layers"]) == (2, 4, 2)
        assert config["recurrence_sampling"] == "lognormal_poisson"
        assert config["use_input_injection"] is True
    config = presets["Instinct V2"]
    assert config["hidden_size"] == 1248
    assert config["intermediate_size"] == 4224
    assert config["num_attention_heads"] == 13
    assert config["num_key_value_heads"] == 13
    assert config["hidden_size"] // config["num_attention_heads"] == 96
    assert config["rope_theta"] == 50_000.0
    assert config["qk_bias"] is True
    assert config["qk_norm"] is False
    assert config["loop_iters"] == 32
    assert config["mean_backprop_depth"] == 8
    assert config["recurrence_log_normal_sigma"] == 1.0
    assert config["max_recurrence"] == 256
    for name in ("Instinct V2 MoE", "Instinct V2 Small"):
        assert presets[name]["loop_iters"] == 8
        assert presets[name]["mean_backprop_depth"] == 4
    assert helpers["_LEGACY_PRESET_NAMES"]["instinct-3"] == "Instinct V2"
    assert helpers["_LEGACY_PRESET_NAMES"]["instinct-3-moe"] == "Instinct V2 MoE"


def test_instinct_v2_parameter_estimate_includes_adapter_and_sandwich_norms():
    helpers = _preset_helpers()
    config = helpers["PRESETS"]["Instinct V2"]
    assert helpers["calc_params"](config)["Total Params"]["value"] == 187_521_984


def test_instinct_v2_architecture_diagram_renders_recurrent_graph():
    helpers = _preset_helpers()
    diagram = helpers["arch_diagram"](helpers["PRESETS"]["Instinct V2"])
    assert "187,521,984" in diagram
    assert "Prelude x2" in diagram
    assert "Input Injection Adapter" in diagram
    assert "Core Transformer x4" in diagram
    assert "target mean 32" in diagram
    assert "gradient through final 8 recurrences" in diagram
    assert "Coda x2" in diagram
    assert "Mean effective depth" in diagram
    assert "Layer 1" not in diagram


def test_instinct_v1_architecture_diagram_remains_sequential():
    helpers = _preset_helpers()
    diagram = helpers["arch_diagram"](helpers["PRESETS"]["Instinct V1"])
    assert "Layer 1" in diagram
    assert "recurrent-shell" not in diagram


def test_instinct_v1_moe_uses_normalized_top2_routing():
    helpers = _preset_helpers()
    config = helpers["PRESETS"]["Instinct V1 MoE"]
    assert config["model_architecture"] == "standard"
    assert config["use_moe"] is True
    assert config["hidden_size"] == 512
    assert config["num_hidden_layers"] == 32
    assert config["num_attention_heads"] == 16
    assert config["num_key_value_heads"] == 4
    assert config["num_experts"] == 8
    assert config["num_experts_per_tok"] == 2
    assert config["norm_topk_prob"] is True
    assert config["moe_intermediate_size"] == 1664
    assert config["rope_scaling_type"] == "longrope"
    assert config["factor"] == 8.0
    assert config["original_max_position_embeddings"] == 4096.0
    assert config["short_factor"] == [1.0] * 16
    assert config["long_factor"] == [8.0] * 16
    assert config["short_attention_factor"] == 1.0
    breakdown = helpers["calc_params"](config)
    assert breakdown["Total Params"]["value"] == 678_726_144
    assert breakdown["Active Total"]["value"] == 187_992_576
    diagram = helpers["arch_diagram"](config)
    assert "MoE-FFN (8E, top-2)" in diagram
    assert "Layer 32" in diagram

    vram = helpers["estimate_bucket_vram_plan"](
        config, target_gb=16.0, bucket_count=2, optimizer="muon",
        param_dtype="fp32", use_compile=True,
        compile_mode="max-autotune-no-cudagraphs",
    )
    assert vram["persistent_gb"] == pytest.approx(7.5854, abs=1e-3)
    assert vram["token_budget"] == 12553
    assert vram["token_budget"] // 4096 == 3

    serialized = json.loads(
        (SOURCE.parents[1] / "trainer" / "config_instinct_v1_moe.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        helpers["_config_preset_value"](serialized, key) == value
        for key, value in config.items()
    )


def test_current_serialized_configs_match_their_training_presets():
    helpers = _preset_helpers()
    for filename, preset_name in (
        ("config_pretrain.json", "Instinct V1 MoE"),
        ("config_full_sft.json", "Instinct V2"),
        ("config_dpo.json", "Instinct V2"),
        ("config_instinct_v2.json", "Instinct V2"),
    ):
        preset = helpers["PRESETS"][preset_name]
        current = json.loads(
            (SOURCE.parents[1] / "trainer" / filename).read_text(encoding="utf-8")
        )
        assert all(
            helpers["_config_preset_value"](current, key) == value
            for key, value in preset.items()
        )


def test_training_defaults_use_single_step_gradient_accumulation():
    source = SOURCE.read_text(encoding="utf-8")
    assert "st.session_state.accumulation_steps = 16" not in source
    assert "st.session_state.accumulation_steps = 4" not in source


def test_preset_fingerprint_tracks_definition_changes():
    helpers = _preset_helpers()
    fingerprint = helpers["_preset_fingerprint"]
    # 同一定义的指纹稳定；旧版预设名映射到现名后指纹一致。
    assert fingerprint("Instinct V2") == fingerprint("instinct-3")
    assert fingerprint("Instinct V2") != fingerprint("Instinct V2 MoE")
    # 非预设名（Custom / 未知）没有指纹，恢复时总会重新套用默认值。
    assert fingerprint("Custom") == ""
    assert fingerprint("missing-preset") == ""
    assert fingerprint(None) == ""


def test_webui_state_restore_keeps_tweaks_for_unchanged_presets():
    """指纹一致时不得重新套用预设，否则用户手动修改每次刷新都被冲掉。"""
    source = SOURCE.read_text(encoding="utf-8")
    assert 'saved_state.get("preset_fingerprint") != _preset_fingerprint(restored_preset)' in source
    assert 'init_from_preset(restored_preset)' in source


def test_training_metrics_parse_only_latest_resume_segment():
    """续训向同一日志追加新一轮（各自 `# epochs:` 开头），两轮的 global step
    都从 1 重新计数；图表若混用整份日志会画成锯齿，必须只解析最后一轮。"""
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    fn = [node for node in module.body
          if isinstance(node, ast.FunctionDef) and node.name == "parse_training_metrics"]
    namespace = {"re": re}
    exec(compile(ast.Module(body=fn, type_ignores=[]), str(SOURCE), "exec"), namespace)
    parse = namespace["parse_training_metrics"]

    resumed_log = (
        "# torch.compile (Triton): ON\n# epochs: 1\n"
        "Epoch:[1/1](1/2) loss: 3.5 lr: 0.0005\n"
        "Epoch:[1/1](2/2) loss: 3.4 lr: 0.0004 checkpoint_saved: 1\n"
        "# torch.compile (Triton): ON\n# epochs: 2\n"
        "Epoch:[1/2](1/2) loss: 3.0 lr: 0.0003\n"
        "Epoch:[2/2](2/2) loss: 2.8 lr: 0.0001 checkpoint_saved: 1\n"
    )
    rows = parse(resumed_log)
    # 续训前的一段不再混入；global step 跨 epoch 连续（ep2 的第 2 步 = 全局第 4 步）。
    assert [row["_global_step"] for row in rows] == [1, 4]
    assert [row["loss"] for row in rows] == [3.0, 2.8]

    # 没有 run 分段头的旧日志仍按整份解析。
    assert len(parse("Epoch:[1/1](1/1) loss: 1.0 lr: 0.1\n")) == 1
    assert parse("no metrics here\n") == []
