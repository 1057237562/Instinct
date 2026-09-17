import ast
import json
import math
from pathlib import Path


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
            "_config_preset_value", "arch_diagram",
        }:
            selected.append(node)
    namespace = {"math": math}
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


def test_instinct_v1_moe_preserves_dense_per_token_ffn_capacity():
    helpers = _preset_helpers()
    config = helpers["PRESETS"]["Instinct V1 MoE"]
    assert config["model_architecture"] == "standard"
    assert config["use_moe"] is True
    assert config["hidden_size"] == 512
    assert config["num_hidden_layers"] == 32
    assert config["num_attention_heads"] == 16
    assert config["num_key_value_heads"] == 4
    assert config["num_experts"] == 8
    assert config["num_experts_per_tok"] == 1
    assert config["moe_intermediate_size"] == 1664
    breakdown = helpers["calc_params"](config)
    assert breakdown["Total Params"]["value"] == 678_726_144
    assert breakdown["Active Total"]["value"] == 106_203_648
    diagram = helpers["arch_diagram"](config)
    assert "MoE-FFN (8E, top-1)" in diagram
    assert "Layer 32" in diagram

    serialized = json.loads(
        (SOURCE.parents[1] / "trainer" / "config_instinct_v1_moe.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        helpers["_config_preset_value"](serialized, key) == value
        for key, value in config.items()
    )


def test_current_serialized_configs_match_instinct_v2_preset():
    helpers = _preset_helpers()
    preset = helpers["PRESETS"]["Instinct V2"]
    for filename in (
        "config_pretrain.json", "config_full_sft.json",
        "config_dpo.json", "config_instinct_v2.json",
    ):
        current = json.loads(
            (SOURCE.parents[1] / "trainer" / filename).read_text(encoding="utf-8")
        )
        assert all(
            helpers["_config_preset_value"](current, key) == value
            for key, value in preset.items()
        )
