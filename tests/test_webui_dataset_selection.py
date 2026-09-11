"""Regression tests for WebUI dataset discovery and staged SFT starts."""

import ast
import os
from pathlib import Path


def _load_helpers():
    source_path = Path(__file__).parents[1] / "scripts" / "config_webui.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    assignments = {
        "_DATASET_KINDS_BY_TRAIN_TYPE",
        "_DEFAULT_DATASET_NAMES",
        "_BASE_WEIGHT_TYPE",
    }
    functions = {
        "_dataset_kind",
        "_available_training_datasets",
        "_base_weight_type",
        "_is_completed_sft_weight",
        "_packing_preprocess_workers",
    }
    selected = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & assignments:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    namespace = {"os": os}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace


def test_any_jsonl_starting_with_sft_is_classified_as_sft():
    helpers = _load_helpers()

    assert helpers["_dataset_kind"]("sft_codealpaca_20k.jsonl") == "sft"
    assert helpers["_dataset_kind"]("SFT-custom.JSONL") == "sft"
    assert helpers["_dataset_kind"]("custom_sft.jsonl") is None


def test_webui_caps_windows_packing_workers():
    workers = _load_helpers()["_packing_preprocess_workers"]

    assert workers(32, platform_name="nt", cpu_count=64) == 4
    assert workers(None, platform_name="nt", cpu_count=64) == 4
    assert workers(32, platform_name="posix", cpu_count=64) == 32


def test_full_sft_discovery_only_lists_sft_prefixed_jsonl(tmp_path):
    helpers = _load_helpers()
    for name in ("sft_t2t.jsonl", "sft_codealpaca.jsonl", "pretrain_t2t.jsonl", "notes.txt"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    discovered = helpers["_available_training_datasets"]("full_sft", tmp_path)

    assert [Path(path).name for path in discovered] == [
        "sft_codealpaca.jsonl",
        "sft_t2t.jsonl",
    ]


def test_default_mini_dataset_is_first(tmp_path):
    helpers = _load_helpers()
    for name in ("sft_t2t.jsonl", "sft_t2t_mini.jsonl", "sft_code.jsonl"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    discovered = helpers["_available_training_datasets"]("full_sft", tmp_path)

    assert Path(discovered[0]).name == "sft_t2t_mini.jsonl"


def test_continue_sft_uses_completed_sft_weights_not_pretrain():
    helpers = _load_helpers()

    assert helpers["_base_weight_type"]("full_sft", False) == "pretrain"
    assert helpers["_base_weight_type"]("full_sft", True) == "full_sft"
    assert helpers["_is_completed_sft_weight"]("out/full_sft_run_768.pth")
    assert not helpers["_is_completed_sft_weight"]("checkpoints/full_sft_run_768_resume.pth")
    assert not helpers["_is_completed_sft_weight"]("out/pretrain_run_768.pth")
