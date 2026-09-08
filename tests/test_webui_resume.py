"""Regression tests for WebUI resume-checkpoint selection."""

import ast
from pathlib import Path
from types import SimpleNamespace


def _load_resolver(session_state, *, existing, latest):
    source_path = Path(__file__).parents[1] / "scripts" / "config_webui.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    resolver = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_save_prefix"
    )
    namespace = {
        "st": SimpleNamespace(session_state=session_state),
        "_arch_tag": lambda: "",
        "_read_paused_state": lambda: None,
        "_resume_checkpoint_exists": lambda weight, hidden_size, use_moe: weight in existing,
        "_matches_arch_tag": lambda prefix, arch_tag: True,
        "_latest_checkpoint_prefix": lambda train_type, hidden_size, use_moe, arch_tag: latest,
        "_default_weight_prefix": lambda train_type: train_type,
    }
    exec(
        compile(ast.Module(body=[resolver], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace["_resolve_save_prefix"]


def test_stale_session_prefix_does_not_mask_valid_checkpoint():
    state = {"save_prefix_pretrain": "pretrain_new_run_without_checkpoint"}
    resolve = _load_resolver(
        state,
        existing={"pretrain_20260901_211419"},
        latest="pretrain_20260901_211419",
    )

    assert resolve("pretrain", 768, False, True, "unused") == "pretrain_20260901_211419"
    assert "save_prefix_pretrain" not in state


def test_existing_session_prefix_remains_resume_target():
    state = {"save_prefix_pretrain": "pretrain_paused_run"}
    resolve = _load_resolver(
        state,
        existing={"pretrain_paused_run"},
        latest="pretrain_older_run",
    )

    assert resolve("pretrain", 768, False, True, "unused") == "pretrain_paused_run"
