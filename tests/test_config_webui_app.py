"""End-to-end smoke tests for the Config WebUI script via Streamlit AppTest."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


SOURCE = Path(__file__).parents[1] / "scripts" / "config_webui.py"
STATE_FILE = Path(__file__).parents[1] / "trainer" / "webui_state.json"


@pytest.fixture()
def guard_state_file():
    """AppTest 运行会真实读写 trainer/webui_state.json；测试前后保护用户存档。"""
    backup = STATE_FILE.read_bytes() if STATE_FILE.exists() else None
    STATE_FILE.unlink(missing_ok=True)
    yield
    if backup is None:
        STATE_FILE.unlink(missing_ok=True)
    else:
        STATE_FILE.write_bytes(backup)


def _preset_fingerprint_fn():
    """Extract _preset_fingerprint without importing the Streamlit script."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_preset_fingerprint":
            selected.append(node)
        elif isinstance(node, ast.Assign) and {
            target.id for target in node.targets if isinstance(target, ast.Name)
        } & {"PRESETS", "_LEGACY_PRESET_NAMES"}:
            selected.append(node)
    namespace = {"json": json, "hashlib": hashlib}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_preset_fingerprint"]


def _run_app():
    app = AppTest.from_file(str(SOURCE), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def test_app_boots_and_autosaves_state(guard_state_file):
    app = _run_app()
    # 页面正常渲染，且本轮交互结束时自动保存已写入状态文件。
    assert STATE_FILE.exists()
    saved = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    assert saved["preset"] == app.session_state["preset"]
    assert saved["preset_fingerprint"] == _preset_fingerprint_fn()(saved["preset"])


def test_restore_keeps_user_tweaks_when_fingerprint_matches(guard_state_file):
    fingerprint = _preset_fingerprint_fn()("Instinct V1 MoE")
    STATE_FILE.write_text(json.dumps({
        "preset": "Instinct V1 MoE",
        "preset_fingerprint": fingerprint,
        "batch_size": 5,
        "max_seq_len": 2048,
    }), encoding="utf-8")
    app = _run_app()
    # 指纹一致：预设定义未变，用户在预设之上的修改必须原样保留。
    assert app.session_state["preset"] == "Instinct V1 MoE"
    assert app.session_state["batch_size"] == 5
    assert app.session_state["max_seq_len"] == 2048


def test_stale_state_without_fingerprint_reapplies_preset(guard_state_file):
    STATE_FILE.write_text(json.dumps({
        "preset": "Instinct V1 MoE",
        "batch_size": 5,
        "max_seq_len": 2048,
    }), encoding="utf-8")
    app = _run_app()
    # 无指纹 = 旧版本存档：重新套用当前预设定义（V1 MoE 强制 batch_size=1）。
    assert app.session_state["batch_size"] == 1
    assert app.session_state["max_seq_len"] == 4096


def test_shutdown_toggle_renders_countdown_setting(guard_state_file):
    app = AppTest.from_file(str(SOURCE), default_timeout=60)
    app.session_state["shutdown_after_train"] = True
    app.run()
    assert not app.exception
    assert any(
        checkbox.key == "shutdown_after_train" and checkbox.value
        for checkbox in app.checkbox
    )
    assert any(
        number_input.key == "shutdown_delay_seconds" and number_input.value == 60
        for number_input in app.number_input
    )


def test_launch_command_preview_rendered(guard_state_file):
    app = _run_app()
    snippets = [c.value for c in app.code]
    # 命令预览与 Start Training 的实际启动共用 _build_training_command：
    # 预览里必须出现训练脚本、保存前缀与所选数据集。
    preview = next(s for s in snippets if "train_pretrain.py" in s)
    assert "--save_weight" in preview
    assert "--epochs" in preview
    assert "--data_path" in preview
    assert preview.startswith('"') or preview.startswith("'") or "python" in preview
