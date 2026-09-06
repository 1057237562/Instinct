"""Regression tests for recovered WebUI training-process status."""

import ast
import os
import re
import time
from pathlib import Path


class _SessionState(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def _load_status_helpers(checkpoints_dir):
    source_path = Path(__file__).parents[1] / "scripts" / "config_webui.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "_training_log_is_complete",
        "_completed_checkpoint_exists",
        "_detached_training_status",
        "_training_script_from_argv",
    }
    functions = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "os": os,
        "re": re,
        "time": time,
        "_checkpoints_dir": lambda: str(checkpoints_dir),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace


def _write_log(path, progress):
    path.write_text(progress, encoding="utf-8")
    return path


def test_detached_run_stays_running_while_process_exists(tmp_path):
    helpers = _load_status_helpers(tmp_path / "checkpoints")
    log_path = _write_log(tmp_path / "train_full_sft_run.log", "starting\n")

    assert helpers["_detached_training_status"](
        str(log_path), True, None
    ) == "running"


def test_training_process_detection_requires_a_script_argv(tmp_path):
    helper = _load_status_helpers(tmp_path / "checkpoints")[
        "_training_script_from_argv"
    ]

    assert helper([
        r"E:\miniconda3\python.exe",
        "-u",
        r"D:\AI\Instinct\trainer\train_full_sft.py",
        "--from_resume",
        "1",
    ]) == "train_full_sft"
    assert helper([
        r"E:\miniconda3\python.exe",
        "-c",
        "print('train_full_sft.py')",
    ]) is None


def test_detached_run_becomes_success_after_final_checkpoint(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    helpers = _load_status_helpers(checkpoints)
    log_path = _write_log(
        tmp_path / "train_full_sft_20260903_010116.log",
        "Epoch:[1/1](18401/18401), loss: 1.5830\n",
    )
    checkpoint = checkpoints / "full_sft_20260903_010116_768_resume.pth"
    checkpoint.write_bytes(b"complete")
    checkpoint_mtime = log_path.stat().st_mtime + 1
    os.utime(checkpoint, (checkpoint_mtime, checkpoint_mtime))

    assert helpers["_detached_training_status"](
        str(log_path), False, None
    ) == "success"


def test_final_log_without_new_checkpoint_is_not_success(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    helpers = _load_status_helpers(checkpoints)
    log_path = _write_log(
        tmp_path / "train_full_sft_run.log",
        "Epoch:[1/1](10/10), loss: 1.0\n",
    )
    stale_checkpoint = checkpoints / "full_sft_run_768_resume.pth"
    stale_checkpoint.write_bytes(b"periodic")
    stale_mtime = log_path.stat().st_mtime - 10
    os.utime(stale_checkpoint, (stale_mtime, stale_mtime))

    assert helpers["_detached_training_status"](
        str(log_path), False, None, now=log_path.stat().st_mtime + 61
    ) == "failed"


def test_paused_marker_wins_when_process_has_exited(tmp_path):
    helpers = _load_status_helpers(tmp_path / "checkpoints")
    log_path = _write_log(tmp_path / "train_pretrain_run.log", "starting\n")

    assert helpers["_detached_training_status"](
        str(log_path), False, {"train_type": "pretrain"}
    ) == "paused"


def test_live_process_overrides_stale_failed_session(tmp_path):
    source_path = Path(__file__).parents[1] / "scripts" / "config_webui.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_try_recover_training_state"
    )
    session_state = _SessionState(train_status="failed")
    namespace = {
        "__file__": str(source_path),
        "os": os,
        "st": type("FakeStreamlit", (), {"session_state": session_state})(),
        "_latest_train_log": lambda _trainer_dir: str(tmp_path / "active.log"),
        "_find_running_train_process": lambda: (60104, "train_full_sft"),
        "_read_paused_state": lambda: None,
        "_detached_training_status": lambda *_args, **_kwargs: "failed",
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )

    namespace["_try_recover_training_state"]()

    assert session_state["train_status"] == "running"
    assert session_state["train_type"] == "full_sft"
    assert session_state["train_log_path"] == str(tmp_path / "active.log")
