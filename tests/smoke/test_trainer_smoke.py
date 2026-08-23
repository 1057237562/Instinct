"""Task T12: 8-trainer gradient-checkpointing smoke tests.

Runs each of the 8 trainers (``train_pretrain`` / ``train_full_sft`` /
``train_lora`` / ``train_dpo`` / ``train_ppo`` / ``train_grpo`` /
``train_agent`` / ``train_distillation``) for a tiny number of training steps
(1-2 with ``batch_size 1`` on 2 synthetic samples) using a minimal config and
``--use_grad_checkpoint 1``, then asserts the child process exits 0, prints no
Traceback, and reports a finite loss.

Data is generated on the fly into a session-scoped temp dir — no real dataset
is downloaded. Each trainer is invoked via ``subprocess`` because every
``trainer/train_*.py`` is an ``if __name__ == "__main__"`` script (not an
importable ``train`` function).

Dependencies investigated per trainer:

- ``train_pretrain``: ``--from_weight none`` + tiny ``{"text": ...}`` JSONL.
- ``train_full_sft`` / ``train_lora`` / ``train_dpo``: ``--from_weight none``
  (``init_model`` in ``trainer_utils.py`` skips weight loading for ``"none"``),
  so they do NOT need a pre-trained tiny weight.
- ``train_distillation``: ``--from_student_weight none --from_teacher_weight
  none --teacher_use_moe 0`` (self- / random-init distillation, both 64-dim).
- ``train_ppo`` / ``train_grpo`` / ``train_agent``: hard-depend on the reward
  model ``internlm2-1_8b-reward`` (sibling of the repo, loaded unconditionally
  via ``LMForRewardModel`` at startup). When it is absent those tests SKIP with
  the reason documented in the evidence files; when present they run with tiny
  rollout settings.

All tests are marked ``@pytest.mark.gpu`` — conftest.py auto-skips them when
CUDA is unavailable (or ``--skip-gpu``), so this suite never blocks CI.

Evidence written by a session-scoped autouse fixture (runs after all tests):
- ``.sisyphus/evidence/task-12-pretrain-smoke.txt``
- ``.sisyphus/evidence/task-12-all-smoke.txt``
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REWARD_MODEL = (REPO_ROOT / ".." / "internlm2-1_8b-reward").resolve()

TIMEOUT_SEC = 300  # per trainer; timeout = failure

LOSS_RE = re.compile(r"(?:loss|Loss|Reward)\s*:\s*([-+]?\d*\.?\d+)")
TRACEBACK_RE = re.compile(r"Traceback")
NAN_RE = re.compile(r"\b(?:nan|inf)\b", re.IGNORECASE)

# trainer name -> config (script file + per-trainer extra CLI args)
TRAINERS = {
    "train_pretrain": {"script": "train_pretrain.py"},
    "train_full_sft": {"script": "train_full_sft.py"},
    "train_lora": {"script": "train_lora.py"},
    "train_dpo": {"script": "train_dpo.py"},
    "train_ppo": {"script": "train_ppo.py", "requires_reward_model": True},
    "train_grpo": {"script": "train_grpo.py", "requires_reward_model": True},
    "train_agent": {"script": "train_agent.py", "requires_reward_model": True},
    "train_distillation": {"script": "train_distillation.py"},
}

# per-session test results, consumed by the evidence fixture
RESULTS = []


# ---------------------------------------------------------------------------
# Session workspace: synthetic data + output dir in a temp folder
# ---------------------------------------------------------------------------

def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.fixture(scope="session")
def smoke_workspace():
    """Session-scoped temp dir with tiny synthetic data for all trainers."""
    tmp = Path(tempfile.mkdtemp(prefix="instinct_smoke_"))
    out = tmp / "out"
    out.mkdir()
    data = tmp / "data"
    data.mkdir()

    _write_jsonl(data / "pretrain.jsonl", [
        {"text": "秋天是收获的季节，树叶金黄，果实累累。"},
        {"text": "人工智能正在改变世界，带来新的可能性。"},
    ])
    _write_jsonl(data / "sft.jsonl", [
        {"conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是Instinct。"},
        ]},
        {"conversations": [
            {"role": "user", "content": "今天天气如何"},
            {"role": "assistant", "content": "今天阳光明媚。"},
        ]},
    ])
    _write_jsonl(data / "dpo.jsonl", [
        {"chosen": [
            {"role": "user", "content": "什么是AI"},
            {"role": "assistant", "content": "AI是人工智能。"},
        ], "rejected": [
            {"role": "user", "content": "什么是AI"},
            {"role": "assistant", "content": "我不知道。"},
        ]},
        {"chosen": [
            {"role": "user", "content": "1加1等于几"},
            {"role": "assistant", "content": "等于2。"},
        ], "rejected": [
            {"role": "user", "content": "1加1等于几"},
            {"role": "assistant", "content": "等于3。"},
        ]},
    ])
    # RL data files (only used if a reward model is present)
    _write_jsonl(data / "rlaif.jsonl", [
        {"conversations": [
            {"role": "user", "content": "写一首短诗"},
            {"role": "assistant", "content": "明月出天山。"},
        ]},
    ])
    _write_jsonl(data / "agent.jsonl", [
        {"conversations": [
            {"role": "user", "content": "1+1=?"},
            {"role": "assistant", "content": "2"},
        ], "gt": "2"},
    ])

    ws = {"tmp": tmp, "out": out, "data": data}
    try:
        yield ws
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_trainer(script, args, timeout=TIMEOUT_SEC):
    """Run one trainer script as a subprocess from the repo root.

    Returns (returncode, combined_output, timed_out). ``returncode`` is None on
    timeout. PYTHONUTF8=1 is required by torch.compile-free eager paths too (the
    trainer modules set their own package/sys.path, but UTF-8 keeps Windows
    child process output decodable).
    """
    cmd = [sys.executable, str(REPO_ROOT / "trainer" / script), *args]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode, output, False
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
        output += "\n" + ((exc.stderr or b"") if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return None, str(output), True


def _build_args(name, ws):
    """Assemble the full CLI arg list for a trainer."""
    out = ws["out"]
    data = ws["data"]
    # train_distillation has no --hidden_size/--num_hidden_layers (it uses the
    # --student_*/--teacher_* variants), so those two are added conditionally.
    size_args = [] if name == "train_distillation" else ["--hidden_size", "64", "--num_hidden_layers", "2"]
    base = size_args + [
        "--max_seq_len", "32",
        "--batch_size", "1",
        "--epochs", "1",
        "--use_grad_checkpoint", "1",
        "--num_workers", "0",
        "--log_interval", "1",
        "--accumulation_steps", "1",
        "--save_dir", str(out),
    ]
    extra = {
        "train_pretrain": [
            "--data_path", str(data / "pretrain.jsonl"),
            "--from_weight", "none",
            "--save_weight", "smoke_pretrain",
        ],
        "train_full_sft": [
            "--data_path", str(data / "sft.jsonl"),
            "--from_weight", "none",
            "--save_weight", "smoke_sft",
        ],
        "train_lora": [
            "--data_path", str(data / "sft.jsonl"),
            "--from_weight", "none",
            "--lora_name", "smoke_lora",
        ],
        "train_dpo": [
            "--data_path", str(data / "dpo.jsonl"),
            "--from_weight", "none",
            "--save_weight", "smoke_dpo",
        ],
        "train_distillation": [
            "--data_path", str(data / "sft.jsonl"),
            "--from_student_weight", "none",
            "--from_teacher_weight", "none",
            "--student_hidden_size", "64",
            "--student_num_layers", "2",
            "--teacher_hidden_size", "64",
            "--teacher_num_layers", "2",
            "--student_use_moe", "0",
            "--teacher_use_moe", "0",
            "--save_weight", "smoke_distill",
        ],
        # RL trainers: only reached when the reward model is present.
        "train_ppo": [
            "--data_path", str(data / "rlaif.jsonl"),
            "--from_weight", "none",
            "--max_gen_len", "32",
            "--thinking_ratio", "0.0",
            "--reward_model_path", str(REWARD_MODEL),
            "--save_weight", "smoke_ppo",
        ],
        "train_grpo": [
            "--data_path", str(data / "rlaif.jsonl"),
            "--from_weight", "none",
            "--max_gen_len", "32",
            "--num_generations", "1",
            "--thinking_ratio", "0.0",
            "--reward_model_path", str(REWARD_MODEL),
            "--save_weight", "smoke_grpo",
        ],
        "train_agent": [
            "--data_path", str(data / "agent.jsonl"),
            "--from_weight", "none",
            "--max_gen_len", "32",
            "--max_total_len", "96",
            "--num_generations", "1",
            "--thinking_ratio", "0.0",
            "--reward_model_path", str(REWARD_MODEL),
            "--save_weight", "smoke_agent",
        ],
    }
    return base + extra[name]


def _check_output(output, returncode, timed_out):
    """Assert exit 0, no Traceback, no NaN, and a finite loss value."""
    if timed_out:
        raise AssertionError(f"trainer timed out after {TIMEOUT_SEC}s")
    if returncode != 0:
        raise AssertionError(f"trainer exited with code {returncode}")
    if TRACEBACK_RE.search(output):
        raise AssertionError("trainer printed a Traceback")
    if NAN_RE.search(output):
        raise AssertionError("trainer output contains nan/inf")
    m = LOSS_RE.search(output)
    if m is None:
        raise AssertionError("no 'loss:' value found in trainer output")
    try:
        loss = float(m.group(1))
    except ValueError:
        raise AssertionError(f"unparseable loss value: {m.group(1)!r}")
    if not math.isfinite(loss):
        raise AssertionError(f"loss is not finite: {loss}")
    return loss


# ---------------------------------------------------------------------------
# The tests (one per trainer)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("name", list(TRAINERS), ids=lambda n: n)
def test_trainer_smoke(name, smoke_workspace):
    cfg = TRAINERS[name]
    rec = {"trainer": name, "status": "RUN", "cmd": None, "detail": ""}
    RESULTS.append(rec)

    if cfg.get("requires_reward_model") and not REWARD_MODEL.exists():
        rec["status"] = "SKIP"
        rec["detail"] = (
            f"reward model not available ({REWARD_MODEL}); hard dependency via "
            "LMForRewardModel at startup"
        )
        pytest.skip(rec["detail"])

    args = _build_args(name, smoke_workspace)
    rec["cmd"] = f"python trainer/{cfg['script']} " + " ".join(args)
    try:
        returncode, output, timed_out = run_trainer(cfg["script"], args)
        loss = _check_output(output, returncode, timed_out)
        rec["status"] = "PASS"
        rec["detail"] = f"exit=0, loss={loss:.6f} (finite)"
    except BaseException as exc:  # record every failure for the evidence file
        rec["status"] = "FAIL"
        rec["detail"] = str(exc)
        raise


# ---------------------------------------------------------------------------
# Evidence: task-12-pretrain-smoke.txt + task-12-all-smoke.txt
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _write_evidence():
    yield
    evidence_dir = REPO_ROOT / ".sisyphus" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    by_name = {r["trainer"]: r for r in RESULTS}

    def line_for(name):
        rec = by_name.get(name)
        if rec is None:
            return f"trainer={name} status=NOT_RUN detail=(test not executed in this session)"
        if rec["status"] == "SKIP":
            return f"trainer={name} status=SKIP detail={rec['detail']}"
        return f"trainer={name} status={rec['status']} detail={rec['detail']}"

    all_lines = [line_for(name) for name in TRAINERS]
    (evidence_dir / "task-12-all-smoke.txt").write_text(
        "Instinct 8-trainer smoke (T12): use_grad_checkpoint=1, hidden=64, layers=2, "
        "max_seq_len=32, batch=1, epochs=1, synthetic data\n"
        + "\n".join(all_lines)
        + "\n",
        encoding="utf-8",
    )

    pretrain = by_name.get("train_pretrain")
    if pretrain is not None:
        pretrain_lines = [
            f"trainer=train_pretrain status={pretrain['status']} detail={pretrain['detail']}",
            f"cmd={pretrain['cmd']}",
        ]
    else:
        pretrain_lines = ["trainer=train_pretrain status=NOT_RUN detail=(not executed)"]
    (evidence_dir / "task-12-pretrain-smoke.txt").write_text(
        "Instinct pretrain smoke (T12): --use_grad_checkpoint 1, tiny config, synthetic data\n"
        + "\n".join(pretrain_lines)
        + "\n",
        encoding="utf-8",
    )
