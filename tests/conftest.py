"""Pytest infrastructure for the Instinct project.

Import order matters: ``datasets`` MUST be imported before ``torch`` to work
around the Windows pyarrow/torch DLL conflict (see AGENTS.md). Do NOT reorder.
"""

from pathlib import Path
import sys

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import datasets  # noqa: F401  # must stay before torch (Windows pyarrow/torch DLL conflict)
import torch  # noqa: F401
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-gpu",
        action="store_true",
        default=False,
        help="Force-skip all tests marked with @pytest.mark.gpu",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run expensive subprocess/integration tests",
    )


def pytest_collection_modifyitems(config, items):
    """Skip unavailable GPU and opt-in expensive integration tests."""
    skip_gpu = None
    if config.getoption("--skip-gpu") or not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(
            reason="CUDA skipped via --skip-gpu" if config.getoption("--skip-gpu") else "no CUDA available"
        )
    skip_slow = None if config.getoption("--run-slow") else pytest.mark.skip(
        reason="expensive integration test; pass --run-slow to enable"
    )
    for item in items:
        if skip_gpu is not None and "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if skip_slow is not None and "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_sessionfinish(session, exitstatus):
    """Treat "no tests collected" (exit 5) as success: test infra must exit 0
    even while it contains no tests yet (T2/T3/T7-T13 will add them)."""
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
