"""Pytest infrastructure for the Instinct project.

Import order matters: ``datasets`` MUST be imported before ``torch`` to work
around the Windows pyarrow/torch DLL conflict (see AGENTS.md). Do NOT reorder.
"""

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


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.gpu tests when CUDA is unavailable (or --skip-gpu)."""
    if not config.getoption("--skip-gpu") and torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(
        reason="CUDA skipped via --skip-gpu" if config.getoption("--skip-gpu") else "no CUDA available"
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


def pytest_sessionfinish(session, exitstatus):
    """Treat "no tests collected" (exit 5) as success: test infra must exit 0
    even while it contains no tests yet (T2/T3/T7-T13 will add them)."""
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
