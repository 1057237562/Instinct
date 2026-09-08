import os

import pytest

from scripts.mofe_ui import parse_expert_rows, validate_expert_bank, write_run_manifest


def test_parse_expert_rows_supports_named_and_path_only_rows():
    rows = parse_expert_rows("code | programming | C:/w/code.pth\nC:/w/math.pth")
    assert rows == [
        {"name": "code", "domain": "programming", "path": "C:/w/code.pth"},
        {"name": "expert_01", "domain": "", "path": "C:/w/math.pth"},
    ]


def test_parse_expert_rows_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        parse_expert_rows("same | a | one.pth\nsame | b | two.pth")


def test_validate_and_write_manifest(tmp_path):
    base = tmp_path / "base.pth"
    first = tmp_path / "one.pth"
    second = tmp_path / "two.pth"
    for path in (base, first, second):
        path.write_bytes(b"checkpoint")
    payload, errors = validate_expert_bank(
        str(base), f"one | d1 | {first}\ntwo | d2 | {second}", 2,
    )
    assert errors == []

    result = write_run_manifest(str(tmp_path), "run", payload)
    assert os.path.isfile(result)
    assert not list(tmp_path.glob("mofe_manifests/.mofe-*"))
