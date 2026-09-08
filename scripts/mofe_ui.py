"""Pure helpers for the MoFE section of config_webui."""

from __future__ import annotations

import json
import os
import tempfile


def parse_expert_rows(text: str) -> list[dict[str, str]]:
    """Parse `name | domain | path` rows; a path-only row is also accepted."""
    rows = []
    for line_number, raw_line in enumerate(str(text).splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) == 1:
            name, domain, path = f"expert_{len(rows):02d}", "", parts[0]
        elif len(parts) == 3:
            name, domain, path = parts
        else:
            raise ValueError(f"line {line_number}: use `name | domain | path` or a path only")
        if not name or not path:
            raise ValueError(f"line {line_number}: expert name and path are required")
        rows.append({"name": name, "domain": domain, "path": path})
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("expert names must be unique")
    return rows


def validate_expert_bank(base_model: str, expert_text: str, expected_count: int) -> tuple[dict, list[str]]:
    base_model = os.path.abspath(os.path.expanduser(os.path.expandvars(str(base_model).strip())))
    rows = parse_expert_rows(expert_text)
    errors = []
    if not os.path.isfile(base_model):
        errors.append(f"shared base checkpoint does not exist: {base_model}")
    if len(rows) != int(expected_count):
        errors.append(f"expected {expected_count} experts, got {len(rows)}")
    resolved = []
    for row in rows:
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(row["path"])))
        if not os.path.isfile(path):
            errors.append(f"expert checkpoint does not exist: {path}")
        resolved.append({**row, "path": path})
    return {"base_model": base_model, "experts": resolved}, errors


def write_run_manifest(trainer_dir: str, save_prefix: str, payload: dict) -> str:
    output_dir = os.path.join(os.path.abspath(trainer_dir), "mofe_manifests")
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, f"{save_prefix}.json")
    fd, temporary = tempfile.mkstemp(prefix=".mofe-", suffix=".json", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, final_path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return final_path
