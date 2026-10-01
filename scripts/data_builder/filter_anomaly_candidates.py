"""Stream datasets and emit suspicious records for AI or human review.

Run from the repository root, for example:

    python scripts/data_builder/filter_anomaly_candidates.py dataset/sft_t2t_mini.jsonl \
        --output dataset/sft_t2t_mini.review_candidates.jsonl

The tool never edits its input. It emits whole source rows with row-level
provenance and heuristic reasons; a reviewer makes the final keep/drop/edit
decision. JSONL, compressed JSONL, Parquet, and directories of shards are
supported through ``scripts.data_loader.source_format``.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_loader.source_format import classify, source_files  # noqa: E402


IDENTITY_TERMS = re.compile(
    r"qwen|通义千问|千问|通义实验室|通义万相|通义听悟|alibaba(?:\s*cloud|\s*group)?|"
    r"阿里云|阿里巴巴",
    re.IGNORECASE,
)
SELF_REFERENCE = re.compile(
    r"我是|我的模型|我由|我来自|我基于|我属于|我的开发|我的创建者|"
    r"开发团队是|开发者是|由.{0,24}(开发|训练|推出|发布)|"
    r"i\s+am|i'm|i\s*,\s*(?:qwen|tongyi|通义千问)|my\s+(model|creator|developer|team)|"
    r"developed\s+by|created\s+by|trained\s+by|based\s+on",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(
    r"\[(?:insert|placeholder|todo|your\s+[^\]]+)\]|"
    r"\b(?:TODO|TBD|FIXME)\b|<\s*(?:question|answer|instruction)\s*>",
    re.IGNORECASE,
)
INCOMPLETE_END = re.compile(r"(?:[:：]|[,，]|[-—])\s*$")
IDENTITY_QUESTION = re.compile(
    r"你是谁|你的(?:真实)?(?:身份|模型|来源|开发者|创建者|开发团队)|"
    r"你背后的模型|你属于(?:哪个|什么)模型|你是(?:哪款|什么|哪个|通义|qwen|阿里云|阿里巴巴)|"
    r"你是由谁(?:开发|创建|训练|设计)|自我介绍|"
    r"who are you|what model are you|your (?:model|identity|creator|developer)|"
    r"who (?:created|developed) you",
    re.IGNORECASE,
)


def _message_texts(value: Any) -> Iterator[tuple[str, str]]:
    """Yield (role, text) pairs from common chat and preference schemas."""
    if isinstance(value, dict):
        role = str(value.get("role", value.get("from", ""))).strip().lower()
        text_value = value.get("content", value.get("value", value.get("text")))
        if isinstance(text_value, str):
            aliases = {"human": "user", "gpt": "assistant", "bot": "assistant"}
            yield aliases.get(role, role), text_value
        for key, child in value.items():
            if key not in {"content", "value", "text"}:
                yield from _message_texts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _message_texts(child)


def _row_texts(row: Any) -> list[tuple[str, str]]:
    if not isinstance(row, dict):
        return [("", str(row))]
    pairs: list[tuple[str, str]] = []
    for key in ("conversations", "messages", "chosen", "rejected", "prompt", "question", "instruction", "text", "response", "answer", "solution", "output", "completion"):
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, str):
            role = "user" if key in {"prompt", "question", "instruction"} else "assistant" if key in {"response", "answer", "solution", "output", "completion"} else ""
            pairs.append((role, value))
        else:
            pairs.extend(_message_texts(value))
    return pairs


def inspect_row(row: Any, profile: str) -> list[dict[str, str]]:
    pairs = _row_texts(row)
    reasons: list[dict[str, str]] = []
    texts_by_role: dict[str, list[str]] = collections.defaultdict(list)
    for role, text in pairs:
        texts_by_role[role].append(text)

    if profile in {"all", "identity"}:
        assistant_messages = [text for role, text in pairs if role == "assistant"]
        direct_claim = any(IDENTITY_TERMS.search(text) and SELF_REFERENCE.search(text) for text in assistant_messages)
        if direct_claim:
            reasons.append({
                "rule": "possible_foreign_model_identity",
                "detail": "Assistant text mentions another model/provider and uses self-identifying language.",
            })
        pending_identity_question = False
        paired_identity_claim = False
        for role, text in pairs:
            if role == "user":
                pending_identity_question = bool(IDENTITY_QUESTION.search(text))
            elif role == "assistant":
                if pending_identity_question and IDENTITY_TERMS.search(text):
                    paired_identity_claim = True
                pending_identity_question = False
        if paired_identity_claim and not direct_claim:
            reasons.append({
                "rule": "identity_question_mentions_foreign_model",
                "detail": "Identity-related question has an answer mentioning a foreign model/provider; verify the answer does not assign that identity to Instinct.",
            })

    if profile in {"all", "questions"}:
        users = [text.strip() for text in texts_by_role.get("user", [])]
        assistants = [text.strip() for text in texts_by_role.get("assistant", [])]
        is_chat = isinstance(row, dict) and any(key in row for key in ("conversations", "messages", "chosen", "rejected"))
        has_question_field = isinstance(row, dict) and any(key in row for key in ("prompt", "question", "instruction"))
        if is_chat:
            if not users:
                reasons.append({"rule": "missing_user_message", "detail": "No user/question text found in this row."})
            if not assistants:
                reasons.append({"rule": "missing_assistant_message", "detail": "No assistant target found in this row."})
            if any(not text for text in users):
                reasons.append({"rule": "empty_user_message", "detail": "At least one user/question message is empty."})
            if any(not text for text in assistants):
                reasons.append({"rule": "empty_assistant_message", "detail": "At least one assistant target is empty."})
        elif has_question_field:
            question_texts = [str(row.get(key, "")).strip() for key in ("prompt", "question", "instruction") if key in row]
            if any(not text for text in question_texts):
                reasons.append({"rule": "empty_question", "detail": "A prompt/question/instruction field is empty."})
        if any(len(text) > 20000 for text in users):
            reasons.append({"rule": "oversized_question", "detail": "A user/question message exceeds 20,000 characters."})
        if users and len(users[-1]) >= 8 and INCOMPLETE_END.search(users[-1]):
            reasons.append({"rule": "possibly_incomplete_question", "detail": "A question ends with a likely unfinished delimiter."})
        if any(PLACEHOLDER.search(text) for _, text in pairs):
            reasons.append({"rule": "placeholder_or_template", "detail": "A message contains a TODO or unresolved template placeholder."})
        normalized_users = [re.sub(r"\s+", " ", text).casefold() for text in users if text]
        if len(normalized_users) != len(set(normalized_users)):
            reasons.append({"rule": "duplicate_user_turn", "detail": "The same user question appears more than once in the conversation."})
        if isinstance(row, dict) and any(key in row for key in ("conversations", "messages")):
            messages = row.get("conversations", row.get("messages"))
            if not isinstance(messages, list) or not messages:
                reasons.append({"rule": "invalid_conversation_structure", "detail": "Conversation field is not a non-empty list."})

    return reasons


def iter_file_rows(path: Path) -> Iterator[tuple[int, Any, str | None, str | None]]:
    """Yield 1-based source row numbers; malformed JSON rows are yielded raw."""
    if classify(path) == "parquet":
        import pyarrow.parquet as pq

        row_number = 0
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                row_number += 1
                yield row_number, row, None, None
        return

    opener = gzip.open if str(path).lower().endswith((".gz", ".gzip")) else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as source:
        for row_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield row_number, json.loads(line), None, None
            except json.JSONDecodeError as error:
                yield row_number, None, f"Invalid JSON: {error.msg} at column {error.colno}", line.rstrip("\r\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(inputs: list[Path], output: Path, report_path: Path, profile: str, max_candidates: int | None, overwrite: bool) -> dict[str, Any]:
    resolved_inputs: list[Path] = []
    for input_path in inputs:
        resolved_inputs.extend(Path(item) for item in source_files(input_path))
    resolved_inputs = list(dict.fromkeys(item.resolve() for item in resolved_inputs))
    output = output.resolve()
    report_path = report_path.resolve()
    if output == report_path:
        raise ValueError("Candidate output and report must use different paths.")
    if output in resolved_inputs or report_path in resolved_inputs:
        raise ValueError("Output and report paths must not overwrite an input dataset.")
    if not overwrite and (output.exists() or report_path.exists()):
        raise FileExistsError("Output/report exists; choose new paths or pass --overwrite.")

    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(output.name + ".tmp")
    totals = collections.Counter()
    reasons_count: collections.Counter[str] = collections.Counter()
    source_reports = []
    candidate_count = 0
    output_hash = hashlib.sha256()
    try:
        with temp_output.open("wb") as destination:
            for source_path in resolved_inputs:
                source_count = 0
                source_candidates = 0
                for row_number, row, parse_error, raw_line in iter_file_rows(source_path):
                    source_count += 1
                    totals["rows_scanned"] += 1
                    if parse_error:
                        reasons = [{"rule": "invalid_json", "detail": parse_error}]
                        original: Any = {"raw_line": raw_line}
                    else:
                        reasons = inspect_row(row, profile)
                        original = row
                    if not reasons:
                        continue
                    if max_candidates is not None and candidate_count >= max_candidates:
                        totals["candidate_limit_reached"] = 1
                        continue
                    candidate = {
                        "_review": {
                            "source": str(source_path),
                            "row_number": row_number,
                            "rules": [item["rule"] for item in reasons],
                            "reasons": reasons,
                            "instruction": "Review this record for training quality and identity contamination. Decide keep, drop, or edit, and explain briefly. Preserve factual discussion of other models when it does not falsely identify the assistant.",
                        },
                        "record": original,
                    }
                    encoded = (json.dumps(candidate, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    destination.write(encoded)
                    output_hash.update(encoded)
                    candidate_count += 1
                    source_candidates += 1
                    totals["candidates"] += 1
                    reasons_count.update(item["rule"] for item in reasons)
                source_reports.append({
                    "path": str(source_path),
                    "rows_scanned": source_count,
                    "candidates": source_candidates,
                    "sha256": sha256_file(source_path),
                })
        os.replace(temp_output, output)
    except Exception:
        temp_output.unlink(missing_ok=True)
        raise

    report = {
        "tool": "filter_anomaly_candidates",
        "version": 1,
        "profile": profile,
        "input_files": source_reports,
        "rows_scanned": totals["rows_scanned"],
        "candidates_written": candidate_count,
        "candidate_limit_reached": bool(totals["candidate_limit_reached"]),
        "candidate_counts_by_rule": dict(sorted(reasons_count.items())),
        "output": str(output),
        "output_sha256": output_hash.hexdigest(),
        "decision": "Candidates are for review; the source data was not modified.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL/.jsonl.gz/.parquet file or shard directory")
    parser.add_argument("--output", required=True, type=Path, help="Candidate JSONL output path")
    parser.add_argument("--report", type=Path, help="Audit report path (default: <output>.report.json)")
    parser.add_argument("--profile", choices=("all", "identity", "questions"), default="all")
    parser.add_argument("--max-candidates", type=int, help="Optional cap; remaining rows are still scanned")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output and report files")
    args = parser.parse_args()
    if args.max_candidates is not None and args.max_candidates < 0:
        parser.error("--max-candidates must be non-negative")
    report_path = args.report or Path(str(args.output) + ".report.json")
    try:
        report = run(args.inputs, args.output, report_path, args.profile, args.max_candidates, args.overwrite)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
