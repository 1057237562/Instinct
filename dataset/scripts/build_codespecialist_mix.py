"""Build and validate a code-majority pretraining mixture without chunking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import orjson


MIX_PLAN = {
    "open_repository_code_and_docs": 750_000_000,
    "competitive_problem_reasoning": 450_000_000,
    "verified_competitive_submissions": 240_000_000,
    "code_instruction": 210_000_000,
    "text_to_sql": 90_000_000,
    "exercism_software_tasks": 60_000_000,
    "general_bilingual": 600_000_000,
    "math_reasoning": 300_000_000,
    "academic_technical": 300_000_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verified-submissions",
        type=Path,
        default=Path("dataset/codespecialist.jsonl"),
        help="Existing mix containing source=verified_competitive_submissions rows.",
    )
    parser.add_argument("--general", type=Path, default=Path("dataset/pretrain_t2t_mini.jsonl"))
    parser.add_argument("--stack", type=Path, default=Path("dataset/pretrain_stackv2_open_750mb.jsonl"))
    parser.add_argument("--code-instruction", type=Path, default=Path("dataset/pretrain_code_prompts_extra.jsonl"))
    parser.add_argument("--math", type=Path, default=Path("dataset/pretrain_math_reasoning_500mb.jsonl"))
    parser.add_argument("--arxiv", type=Path, default=Path("dataset/pretrain_arxiv_open_500mb.jsonl"))
    parser.add_argument("--competitive-dir", type=Path, default=Path("dataset/competitive-coding/data"))
    parser.add_argument("--output", type=Path, default=Path("dataset/codespecialist.next.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("dataset/codespecialist.next.report.json"))
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def render_messages(messages: object) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            return ""
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return ""
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def selected_by_hash(text_bytes: bytes, probability: float, seed: int) -> bool:
    if probability >= 1:
        return True
    seed_bytes = seed.to_bytes(8, "little", signed=False)
    value = int.from_bytes(
        hashlib.blake2b(text_bytes, digest_size=8, key=seed_bytes).digest(), "big"
    )
    return value < int(probability * (1 << 64))


def main() -> None:
    args = parse_args()
    paths = {
        "verified": args.verified_submissions.resolve(),
        "general": args.general.resolve(),
        "stack": args.stack.resolve(),
        "instruction": args.code_instruction.resolve(),
        "math": args.math.resolve(),
        "arxiv": args.arxiv.resolve(),
        "competitive": args.competitive_dir.resolve(),
        "output": args.output.resolve(),
        "report": args.report.resolve(),
    }
    for name in ("verified", "general", "stack", "instruction", "math", "arxiv"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    if not paths["competitive"].is_dir():
        raise FileNotFoundError(paths["competitive"])
    for name in ("output", "report"):
        if paths[name].exists():
            raise FileExistsError(paths[name])

    temporary = paths["output"].with_name(paths["output"].name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    seen_text: set[bytes] = set()
    component_stats: dict[str, Counter] = {
        name: Counter() for name in MIX_PLAN
    }
    unique_message_ids: dict[str, Counter[str]] = {}
    output_rows = 0
    output_bytes = 0

    def add(text: str, source: str, source_id: object = "", license_name: object = "", url: object = "") -> bool:
        nonlocal output_rows, output_bytes
        if not text:
            component_stats[source]["rejected_empty"] += 1
            return False
        raw_text = text.encode("utf-8")
        target = MIX_PLAN[source]
        if component_stats[source]["text_bytes"] + len(raw_text) > target:
            component_stats[source]["rejected_over_budget"] += 1
            return False
        text_hash = hashlib.blake2b(raw_text, digest_size=16).digest()
        if text_hash in seen_text:
            component_stats[source]["rejected_duplicate_text"] += 1
            return False
        record = {
            "text": text,
            "source": source,
            "source_id": str(source_id or ""),
            "license": str(license_name or ""),
            "url": str(url or ""),
        }
        line = orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
        destination.write(line)
        digest.update(line)
        seen_text.add(text_hash)
        output_rows += 1
        output_bytes += len(line)
        component_stats[source]["rows"] += 1
        component_stats[source]["text_bytes"] += len(raw_text)
        component_stats[source]["output_bytes"] += len(line)
        return True

    def full(source: str) -> bool:
        return component_stats[source]["text_bytes"] >= MIX_PLAN[source] - 1024

    def sample_text_jsonl(path: Path, source: str, probability: float, license_name: str) -> None:
        # First pass is content-hash sampling, which avoids selecting only the
        # beginning of domain-grouped files.  A complement pass fills tiny gaps.
        for complement in (False, True):
            with path.open("rb") as handle:
                for raw_line in handle:
                    row = orjson.loads(raw_line)
                    text = row.get("text") if isinstance(row, dict) else None
                    if not isinstance(text, str):
                        component_stats[source]["rejected_invalid"] += 1
                        continue
                    chosen = selected_by_hash(text.encode("utf-8"), probability, args.seed)
                    if chosen == complement:
                        continue
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    add(
                        text,
                        source,
                        row.get("source_id") or row.get("arxiv_id") or row.get("id"),
                        row.get("license") or metadata.get("license") or license_name,
                        row.get("url") or metadata.get("url"),
                    )
                    if full(source):
                        return

    def sample_message_jsonl(
        path: Path,
        source: str,
        target_share: int,
        unique_scope: str | None = None,
        question_cap: int = 0,
    ) -> None:
        seen_questions = unique_message_ids.setdefault(unique_scope, Counter()) if unique_scope else None
        local_target = target_share
        with path.open("rb") as handle:
            for raw_line in handle:
                row = orjson.loads(raw_line)
                question_id = str(row.get("question_id") or row.get("uuid") or "")
                if seen_questions is not None and question_id:
                    if seen_questions[question_id] >= question_cap:
                        component_stats[source]["rejected_duplicate_question"] += 1
                        continue
                    seen_questions[question_id] += 1
                text = render_messages(row.get("messages"))
                add(text, source, question_id, row.get("license"), "")
                if component_stats[source]["text_bytes"] >= local_target - 1024:
                    return

    try:
        with temporary.open("xb", buffering=8 * 1024 * 1024) as destination:
            # High-quality real repository code and developer documentation.
            sample_text_jsonl(paths["stack"], "open_repository_code_and_docs", 1.0, "open licenses")

            # Diverse question + reasoning + executable answer, balanced across
            # Python/C++ and two independent shards for each language.
            for language in ("python", "cpp"):
                for shard in ("00", "01"):
                    share = (225_000_000 * (int(shard) + 1)) // 2
                    sample_message_jsonl(
                        paths["competitive"] / f"competitive_programming_{language}_{shard}.jsonl",
                        "competitive_problem_reasoning",
                        (225_000_000 if language == "python" else 450_000_000) - 225_000_000 + share,
                        unique_scope=language,
                        question_cap=2,
                    )

            # Reuse the already selected verified-submission component from the
            # current mix, so obsolete multi-gigabyte backup mixes can be removed.
            with paths["verified"].open("rb") as handle:
                for raw_line in handle:
                    row = orjson.loads(raw_line)
                    if row.get("source") != "verified_competitive_submissions":
                        continue
                    text = row.get("text")
                    if not isinstance(text, str):
                        continue
                    add(
                        text,
                        "verified_competitive_submissions",
                        row.get("source_id"),
                        row.get("license") or "ODC-By",
                        row.get("url"),
                    )
                    if full("verified_competitive_submissions"):
                        break

            sample_text_jsonl(paths["instruction"], "code_instruction", 0.85, "mixed open licenses")
            sample_message_jsonl(paths["competitive"] / "text_to_sql.jsonl", "text_to_sql", 90_000_000)
            sample_message_jsonl(paths["competitive"] / "exercism.jsonl", "exercism_software_tasks", 60_000_000)
            sample_text_jsonl(paths["general"], "general_bilingual", 0.58, "mixed; see Instinct dataset docs")
            sample_text_jsonl(paths["math"], "math_reasoning", 0.90, "Apache-2.0")
            sample_text_jsonl(paths["arxiv"], "academic_technical", 0.70, "CC BY/CC BY-SA/Public Domain")

            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, paths["output"])
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    planned_total = sum(MIX_PLAN.values())
    report = {
        "name": "codespecialist",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(paths["output"]),
        "design": {
            "objective": "code-majority pretraining with retained general, mathematical, and technical knowledge",
            "planned_text_bytes": planned_total,
            "whole_records_only": True,
            "text_truncated": False,
            "text_split": False,
            "global_exact_text_deduplication": True,
        },
        "components": {
            name: {
                "target_text_bytes": target,
                "target_percent": target / planned_total * 100,
                **dict(component_stats[name]),
            }
            for name, target in MIX_PLAN.items()
        },
        "output_rows": output_rows,
        "output_bytes": output_bytes,
        "output_text_utf8_bytes": sum(s["text_bytes"] for s in component_stats.values()),
        "output_sha256": digest.hexdigest(),
        "source_files": {key: str(value) for key, value in paths.items() if key not in ("output", "report")},
    }
    paths["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
