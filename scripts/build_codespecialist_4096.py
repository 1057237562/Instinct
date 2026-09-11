"""Build a 1.6B-token code-majority corpus; reject every record over 4096 tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import orjson
from datasets import load_dataset
from transformers import AutoTokenizer

if __package__ in (None, ""):
    __package__ = "scripts"
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.build_codespecialist_mix import render_messages, selected_by_hash


MAX_TOKENS = 4096  # Includes the BOS/EOS added by PretrainDataset.
MIX_PLAN = {
    "open_repository_code_and_docs": 469_914_737,
    "competitive_problem_reasoning": 286_617_283,
    "verified_competitive_submissions": 107_467_980,
    "code_instruction": 64_000_000,
    "text_to_sql": 16_000_000,
    "exercism_software_tasks": 16_000_000,
    "general_bilingual": 320_028_801,
    "math_reasoning": 160_000_000,
    "academic_technical": 159_971_199,
}
CODE_COMPONENTS = tuple(list(MIX_PLAN)[:6])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=Path("dataset/codespecialist.jsonl"))
    parser.add_argument("--general", type=Path, default=Path("dataset/pretrain_t2t.jsonl"))
    parser.add_argument("--stack", type=Path, default=Path("dataset/pretrain_stackv2_code_480m_tokens_4096.jsonl"))
    parser.add_argument("--instruction", type=Path, default=Path("dataset/pretrain_code_prompts_extra.jsonl"))
    parser.add_argument("--math", type=Path, default=Path("dataset/pretrain_math_reasoning_500mb.jsonl"))
    parser.add_argument("--math-instruct", type=Path, default=Path("dataset/math-instruct/MathInstruct.json"))
    parser.add_argument("--abstracts", type=Path, default=Path("dataset/pretrain_arxiv_abstracts_160m_tokens_4096.jsonl"))
    parser.add_argument("--competitive-dir", type=Path, default=Path("dataset/competitive-coding/data"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--output", type=Path, default=Path("dataset/codespecialist_4096.next.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("dataset/codespecialist_4096.next.report.json"))
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {name: value.resolve() for name, value in {
        "current": args.current, "general": args.general, "stack": args.stack,
        "instruction": args.instruction, "math": args.math,
        "math_instruct": args.math_instruct, "abstracts": args.abstracts,
        "competitive": args.competitive_dir, "output": args.output, "report": args.report,
    }.items()}
    for name in ("current", "general", "stack", "instruction", "math", "math_instruct", "abstracts"):
        if not paths[name].is_file(): raise FileNotFoundError(paths[name])
    if not paths["competitive"].is_dir(): raise FileNotFoundError(paths["competitive"])
    for name in ("output", "report"):
        if paths[name].exists(): raise FileExistsError(paths[name])
    temporary = paths["output"].with_name(paths["output"].name + ".tmp")
    if temporary.exists(): raise FileExistsError(temporary)

    backend = AutoTokenizer.from_pretrained(args.tokenizer_path.resolve(), trust_remote_code=True).backend_tokenizer
    digest = hashlib.sha256(); seen_text: set[bytes] = set(); stats = {name: Counter() for name in MIX_PLAN}
    question_counts: dict[str, Counter[str]] = defaultdict(Counter)
    output_rows = output_bytes = 0

    def full(source: str) -> bool:
        return stats[source]["tokens"] >= MIX_PLAN[source] - MAX_TOKENS

    def add(text: str, source: str, source_id="", license_name="", url="", known_tokens=None) -> bool:
        nonlocal output_rows, output_bytes
        if not text:
            stats[source]["rejected_empty"] += 1; return False
        raw = text.encode("utf-8"); text_hash = hashlib.blake2b(raw, digest_size=16).digest()
        if text_hash in seen_text:
            stats[source]["rejected_duplicate_text"] += 1; return False
        token_count = int(known_tokens) if known_tokens is not None else len(backend.encode(text, add_special_tokens=False).ids) + 2
        if token_count > MAX_TOKENS:
            stats[source]["rejected_over_4096"] += 1; return False
        if stats[source]["tokens"] + token_count > MIX_PLAN[source]:
            stats[source]["rejected_over_budget"] += 1; return False
        record = {"text": text, "source": source, "source_id": str(source_id or ""), "license": str(license_name or ""), "url": str(url or ""), "token_count": token_count}
        line = orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
        destination.write(line); digest.update(line); seen_text.add(text_hash)
        stats[source]["rows"] += 1; stats[source]["tokens"] += token_count; stats[source]["text_bytes"] += len(raw); stats[source]["output_bytes"] += len(line)
        output_rows += 1; output_bytes += len(line); return True

    def direct(path: Path, source: str, probability=1.0, required_source=None, default_license="") -> None:
        for complement in (False, True):
            with path.open("rb") as handle:
                for raw_line in handle:
                    row = orjson.loads(raw_line)
                    if required_source and row.get("source") != required_source: continue
                    text = row.get("text")
                    if not isinstance(text, str): stats[source]["rejected_invalid"] += 1; continue
                    chosen = selected_by_hash(text.encode("utf-8"), probability, args.seed)
                    if chosen == complement: continue
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    known = row.get("token_count")
                    add(text, source, row.get("source_id") or row.get("arxiv_id") or row.get("id"), row.get("license") or metadata.get("license") or default_license, row.get("url") or metadata.get("url"), known)
                    if full(source): return

    def message_file(path: Path, source: str, cumulative_target: int, scope=None, cap=0) -> None:
        counts = question_counts[scope] if scope else None
        with path.open("rb") as handle:
            for raw_line in handle:
                row = orjson.loads(raw_line); source_id = str(row.get("question_id") or row.get("uuid") or "")
                if counts is not None and source_id:
                    if counts[source_id] >= cap:
                        stats[source]["rejected_question_cap"] += 1; continue
                added = add(render_messages(row.get("messages")), source, source_id, row.get("license"))
                if added and counts is not None and source_id:
                    counts[source_id] += 1
                if stats[source]["tokens"] >= cumulative_target - MAX_TOKENS: return

    try:
        with temporary.open("xb", buffering=8 * 1024 * 1024) as destination:
            direct(paths["stack"], "open_repository_code_and_docs")
            print("stack", stats["open_repository_code_and_docs"]["tokens"], flush=True)

            for language in ("python", "cpp"):
                base = 0 if language == "python" else 144_000_000
                for shard_index, shard in enumerate(("00", "01"), start=1):
                    message_file(paths["competitive"] / f"competitive_programming_{language}_{shard}.jsonl", "competitive_problem_reasoning", base + 144_000_000 * shard_index // 2, scope=language, cap=4)
            print("competitive", stats["competitive_problem_reasoning"]["tokens"], flush=True)

            direct(paths["current"], "verified_competitive_submissions", required_source="verified_competitive_submissions", default_license="ODC-By")
            direct(paths["instruction"], "code_instruction", default_license="mixed open licenses")
            message_file(paths["competitive"] / "text_to_sql.jsonl", "text_to_sql", MIX_PLAN["text_to_sql"])
            message_file(paths["competitive"] / "exercism.jsonl", "exercism_software_tasks", MIX_PLAN["exercism_software_tasks"])
            print("other code", sum(stats[s]["tokens"] for s in CODE_COMPONENTS[2:]), flush=True)

            direct(paths["general"], "general_bilingual", probability=0.42, default_license="mixed; see Instinct dataset docs")
            print("general", stats["general_bilingual"]["tokens"], flush=True)
            direct(paths["math"], "math_reasoning", default_license="Apache-2.0")
            if not full("math_reasoning"):
                dataset = load_dataset("json", data_files=str(paths["math_instruct"]), split="train")
                for row in dataset:
                    text = render_messages([{"role": "user", "content": row["instruction"]}, {"role": "assistant", "content": row["output"]}])
                    add(text, "math_reasoning", row.get("source"), "Apache-2.0")
                    if full("math_reasoning"): break
            print("math", stats["math_reasoning"]["tokens"], flush=True)
            direct(paths["abstracts"], "academic_technical", default_license="CC0")
            print("academic", stats["academic_technical"]["tokens"], flush=True)

            short = {name: MIX_PLAN[name] - stats[name]["tokens"] for name in MIX_PLAN if not full(name)}
            if short: raise RuntimeError(f"Components below token budget: {short}")
            destination.flush(); os.fsync(destination.fileno())
        os.replace(temporary, paths["output"])
    except BaseException:
        if temporary.exists(): temporary.unlink()
        raise

    total_tokens = sum(value["tokens"] for value in stats.values())
    code_tokens = sum(stats[name]["tokens"] for name in CODE_COMPONENTS)
    report = {
        "name": "codespecialist_4096", "created_at": datetime.now(timezone.utc).isoformat(),
        "model_parameters": 152_406_528, "epochs_planned": 2,
        "max_tokens_including_bos_eos": MAX_TOKENS, "overlength_policy": "discard whole record",
        "text_split": False, "text_truncated": False, "global_exact_text_deduplication": True,
        "target_unique_tokens": sum(MIX_PLAN.values()), "output_unique_tokens": total_tokens,
        "planned_training_tokens": total_tokens * 2, "training_tokens_per_parameter": total_tokens * 2 / 152_406_528,
        "code_token_percent": code_tokens / total_tokens * 100,
        "components": {name: {"target_tokens": MIX_PLAN[name], "target_percent": MIX_PLAN[name] / sum(MIX_PLAN.values()) * 100, **dict(stats[name])} for name in MIX_PLAN},
        "output": str(paths["output"]), "output_rows": output_rows, "output_bytes": output_bytes,
        "output_sha256": digest.hexdigest(), "source_files": {name: str(path) for name, path in paths.items() if name not in ("output", "report")},
    }
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
