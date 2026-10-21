"""Full identity audit/cleaning of pretrain_coder_12b; raw source is immutable.

Byte-range workers preserve original text. Ordinary factual/code mentions are
kept with evidence; explicit foreign-assistant persona statements are excluded
under a reviewed policy. Ambiguous cases remain in whole-row review files.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from scripts.data_builder import clean_ultradata_l3_cpt as common
from scripts.data_builder import review_ultradata_l3_identity  # bounded second pass
from scripts.data_builder.coder_identity_review import review_reasons
from scripts.data_builder.filter_anomaly_candidates import inspect_row
import orjson

SOURCE = ROOT / "dataset/pretrain_coder_12b.jsonl"
OUTPUT = ROOT / "dataset/pretrain_coder_12b_identity_clean.jsonl"
WORK = ROOT / "dataset/coder_replay_identity_review"
SOURCE_REPORT = ROOT / "dataset/pretrain_coder_12b.report.json"
STRONG_MODEL = re.compile(r"qwen|通义千问|千问|chat\s*gpt|openai|deepseek|minicpm|openbmb|modelbest|面壁智能", re.I)
ASSISTANT_GENRE = re.compile(r"\b(?:AI|assistant|language model|large language)\b|人工智能|语言模型|智能助手|我是(?:通义千问|Qwen|ChatGPT|DeepSeek)", re.I)


def process(task):
    number, start, end = task
    prefix = WORK / f"part_{number:03d}"
    output, candidates = Path(str(prefix) + ".clean.jsonl"), Path(str(prefix) + ".candidates.jsonl")
    decisions = Path(str(prefix) + ".decisions.jsonl")
    report_path = Path(str(prefix) + ".report.json")
    signature = SOURCE.stat()
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["source_signature"] != [signature.st_size, signature.st_mtime_ns]:
            raise ValueError("Raw corpus changed")
        return report
    counts, tokens, keywords = Counter(), Counter(), Counter()
    digest = hashlib.sha256()
    with SOURCE.open("rb") as source, output.open("wb") as writer, candidates.open("wb") as pending, decisions.open("wb") as ledger:
        if start:
            source.seek(start - 1)
            if source.read(1) != b"\n":
                source.readline()
        while source.tell() < end:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            counts["input_rows"] += 1
            row = orjson.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                counts["drop:invalid_text"] += 1
                continue
            if "\x00" in text or "\ufffd" in text:
                counts["drop:encoding_corruption"] += 1
                continue
            category = row.get("mix_category")
            if category not in ("code", "math", "natural"):
                raise ValueError("Unknown baseline category")
            row = {**row, "text_id": common.fp(text).hex(), "base_source_byte_offset": offset,
                   "base_source_row_within_partition": counts["input_rows"], "base_source_partition": number}
            # Mandatory triage precedes verdict/preservation policy.
            triage = inspect_row({"response": text}, "identity")
            verdict, hits, vendor = common.upstream.identity_scan(text)
            targeted = review_reasons(text, category) if vendor or hits else []
            keywords.update(vendor)
            decision, reason = "keep", "No foreign-model identity claim or targeted persona pattern."
            if vendor or hits or targeted or triage:
                algorithm_use = (set(vendor) <= {"minimax", "glm"} and bool(vendor) and
                                 re.search(r"algorithm|alpha.beta|game|spanning.tree|glm::|vec[234]|matrix|regression", text, re.I))
                if not targeted and (verdict == "clean" or verdict == "review" and algorithm_use):
                    reason = "Reviewed factual/code preservation policy: ordinary names, model discussion, minimax/GLM algorithms or code; no targeted assistant-persona claim."
                elif verdict == "claim" and STRONG_MODEL.search(text) and ASSISTANT_GENRE.search(text):
                    decision = "drop"
                    reason = "Reviewed persona exclusion policy: non-negated explicit foreign-assistant/model self-identification; preserve raw row and evidence, exclude from InstinctV1Moe training."
                else:
                    decision = "pending"
                    reason = "Ambiguous persona/model-card context requires a fingerprint-bound contextual decision."
                payload = {"record": row, "triage": triage, "identity_verdict": verdict,
                           "identity_hits": hits, "contextual_reasons": targeted,
                           "vendor_keywords": dict(vendor), "decision": decision, "reason": reason,
                           "source": str(SOURCE), "source_byte_offset": offset}
                pending.write(orjson.dumps(payload) + b"\n")
                ledger.write(orjson.dumps({"text_id": row["text_id"], "decision": decision, "reason": reason,
                                          "source_byte_offset": offset, "vendor_keywords": dict(vendor)}) + b"\n")
            if decision != "keep":
                counts["identity_" + decision] += 1
                continue
            serialized = orjson.dumps(row) + b"\n"
            writer.write(serialized)
            digest.update(serialized)
            counts["output_rows"] += 1
            tokens[category] += int(row["token_count"])
    report = {"number": number, "source_signature": [signature.st_size, signature.st_mtime_ns],
              "byte_range": [start, end], "output": str(output), "output_sha256": digest.hexdigest(),
              "candidates": str(candidates), "candidates_sha256": common.upstream.sha256_file(candidates),
              "decisions": str(decisions), "counts": dict(counts), "component_tokens": dict(tokens),
              "vendor_keyword_hits": dict(keywords)}
    common.dump(report_path, report)
    return report


def scan(workers):
    WORK.mkdir(parents=True, exist_ok=True)
    size = SOURCE.stat().st_size
    tasks = [(i, size * i // workers, size * (i + 1) // workers) for i in range(workers)]
    reports, counts, keywords = [], Counter(), Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(process, task) for task in tasks]):
            report = future.result()
            reports.append(report)
            counts.update(report["counts"])
            keywords.update(report["vendor_keyword_hits"])
            print(f"[baseline-clean] {len(reports)}/{workers} parts; {counts['input_rows']:,} source rows; {counts.get('identity_pending',0):,} pending", flush=True)
            common.dump(WORK / "status.json", {"phase": "scan", "counts": dict(counts), "completed_parts": len(reports)})
    reports.sort(key=lambda r: r["number"])
    original = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    if counts["input_rows"] != original["rows"]:
        raise ValueError("Byte-range partitioning does not reconcile with original row count")
    if common.upstream.sha256_file(SOURCE) != original["output_sha256"]:
        raise ValueError("Baseline corpus differs from original SHA-256 report")
    common.dump(WORK / "manifest.json", {"source": str(SOURCE), "source_sha256": original["output_sha256"],
                "source_rows": counts["input_rows"], "parts": reports, "counts": dict(counts),
                "vendor_keyword_hits": dict(keywords), "source_report": original})


def merge():
    manifest = json.loads((WORK / "manifest.json").read_text(encoding="utf-8"))
    manual_path = WORK / "manual_decisions.jsonl"
    manual = {}
    if manual_path.exists():
        for line in manual_path.open("rb"):
            row = orjson.loads(line)
            manual[row["text_id"]] = row
    pending_path = WORK / "pending_review.jsonl"
    unresolved = 0
    with pending_path.open("wb") as pending:
        for part in manifest["parts"]:
            with Path(part["candidates"]).open("rb") as source:
                for line in source:
                    row = orjson.loads(line)
                    if row["decision"] == "pending" and row["record"]["text_id"] not in manual:
                        pending.write(line)
                        unresolved += 1
    if unresolved:
        print(f"[baseline-clean] {unresolved} unresolved rows in {pending_path}", flush=True)
        return
    if OUTPUT.exists():
        raise FileExistsError("Cleaned baseline already exists")
    temporary = Path(str(OUTPUT) + ".tmp")
    counts, tokens, digest = Counter(), Counter(), hashlib.sha256()
    row_offset = 0
    def write(row, target):
        row["base_source_row"] = row_offset + row["base_source_row_within_partition"]
        line = orjson.dumps(row) + b"\n"
        target.write(line)
        digest.update(line)
        counts["output_rows"] += 1
        tokens[row["mix_category"]] += int(row["token_count"])
    with temporary.open("wb") as target:
        for part in manifest["parts"]:
            if common.upstream.sha256_file(Path(part["output"])) != part["output_sha256"]:
                raise ValueError("Intermediate baseline output changed")
            with Path(part["output"]).open("rb") as source:
                for line in source:
                    write(orjson.loads(line), target)
            with Path(part["candidates"]).open("rb") as source:
                for line in source:
                    case = orjson.loads(line)
                    if case["decision"] != "pending":
                        continue
                    decision = manual[case["record"]["text_id"]]
                    if decision["decision"] == "keep":
                        write(case["record"], target)
                    else:
                        counts["manual_drop"] += 1
            row_offset += part["counts"]["input_rows"]
            print(f"[baseline-merge] {counts['output_rows']:,} cleaned rows", flush=True)
    os.replace(temporary, OUTPUT)
    report = {"name": "pretrain_coder_12b identity-clean replay source", "source": str(SOURCE),
              "source_sha256": manifest["source_sha256"], "original_provenance_report": str(SOURCE_REPORT),
              "input_rows": manifest["source_rows"], "rows": counts["output_rows"],
              "dropped_rows": manifest["source_rows"] - counts["output_rows"], "rewritten_text_rows": 0,
              "output": str(OUTPUT), "output_sha256": digest.hexdigest(),
              "component_tokens": dict(tokens), "tokens": sum(tokens.values()),
              "identity_clean": {"target_identity": "InstinctV1Moe", "vendor_keyword_hits": manifest["vendor_keyword_hits"],
                                 "full_scan": True, "unresolved_candidates": 0,
                                 "manual_decisions": str(manual_path), "counts": manifest["counts"]},
              "provenance": manifest,
              "limitations": ["Source token counts are preserved; selected replay rows are independently retokenized before mixing.",
                              "Identity/schema audit does not certify every factual claim or code solution.",
                              "Ambiguous raw records are preserved separately; reviewed exclusions do not alter the raw corpus."]}
    common.dump(OUTPUT.with_suffix(".report.json"), report)
    common.dump(WORK / "status.json", {"phase": "clean_complete", "rows": counts["output_rows"], "output": str(OUTPUT)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("scan", "merge"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    scan(args.workers) if args.command == "scan" else merge()
