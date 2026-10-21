"""Select unbiased language quotas from all reviewed L3 candidates, then shuffle.

A compact random-priority index avoids selecting only the first source shards.
One uniform random order per language selects complete documents up to its token
budget. Selected rows are then read sequentially and independently mixed with
the existing external shuffler. Raw/v1/supplement files are never overwritten.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data_builder import clean_ultradata_l3_cpt as clean
import numpy as np
import orjson

WORK = clean.WORK
DTYPE = np.dtype([("priority", "V16"), ("file", "u4"), ("offset", "u8"), ("tokens", "u4")])


def assembly():
    if clean.OUTPUT.exists() or clean.REPORT.exists():
        raise FileExistsError("Final artifact already exists")
    originals, supplements = [], []
    for path in sorted(WORK.glob("UltraData*.report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        (supplements if report.get("kind") == "bias_correction_supplement" else originals).append(report)
    if len(originals) != 74 or len(supplements) != 74 or not (WORK / "rescue_manifest.json").exists():
        raise ValueError("Wait for all 74 original and bias-correction shards")
    decisions_path = WORK / "identity_decisions.jsonl"
    decisions = {}
    with decisions_path.open("rb") as stream:
        for line in stream:
            row = orjson.loads(line)
            decisions[row["text_id"]] = row
    parts, buffers = {k: [] for k in clean.WEIGHTS}, {k: [] for k in clean.WEIGHTS}
    files, seen_text, seen_exercise = [], set(), set()
    available, rejected = Counter(), Counter()
    counts, keywords = Counter(), Counter()
    seed = struct.pack(">Q", clean.upstream.SEED)
    for report in originals + supplements:
        counts.update(report["counts"])
        keywords.update(report["vendor_keyword_hits"])
        for candidate, key, sha_key in ((False, "output", "output_sha256"), (True, "candidates", "candidates_sha256")):
            path = Path(report[key])
            file_number = len(files)
            files.append({"path": str(path), "candidate": candidate, "bytes": path.stat().st_size,
                          "mtime_ns": path.stat().st_mtime_ns})
            digest, offset = hashlib.sha256(), 0
            with path.open("rb") as stream:
                for line in stream:
                    digest.update(line)
                    payload = orjson.loads(line)
                    if candidate:
                        if not (payload["vendor_keywords"] or payload["identity_hits"]):
                            offset += len(line)
                            continue
                        row = payload["record"]
                        decision = decisions.get(row["text_id"])
                        if not decision or decision["decision"] != "keep":
                            key = ("manual_static_quality_drop" if decision and decision.get("scope") == "manual_static_quality_review"
                                   else "identity_review_drop" if decision else "identity_unreviewed_quarantine")
                            rejected[key] += 1
                            offset += len(line)
                            continue
                    else:
                        row = payload
                    text_id, exercise_id = bytes.fromhex(row["text_id"]), bytes.fromhex(row["exercise_id"])
                    if text_id in seen_text or exercise_id in seen_exercise:
                        rejected["duplicate_text_or_exercise"] += 1
                        offset += len(line)
                        continue
                    seen_text.add(text_id)
                    seen_exercise.add(exercise_id)
                    lang, token_count = row["mix_category"], row["token_count"]
                    priority = hashlib.blake2b(seed + text_id, digest_size=16).digest()
                    buffers[lang].append((priority, file_number, offset, token_count))
                    available[lang] += token_count
                    if len(buffers[lang]) >= 100_000:
                        parts[lang].append(np.array(buffers[lang], dtype=DTYPE))
                        buffers[lang].clear()
                    offset += len(line)
            if digest.hexdigest() != report[sha_key]:
                raise ValueError(f"Candidate artifact changed: {path}")
        print(f"[selection-index] {Path(report['source']).name}: {sum(available.values()):,} unique candidate tokens", flush=True)
    del seen_text, seen_exercise
    quotas = clean.quotas_for(available)
    stats = {lang: Counter() for lang in clean.WEIGHTS}
    selected = {i: [] for i in range(len(files))}
    per_file_tokens = Counter()
    for lang in clean.WEIGHTS:
        if buffers[lang]:
            parts[lang].append(np.array(buffers[lang], dtype=DTYPE))
            buffers[lang].clear()
        index = np.concatenate(parts[lang])
        parts[lang].clear()
        index.sort(order="priority")
        for entry in index:
            tokens = int(entry["tokens"])
            if stats[lang]["tokens"] + tokens > quotas[lang]:
                rejected[lang + ":language_quota"] += 1
                continue
            number = int(entry["file"])
            selected[number].append(int(entry["offset"]))
            per_file_tokens[number] += tokens
            stats[lang]["tokens"] += tokens
            stats[lang]["rows"] += 1
        if quotas[lang] - stats[lang]["tokens"] > 4096:
            raise ValueError(f"Unfilled quota for {lang}")
        print(f"[selection] {lang}: {stats[lang]['rows']:,} documents / {stats[lang]['tokens']:,} tokens", flush=True)
        del index
    shuffler = clean.ExternalRandomShuffler(clean.OUTPUT, clean.upstream.SEED + 1, 60_000,
                                            chunk_bytes=256 * 1024 ** 2, deduplicate=False)
    try:
        for number, info in enumerate(files):
            positions = sorted(selected.pop(number))
            if not positions:
                continue
            path = Path(info["path"])
            signature = path.stat()
            if (signature.st_size, signature.st_mtime_ns) != (info["bytes"], info["mtime_ns"]):
                raise ValueError("Indexed candidate file changed")
            # Sorted seeks let the OS read ahead; unlike random-priority order,
            # this never performs millions of unordered disk reads.
            with path.open("rb") as source:
                for position in positions:
                    source.seek(position)
                    line = source.readline()
                    if info["candidate"]:
                        line = orjson.dumps(orjson.loads(line)["record"])
                    shuffler.add_serialized(line)
            print(f"[shuffle-feed] {number+1}/{len(files)} files; {shuffler.rows_added:,} selected rows", flush=True)
        result = shuffler.finish(progress_label="CPT final shuffle")
    except BaseException:
        shuffler.temporary_dir.cleanup()
        raise
    mixing = clean.upstream.audit_contiguous_mixing(clean.OUTPUT, {k: {"quota": v} for k, v in quotas.items()})
    rescue = json.loads((WORK / "rescue_manifest.json").read_text(encoding="utf-8"))
    length_audit = json.loads((WORK / "length_distribution.json").read_text(encoding="utf-8"))
    report = {"name": "InstinctV1Moe UltraData-Code-L3 Coding CPT", "quality_policy": "l3-quality-v2-bias-correction",
              "source_repository": clean.upstream.REPO, "source_revision": clean.upstream.REVISION,
              "output": str(clean.OUTPUT), "output_sha256": result["sha256"], "rows": result["rows"],
              "tokens": sum(v["tokens"] for v in stats.values()), "target_tokens": clean.TARGET,
              "tokenizer_sha256": clean.upstream.sha256_file(clean.ROOT / "model/tokenizer.json"),
              "token_count_policy": "actual project tokenizer, add_special_tokens=False, plus BOS/EOS=2",
              "min_tokens": 128, "max_tokens": 4096, "rewritten_rows": 0, "text_truncated": False,
              "components": {k: dict(v) for k, v in stats.items()}, "language_quotas": quotas,
              "python_cpp_share": (stats["py"]["tokens"] + stats["cpp"]["tokens"]) / clean.TARGET,
              "filter_counts_after_bias_correction": dict(counts), "selection_rejections": dict(rejected),
              "length_filter_audit": length_audit,
              "bias_review": {"sample_report": str(WORK / "filter_review_samples.report.json"),
                              "sampled_rows": 42000, "recovered_records": rescue["counts"]["recovered_rows"],
                              "recovery_counts": rescue["counts"],
                              "selection_fix": "Uniform deterministic random-priority order over ALL unique reviewed candidates per language; not source-order quota filling"},
              "identity_clean": {"target_identity": "InstinctV1Moe", "vendor_keyword_hits": dict(keywords),
                                 "review_decisions": str(decisions_path), "review_decisions_sha256": clean.upstream.sha256_file(decisions_path),
                                 "unreviewed_mentions": "quarantined and excluded from final selection"},
              "prior_corpus_exclusion": json.loads((WORK / "prior.report.json").read_text(encoding="utf-8")),
              "benchmark_screening": json.loads((WORK / "benchmark_v2.report.json").read_text(encoding="utf-8")),
              "original_inputs": originals, "bias_correction_supplements": supplements,
              "random_selection": {"seed": clean.upstream.SEED, "priority": "BLAKE2b-128(seed||text_fingerprint)",
                                   "unique_candidate_tokens": dict(available),
                                   "selected_tokens_per_candidate_file": {files[k]["path"]: v for k, v in per_file_tokens.items()}},
              "shuffle": {**result, "seed": clean.upstream.SEED + 1, "mixing_audit": mixing},
              "limitations": ["Heuristic quality selection; generated solutions, analyses and tests are not execution-verified.",
                              "Python syntax is checked; other languages are not compiler-verified.",
                              "Remaining lexical benchmark signatures can still overmatch generic code and do not prove semantic decontamination.",
                              "Exact text/task+solution dedup does not establish semantic uniqueness.",
                              "Length filtering intentionally favors shorter records; global filtered fraction is below the user's 10% threshold.",
                              "Apache-2.0 release terms do not replace original seed-repository licenses."],
              "decision": "quality-filtered and bias-corrected CPT artifact; see audit scope and limitations"}
    clean.dump(clean.REPORT, report)
    clean.dump(WORK / "status.json", {"phase": "complete", "output": str(clean.OUTPUT), "report": str(clean.REPORT),
                                     "rows": result["rows"], "tokens": report["tokens"]})
    print(json.dumps({k: report[k] for k in ("output", "rows", "tokens", "python_cpp_share")}), flush=True)


if __name__ == "__main__":
    assembly()
