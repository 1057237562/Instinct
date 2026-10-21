"""Apply factual-code preservation policy to completed L3 candidate shards.

Unambiguous vendor mentions without persona cues are retained with a text-bound
reason. Claims and ambiguous self-reference remain quarantined for contextual
review in identity_manual_review.jsonl. This never edits training source text.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import orjson
from scripts.data_builder import build_ultradata_code_l3_12b as upstream
from scripts.data_builder.coder_identity_review import review_reasons

WORK = ROOT / "dataset/ultradata_code_l3_12b/quality_clean_v1"
# Contextual second pass: do not read "bing" inside "describing", or "Alexa"
# inside an animal named "AlexanderTheGreat", as a model/vendor mention.
upstream.VENDOR_RE = re.compile(upstream.VENDOR_COUNT_RE.pattern +
    r"|(?<![a-z0-9_])(?:qwen[a-z0-9.\-]*|minicpm[a-z0-9.\-]*)(?![a-z0-9_])", re.I)


def main():
    counts, keywords = Counter(), Counter()
    path = WORK / "identity_decisions.jsonl"
    manual = WORK / "identity_manual_review.jsonl"
    existing = {}
    if path.exists():
        for line in path.open("rb"):
            decision = orjson.loads(line)
            if decision.get("reviewer") == "contextual_agent_review":
                existing[decision["text_id"]] = decision
    temporary, manual_tmp = Path(str(path) + ".tmp"), Path(str(manual) + ".tmp")
    seen = set()
    with temporary.open("wb") as output, manual_tmp.open("wb") as pending:
        for report_path in sorted(WORK.glob("UltraData*.report.json")):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            with Path(report["candidates"]).open("rb") as source:
                for line in source:
                    candidate = orjson.loads(line)
                    if not (candidate["vendor_keywords"] or candidate["identity_hits"]):
                        continue
                    row = candidate["record"]
                    text_id = row["text_id"]
                    if text_id in seen:
                        continue
                    seen.add(text_id)
                    keywords.update(candidate["vendor_keywords"])
                    bounded_verdict, bounded_hits, bounded_keywords = upstream.identity_scan(row["text"])
                    persona = review_reasons(row["text"], "code")
                    algorithm_use = (set(bounded_keywords) <= {"minimax", "glm"}
                                     and bool(bounded_keywords)
                                     and re.search(r"\b(?:algorithm|alpha.beta|game|spanning.tree|glm::|vec[234]|matrix|regression)\b", row["text"], re.I))
                    if text_id in existing:
                        decision = existing[text_id]
                    elif not persona and (bounded_verdict == "clean" or (bounded_verdict == "review" and algorithm_use)):
                        decision = {"text_id": text_id, "decision": "keep",
                                    "reviewer": "reviewed_factual_code_policy_v1",
                                    "reason": "Contextually preserve complete code exercises: bounded vendor scan finds no identity claim/persona; algorithm-only minimax/GLM references and substring false positives (describing/probing/Alexander) are ordinary code/facts. Text unchanged.",
                                    "bounded_identity_verdict": bounded_verdict,
                                    "source_shard": row["source_shard"], "source_row": row["source_row"],
                                    "vendor_keywords": candidate["vendor_keywords"]}
                    else:
                        counts["requires_contextual_review"] += 1
                        pending.write(line)
                        continue
                    counts[decision["decision"]] += 1
                    output.write(orjson.dumps(decision) + b"\n")
    os.replace(temporary, path)
    os.replace(manual_tmp, manual)
    summary = {"counts": dict(counts), "vendor_keyword_hits": dict(keywords),
               "decisions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "manual_review_sha256": hashlib.sha256(manual.read_bytes()).hexdigest(),
               "policy": "Contextual factual-code preservation; ambiguous persona records remain quarantined"}
    (WORK / "identity_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
