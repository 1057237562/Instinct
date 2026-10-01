"""Remove rows matching a topic profile from a chat/SFT corpus.

The profile lists unambiguous terms plus ambiguous ones. An ambiguous term (one
that also carries an unrelated everyday meaning, e.g. 同志 "comrade", 同性 in the
physics phrase 同性相斥, "trans" inside "transformer") removes a row only when a
second profile term appears in the same row, so ordinary text is not swept up.
The input is never modified; removed rows are written to a sidecar with the
matched terms so the decision can be reviewed or reverted.

Run from the repository root::

    python dataset/scripts/remove_topic_rows.py dataset/sft_t2t_mini.identity_repaired.jsonl \
        --profile lgbt --output dataset/sft_t2t_mini.identity_repaired.lgbt_removed.jsonl
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

from dataset.source_format import classify  # noqa: E402


def _ascii(pattern: str) -> str:
    # "\b" misbehaves next to CJK, so bound ASCII terms explicitly.
    return rf"(?<![a-z0-9]){pattern}(?![a-z0-9])"


PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "lgbt": {
        "strict": (
            "lgbt", "lgbtq", "lgbtqia", "gay", "lesbian", "bisexual", "bisexuality",
            "transgender", "transsexual", "homosexual", "homosexuality", "queer",
            "nonbinary", "intersex", "drag queen", "drag king", "same-sex",
            "gender identity", "sexual orientation", "coming out of the closet",
            "同性恋", "男同性恋", "女同性恋", "同性婚姻", "同性伴侣", "同性配偶",
            "双性恋", "跨性别", "变性人", "变性手术", "性别认同", "性取向", "性倾向",
            "出柜", "彩虹旗", "骄傲月", "骄傲游行", "同性结婚", "同婚",
        ),
        "ambiguous": (
            "trans", "pride", "同志", "同性", "百合", "拉拉",
        ),
        # Terms with a dominant everyday meaning (同志 "comrade", 百合 "lily",
        # 同性 in 同性相斥, "trans" inside unrelated words, "pride" in book
        # titles). They never remove a row on their own; a second profile term
        # must appear in the same row.
        "weak": (
            "同志", "同性", "百合", "拉拉", "trans", "pride",
        ),
    },
}

# ASCII terms get explicit boundaries: "\b" misbehaves next to CJK, and a bare
# "trans" would otherwise match "translate" or "transformer".
def _compile(terms: tuple[str, ...]) -> re.Pattern[str]:
    parts = [_ascii(re.escape(term)) if term.isascii() else term for term in terms]
    return re.compile("|".join(parts), re.IGNORECASE)


COMPILED = {
    name: {kind: _compile(terms) for kind, terms in kinds.items()}
    for name, kinds in PROFILES.items()
}

# Sentences that contain an ambiguous term but are plainly about something else.
FALSE_FRIEND_RE = re.compile(
    r"同性相斥|同性电荷|同种电荷|同性磁极|翻译|转录|转运|转让|传输|变电|变压器|转学|转账|"
    r"pride\s+and\s+prejudice",
    re.IGNORECASE,
)


def row_text(row: Any) -> str:
    chunks: list[str] = []
    if isinstance(row, dict):
        for value in row.values():
            chunks.append(_flatten(value))
    return "\n".join(chunk for chunk in chunks if chunk)


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(child) for child in value.values())
    if isinstance(value, list):
        return " ".join(_flatten(child) for child in value)
    return ""


def classify_row(row: Any, profile: str) -> list[str]:
    """Return the matched terms that make this row a removal candidate."""
    text = row_text(row)
    patterns = COMPILED[profile]
    strict_hits = {match.group(0).lower() for match in patterns["strict"].finditer(text)}
    ambiguous_hits = {match.group(0).lower() for match in patterns["ambiguous"].finditer(text)}
    if strict_hits:
        return sorted(strict_hits | ambiguous_hits)
    if len(ambiguous_hits) > 1:
        return sorted(ambiguous_hits)
    if not ambiguous_hits:
        return []
    weak = patterns["weak"] if "weak" in patterns else patterns["ambiguous"]
    for match in weak.finditer(text):
        window = text[max(0, match.start() - 40):match.end() + 40]
        if FALSE_FRIEND_RE.search(window):
            return []
    return []


def iter_records(path: Path) -> Iterator[tuple[int, str, Any]]:
    if classify(path) == "parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        row_number = 0
        for batch in parquet_file.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                row_number += 1
                yield row_number, "", row
        return
    opener = gzip.open if str(path).lower().endswith((".gz", ".gzip")) else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as source:
        for row_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield row_number, line, json.loads(line)
            except json.JSONDecodeError:
                yield row_number, line, None


def run(source: Path, output: Path, removed_path: Path, report_path: Path, profile: str, overwrite: bool) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    removed_path, report_path = removed_path.resolve(), report_path.resolve()
    for path in (output, removed_path, report_path):
        if path == source:
            raise ValueError("Output paths must not overwrite the input dataset.")
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    removed_path.parent.mkdir(parents=True, exist_ok=True)

    counts: collections.Counter[str] = collections.Counter()
    term_counts: collections.Counter[str] = collections.Counter()
    output_hash = hashlib.sha256()
    kept = removed = 0
    temp_output = output.with_name(output.name + ".tmp")
    temp_removed = removed_path.with_name(removed_path.name + ".tmp")
    try:
        with temp_output.open("wb") as out_stream, temp_removed.open("wb") as removed_stream:
            for row_number, raw_line, row in iter_records(source):
                counts["rows_read"] += 1
                matches = classify_row(row, profile) if row is not None else []
                if matches:
                    removed += 1
                    term_counts.update(matches)
                    record = {"_review": {"source": str(source), "row_number": row_number, "profile": profile, "matched_terms": matches}, "record": row}
                    removed_stream.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
                    continue
                kept += 1
                encoded = raw_line.encode("utf-8") if raw_line else (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                out_stream.write(encoded)
                output_hash.update(encoded)
        os.replace(temp_output, output)
        os.replace(temp_removed, removed_path)
    except Exception:
        temp_output.unlink(missing_ok=True)
        temp_removed.unlink(missing_ok=True)
        raise

    report = {
        "tool": "remove_topic_rows",
        "version": 1,
        "profile": profile,
        "source": {"path": str(source), "rows": counts["rows_read"]},
        "output": {"path": str(output), "rows": kept, "sha256": output_hash.hexdigest(),
                   "policy": "kept rows copied byte for byte"},
        "removed": {"path": str(removed_path), "rows": removed, "terms": dict(sorted(term_counts.items(), key=lambda item: -item[1]))},
        "decision": "The source dataset was not modified.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--removed", type=Path, help="Sidecar listing removed rows")
    parser.add_argument("--report", type=Path, help="Default: <output>.report.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    removed = args.removed or Path(str(args.output) + ".removed.jsonl")
    report = args.report or Path(str(args.output) + ".report.json")
    try:
        result = run(args.input, args.output, removed, report, args.profile, args.overwrite)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))
    print(json.dumps({k: result[k] for k in ("profile", "source", "output", "removed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
