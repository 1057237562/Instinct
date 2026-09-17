"""Independently recompute every persisted instruction-understanding label."""
from collections import Counter
import ast
import csv
import gzip
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "instruction_understanding_sft"


def expected(meta):
    family, op, p = meta["family"], meta["operation"], meta["parameters"]
    if family == "sequence_contract":
        xs, limit = p["values"], p["limit"]
        counts = {x: xs.count(x) for x in xs}
        if op == "first_occurrence":
            return [x for i, x in enumerate(xs) if x not in xs[:i]]
        if op == "occurs_once": return [x for x in xs if counts[x] == 1]
        if op == "distinct_sorted": return sorted(counts)
        return [x for x in xs if x % 2 == 0 and x > limit]
    if family in ("entity_binding", "condition_composition", "output_protocol"):
        rs = p["records"]
        if op == "ready_names": return [r["name"] for r in rs if r["status"] == "ready"]
        if op == "color_ids": return [r["id"] for r in rs if r["color"] == p["target_color"]]
        if op == "target_projection":
            found = [r for r in rs if r["id"] == p["target_id"]][0]
            return {"name": found["name"], "score": found["score"]}
        if op == "urgent_count": return sum(1 for r in rs if "urgent" in r["tags"])
        if op == "ready": return [r["id"] for r in rs if r["status"] == "ready"]
        if op == "ready_score_strict": return [r["id"] for r in rs if r["status"] == "ready" and r["score"] > p["threshold"]]
        if op == "composed":
            return [r["id"] for r in rs if r["status"] != "archived" and
                    ((r["status"] == "ready" and r["score"] >= p["threshold"]) or "urgent" in r["tags"])]
        ready = [r for r in rs if r["status"] == "ready"]
        if op == "json_objects": return [{"id": r["id"], "score": r["score"]} for r in ready]
        if op == "numeric_count": return len(ready)
        if op == "id_lines": return "\n".join(r["id"] for r in ready)
        if op == "csv_with_header":
            out = io.StringIO(newline=""); writer = csv.writer(out, lineterminator="\n")
            writer.writerow(["id", "score"]); writer.writerows((r["id"], r["score"]) for r in ready)
            return out.getvalue().rstrip("\n")
    if family == "content_instruction_boundary":
        doc = p["document"]
        if op == "extract_metadata": return {"title": doc["title"], "owner": doc["owner"]}
        if op == "count_quoted_chars": return len(doc["quoted_text"])
        return {"type": "embedded_instruction", "executed": False}
    if family == "missing_vs_empty":
        if op == "missing_field": return {"status": "missing", "field": p["missing_field"]}
        if op == "valid_empty": return []
        return {"status": "conflict", "reason": "same status required and excluded"}
    if family == "multi_turn_revision":
        rs, threshold = p["records"], p["threshold"]
        return ([r["id"] for r in rs if r["score"] >= threshold],
                [r["name"] for r in rs if r["score"] > threshold])
    if family == "code_contract": return None
    raise AssertionError((family, op))


def test_code(meta, source):
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert tree.body[0].name == meta["parameters"]["function_name"]
    namespace = {}
    # The generated code needs only this explicitly allowed pure builtin.
    exec(compile(tree, "<audited_sft>", "exec"), {"__builtins__": {"abs": abs}}, namespace)
    fn = namespace[tree.body[0].name]
    op = meta["operation"]
    cases = {
        "keep_first": [([], []), ([2, 1, 2, 3, 1], [2, 1, 3]), ([0, 0], [0])],
        "keep_singletons": [([], []), ([2, 1, 2, 3, 1], [3]), ([0, -1, 0], [-1])],
        "last_digit_product": [((12, 34), 8), ((-27, 15), 35), ((0, -99), 0)],
        "prefix_max": [([], []), ([-3, -5, -1], [-3, -3, -1]), ([2, 2, 1], [2, 2, 2])],
        "strict_even_filter": [(([2, 4, 4, 7, 8], 4), [8]), (([-4, -2, 0], -2), [0]), (([], 0), [])],
    }[op]
    for args, want in cases:
        call_args = args if isinstance(args, tuple) else (args,)
        snapshots = json.loads(json.dumps(call_args))
        got = fn(*call_args)
        assert got == want, (op, call_args, got, want)
        assert json.loads(json.dumps(call_args)) == snapshots


def compare(meta, answers):
    if meta["family"] == "code_contract":
        assert len(answers) == 1
        test_code(meta, answers[0]); return
    want = expected(meta)
    if meta["answer_format"] == "multi_json":
        assert tuple(json.loads(x) for x in answers) == want; return
    assert len(answers) == 1
    actual = answers[0]
    if meta["answer_format"] == "json": actual = json.loads(actual)
    elif meta["answer_format"] == "integer": actual = int(actual)
    assert actual == want, (meta["family"], meta["operation"], actual, want)


def main():
    metadata = []
    with gzip.open(DATA / "provenance.jsonl.gz", "rt", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
    by_file = {}
    for m in metadata: by_file.setdefault(m["file_split"], {})[m["row"]] = m
    errors = []
    checked = Counter()
    hashes = {"train": set(), "validation": set(), "pilot_train": set()}
    group_variants = {}
    for split in ("train", "validation", "pilot_train"):
        with (DATA / f"{split}.jsonl").open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == len(by_file[split])
        for index, persisted in enumerate(rows):
            meta = by_file[split][index]
            answers = [m["content"] for m in persisted["conversations"] if m["role"] == "assistant"]
            assert answers == meta["answers"]
            try: compare(meta, answers)
            except Exception as exc: errors.append({"split": split, "row": index, "error": repr(exc)})
            checked[meta["family"]] += 1
            hashes[split].add(meta["semantic_hash"])
            group_variants.setdefault((split, meta["group_id"]), set()).add((meta["operation"], meta["language"]))
    assert not hashes["train"] & hashes["validation"]
    assert hashes["pilot_train"] <= hashes["train"]
    # Every pilot/train/validation semantic group must contain all of its contrast and language variants.
    expected_sizes = {"sequence_contract": 8, "entity_binding": 8, "condition_composition": 6,
                      "output_protocol": 8, "content_instruction_boundary": 6,
                      "multi_turn_revision": 2, "missing_vs_empty": 6, "code_contract": 10}
    incomplete = []
    for (split, gid), variants in group_variants.items():
        family = gid.split("/", 1)[0]
        if len(variants) != expected_sizes[family]: incomplete.append({"split": split, "group": gid, "size": len(variants)})
    report = {
        "method": "Independent full recomputation from persisted parameters; the builder's label functions are not imported.",
        "persisted_records_checked": sum(checked.values()),
        "source_records_checked": len(by_file["train"]) + len(by_file["validation"]),
        "by_family_including_pilot": dict(checked),
        "answer_or_execution_errors": len(errors),
        "first_errors": errors[:20],
        "incomplete_contrast_groups": len(incomplete),
        "first_incomplete_groups": incomplete[:20],
        "train_validation_semantic_intersection": len(hashes["train"] & hashes["validation"]),
        "pilot_hashes_outside_train": len(hashes["pilot_train"] - hashes["train"]),
        "scope": "Verifies exact labels and executable code inside the generated task domains; not a claim of general instruction understanding.",
    }
    (DATA / "full_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors or incomplete: raise SystemExit(1)


if __name__ == "__main__": main()
