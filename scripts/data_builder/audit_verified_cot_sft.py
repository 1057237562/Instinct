"""Independently audit every persisted verified-CoT record."""
from fractions import Fraction
import ast
from collections import Counter
import gzip
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset" / "verified_cot_sft"


def formula(kind, p):
    if kind == "inventory_flow": return p["start"] - p["sold"] - p["damaged"] + p["restock"]
    if kind == "entity_chain":
        b = p["a"] + p["more"]; return p["a"] + b + (b - p["fewer"])
    if kind == "discount_shipping": return p["price"] * p["count"] * (100 - p["discount"]) // 100 + p["shipping"]
    if kind == "unit_rate_batches": return (p["per_box"] * p["boxes"] + p["extra"]) * p["groups"]
    if kind == "average_speed": return (p["d1"] + p["d2"]) // (p["t1"] + p["t2"])
    if kind == "time_budget": return (p["hours"] * 60 + p["remainder"] - p["breaks"] * p["break_minutes"]) // p["task_minutes"]
    if kind == "ticket_revenue": return p["adult_n"] * p["adult_price"] + p["child_n"] * p["child_price"] - p["cost"]
    if kind == "rectangle_tiles": return p["length"] * p["width"] // (p["tile"] ** 2) - p["excluded"]
    if kind == "ratio_split": return p["total"] // (p["left"] + p["right"]) * p["left"]
    if kind == "percent_then_change": return p["base"] * p["percent"] // 100 + p["added"]
    raise AssertionError(kind)


def arithmetic(node):
    if isinstance(node, ast.Expression): return arithmetic(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float): return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = arithmetic(node.operand); return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        a, b = arithmetic(node.left), arithmetic(node.right)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Div): return a / b
    raise AssertionError(ast.dump(node))


def audit_answer(answer, wanted):
    equations = re.findall(r"`([^`]+)`", answer)
    assert len(equations) in (2, 3)
    for equation in equations:
        pieces = equation.split("=")
        assert len(pieces) >= 2
        values = [arithmetic(ast.parse(piece.strip(), mode="eval")) for piece in pieces]
        assert all(x == values[0] for x in values[1:])
    assert answer.splitlines()[-1] == f"#### {wanted}"
    assert str(wanted) in answer.splitlines()[-2]


def main():
    by_file = {}
    with gzip.open(DATA / "provenance.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            meta = json.loads(line); by_file.setdefault(meta["file_split"], {})[meta["row"]] = meta
    errors, counts = [], Counter()
    hashes = {"train": set(), "validation": set()}
    prompts = set()
    for split in ("train", "validation"):
        with (DATA / f"{split}.jsonl").open("r", encoding="utf-8") as f:
            persisted = [json.loads(line) for line in f]
        assert len(persisted) == len(by_file[split])
        for index, row in enumerate(persisted):
            meta = by_file[split][index]
            try:
                assert row["conversations"][0]["content"] == meta["question"]
                assert row["conversations"][1]["content"] == meta["answer"]
                wanted = formula(meta["family"], meta["parameters"])
                assert wanted == meta["final"]
                audit_answer(meta["answer"], wanted)
                # Every numeric parameter must be rendered into the question.
                for value in meta["parameters"].values(): assert str(value) in meta["question"]
            except Exception as exc:
                errors.append({"split": split, "row": index, "family": meta["family"], "error": repr(exc)})
            counts[meta["family"]] += 1; hashes[split].add(meta["semantic_hash"])
            prompts.add(meta["question"])
    report = {
        "method": "Independent full formula recomputation and restricted-AST checking of every displayed equation; builder not imported.",
        "records_checked": sum(counts.values()), "by_family": dict(counts),
        "answer_equation_or_prompt_errors": len(errors), "first_errors": errors[:20],
        "train_validation_semantic_intersection": len(hashes["train"] & hashes["validation"]),
        "unique_prompts": len(prompts), "scope": "Checks generated arithmetic and parameter rendering, not open-ended reasoning quality.",
    }
    (DATA / "full_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors or hashes["train"] & hashes["validation"]: raise SystemExit(1)


if __name__ == "__main__": main()
