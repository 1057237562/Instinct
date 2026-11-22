import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import clean_think_repetition as clean  # noqa: E402


def messages_row(think, answer="answer", content=None):
    row = {"messages": [{"role": "user", "content": "solve it"},
                        {"role": "assistant", "content": answer, "reasoning_content": think}]}
    if content is not None:
        row["messages"][-1]["content"] = content
    return row


def unique_words(count):
    return " ".join(f"token{i}x" for i in range(count))


def test_locate_reasoning_schema_variants():
    assert clean.locate_reasoning({"reasoning": "abc"}).text == "abc"
    located = clean.locate_reasoning(messages_row("think body"))
    assert (located.kind, located.key) == ("message_field", "reasoning_content")
    inline = {"conversations": [{"role": "user", "content": "q"},
                                {"role": "assistant", "content": "<think>thinking</think>`answer`"}]}
    located = clean.locate_reasoning(inline)
    assert located.kind == "inline" and located.text == "thinking" and not located.unclosed
    unclosed = {"conversations": [{"role": "assistant", "content": "<think>no close"}]}
    assert clean.locate_reasoning(unclosed).unclosed
    assert clean.locate_reasoning({"foo": "bar"}) is None


def test_set_reasoning_rewrites_inline_and_fields():
    row = {"conversations": [{"role": "assistant", "content": "<think>old</think>final"}]}
    located = clean.locate_reasoning(row)
    clean.set_reasoning(row, located, "new")
    assert row["conversations"][0]["content"] == "<think>new</think>final"
    row = messages_row("old")
    clean.set_reasoning(row, clean.locate_reasoning(row), "new")
    assert row["messages"][-1]["reasoning_content"] == "new"


def test_signals_detect_line_token_and_code_repetition():
    line = "This exact line is repeated many times inside reasoning.\n"
    info = clean.signals(line * 6)
    assert info["max_line"] >= 5 and "T3_repeated_line" in info["flags"]
    info = clean.signals("x x x x x " * 8 + unique_words(40))
    assert info["degenerate"] >= 5 and "T5_degenerate_tokens" in info["flags"]
    code = "```python\nprint(1)\n```\n"
    info = clean.signals(code * 3 + unique_words(30))
    assert info["dup_code"] >= 2 and "T6_repeated_code_block" in info["flags"]


def test_rewrite_stages():
    code = "```python\nprint(1)\n```"
    text = f"{code}\n{code}\n{unique_words(40)}"
    repaired = clean.rewrite(text)
    assert repaired.count("print(1)") == 1
    line = "The same long line appears again and again in the trace.\n"
    repaired = clean.rewrite(unique_words(30) + "\n" + line * 5 + unique_words(10))
    assert repaired.count("The same long line") == 1
    repaired = clean.rewrite("x x x x x x " + unique_words(20))
    assert repaired.startswith("x ")
    # fenced code content itself is preserved when it is not an exact duplicate
    unique_code = "```python\na = b + 1\n```"
    repaired = clean.rewrite(unique_code + " " + unique_words(30))
    assert "a = b + 1" in repaired


def test_clean_row_keep_rewrite_drop():
    clean_row = messages_row(unique_words(120))
    original = json.loads(json.dumps(clean_row))
    status, row, reason = clean.clean_row(clean_row)
    assert status == "keep_clean" and reason is None and row == original

    # mild sentence repetition: cheap repair removes <35% of the words
    repeated = "We should check the edge case carefully. "
    dirty = messages_row(unique_words(80) + ". " + repeated * 6)
    status, row, reason = clean.clean_row(dirty)
    assert status == "rewritten" and reason is None
    assert row["messages"][-1]["reasoning_content"].count("We should check") == 1
    assert row["messages"][-1]["content"] == "answer"

    # a pure degenerate loop is not repairable within the removal budget
    status, row, reason = clean.clean_row(messages_row("x x x x x " * 60))
    assert status == "dropped" and reason in ("removed_too_much", "too_short_after")

    assert clean.clean_row({"foo": "bar"})[2] == "no_reasoning"
    assert clean.clean_row(messages_row("ok", answer=""))[2] == "empty_answer"
    unclosed = {"conversations": [{"role": "assistant", "content": "<think>gone"}]}
    assert clean.clean_row(unclosed)[2] == "unclosed_think"


def test_allocate_quotas_short_floor_and_supply():
    quotas = clean.allocate_quotas(100, [10, 90, 0, 0, 0, 0], 0, 0.25)
    assert quotas[0] == 10  # supply < requested floor
    assert sum(quotas) == 100
    quotas = clean.allocate_quotas(100, [200, 200, 0, 0, 0, 0], 0, 0.25)
    assert quotas[0] == 25 and quotas[1] == 75
    quotas = clean.allocate_quotas(1000, [5, 10, 0, 0, 0, 0], 0, 0.25)
    assert sum(quotas) == 15  # cannot reach target


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_scan_aggregates_buckets_and_lanes(tmp_path):
    rows = []
    for index in range(3):
        row = messages_row(unique_words(100))
        row["source"] = "clean_source"
        rows.append(row)
    rows.append({**messages_row("x x x x x " * 20), "source": "dirty_source"})
    path = tmp_path / "in.jsonl"
    write_rows(path, rows)
    payload = clean.command_scan(type("A", (), dict(inputs=[str(path)], workers=1, report=None,
                                                    reasoning_key=None, lane_key=None))())
    assert payload["rows"] == 4 and payload["with_reasoning"] == 4
    assert payload["length_buckets"][0]["rows"] == 4
    assert payload["repetitive"] == 1
    lanes = {row["lane"]: row for row in payload["lanes"]}
    assert lanes["clean_source"]["repetitive"] == 0
    assert lanes["dirty_source"]["repetitive"] == 1


def test_clean_command_and_rebuild_lane(tmp_path):
    pool = []
    sizes = [80, 220, 600, 1000, 2000, 3000]
    for index, size in enumerate(sizes):
        pool.append(messages_row(unique_words(size)))
    pool.append(messages_row("x x x x x " * 40))  # should be dropped by cleaning
    source = tmp_path / "pool.jsonl"
    write_rows(source, pool)

    out = tmp_path / "clean.jsonl"
    report = tmp_path / "clean.report.json"
    args = type("A", (), dict(inputs=[str(source)], out=str(out), out_dir=None,
                              report=str(report), tmp_dir=str(tmp_path / "tmp"), workers=1,
                              reasoning_key=None, min_unique_5gram=0.88, max_occ=1,
                              max_removed_ratio=0.35, max_pingpong=2.0))()
    payload = clean.command_clean(args)
    assert payload["rows"] == 7 and payload["outputs"] == str(out)
    assert json.loads(report.read_text("utf-8"))["rows"] == 7
    lines = out.read_text("utf-8").splitlines()
    assert len(lines) == 6

    lane = tmp_path / "lane.jsonl"
    lane_report = tmp_path / "lane.report.json"
    rebuild_args = type("A", (), dict(inputs=[str(source)], out=str(lane),
                                      report=str(lane_report), target_rows=4, max_tokens=6500,
                                      short_quota=0.5, seed=7, tmp_dir=str(tmp_path / "rtmp"),
                                      reasoning_key=None, min_unique_5gram=0.88, max_occ=1,
                                      max_removed_ratio=0.35, max_pingpong=2.0, workers=1))()
    payload = clean.command_rebuild(rebuild_args)
    assert payload["selected_rows"] == 4
    assert payload["dropped"].get("too_long", 0) == 0
    written = lane.read_text("utf-8").splitlines()
    assert len(written) == 4
    # deterministic
    lane2 = tmp_path / "lane2.jsonl"
    rebuild_args.out = str(lane2)
    rebuild_args.tmp_dir = str(tmp_path / "rtmp2")
    clean.command_rebuild(rebuild_args)
    assert lane.read_bytes() == lane2.read_bytes()
