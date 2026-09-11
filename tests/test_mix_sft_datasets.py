"""Tests for bounded-memory SFT dataset mixing."""

import json
import random

from scripts.mix_sft_datasets import (
    iter_json_array,
    mix_datasets,
    replay_rows_for_fraction,
    reservoir_sample_jsonl,
)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row) + "\n")


def test_stream_json_array_across_tiny_read_chunks(tmp_path):
    path = tmp_path / "array.json"
    rows = [{"text": "alpha"}, {"text": "中文"}, {"number": 3}]
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    assert list(iter_json_array(path, read_chars=3)) == rows


def test_reservoir_sample_is_exact_and_deterministic(tmp_path):
    path = tmp_path / "source.jsonl"
    _write_jsonl(path, [{"id": index} for index in range(100)])

    first, seen = reservoir_sample_jsonl(path, 10, random.Random(7))
    second, _ = reservoir_sample_jsonl(path, 10, random.Random(7))

    assert seen == 100
    assert first == second
    assert len(first) == 10
    assert len({json.loads(line)["id"] for line in first}) == 10


def test_reservoir_returns_all_rows_when_request_is_larger(tmp_path):
    path = tmp_path / "source.jsonl"
    _write_jsonl(path, [{"id": index} for index in range(3)])

    sample, seen = reservoir_sample_jsonl(path, 10, random.Random(7))

    assert seen == 3
    assert len(sample) == 3


def test_replay_fraction_is_fraction_of_final_mix():
    assert replay_rows_for_fraction(300, 0.20) == 75


def test_end_to_end_mix_normalizes_and_preserves_original_metadata(tmp_path):
    magicoder = tmp_path / "magic.jsonl"
    math = tmp_path / "math.json"
    original = tmp_path / "sft_t2t.jsonl"
    output = tmp_path / "sft_mix.jsonl"
    _write_jsonl(magicoder, [
        {"instruction": f"code-{index}", "response": f"answer-{index}"}
        for index in range(3)
    ] + [{"instruction": "code-0", "response": "answer-0"}])
    math.write_text(json.dumps([
        {"instruction": f"math-{index}", "output": f"solution-{index}"}
        for index in range(2)
    ] + [{"instruction": "invalid", "output": ""}]), encoding="utf-8")
    _write_jsonl(original, [
        {"conversations": [
            {"role": "user", "content": f"old-{index}"},
            {"role": "assistant", "content": "reply", "reasoning_content": "kept"},
        ]}
        for index in range(20)
    ])

    stats = mix_datasets(
        magicoder, math, original, output,
        replay_fraction=0.20, seed=11, chunk_rows=2,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert stats["magicoder_rows"] == 3
    assert stats["duplicate_magicoder_rows"] == 1
    assert stats["math_rows"] == 2
    assert stats["invalid_math_rows"] == 1
    assert stats["replay_rows"] == 1
    assert stats["total_rows"] == 6
    assert len(rows) == 6
    assert len({json.dumps(row, sort_keys=True) for row in rows}) == 6
    assert all(set(row) == {"conversations"} for row in rows)
    replay = next(row for row in rows if row["conversations"][0]["content"].startswith("old-"))
    assert replay["conversations"][1]["reasoning_content"] == "kept"
