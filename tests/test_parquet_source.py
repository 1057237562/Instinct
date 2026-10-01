"""Parquet dataset sources: the compiler contract and the loader paths.

The parquet fixtures are written with pyarrow rather than by the Rust
``dataset_compiler`` so the loader and streaming tests always run; the compiler
itself is exercised end to end by the tests that skip when it is not built.
"""

import itertools
import json
import os
import random
import subprocess
from pathlib import Path

import datasets  # noqa: F401  # Windows: pyarrow must load before torch.
import pytest
import torch
from transformers import AutoTokenizer

from scripts.data_loader import source_format
from scripts.data_loader.lm_dataset import (
    AgentRLDataset, DPODataset, PretrainDataset, SFTDataset,
)
from scripts.data_loader.streaming_chunks import (
    build_chunk_plan, materialize_parquet_range, materialize_range,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPILER = REPO_ROOT / "dataset_compiler" / "target" / "release" / (
    "dataset_compiler.exe" if os.name == "nt" else "dataset_compiler"
)

CHAT_FIELDS = ('role', 'content', 'reasoning_content', 'tools', 'tool_calls')

PRETRAIN_ROWS = [
    {"text": "秋天的早晨，清风拂面。", "token_count": 8},
    {"text": "def add(a, b):\n    return a + b", "token_count": 12},
    {"text": "hello world", "token_count": 3},
    {"text": "机器学习很实用。", "token_count": 7},
]

SFT_ROWS = [
    {"conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的？"},
    ]},
    {"conversations": [
        {"role": "system", "content": "You are helpful.", "tools": '[{"name": "search"}]'},
        {"role": "user", "content": "write a function"},
        {"role": "assistant", "content": "def f():\n    pass"},
    ]},
    {"conversations": [
        {"role": "user", "content": "think step by step"},
        {"role": "assistant", "content": "ok", "reasoning_content": "first, ..."},
    ]},
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("./model")


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def write_pretrain_parquet(path, rows=PRETRAIN_ROWS, row_group_size=2):
    import pyarrow as pa
    import pyarrow.parquet as parquet

    table = pa.table({
        "text": pa.array([row["text"] for row in rows], pa.string()),
        "token_count": pa.array([row["token_count"] for row in rows], pa.int64()),
    })
    parquet.write_table(table, path, row_group_size=row_group_size, compression="zstd")
    return path


def write_chat_parquet(path, rows=SFT_ROWS, row_group_size=2):
    import pyarrow as pa
    import pyarrow.parquet as parquet

    struct = pa.struct([(name, pa.string()) for name in CHAT_FIELDS])
    conversations = []
    for row in rows:
        messages = []
        for message in row["conversations"]:
            messages.append({name: message.get(name) for name in CHAT_FIELDS})
        conversations.append(messages)
    table = pa.table({"conversations": pa.array(conversations, pa.list_(struct))})
    parquet.write_table(table, path, row_group_size=row_group_size, compression="zstd")
    return path


def require_compiler():
    override = os.environ.get("INSTINCT_DATASET_COMPILER")
    binary = Path(override) if override else COMPILER
    if not binary.is_file():
        pytest.skip(
            "dataset_compiler is not built; run `cargo build --release` in dataset_compiler/"
        )
    return binary


# --------------------------------------------------------------------------- #
# source resolution
# --------------------------------------------------------------------------- #

def test_classify_recognises_both_formats():
    assert source_format.classify("a.jsonl") == "json"
    assert source_format.classify("a.JSONL") == "json"
    assert source_format.classify("a.jsonl.gz") == "json"
    assert source_format.classify("a.parquet") == "parquet"
    assert source_format.classify("a.pq") == "parquet"
    assert source_format.classify("a.txt") is None
    assert source_format.classify("pretrain_x.report.json") is None


def test_resolve_source_orders_directory_shards(tmp_path):
    write_pretrain_parquet(tmp_path / "b.parquet")
    write_pretrain_parquet(tmp_path / "a.parquet")
    source = source_format.resolve_source(tmp_path)

    assert source.is_parquet
    assert [Path(item).name for item in source.files] == ["a.parquet", "b.parquet"]
    assert source.data_files == source.files  # several files stay a list


def test_resolve_source_rejects_missing_and_unsupported_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        source_format.resolve_source(tmp_path / "nope.jsonl")
    write_jsonl(tmp_path / "rows.jsonl", PRETRAIN_ROWS)
    write_pretrain_parquet(tmp_path / "rows.parquet")
    (tmp_path / "rows.txt").write_text("text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported dataset file"):
        source_format.resolve_source(tmp_path / "rows.txt")


def test_resolve_source_prefers_parquet_for_a_mixed_directory(tmp_path):
    write_jsonl(tmp_path / "rows.jsonl", PRETRAIN_ROWS)
    write_pretrain_parquet(tmp_path / "rows.parquet")

    source = source_format.resolve_source(tmp_path)

    assert source.is_parquet
    assert [Path(item).name for item in source.files] == ["rows.parquet"]


def test_canonical_source_key_ignores_the_container_only(tmp_path):
    key = source_format.canonical_source_key

    assert key(tmp_path / "corpus.jsonl") == key(tmp_path / "corpus.parquet")
    assert key(tmp_path / "corpus.jsonl.gz") == key(tmp_path / "corpus.pq")
    assert key(tmp_path / "corpus.jsonl") != key(tmp_path / "other.jsonl")
    assert key(tmp_path / "corpus.jsonl") != key(tmp_path / "sub" / "corpus.jsonl")
    # A report sidecar is not a corpus, so it never collapses onto one.
    assert key(tmp_path / "corpus.report.json") != key(tmp_path / "corpus.jsonl")


def test_arrow_estimate_sees_through_parquet_compression(tmp_path):
    jsonl = write_jsonl(tmp_path / "rows.jsonl", PRETRAIN_ROWS)
    assert source_format.estimated_arrow_bytes(jsonl) == int(jsonl.stat().st_size * 1.25)

    # A compressible corpus, so the footer cannot dominate the comparison.
    rows = [{"text": "abc " * 200, "token_count": 5} for _ in range(200)]
    parquet = write_pretrain_parquet(tmp_path / "big.parquet", rows, row_group_size=50)
    metadata = parquet_file(parquet).metadata
    uncompressed = sum(
        metadata.row_group(index).total_byte_size
        for index in range(metadata.num_row_groups)
    )

    estimate = source_format.estimated_arrow_bytes(parquet)
    # The estimate tracks the footer's uncompressed totals, not the compressed
    # file size the JSONL heuristic would use.
    assert estimate != int(parquet.stat().st_size * 1.25)
    assert estimate >= uncompressed
    assert estimate < uncompressed * 1.2 + 64 * 1024


def parquet_file(path):
    import pyarrow.parquet as parquet

    return parquet.ParquetFile(path)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

def test_pretrain_loader_matches_across_formats(tmp_path, tokenizer):
    jsonl = PretrainDataset(
        str(write_jsonl(tmp_path / "p.jsonl", PRETRAIN_ROWS)), tokenizer, max_length=64,
    )
    parquet = PretrainDataset(
        str(write_pretrain_parquet(tmp_path / "p.parquet")), tokenizer, max_length=64,
    )

    assert len(parquet) == len(jsonl) == len(PRETRAIN_ROWS)
    assert parquet.samples.features["text"] == datasets.Value("string")
    for index in range(len(jsonl)):
        left = jsonl[index]
        right = parquet[index]
        assert torch.equal(left[0], right[0])
        assert torch.equal(left[1], right[1])


def test_sft_loader_matches_across_formats(tmp_path, tokenizer):
    jsonl = SFTDataset(
        str(write_jsonl(tmp_path / "s.jsonl", SFT_ROWS)), tokenizer, max_length=96,
    )
    parquet = SFTDataset(
        str(write_chat_parquet(tmp_path / "s.parquet")), tokenizer, max_length=96,
    )

    assert len(parquet) == len(jsonl) == len(SFT_ROWS)
    # Parquet keeps its own schema; the loader must not narrow it.
    assert set(parquet.samples.features["conversations"].feature) >= {"role", "content"}
    for index in range(len(jsonl)):
        # ``pre_processing_chat`` samples a system prompt from the global RNG.
        random.seed(20240924)
        left = jsonl[index]
        random.seed(20240924)
        right = parquet[index]
        assert torch.equal(left[0], right[0])
        assert torch.equal(left[1], right[1])


def test_packed_sft_loader_matches_across_formats(tmp_path, tokenizer):
    def build(path):
        return SFTDataset(
            str(path), tokenizer, max_length=96, packing=True,
            packing_batch_size=2, packing_seed=42, packing_mode='fixed',
            packing_num_proc=1,
        )

    jsonl = build(write_jsonl(tmp_path / "pack.jsonl", SFT_ROWS))
    parquet = build(write_chat_parquet(tmp_path / "pack.parquet"))

    assert len(jsonl) == len(parquet)
    for index in range(len(jsonl)):
        left = jsonl[index]
        right = parquet[index]
        assert torch.equal(left[0], right[0])
        assert torch.equal(left[1], right[1])
        assert torch.equal(left[2], right[2])


def test_sft_loader_rejects_a_pretrain_shaped_parquet(tmp_path, tokenizer):
    path = write_pretrain_parquet(tmp_path / "wrong.parquet")
    with pytest.raises(ValueError, match="conversations"):
        SFTDataset(str(path), tokenizer, max_length=64)


def test_dpo_loader_reads_parquet(tmp_path, tokenizer):
    import pyarrow as pa
    import pyarrow.parquet as parquet

    struct = pa.struct([("role", pa.string()), ("content", pa.string())])
    chosen = [[{"role": "assistant", "content": "good"}]]
    rejected = [[{"role": "assistant", "content": "bad"}]]
    table = pa.table({
        "chosen": pa.array(chosen, pa.list_(struct)),
        "rejected": pa.array(rejected, pa.list_(struct)),
    })
    path = tmp_path / "dpo.parquet"
    parquet.write_table(table, path, compression="zstd")

    dataset = DPODataset(str(path), tokenizer, max_length=32)
    sample = dataset[0]
    assert set(sample) == {
        "x_chosen", "y_chosen", "mask_chosen",
        "x_rejected", "y_rejected", "mask_rejected",
    }
    assert sample["x_chosen"].shape == (31,)


def test_agent_loader_keeps_gt_as_a_list(tmp_path, tokenizer):
    import pyarrow as pa
    import pyarrow.parquet as parquet

    struct = pa.struct([(name, pa.string()) for name in CHAT_FIELDS])
    table = pa.table({
        "conversations": pa.array([[
            {"role": "system", "content": "solve it", "reasoning_content": None,
             "tools": None, "tool_calls": None},
            {"role": "user", "content": "2+2?", "reasoning_content": None,
             "tools": None, "tool_calls": None},
        ]], pa.list_(struct)),
        "gt": pa.array([["4", "four"]], pa.list_(pa.string())),
    })
    path = tmp_path / "agent.parquet"
    parquet.write_table(table, path, compression="zstd")

    sample = AgentRLDataset(str(path), tokenizer, max_length=32)[0]
    assert sample["gt"] == ["4", "four"]
    assert sample["messages"][0]["role"] == "system"


def test_agent_loader_rejects_a_conversations_only_parquet(tmp_path, tokenizer):
    path = write_pretrain_parquet(tmp_path / "wrong_agent.parquet")
    with pytest.raises(ValueError, match="conversations"):
        AgentRLDataset(str(path), tokenizer, max_length=32)


# --------------------------------------------------------------------------- #
# streaming chunks
# --------------------------------------------------------------------------- #

def test_parquet_plan_tiles_the_source_and_materializes_in_order(tmp_path):
    rows = [
        {"text": f"row {index}", "token_count": index + 2}
        for index in range(10)
    ]
    source = write_pretrain_parquet(tmp_path / "plan.parquet", rows, row_group_size=2)
    plan_dir = tmp_path / "plans"
    plan = build_chunk_plan(
        source, chunk_bytes=1, max_length=8,
        tokenizer=PytestTokenizer(), plan_dir=plan_dir,
    )

    assert plan["identity"]["unit"] == "rows"
    assert plan["rows"] == len(rows)
    assert plan["tokens"] == sum(min(row["token_count"], 8) for row in rows)
    assert plan["chunks"][0]["start"] == 0
    assert plan["chunks"][-1]["end"] == len(rows)
    assert all(
        left["end"] == right["start"]
        for left, right in zip(plan["chunks"], plan["chunks"][1:])
    )

    import pyarrow.parquet as parquet

    restored = []
    for chunk in plan["chunks"]:
        target = tmp_path / f"chunk-{chunk['index']}.parquet"
        written = materialize_range(source, target, chunk["start"], chunk["end"])
        assert written == chunk["rows"]
        table = parquet.read_table(target)
        assert table.num_rows == chunk["rows"]
        restored.extend(table.to_pylist())
    assert [row["text"] for row in restored] == [row["text"] for row in rows]
    assert [row["token_count"] for row in restored] == [
        row["token_count"] for row in rows
    ]


def test_parquet_plan_is_reused_until_the_source_changes(tmp_path):
    source = write_pretrain_parquet(tmp_path / "reuse.parquet", row_group_size=2)
    plan_dir = tmp_path / "plans"
    first = build_chunk_plan(
        source, chunk_bytes=1, max_length=8,
        tokenizer=PytestTokenizer(), plan_dir=plan_dir,
    )
    plan_file = next(plan_dir.glob("*.json"))
    stamp = plan_file.stat().st_mtime_ns

    again = build_chunk_plan(
        source, chunk_bytes=1, max_length=8,
        tokenizer=PytestTokenizer(), plan_dir=plan_dir,
    )
    assert again == first
    assert plan_file.stat().st_mtime_ns == stamp


def test_parquet_plan_tokenizes_text_when_token_count_is_absent(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as parquet

    table = pa.table({"text": pa.array(["alpha beta", "gamma"], pa.string())})
    source = tmp_path / "no_counts.parquet"
    parquet.write_table(table, source, row_group_size=1, compression="zstd")

    tokenizer = AutoTokenizer.from_pretrained("./model")
    plan = build_chunk_plan(
        source, chunk_bytes=1, max_length=16, tokenizer=tokenizer,
        plan_dir=tmp_path / "plans",
    )
    expected = sum(
        min(len(tokenizer(text, add_special_tokens=False).input_ids) + 2, 16)
        for text in ("alpha beta", "gamma")
    )
    assert plan["tokens"] == expected


def test_materialize_parquet_range_honors_exact_bounds(tmp_path):
    import pyarrow.parquet as parquet

    rows = [{"text": f"row {index}", "token_count": index} for index in range(9)]
    source = write_pretrain_parquet(tmp_path / "bounds.parquet", rows, row_group_size=3)
    target = tmp_path / "slice.parquet"

    # Deliberately mid-row-group: the plan is row-group aligned, but an
    # arbitrary range must still produce exactly the rows it names.
    assert materialize_parquet_range(source, target, 2, 7) == 5
    table = parquet.read_table(target)
    assert [row["text"] for row in table.to_pylist()] == [
        f"row {index}" for index in range(2, 7)
    ]


def test_materialize_parquet_range_rejects_bad_bounds(tmp_path):
    source = write_pretrain_parquet(tmp_path / "bad.parquet", row_group_size=2)
    with pytest.raises(ValueError, match="invalid parquet row range"):
        materialize_parquet_range(source, tmp_path / "out.parquet", 0, 999)
    with pytest.raises(ValueError, match="invalid parquet row range"):
        materialize_parquet_range(source, tmp_path / "out.parquet", 3, 3)


class PytestTokenizer:
    """Fail loudly if planning tokenizes a corpus that carries token counts."""

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("token_count rows must not be tokenized during planning")


# --------------------------------------------------------------------------- #
# compiled corpora (needs the Rust binary)
# --------------------------------------------------------------------------- #

def test_compiler_output_matches_the_jsonl_source(tmp_path, tokenizer):
    binary = require_compiler()
    source = write_jsonl(tmp_path / "corpus.jsonl", PRETRAIN_ROWS)
    output = tmp_path / "corpus.parquet"
    result = subprocess.run(
        [str(binary), "compile", "-i", str(source), "-o", str(output),
         "--row-group-rows", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()

    jsonl = PretrainDataset(str(source), tokenizer, max_length=48)
    parquet = PretrainDataset(str(output), tokenizer, max_length=48)
    assert len(jsonl) == len(parquet)
    for index in range(len(jsonl)):
        assert torch.equal(jsonl[index][0], parquet[index][0])
        assert torch.equal(jsonl[index][1], parquet[index][1])

    verify = subprocess.run(
        [str(binary), "verify", str(source), str(output), "--rows", "100"],
        capture_output=True, text=True,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "value_mismatches=0" in verify.stdout


def test_compiled_chat_corpus_keeps_tool_payloads(tmp_path, tokenizer):
    binary = require_compiler()
    source = write_jsonl(tmp_path / "chat.jsonl", SFT_ROWS)
    output = tmp_path / "chat.parquet"
    result = subprocess.run(
        [str(binary), "compile", "-i", str(source), "-o", str(output), "--format", "sft"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    dataset = SFTDataset(str(output), tokenizer, max_length=96)
    messages = dataset.samples[1]["conversations"]
    assert messages[0]["tools"] == '[{"name": "search"}]'
    assert messages[0]["reasoning_content"] is None


def test_recompiled_resume_note_reports_the_cursor_shift(tmp_path):
    from trainer.train_pretrain import _recompiled_resume_note

    plan = {
        "rows": 1000,
        "chunks": [
            {"index": 0, "start": 0, "end": 400, "rows": 400, "tokens": 10},
            {"index": 1, "start": 400, "end": 1000, "rows": 600, "tokens": 10},
        ],
    }
    jsonl = str(tmp_path / "corpus.jsonl")
    parquet = str(tmp_path / "corpus.parquet")

    note = _recompiled_resume_note(
        {"data_path": jsonl, "streaming_chunk_index": 1, "streaming_chunk_step": 7},
        plan, parquet,
    )
    assert note is not None
    assert "chunk 2 step 7" in note
    assert "row 400 of 1,000" in note

    # Same path, an epoch boundary, or another corpus: nothing to report.
    assert _recompiled_resume_note(
        {"data_path": jsonl, "streaming_chunk_index": 1}, plan, jsonl,
    ) is None
    assert _recompiled_resume_note(
        {"data_path": jsonl, "streaming_chunk_index": 0, "streaming_chunk_step": 0},
        plan, parquet,
    ) is None
    assert _recompiled_resume_note(
        {"data_path": str(tmp_path / "other.jsonl"), "streaming_chunk_index": 1},
        plan, parquet,
    ) is None


def _aligned_parquet(path, rows, boundaries, chunk_bytes):
    """Write a parquet whose row groups and footer match --align-chunk-bytes."""
    import pyarrow as pa
    import pyarrow.parquet as parquet

    table = pa.table({
        "text": pa.array([row["text"] for row in rows], pa.string()),
        "token_count": pa.array([row["token_count"] for row in rows], pa.int64()),
    })
    table = table.replace_schema_metadata({
        "instinct.aligned_chunk_bytes": str(chunk_bytes),
        "instinct.aligned_chunk_rows": ",".join(str(item) for item in boundaries),
    })
    writer = parquet.ParquetWriter(path, table.schema, compression="zstd")
    start = 0
    for end in boundaries:
        batch = table.slice(start, end - start).to_batches(max_chunksize=end - start)[0]
        writer.write_batch(batch, row_group_size=end - start)
        start = end
    writer.close()
    return path


def test_aligned_parquet_plan_reuses_the_recorded_boundaries(tmp_path):
    rows = [{"text": f"row {index}", "token_count": index + 1} for index in range(10)]
    source = _aligned_parquet(tmp_path / "aligned.parquet", rows, [4, 7, 10], 64)
    plan = build_chunk_plan(
        source, chunk_bytes=64, max_length=32,
        tokenizer=PytestTokenizer(), plan_dir=tmp_path / "plans",
    )

    assert plan["identity"]["aligned_chunk_bytes"] == 64
    assert [(c["start"], c["end"]) for c in plan["chunks"]] == [(0, 4), (4, 7), (7, 10)]
    assert [c["rows"] for c in plan["chunks"]] == [4, 3, 3]
    assert [c["tokens"] for c in plan["chunks"]] == [10, 18, 27]


def test_aligned_boundaries_win_over_a_different_requested_chunk_size(tmp_path, capsys):
    rows = [{"text": f"row {index}", "token_count": 1} for index in range(8)]
    source = _aligned_parquet(tmp_path / "aligned2.parquet", rows, [4, 8], 1024)
    plan = build_chunk_plan(
        source, chunk_bytes=64, max_length=32,
        tokenizer=PytestTokenizer(), plan_dir=tmp_path / "plans",
    )

    assert [c["rows"] for c in plan["chunks"]] == [4, 4]
    assert "is aligned to" in capsys.readouterr().out


def test_malformed_alignment_metadata_falls_back(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as parquet

    table = pa.table({
        "text": pa.array(["a", "b", "c"], pa.string()),
        "token_count": pa.array([1, 2, 3], pa.int64()),
    })
    # The last boundary must be the row count; a stale list is ignored.
    table = table.replace_schema_metadata({"instinct.aligned_chunk_rows": "2,5"})
    source = tmp_path / "stale.parquet"
    parquet.write_table(table, source, row_group_size=1, compression="zstd")

    plan = build_chunk_plan(
        source, chunk_bytes=1, max_length=8,
        tokenizer=PytestTokenizer(), plan_dir=tmp_path / "plans",
    )
    assert "aligned_chunk_bytes" not in plan["identity"]
    assert plan["rows"] == 3


def test_compiled_alignment_matches_the_jsonl_chunk_plan(tmp_path, tokenizer):
    binary = require_compiler()
    rows = [
        {"text": f"row {index}: " + "x" * (30 + index * 7), "token_count": index + 2}
        for index in range(40)
    ]
    source = write_jsonl(tmp_path / "corpus.jsonl", rows)
    output = tmp_path / "corpus.parquet"
    chunk_bytes = source.stat().st_size // 4          # a handful of chunks
    result = subprocess.run(
        [str(binary), "compile", "-i", str(source), "-o", str(output),
         "--align-chunk-bytes", str(chunk_bytes), "--overwrite"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    plans = tmp_path / "plans"
    jsonl_plan = build_chunk_plan(
        source, chunk_bytes=chunk_bytes, max_length=64,
        tokenizer=tokenizer, plan_dir=plans,
    )
    parquet_plan = build_chunk_plan(
        output, chunk_bytes=chunk_bytes, max_length=64,
        tokenizer=tokenizer, plan_dir=plans,
    )

    assert len(jsonl_plan["chunks"]) > 1
    # JSONL ranges are bytes and parquet ranges are rows, so compare the row
    # tiling: each parquet chunk must end where the JSONL chunk ends.
    assert [c["end"] for c in parquet_plan["chunks"]] == list(
        itertools.accumulate(c["rows"] for c in jsonl_plan["chunks"])
    )
    assert [c["rows"] for c in parquet_plan["chunks"]] == [
        c["rows"] for c in jsonl_plan["chunks"]
    ]
    assert [c["tokens"] for c in parquet_plan["chunks"]] == [
        c["tokens"] for c in jsonl_plan["chunks"]
    ]
    # One row group per chunk, so materializing a chunk copies a single group.
    import pyarrow.parquet as parquet
    metadata = parquet.ParquetFile(output).metadata
    assert metadata.num_row_groups == len(parquet_plan["chunks"])


def test_resume_note_reports_an_aligned_plan_as_exact(tmp_path):
    from trainer.train_pretrain import _recompiled_resume_note

    jsonl = str(tmp_path / "corpus.jsonl")
    parquet = str(tmp_path / "corpus.parquet")
    saved = {
        "data_path": jsonl,
        "streaming_chunk_index": 1,
        "streaming_chunk_step": 7,
        "streaming_chunk_mb": 64,
    }
    plan = {
        "rows": 1000,
        "identity": {"aligned_chunk_bytes": 64 * 1024 ** 2},
        "chunks": [
            {"index": 0, "start": 0, "end": 400, "rows": 400, "tokens": 10},
            {"index": 1, "start": 400, "end": 1000, "rows": 600, "tokens": 10},
        ],
    }
    note = _recompiled_resume_note(dict(saved), plan, parquet)
    assert note is not None and "same rows" in note

    # A file aligned for a different chunk size still reports the shift.
    drifted = {
        **plan,
        "identity": {"aligned_chunk_bytes": 128 * 1024 ** 2},
    }
    shift = _recompiled_resume_note(dict(saved), drifted, parquet)
    assert shift is not None and "row 400" in shift
