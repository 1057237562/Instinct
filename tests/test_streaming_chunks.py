import json
from pathlib import Path
from types import SimpleNamespace
import threading

import datasets  # noqa: F401  # Windows: pyarrow must load before torch.
import pytest
import torch

from dataset.streaming_chunks import build_jsonl_chunk_plan, materialize_jsonl_range
from trainer.streaming_pretrain import (
    ChunkedPackedEpochLoader,
    should_stream_pretrain,
    validate_streaming_budget,
    streaming_token_progress,
)


class NeverCalledTokenizer:
    def __call__(self, *_args, **_kwargs):
        raise AssertionError("token_count rows must not be tokenized during planning")


def test_streaming_eta_uses_tokens_since_resume_and_current_epoch():
    # Epoch 2: resumed at 200 tokens, consumed another 100 in one minute.
    done, eta = streaming_token_progress(1300, 1200, 1000, 1, 60)
    assert done == 300
    assert eta == 7
    assert streaming_token_progress(2000, 1200, 1000, 1, 60) == (1000, 0)


def test_completed_streaming_cursor_does_not_read_another_chunk():
    def no_dataset(*args):
        raise AssertionError('Completed epoch must not read any data')

    loader = ChunkedPackedEpochLoader(
        plan={'chunks': [None] * 35}, dataset_factory=no_dataset,
        packing_plan=None, args=SimpleNamespace(streaming_prefetch_chunks=0),
        epoch=0, data_config={},
        resume_config={'streaming_chunk_index': 35, 'streaming_chunk_step': 0},
    )
    assert list(iter(loader)) == []


def _write_rows(path: Path, count=12):
    rows = [
        {"text": "x" * (index + 3), "token_count": index + 2}
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return rows


def test_plan_ranges_are_lossless_bounded_and_reused(tmp_path):
    source = tmp_path / "train.jsonl"
    rows = _write_rows(source)
    plan_dir = tmp_path / "plans"
    plan = build_jsonl_chunk_plan(
        source, chunk_bytes=100, max_length=9,
        tokenizer=NeverCalledTokenizer(), plan_dir=plan_dir,
    )
    assert len(plan["chunks"]) > 1
    assert plan["rows"] == len(rows)
    assert plan["tokens"] == sum(min(row["token_count"], 9) for row in rows)
    assert plan["chunks"][0]["start"] == 0
    assert plan["chunks"][-1]["end"] == source.stat().st_size
    assert all(
        left["end"] == right["start"]
        for left, right in zip(plan["chunks"], plan["chunks"][1:])
    )

    restored = bytearray()
    for chunk in plan["chunks"]:
        target = tmp_path / f"chunk-{chunk['index']}.jsonl"
        materialize_jsonl_range(source, target, chunk["start"], chunk["end"])
        restored.extend(target.read_bytes())
    assert bytes(restored) == source.read_bytes()

    plan_file = next(plan_dir.glob("*.json"))
    before = plan_file.stat().st_mtime_ns
    again = build_jsonl_chunk_plan(
        source, chunk_bytes=100, max_length=9,
        tokenizer=NeverCalledTokenizer(), plan_dir=plan_dir,
    )
    assert again == plan
    assert plan_file.stat().st_mtime_ns == before


def test_streaming_auto_uses_disk_budget_and_validates_peak(tmp_path):
    source = tmp_path / "large.jsonl"
    source.write_bytes(b"x" * 1024)
    args = SimpleNamespace(
        dataset_streaming="auto", data_path=str(source),
        data_cache_max_gb=0.0000005, streaming_chunk_mb=1,
    )
    assert should_stream_pretrain(args) is True
    with pytest.raises(ValueError, match="too large"):
        validate_streaming_budget(args)
    args.dataset_streaming = "off"
    assert should_stream_pretrain(args) is False
    args.dataset_streaming = "on"
    assert should_stream_pretrain(args) is True


def test_prefetch_budget_accounts_for_current_and_next_chunk():
    args = SimpleNamespace(
        data_cache_max_gb=4.5,
        streaming_chunk_mb=1024,
        streaming_prefetch_chunks=1,
    )
    with pytest.raises(ValueError, match="too large"):
        validate_streaming_budget(args)
    args.streaming_prefetch_chunks = 0
    assert validate_streaming_budget(args) == 1024 ** 3


def test_chunk_loader_packs_next_chunk_in_thread_while_current_trains(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = []
    cache_enforcements = []
    main_thread = threading.get_ident()

    class FakeDataset(torch.utils.data.Dataset):
        samples = SimpleNamespace(cache_files=[])

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return torch.tensor([1]), torch.tensor([1]), torch.tensor([0])

    def factory(_packing, _sample_indices, byte_range):
        calls.append((byte_range, threading.get_ident()))
        if byte_range == (10, 20):
            started.set()
            assert release.wait(timeout=5), "test did not let background packing finish"
        return FakeDataset()

    class FakePackingPlan:
        packing_mode = "fixed"

        @staticmethod
        def batch_sampler(_dataset, **_kwargs):
            return [[0]]

        @staticmethod
        def loader_num_workers():
            return 0

    monkeypatch.setattr(
        "trainer.streaming_pretrain.release_cache_use", lambda _paths: None
    )
    monkeypatch.setattr(
        "trainer.streaming_pretrain.enforce_cache_budget",
        lambda: cache_enforcements.append(True) or {
            "removed": [], "removed_bytes": 0,
        },
    )
    loader = ChunkedPackedEpochLoader(
        plan={
            "identity": {"max_length": 8},
            "chunks": [
                {"index": 0, "start": 0, "end": 10, "rows": 1, "tokens": 4},
                {"index": 1, "start": 10, "end": 20, "rows": 1, "tokens": 4},
            ],
        },
        dataset_factory=factory,
        packing_plan=FakePackingPlan(),
        args=SimpleNamespace(
            streaming_prefetch_chunks=1,
            batch_size=1,
            bucket_gpu_memory_gb=16.0,
        ),
        epoch=0,
        data_config={},
    )

    iterator = iter(loader)
    try:
        first = next(iterator)
        assert first[0].shape == (1, 1)
        assert not started.is_set(), "prefetch stole CPU before the first train batch"
        second_result = []
        second_error = []

        def consume_second():
            try:
                second_result.append(next(iterator))
            except BaseException as error:
                second_error.append(error)

        consumer = threading.Thread(target=consume_second)
        consumer.start()
        assert started.wait(timeout=2), "next chunk did not start after the first batch"
        assert [item[0] for item in calls] == [(0, 10), (10, 20)]
        assert calls[0][1] == main_thread
        assert calls[1][1] != main_thread
    finally:
        release.set()

    consumer.join(timeout=5)
    assert not consumer.is_alive()
    assert not second_error
    second = second_result[0]
    assert second[0].shape == (1, 1)
    with pytest.raises(StopIteration):
        next(iterator)
    assert [item[0] for item in calls] == [(0, 10), (10, 20)]
    assert len(cache_enforcements) == 2
