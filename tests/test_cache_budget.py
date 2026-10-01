import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.data_loader import cache_budget


def _file(path: Path, size: int, mtime_ns: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_lru_evicts_old_arrow_and_keeps_recent(tmp_path):
    old = _file(tmp_path / "json" / "old.arrow", 1000, 1_000_000)
    recent = _file(tmp_path / "json" / "recent.arrow", 1000, 2_000_000)
    report = cache_budget.enforce_cache_budget(tmp_path, max_bytes=1500)
    assert not old.exists()
    assert recent.exists()
    assert report["after_bytes"] <= 1500
    assert str(old) in report["removed"]


def test_active_lease_protects_mmap_cache(tmp_path):
    active = _file(tmp_path / "json" / "active.arrow", 1000, 1_000_000)
    disposable = _file(tmp_path / "json" / "disposable.arrow", 1000, 2_000_000)
    cache_budget.register_cache_use([active], tmp_path)
    report = cache_budget.enforce_cache_budget(tmp_path, max_bytes=1600)
    assert active.exists()
    assert not disposable.exists()
    assert report["after_bytes"] <= 1600


def test_expensive_packed_cache_is_evicted_after_plain_arrow(tmp_path):
    plain = _file(tmp_path / "json" / "plain.arrow", 900, 2_000_000)
    packed = _file(
        tmp_path / "instinct-packed" / ("a" * 64) / "part-00000.arrow",
        900,
        1_000_000,
    )
    manifest = packed.parent / "manifest.json"
    manifest.write_text(json.dumps({"files": []}), encoding="utf-8")
    report = cache_budget.enforce_cache_budget(tmp_path, max_bytes=1200)
    assert not plain.exists()
    assert packed.exists()
    assert str(plain) in report["removed"]


def test_budget_errors_instead_of_deleting_only_active_cache(tmp_path):
    active = _file(tmp_path / "json" / "active.arrow", 2000, 1_000_000)
    cache_budget.register_cache_use([active], tmp_path)
    with pytest.raises(cache_budget.CacheBudgetExceeded, match="Active cache files are protected"):
        cache_budget.enforce_cache_budget(tmp_path, max_bytes=1000)
    assert active.exists()


def test_zero_gb_disables_budget(tmp_path):
    _file(tmp_path / "json" / "large.arrow", 2000, 1_000_000)
    report = cache_budget.enforce_cache_budget(tmp_path, max_gb=0)
    assert report["budget_bytes"] is None
    assert (tmp_path / "json" / "large.arrow").exists()


def test_reservation_evicts_before_new_output_is_written(tmp_path):
    active = _file(tmp_path / "json" / "active.arrow", 800, 2_000_000)
    historical = _file(tmp_path / "json" / "historical.arrow", 800, 1_000_000)
    cache_budget.register_cache_use([active], tmp_path)
    report = cache_budget.enforce_cache_budget(
        tmp_path, max_bytes=3000, reserve_bytes=1200
    )
    assert active.exists()
    assert not historical.exists()
    assert report["after_bytes"] + report["reserve_bytes"] <= 3000


def test_released_lease_becomes_evictable(tmp_path):
    active = _file(tmp_path / "json" / "active.arrow", 1000, 1_000_000)
    cache_budget.register_cache_use([active], tmp_path)
    cache_budget.release_cache_use([active], tmp_path)
    cache_budget.enforce_cache_budget(tmp_path, max_bytes=500)
    assert not active.exists()


def test_completed_streaming_chunk_is_discarded_without_deleting_plan(tmp_path):
    chunk = tmp_path / "instinct-packed" / ("c" * 64)
    packed = _file(chunk / "part-00000.arrow", 1000, 1_000_000)
    plan = _file(tmp_path / "instinct-stream-plans" / "plan.json", 100, 1_000_000)
    cache_budget.register_cache_use([chunk], tmp_path)
    assert cache_budget.discard_rebuildable_cache([chunk], tmp_path) == []
    cache_budget.release_cache_use([chunk], tmp_path)
    assert cache_budget.discard_rebuildable_cache([chunk], tmp_path) == [str(chunk)]
    assert not packed.exists()
    assert plan.exists()


def test_dead_process_lease_is_removed(tmp_path):
    active = _file(tmp_path / "json" / "stale.arrow", 1000, 1_000_000)
    lease_dir = tmp_path / ".instinct-leases"
    lease_dir.mkdir()
    lease = lease_dir / "999999999.json"
    lease.write_text(json.dumps({"pid": 999999999, "paths": [str(active)]}), encoding="utf-8")
    cache_budget.enforce_cache_budget(tmp_path, max_bytes=500)
    assert not lease.exists()
    assert not active.exists()


def test_nested_worker_counts_against_parent_budget(tmp_path, monkeypatch):
    parent = tmp_path / "datasets"
    worker = parent / "instinct-packed" / "build-test" / "intermediate"
    active = _file(worker / "json" / "active.arrow", 900, 2_000_000)
    historical = _file(parent / "json" / "historical.arrow", 900, 1_000_000)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(worker))
    monkeypatch.setenv("INSTINCT_DATA_CACHE_BUDGET_ROOT", str(parent))
    cache_budget.register_cache_use([active])
    cache_budget.enforce_cache_budget(max_bytes=1400)
    assert active.exists()
    assert not historical.exists()


def test_managed_worker_does_not_require_hf_datasets_cache(tmp_path, monkeypatch):
    """Regression: WebUI commonly sets HF_HOME without exporting the cache leaf."""
    import datasets
    from scripts.data_loader import lm_dataset
    from scripts.data_loader.managed_cache import _worker

    # _worker normally exits with its process. Restore this module-global in
    # the direct unit test so the temporary cache path cannot leak downstream.
    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_CACHE", datasets.config.HF_DATASETS_CACHE
    )

    budget_root = tmp_path / "datasets"
    work = budget_root / "instinct-packed" / "build-test"
    packed = work / "intermediate" / "packed.arrow"
    _file(packed, 32, 1_000_000)

    fake = SimpleNamespace(
        bucket_ranges=[{"blocks": 1}],
        full_raw_sample_count=1,
        raw_sample_count=1,
        packing_mode="fixed",
        discarded_long_sample_count=0,
        samples=SimpleNamespace(cache_files=[{"filename": str(packed)}]),
    )
    monkeypatch.setattr(lm_dataset, "PretrainDataset", lambda **kwargs: fake)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.setenv("INSTINCT_DATA_CACHE_WORKER", "0")
    monkeypatch.setenv("INSTINCT_DATA_CACHE_MAX_GB", "0")
    monkeypatch.setenv("INSTINCT_DATA_CACHE_BUDGET_ROOT", str(budget_root))

    _worker("pretrain", {}, str(work), str(budget_root))

    assert os.environ["HF_DATASETS_CACHE"] == str(work / "intermediate")
    result = json.loads((work / "result.json").read_text(encoding="utf-8"))
    assert result["files"] == [str(packed)]


def test_managed_worker_materializes_only_requested_jsonl_range(tmp_path, monkeypatch):
    import datasets
    from scripts.data_loader import lm_dataset
    from scripts.data_loader.managed_cache import _worker

    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_CACHE", datasets.config.HF_DATASETS_CACHE
    )
    budget_root = tmp_path / "datasets"
    work = budget_root / "instinct-packed" / "build-range"
    packed = work / "intermediate" / "packed.arrow"
    _file(packed, 32, 1_000_000)
    source = tmp_path / "source.jsonl"
    lines = [b'{"text":"a"}\n', b'{"text":"b"}\n', b'{"text":"c"}\n']
    source.write_bytes(b"".join(lines))
    captured = {}

    def fake_dataset(**kwargs):
        captured["bytes"] = Path(kwargs["data_path"]).read_bytes()
        return SimpleNamespace(
            bucket_ranges=[{"blocks": 1}], full_raw_sample_count=1,
            raw_sample_count=1, packing_mode="fixed",
            discarded_long_sample_count=0,
            samples=SimpleNamespace(cache_files=[{"filename": str(packed)}]),
        )

    monkeypatch.setattr(lm_dataset, "PretrainDataset", fake_dataset)
    monkeypatch.setenv("INSTINCT_DATA_CACHE_WORKER", "0")
    monkeypatch.setenv("INSTINCT_DATA_CACHE_MAX_GB", "0")
    monkeypatch.setenv("INSTINCT_DATA_CACHE_BUDGET_ROOT", str(budget_root))
    _worker(
        "pretrain",
        {"data_path": str(source), "byte_range": (len(lines[0]), len(lines[0]) + len(lines[1]))},
        str(work), str(budget_root),
    )
    assert captured["bytes"] == lines[1]


def test_trainer_cli_defaults_to_five_gibibytes(monkeypatch):
    monkeypatch.delenv("INSTINCT_DATA_CACHE_MAX_GB", raising=False)
    from trainer.trainer_cli import build_trainer_parser

    args = build_trainer_parser("test").parse_args([])
    assert args.data_cache_max_gb == 5.0
    assert args.dataset_streaming == "auto"
    assert args.streaming_chunk_mb == 1024
    assert args.streaming_prefetch_chunks == 1
    assert args.cache_build_mode == "inline"


def test_source_digest_memo_invalidates_from_head_tail_signature(tmp_path):
    from scripts.data_loader.managed_cache import _source_digest

    source = tmp_path / "source.jsonl"
    source.write_bytes(b"a" * (128 * 1024) + b"b")
    cache = tmp_path / "cache"
    cache.mkdir()
    first = _source_digest(source, cache)
    index_mtime = (cache / ".source-digests.json").stat().st_mtime_ns
    assert _source_digest(source, cache) == first
    assert (cache / ".source-digests.json").stat().st_mtime_ns == index_mtime
    source.write_bytes(b"a" * (128 * 1024) + b"c")
    assert _source_digest(source, cache) != first
