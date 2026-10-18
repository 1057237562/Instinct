import hashlib
import io
import json
import os
import sys
import time

import pytest

from scripts.data_builder.coder_novelty import ExclusionIndex, IdentityReview, build_exclusion_index, build_merged_exclusion_index
from scripts.data_builder import collect_coder_pretrain_12b as collector


def make_index(tmp_path, texts):
    source = tmp_path / "baseline.jsonl"
    source.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
    report = tmp_path / "baseline.report.json"
    report.write_text(json.dumps({"rows": len(texts), "output_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}), encoding="utf-8")
    index = tmp_path / "prior.blake2b128"
    build_exclusion_index(source, index, report)
    return ExclusionIndex(index)


def test_baseline_exclusion_normalizes_outer_whitespace_and_validates_source(tmp_path):
    index = make_index(tmp_path, ["old code", " old code\n", "other text"])
    assert index.report["source_rows"] == 3
    assert index.report["fingerprints"] == 2
    assert index.contains("\nold code  ")
    assert index.contains("other text")
    assert not index.contains("new code")
    source = tmp_path / "baseline.jsonl"
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Baseline changed"):
        ExclusionIndex(index.path)


def test_bad_baseline_checksum_never_publishes_index(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"text":"hello"}\n', encoding="utf-8")
    report = tmp_path / "source.report.json"
    report.write_text(json.dumps({"rows": 1, "output_sha256": "wrong"}), encoding="utf-8")
    target = tmp_path / "index"
    with pytest.raises(ValueError, match="release report"):
        build_exclusion_index(source, target, report)
    assert not target.exists()


def test_collector_excludes_prior_rows_and_keeps_identity_hits_for_review(tmp_path, monkeypatch):
    old = "Educational material about numbers and proofs. " * 8
    identity = "I am Qwen, a language model developed by Alibaba Cloud. " * 5
    factual = "Qwen is a family of language models. This article discusses its tokenizer and architecture. " * 3
    index = make_index(tmp_path, [old])
    monkeypatch.setattr(collector, "EXCLUSION_INDEX", index)
    monkeypatch.setattr(collector, "IDENTITY_REVIEW", IdentityReview())

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return type("Encoded", (), {"ids": list(range(100))})()

    output, reviews = io.BytesIO(), io.BytesIO()
    result = collector.process_rows("natural_zh", [{"text": old, "id": "old"}, {"text": identity, "id": "identity"}, {"text": factual, "id": "fact"}], "new-shard", output, Tokenizer(), 1000, reviews)
    assert result["selection"]["rejected_prior_corpus_text"] == 1
    assert result["rows"] == 2
    assert result["tokens"] == 204
    assert result["identity_review"]["candidate_rows"] == 2
    assert result["identity_review"]["training_ready"] is False
    assert result["identity_review"]["dropped_rows"] == 0
    assert result["identity_review"]["candidate_sha256"] == hashlib.sha256(reviews.getvalue()).hexdigest()
    assert [json.loads(line)["text"] for line in output.getvalue().splitlines()] == [identity.strip(), factual.strip()]


def test_factual_mentions_are_reviewed_without_rewriting():
    row = {"text": "Qwen was released by Alibaba. The models have several sizes."}
    reasons, hits = IdentityReview().inspect(row)
    assert hits
    assert any(item["rule"] == "foreign_model_or_provider_mention" for item in reasons)
    assert row["text"].startswith("Qwen was released")


def test_capped_http_range_stays_partial_and_resumes(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 206

        def __init__(self, data, content_range):
            super().__init__(data)
            self.headers = {"Content-Length": str(len(data)), "Content-Range": content_range}

        def getcode(self):
            return self.status

    partial, destination = tmp_path / "download.part", tmp_path / "download"
    monkeypatch.setattr(collector, "urlopen", lambda request, timeout: Response(b"abc", "bytes 0-2/6"))
    with pytest.raises(EOFError, match="short download"):
        collector._download_to("https://example.com/data", partial, 0, 1, "data", destination)
    assert partial.read_bytes() == b"abc"
    assert not destination.exists()
    monkeypatch.setattr(collector, "urlopen", lambda request, timeout: Response(b"def", "bytes 3-5/6"))
    collector._download_to("https://example.com/data", partial, 3, 1, "data", destination)
    assert destination.read_bytes() == b"abcdef"


def test_misaligned_http_range_cannot_corrupt_partial(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Length": "3", "Content-Range": "bytes 0-2/6"}

        def getcode(self):
            return self.status

    partial = tmp_path / "download.part"
    partial.write_bytes(b"abc")
    monkeypatch.setattr(collector, "urlopen", lambda request, timeout: Response(b"wrong"))
    with pytest.raises(OSError, match="Invalid Content-Range"):
        collector._download_to("https://example.com/data", partial, 3, 1, "data", tmp_path / "download")
    assert partial.read_bytes() == b"abc"


@pytest.mark.skipif(os.name != "nt", reason="Windows WMI process creation")
def test_independent_windows_launch_preserves_environment_and_leaves_job(tmp_path, monkeypatch):
    from scripts.data_builder.start_coder_novel_12b import launch_windows_independent

    monkeypatch.setenv("INSTINCT_TEST_LAUNCH_VALUE", "preserved")
    root = tmp_path / "launch with spaces"
    (root / "logs").mkdir(parents=True)
    result_path = root / "probe.json"
    code = """
import ctypes, json, os, sys
from ctypes import wintypes
from pathlib import Path
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=wintypes.HANDLE
k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
k.IsProcessInJob.restype=wintypes.BOOL
inside=wintypes.BOOL()
if not k.IsProcessInJob(k.GetCurrentProcess(),None,ctypes.byref(inside)):
    raise ctypes.WinError(ctypes.get_last_error())
Path(sys.argv[1]).write_text(json.dumps({'in_job':bool(inside.value),'value':os.environ.get('INSTINCT_TEST_LAUNCH_VALUE')}),encoding='utf-8')
"""
    result = launch_windows_independent([sys.executable, "-c", code, str(result_path)], root)
    assert result["pid"] > 0
    deadline = time.monotonic() + 10
    while not result_path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert json.loads(result_path.read_text(encoding="utf-8")) == {"in_job": False, "value": "preserved"}


def test_process_detection_rejects_unrelated_pid_and_other_collection_root(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.data_builder import start_coder_novel_12b as runner

    script = str(runner.ROOT / "scripts/data_builder/collect_coder_pretrain_12b.py")
    def proc(pid, command):
        return SimpleNamespace(info={"pid": pid, "cmdline": command, "cwd": str(runner.ROOT), "status": "running", "create_time": 1.0})
    processes = [proc(901, [sys.executable, "unrelated.py"]),
                 proc(902, [sys.executable, script, "--out-dir", str(tmp_path / "other"), "--component", "code"]),
                 proc(903, [sys.executable, script, "--out-dir", str(tmp_path), "--component", "code"])]
    monkeypatch.setattr(runner.psutil, "process_iter", lambda attrs: iter(processes))
    result = runner.active_collection_processes(tmp_path)
    assert [item["pid"] for item in result] == [903]
    assert result[0]["component"] == "code"


def test_stale_status_and_unlocked_lock_file_allow_resume(tmp_path, monkeypatch, capsys):
    from scripts.data_builder import start_coder_novel_12b as runner

    (tmp_path / "run.lock").write_bytes(b"0")
    (tmp_path / ".pause_request").touch()
    (tmp_path / "status.json").write_text(json.dumps({"pid": 99999999, "phase": "collecting"}), encoding="utf-8")
    monkeypatch.setattr(runner, "active_collection_processes", lambda root: [])
    launches = []
    def launch(command, root):
        launches.append(command)
        return {"pid": 123, "launch_method": "test"}
    monkeypatch.setattr(runner, "launch_windows_independent", launch)
    if os.name != "nt":
        from types import SimpleNamespace
        monkeypatch.setattr(runner.subprocess, "Popen", lambda command, **kwargs: (launches.append(command) or SimpleNamespace(pid=123)))
    monkeypatch.setattr(sys, "argv", ["start_coder_novel_12b.py", "--background", "--out-dir", str(tmp_path)])
    runner.main()
    assert len(launches) == 1
    assert not (tmp_path / ".pause_request").exists()
    assert '"pid": 123' in capsys.readouterr().out


def test_progress_does_not_call_dead_worker_running(tmp_path, capsys):
    from scripts.data_builder import start_coder_novel_12b as runner

    runner.print_running_progress(tmp_path, {"phase": "collecting", "components": {"code": {"pid": 99999999, "exit_code": None}}}, [])
    output = capsys.readouterr().out
    code_line = next(line for line in output.splitlines() if line.startswith("Code"))
    assert "Stopped" in code_line
    assert "Running" not in code_line


def test_metadata_uses_fallback_and_pinned_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "ACTIVE_OUT", tmp_path)
    monkeypatch.setenv("HF_ENDPOINT", "https://primary.example")
    monkeypatch.setenv("HF_FALLBACK_ENDPOINT", "https://fallback.example")
    source = collector.SOURCES["natural_en"]
    files = ["fineweb-edu-dedup/train-00001-of-00234.parquet", "fineweb-edu-dedup/train-00000-of-00234.parquet"]
    calls = []
    def metadata(request, timeout):
        calls.append(request.full_url)
        if request.full_url.startswith("https://primary.example"):
            raise OSError("TLS failure")
        return io.BytesIO(json.dumps({"sha": source["revision"], "siblings": [{"rfilename": p} for p in files]}).encode())
    monkeypatch.setattr(collector, "urlopen", metadata)
    result = collector.list_source_files("natural_en", retries=1)
    assert result == collector.stable_order(sorted(files))
    assert len(calls) == 2
    monkeypatch.setattr(collector, "urlopen", lambda *args, **kwargs: pytest.fail("cached metadata must not request network"))
    assert collector.list_source_files("natural_en") == result


def test_metadata_cannot_switch_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "ACTIVE_OUT", tmp_path)
    monkeypatch.setattr(collector, "urlopen", lambda *args, **kwargs: io.BytesIO(b'{"sha":"different","siblings":[]}'))
    with pytest.raises(ValueError, match="pinned revision"):
        collector.list_source_files("natural_en", retries=1)
    assert not (tmp_path / "natural_en/upstream_files.json").exists()


def test_completed_component_does_not_contact_upstream(monkeypatch):
    monkeypatch.setattr(collector, "completed_state", lambda component: (set(), collector.TARGETS[component] - 100))
    monkeypatch.setattr(collector, "list_source_files", lambda *args, **kwargs: pytest.fail("completed component must not contact upstream"))
    assert collector.collect_component("natural_en", None, None) == collector.TARGETS["natural_en"] - 100


def test_download_pause_flushes_and_retains_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "ACTIVE_OUT", tmp_path)
    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": "6"}

        def getcode(self):
            return self.status

        def read(self, size=-1):
            (tmp_path / ".pause_request").touch()
            return super().read(3)

    monkeypatch.setattr(collector, "urlopen", lambda *args, **kwargs: Response(b"abcdef"))
    partial, output = tmp_path / "data.part", tmp_path / "data"
    with pytest.raises(collector.CollectionPaused):
        collector._download_to("https://example.com/data", partial, 0, 1, "data", output)
    assert partial.read_bytes() == b"abc"
    assert not output.exists()


def test_pause_keeps_reports_and_raw_files(tmp_path, monkeypatch):
    from scripts.data_builder import start_coder_novel_12b as runner
    monkeypatch.setattr(runner, "active_collection_processes", lambda root: [])
    raw = tmp_path / "natural_en/raw/file.part"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"downloaded data")
    report = tmp_path / "natural_en/reports/file.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"rows": 2, "tokens": 100}), encoding="utf-8")
    original_report = report.read_bytes()
    status = runner.pause_collection(tmp_path, grace_seconds=0)
    assert status["phase"] == "paused"
    assert status["pid"] is None
    assert status["components"]["natural_en"]["raw_selected_tokens"] == 100
    assert status["components"]["natural_en"]["exit_code"] == 42
    assert raw.read_bytes() == b"downloaded data"
    assert report.read_bytes() == original_report
    assert (tmp_path / ".pause_request").exists()


def test_merged_index_excludes_both_prior_corpora_and_rejects_changes(tmp_path):
    base = make_index(tmp_path, ["segment one text", "shared text"])
    shard = tmp_path / "segment_two.jsonl"
    shard.write_text(''.join(json.dumps({"text": t}) + '\n' for t in ["segment two text", " shared text "]), encoding="utf-8")
    sources = [{"path": str(shard), "rows": 2, "sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}]
    target = tmp_path / "merged.blake2b128"
    report = build_merged_exclusion_index(base.path, sources, target)
    assert report["source_rows"] == 4
    assert report["fingerprints"] == 3
    merged = ExclusionIndex(target)
    assert merged.contains("segment one text")
    assert merged.contains("segment two text")
    assert merged.contains("\nshared text\n")
    assert not merged.contains("segment three text")
    assert build_merged_exclusion_index(base.path, sources, target) == report
    shard.write_text('changed', encoding="utf-8")
    with pytest.raises(ValueError, match="Baseline changed"):
        ExclusionIndex(target)


def test_merged_index_cannot_publish_a_corrupt_prior_shard(tmp_path):
    base = make_index(tmp_path, ["first"])
    shard = tmp_path / "second.jsonl"
    shard.write_text('{"text":"second"}\n', encoding="utf-8")
    target = tmp_path / "merged"
    with pytest.raises(ValueError, match="audit report"):
        build_merged_exclusion_index(base.path, [{"path": str(shard), "rows": 1, "sha256": "wrong"}], target)
    assert not target.exists()


def test_segment_three_does_not_change_segment_two_targets():
    from scripts.data_builder import start_coder_novel_12b as runner
    assert sum(runner.segment_collection_targets(2).values()) == 13_500_000_000
    assert sum(runner.segment_collection_targets(3).values()) == 13_100_000_000
    assert runner.segment_collection_targets(3)["code"] > runner.FINAL_TARGETS["code"]
    assert runner.segment_output_root(2) != runner.segment_output_root(3)
