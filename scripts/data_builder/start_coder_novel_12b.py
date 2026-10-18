"""Collect a new 12B coder segment in four resumable background workers.

This starts collection, not training or assembly. Raw shards and contextual
identity-review candidates are retained; a reviewed, cross-shard-deduplicated
clean release must be produced separately before training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data_builder.coder_novelty import build_exclusion_index, build_merged_exclusion_index


FINAL_TARGETS = {"code": 7_200_000_000, "math": 1_200_000_000,
                 "natural_en": 2_400_000_000, "natural_zh": 1_200_000_000}
DEFAULT_COLLECTION_TARGETS = {"code": 8_000_000_000, "math": 1_400_000_000,
                              "natural_en": 2_700_000_000, "natural_zh": 1_400_000_000}
COLLECTION_TARGETS = dict(DEFAULT_COLLECTION_TARGETS)


def segment_output_root(segment):
    return ROOT / f"dataset/coder_pretrain_12b_seg{segment}_novel_raw"


def segment_collection_targets(segment):
    return {**DEFAULT_COLLECTION_TARGETS, "code": 7_600_000_000 if segment == 3 else 8_000_000_000}


def prior_segment_inputs(segment):
    roots = [ROOT / "dataset/coder_pretrain_12b"]
    additions = []
    if segment == 3:
        prior = segment_output_root(2)
        roots.append(prior)
        if active_collection_processes(prior):
            raise RuntimeError("Segment 2 must finish before segment 3 can freeze its prior-corpus manifest")
        for component, target in DEFAULT_COLLECTION_TARGETS.items():
            if progress(prior, component)["raw_selected_tokens"] < target - 4096:
                raise RuntimeError(f"Segment 2 is incomplete for {component}")
            for report_path in sorted((prior / component / "reports").glob("*.json")):
                report = json.loads(report_path.read_text(encoding="utf-8"))
                path = Path(report["output"]).resolve()
                if not path.is_relative_to((prior / component / "shards").resolve()):
                    raise ValueError(f"Prior report points outside its shard directory: {report_path}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                additions.append({"path": str(path), "rows": report["rows"], "sha256": report["sha256"],
                                  "source_report": str(report_path.resolve()), "upstream_repo": report["upstream_repo"],
                                  "upstream_revision": report["upstream_revision"], "upstream_file": report["upstream_file"]})
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    return roots, additions


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def collection_environment():
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false")
    env["HF_ENDPOINT"] = os.environ.get("INSTINCT_NOVEL_HF_ENDPOINT", "https://huggingface.co")
    env["HF_FALLBACK_ENDPOINT"] = os.environ.get("INSTINCT_NOVEL_HF_FALLBACK_ENDPOINT", "https://aifasthub.com")
    return env


def launch_windows_independent(command, root):
    """WMI creates the process outside the caller's Windows Job.

    CREATE_NO_WINDOW/CREATE_NEW_PROCESS_GROUP alone still inherit the app's
    Job and therefore die when the app closes it. WMI's process service owns
    this launch; the child redirects its own logs and sets its environment.
    """
    def quote_ps(value):
        return "'" + str(value).replace("'", "''") + "'"

    script = root / "logs/launch_background.ps1"
    script.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
        "$startup = ([wmiclass]'Win32_ProcessStartup').CreateInstance()\n"
        "$startup.ShowWindow = [uint16]0\n"
        "$startup.EnvironmentVariables = [string[]](ConvertFrom-Json -InputObject ([Console]::In.ReadToEnd()))\n"
        "$processClass = [wmiclass]'Win32_Process'\n"
        f"$result = $processClass.Create({quote_ps(subprocess.list2cmdline(command))}, {quote_ps(ROOT)}, $startup)\n"
        "if ($result.ReturnValue -ne 0) { throw ('WMI process creation failed: ' + $result.ReturnValue) }\n"
        "@{pid = [int]$result.ProcessId; launch_method = 'windows_wmi'} | ConvertTo-Json -Compress\n",
        encoding="utf-8-sig",
    )
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(script)],
                            input=json.dumps([f"{key}={value}" for key, value in collection_environment().items()]),
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"Independent Windows launch failed: {result.stderr.strip()}")
    return json.loads(result.stdout.strip())


def progress(root, component):
    rows = tokens = candidates = excluded = 0
    files = sorted((root / component / "reports").glob("*.json"))
    for path in files:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows += report["rows"]
        tokens += report["tokens"]
        candidates += report.get("identity_review", {}).get("candidate_rows", 0)
        excluded += report.get("selection", {}).get("rejected_prior_corpus_text", 0)
    return {"completed_shards": len(files), "rows": rows, "raw_selected_tokens": tokens,
            "review_candidate_rows": candidates, "excluded_prior_corpus_rows": excluded}


def acquire_run_lock(root):
    lock = (root / "run.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    return lock


def active_collection_processes(root):
    """Match the script and output directory, never just a possibly reused PID."""
    expected = {str((ROOT / "scripts/data_builder" / name).resolve()).casefold(): role
                for name, role in (("start_coder_novel_12b.py", "supervisor"), ("collect_coder_pretrain_12b.py", "collector"))}
    matches = []
    for process in psutil.process_iter(["pid", "cmdline", "cwd", "status", "create_time"]):
        try:
            info = process.info
            if info["pid"] == os.getpid() or info["status"] == psutil.STATUS_ZOMBIE:
                continue
            command = info["cmdline"] or []
            if "--background" in command:
                continue
            cwd = Path(info["cwd"] or ROOT)
            role = next((expected[str((cwd / arg).resolve()).casefold()] for arg in command[1:]
                         if str((cwd / arg).resolve()).casefold() in expected), None)
            if role is None:
                continue
            if "--out-dir" in command:
                output = (cwd / command[command.index("--out-dir") + 1]).resolve()
            elif role == "supervisor":
                segment = int(command[command.index("--segment") + 1]) if "--segment" in command else 2
                output = segment_output_root(segment).resolve()
            else:
                continue
            if output != root.resolve():
                continue
            component = command[command.index("--component") + 1] if "--component" in command else None
            matches.append({"pid": info["pid"], "role": role, "component": component, "create_time": info["create_time"]})
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError, IndexError):
            continue
    return matches


def stop_collection_processes(root):
    matches = active_collection_processes(root)
    # Stop the supervisor before its children so it cannot launch more work.
    for match in sorted(matches, key=lambda item: item["role"] != "supervisor"):
        try:
            process = psutil.Process(match["pid"])
            if process.create_time() != match["create_time"]:
                continue
            process.terminate()
            process.wait(timeout=10)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    return matches


def pause_collection(root, grace_seconds=30):
    (root / ".pause_request").touch()
    print("Pause requested. Waiting for safe download/processing boundaries...", flush=True)
    deadline = time.monotonic() + grace_seconds
    live = active_collection_processes(root)
    while live and time.monotonic() < deadline:
        time.sleep(1)
        live = active_collection_processes(root)
    if live:
        print("Stopping older or blocked task workers; existing raw partial files are retained.", flush=True)
        stop_collection_processes(root)
    if active_collection_processes(root):
        raise RuntimeError("Task workers are still alive; pause has not completed")
    lock = acquire_run_lock(root)
    try:
        path = root / "status.json"
        status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        status.update(phase="paused", pid=None, updated_at_utc=datetime.now(timezone.utc).isoformat())
        status.setdefault("components", {})
        for component, target in COLLECTION_TARGETS.items():
            current = progress(root, component)
            status["components"].setdefault(component, {}).update(current)
            status["components"][component].update(pid=None, collection_target_tokens=target,
                                                   exit_code=0 if current["raw_selected_tokens"] >= target - 4096 else 42)
        write_json(path, status)
    finally:
        lock.close()
    print("Paused. Run the BAT normally to resume; do not delete raw .part files.", flush=True)
    return status


def print_running_progress(root, status, live_processes=None):
    live_processes = active_collection_processes(root) if live_processes is None else live_processes
    supervisors = [item["pid"] for item in live_processes if item["role"] == "supervisor"]
    live_components = {item["component"]: item["pid"] for item in live_processes if item["role"] == "collector"}
    print(f"Collection active. Supervisor PID: {supervisors or 'none'}")
    print(f"Verified live process PIDs: {[item['pid'] for item in live_processes]}")
    print(f"Phase: {status.get('phase', 'starting')}")
    if status.get("updated_at_utc"):
        updated = datetime.fromisoformat(status["updated_at_utc"]).astimezone(timezone(timedelta(hours=8)))
        print(f"Worker status updated: {updated:%Y-%m-%d %H:%M:%S} UTC+08:00")
    print()
    print(f"{'Component':<12} {'Collected / target (B tokens)':>30} {'Progress':>10} {'Shards':>7}  State")
    total = target_total = review_total = excluded_total = 0
    labels = {"code": "Code", "math": "Math", "natural_en": "English", "natural_zh": "Chinese"}
    for component, default_target in COLLECTION_TARGETS.items():
        snapshot = status.get("components", {}).get(component, {})
        current = progress(root, component)
        tokens = current["raw_selected_tokens"]
        target = snapshot.get("collection_target_tokens", default_target)
        code = snapshot.get("exit_code")
        state = "Done" if tokens >= target - 4096 else "Running" if component in live_components else f"Failed ({code})" if code not in (None, 0) else "Waiting" if status.get("phase") == "indexing_baseline" else "Stopped"
        amounts = f"{tokens / 1e9:.3f} / {target / 1e9:.3f}"
        print(f"{labels[component]:<12} {amounts:>30} {tokens / target:>9.1%} {current['completed_shards']:>7}  {state}")
        total += tokens
        target_total += target
        review_total += current["review_candidate_rows"]
        excluded_total += current["excluded_prior_corpus_rows"]
    print(f"\nTotal collected: {total / 1e9:.3f}B / {target_total / 1e9:.3f}B ({total / target_total:.1%})")
    print(f"Prior-corpus rows excluded: {excluded_total:,}; review candidates: {review_total:,}")
    print("Counts include completed raw shards only. Final clean release target: 12B tokens.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=int, choices=(2, 3), default=2,
                        help="Collect segment 2 or segment 3; segment 3 excludes both earlier corpora.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=30)
    parser.add_argument("--background", action="store_true", help="Launch a detached, hidden worker with persistent logs.")
    parser.add_argument("--log-to-files", action="store_true", help="Redirect logs in the independently launched worker.")
    parser.add_argument("--restart", action="store_true", help="Stop only matching collection processes, then resume their completed shards.")
    parser.add_argument("--pause", action="store_true", help="Pause the collection while retaining completed shards and resumable downloads.")
    args = parser.parse_args()
    global COLLECTION_TARGETS
    COLLECTION_TARGETS = segment_collection_targets(args.segment)
    root = args.out_dir.resolve() if args.out_dir else segment_output_root(args.segment).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.pause:
        if args.background or args.restart:
            parser.error("--pause cannot be combined with --background or --restart")
        pause_collection(root)
        return
    if args.restart and not args.background:
        parser.error("--restart requires --background")
    if args.background:
        if args.restart:
            stopped = stop_collection_processes(root)
            print(f"Restart: stopped matching task PIDs {[item['pid'] for item in stopped]}; completed shards retained.", flush=True)
        else:
            live = active_collection_processes(root)
            if live:
                status_path = root / "status.json"
                status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
                print_running_progress(root, status, live)
                return
        try:
            probe = acquire_run_lock(root)
        except OSError as error:
            if error.errno not in (11, 13):
                raise
            status_path = root / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            live = active_collection_processes(root)
            if live:
                print_running_progress(root, status, live)
                return
            raise RuntimeError("The lock cannot be acquired, but no matching task is alive. Check run.lock file access; a stale status PID is not proof of a running task.") from error
        else:
            probe.close()
        (root / ".pause_request").unlink(missing_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--out-dir", str(root),
                   "--segment", str(args.segment), "--timeout", str(args.timeout), "--retries", str(args.retries), "--log-to-files"]
        if os.name == "nt":
            result = launch_windows_independent(command, root)
        else:
            process = subprocess.Popen(command, cwd=ROOT, env=collection_environment(), stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            result = {"pid": process.pid, "launch_method": "posix_new_session"}
        print(json.dumps({**result, "output_dir": str(root), "log": str(root / "logs/run.log")}), flush=True)
        return
    if args.log_to_files:
        (root / "logs").mkdir(exist_ok=True)
        sys.stdout = (root / "logs/run.log").open("a", encoding="utf-8", buffering=1)
        sys.stderr = (root / "logs/run.stderr.log").open("a", encoding="utf-8", buffering=1)
    lock = acquire_run_lock(root)
    baseline = ROOT / "dataset/pretrain_coder_12b.jsonl"
    baseline_report = ROOT / "dataset/pretrain_coder_12b.report.json"
    index = root / "baseline_texts.blake2b128"
    prior_roots, additional_sources = prior_segment_inputs(args.segment)
    config = {"baseline": str(baseline), "baseline_report": str(baseline_report),
              "prior_shard_reports": str(ROOT / "dataset/coder_pretrain_12b"),
              "final_targets": FINAL_TARGETS, "collection_targets": COLLECTION_TARGETS,
              "replay_tokens": 0, "training_ready": False,
              "review_policy": "Retain raw rows and all identity keyword hits for contextual review; do not automatically delete candidates.",
              "novelty_policy": "Exclude prior released corpus texts; skip its collector shards; cross-shard deduplication and review required before final release."}
    if args.segment == 3:
        config.update(segment=3, prior_shard_roots=[str(path) for path in prior_roots],
                      additional_prior_sources=additional_sources)
    config_path = root / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Run configuration changed; select a new output directory")
    write_json(config_path, config)
    status_path = root / "status.json"
    status = {"phase": "indexing_baseline", "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "output_dir": str(root), "training_ready": False, "components": {}}
    write_json(status_path, status)
    processes = {}
    try:
        if args.segment == 3:
            base_index = segment_output_root(2) / "baseline_texts.blake2b128"
            if not base_index.is_file() or not base_index.with_suffix(base_index.suffix + ".report.json").is_file():
                base_index = root / "first_segment.blake2b128"
                build_exclusion_index(baseline, base_index, baseline_report)
            baseline_index = build_merged_exclusion_index(base_index, additional_sources, index)
        else:
            baseline_index = build_exclusion_index(baseline, index, baseline_report)
        status["baseline_sha256"] = baseline_index["source_sha256"]
        status["phase"] = "collecting"
        (root / "logs").mkdir(exist_ok=True)
        env = collection_environment()
        for component, target in COLLECTION_TARGETS.items():
            if (root / ".pause_request").exists():
                break
            command = [sys.executable, "-u", str(ROOT / "scripts/data_builder/collect_coder_pretrain_12b.py"),
                       "--component", component, "--segment", "1", "--out-dir", str(root),
                       "--target-tokens", str(target), "--novel-index", str(index), "--identity-review",
                       "--timeout", str(args.timeout), "--retries", str(args.retries)]
            for prior in prior_roots:
                command.extend(["--skip-reports-from", str(prior)])
            log_path = root / "logs" / f"{component}.log"
            with log_path.open("ab", buffering=0) as log:
                processes[component] = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            status["components"][component] = {"pid": processes[component].pid, "log": str(log_path), "collection_target_tokens": target}
            print(f"[collect] {component}: pid={processes[component].pid}, target={target:,}, log={log_path}", flush=True)
        write_json(status_path, status)
        while True:
            running = False
            failed = False
            paused = (root / ".pause_request").exists()
            for component, process in processes.items():
                code = process.poll()
                running |= code is None
                failed |= code is not None and code not in (0, 42)
                paused |= code == 42
                status["components"][component].update(progress(root, component))
                status["components"][component]["exit_code"] = code
            status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            status["phase"] = ("pausing" if running else "paused") if paused else "collecting_with_failed_component" if running and failed else "collecting" if running else "collection_failed" if failed else "collected_pending_review"
            write_json(status_path, status)
            if not running:
                if paused:
                    print("[collect] paused; completed shards and raw partials retained", flush=True)
                    break
                if failed:
                    raise RuntimeError("A collector failed; see component logs and rerun to resume")
                print("[collect] done; raw shards require review and final deduplicated assembly", flush=True)
                break
            time.sleep(15)
    except BaseException as error:
        # A controlled interruption stops children so restarting cannot overlap workers.
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        status["phase"] = "failed"
        status["error"] = str(error)
        write_json(status_path, status)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
