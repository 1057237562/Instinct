"""Relay segment-1 boundary runs into segment-2 collection, component by component.

Segment 1 (dataset/coder_pretrain_12b) reproduces the shard boundary of the
already-trained first 12B corpus; its shard DATA is not needed once a component
reaches target (the per-shard reports stay as the skip evidence). When a
segment-1 component completes, this relay:

1. frees that component's segment-1 shard data (reports kept),
2. prefetches the first candidate shards of the component into segment 2's raw
   directory with parallel-range downloads,
3. runs the segment-2 collector for that component (which resumes/reruns
   safely; the collector itself refuses if the boundary is incomplete).

Segment 2's own components are serialized; each starts when its segment-1
counterpart is complete and the previous segment-2 component has finished.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from scripts.data_builder.collect_coder_pretrain_12b import (  # noqa: E402
    MAX_TOKENS, TARGETS, list_source_files, segment_out_root,
)
from scripts.data_builder.collect_arxiv_pretrain import hf_endpoint  # noqa: E402
from scripts.data_builder.fetch_parallel import download  # noqa: E402

COMPONENTS = ["code", "math", "natural_en", "natural_zh"]
SEG1 = segment_out_root(1)
SEG2 = segment_out_root(2)
STATE = SEG2 / ".relay_state.json"
PREFETCH_SHARDS = {"code": 10, "math": 8, "natural_en": 4, "natural_zh": 4}
POLL_SECONDS = 300


def component_tokens(root: Path, component: str) -> int:
    reports = root / component / "reports"
    tokens = 0
    if reports.is_dir():
        for path in reports.glob("*.json"):
            try:
                tokens += int(json.loads(path.read_text(encoding="utf-8"))["tokens"])
            except Exception:
                pass
    return tokens


def complete(root: Path, component: str) -> bool:
    return component_tokens(root, component) >= TARGETS[component] - MAX_TOKENS


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"inflight": None, "done": []}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def free_seg1_data(component: str) -> None:
    """Delete segment-1 shard data for a completed component (reports kept)."""
    shards = SEG1 / component / "shards"
    if shards.is_dir():
        for path in shards.glob("*.jsonl"):
            path.unlink()
    if component != "code":  # code raws still feed collect_stackv2_code_4096
        raw = SEG1 / component / "raw"
        if raw.is_dir():
            shutil.rmtree(raw, ignore_errors=True)


def prefetch_candidates(component: str) -> None:
    try:
        files = list_source_files(component)
    except Exception as error:
        print(f"[relay] candidate listing failed for {component}: {error}", flush=True)
        return
    skip = set()
    reports = SEG1 / component / "reports"
    if reports.is_dir():
        for path in reports.glob("*.json"):
            skip.add(json.loads(path.read_text(encoding="utf-8"))["upstream_file"])
    candidates = [name for name in files if name not in skip]
    if component == "code":
        # hardlinks are free, so take a wide window: every candidate whose raw
        # already sits in segment-1's raw dir gets reused, the rest are ignored
        candidates = candidates[: PREFETCH_SHARDS[component] * 2]
    else:
        candidates = candidates[: PREFETCH_SHARDS[component]]
    print(f"[relay] prefetching {len(candidates)} {component} shards for segment 2", flush=True)
    source = {  # mirrors collect_coder_pretrain_12b.SOURCES
        "code": ("common-pile/stackv2_edu_filtered", "c354dbe88469a1153e97c6a63ac50591849654de"),
        "math": ("HuggingFaceTB/finemath", "e92b25a616738fe95dc186b64dfb19f9c8525594"),
        "natural_en": ("HuggingFaceTB/smollm-corpus", "3ba9d605774198c5868892d7a8deda78031a781f"),
        "natural_zh": ("opencsg/chinese-fineweb-edu", "784f118864c0741ce32aa74d9588795bdf1a896b"),
    }[component]
    repo, revision = source
    raw = SEG2 / component / "raw"
    for name in candidates:
        out = raw / name
        if out.is_file() or out.with_suffix(out.suffix + ".ppart").exists():
            continue
        # reuse a segment-1 raw for the same upstream shard when one exists
        # (the 480M/ahead prefetches over-covered the code component): a
        # hardlink costs nothing and the collector then reads it locally.
        seg1_raw = SEG1 / component / "raw" / name
        if seg1_raw.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                out.hardlink_to(seg1_raw)
                print(f"[relay] {component}/{name}: hardlinked from segment-1 raw", flush=True)
                continue
            except OSError:
                pass  # cross-device or link limits: fall through to download
        url = f"{hf_endpoint()}/datasets/{repo}/resolve/{revision}/{name}?download=true"
        for attempt in range(1, 3):
            try:
                download(url, out, connections=8)
                break
            except Exception as error:
                print(f"[relay] prefetch {name} attempt {attempt}: {type(error).__name__}", flush=True)
                time.sleep(10)


def launch_collector(component: str) -> int:
    command = [sys.executable, str(ROOT / "scripts/data_builder/collect_coder_pretrain_12b.py"),
               "--component", component, "--segment", "2"]
    print(f"[relay] launching segment-2 collector for {component}", flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def main() -> None:
    state = load_state()
    running = {}  # component -> Popen we spawned
    print(f"[relay] state: {state}", flush=True)
    while True:
        progressed = False
        for component in COMPONENTS:
            if component in state["done"]:
                continue
            if not complete(SEG1, component):
                continue
            if complete(SEG2, component):
                state["done"].append(component)
                state["inflight"] = None
                save_state(state)
                running.pop(component, None)
                progressed = True
                continue
            if component in running and running[component].poll() is None:
                continue  # still collecting
            if component in running:
                code = running.pop(component).returncode
                print(f"[relay] segment-2 {component} collector exited code={code}", flush=True)
                if code == 0 and complete(SEG2, component):
                    state["done"].append(component)
                    save_state(state)
                progressed = True
                continue
            # adopt a collector left running by a previous relay process: skip
            # while its reports are fresh; relaunch once they go stale
            reports = SEG2 / component / "reports"
            newest = max((p.stat().st_mtime for p in reports.glob("*.json")), default=0)
            if time.time() - newest < 1800 and state.get("inflight") == component:
                continue
            state["inflight"] = component
            save_state(state)
            free_seg1_data(component)
            prefetch_candidates(component)
            print(f"[relay] launching segment-2 collector for {component}", flush=True)
            command = [sys.executable, str(ROOT / "scripts/data_builder/collect_coder_pretrain_12b.py"),
                       "--component", component, "--segment", "2"]
            running[component] = subprocess.Popen(command, cwd=ROOT)
            progressed = True
        if len(state["done"]) == len(COMPONENTS):
            print("[relay] all segment-2 components complete", flush=True)
            return
        if not progressed:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
