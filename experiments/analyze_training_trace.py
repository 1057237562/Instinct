"""Summarize GPU kernels once, associating CUDA launches with trainer phases.

This is a timing analysis, not a hardware-counter bandwidth/occupancy measurement.
The trace is diagnostic; use a separate unprofiled run for end-to-end speed.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def summarize(path):
    events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
    phases = [e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation"
              and e.get("name", "").startswith("train::")]
    cpu_ops = {e["args"]["External id"]: e for e in events
               if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})}
    launches = {e.get("args", {}).get("correlation"): e for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver")
                and "correlation" in e.get("args", {})}
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    names = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    categories = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    spans = defaultdict(list)
    for event in kernels:
        launch = launches.get(event.get("args", {}).get("correlation"))
        containing = ([] if launch is None else [
            p for p in phases if p["ts"] <= launch["ts"] < p["ts"] + p["dur"]
            and p.get("pid") == launch.get("pid")])
        phase = min(containing, key=lambda p: p["dur"])["name"] if containing else "unattributed"
        name = event["name"]
        parent = cpu_ops.get(event.get("args", {}).get("External id"), {})
        if "flex" in name:
            category = "attention"
        elif (parent.get("name") == "aten::_scaled_mm" or "e4m3" in name
              or "e5m2" in name or "f8f8" in name):
            category = "FP8 GEMM"
        elif "_baddbmm" in name or (name == "_gemm" and phase == "train::optimizer"):
            category = "Muon NS Triton GEMM"
        elif any(term in name.lower() for term in ("gemm", "mma", "cublas", "cutlass")):
            category = "other GEMM"
        elif "triton_" in name:
            category = "fused pointwise/reduction"
        else:
            category = "other"
        for key, group in (((phase, name), names), ((phase, category), categories)):
            group[key]["calls"] += 1
            group[key]["gpu_ms"] += event["dur"] / 1000
        spans[phase].append((event["ts"], event["ts"] + event["dur"]))

    timing = {}
    for phase, intervals in spans.items():
        intervals.sort()
        end = intervals[0][0]
        busy = 0
        for start, stop in intervals:
            busy += max(0, stop - max(start, end))
            end = max(end, stop)
        elapsed = end - intervals[0][0]
        timing[phase] = dict(kernel_count=len(intervals), gpu_span_ms=elapsed / 1000,
                             kernel_busy_ms=busy / 1000, gaps_ms=(elapsed-busy) / 1000)
    operators = Counter()
    for event in events:
        if event.get("cat") == "cpu_op" and event.get("name") in (
                "aten::_scaled_mm", "aten::mm", "aten::bmm", "aten::baddbmm"):
            operators[(event["name"], json.dumps(event.get("args", {}).get("Input Dims")))] += 1
    return dict(trace=str(path.resolve()), kernel_count=len(kernels), phases=timing,
                categories=[dict(phase=p, category=n, **v) for (p, n), v in categories.items()],
                top_kernels=[dict(phase=p, name=n, **v) for (p, n), v in
                             sorted(names.items(), key=lambda kv: -kv[1]["gpu_ms"])[:40]],
                matmul_shapes=[dict(operator=n, input_dims=json.loads(s), calls=c)
                               for (n, s), c in operators.items()],
                caveats=["Profiler overhead can inflate gaps and kernel timings.",
                         "Categories are kernel-name heuristics; inspect top_kernels.",
                         "No hardware counters: DRAM bandwidth and occupancy are not measured."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.trace)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("top_kernels", "matmul_shapes")},
                     indent=2))
