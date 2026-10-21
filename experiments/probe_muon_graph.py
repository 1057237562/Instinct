"""Isolated fixed-LR Muon chunk probe; no model/checkpoint or production edits.

This measures a reusable graph per chunk shape, not a complete optimizer.
Production integration must separately solve dynamic LR, grad=None and pointers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: F401 -- Windows DLL ordering
import trainer.compile_cache  # noqa: F401
import json
import os
import statistics
import time
import torch
from trainer.batched_muon import BatchedMuon


def measure(fn):
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / 20)
    return {"median_ms": statistics.median(samples), "windows_ms": samples}


def probe(shape, count):
    torch.manual_seed(7)
    params = [torch.nn.Parameter(torch.randn(shape, device='cuda')) for _ in range(count)]
    for p in params:
        p.grad = torch.randn_like(p)
    opt = BatchedMuon(params, lr=5e-4)
    group = opt.param_groups[0]
    @torch.no_grad()
    def step():
        opt._step_chunk(params, group)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    eager = measure(step)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    # Each replay must match the *next* eager update including momentum.
    exact = True
    for _ in range(3):
        for p in params:
            p.grad.copy_(torch.randn_like(p))
        before = [(p.detach().clone(), opt.state[p]['momentum_buffer'].clone()) for p in params]
        step()
        expected = [(p.detach().clone(), opt.state[p]['momentum_buffer'].clone()) for p in params]
        with torch.no_grad():
            for p, (weight, momentum) in zip(params, before):
                p.copy_(weight)
                opt.state[p]['momentum_buffer'].copy_(momentum)
        graph.replay()
        exact &= all(torch.equal(p, w) and torch.equal(opt.state[p]['momentum_buffer'], m)
                     for p, (w, m) in zip(params, expected))
    graphed = measure(graph.replay)
    eager_after = measure(step)
    result = dict(shape=shape, count=count, eager=eager, graph=graphed,
                  eager_after=eager_after,
                  baseline_drift_pct=100*(eager_after['median_ms']/eager['median_ms']-1),
                  bitwise_next_update_and_momentum=exact)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    os.environ['INSTINCT_MUON_NS_BACKEND'] = 'torch'
    os.environ['INSTINCT_MUON_FUSED_UPDATES'] = '0'
    results = [probe(shape, count) for shape, count in [
        ((1664, 512), 4), ((512, 1664), 4), ((512, 512), 10),
        ((512, 512), 4), ((128, 512), 16), ((8, 512), 16),
        ((1664, 512), 2), ((512, 1664), 2),
        ((1664, 512), 8), ((512, 1664), 8)]]
    counts = [128, 64, 6, 1, 4, 2]
    totals = {kind: sum(n * r[kind]['median_ms'] for n, r in zip(counts, results))
              for kind in ('eager', 'graph', 'eager_after')}
    output = Path('experiments/results/mfu_20261012/muon_graph_probe.json')
    output.write_text(json.dumps(dict(
        torch_version=torch.__version__, device=torch.cuda.get_device_name(),
        scope='chunk-only projection excludes dispatch, gradient staging, LR, clipping, AdamW and zero_grad',
        cases=results, chunk_counts_first_six=counts, projected_chunk_total_ms=totals), indent=2), encoding='utf-8')
    print(totals)
