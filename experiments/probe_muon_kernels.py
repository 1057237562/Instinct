"""GEMM and full NS numerical/timing probe on actual expert shapes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets
import trainer.compile_cache
import torch
import json
from trainer.batched_muon import _batched_zeropower_reference as _batched_zeropower, _NS_COEFFICIENTS
from trainer.muon_kernels import zeropower, baddbmm
from experiments.probe_muon_graph import measure


torch.set_num_threads(1)
torch.manual_seed(17)
options = dict(ns_coefficients=_NS_COEFFICIENTS, ns_steps=5, eps=1e-7)
rows = []
for shape in [(4, 1664, 512), (4, 512, 1664), (10, 512, 512), (16, 8, 512)]:
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    ref = _batched_zeropower(x, **options)
    y = zeropower(x, **options)
    error = ((ref.float()-y.float()).norm()/ref.float().norm()).item()
    for _ in range(5):
        zeropower(x, **options)
    row = dict(shape=shape, error=error, exact=torch.equal(ref,y),
               eager=measure(lambda: _batched_zeropower(x, **options)),
               triton=measure(lambda: zeropower(x, **options)))
    rows.append(row)
    print(json.dumps(row), flush=True)
Path('experiments/results/mfu_20261012/muon_kernel_probe.json').write_text(json.dumps(rows, indent=2))
