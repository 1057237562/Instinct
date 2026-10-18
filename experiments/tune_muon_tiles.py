"""Offline tile probe; changing tile is gated against the existing NS result."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datasets
import trainer.compile_cache
import torch, json, gc
import trainer.muon_kernels as kernels
from trainer.batched_muon import _batched_zeropower_reference, _NS_COEFFICIENTS
from experiments.probe_muon_graph import measure
torch.set_num_threads(1)
torch.manual_seed(17)
x = torch.randn(4,1664,512,device='cuda',dtype=torch.bfloat16)
opts=dict(ns_coefficients=_NS_COEFFICIENTS, ns_steps=5, eps=1e-7)
ref = _batched_zeropower_reference(x,**opts)
rows=[]
for tile in [(64,64,32,4,3),(32,64,32,4,3),(64,128,32,4,3),
             (64,64,64,4,3),(64,128,64,4,3),(128,64,32,4,3),
             (128,128,32,8,3),(32,128,64,4,3)]:
 kernels._TILE=tile
 cache=kernels.ZeropowerGraphCache()
 y=cache(x,**opts)
 row=dict(tile=tile, exact=torch.equal(ref,y),
          error=((ref.float()-y.float()).norm()/ref.float().norm()).item(),
          timing=measure(lambda:cache(x,**opts)))
 print(json.dumps(row),flush=True)
 rows.append(row)
 del cache
 gc.collect()
Path('experiments/results/mfu_20261012/muon_tiles.json').write_text(json.dumps(rows,indent=2))
