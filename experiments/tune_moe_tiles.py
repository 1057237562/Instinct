"""Tune BF16 MoE GEMMs; compare every candidate with original device kernels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datasets
import trainer.compile_cache
import torch,json
import model.grouped_mm_triton as kernels
from experiments.probe_muon_graph import measure
torch.set_num_threads(1)
torch.manual_seed(17)
ends=torch.arange(1,9,device='cuda',dtype=torch.int32)*1024
inputs=[]
for k,n,transposed in [(512,1664,True),(1664,512,True),(1664,512,False),(512,1664,False)]:
 x=torch.randn(8192,k,device='cuda',dtype=torch.bfloat16)
 w=torch.randn(8,n,k,device='cuda',dtype=torch.bfloat16).transpose(1,2) if transposed else torch.randn(8,k,n,device='cuda',dtype=torch.bfloat16)
 inputs.append((x,w,kernels.ragged_mm(x,w,ends)))
dw=[]
for k,n in [(512,1664),(1664,512)]:
 x=torch.randn(8192,k,device='cuda',dtype=torch.bfloat16)
 dy=torch.randn(8192,n,device='cuda',dtype=torch.bfloat16)
 dw.append((x,dy,kernels.ragged_weight_grad(x,dy,ends,8)))
rows=[]
for tile in [(64,64,32,4,3),(128,64,32,4,3),(64,128,32,4,3),
             (128,128,32,8,3),(64,64,64,4,3),(64,128,64,4,3)]:
 kernels._FORWARD_TILE=tile
 kernels._WEIGHT_TILE=tile
 cases=[]
 for x,w,ref in inputs:
  y=kernels.ragged_mm(x,w,ends)
  cases.append(dict(exact=torch.equal(y,ref),ms=measure(lambda:kernels.ragged_mm(x,w,ends))['median_ms']))
 for x,dy,ref in dw:
  y=kernels.ragged_weight_grad(x,dy,ends,8)
  cases.append(dict(exact=torch.equal(y,ref),ms=measure(lambda:kernels.ragged_weight_grad(x,dy,ends,8))['median_ms']))
 row=dict(tile=tile,cases=cases,projected_ms_per_step=32*sum(c['ms']*n for c,n in zip(cases,[4,2,2,1,2,1])))
 print(json.dumps(row),flush=True)
 rows.append(row)
Path('experiments/results/mfu_20261012/moe_tiles.json').write_text(json.dumps(rows,indent=2))
