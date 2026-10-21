import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datasets
import trainer.compile_cache
import torch
from trainer.muon_kernels import baddbmm
torch.manual_seed(17)
x=torch.randn(4,512,1664,device='cuda',dtype=torch.bfloat16)
x=x/torch.linalg.vector_norm(x,dim=(-2,-1),keepdim=True)
def compare(name,a,b):
 print(name, 'mismatch', (a!=b).float().mean().item(), 'rel', ((a.float()-b.float()).norm()/a.float().norm()).item(),flush=True)
for i in range(5):
 g=torch.bmm(x,x.transpose(-2,-1))
 compare('gram',g,baddbmm(x,x,x.transpose(-2,-1)))
 p=torch.baddbmm(g,g,g,beta=-4.775,alpha=2.0315)
 compare('poly',p,baddbmm(g,g,g,beta=-4.775,alpha=2.0315))
 y=torch.baddbmm(x,p,x,beta=3.4445)
 compare('update',y,baddbmm(x,p,x,beta=3.4445))
 x=y
