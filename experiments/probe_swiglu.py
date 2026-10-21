import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datasets
import trainer.compile_cache
import torch
from model.swiglu_triton import swiglu
torch.manual_seed(73)
for scale in [0.1,1.,10.]:
 g=(torch.randn(8192,1664,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_()
 u=torch.randn_like(g).requires_grad_()
 d=torch.randn_like(g)
 a=swiglu(g,u)
 e=torch.nn.functional.silu(g)*u
 ag=torch.autograd.grad(a,(g,u),d)
 eg=torch.autograd.grad(e,(g,u),d)
 print(scale,[(torch.equal(x,y),(x!=y).float().mean().item()) for x,y in [(a,e),*zip(ag,eg)]],flush=True)
