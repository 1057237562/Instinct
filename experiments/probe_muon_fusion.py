import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datasets
import trainer.compile_cache
import torch
from trainer.muon_kernels import prepare_update,apply_update
torch.manual_seed(17)
for mom in [0.,0.5,0.8,0.95,1.]:
 for nesterov in [True,False]:
  grads=[torch.randn(1664,512,device='cuda') for _ in range(4)]
  buffers=[torch.randn_like(t) for t in grads]
  expected=[t.clone() for t in buffers]
  torch._foreach_lerp_(expected,grads,1-mom)
  u=torch.stack(torch._foreach_lerp(grads,expected,mom) if nesterov else expected).bfloat16()
  got=prepare_update(grads,buffers,momentum=mom,nesterov=nesterov)
  print(mom,nesterov,'momentum',all(torch.equal(a,b) for a,b in zip(expected,buffers)), 'update',torch.equal(u,got),flush=True)
  params=[torch.randn_like(t) for t in grads]
  ref=[t.clone() for t in params]
  torch._foreach_mul_(ref,0.99995)
  torch._foreach_add_(ref,list(u.float().unbind()),alpha=-0.0012345)
  apply_update(params,got,decay=0.99995,alpha=-0.0012345)
  print('apply',all(torch.equal(a,b) for a,b in zip(params,ref)),flush=True)
