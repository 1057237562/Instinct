"""Exact permutation and isolated floating point parity for fused routing."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import torch
import json
from model.moe_routing_triton import permutation,gather,combine


def main():
    torch.manual_seed(171); rows=[]
    for tokens in (17,8192,11712):
        for top in (1,2):
            for skew in (False,True):
                ids=torch.rand(tokens,8,device='cuda').topk(top,dim=-1).indices
                if skew: ids=torch.arange(top,device='cuda').expand(tokens,-1).contiguous()
                order,inv,ends=permutation(ids,8)
                reforder=ids.flatten().argsort(stable=True)
                assert torch.equal(order.long(),reforder)
                assert torch.equal(inv[order.long()],torch.arange(tokens*top,device='cuda',dtype=torch.int32))
                assert torch.equal(ends,torch.bincount(ids.flatten(),minlength=8).cumsum(0))
                route_token=reforder//top
                x=torch.randn(tokens,512,device='cuda',requires_grad=True)
                xx=x.detach().clone().requires_grad_(True)
                actual=gather(x,order,inv,top)
                expected=xx[route_token].bfloat16()
                assert torch.equal(actual,expected)
                grad=torch.randn_like(actual)
                actual.backward(grad); expected.backward(grad)
                assert torch.equal(x.grad,xx.grad)
                y=torch.randn(tokens*top,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
                w=torch.rand(tokens,top,device='cuda',dtype=torch.bfloat16,requires_grad=True)
                yy=y.detach().clone().requires_grad_(True); ww=w.detach().clone().requires_grad_(True)
                actual=combine(y,w,order,inv,torch.float32)
                weighted=(yy*ww.flatten()[reforder,None]).float()
                expected=torch.zeros(tokens,512,device='cuda').index_add_(0,route_token,weighted)
                assert torch.equal(actual,expected)
                grad=torch.randn_like(actual)
                actual.backward(grad); expected.backward(grad)
                assert torch.equal(y.grad,yy.grad)
                relative=float((w.grad.float()-ww.grad.float()).norm()/ww.grad.float().norm())
                assert relative < 1e-5
                rows.append(dict(tokens=tokens,top=top,skew=skew,
                                 forward_dx_dy_exact=True,weight_grad_relative_l2=relative))
                print(rows[-1],flush=True)
    Path('experiments/results/mfu_20261013/routing_parity.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__': main()
