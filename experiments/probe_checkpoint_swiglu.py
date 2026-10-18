"""Compare operator-local checkpointing with the original gate/up replay."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import torch
from model.grouped_mm import expert_grouped_mm,make_grouped_mm_plan
from model.expert_weight_pack import pack_weights
from model.swiglu_triton import swiglu
from model.checkpoint_swiglu_triton import checkpoint_swiglu
from torch.utils.checkpoint import checkpoint


def main():
    torch.set_num_threads(1);torch.manual_seed(52);rows=[]
    for routes in (16384,23424):
        x=torch.randn(routes,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        ends=torch.arange(1,9,device='cuda',dtype=torch.int32)*(routes//8)
        weights=[torch.randn(1664,512,device='cuda',requires_grad=True)*.02 for _ in range(16)]
        weights=[t.detach().requires_grad_(True) for t in weights]
        g,u=weights[:8],weights[8:]
        dy=torch.randn(routes,1664,device='cuda',dtype=torch.bfloat16)
        def old(h):
            plan=make_grouped_mm_plan(ends)
            return swiglu(expert_grouped_mm(h,pack_weights(g,torch.bfloat16),plan),
                          expert_grouped_mm(h,pack_weights(u,torch.bfloat16),plan))
        def execute(candidate):
            y=checkpoint_swiglu(x,g,u,ends) if candidate else checkpoint(old,x,use_reentrant=False)
            return y,torch.autograd.grad(y,(x,*weights),dy)
        ref,rg=execute(False);got,gg=execute(True)
        row=dict(routes=routes,output_exact=torch.equal(ref,got),
                 gradient_exact=all(torch.equal(a,b) for a,b in zip(rg,gg)),
                 output_relative=float((ref.detach().float()-got.detach().float()).norm()/ref.detach().float().norm()),
                 gradient_relative_max=max(float((a.float()-b.float()).norm()/a.float().norm()) for a,b in zip(rg,gg)))
        assert row['output_exact'] and row['gradient_exact'],row
        for candidate in (False,True):
            for _ in range(3):execute(candidate)
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(20):execute(candidate)
            torch.cuda.synchronize();row['candidate_ms' if candidate else 'reference_ms']=(time.perf_counter()-start)*1000/20
        print(row,flush=True);rows.append(row)
    Path('experiments/results/mfu_20261013/checkpoint_swiglu.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
