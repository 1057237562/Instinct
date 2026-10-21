"""Isolation/copy/gradient checks and timing for the shared expert graph pool."""
import sys,os,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import torch
os.environ['INSTINCT_MOE_FUSED_ROUTING']='1'
from model.model_instinct import InstinctConfig,MOEFeedForward
from model.moe_dispatch import _grouped_expert_forward
from experiments.moe_training_graph_candidate import checkpointed_experts


def main():
    torch.set_num_threads(1);torch.manual_seed(61)
    config=InstinctConfig(**json.loads(Path('trainer/config_instinct_v1_moe.json').read_text()))
    moe=MOEFeedForward(config).cuda()
    x=torch.randn(8192,512,device='cuda',requires_grad=True)
    ids=torch.rand(8192,8,device='cuda').topk(2,dim=-1).indices
    w=torch.rand(8192,2,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    params=[getattr(e,n).weight for n in ('gate_proj','up_proj','down_proj') for e in moe.experts]
    dy=torch.randn_like(x)
    def execute(fn):
        with torch.autocast('cuda',dtype=torch.bfloat16): y=fn(x,ids,w,moe.experts,moe.act_fn)
        grads=torch.autograd.grad(y,(x,w,*params),dy)
        return y,grads
    rows=[]
    for cycle in range(3):
        ref,rg=execute(_grouped_expert_forward)
        got,gg=execute(checkpointed_experts)
        errors=[float((a-b).float().norm()/a.float().norm()) for a,b in zip(rg,gg)]
        row=dict(cycle=cycle,output_exact=torch.equal(ref,got),
                 output_rel=float((ref-got).norm()/ref.norm()),gradient_exact=all(torch.equal(a,b) for a,b in zip(rg,gg)),
                 gradient_relative_max=max(errors),errors=errors)
        print(json.dumps(row),flush=True); rows.append(row)
        assert row['output_exact'] and max(errors)<1e-6
        with torch.no_grad():
            for p in params: p.add_(.00001)
            x.add_(.1); w.mul_(.99); ids.copy_(torch.rand(8192,8,device='cuda').topk(2,dim=-1).indices)
    timings={}
    for label,fn in [('reference',_grouped_expert_forward),('graph',checkpointed_experts)]:
        for _ in range(3): execute(fn)
        torch.cuda.synchronize(); start=time.perf_counter()
        for _ in range(20): execute(fn)
        torch.cuda.synchronize(); timings[label]=(time.perf_counter()-start)*1000/20
    result=dict(checks=rows,milliseconds=timings)
    Path('experiments/results/mfu_20261013/graph_isolation.json').write_text(json.dumps(result,indent=2))
    print(timings,flush=True)


if __name__=='__main__': main()
