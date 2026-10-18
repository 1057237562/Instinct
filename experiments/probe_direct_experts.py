"""Compare direct FP32 parameter loads with BF16-stacked expert GEMMs."""
import sys,os,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import torch
from model.model_instinct import InstinctConfig,MOEFeedForward
from model.moe_dispatch import _grouped_expert_forward


def main():
    torch.set_num_threads(1);torch.manual_seed(61)
    config=InstinctConfig(**json.loads(Path('trainer/config_instinct_v1_moe.json').read_text()))
    moe=MOEFeedForward(config).cuda()
    rows=[]
    for tokens in (8192,11712):
        x=torch.randn(tokens,512,device='cuda',requires_grad=True)
        ids=torch.rand(tokens,8,device='cuda').topk(2,dim=-1).indices
        w=torch.rand(tokens,2,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        params=[getattr(e,n).weight for n in ('gate_proj','up_proj','down_proj') for e in moe.experts]
        dy=torch.randn_like(x)
        def execute():
            with torch.autocast('cuda',dtype=torch.bfloat16): y=_grouped_expert_forward(x,ids,w,moe.experts,moe.act_fn)
            grads=torch.autograd.grad(y,(x,w,*params),dy)
            return y,grads
        os.environ['INSTINCT_MOE_DIRECT_WEIGHTS']='0';ref,rg=execute()
        os.environ['INSTINCT_MOE_DIRECT_WEIGHTS']='1';got,gg=execute()
        row=dict(tokens=tokens,output_exact=torch.equal(ref,got),
                 gradient_exact=all(torch.equal(a,b) for a,b in zip(rg,gg)))
        assert row['output_exact'] and row['gradient_exact'],row
        for mode in ('0','1'):
            os.environ['INSTINCT_MOE_DIRECT_WEIGHTS']=mode
            for _ in range(3): execute()
            torch.cuda.synchronize(); start=time.perf_counter()
            for _ in range(20): execute()
            torch.cuda.synchronize(); row['ms_'+mode]=(time.perf_counter()-start)*1000/20
        rows.append(row);print(row,flush=True)
    Path('experiments/results/mfu_20261013/direct_experts.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
