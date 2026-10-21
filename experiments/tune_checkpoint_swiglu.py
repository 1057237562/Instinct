"""Tune the operator-local recomputation kernel, including its backward."""
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import torch
from model.checkpoint_swiglu_triton import checkpoint_swiglu


def main():
    torch.set_num_threads(1);torch.manual_seed(731)
    reports=[]
    for routes in (16384,23424):
        x=torch.randn(routes,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        ends=torch.arange(1,9,device='cuda',dtype=torch.int32)*(routes//8)
        weights=[(torch.randn(1664,512,device='cuda')*.02).requires_grad_() for _ in range(16)]
        dy=torch.randn(routes,1664,device='cuda',dtype=torch.bfloat16)
        def run():
            y=checkpoint_swiglu(x,weights[:8],weights[8:],ends)
            return y,torch.autograd.grad(y,(x,*weights),dy)
        os.environ['INSTINCT_MOE_CHECKPOINT_TILE']='64,64,32,4,3'
        ref,rg=run()
        for tile in ('64,64,32,4,3','64,64,64,4,3','64,64,128,4,3','32,64,64,4,3',
                     '32,128,64,4,3','64,128,32,8,3','64,128,64,8,3','32,64,128,4,3',
                     '128,64,64,8,3','32,128,128,8,3','64,64,64,4,4'):
            os.environ['INSTINCT_MOE_CHECKPOINT_TILE']=tile
            row=dict(routes=routes,tile=tile)
            try:
                got,gg=run()
                row['output_exact']=torch.equal(ref,got)
                row['gradients_exact']=all(torch.equal(a,b) for a,b in zip(rg,gg))
                for _ in range(3):run()
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(20):run()
                torch.cuda.synchronize();row['ms']=(time.perf_counter()-start)*50
            except Exception as exc:
                row['error']=str(exc)[:500]
            reports.append(row);print(row,flush=True)
            Path('experiments/results/mfu_20261013/checkpoint_tiles.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':main()
