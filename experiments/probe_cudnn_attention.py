"""Check installed cuDNN SDPA, including zero-padded head dimensions."""
import sys,math,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import torch,triton
from torch.nn.attention import sdpa_kernel,SDPBackend


def main():
    torch.set_num_threads(1);torch.manual_seed(11)
    rows=[]
    for dim in (32,64):
        q,k,v=[torch.randn(2,4096,16,32,device='cuda',dtype=torch.bfloat16).transpose(1,2).detach().requires_grad_(True) for _ in range(3)]
        g=torch.randn_like(q)
        for backend in (SDPBackend.EFFICIENT_ATTENTION,SDPBackend.CUDNN_ATTENTION):
            try:
                def run():
                    with sdpa_kernel(backend):
                        inputs=[torch.nn.functional.pad(t,(0,dim-32)) if dim!=32 else t for t in (q,k,v)]
                        y=torch.nn.functional.scaled_dot_product_attention(*inputs,is_causal=True,scale=1/math.sqrt(32))[...,:32]
                        return torch.autograd.grad(y,(q,k,v),g)
                run()
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(20): run()
                b.record();b.synchronize()
                row=dict(dim=dim,backend=str(backend),ms=a.elapsed_time(b)/20)
            except Exception as exc:row=dict(dim=dim,backend=str(backend),error=str(exc))
            rows.append(row);print(row,flush=True)
    Path('experiments/results/mfu_20261013/cudnn_attention.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
