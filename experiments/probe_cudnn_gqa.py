"""Benchmark installed cuDNN compact-GQA attention with causal/packed masks."""
from experiments.probe_cudnn_attention import torch,sdpa_kernel,SDPBackend,Path,json
from model.packed_attention import _causal_flex_mask,build_packed_flex_mask,_flex_functions


def main():
    torch.set_num_threads(1);torch.manual_seed(11);rows=[]
    for batch,seq,packed in [(2,4096,False),(4,2928,True)]:
        q,k,v=[torch.randn(batch,seq,h,32,device='cuda',dtype=torch.bfloat16).transpose(1,2).detach().requires_grad_(True) for h in (16,4,4)]
        g=torch.randn_like(q)
        mask=None
        ids=(torch.arange(seq,device='cuda')//976)[None].expand(batch,-1)
        if packed:
            allowed=(ids[:,:,None]==ids[:,None,:])&torch.ones(seq,seq,device='cuda',dtype=torch.bool).tril()
            mask=torch.zeros(batch,1,seq,seq,device='cuda',dtype=torch.bfloat16).masked_fill_(~allowed[:,None],float('-inf'))
        blockmask=build_packed_flex_mask(ids).block_mask if packed else _causal_flex_mask(seq,'cuda:0')
        for backend in ('flex','cudnn'):
            try:
                def run():
                    if backend=='flex':
                        y=_flex_functions(True)[1](q,k,v,block_mask=blockmask,enable_gqa=True,kernel_options={'BACKEND':'TRITON'})
                    else:
                        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                            y=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=not packed,enable_gqa=True)
                    return torch.autograd.grad(y,(q,k,v),g)
                for _ in range(3):run()
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(20):run()
                b.record();b.synchronize()
                row=dict(batch=batch,seq=seq,packed=packed,backend=backend,ms=a.elapsed_time(b)/20)
            except Exception as exc:row=dict(batch=batch,seq=seq,packed=packed,backend=backend,error=str(exc))
            rows.append(row);print(row,flush=True)
    Path('experiments/results/mfu_20261013/cudnn_gqa.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
