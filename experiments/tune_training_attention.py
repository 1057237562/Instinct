"""Tune the real GQA attention dimensions, checking output/gradient drift."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import torch,triton
from model.packed_attention import build_packed_flex_mask,_causal_flex_mask
from torch.nn.attention.flex_attention import flex_attention


def main():
    torch.set_num_threads(1);torch.manual_seed(131)
    torch.autograd.graph.set_override_stale_capture_stream(True)
    original=torch.compile(flex_attention,fullgraph=True)
    tuned=torch.compile(flex_attention,fullgraph=True,mode='max-autotune-no-cudagraphs')
    variants=[('default',original,{}),('autotune',tuned,{}),
        ('safe',original,{'ROWS_GUARANTEED_SAFE':True}),
        ('dq_off',original,{'ROWS_GUARANTEED_SAFE':True,'WRITE_DQ':False}),
        ('tile64',original,{'BLOCK_M':64,'BLOCK_N':64,'bwd_BLOCK_M1':64,'bwd_BLOCK_N1':64,'bwd_BLOCK_M2':64,'bwd_BLOCK_N2':64}),
        ('tile128',original,{'BLOCK_M':128,'BLOCK_N':64,'bwd_BLOCK_M1':64,'bwd_BLOCK_N1':128,'bwd_BLOCK_M2':128,'bwd_BLOCK_N2':64}),
        ('tile128_dqoff',original,{'BLOCK_M':128,'BLOCK_N':64,'bwd_BLOCK_M1':64,'bwd_BLOCK_N1':128,'bwd_BLOCK_M2':128,'bwd_BLOCK_N2':64,'WRITE_DQ':False}),
        ('prescale',original,{'PRESCALE_QK':True,'ROWS_GUARANTEED_SAFE':True})]
    rows=[]
    for batch,seq,packed in [(2,4096,False),(4,2928,True)]:
        q,k,v=[torch.randn(batch,seq,h,32,device='cuda',dtype=torch.bfloat16).transpose(1,2).detach().requires_grad_(True) for h in (16,4,4)]
        ids=(torch.arange(seq,device='cuda')//976)[None].expand(batch,-1)
        mask=build_packed_flex_mask(ids).block_mask if packed else _causal_flex_mask(seq,'cuda:0')
        dy=torch.randn_like(q)
        ref=None
        for name,fn,opts in variants:
            options={'BACKEND':'TRITON',**opts}
            def execute():
                y=fn(q,k,v,block_mask=mask,enable_gqa=True,kernel_options=options)
                g=torch.autograd.grad(y,(q,k,v),dy)
                return y,g
            try:
                y,g=execute()
                if ref is None: ref=(y.detach().clone(),[t.detach().clone() for t in g])
                errors=[float((a.detach().float()-b.float()).norm()/b.float().norm()) for a,b in zip((y,*g),(ref[0],*ref[1]))]
                del y,g
                ms=triton.testing.do_bench_cudagraph(execute,rep=100)
                row=dict(batch=batch,seq=seq,packed=packed,name=name,options=opts,ms=ms,relative_l2=errors)
            except Exception as exc:
                row=dict(batch=batch,seq=seq,packed=packed,name=name,error=str(exc))
            rows.append(row);print(json.dumps(row),flush=True)
            Path('experiments/results/mfu_20261013/attention_tiles.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
