"""Measure standard NS with a faster inner accumulator, without changing its polynomial."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import torch
import triton
import trainer.muon_kernels as kernels
from trainer.batched_muon import _NS_COEFFICIENTS
from experiments.muon_fp16acc_candidate import _gemm_fp16acc


def main():
    torch.set_num_threads(1);torch.manual_seed(81)
    original=kernels._gemm;rows=[]
    for batch in (4,16):
        x=torch.randn(batch,1664,512,device='cuda',dtype=torch.bfloat16)
        options=dict(ns_coefficients=_NS_COEFFICIENTS,ns_steps=5,eps=1e-7)
        kernels._gemm=original; ref_cache=kernels.ZeropowerGraphCache()
        ref=ref_cache(x,**options).clone()
        baseline=triton.testing.do_bench_cudagraph(lambda:ref_cache(x,**options),rep=100)
        kernels._gemm=_gemm_fp16acc; cache=kernels.ZeropowerGraphCache()
        got=cache(x,**options).clone()
        error=float((ref.float()-got.float()).norm()/ref.float().norm())
        assert torch.isfinite(got).all() and error<.2
        candidate=triton.testing.do_bench_cudagraph(lambda:cache(x,**options),rep=100)
        row=dict(batch=batch,reference_ms=baseline,candidate_ms=candidate,relative_l2=error)
        rows.append(row);print(row,flush=True)
        del cache,ref_cache
    kernels._gemm=original
    Path('experiments/results/mfu_20261013/muon_accumulation.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
