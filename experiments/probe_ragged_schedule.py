"""Compare device-only flat scheduling with the persistent expert loop.

This is an isolated GEMM experiment, not an end-to-end training result.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import torch
import triton
import triton.language as tl
import json
import model.grouped_mm_triton as old


@triton.jit
def flat_mm(X, W, Ends, Y, K: tl.constexpr, N: tl.constexpr,
            E: tl.constexpr, X0: tl.constexpr, X1: tl.constexpr,
            W0: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    ends = tl.load(Ends + tl.arange(0, E))
    starts = tl.load(Ends + tl.arange(0, E) - 1, tl.arange(0, E) > 0, other=0)
    blocks = tl.cdiv(ends - starts, BM)
    cumulative = tl.cumsum(blocks)
    expert = tl.sum((tile >= cumulative).to(tl.int32))
    if expert < E:
        start = tl.sum(tl.where(tl.arange(0, E) == expert, starts, 0))
        end = tl.sum(tl.where(tl.arange(0, E) == expert, ends, 0))
        prev = tl.sum(tl.where(tl.arange(0, E) == expert - 1, cumulative, 0))
        rm = (tile - prev) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for block in range(tl.cdiv(K, BK)):
            kk = block * BK + rk
            a = tl.load(X + (start + rm[:, None]) * X0 + kk[None, :] * X1,
                        (start + rm[:, None] < end) & (kk[None, :] < K), 0)
            b = tl.load(W + expert * W0 + kk[:, None] * W1 + rn[None, :] * W2,
                        (kk[:, None] < K) & (rn[None, :] < N), 0)
            acc = tl.dot(a, b, acc)
        tl.store(Y + (start + rm[:, None]) * N + rn[None, :], acc,
                 (start + rm[:, None] < end) & (rn[None, :] < N))


def run(x, w, ends, tile):
    bm, bn, bk, warps, stages = tile
    y = torch.empty((x.shape[0], w.shape[2]), device=x.device, dtype=x.dtype)
    flat_mm[(triton.cdiv(x.shape[0], bm)+w.shape[0]-1, triton.cdiv(w.shape[2], bn))](
        x,w,ends,y,x.shape[1],w.shape[2],w.shape[0],*x.stride(),*w.stride(),
        bm,bn,bk,num_warps=warps,num_stages=stages)
    return y


def measure(fn):
    return triton.testing.do_bench_cudagraph(fn, rep=100)


def main():
    torch.set_num_threads(1); torch.manual_seed(17)
    rows=[]
    for counts in ([2048]*8, [0,100,500,4000,2000,4000,3000,2784]):
        ends=torch.tensor(counts,device='cuda',dtype=torch.int32).cumsum(0,dtype=torch.int32)
        for k,n,transposed in [(512,1664,True),(1664,512,True),(1664,512,False),(512,1664,False)]:
            x=torch.randn(sum(counts),k,device='cuda',dtype=torch.bfloat16)
            w=torch.randn(8,n,k,device='cuda',dtype=torch.bfloat16).transpose(1,2) if transposed else torch.randn(8,k,n,device='cuda',dtype=torch.bfloat16)
            ref=old.ragged_mm(x,w,ends)
            baseline=measure(lambda:old.ragged_mm(x,w,ends))
            cases=[]
            for tile in [(64,64,32,4,3),(64,128,32,4,3),(128,64,32,4,3),
                         (64,128,64,4,3),(128,128,32,8,3),(128,128,64,8,3)]:
                actual=run(x,w,ends,tile)
                old._FORWARD_TILE=tile
                cases.append(dict(tile=tile,exact=torch.equal(ref,actual),
                    rel_l2=float(torch.linalg.vector_norm(actual.float()-ref.float())/torch.linalg.vector_norm(ref.float())),
                    ms=measure(lambda:run(x,w,ends,tile)),
                    persistent_ms=measure(lambda:old.ragged_mm(x,w,ends))))
            old._FORWARD_TILE=(64,64,32,4,3)
            row=dict(counts=counts,k=k,n=n,transposed=transposed,baseline_ms=baseline,cases=cases)
            print(json.dumps(row),flush=True); rows.append(row)
    Path('experiments/results/mfu_20261013/ragged_schedule.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__': main()
