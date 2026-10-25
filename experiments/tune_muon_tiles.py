"""Offline tile probe; changing tile is gated against the existing NS result."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import datasets
import trainer.compile_cache
import torch, json, gc, argparse, statistics, time
import trainer.muon_kernels as kernels
from trainer.batched_muon import _batched_zeropower_reference, _NS_COEFFICIENTS
from experiments.probe_muon_graph import measure

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2',action='store_true',help='V2 chunks at 512 MiB; pure NS only, no optimizer updates')
    parser.add_argument('--output',default='experiments/results/mfu_20261012/muon_tiles.json')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    opts=dict(ns_coefficients=_NS_COEFFICIENTS, ns_steps=5, eps=1e-7)
    rows=[]
    shapes=[(3,5504,1536),(3,1536,5504),(8,1536,1536),(16,384,1536)] if args.v2 else [(4,1664,512)]
    for shape in shapes:
        x=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
        kernels._TILE=(64,64,32,4,3)
        ref=kernels.zeropower(x,**opts) if args.v2 else _batched_zeropower_reference(x,**opts)
        for tile in [(64,64,32,4,3),(32,64,32,4,3),(64,128,32,4,3),
                     (64,64,64,4,3),(64,128,64,4,3),(128,64,32,4,3),
                     (128,128,32,8,3),(32,128,64,4,3)]:
            kernels._TILE=tile
            cache=kernels.zeropower if args.v2 else kernels.ZeropowerGraphCache()
            y=cache(x,**opts)
            if args.v2:
                for _ in range(3): cache(x,**opts)
                torch.cuda.synchronize()
                samples=[]
                for _ in range(5):
                    start=time.perf_counter()
                    for _ in range(3): cache(x,**opts)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter()-start)*1000/3)
                timing=dict(median_ms=statistics.median(samples),windows_ms=samples)
            else:
                timing=measure(lambda:cache(x,**opts))
            row=dict(shape=shape,tile=tile,exact=torch.equal(ref,y),
                error=float((ref.float()-y.float()).norm()/ref.float().norm()),timing=timing)
            print(json.dumps(row),flush=True)
            rows.append(row)
            target=Path(args.output)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(rows,indent=2))
            del cache
            gc.collect()


if __name__=='__main__': main()
