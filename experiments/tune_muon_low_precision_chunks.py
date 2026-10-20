"""Isolated BF16-momentum chunk sweep; repeated V1 block shapes, no training data.

All candidates use the same seed, gradients and number of steps. Final hashes
check that batching preserves weights, uint16 low halves and BF16 momentum.
Use --training for fresh-process full V1 comparisons, including original FP32.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import datasets  # noqa: F401,E402 -- Windows DLL order
import trainer.compile_cache  # noqa: F401,E402
import torch  # noqa: E402
from experiments.audit_muon_precision import production_shapes
from experiments.validate_full_training_kernels import fingerprint
from trainer.batched_muon import BatchedMuon


def run(workspace, batch_size, args):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(73)
    shapes, _ = production_shapes()
    params = [torch.nn.Parameter(torch.randn(s, device='cuda').bfloat16()*.02) for s in shapes*2]
    for p in params:
        p.grad = torch.randn_like(p)
    opt = BatchedMuon(params, use_mantissa=True, momentum_dtype='bf16', lr=5e-4,
                      adjust_lr_fn='match_rms_adamw', workspace_mb=workspace, batch_size=batch_size)
    for _ in range(args.warmup):
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings, wall = [], []
    for _ in range(args.steps):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        t0 = time.perf_counter()
        start.record()
        opt.step()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter()-t0)*1000)
        timings.append(start.elapsed_time(end))
    row = dict(workspace_mb=workspace, batch_size=batch_size,
               chunks={str(s): opt._chunk_size(p) for s,p in zip(shapes*2,params)},
               shape_counts={str(s): n for s,n in Counter(shapes*2).items()},
               median_ms=statistics.median(timings), median_wall_ms=statistics.median(wall),
               peak_bytes=torch.cuda.max_memory_allocated(),
               retained_bytes=torch.cuda.memory_allocated(), timings_ms=timings,
               model_sha256=fingerprint([(str(i), p) for i,p in enumerate(params)]),
               optimizer_sha256=fingerprint([(f'{i}/{key}', value) for i,p in enumerate(params)
                                             for key,value in sorted(opt.state[p].items())]))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--seq', type=int, default=2048)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['INSTINCT_MUON_FUSED_UPDATES'] = 'auto'
    os.environ['INSTINCT_MUON_NS_BACKEND'] = 'auto'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = ['trainer/batched_muon.py', 'trainer/muon_kernels.py', 'trainer/trainer_utils.py',
             'experiments/compare_muon_fp32_mantissa.py', str(Path(__file__).relative_to(ROOT))]
    report = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  protocol=vars(args)|{'output':str(args.output)},
                  revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}, runs=[])
    candidates = [(64,16), (144,8), (272,16), (544,32)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.training:
        if args.steps <= args.warmup:
            parser.error('steps must exceed warmup')
        candidates = [('fp32',256,16), ('mantissa',64,16), ('mantissa',272,16)]
        for index,(arm,workspace,batch_size) in enumerate(candidates+candidates[::-1]):
            path = args.output.with_name(args.output.stem+f'.run{index}.json')
            command = [sys.executable,'-u',str(ROOT/'experiments/compare_muon_fp32_mantissa.py'),
                       '--arm',arm,'--run',str(index),'--steps',str(args.steps),'--warmup',str(args.warmup),
                       '--batch',str(args.batch),'--seq',str(args.seq),'--workspace-mb',str(workspace),
                       '--muon-batch-size',str(batch_size),'--momentum-dtype','bf16','--output',str(path)]
            if args.deterministic:
                command.append('--deterministic')
            subprocess.run(command,cwd=ROOT,check=True)
            report['runs'].append(json.loads(path.read_text(encoding='utf-8')))
            args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
            path.unlink()
        assert len({r['input_sha256'] for r in report['runs']}) == 1
        low = [r for r in report['runs'] if r['arm'] == 'mantissa']
        report['low_precision_all_exact'] = all(all(r[key] == low[0][key] for r in low)
                                               for key in ('losses','model_sha256','optimizer_sha256'))
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        if args.deterministic:
            assert report['low_precision_all_exact']
        return
    for trial in range(2):
        for workspace,batch_size in (candidates if trial == 0 else candidates[::-1]):
            row = run(workspace,batch_size,args)
            row['trial'] = trial
            report['runs'].append(row)
            args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('RESULT', json.dumps({k:v for k,v in row.items() if k != 'timings_ms'}),flush=True)
    report['all_exact'] = all(len({r[key] for r in report['runs']}) == 1
                              for key in ('model_sha256','optimizer_sha256'))
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert report['all_exact'], 'Chunk sizes changed final weights or optimizer state'


if __name__ == '__main__':
    main()
