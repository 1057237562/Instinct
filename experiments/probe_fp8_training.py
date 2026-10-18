"""Local FP8 capability and expert-size forward/backward benchmark.

This is a synthetic kernel probe, not a production MFU or quality result.
FP32 master parameters, BF16 inputs/outputs, dynamic FP8 quantization included.
"""
import argparse
import copy
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
import trainer.compile_cache
import datasets  # Windows DLL ordering
import torch
from torchao.float8 import Float8LinearConfig, convert_to_float8_training


def relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / a.detach().float().norm().clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='experiments/results/mfu_20261013/fp8_training_probe.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(731)
    report = dict(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  torch=torch.__version__, cuda=torch.version.cuda, cases=[],
                  scope='Synthetic single linear fwd+dx+dw; includes quantization; no optimizer or routing')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for rows, inputs, outputs in ((2048,512,1664),(2928,512,1664),(2048,1664,512),(2928,1664,512)):
        baseline = torch.nn.Sequential(torch.nn.Linear(inputs,outputs,bias=False,device='cuda'))
        x = torch.randn(rows,inputs,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        dy = torch.randn(rows,outputs,device='cuda',dtype=torch.bfloat16)
        def run(module):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                y=module(x)
            grads=torch.autograd.grad(y,(x,*module.parameters()),dy)
            return y,grads
        ref, refgrads=run(baseline)
        for recipe in ('bf16','tensorwise','rowwise'):
            entry=dict(rows=rows,inputs=inputs,outputs=outputs,recipe=recipe)
            try:
                torch._dynamo.reset()
                module=copy.deepcopy(baseline)
                if recipe!='bf16':
                    config=replace(Float8LinearConfig.from_recipe_name(recipe),pad_inner_dim=True)
                    convert_to_float8_training(module,config=config)
                compiled=torch.compile(module,fullgraph=True)
                got,grads=run(compiled)
                entry.update(output_relative_l2=relative(ref,got),dx_relative_l2=relative(refgrads[0],grads[0]),
                             dw_relative_l2=relative(refgrads[1],grads[1]),
                             finite=bool(torch.isfinite(got).all() and all(torch.isfinite(g).all() for g in grads)))
                for _ in range(5): run(compiled)
                torch.cuda.synchronize()
                timings=[]
                for _ in range(3):
                    start=time.perf_counter()
                    for _ in range(100): run(compiled)
                    torch.cuda.synchronize()
                    timings.append((time.perf_counter()-start)*10)
                entry['wall_ms']=statistics.median(timings)
                entry['wall_ms_repeats']=timings
                start_event=torch.cuda.Event(enable_timing=True)
                end_event=torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(100): run(compiled)
                end_event.record()
                end_event.synchronize()
                entry['cuda_timeline_ms']=start_event.elapsed_time(end_event)/100
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    run(compiled)
                    torch.cuda.synchronize()
                entry['gemm_events']=[e.key for e in prof.key_averages() if any(s in e.key.lower() for s in ('gemm','scaled_mm','cutlass','f8f8','mma'))]
                try:
                    # Probe-only: fixed leaf inputs retain AccumulateGrad stream
                    # metadata from the preceding numerical comparison.
                    torch.autograd.graph.set_override_stale_capture_stream(True)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured=run(compiled)
                    for _ in range(5): graph.replay()
                    start_event.record()
                    for _ in range(100): graph.replay()
                    end_event.record()
                    end_event.synchronize()
                    entry['graph_cuda_ms']=start_event.elapsed_time(end_event)/100
                    del graph,captured
                except Exception as exc:
                    entry['graph_error']=str(exc)[:1000]
                del compiled,module,got,grads
            except Exception as exc:
                entry['error']=str(exc)[:2000]
            report['cases'].append(entry)
            output.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(entry),flush=True)
        del baseline,x,dy,ref,refgrads
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
