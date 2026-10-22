"""Scratch bisect (deleted after run): isolate the fp8-KV 6x slowdown.

Splits wall time into Python-call time (before final sync) vs GPU drain.
Cross-tests dtype (bf16/fp16) and kv dtype on the SAME 20L dense weights.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.inference_runtime import optimize_inference

with open('checkpoints/full_sft_20260916_000740_768.json', encoding='utf-8') as f:
    cfg_dict = json.load(f)

ids = torch.randint(3, 6000, (1, 24), device='cuda')


def run(kv_dtype, weight_dtype, mode='auto', n=128):
    cfg_dict['kv_cache_dtype'] = kv_dtype
    model = InstinctForCausalLM(InstinctConfig(**cfg_dict))
    sd = torch.load('out/full_sft_20260916_000740_768.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(dtype=weight_dtype).eval().to('cuda')
    model = optimize_inference(model, mode)
    kw = dict(input_ids=ids, max_new_tokens=n, do_sample=True, temperature=0.9, eos_token_id=None)
    with torch.inference_mode():
        model.generate(**kw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**kw)          # python-side duration: launches + python overhead
        t_cpu = time.perf_counter() - t0
        torch.cuda.synchronize()      # then drain the GPU queue
        t_wall = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return n / t_wall, t_cpu * 1000 / n, t_wall * 1000 / n


for kv, dt in [('fp8_e5m2', torch.bfloat16), ('fp32', torch.bfloat16),
               ('fp8_e5m2', torch.float16)]:
    tps, cpu_ms, wall_ms = run(kv, dt)
    print(f'kv={kv:8s} weights={str(dt).split(".")[-1]:8s}: {tps:6.1f} tok/s | '
          f'cpu-call {cpu_ms:6.1f} ms/tok | wall {wall_ms:6.1f} ms/tok', flush=True)
