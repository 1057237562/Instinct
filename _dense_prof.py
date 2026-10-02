"""Scratch profile (deleted after run): why is fp8 KV cache 6x slower than fp32 in eager?"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.inference_runtime import optimize_inference, select_inference_dtype

with open('checkpoints/full_sft_20260916_000740_768.json', encoding='utf-8') as f:
    cfg_dict = json.load(f)

ids = torch.randint(3, 6000, (1, 24), device='cuda')


def profile_cfg(kv_dtype, n=48):
    cfg_dict['kv_cache_dtype'] = kv_dtype
    model = InstinctForCausalLM(InstinctConfig(**cfg_dict))
    sd = torch.load('out/full_sft_20260916_000740_768.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(dtype=select_inference_dtype('cuda')).eval().to('cuda')
    model = optimize_inference(model, 'auto')
    kw = dict(input_ids=ids, max_new_tokens=n, do_sample=True, temperature=0.9, eos_token_id=None)
    with torch.inference_mode():
        model.generate(**kw)
        torch.cuda.synchronize()
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model.generate(**kw)
            torch.cuda.synchronize()
    events = prof.key_averages()
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA
               and e.self_device_time_total > 0]
    launches = sum(e.count for e in kernels)
    gpu_ms = sum(e.self_device_time_total for e in kernels) / 1000.0
    print(f'\n=== kv={kv_dtype}: {launches / n:.0f} kernels/token, GPU busy {gpu_ms / n:.2f} ms/token')
    for e in sorted(kernels, key=lambda e: -e.self_device_time_total)[:6]:
        print(f'   gpu {e.self_device_time_total / 1000:7.1f} ms x{e.count:<5d} {e.key[:70]}')
    for e in sorted(events, key=lambda e: -e.self_cpu_time_total)[:6]:
        print(f'   cpu {e.self_cpu_time_total / 1000:7.1f} ms x{e.count:<5d} {e.key[:70]}')
    del model
    torch.cuda.empty_cache()


profile_cfg('fp8_e5m2')
profile_cfg('fp32')
