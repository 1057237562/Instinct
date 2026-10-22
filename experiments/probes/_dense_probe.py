"""Scratch investigation (deleted after run): dense SFT 20L decode throughput.

Real weights, current WebUI load path (bf16 + fused kernels). Measures:
  1. auto mode (fused kernels only)      <- current dense WebUI behavior w/ full default off? (we test both)
  2. full mode (trunk torch.compile)
  3. kv cache dtype effect: fp8_e5m2 (config) vs fp32 (pure cat, no quant)
  4. sync count per token
  5. profile: kernel launches + GPU busy per token for the best config
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.inference_runtime import optimize_inference, select_inference_dtype, CompiledTrunk

CONFIG = 'checkpoints/full_sft_20260916_000740_768.json'
WEIGHT = 'out/full_sft_20260916_000740_768.pth'


def load(kv_dtype=None, mode='auto'):
    with open(CONFIG, encoding='utf-8') as f:
        cfg_dict = json.load(f)
    if kv_dtype:
        cfg_dict['kv_cache_dtype'] = kv_dtype
    config = InstinctConfig(**cfg_dict)
    model = InstinctForCausalLM(config)
    sd = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)
    model = model.to(dtype=select_inference_dtype('cuda')).eval().to('cuda')
    return optimize_inference(model, mode)


tokenizer = AutoTokenizer.from_pretrained('model', trust_remote_code=True)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "用几句话介绍一下太阳系"}],
    tokenize=False, add_generation_prompt=True)
ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
print(f'prompt tokens: {ids.shape[1]}')


def bench(model, n=128, warm=True):
    with torch.inference_mode():
        kw = dict(input_ids=ids, max_new_tokens=n, do_sample=True, temperature=0.9,
                  eos_token_id=tokenizer.eos_token_id)
        if warm:
            model.generate(**kw)
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**kw)
        torch.cuda.synchronize()
        return (out.shape[1] - ids.shape[1]) / (time.perf_counter() - t0), out


def syncs_per_token(model, n=32):
    import warnings
    with torch.inference_mode():
        model.generate(input_ids=ids, max_new_tokens=n, do_sample=True,
                       temperature=0.9, eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode('warn')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.generate(input_ids=ids, max_new_tokens=n, do_sample=True,
                           temperature=0.9, eos_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode('default')
    return len(caught) / n


for label, kw in [
    ('auto,  fp8 kv (config default)', dict(kv_dtype=None, mode='auto')),
    ('full,  fp8 kv', dict(kv_dtype=None, mode='full')),
    ('full,  fp32 kv (no quant)', dict(kv_dtype='fp32', mode='full')),
    ('auto,  fp32 kv (no quant)', dict(kv_dtype='fp32', mode='auto')),
]:
    model = load(**kw)
    tps, _ = bench(model)
    sp = syncs_per_token(model)
    print(f'{label}: {tps:6.1f} tok/s | syncs/token={sp:.2f}', flush=True)
    del model
    torch.cuda.empty_cache()
