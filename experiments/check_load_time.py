"""Phase-by-phase model load timing, mirroring scripts/web_demo.py's loader.

Load time is easy to regress without noticing: it is dominated by
``optimize_inference``, not by reading the weights. Run this after touching the
inference path.

    python experiments/check_load_time.py                 # WebUI default (auto)
    python experiments/check_load_time.py full            # compiles the trunk
    LOAD_CONFIG=checkpoints/other.json LOAD_WEIGHT=out/other.pth \
        python experiments/check_load_time.py auto

Interpreting the numbers, for the 512-dim MoE model:

  construct/read/load_state   ~5 s    unavoidable, same as before this tool existed
  optimize(auto)              ~0 s    just installs kernels/dispatch/cache opt-in
  optimize(full)              ~50 s   Inductor compiles the whole trunk; the first
                                      real prompt then compiles again (~37 s) because
                                      each generate() allocates a new static KV cache
                                      and the graph is guarded on those buffers
  first generate              ~2 s    one-time Triton autotuning + capture

So 'auto' (the WebUI default) is the interactive choice, and 'full' buys about
30% more decode throughput for a minute of compiling.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache  # noqa: F401 -- same Inductor cache config as web_demo
import datasets  # noqa: F401 -- Windows DLL order
import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.inference_runtime import optimize_inference

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = 'checkpoints/pretrain_20260919_174858_512_moe.json'
DEFAULT_WEIGHT = 'out/pretrain_20260919_174858_512_moe.pth'
mode = sys.argv[1] if len(sys.argv) > 1 else 'auto'
config_path = os.environ.get('LOAD_CONFIG', DEFAULT_CONFIG)
weight_path = os.environ.get('LOAD_WEIGHT', DEFAULT_WEIGHT)

with open(REPO / config_path, encoding='utf-8') as fh:
    config = InstinctConfig(**json.load(fh))

clock = time.perf_counter
t0 = clock()
model = InstinctForCausalLM(config)
t1 = clock()
state = torch.load(REPO / weight_path, map_location='cpu', weights_only=True)
t2 = clock()
model.load_state_dict(state, strict=False)
del state
model = model.to(torch.bfloat16).eval().cuda()
t3 = clock()
optimize_inference(model, mode)
t4 = clock()
print(f'[load] construct        {t1 - t0:6.2f}s')
print(f'[load] read weights     {t2 - t1:6.2f}s')
print(f'[load] load_state+cast  {t3 - t2:6.2f}s')
print(f'[load] optimize({mode:5s})    {t4 - t3:6.2f}s')
print(f'[load] TOTAL            {t4 - t0:6.2f}s')

# Separately: what the first chat turn costs beyond the load.
ids = torch.randint(10, 6000, (1, 128), device='cuda')
mask = torch.ones_like(ids)
t5 = clock()
with torch.inference_mode():
    model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=16,
                   do_sample=False, eos_token_id=None)
t6 = clock()
print(f'[load] first generate   {t6 - t5:6.2f}s')
