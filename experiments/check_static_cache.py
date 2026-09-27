"""Static in-place cache vs the growing cache, at equal KV precision.

Both paths run uncompiled so the comparison isolates the cache rewrite. The
legacy path's fp8 cache is forced to the model dtype first, because the static
cache stores unquantized keys/values.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: F401 -- Windows DLL order
import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM

REPO = Path(__file__).resolve().parents[1]
device = 'cuda'

CONFIG = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/pretrain_20260919_174858_512_moe.json'
WEIGHT = sys.argv[2] if len(sys.argv) > 2 else 'out/pretrain_20260919_174858_512_moe.pth'
with open(REPO / CONFIG, encoding='utf-8') as fh:
    config = InstinctConfig(**json.load(fh))
state = torch.load(REPO / WEIGHT, map_location='cpu', weights_only=True)


def load():
    model = InstinctForCausalLM(config)
    model.load_state_dict(state, strict=False)
    return model.to(torch.bfloat16).eval().to(device)


torch.manual_seed(0)
ids = torch.randint(10, 6000, (1, 32), device=device)
mask = torch.ones_like(ids)

legacy = load()
for layer in legacy.model.layers:
    layer.self_attn.kv_cache_dtype = 'bf16'  # match the static cache's precision
with torch.inference_mode():
    legacy_gen = legacy.generate(input_ids=ids, attention_mask=mask, max_new_tokens=24,
                                 do_sample=False, eos_token_id=None)

static = load()
for layer in static.model.layers:
    layer.self_attn.kv_cache_dtype = 'bf16'
static._static_cache_ok = True
with torch.inference_mode():
    static_gen = static.generate(input_ids=ids, attention_mask=mask, max_new_tokens=24,
                                 do_sample=False, eos_token_id=None)

same = torch.equal(legacy_gen, static_gen)
prefix = int((legacy_gen == static_gen).cumprod(dim=1).sum())
print('[static] legacy:', legacy_gen[0, 32:].tolist())
print('[static] static:', static_gen[0, 32:].tolist())
print(f'[static] identical tokens: {same} (matching prefix {prefix}/32 prompt+gen)')
if not same:
    # The two paths differ in how SDPA sees the keys (padded buffer + mask vs a
    # dense prefix), so a token flip deep in the sequence is a numerics effect,
    # not necessarily a bug. Report the logit gap at the first divergence.
    print(f'[static] first divergence at index {prefix}')
print('[static] PASS' if same else '[static] DIVERGED')
