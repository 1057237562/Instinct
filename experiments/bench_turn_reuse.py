"""Does reusing the decode state make a turn faster or slower than rebuilding it?

Greedy decode, identical prompt every turn, so only the state differs. `fresh`
drops the cached state (and its recorded graph) before each turn, which is what
the code did before state reuse existed.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache  # noqa: F401
import datasets  # noqa: F401 -- Windows DLL order
import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.inference_runtime import optimize_inference

REPO = Path(__file__).resolve().parents[1]
mode = sys.argv[1] if len(sys.argv) > 1 else 'auto'
variant = sys.argv[2] if len(sys.argv) > 2 else 'reuse'
with open(REPO / 'checkpoints' / 'pretrain_20260919_174858_512_moe.json', encoding='utf-8') as fh:
    config = InstinctConfig(**json.load(fh))
model = InstinctForCausalLM(config)
state_dict = torch.load(REPO / 'out' / 'pretrain_20260919_174858_512_moe.pth',
                        map_location='cpu', weights_only=True)
model.load_state_dict(state_dict, strict=False)
del state_dict
model = model.to(torch.bfloat16).eval().cuda()
optimize_inference(model, mode)

ids = torch.randint(10, 6000, (1, 96), device='cuda')
mask = torch.ones_like(ids)
captures = []
original_capture = model._capture_decode_step


def recording(*args, **kwargs):
    result = original_capture(*args, **kwargs)
    captures.append(result is not None)
    return result


model._capture_decode_step = recording

for turn in range(4):
    if variant == 'fresh':
        model._decode_states.clear()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=32,
                             do_sample=False, eos_token_id=None)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1e3
    print(f'[turn] {variant} turn {turn}: {elapsed:8.1f} ms for {out.shape[1] - 96} tokens '
          f'({elapsed / max(out.shape[1] - 96, 1):5.2f} ms/token), captures so far {len(captures)}')
