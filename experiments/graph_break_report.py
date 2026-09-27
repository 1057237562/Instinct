"""Report every torch.compile graph break in the eval-mode inference trunk.

A faithful but 2-layer copy of a real checkpoint config, so it runs in seconds
while exercising the same features (MoE dispatch, quantized KV config, LongRoPE,
attention mask). Run it first whenever the compiled trunk gets slower: a break
per layer is the usual cause, and the reported file:line names the culprit.

    python experiments/graph_break_report.py
    python experiments/graph_break_report.py checkpoints/other_moe.json

The reference point is a single graph with zero breaks for the standard
topology; anything else is a regression worth investigating.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: F401 -- Windows DLL order
import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.inference_runtime import optimize_inference

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = 'checkpoints/pretrain_20260919_174858_512_moe.json'
config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
with open(REPO / config_path, encoding='utf-8') as fh:
    raw = json.load(fh)
raw['num_hidden_layers'] = 2  # keep the trace short; breaks are per layer
config = InstinctConfig(**raw)
print('[cfg] layers', config.num_hidden_layers, 'kv', config.kv_cache_dtype,
      'rope', type(config.rope_scaling).__name__, 'moe', config.use_moe)

torch.manual_seed(0)
model = InstinctForCausalLM(config).to(torch.bfloat16).eval().cuda()
trunk = model.model
optimize_inference(model, 'full')

ids = torch.randint(3, 100, (1, 6), device='cuda')
mask = torch.ones_like(ids)
with torch.inference_mode():
    explanation = torch._dynamo.explain(trunk)(ids, attention_mask=mask, use_cache=True)
print(f'[explain] graph count: {explanation.graph_count}')
print(f'[explain] ops per graph: {[len(g.graph.nodes) for g in explanation.graphs]}')
for i, reason in enumerate(explanation.break_reasons):
    stack = reason.user_stack[-1] if reason.user_stack else None
    where = f'{Path(stack.filename).name}:{stack.lineno}' if stack else '?'
    print(f'[break {i}] {where} :: {reason.reason.splitlines()[0][:100]}')
