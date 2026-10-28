"""Per-layer selective replay plans for native Dense training.

Only projection outputs and fused attention O/LSE survive the forward.
Quantization inputs/scales and all VJP nodes are rebuilt during replay.
"""
from dataclasses import dataclass, field
import math


CACHE_BYTES = 7 * 2**28  # Same 1.75 GiB candidate-output budget as V2 ckpt4.


def parse_cache_budgets(value):
    """Exact sequence-length overrides, e.g. 3072:768,4096:1024 (MiB)."""
    if not value:return {}
    result={}
    for entry in value.split(','):
        parts=entry.strip().split(':')
        if len(parts)!=2:raise ValueError('cache budgets require sequence_length:MiB entries')
        length,budget=int(parts[0]),float(parts[1])
        if length<1 or not math.isfinite(budget) or budget<0 or str(length) in result:
            raise ValueError('cache budgets require unique positive lengths and finite nonnegative MiB')
        result[str(length)]=budget
    return result


def recipe_plan(config, shape, dtype, layer, *, attention_min_seq=4096, auxiliary_kv=False):
    tokens = shape[0] * shape[1]
    size = 4 if dtype == 'fp32' else 2
    h, q, kv, f = (config['hidden_size'], config['num_attention_heads'] * config['head_dim'],
                   config['num_key_value_heads'] * config['head_dim'], config['intermediate_size'])
    layers = config['num_hidden_layers']
    selected=config.get('selective_cache_by_seq',{}).get(str(shape[1]),
        config.get('selective_cache_mib',CACHE_BYTES/2**20))
    if not math.isfinite(selected):raise ValueError('selective cache budget must be finite')
    budget = int(selected * 2**20)
    if budget < 0:raise ValueError('selective_cache_mib must be nonnegative')
    attention = shape[1] >= attention_min_seq and config['dropout'] == 0.
    per_token = size * q + 4 * config['num_attention_heads'] if attention else size * 2 * kv
    base_layers = min(layers, budget // max(1, tokens * per_token))
    remaining=budget-tokens*base_layers*per_token
    # Measured V2 BF16 K/V projections save more time per resident byte than
    # the much larger tensor-core FFN projections. Fill the remainder only;
    # the O/LSE priority and the user's hard cache-byte budget are preserved.
    kv_layers=min(layers,remaining//max(1,tokens*size*2*kv)) if attention and auxiliary_kv and kv!=q and f!=kv else 0
    remaining-=tokens*size*2*kv*kv_layers
    gate_layers = min(layers, max(0, remaining
                                // max(1, tokens * size * 2 * f)))
    prefix = f'model.layers.{layer}.'
    names = set()
    if layer >= layers-(kv_layers if attention else base_layers) and kv != q and f != kv:
        names.update(prefix + 'self_attn.' + k + '_proj.weight' for k in ('k', 'v'))
    if layer >= layers - gate_layers and f not in (h, q, kv):
        names.update(prefix + 'mlp.' + k + '_proj.weight' for k in ('gate', 'up'))
    return ReplayCache(names, attention and layer >= layers - base_layers)


@dataclass
class ReplayCache:
    linear_names: set
    attention: bool = False
    replay: bool = False
    values: dict = field(default_factory=dict)

    def tensors(self):
        return tuple(t for value in self.values.values()
                     for t in (value if isinstance(value, tuple) else (value,)))
