"""Per-layer selective replay plans for native Dense training.

Only projection outputs and fused attention O/LSE survive the forward.
Quantization inputs/scales and all VJP nodes are rebuilt during replay.
"""
from dataclasses import dataclass, field


CACHE_BYTES = 7 * 2**28  # Same 1.75 GiB candidate-output budget as V2 ckpt4.


def recipe_plan(config, shape, dtype, layer):
    tokens = shape[0] * shape[1]
    size = 4 if dtype == 'fp32' else 2
    h, q, kv, f = (config['hidden_size'], config['num_attention_heads'] * config['head_dim'],
                   config['num_key_value_heads'] * config['head_dim'], config['intermediate_size'])
    layers = config['num_hidden_layers']
    budget = int(config.get('selective_cache_mib', CACHE_BYTES / 2**20) * 2**20)
    if budget < 0:raise ValueError('selective_cache_mib must be nonnegative')
    attention = shape[1] >= 4096 and config['dropout'] == 0.
    per_token = size * q + 4 * config['num_attention_heads'] if attention else size * 2 * kv
    base_layers = min(layers, budget // max(1, tokens * per_token))
    gate_layers = min(layers, max(0, (budget - tokens * base_layers * per_token)
                                // max(1, tokens * size * 2 * f)))
    prefix = f'model.layers.{layer}.'
    names = set()
    if not attention and layer >= layers - base_layers and kv != q and f != kv:
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
