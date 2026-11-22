"""NumPy LongRoPE tables for the Torch-free native runtime."""
import math
import os
import numpy as np


def is_longrope(scaling):
    return isinstance(scaling, dict) and scaling.get('rope_type', scaling.get('type')) == 'longrope'


def rope_switch_mode(scaling):
    """Resolve the LongRoPE table mode: position / smooth / legacy batch."""
    configured = (scaling or {}).get('switch_mode')
    mode = os.environ.get('INSTINCT_ROPE_SWITCH_MODE') or configured or 'position'
    if mode not in ('position', 'batch', 'smooth'):
        raise ValueError('LongRoPE switch_mode must be position, batch or smooth')
    return mode


def validate_longrope(scaling, head_dim, maximum):
    allowed = {'type', 'rope_type', 'factor', 'original_max_position_embeddings',
               'short_factor', 'long_factor', 'attention_factor',
               'short_attention_factor', 'long_attention_factor',
               'retained_start_tokens', 'short_retained_start_tokens',
               'long_retained_start_tokens', 'switch_mode',
               'smooth_start', 'smooth_end'}
    if set(scaling) - allowed:
        raise ValueError('unknown LongRoPE fields')
    rope_switch_mode(scaling)
    original = scaling.get('original_max_position_embeddings')
    if type(original) is not int or original < 2 or original > maximum:
        raise ValueError('invalid LongRoPE original context')
    smooth_start = scaling.get('smooth_start', original)
    smooth_end = scaling.get('smooth_end', min(2 * original, maximum))
    if (type(smooth_start) is not int or type(smooth_end) is not int
            or not (original <= smooth_start < smooth_end <= maximum)):
        raise ValueError('LongRoPE smooth_start/smooth_end out of range')
    factor = scaling.get('factor', maximum / original)
    if not math.isfinite(factor) or factor < 1 or head_dim % 2:
        raise ValueError('invalid LongRoPE factor or head dimension')
    for prefix in ('short', 'long'):
        values = scaling.get(prefix + '_factor')
        if not isinstance(values, list) or len(values) != head_dim // 2:
            raise ValueError('LongRoPE requires head_dim / 2 factors')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 1 for v in values):
            raise ValueError('LongRoPE factors must be finite and >= 1')
        retained = scaling.get(prefix + '_retained_start_tokens', scaling.get('retained_start_tokens', 0))
        if type(retained) is not int or not 0 <= retained <= original:
            raise ValueError('invalid LongRoPE retained-start count')
    for key in ('attention_factor', 'short_attention_factor', 'long_attention_factor'):
        if key in scaling and (not math.isfinite(scaling[key]) or scaling[key] <= 0):
            raise ValueError('invalid LongRoPE attention factor')


def longrope_tables(config, long_context=True):
    scaling = config['rope_scaling']
    dim, end = config['head_dim'], config['max_position_embeddings']
    validate_longrope(scaling, dim, end)
    prefix = 'long' if long_context else 'short'
    original = scaling['original_max_position_embeddings']
    factor = scaling.get('factor', end / original)
    if not long_context:
        end = original
    amplitude = scaling.get(prefix + '_attention_factor', scaling.get(
        'attention_factor', math.sqrt(1 + math.log(factor) / math.log(original))))
    frequencies = 1. / (config['rope_theta'] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    phase = np.outer(np.arange(end, dtype=np.float32),
                     frequencies / np.asarray(scaling[prefix + '_factor'], np.float32))
    result = [np.tile(fn(phase), (1, 2)).astype(np.float32) * amplitude
              for fn in (np.cos, np.sin)]
    retained = scaling.get(prefix + '_retained_start_tokens', scaling.get('retained_start_tokens', 0))
    if retained:
        native = np.outer(np.arange(retained, dtype=np.float32), frequencies)
        for table, fn in zip(result, (np.cos, np.sin)):
            table[:retained] = np.tile(fn(native), (1, 2))
    return tuple(result)


def smooth_longrope_tables(config):
    """Single LongRoPE table with a smooth post-boundary ramp to the long factors.

    Positions below ``smooth_start`` use the short (original RoPE) factors and
    attention scale exactly. Over ``[smooth_start, smooth_end]`` each rotary
    pair's factor is interpolated geometrically between short and long and the
    attention scale is ramped linearly, so the encoding is continuous at the
    boundary. Positions beyond ``smooth_end`` are exactly the long table.
    """
    scaling = config['rope_scaling']
    dim, maximum = config['head_dim'], config['max_position_embeddings']
    validate_longrope(scaling, dim, maximum)
    original = scaling['original_max_position_embeddings']
    start = int(scaling.get('smooth_start', original))
    stop = int(scaling.get('smooth_end', min(2 * original, maximum)))
    factor = scaling.get('factor', maximum / original)
    short_amp = scaling.get('short_attention_factor',
                            scaling.get('attention_factor', 1.0))
    long_amp = scaling.get('long_attention_factor', scaling.get(
        'attention_factor', math.sqrt(1 + math.log(factor) / math.log(original))))
    frequencies = 1. / (config['rope_theta'] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    positions = np.arange(maximum, dtype=np.float32)
    t = np.clip((positions - start) / max(stop - start, 1), 0.0, 1.0).astype(np.float32)
    factors = (np.power(np.asarray(scaling['short_factor'], np.float32)[None, :], (1.0 - t)[:, None])
               * np.power(np.asarray(scaling['long_factor'], np.float32)[None, :], t[:, None]))
    amplitude = short_amp + (long_amp - short_amp) * t
    phase = positions[:, None] * frequencies[None, :] / factors
    return tuple(np.tile(fn(phase), (1, 2)).astype(np.float32) * amplitude[:, None]
                 for fn in (np.cos, np.sin))


def mixed_longrope_tables(config):
    """Position-unique LongRoPE tables (short rows then long rows).

    Rows below ``original_max_position_embeddings`` come from the short table
    (normally the original RoPE, since short factors are 1.0); rows at or beyond
    it come from the scaled long table. A position therefore has exactly one
    encoding regardless of document length, batch packing or KV-cache decoding.
    """
    scaling = config['rope_scaling']
    original = scaling['original_max_position_embeddings']
    long_cos, long_sin = longrope_tables(config, True)
    short_cos, short_sin = longrope_tables(config, False)
    cos, sin = long_cos.copy(), long_sin.copy()
    cos[:original] = short_cos[:original]
    sin[:original] = short_sin[:original]
    return cos, sin
