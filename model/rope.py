"""Shared rotary-position helpers for RoPE, YaRN, and LongRoPE.

LongRoPE follows Ding et al. (ICML 2024): every rotary frequency has an
independent interpolation factor and the first ``n_hat`` positions may retain
the original (unscaled) RoPE.  The factors are model-specific search results;
this module deliberately validates and consumes them instead of inventing an
"optimal" vector.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def rope_scaling_type(rope_scaling: Optional[dict]) -> str:
    """Return the normalized scaling type while accepting HF's two spellings."""
    if not rope_scaling:
        return "default"
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "yarn"))
    rope_type = str(rope_type).lower()
    if rope_type == "rope":
        rope_type = "default"
    return rope_type


def _validate_factor_vector(name: str, values, half_dim: int) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"LongRoPE {name} must be a list of {half_dim} numbers")
    if len(values) != half_dim:
        raise ValueError(
            f"LongRoPE {name} must contain head_dim / 2 = {half_dim} values, "
            f"got {len(values)}"
        )
    vector = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 1.0 for value in vector):
        raise ValueError(f"LongRoPE {name} values must be finite and >= 1.0")
    return vector


def validate_rope_scaling(
    rope_scaling: Optional[dict],
    dim: int,
    max_position_embeddings: int,
) -> Optional[dict]:
    """Validate and normalize a RoPE-scaling configuration.

    Both ``type`` (legacy/Hugging Face configs) and ``rope_type`` are accepted.
    LongRoPE needs one factor per rotary pair.  ``retained_start_tokens`` is the
    paper's n-hat; separate short/long values are accepted for the paper's
    short-context recovery pass.
    """
    if rope_scaling is None:
        return None
    if not isinstance(rope_scaling, dict):
        raise ValueError("rope_scaling must be a dictionary or None")
    if dim < 2 or dim % 2:
        raise ValueError("RoPE head_dim must be a positive even integer")

    config = dict(rope_scaling)
    rope_type = rope_scaling_type(config)
    if rope_type not in {"default", "yarn", "longrope"}:
        raise ValueError("rope_scaling type must be one of: default, yarn, longrope")
    config["type"] = rope_type
    config["rope_type"] = rope_type

    if rope_type == "default":
        return config

    original_max = int(config.get("original_max_position_embeddings", 0))
    if original_max < 2:
        raise ValueError("original_max_position_embeddings must be >= 2")
    if int(max_position_embeddings) < original_max:
        raise ValueError(
            "max_position_embeddings must be >= original_max_position_embeddings"
        )
    factor = float(config.get("factor", max_position_embeddings / original_max))
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("rope_scaling factor must be finite and >= 1.0")
    config["factor"] = factor
    config["original_max_position_embeddings"] = original_max

    attention_factor = config.get("attention_factor")
    if attention_factor is not None:
        attention_factor = float(attention_factor)
        if not math.isfinite(attention_factor) or attention_factor <= 0:
            raise ValueError("attention_factor must be finite and > 0")
        config["attention_factor"] = attention_factor

    if rope_type == "yarn":
        beta_fast = float(config.get("beta_fast", 32.0))
        beta_slow = float(config.get("beta_slow", 1.0))
        if beta_fast < beta_slow:
            raise ValueError("YaRN beta_fast must be >= beta_slow")
        config["beta_fast"] = beta_fast
        config["beta_slow"] = beta_slow
        return config

    half_dim = dim // 2
    config["short_factor"] = _validate_factor_vector(
        "short_factor", config.get("short_factor"), half_dim
    )
    config["long_factor"] = _validate_factor_vector(
        "long_factor", config.get("long_factor"), half_dim
    )
    for key in ("short_attention_factor", "long_attention_factor"):
        value = config.get(key)
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"LongRoPE {key} must be finite and > 0")
            config[key] = value
    retained = int(config.get("retained_start_tokens", 0))
    short_retained = int(config.get("short_retained_start_tokens", retained))
    long_retained = int(config.get("long_retained_start_tokens", retained))
    if short_retained < 0 or long_retained < 0:
        raise ValueError("LongRoPE retained start-token counts must be >= 0")
    if short_retained > original_max or long_retained > original_max:
        raise ValueError(
            "LongRoPE retained start-token counts must not exceed the original context"
        )
    config["short_retained_start_tokens"] = short_retained
    config["long_retained_start_tokens"] = long_retained
    return config


def _longrope_attention_factor(config: dict, long_context: bool) -> float:
    contextual_key = "long_attention_factor" if long_context else "short_attention_factor"
    contextual = config.get(contextual_key)
    if contextual is not None:
        return float(contextual)
    explicit = config.get("attention_factor")
    if explicit is not None:
        return float(explicit)
    factor = float(config["factor"])
    if factor <= 1.0:
        return 1.0
    original_max = float(config["original_max_position_embeddings"])
    # Appendix A.2 / Microsoft's reference implementation ("su" policy).
    return math.sqrt(1.0 + math.log(factor) / math.log(original_max))


def precompute_freqs_cis(
    dim: int,
    end: int = 32 * 1024,
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
    *,
    long_context: bool = True,
):
    """Precompute a RoPE cache for default RoPE, YaRN, or LongRoPE.

    ``long_context`` only affects LongRoPE.  A model keeps both caches and
    switches at ``original_max_position_embeddings`` as Hugging Face Phi-3
    does.  The initial n-hat positions in each cache are then overwritten with
    the exact original RoPE, matching Eq. (3) and ``MixedLongRoPE``.
    """
    config = validate_rope_scaling(rope_scaling, dim, end)
    exponent = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    original_inv_freq = 1.0 / (float(rope_base) ** exponent)
    inv_freq = original_inv_freq
    attention_factor = 1.0
    retained_start_tokens = 0
    rope_type = rope_scaling_type(config)

    if rope_type == "yarn":
        original_max = config["original_max_position_embeddings"]
        factor = config["factor"]
        beta_fast = config["beta_fast"]
        beta_slow = config["beta_slow"]
        attention_factor = float(config.get("attention_factor", 1.0))
        if end / original_max > 1.0:
            inv_dim = lambda beta: (
                dim * math.log(original_max / (beta * 2 * math.pi))
            ) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2).float() - low) / max(high - low, 0.001),
                0,
                1,
            )
            inv_freq = original_inv_freq * (1 - ramp + ramp / factor)
    elif rope_type == "longrope":
        prefix = "long" if long_context else "short"
        factors = torch.tensor(config[f"{prefix}_factor"], dtype=torch.float32)
        inv_freq = original_inv_freq / factors
        attention_factor = _longrope_attention_factor(config, long_context)
        retained_start_tokens = int(config[f"{prefix}_retained_start_tokens"])

    positions = torch.arange(end, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq).float()
    freqs_cos = torch.cat((angles.cos(), angles.cos()), dim=-1) * attention_factor
    freqs_sin = torch.cat((angles.sin(), angles.sin()), dim=-1) * attention_factor

    if retained_start_tokens:
        retained_positions = positions[:retained_start_tokens]
        original_angles = torch.outer(retained_positions, original_inv_freq).float()
        original_cos = torch.cat((original_angles.cos(), original_angles.cos()), dim=-1)
        original_sin = torch.cat((original_angles.sin(), original_angles.sin()), dim=-1)
        freqs_cos[:retained_start_tokens] = original_cos
        freqs_sin[:retained_start_tokens] = original_sin
    return freqs_cos, freqs_sin


def build_rope_caches(dim: int, end: int, rope_base: float, rope_scaling: Optional[dict]):
    """Build the active cache and, for LongRoPE, its short-context companion."""
    long_cos, long_sin = precompute_freqs_cis(
        dim, end, rope_base, rope_scaling, long_context=True
    )
    if rope_scaling_type(rope_scaling) != "longrope":
        return long_cos, long_sin, None, None
    short_cos, short_sin = precompute_freqs_cis(
        dim, end, rope_base, rope_scaling, long_context=False
    )
    return long_cos, long_sin, short_cos, short_sin


def select_rope_cache(
    long_cos: torch.Tensor,
    long_sin: torch.Tensor,
    short_cos: Optional[torch.Tensor],
    short_sin: Optional[torch.Tensor],
    rope_scaling: Optional[dict],
    *,
    start_pos: int,
    seq_length: int,
    position_ids: Optional[torch.Tensor] = None,
):
    """Select and index the correct LongRoPE cache for the current context."""
    if short_cos is None:
        cos, sin = long_cos, long_sin
        if position_ids is not None:
            return cos[position_ids], sin[position_ids]
        return (
            cos[start_pos : start_pos + seq_length],
            sin[start_pos : start_pos + seq_length],
        )

    original_max = int(rope_scaling["original_max_position_embeddings"])
    if position_ids is None:
        cos, sin = (
            (short_cos, short_sin)
            if start_pos + seq_length <= original_max
            else (long_cos, long_sin)
        )
        return (
            cos[start_pos : start_pos + seq_length],
            sin[start_pos : start_pos + seq_length],
        )

    # Match the dynamic LongRoPE rule used by Phi-3: switch globally according
    # to the largest logical position. torch.where avoids Tensor.item() graph
    # breaks and also handles position resets produced by sequence packing.
    use_long = position_ids.amax() + 1 > original_max
    short_selected = (short_cos[position_ids], short_sin[position_ids])
    long_selected = (long_cos[position_ids], long_sin[position_ids])
    return (
        torch.where(use_long, long_selected[0], short_selected[0]),
        torch.where(use_long, long_selected[1], short_selected[1]),
    )
