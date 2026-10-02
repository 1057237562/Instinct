"""Attention-mask helpers shared by inference and training backbones."""

from typing import NamedTuple

import torch


class PreparedAttentionBias(NamedTuple):
    """Internal SDPA-only additive bias, already including causal constraints.

    The explicit wrapper avoids confusing public floating-point 0/1 masks with
    additive 0/-inf masks. It is a pytree understood by checkpoint and compile.
    """

    tensor: torch.Tensor


def prepare_sdpa_attention_bias(
    attention_mask: torch.Tensor,
    reference: torch.Tensor,
    *,
    query_length: int,
    is_causal: bool = True,
) -> PreparedAttentionBias:
    """Build one read-only, aligned attention bias for all layers in a forward.

    Use the attention compute dtype, not necessarily the embedding/master-weight
    dtype. Pad storage (not logical token length) to avoid per-layer SDPA bias
    alignment copies. This is per-forward data, never a persistent model cache.
    """
    allowed = normalize_attention_mask(attention_mask)
    key_length = allowed.size(-1)
    if is_causal:
        causal = torch.ones(
            (query_length, key_length), device=allowed.device, dtype=torch.bool
        ).tril(diagonal=key_length - query_length)
        allowed = allowed & causal
    device_type = reference.device.type
    dtype = (torch.get_autocast_dtype(device_type)
             if torch.is_autocast_enabled(device_type) else reference.dtype)
    # Match Inductor's padded attention-bias layout. Keep heads broadcastable.
    padded_length = ((key_length + 63) // 64) * 64
    storage = torch.full(
        (*allowed.shape[:-1], padded_length), float('-inf'),
        dtype=dtype, device=allowed.device,
    )
    bias = storage[..., :key_length]
    bias.masked_fill_(allowed, 0.0)
    return PreparedAttentionBias(bias)


def normalize_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Normalize 2D/3D/4D 0/1 masks to an SDPA-compatible boolean mask."""
    allowed = attention_mask if attention_mask.dtype == torch.bool else attention_mask != 0
    if allowed.ndim == 2:
        return allowed[:, None, None, :]
    if allowed.ndim == 3:
        return allowed[:, None, :, :]
    if allowed.ndim == 4:
        return allowed
    raise ValueError("attention_mask must have 2, 3, or 4 dimensions")


def apply_attention_mask(scores: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask attention scores with a fill value representable by their dtype."""
    # ``masked_fill`` converts the scalar to ``scores.dtype`` even when the
    # mask contains no False entries. A fixed -1e9 therefore overflows for
    # float16 scores, whose lowest finite value is -65504.
    mask_value = torch.finfo(scores.dtype).min
    return scores.masked_fill(~normalize_attention_mask(attention_mask), mask_value)
