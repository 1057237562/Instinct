"""Attention-mask helpers shared by inference and training backbones."""

import torch


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
