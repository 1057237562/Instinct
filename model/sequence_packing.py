"""Sequence-packing metadata helpers shared by all model backbones."""

import torch

from model.attention_mask import normalize_attention_mask as _normalize_attention_mask


def positions_from_sequence_ids(sequence_ids: torch.Tensor) -> torch.Tensor:
    """Return per-example positions, resetting to zero at every segment."""
    if sequence_ids.ndim != 2:
        raise ValueError("sequence_ids must have shape [batch, sequence]")
    batch_size, seq_len = sequence_ids.shape
    indices = torch.arange(seq_len, device=sequence_ids.device, dtype=torch.long)
    indices = indices.unsqueeze(0).expand(batch_size, -1)
    boundaries = torch.ones_like(sequence_ids, dtype=torch.bool)
    boundaries[:, 1:] = sequence_ids[:, 1:] != sequence_ids[:, :-1]
    starts = torch.where(boundaries, indices, torch.zeros_like(indices))
    starts = torch.cummax(starts, dim=-1).values
    return indices - starts


def block_diagonal_attention_mask(sequence_ids: torch.Tensor) -> torch.Tensor:
    """Build a boolean [batch, query, key] same-segment mask.

    Causality is applied separately by the attention implementation.  Padding
    uses sequence id -1 and therefore remains numerically well-defined while
    staying isolated from all real examples.
    """
    if sequence_ids.ndim != 2:
        raise ValueError("sequence_ids must have shape [batch, sequence]")
    return sequence_ids.unsqueeze(-1) == sequence_ids.unsqueeze(-2)


def merge_packed_attention_mask(
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Combine packed segment isolation with an optional caller mask."""
    packed_mask = block_diagonal_attention_mask(sequence_ids)
    if attention_mask is None:
        return packed_mask
    allowed = _normalize_attention_mask(attention_mask)
    if allowed.ndim == 4:
        allowed = allowed.squeeze(1)
    if allowed.ndim == 2:
        allowed = allowed.unsqueeze(1)
    if allowed.ndim != 3:
        raise ValueError("attention_mask must have 2, 3, or 4 dimensions")
    return packed_mask & allowed
