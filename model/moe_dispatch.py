"""Shared MoE routing; grouped-MM backend synchronization is platform dependent."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _cuda_compute_dtype(x: torch.Tensor) -> torch.dtype:
    """Return the dtype CUDA autocast will use for linear algebra."""
    if not x.is_cuda:
        return x.dtype
    try:
        autocast_enabled = torch.is_autocast_enabled("cuda")
    except TypeError:  # PyTorch < 2.4
        autocast_enabled = torch.is_autocast_enabled()
    if not autocast_enabled:
        return x.dtype
    try:
        return torch.get_autocast_dtype("cuda")
    except AttributeError:  # PyTorch < 2.4
        return torch.get_autocast_gpu_dtype()


def _can_use_grouped_mm(x: torch.Tensor, experts: Sequence[nn.Module]) -> bool:
    """Whether the native differentiable BF16 grouped GEMM is available."""
    if (
        not x.is_cuda
        or _cuda_compute_dtype(x) != torch.bfloat16
        or not hasattr(F, "grouped_mm")
        or torch.cuda.get_device_capability(x.device)[0] < 8
        or not experts
    ):
        return False
    first = experts[0]
    hidden = first.gate_proj.in_features
    intermediate = first.gate_proj.out_features
    return hidden % 16 == 0 and intermediate % 16 == 0


def _stack_linear_weights(
    experts: Sequence[nn.Module], name: str, dtype: torch.dtype
) -> torch.Tensor:
    """Stack legacy expert weights in grouped-MM right-hand layout.

    Keeping the original Linear modules preserves existing checkpoint keys and
    optimizer state. The transpose produces the per-group column-major layout
    accepted by ``torch.nn.functional.grouped_mm``.
    """
    return torch.stack(
        [getattr(expert, name).weight for expert in experts], dim=0
    ).transpose(1, 2).to(dtype=dtype)


def _sorted_expert_offsets(
    sorted_experts: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Return int32 exclusive ends for sorted expert IDs in [0, num_experts).

    The right insertion point of expert e equals the number of routes with ID
    <= e, including when that expert is empty. Unlike CUDA bincount (even with
    minlength), this fixed-size search does not read min/max IDs back to CPU.
    """
    expert_ids = torch.arange(
        num_experts, device=sorted_experts.device, dtype=sorted_experts.dtype
    )
    return torch.searchsorted(
        sorted_experts, expert_ids, right=True, out_int32=True
    )


def _grouped_expert_forward(
    x_flat: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    experts: Sequence[nn.Module],
    act_fn,
) -> torch.Tensor:
    """Run all experts with three grouped GEMMs and one final scatter-add."""
    num_tokens = x_flat.shape[0]
    num_experts = len(experts)
    top_k = topk_idx.shape[-1]

    route_expert = topk_idx.reshape(-1)
    route_token = torch.arange(
        num_tokens, device=x_flat.device, dtype=torch.long
    ).view(-1, 1).expand(-1, top_k).reshape(-1)

    # Sort routes and find group boundaries on device. The grouped-MM backend
    # below may still synchronize on platforms that use its per-expert fallback.
    order = torch.argsort(route_expert, stable=True)
    route_expert = route_expert.index_select(0, order)
    route_token = route_token.index_select(0, order)
    route_weight = topk_weight.reshape(-1).index_select(0, order).unsqueeze(-1)
    offsets = _sorted_expert_offsets(route_expert, num_experts)

    compute_dtype = torch.bfloat16
    routed_x = x_flat.index_select(0, route_token).to(dtype=compute_dtype)
    gate = F.grouped_mm(
        routed_x,
        _stack_linear_weights(experts, "gate_proj", compute_dtype),
        offs=offsets,
    )
    up = F.grouped_mm(
        routed_x,
        _stack_linear_weights(experts, "up_proj", compute_dtype),
        offs=offsets,
    )
    hidden = act_fn(gate) * up
    routed_y = F.grouped_mm(
        hidden,
        _stack_linear_weights(experts, "down_proj", compute_dtype),
        offs=offsets,
    )
    routed_y = (routed_y * route_weight.to(routed_y.dtype)).to(x_flat.dtype)

    y = torch.zeros_like(x_flat)
    y.index_add_(0, route_token, routed_y)
    return y


# PyTorch 2.13 Inductor can select an invalid TMA layout for the grouped-MM
# weight-gradient on Blackwell consumer GPUs. Keep this compact region eager:
# Dynamo makes one graph break per MoE block, while the native grouped kernels
# still run on CUDA and their autograd remains fully differentiable. This also
# avoids the eight data-dependent graph breaks made by the previous expert
# loop. Remove the barrier once the upstream grouped-MM TMA lowering is fixed.
if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    _grouped_expert_forward = torch.compiler.disable(_grouped_expert_forward)


def _synchronization_free_expert_loop(
    x_flat: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    experts: Sequence[nn.Module],
) -> torch.Tensor:
    """Portable fallback without the old ``if mask.any()`` host sync.

    Linear/SwiGLU operations accept an empty leading dimension. Evaluating an
    empty expert also attaches zero gradients to its parameters, replacing the
    former Python branch and its explicit zero-gradient workaround.
    """
    y = torch.zeros_like(x_flat)
    for expert_id, expert in enumerate(experts):
        mask = topk_idx == expert_id
        token_idx = mask.any(dim=-1).nonzero().flatten()
        weight = topk_weight[mask].view(-1, 1)
        expert_y = (expert(x_flat.index_select(0, token_idx)) * weight).to(y.dtype)
        y.index_add_(0, token_idx, expert_y)
    return y


def routed_moe_forward(
    x: torch.Tensor,
    gate: nn.Module,
    experts: Sequence[nn.Module],
    *,
    num_experts_per_tok: int,
    norm_topk_prob: bool,
    act_fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route tokens and return ``(output, router_scores, selected_experts)``."""
    batch_size, seq_len, hidden_dim = x.shape
    x_flat = x.reshape(-1, hidden_dim)
    scores = F.softmax(gate(x_flat), dim=-1)
    topk_weight, topk_idx = torch.topk(
        scores, k=num_experts_per_tok, dim=-1, sorted=False
    )
    if norm_topk_prob:
        topk_weight = topk_weight / (
            topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        )

    if _can_use_grouped_mm(x_flat, experts):
        y = _grouped_expert_forward(
            x_flat, topk_idx, topk_weight, experts, act_fn
        )
    else:
        y = _synchronization_free_expert_loop(
            x_flat, topk_idx, topk_weight, experts
        )
    return y.view(batch_size, seq_len, hidden_dim), scores, topk_idx
