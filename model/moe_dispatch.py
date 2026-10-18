"""Shared MoE routing; grouped-MM backend synchronization is platform dependent."""

from __future__ import annotations

from collections.abc import Sequence
import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import SiLUActivation
from model.grouped_mm import (
    expert_grouped_mm,
    inference_backend,
    inference_grouped_mm,
    make_grouped_mm_plan,
)


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
    """Whether the packed BF16 expert path can run on this device."""
    if (
        not x.is_cuda
        or _cuda_compute_dtype(x) != torch.bfloat16
        or torch.cuda.get_device_capability(x.device)[0] < 8
        or not experts
    ):
        return False
    first = experts[0]
    hidden = first.gate_proj.in_features
    intermediate = first.gate_proj.out_features
    return hidden % 16 == 0 and intermediate % 16 == 0


_STACK_WEIGHTS, _STACK_VERSIONS, _STACK_DTYPE, _STACK_TENSOR = range(4)
"""Field offsets of a stack-cache entry.

The entry is a plain tuple rather than a NamedTuple on purpose: Dynamo recurses
until the stack overflows when a NamedTuple carrying tensors is constructed
inside a traced region compiled with ``dynamic=True``, which is exactly where
the inference MoE forward now lives.
"""


def _stacked_expert_weights(
    experts: Sequence[nn.Module], name: str, dtype: torch.dtype
) -> torch.Tensor:
    """Stack legacy expert weights in grouped-MM right-hand layout.

    Keeping the original Linear modules preserves existing checkpoint keys and
    optimizer state. The transpose produces the per-group column-major layout
    accepted by ``torch.nn.functional.grouped_mm``.

    Inference callers marked by ``optimize_inference`` (via an
    ``_inference_stack_cache`` dict on the experts module) reuse the stack
    across forwards instead of recopying every expert weight each token.
    Entries validate weight-tensor identity and in-place version counters, so
    weight replacement, dtype/device casts, optimizer steps and
    ``load_state_dict`` all force a rebuild. Grad-enabled forwards skip the
    cache entirely: the stack must stay in the autograd graph so expert weight
    gradients keep flowing through the grouped GEMMs.
    """
    weights = [getattr(expert, name).weight for expert in experts]
    cache = (
        None
        if torch.is_grad_enabled()
        else getattr(experts, "_inference_stack_cache", None)
    )
    versions = None
    if cache is not None:
        versions = tuple(weight._version for weight in weights)
        entry = cache.get(name)
        if (
            entry is not None
            and entry[_STACK_DTYPE] == dtype
            and entry[_STACK_VERSIONS] == versions
            and all(stored is weight for stored, weight in zip(entry[_STACK_WEIGHTS], weights))
        ):
            return entry[_STACK_TENSOR]
    pack_setting = os.environ.get('INSTINCT_MOE_FUSED_PACK', 'auto')
    if pack_setting not in ('0', '1', 'auto'):
        raise ValueError('INSTINCT_MOE_FUSED_PACK must be 0, 1 or auto')
    from model.cuda_fastpath import blackwell_training_kernels
    if (torch.is_grad_enabled() and (pack_setting == '1' or
            (pack_setting == 'auto' and blackwell_training_kernels(weights[0].device)))
            and weights[0].is_cuda
            and dtype == torch.bfloat16 and weights[0].dtype == torch.float32
            and 0 < len(weights) <= 16 and all(w.is_contiguous() for w in weights)):
        from model.expert_weight_pack import pack_weights
        stacked = pack_weights(weights, dtype)
    else:
        stacked = torch.stack(weights, dim=0).transpose(1, 2).to(dtype=dtype)
    if cache is not None:
        cache[name] = (tuple(weights), versions, dtype, stacked)
    return stacked


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


def inner_checkpoint_eligible(x, experts, act_fn, mode):
    """Narrow opt-in for recomputing gate/up inside their fused autograd op."""
    setting = os.environ.get('INSTINCT_MOE_INNER_CHECKPOINT', '0')
    if setting not in ('0', '1', '2'):
        raise ValueError('INSTINCT_MOE_INNER_CHECKPOINT must be 0, 1 or 2')
    if setting == '0' or mode != 1 or not torch.is_grad_enabled():
        return False
    from model.cuda_fastpath import blackwell_training_kernels
    from model.grouped_mm import _auto_backend
    backend = os.environ.get('INSTINCT_GROUPED_MM_BACKEND', 'auto')
    if backend == 'auto':
        backend = _auto_backend(x.device)
    if (not blackwell_training_kernels(x.device) or x.dtype != torch.float32
            or backend != 'triton'
            or _cuda_compute_dtype(x) != torch.bfloat16 or len(experts) != 8
            or type(act_fn) not in (SiLUActivation, nn.SiLU)
            or getattr(act_fn, 'inplace', False)):
        return False
    return all(
        expert.gate_proj.in_features == 512 and expert.gate_proj.out_features == 1664
        and all(getattr(expert, name).weight.dtype == torch.float32
                and getattr(expert, name).weight.is_contiguous()
                and getattr(expert, name).weight.requires_grad
                and getattr(expert, name).bias is None
                for name in ('gate_proj', 'up_proj', 'down_proj'))
        for expert in experts)


def _grouped_expert_forward(
    x_flat: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    experts: Sequence[nn.Module],
    act_fn,
    inner_checkpoint=False,
) -> torch.Tensor:
    """Run all experts with three grouped GEMMs and one final scatter-add."""
    num_tokens = x_flat.shape[0]
    num_experts = len(experts)
    top_k = topk_idx.shape[-1]

    routing_setting = os.environ.get('INSTINCT_MOE_FUSED_ROUTING', 'auto')
    if routing_setting not in ('auto','0','1'):
        raise ValueError('INSTINCT_MOE_FUSED_ROUTING must be auto, 0 or 1')
    from model.cuda_fastpath import blackwell_training_kernels
    fused_routing = ((routing_setting == '1' or (routing_setting == 'auto'
                      and blackwell_training_kernels(x_flat.device)))
                     and top_k in (1, 2) and num_experts in (2,4,8,16)
                     and 0 < num_tokens * top_k <= 65536
                     and x_flat.is_contiguous() and x_flat.dtype == torch.float32
                     and topk_idx.is_contiguous() and topk_weight.is_contiguous())
    if fused_routing:
        from model.moe_routing_triton import permutation, gather, combine
        order, inverse, offsets = permutation(topk_idx, num_experts)
        plan = make_grouped_mm_plan(offsets)
        routed_x = gather(x_flat, order, inverse, top_k)
    else:
        route_expert = topk_idx.reshape(-1)
        route_token = torch.arange(
            num_tokens, device=x_flat.device, dtype=torch.long
        ).view(-1, 1).expand(-1, top_k).reshape(-1)

        # Route metadata is shared by all three projections and their backward.
        order = torch.argsort(route_expert, stable=True)
        route_expert = route_expert.index_select(0, order)
        route_token = route_token.index_select(0, order)
        route_weight = topk_weight.reshape(-1).index_select(0, order).unsqueeze(-1)
        offsets = _sorted_expert_offsets(route_expert, num_experts)
        plan = make_grouped_mm_plan(offsets)
        routed_x = x_flat.index_select(0, route_token).to(dtype=torch.bfloat16)

    compute_dtype = torch.bfloat16
    direct_setting=os.environ.get('INSTINCT_MOE_DIRECT_WEIGHTS','0')
    def project(value,name):
        weights=[getattr(expert,name).weight for expert in experts]
        if (direct_setting=='1' and plan.backend=='triton'
                and all(w.dtype==torch.float32 and w.is_contiguous() for w in weights)):
            from model.expert_linear_triton import expert_linear
            return expert_linear(value,weights,offsets)
        return expert_grouped_mm(value,_stacked_expert_weights(experts,name,compute_dtype),plan)
    if inner_checkpoint and plan.backend == 'triton':
        from model.checkpoint_swiglu_triton import checkpoint_swiglu, checkpoint_experts
        from model.expert_linear_triton import expert_linear
        gate_weights = [expert.gate_proj.weight for expert in experts]
        up_weights = [expert.up_proj.weight for expert in experts]
        down_weights = [expert.down_proj.weight for expert in experts]
        if os.environ.get('INSTINCT_MOE_INNER_CHECKPOINT', '0') == '2':
            routed_y = checkpoint_experts(routed_x,gate_weights,up_weights,down_weights,offsets)
        else:
            hidden = checkpoint_swiglu(routed_x,gate_weights,up_weights,offsets)
            routed_y = expert_linear(hidden,down_weights,offsets)
        if fused_routing:
            return combine(routed_y, topk_weight, order, inverse, x_flat.dtype)
        routed_y = (routed_y * route_weight.to(routed_y.dtype)).to(x_flat.dtype)
        y = torch.zeros_like(x_flat)
        y.index_add_(0, route_token, routed_y)
        return y
    gate = project(routed_x,'gate_proj')
    up = project(routed_x,'up_proj')
    swiglu_setting = os.environ.get('INSTINCT_MOE_FUSED_SWIGLU', 'auto')
    if swiglu_setting not in ('0', '1', 'auto'):
        raise ValueError('INSTINCT_MOE_FUSED_SWIGLU must be 0, 1 or auto')
    from model.cuda_fastpath import blackwell_training_kernels
    if ((swiglu_setting == '1' or (swiglu_setting == 'auto'
            and blackwell_training_kernels(gate.device)))
            and type(act_fn) in (SiLUActivation, nn.SiLU)
            and not getattr(act_fn, 'inplace', False)
            and gate.is_cuda and gate.dtype == torch.bfloat16
            and gate.is_contiguous() and up.is_contiguous() and gate.numel() > 0):
        from model.swiglu_triton import swiglu
        hidden = swiglu(gate, up)
    else:
        hidden = act_fn(gate) * up
    routed_y = project(hidden,'down_proj')
    if fused_routing:
        return combine(routed_y, topk_weight, order, inverse, x_flat.dtype)
    routed_y = (routed_y * route_weight.to(routed_y.dtype)).to(x_flat.dtype)

    y = torch.zeros_like(x_flat)
    y.index_add_(0, route_token, routed_y)
    return y


def _inference_expert_weights(experts: Sequence[nn.Module], dtype: torch.dtype):
    """Gate/up/down stacks for the traced inference forward.

    ``_stacked_expert_weights`` stays the single source of truth for identity
    and version validation, so this only batches its three lookups into one call.
    """
    return (
        _stacked_expert_weights(experts, "gate_proj", dtype),
        _stacked_expert_weights(experts, "up_proj", dtype),
        _stacked_expert_weights(experts, "down_proj", dtype),
    )


def refresh_inference_stacks(modules: Sequence[nn.Module], dtype=torch.bfloat16) -> None:
    """Refresh the weight stacks read by the traced inference forward.

    Must run outside any compiled region, which is why it is called from the
    eager ``InstinctForCausalLM.forward``: the ``_version`` comparisons that make
    the stack cache safe are unguessable to Dynamo (it cannot guard on those
    symbolic ints), and comparing them inside the graph cost one graph break per
    expert projection per layer. The graph itself only reads the resulting tuple,
    so an in-place ``load_state_dict`` still takes effect on the next token.
    """
    for module in modules:
        module._inference_stacked = _inference_expert_weights(module.experts, dtype)


def _inference_grouped_expert_forward(
    x_flat: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    experts: Sequence[nn.Module],
    act_fn,
    backend: str,
    stacks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Grouped expert forward that stays inside ``torch.compile``.

    Same maths and same routing metadata as ``_grouped_expert_forward``, but it
    calls no autograd Function, no ``torch.autocast`` context, no plan object
    with host-side expert ends and no cache validation, so Dynamo can trace the
    whole block and Inductor can fuse the routing and the final scatter with the
    surrounding layers. ``backend`` and ``stacks`` are both resolved outside the
    graph by ``optimize_inference`` and ``refresh_inference_stacks``.
    """
    num_tokens = x_flat.shape[0]
    num_experts = len(experts)
    top_k = topk_idx.shape[-1]

    route_expert = topk_idx.reshape(-1)
    route_token = torch.arange(
        num_tokens, device=x_flat.device, dtype=torch.long
    ).view(-1, 1).expand(-1, top_k).reshape(-1)

    order = torch.argsort(route_expert, stable=True)
    route_expert = route_expert.index_select(0, order)
    route_token = route_token.index_select(0, order)
    route_weight = topk_weight.reshape(-1).index_select(0, order).unsqueeze(-1)
    offsets = _sorted_expert_offsets(route_expert, num_experts)

    compute_dtype = torch.bfloat16
    gate_stack, up_stack, down_stack = stacks
    routed_x = x_flat.index_select(0, route_token).to(dtype=compute_dtype)
    gate = inference_grouped_mm(routed_x, gate_stack, offsets, backend)
    up = inference_grouped_mm(routed_x, up_stack, offsets, backend)
    hidden = act_fn(gate) * up
    routed_y = inference_grouped_mm(hidden, down_stack, offsets, backend)
    routed_y = (routed_y * route_weight.to(routed_y.dtype)).to(x_flat.dtype)

    y = torch.zeros_like(x_flat)
    y.index_add_(0, route_token, routed_y)
    return y


# PyTorch 2.13 Inductor can select an invalid TMA layout for the grouped-MM
# weight-gradient on Blackwell consumer GPUs. Keep this compact region eager:
# Dynamo makes one graph break per MoE block; our device-offset kernels avoid
# that lowering and provide explicit first-order backward kernels. This also
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
    inference_backend: str | None = None,
    inference_stacks=None,
    checkpoint_mode=0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route tokens and return ``(output, router_scores, selected_experts)``.

    ``inference_backend``/``inference_stacks`` are installed by
    ``optimize_inference`` on inference models; when present and gradients are
    off, the expert dispatch takes the traceable path so ``torch.compile`` can
    keep the whole MoE block in one graph. Both are plain values resolved
    outside the graph: resolving them here would put an ``lru_cache``-wrapped
    callable and un-guardable ``_version`` comparisons inside the traced region.
    """
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
        traceable = (
            inference_backend in ('native', 'triton')
            and inference_stacks is not None
            and not torch.is_grad_enabled()
        )
        if traceable:
            y = _inference_grouped_expert_forward(
                x_flat, topk_idx, topk_weight, experts, act_fn,
                inference_backend, inference_stacks,
            )
        else:
            y = _grouped_expert_forward(
                x_flat, topk_idx, topk_weight, experts, act_fn,
                inner_checkpoint=inner_checkpoint_eligible(x_flat, experts, act_fn, checkpoint_mode),
            )
    else:
        y = _synchronization_free_expert_loop(
            x_flat, topk_idx, topk_weight, experts
        )
    return y.view(batch_size, seq_len, hidden_dim), scores, topk_idx
