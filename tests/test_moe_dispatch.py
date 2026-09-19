"""Correctness tests for synchronization-free and grouped MoE dispatch."""

import copy

import pytest
import torch
import torch.nn.functional as F

from model.model_instinct import MOEFeedForward
from tests.helpers import make_tiny_config


def _legacy_reference(module, x):
    """Small correctness oracle matching the former per-expert implementation."""
    batch_size, seq_len, hidden_dim = x.shape
    x_flat = x.reshape(-1, hidden_dim)
    scores = F.softmax(module.gate(x_flat), dim=-1)
    topk_weight, topk_idx = torch.topk(
        scores, k=module.config.num_experts_per_tok, dim=-1, sorted=False
    )
    if module.config.norm_topk_prob:
        topk_weight = topk_weight / (
            topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        )
    y = torch.zeros_like(x_flat)
    for expert_id, expert in enumerate(module.experts):
        mask = topk_idx == expert_id
        token_idx = mask.any(dim=-1).nonzero().flatten()
        weight = topk_weight[mask].view(-1, 1)
        y.index_add_(
            0, token_idx,
            (expert(x_flat.index_select(0, token_idx)) * weight).to(y.dtype),
        )
    load = F.one_hot(topk_idx, module.config.num_experts).float().mean(0)
    aux = (
        (load * scores.mean(0)).sum()
        * module.config.num_experts
        * module.config.router_aux_loss_coef
    )
    return y.view(batch_size, seq_len, hidden_dim), aux


def test_empty_experts_receive_zero_grad_without_branch():
    """An empty expert remains in autograd without a device-to-host condition."""
    module = MOEFeedForward(make_tiny_config(use_moe=True)).train()
    with torch.no_grad():
        module.gate.weight.fill_(-1.0)
        module.gate.weight[0].fill_(1.0)

    x = torch.ones(2, 8, 64, requires_grad=True)
    output = module(x)
    (output.sum() + module.aux_loss).backward()

    expert_grad_sums = []
    for expert in module.experts:
        grad = expert.gate_proj.weight.grad
        assert grad is not None
        expert_grad_sums.append(grad.abs().sum().item())
    assert expert_grad_sums[0] > 0
    assert expert_grad_sums[1:] == [0.0] * (len(module.experts) - 1)


@pytest.mark.gpu
def test_grouped_cuda_matches_legacy_forward_and_backward():
    """Native BF16 grouped GEMM preserves routed output and parameter grads."""
    torch.manual_seed(7)
    grouped = MOEFeedForward(make_tiny_config(use_moe=True)).cuda().train()
    reference = copy.deepcopy(grouped).cuda().train()
    x_grouped = torch.randn(2, 64, 64, device="cuda", requires_grad=True)
    x_reference = x_grouped.detach().clone().requires_grad_(True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        grouped_output = grouped(x_grouped)
        reference_output, reference_aux = _legacy_reference(reference, x_reference)
    grouped_loss = grouped_output.float().square().mean() + grouped.aux_loss
    reference_loss = reference_output.float().square().mean() + reference_aux
    grouped_loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(
        grouped_output.float(), reference_output.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        x_grouped.grad, x_reference.grad, rtol=3e-2, atol=3e-3
    )
    for (grouped_name, grouped_param), (reference_name, reference_param) in zip(
        grouped.named_parameters(), reference.named_parameters()
    ):
        assert grouped_name == reference_name
        assert grouped_param.grad is not None, grouped_name
        assert reference_param.grad is not None, reference_name
        torch.testing.assert_close(
            grouped_param.grad,
            reference_param.grad,
            rtol=5e-2,
            atol=5e-3,
            msg=lambda message, name=grouped_name: f"{name}: {message}",
        )
