"""Correctness tests for synchronization-free and grouped MoE dispatch."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_instinct import MOEFeedForward
from model.moe_dispatch import _sorted_expert_offsets, _stacked_expert_weights
from tests.helpers import make_tiny_config


@pytest.mark.parametrize('counts', [
    [0] * 8,                         # no routes
    [19] + [0] * 7,                  # trailing empty experts
    [0] * 7 + [19],                  # leading empty experts
    [0, 3, 0, 1, 0, 0, 12, 0],      # internal gaps and skew
    [4] * 8,
    [1],
])
def test_sorted_expert_offsets_match_histogram(counts):
    ids = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    expected = torch.bincount(ids, minlength=len(counts)).cumsum(0, dtype=torch.int32)
    actual = _sorted_expert_offsets(ids, len(counts))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.dtype == torch.int32
    assert actual.device == ids.device


@pytest.mark.parametrize('top_k', [1, 2, 8])
def test_sorted_expert_offsets_for_topk_routes(top_k):
    generator = torch.Generator().manual_seed(23)
    choices = torch.rand(37, 8, generator=generator).topk(top_k, dim=-1).indices
    ids = choices.flatten().sort().values
    actual = _sorted_expert_offsets(ids, 8)
    expected = torch.bincount(ids, minlength=8).cumsum(0, dtype=torch.int32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual[-1] == 37 * top_k


@pytest.mark.gpu
def test_cuda_expert_offsets_do_not_synchronize():
    cpu_ids = torch.tensor([0, 0, 2, 2, 2, 7], dtype=torch.long)
    expected = torch.bincount(cpu_ids, minlength=8).cumsum(0, dtype=torch.int32)
    ids = cpu_ids.cuda()
    _sorted_expert_offsets(ids, 8)  # warm up allocation/kernel initialization
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode('error')
        actual = _sorted_expert_offsets(ids, 8)
        empty = _sorted_expert_offsets(ids[:0], 8)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(empty.cpu(), torch.zeros(8, dtype=torch.int32))


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
@pytest.mark.parametrize('top_k,empty_experts', [(1, False), (2, False), (1, True)])
def test_grouped_cuda_matches_legacy_forward_and_backward(top_k, empty_experts):
    """Native BF16 grouped GEMM preserves routed output and parameter grads."""
    torch.manual_seed(7)
    grouped = MOEFeedForward(make_tiny_config(use_moe=True)).cuda().train()
    grouped.config.num_experts_per_tok = top_k
    if empty_experts:
        with torch.no_grad():
            grouped.gate.weight.fill_(-1)
            grouped.gate.weight[0].fill_(1)
    reference = copy.deepcopy(grouped).cuda().train()
    x_grouped = (torch.ones if empty_experts else torch.randn)(
        2, 64, 64, device="cuda", requires_grad=True
    )
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


def test_stacked_expert_weights_cache_reuse_and_invalidation():
    """Inference forwards reuse the stack; any weight change forces a rebuild."""
    module = MOEFeedForward(make_tiny_config(use_moe=True)).eval()
    experts = module.experts
    experts._inference_stack_cache = {}

    def expected_stack():
        return torch.stack([e.gate_proj.weight for e in experts], 0).transpose(1, 2)

    with torch.no_grad():
        first = _stacked_expert_weights(experts, "gate_proj", torch.float32)
        reused = _stacked_expert_weights(experts, "gate_proj", torch.float32)
    assert reused is first
    torch.testing.assert_close(reused, expected_stack())

    with torch.no_grad():  # in-place load (load_state_dict) bumps the version
        experts[1].gate_proj.weight.copy_(torch.randn_like(experts[1].gate_proj.weight))
        after_update = _stacked_expert_weights(experts, "gate_proj", torch.float32)
    assert after_update is not first
    torch.testing.assert_close(after_update, expected_stack())

    experts[0].gate_proj.weight = nn.Parameter(  # replaced tensors fail identity
        torch.randn_like(experts[0].gate_proj.weight)
    )
    with torch.no_grad():
        after_replace = _stacked_expert_weights(experts, "gate_proj", torch.float32)
    assert after_replace is not after_update

    fresh = _stacked_expert_weights(experts, "gate_proj", torch.float32)
    assert fresh is not after_replace  # grad-enabled forwards bypass the cache


@pytest.mark.gpu
def test_grouped_eval_forward_with_stack_cache_matches_legacy():
    """Cached decode path (eval + no-grad + BF16) keeps the routed output."""
    torch.manual_seed(11)
    module = MOEFeedForward(make_tiny_config(use_moe=True)).cuda().to(torch.bfloat16).eval()
    module.experts._inference_stack_cache = {}
    reference = copy.deepcopy(module)
    x = torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        first = module(x)
        second = module(x)  # expert stacks served from the cache
        expected, _ = _legacy_reference(reference, x)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.float(), expected.float(), rtol=2e-2, atol=2e-2)
