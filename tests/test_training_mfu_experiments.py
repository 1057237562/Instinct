import pytest
import torch

from experiments.bench_training_mfu import gram_candidate, useful_flops
from model.model_instinct import InstinctConfig
from trainer.batched_muon import _NS_COEFFICIENTS
from tests.test_gram_newton_schulz import gram_zeropower, standard_zeropower


@pytest.mark.parametrize('shape', [(2, 32, 100), (2, 100, 32), (2, 32, 32), (2, 8, 32)])
@pytest.mark.parametrize('steps', [0, 1, 3, 5, 7])
def test_optimized_gram_preserves_polynomial_in_float64(shape, steps):
    torch.manual_seed(128)
    update = torch.randn(shape, dtype=torch.float64)
    options = dict(ns_coefficients=_NS_COEFFICIENTS, ns_steps=steps, eps=1e-7, dtype=torch.float64)
    expected = standard_zeropower(update, **options)
    actual = gram_candidate(update, **options)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_optimized_gram_matches_independent_stabilized_reference(dtype):
    torch.manual_seed(73)
    update = torch.randn(3, 32, 100)
    options = dict(ns_coefficients=_NS_COEFFICIENTS, ns_steps=5, eps=1e-7, dtype=dtype)
    actual = gram_candidate(update, **options)
    expected = gram_zeropower(update, restart_after=3, **options)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mfu_counts_only_active_experts_and_useful_causal_pairs():
    cfg = InstinctConfig(hidden_size=512, num_hidden_layers=32, num_attention_heads=16,
                         head_dim=32, use_moe=True, num_experts=8, num_experts_per_tok=1,
                         moe_intermediate_size=1664)
    total = 678726144
    active, flops = useful_flops(cfg, total, batch=4, seq=2048, segments=4)
    assert active == total - 32 * 7 * 3 * 512 * 1664
    pairs = 4 * 4 * 512 * 513 // 2
    assert flops == 6 * active * 4 * 2048 + 12 * 32 * 16 * 32 * pairs
    _, unpacked = useful_flops(cfg, total, 4, 2048, 1)
    assert unpacked > flops
