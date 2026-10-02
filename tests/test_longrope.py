import math

import pytest
import torch

from model.rope import (
    build_rope_caches,
    precompute_freqs_cis,
    select_rope_cache,
    validate_rope_scaling,
)
from scripts.search_longrope import Candidate, make_rope_scaling, project_factors


def _longrope_config(**overrides):
    config = {
        "type": "longrope",
        "factor": 4.0,
        "original_max_position_embeddings": 4,
        "short_factor": [1.0, 1.0, 1.0, 1.0],
        "long_factor": [1.0, 2.0, 3.0, 4.0],
        "short_retained_start_tokens": 0,
        "long_retained_start_tokens": 2,
        "attention_factor": 1.0,
    }
    config.update(overrides)
    return config


def test_longrope_uses_per_dimension_factors_and_retains_initial_positions():
    native_cos, native_sin = precompute_freqs_cis(8, 16, 10_000.0, None)
    long_cos, long_sin = precompute_freqs_cis(
        8, 16, 10_000.0, _longrope_config(), long_context=True
    )

    torch.testing.assert_close(long_cos[:2], native_cos[:2])
    torch.testing.assert_close(long_sin[:2], native_sin[:2])
    assert not torch.allclose(long_cos[2], native_cos[2])
    # lambda_0=1 remains unscaled while the remaining rotary pairs differ.
    torch.testing.assert_close(long_cos[2, 0], native_cos[2, 0])
    assert not torch.allclose(long_cos[2, 1:4], native_cos[2, 1:4])


def test_longrope_switches_from_short_to_long_cache_at_original_context():
    scaling = _longrope_config(long_retained_start_tokens=0)
    long_cos, long_sin, short_cos, short_sin = build_rope_caches(
        8, 16, 10_000.0, scaling
    )
    short = select_rope_cache(
        long_cos, long_sin, short_cos, short_sin, scaling,
        start_pos=0, seq_length=4,
    )
    long = select_rope_cache(
        long_cos, long_sin, short_cos, short_sin, scaling,
        start_pos=0, seq_length=5,
    )
    torch.testing.assert_close(short[0], short_cos[:4])
    torch.testing.assert_close(long[0], long_cos[:5])
    assert not torch.allclose(short_cos[3], long_cos[3])

    packed_positions = torch.tensor([[0, 1, 2, 3, 0, 1]])
    packed = select_rope_cache(
        long_cos, long_sin, short_cos, short_sin, scaling,
        start_pos=0, seq_length=6, position_ids=packed_positions,
    )
    torch.testing.assert_close(packed[0], short_cos[packed_positions])


def test_longrope_defaults_to_paper_attention_scaling():
    scaling = _longrope_config(
        attention_factor=None, long_retained_start_tokens=0
    )
    long_cos, _ = precompute_freqs_cis(
        8, 16, 10_000.0, scaling, long_context=True
    )
    expected = math.sqrt(1.0 + math.log(4.0) / math.log(4.0))
    torch.testing.assert_close(long_cos[0], torch.full((8,), expected))


@pytest.mark.parametrize(
    "module_name",
    [
        "model.model_instinct",
        "model.model_instinct_linear",
        "model.model_instinct_loop",
    ],
)
def test_every_instinct_backbone_accepts_longrope(module_name):
    module = __import__(module_name, fromlist=["InstinctConfig"])
    config = module.InstinctConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        inference_rope_scaling=True,
        rope_scaling=_longrope_config(),
    )
    assert config.rope_scaling["type"] == "longrope"
    assert config.rope_scaling["rope_type"] == "longrope"
    assert config.rope_scaling["long_retained_start_tokens"] == 2


def test_longrope_rejects_wrong_factor_vector_length():
    with pytest.raises(ValueError, match="head_dim / 2 = 4"):
        validate_rope_scaling(
            _longrope_config(long_factor=[1.0, 2.0]),
            dim=8,
            max_position_embeddings=16,
        )


def test_longrope_search_projects_to_paper_monotonic_space():
    projected = project_factors([2.03, 1.01, 7.0, 3.0], scale=4.0, step=0.01)
    assert projected == (2.03, 2.03, 5.0, 5.0)


def test_longrope_search_emits_runtime_compatible_config():
    candidate = Candidate((1.0, 2.0, 3.0, 4.0), retained_start_tokens=16)
    scaling = make_rope_scaling(candidate, dim=8, original_max=32, target_length=128)
    assert scaling["long_factor"] == [1.0, 2.0, 3.0, 4.0]
    assert scaling["long_retained_start_tokens"] == 16
    assert scaling["short_factor"] == [1.0] * 4
    assert scaling["short_attention_factor"] == 1.0
