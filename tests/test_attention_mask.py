import pytest
import torch

from model.attention_mask import apply_attention_mask, normalize_attention_mask


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_attention_mask_fill_value_supports_score_dtype(dtype):
    scores = torch.zeros((1, 1, 2, 2), dtype=dtype)
    attention_mask = torch.tensor([[1, 0]])

    masked = apply_attention_mask(scores, attention_mask)

    assert masked[0, 0, 0, 0] == 0
    assert masked[0, 0, 0, 1] == torch.finfo(dtype).min
    assert torch.isfinite(masked).all()


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4), (2, 1, 3, 4)])
def test_normalize_attention_mask_returns_boolean_4d_mask(shape):
    attention_mask = torch.ones(shape, dtype=torch.int64)

    normalized = normalize_attention_mask(attention_mask)

    assert normalized.dtype == torch.bool
    assert normalized.ndim == 4
