"""Shared test helpers for the Instinct test suite."""

import torch


def make_tiny_config(use_moe=False, variant="dense"):
    """Return a tiny InstinctConfig for fast CPU tests.

    Each model file defines its own ``InstinctConfig`` class with an identical
    interface (``model_instinct.py`` / ``model_instinct_loop.py`` /
    ``model_instinct_linear.py``). ``model_instinct_loop`` and
    ``model_instinct_linear`` are standalone top-level modules (no ``sys.modules``
    aliasing required), so a plain import is sufficient here. ``flash_attn=False``
    keeps tests on CPU-friendly math-attention. ``head_dim`` resolves
    automatically to ``hidden_size // num_attention_heads = 64 // 4 = 16``.

    Args:
        use_moe: Whether to enable the MoE branch (ignored if the model uses a
            variant without MoE support at runtime).
        variant: One of "dense" (default), "loop", or "linear".
    """
    if variant == "dense":
        from model.model_instinct import InstinctConfig
    elif variant == "loop":
        from model.model_instinct_loop import InstinctConfig
    elif variant == "linear":
        from model.model_instinct_linear import InstinctConfig
    else:
        raise ValueError(f"Unknown variant: {variant!r} (expected 'dense', 'loop' or 'linear')")

    return InstinctConfig(
        hidden_size=64,
        num_hidden_layers=2,
        use_moe=use_moe,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        vocab_size=256,
        flash_attn=False,
        dropout=0.0,
    )


def assert_grads_equal(m0, m1, atol=1e-5, rtol=1e-4):
    """Assert that two models have identical gradients on every parameter.

    fp32 grads must match exactly (``torch.equal``); other dtypes (bf16/fp16)
    are compared with ``torch.allclose(atol, rtol)``. Fails with a per-parameter
    message when a gradient is missing or mismatched.

    Uses ``named_parameters()`` (not ``state_dict()``): ``state_dict()`` returns
    detached copies of the tensors, whose ``.grad`` is always ``None``. Tied
    embeddings appear twice (e.g. ``lm_head.weight`` / ``model.embed_tokens.weight``)
    and are compared independently, which is harmless.
    """
    p0 = dict(m0.named_parameters())
    p1 = dict(m1.named_parameters())
    if set(p0.keys()) != set(p1.keys()):
        only0 = sorted(set(p0) - set(p1))
        only1 = sorted(set(p1) - set(p0))
        raise AssertionError(
            f"parameter names differ; only in m0: {only0}, only in m1: {only1}"
        )
    for name in p0:
        g0 = p0[name].grad
        g1 = p1[name].grad
        if g0 is None or g1 is None:
            raise AssertionError(
                f"missing gradient on parameter '{name}' "
                f"(m0 has grad: {g0 is not None}, m1 has grad: {g1 is not None})"
            )
        if g0.dtype == torch.float32 and g1.dtype == torch.float32:
            if not torch.equal(g0, g1):
                raise AssertionError(f"gradient mismatch on parameter '{name}' (fp32, not bitwise equal)")
        elif not torch.allclose(g0, g1, atol=atol, rtol=rtol):
            diff = (g0 - g1).abs().max().item()
            raise AssertionError(
                f"gradient mismatch on parameter '{name}' "
                f"(atol={atol}, rtol={rtol}, max_abs_diff={diff:.3e})"
            )
