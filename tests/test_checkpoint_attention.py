"""TDD tests for ``RecomputeAttention`` (selective attention recompute, plan T2).

RED first: this file was written before ``model/checkpointing.py`` existed, so the
import below fails to collect → the suite is red. After implementing
``RecomputeAttention`` the suite must go fully green, with every assertion using
``torch.equal`` (bitwise) against a hand-written eager reference that is a
byte-for-byte copy of the math-attention branch of ``Attention.forward``
(``model/model_instinct.py:142-147``).

All tests run on CPU fp32. head_dim = 16 so ``* (1/sqrt(16)) == / sqrt(16)``
exactly (both are an exact power-of-two scale) and ``torch.equal`` holds.
"""

import math

import torch
import torch.nn.functional as F

from model.checkpointing import RecomputeAttention


# ---------------------------------------------------------------------------
# Oracles — byte-for-byte copies of the eager branch semantics
# ---------------------------------------------------------------------------

def _repeat_kv(x, n_rep):
    """Byte-copy of ``repeat_kv`` in model/model_instinct.py:95-98."""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(
        bs, slen, num_key_value_heads * n_rep, head_dim
    )


def eager_attention(q, k, v, attention_mask, is_causal, dropout_p):
    """Eager math-attention oracle (model_instinct.py:142-147), ``/ sqrt(head_dim)``.

    q/k/v are pre-RoPE, pre-transpose [bs, seq, heads, hd]. Returns
    [bs, seq, n_heads * hd] exactly like the model's eager branch.
    """
    n_rep = q.shape[2] // k.shape[2]
    head_dim = q.shape[-1]
    bs, seq_len = q.shape[0], q.shape[1]
    q_t = q.transpose(1, 2)
    k_t = _repeat_kv(k, n_rep).transpose(1, 2)
    v_t = _repeat_kv(v, n_rep).transpose(1, 2)
    scores = (q_t @ k_t.transpose(-2, -1)) / math.sqrt(head_dim)
    if is_causal:
        scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
    if attention_mask is not None:
        if attention_mask.ndim == 2:
            scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        else:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).bool(), -1e9)
    out = F.dropout(F.softmax(scores.float(), dim=-1).type_as(q_t), p=dropout_p) @ v_t
    return out.transpose(1, 2).reshape(bs, seq_len, -1)


def eager_backward(q, k, v, attention_mask, is_causal, dropout_p):
    """Gradients of ``out.sum()`` through the eager reference."""
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2.requires_grad = True
    k2.requires_grad = True
    v2.requires_grad = True
    out = eager_attention(q2, k2, v2, attention_mask, is_causal, dropout_p)
    out.sum().backward()
    return q2.grad, k2.grad, v2.grad


def _make_qkv(bs=2, seq=6, n_heads=4, n_kv_heads=2, hd=16, seed=0):
    """Deterministic random q/k/v (pre-RoPE, pre-transpose layout)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(bs, seq, n_heads, hd, generator=g)
    k = torch.randn(bs, seq, n_kv_heads, hd, generator=g)
    v = torch.randn(bs, seq, n_kv_heads, hd, generator=g)
    return q, k, v


def _scale(hd):
    return 1.0 / math.sqrt(hd)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_causal():
    """is_causal=True, dropout=0, no mask, GQA n_rep=2 → bitwise equal to eager."""
    q, k, v = _make_qkv(n_heads=4, n_kv_heads=2)
    out = RecomputeAttention.apply(q, k, v, None, True, 0.0, _scale(q.shape[-1]))
    ref = eager_attention(q, k, v, None, True, 0.0)
    assert out.shape == (2, 6, 4 * 16)
    assert torch.equal(out, ref)


def test_output_no_causal_with_mask():
    """is_causal=False with a 0/1 padding mask → bitwise equal to eager."""
    q, k, v = _make_qkv()
    mask = torch.ones(2, 6)
    mask[0, 4:] = 0  # pad the last two positions of batch 0
    out = RecomputeAttention.apply(q, k, v, mask, False, 0.0, _scale(q.shape[-1]))
    ref = eager_attention(q, k, v, mask, False, 0.0)
    assert torch.equal(out, ref)


def test_output_and_grad_with_block_diagonal_mask():
    """Packed [batch, query, key] masks work in selective recomputation."""
    q, k, v = _make_qkv()
    sequence_ids = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]])
    mask = sequence_ids.unsqueeze(-1) == sequence_ids.unsqueeze(-2)
    qg, kg, vg = q.clone(), k.clone(), v.clone()
    qg.requires_grad = kg.requires_grad = vg.requires_grad = True

    out = RecomputeAttention.apply(qg, kg, vg, mask, True, 0.0, _scale(q.shape[-1]))
    ref = eager_attention(q, k, v, mask, True, 0.0)
    assert torch.equal(out, ref)
    out.sum().backward()
    rq, rk, rv = eager_backward(q, k, v, mask, True, 0.0)
    assert torch.equal(qg.grad, rq)
    assert torch.equal(kg.grad, rk)
    assert torch.equal(vg.grad, rv)


def test_grad_output():
    """q/k/v grads (d(out.sum())/d*) bitwise equal to the eager reference."""
    q, k, v = _make_qkv(n_heads=4, n_kv_heads=2)
    qg, kg, vg = q.clone(), k.clone(), v.clone()
    qg.requires_grad = True
    kg.requires_grad = True
    vg.requires_grad = True
    out = RecomputeAttention.apply(qg, kg, vg, None, True, 0.0, _scale(q.shape[-1]))
    out.sum().backward()
    rq, rk, rv = eager_backward(q, k, v, None, True, 0.0)
    assert qg.grad is not None and kg.grad is not None and vg.grad is not None
    assert torch.equal(qg.grad, rq)
    assert torch.equal(kg.grad, rk)
    assert torch.equal(vg.grad, rv)


def test_grad_output_with_mask():
    """Same as test_grad_output but through the masked non-causal path."""
    q, k, v = _make_qkv()
    mask = torch.ones(2, 6)
    mask[1, 5:] = 0
    qg, kg, vg = q.clone(), k.clone(), v.clone()
    qg.requires_grad = True
    kg.requires_grad = True
    vg.requires_grad = True
    out = RecomputeAttention.apply(qg, kg, vg, mask, False, 0.0, _scale(q.shape[-1]))
    out.sum().backward()
    rq, rk, rv = eager_backward(q, k, v, mask, False, 0.0)
    assert torch.equal(qg.grad, rq)
    assert torch.equal(kg.grad, rk)
    assert torch.equal(vg.grad, rv)


def test_dropout_rng():
    """dropout_p=0.1, fixed seed:
    (a) two applies with the same seed are bitwise equal — forward dropout is
        reproducible and the saved RNG state lets backward replay the same mask;
    (b) dropout genuinely differs from the p=0 path;
    (c) backward grads equal a same-seed eager reference — proves the backward
        replay used the *same* dropout mask as forward.
    """
    q, k, v = _make_qkv(seed=123)
    scale = _scale(q.shape[-1])

    torch.manual_seed(7)
    out1 = RecomputeAttention.apply(q, k, v, None, True, 0.1, scale)
    torch.manual_seed(7)
    out2 = RecomputeAttention.apply(q, k, v, None, True, 0.1, scale)
    assert torch.equal(out1, out2)
    ref_no_drop = eager_attention(q, k, v, None, True, 0.0)
    assert not torch.equal(out1, ref_no_drop)  # dropout is actually active

    qg, kg, vg = q.clone(), k.clone(), v.clone()
    qg.requires_grad = True
    kg.requires_grad = True
    vg.requires_grad = True
    torch.manual_seed(7)
    out = RecomputeAttention.apply(qg, kg, vg, None, True, 0.1, scale)
    out.sum().backward()
    torch.manual_seed(7)
    rq, rk, rv = eager_backward(q, k, v, None, True, 0.1)
    assert torch.equal(qg.grad, rq)
    assert torch.equal(kg.grad, rk)
    assert torch.equal(vg.grad, rv)


def test_repeat_kv_identity():
    """n_rep=1 (n_heads == n_kv_heads) matches eager exactly, as does n_rep=2."""
    q, k, v = _make_qkv(n_heads=4, n_kv_heads=4)  # n_rep == 1 → repeat_kv is identity
    out = RecomputeAttention.apply(q, k, v, None, True, 0.0, _scale(q.shape[-1]))
    ref = eager_attention(q, k, v, None, True, 0.0)
    assert torch.equal(out, ref)

    q2, k2, v2 = _make_qkv(n_heads=4, n_kv_heads=2, seed=5)  # n_rep == 2
    out2 = RecomputeAttention.apply(q2, k2, v2, None, True, 0.0, _scale(q2.shape[-1]))
    ref2 = eager_attention(q2, k2, v2, None, True, 0.0)
    assert torch.equal(out2, ref2)
