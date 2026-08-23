"""Selective attention recomputation — gradient-checkpointing Mode 1 (plan T2).

``RecomputeAttention`` is a custom ``torch.autograd.Function`` that keeps only
Q/K/V (plus attention mask and RNG state) across the forward pass and
recomputes the attention core in backward:

    QK^T -> *scale -> causal / attention mask -> softmax -> dropout -> @V

The large intermediates (scores ``[bs, heads, seq, seq]``, softmax probabilities,
dropout mask) are never saved — the memory win grows with ``seq``.

Semantics are a byte-for-byte re-implementation of the eager math-attention
branch of ``Attention.forward`` in ``model/model_instinct.py:142-147``:

    q_t = q.transpose(1, 2)
    k_t = repeat_kv(k, n_rep).transpose(1, 2)
    v_t = repeat_kv(v, n_rep).transpose(1, 2)
    scores = (q_t @ k_t.transpose(-2, -1)) * scale          # eager: / sqrt(head_dim)
    if is_causal:
        scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"),
                                                  device=scores.device).triu(1)
    if attention_mask is not None:
        scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
    out = F.dropout(F.softmax(scores.float(), dim=-1).type_as(q_t), p=dropout_p) @ v_t
    out = out.transpose(1, 2).reshape(bs, seq_len, -1)

Key constraints honored here:

* **No parameterized layers inside the Function.** QKV projections, RoPE, norms
  and o_proj all live outside — otherwise their parameter gradients would be
  computed under the forward's ``no_grad`` and silently dropped.
* **``scale`` is ``1/sqrt(head_dim)`` passed by the caller.** For head_dim a
  power of two (16 in the tiny config) ``* scale`` is bitwise identical to the
  eager ``/ sqrt(head_dim)``, so tests can assert ``torch.equal``.
* **RNG replay.** forward saves the CPU (and CUDA, if available) generator state
  *before* dropout; backward restores it so the dropout mask is bitwise
  identical to the one forward used.
* **Autograd isolation.** forward runs under ``torch.no_grad()`` so nothing is
  kept alive; backward re-runs the core under ``torch.enable_grad()`` and
  differentiates with ``torch.autograd.grad``, returning first-order grads.
"""

import math

import torch
import torch.nn.functional as F


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Byte-copy of ``repeat_kv`` (model/model_instinct.py:95-98).

    Defined locally (not imported) to avoid a circular import once
    ``model_instinct`` starts importing this module (T7 integration).
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(
        bs, slen, num_key_value_heads * n_rep, head_dim
    )


def _get_rng_state():
    """Snapshot the generators used by dropout: CPU always, current CUDA if any."""
    states = [torch.get_rng_state()]
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def _set_rng_state(states):
    torch.set_rng_state(states[0])
    if torch.cuda.is_available() and len(states) > 1:
        torch.cuda.set_rng_state(states[1])


class RecomputeAttention(torch.autograd.Function):
    """Recompute the attention core in backward; save only Q/K/V + RNG state.

    Inputs (all pre-RoPE, pre-transpose, matching the layout produced by
    ``Attention.forward`` before the eager branch):

        q: [bs, seq, n_heads, head_dim]
        k: [bs, seq, n_kv_heads, head_dim]
        v: [bs, seq, n_kv_heads, head_dim]
        attention_mask: [bs, seq] 0/1 tensor (or None)
        is_causal: bool — apply the ``triu(1)`` -inf causal mask
        dropout_p: float — attention-dropout probability (caller passes 0.0
            when the model is in eval mode)
        scale: float — 1/sqrt(head_dim)

    Returns [bs, seq, n_heads * head_dim], identical to the eager branch.
    """

    @staticmethod
    def forward(ctx, q, k, v, attention_mask, is_causal, dropout_p, scale):
        n_rep = q.shape[2] // k.shape[2]  # GQA: n_heads / n_kv_heads
        ctx.save_for_backward(q, k, v, attention_mask)
        ctx.is_causal = bool(is_causal)
        ctx.dropout_p = float(dropout_p)
        ctx.scale = float(scale)
        ctx.n_rep = n_rep
        # Save RNG *before* dropout runs so backward replays the identical mask.
        ctx.rng_state = _get_rng_state()
        with torch.no_grad():
            output = RecomputeAttention._attention_core(
                q, k, v, attention_mask, ctx.is_causal, ctx.dropout_p, ctx.scale, n_rep
            )
        return output

    @staticmethod
    def _attention_core(q, k, v, attention_mask, is_causal, dropout_p, scale, n_rep):
        """Eager math-attention core, ``* scale`` instead of ``/ sqrt(head_dim)``."""
        bs, seq_len = q.shape[0], q.shape[1]
        q_t = q.transpose(1, 2)
        k_t = _repeat_kv(k, n_rep).transpose(1, 2)
        v_t = _repeat_kv(v, n_rep).transpose(1, 2)
        scores = (q_t @ k_t.transpose(-2, -1)) * scale
        if is_causal:
            scores[:, :, :, -seq_len:] += torch.full(
                (seq_len, seq_len), float("-inf"), device=scores.device
            ).triu(1)
        if attention_mask is not None:
            scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        out = F.dropout(F.softmax(scores.float(), dim=-1).type_as(q_t), p=dropout_p) @ v_t
        return out.transpose(1, 2).reshape(bs, seq_len, -1)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, attention_mask = ctx.saved_tensors
        _set_rng_state(ctx.rng_state)  # replay forward's exact dropout mask
        with torch.enable_grad():
            output = RecomputeAttention._attention_core(
                q, k, v, attention_mask, ctx.is_causal, ctx.dropout_p, ctx.scale, ctx.n_rep
            )
            # Differentiate only the inputs that actually require grad; a frozen
            # q/k/v would make torch.autograd.grad raise otherwise.
            grads = {0: None, 1: None, 2: None}
            diff = [(i, t) for i, t in enumerate((q, k, v)) if t.requires_grad]
            if diff:
                idxs = [i for i, _ in diff]
                tensors = [t for _, t in diff]
                got = torch.autograd.grad(output, tensors, grad_outputs=grad_out, allow_unused=True)
                for i, g in zip(idxs, got):
                    grads[i] = g
        return (grads[0], grads[1], grads[2], None, None, None, None)


def recompute_attention(q, k, v, attention_mask=None, is_causal=True, dropout_p=0.0, head_dim=None):
    """Functional wrapper: convenience API for the T7 integration.

    ``scale`` is derived from ``head_dim`` (defaults to ``q.shape[-1]``).
    """
    if head_dim is None:
        head_dim = q.shape[-1]
    return RecomputeAttention.apply(q, k, v, attention_mask, is_causal, dropout_p, 1.0 / math.sqrt(head_dim))
