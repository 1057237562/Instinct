"""Selective attention recomputation — gradient-checkpointing Mode 1.

``RecomputeAttention`` keeps only Q/K/V (plus attention mask and RNG state)
across the explicit eager-attention fallback and recomputes its core in
backward. The fused FA4/SDPA fast path performs its own memory-efficient
attention and is intentionally not wrapped here; mode 1 checkpoints its FFN.

Semantics are a byte-for-byte re-implementation of the eager math-attention
branch of ``Attention.forward`` in ``model/model_instinct.py:142-147``;
the exact math lives in ``_attention_core`` below.

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
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from model.sequence_packing import apply_attention_mask


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


def _get_rng_state() -> List[torch.Tensor]:
    """Snapshot the generators used by dropout: CPU always, current CUDA if any."""
    states = [torch.get_rng_state()]
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def _set_rng_state(states: List[torch.Tensor]) -> None:
    """Restore the RNG state saved by ``_get_rng_state`` (CPU always, CUDA if any).

    Called at the top of ``backward`` so the replayed dropout mask is bitwise
    identical to the one forward used.
    """
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
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
        dropout_p: float,
        scale: float,
    ) -> torch.Tensor:
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
    def _attention_core(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
        dropout_p: float,
        scale: float,
        n_rep: int,
    ) -> torch.Tensor:
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
            scores = apply_attention_mask(scores, attention_mask)
        out = F.dropout(F.softmax(scores.float(), dim=-1).type_as(q_t), p=dropout_p) @ v_t
        return out.transpose(1, 2).reshape(bs, seq_len, -1)

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Restore the forward RNG state, recompute the core under ``enable_grad``,
        and return first-order grads for q/k/v. Non-differentiable inputs
        (attention_mask, is_causal, dropout_p, scale) map to ``None``.
        """
        q, k, v, attention_mask = ctx.saved_tensors
        _set_rng_state(ctx.rng_state)
        with torch.enable_grad():
            output = RecomputeAttention._attention_core(
                q, k, v, attention_mask, ctx.is_causal, ctx.dropout_p, ctx.scale, ctx.n_rep
            )
            # Differentiate only the tensors that require grad — a frozen q/k/v
            # would make torch.autograd.grad raise otherwise.
            grads = [None, None, None]
            diff = [(i, t) for i, t in enumerate((q, k, v)) if t.requires_grad]
            if diff:
                idxs = [i for i, _ in diff]
                tensors = [t for _, t in diff]
                got = torch.autograd.grad(
                    output, tensors, grad_outputs=grad_out, allow_unused=True
                )
                for i, g in zip(idxs, got):
                    grads[i] = g
        return (grads[0], grads[1], grads[2], None, None, None, None)


def recompute_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    head_dim: Optional[int] = None,
) -> torch.Tensor:
    """Functional wrapper: convenience API for the T7 integration.

    ``scale`` is derived from ``head_dim`` (defaults to ``q.shape[-1]``).
    """
    if head_dim is None:
        head_dim = q.shape[-1]
    return RecomputeAttention.apply(q, k, v, attention_mask, is_causal, dropout_p, 1.0 / math.sqrt(head_dim))


def checkpoint_ffn(
    ffn_module: torch.nn.Module,
    hidden_states: torch.Tensor,
    use_reentrant: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Gradient-checkpoint the FFN/MoE block (plan T3) — recompute, don't stash.

    Wraps ``FeedForward`` (dense SwiGLU) or ``MOEFeedForward`` (top-1 routing)
    in ``torch.utils.checkpoint`` so the large intermediate activations
    (``[bs*seq, intermediate]``, expert expert-activations) are never kept
    alive between forward and backward: only ``hidden_states`` is saved, and
    the whole FFN forward is re-run inside backward under ``torch.enable_grad``.

    Returns ``(output, aux_loss)``. For MoE, ``aux_loss`` is the router
    auxiliary-load loss, surfaced **through the return value** rather than the
    module side-channel attribute ``ffn_module.aux_loss``. Dense ``FeedForward``
    has no ``aux_loss`` attribute, so ``getattr(..., None)`` yields ``None`` and
    the dense path simply ignores it.

    Key semantics (the tests pin these down):

    * **``torch.utils.checkpoint``, not a custom ``torch.autograd.Function``.**
      A custom Function's forward runs under ``torch.no_grad()``, so any
      parameter gradient (``gate_proj``/``up_proj``/``down_proj``, MoE router
      ``gate``, expert weights) computed against its outputs is silently
      dropped. ``checkpoint`` has no such trap: the forward runs with autograd
      attached and parameter gradients accumulate normally.
    * **``use_reentrant=False``.** The non-reentrant variant supports tuple /
      non-tensor return values, and its backward re-runs ``run`` under
      ``torch.enable_grad()`` then calls ``torch.autograd.backward`` on **all**
      returned outputs — including ``aux_loss``. That is what carries the aux
      gradient back to ``gate.weight``. (``use_reentrant=True`` is deprecated
      and rejects in-place / tuple outputs.)
    * **``preserve_rng_state=True``.** Forward saves the generator state before
      any dropout; backward restores it so the recomputed dropout mask is
      bitwise identical to the one forward used.
    """
    def run(h):
        out = ffn_module(h)
        aux = getattr(ffn_module, "aux_loss", None)
        return out, aux

    return torch.utils.checkpoint.checkpoint(
        run,
        hidden_states,
        use_reentrant=use_reentrant,
        preserve_rng_state=True,
    )
