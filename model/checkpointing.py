"""Attention-only recomputation and explicit experimental checkpoint helpers.

``RecomputeAttention`` keeps only Q/K/V (plus attention mask and RNG state)
across the explicit eager-attention fallback and recomputes its core in
backward. The fused FA4/SDPA fast path performs its own memory-efficient
attention. Mode 1 wraps that fused core with PyTorch's non-reentrant checkpoint
at the model call site, preserving its backend. QKV/O projections and FFNs are
not checkpointed by mode 1. ``checkpoint_ffn`` remains an explicit experiment
helper; the models do not call it as part of mode 1.

Mode 3 uses native selective activation checkpointing around each standard
Dense block: keep GEMM and supported fused-attention results, recompute cheap
intermediates. Modes 0/1/2 retain their original behavior.
Mode 4 bounds the retained outputs to a 1.75 GiB analytical cache allowance.
Long contexts prioritize fused attention; shorter contexts retain compact K/V.
The remaining allowance retains gate/up in upper layers, releasing them early
in backward. This allowance is not a total GPU-memory or allocator guarantee.

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
import os
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from model.attention_mask import apply_attention_mask


DENSE_RECIPE_CACHE_BYTES = 1792 * 2**20


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

    Wraps ``FeedForward`` (dense SwiGLU) or ``MOEFeedForward``
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

    if (not use_reentrant and hasattr(ffn_module, 'experts')
            and hasattr(ffn_module, 'config') and ffn_module.training):
        from model.moe_dispatch import inner_checkpoint_eligible
        if inner_checkpoint_eligible(hidden_states, ffn_module.experts,
                ffn_module.act_fn, ffn_module.config.use_grad_checkpoint):
            # Gate/up activations are recomputed by the fused operator's backward.
            # Avoid wrapping the entire FFN and recomputing it a second time.
            return run(hidden_states)

    options = {}
    selective = os.environ.get('INSTINCT_MOE_SELECTIVE_CHECKPOINT', '0')
    if selective not in ('0','1','2','3','down','gate_down'):
        raise ValueError('INSTINCT_MOE_SELECTIVE_CHECKPOINT must be 0, 1, 2, 3, down or gate_down')
    if selective != '0' and hasattr(ffn_module, 'experts') and not use_reentrant:
        from functools import partial
        options['context_fn'] = partial(_moe_checkpoint_contexts, selective)
    return torch.utils.checkpoint.checkpoint(
        run,
        hidden_states,
        use_reentrant=use_reentrant,
        preserve_rng_state=True,
        **options,
    )


def _moe_checkpoint_contexts(count):
    from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
    counters = {False: 0, True: 0}
    selected={'down':(2,), 'gate_down':(0,2)}.get(str(count))
    if selected is None:
        selected=tuple(range(int(count)))
    def policy(ctx, op, *args, **kwargs):
        if op._schema.name in ('instinct::checkpoint_expert_mm','instinct::checkpoint_direct_mm'):
            index = counters[ctx.is_recompute]
            counters[ctx.is_recompute] += 1
            if index % 3 in selected:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE
    return create_selective_checkpoint_contexts(policy)


def dense_checkpoint_options(config, hidden_states, attention_mask, layer_index=0,
                             *, mode4_policy='recipe', mode4_gate_layers=None):
    """Mode 3/4 SAC, or opt-in mode 2 V2 Flex result retention.

    Replacing ``torch.utils.checkpoint.checkpoint`` with a Python wrapper hides
    the checkpoint higher-order op from Dynamo. Pass context_fn at the actual
    call site so the FP8 trunk stays compiled.
    """
    if getattr(config, 'use_grad_checkpoint', 0) == 3:
        validate_mfu_checkpoint_config(config)
        # Skipping a stochastic fused attention op on replay would shift the
        # RNG stream used by later dropout. Keep its replay when dropout > 0.
        return {'context_fn': partial(dense_mfu_checkpoint_contexts,
                                      save_attention=getattr(config, 'dropout', 0.0) == 0.0)}
    if getattr(config, 'use_grad_checkpoint', 0) == 4:
        validate_mfu_checkpoint_config(config)
        if mode4_policy == 'recipe':
            mode4_policy = ('attention_gate' if hidden_states.shape[1] >= 4096
                            and config.flash_attn and config.dropout == 0. else 'kv_gate')
        if mode4_policy not in ('kv_gate', 'attention_gate', 'projections'):
            raise ValueError('unknown mode 4 cache policy')
        base_layers = config.num_hidden_layers
        if mode4_gate_layers is None:
            dtype = (torch.get_autocast_dtype(hidden_states.device.type)
                     if torch.is_autocast_enabled(hidden_states.device.type) else hidden_states.dtype)
            element_bytes = 4 if dtype == torch.float32 else 2
            tokens = hidden_states.shape[0] * hidden_states.shape[1]
            kv_width = config.num_key_value_heads * config.head_dim
            q_width = config.num_attention_heads * config.head_dim
            base_per_token = (element_bytes * q_width + 4 * config.num_attention_heads
                              if mode4_policy == 'attention_gate' else element_bytes * 2 * kv_width)
            base_layers = min(config.num_hidden_layers, DENSE_RECIPE_CACHE_BYTES // max(1, tokens * base_per_token))
            base_bytes = tokens * base_layers * base_per_token
            gate_bytes = tokens * 2 * config.intermediate_size * element_bytes
            mode4_gate_layers = max(0, min(config.num_hidden_layers,
                (DENSE_RECIPE_CACHE_BYTES - base_bytes) // max(1, gate_bytes)))
        keep_gate = layer_index >= max(0, config.num_hidden_layers - mode4_gate_layers)
        keep_base = layer_index >= config.num_hidden_layers - base_layers
        return {'context_fn': partial(dense_recipe_checkpoint_contexts,
            hidden_size=config.hidden_size,
            query_dim=config.num_attention_heads * config.head_dim,
            kv_dim=config.num_key_value_heads * config.head_dim,
            ffn_dim=config.intermediate_size,
            save_gate_up=keep_gate and mode4_policy != 'projections',
            projections=('all' if layer_index >= config.num_hidden_layers // 2 else 'none')
                        if mode4_policy == 'projections' else ('kv' if mode4_policy == 'kv_gate' and keep_base else 'none'),
            save_attention=mode4_policy == 'attention_gate' and keep_base and getattr(config, 'dropout', 0.) == 0.)}
    setting = os.environ.get('INSTINCT_DENSE_SELECTIVE_CHECKPOINT', '0')
    if setting not in ('0', 'attention'):
        raise ValueError('INSTINCT_DENSE_SELECTIVE_CHECKPOINT must be 0 or attention')
    if setting == '0':
        return {}
    from model.packed_attention import PackedFlexMask
    if (getattr(config, 'use_moe', False)
            or getattr(config, 'residual_type', 'standard') != 'standard'
            or config.hidden_size != 1536 or config.head_dim != 96
            or config.num_attention_heads != 16 or config.num_key_value_heads != 4
            or not hidden_states.is_cuda or not isinstance(attention_mask, PackedFlexMask)):
        return {}
    return {'context_fn': dense_attention_checkpoint_contexts}


def dense_attention_checkpoint_contexts():
    """Retain Flex output/LSE, recompute the remaining full-layer region."""
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
    from torch._higher_order_ops.flex_attention import flex_attention

    def policy(ctx, op, *args, **kwargs):
        return (CheckpointPolicy.MUST_SAVE if op == flex_attention
                else CheckpointPolicy.PREFER_RECOMPUTE)

    return create_selective_checkpoint_contexts(policy)


def validate_mfu_checkpoint_config(config, *, architecture=None):
    """Fail explicitly where Dense SAC has no validated checkpoint boundary."""
    mode = getattr(config, 'use_grad_checkpoint', 0)
    if mode not in (3, 4):
        return
    architecture = architecture or getattr(config, 'model_architecture', 'standard')
    if (architecture != 'standard' or getattr(config, 'use_moe', False)
            or getattr(config, 'residual_type', 'standard') != 'standard'):
        raise ValueError(f'use_grad_checkpoint={mode} requires standard Dense with '
                         'residual_type=standard; use mode 2 for MoE, linear, '
                         'RecurrsiveInstinct or other residual topologies')


def dense_mfu_checkpoint_contexts(*, save_attention=True):
    """Mode 3: keep dense GEMMs and fused attention, replay cheap operations.

    MUST_SAVE survives AOTAutograd partitioning. Do not save arbitrary bmm:
    eager/math attention's score matrix is quadratic in sequence length and
    is mutated by the causal mask. Its core is intentionally recomputed.
    Contexts are fresh per layer/forward; never share a SAC cache across layers.
    No parameter is detached and no custom parameter backward is needed.
    """
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
    from torch._higher_order_ops.flex_attention import flex_attention

    gemms = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten._scaled_mm.default,
    }
    fused_attention = {
        flex_attention,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    }

    def policy(ctx, op, *args, **kwargs):
        if op in gemms or (save_attention and op in fused_attention):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy)


def dense_recipe_checkpoint_contexts(*, hidden_size, query_dim, kv_dim, ffn_dim,
                                    projections='kv', save_gate_up=False, save_attention=False):
    """Mode 4: bound large FFN caches to a small number of upper layers.

    Match logical Linear dimensions, not call order or batch shape. This also
    handles TorchAO _scaled_mm while selecting gate/up only in planned layers.
    Nonselected GEMMs and attention kernels MUST_RECOMPUTE so AOT partitioning
    cannot silently restore mode 3's expensive activation cache.
    """
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
    from torch._higher_order_ops.flex_attention import flex_attention

    projection_shapes = ({(hidden_size, query_dim), (hidden_size, kv_dim), (query_dim, hidden_size)}
                         if projections == 'all' else
                         {(hidden_size, kv_dim)} if projections == 'kv' and kv_dim != query_dim else set())
    ffn_shapes = {(hidden_size, ffn_dim), (ffn_dim, hidden_size)}
    gemms = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default,
             torch.ops.aten._scaled_mm.default}
    attention = {flex_attention, torch.ops.aten.bmm.default,
                 torch.ops.aten._scaled_dot_product_flash_attention.default,
                 torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
                 torch.ops.aten._scaled_dot_product_efficient_attention.default,
                 torch.ops.aten._scaled_dot_product_cudnn_attention.default}

    def policy(ctx, op, *args, **kwargs):
        if op in gemms:
            lhs, rhs = args[1:3] if op == torch.ops.aten.addmm.default else args[:2]
            shape = (lhs.shape[-1], rhs.shape[-1])
            # Ambiguous equal-width FFNs favor recompute, never an unbounded
            # FFN cache disguised as a small attention projection.
            if (save_gate_up and ffn_dim not in (hidden_size, query_dim, kv_dim)
                    and shape == (hidden_size, ffn_dim)):
                return CheckpointPolicy.MUST_SAVE
            if shape in projection_shapes and shape not in ffn_shapes:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_RECOMPUTE
        if op in attention:
            if save_attention and op != torch.ops.aten.bmm.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_RECOMPUTE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy)
