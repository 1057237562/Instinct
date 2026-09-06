"""Cross-model gradient / step consistency suite (plan task T10).

This is NOT a RED-first suite: T7/T8/T9 have already wired
``config.use_grad_checkpoint`` (0/1/2) into the dense / loop / linear model
files and their per-model suites are green. This suite is the *cross-model*
regression net that pins the wiring down across all four variants
{dense, moe, loop, linear} x {Mode 0 vs 1, Mode 0 vs 2}:

1. **Gradient matrix** (``test_grad_matrix``): for every (variant, mode) pair,
   two same-seed models (mode 0 and mode m) forward with labels and backward
   through ``loss + aux_loss`` (trainer semantics — ``train_pretrain.py:37``
   trains on ``res.loss + res.aux_loss``). Loss, aux_loss and every parameter
   gradient must be bitwise equal (CPU fp32 ``torch.equal`` via
   ``assert_grads_equal``).

2. **Step equality** (``test_step_equality``): seed 42, two dense models
   (Mode 0 / Mode 1), 3 full AdamW training steps following the
   ``trainer/train_pretrain.py:35-49`` pattern (forward -> backward -> step ->
   zero_grad). The 3-step loss sequence must stay bitwise locked — this catches
   *silent drift*: any sub-bitwise divergence in the Mode 1 recomputed grads
   would skew AdamW moments and blow up the loss sequence.

3. **Path coverage** (``test_path_*``): the four attention execution paths —
   flash (SDPA), eager + padding mask, KV cache (``past_key_value``), and
   ``seq_len == 1`` — must keep Mode 1/2 forward losses and gradients bitwise
   equal to Mode 0. Paths that cannot run on this host are skipped explicitly
   with a documented reason (only the flash path is host-conditional: it needs
   ``F.scaled_dot_product_attention``).

All tests are CPU fp32 (no GPU dependency); fp16/bf16 dtype coverage belongs to
``@pytest.mark.gpu`` suites and is out of scope here.
"""

import torch
import torch.nn.functional as F
import pytest

from model.model_instinct import InstinctForCausalLM as DenseForCausalLM
from model.model_instinct_loop import InstinctForCausalLM as LoopForCausalLM
from model.model_instinct_linear import InstinctForCausalLM as LinearForCausalLM
from tests.helpers import assert_grads_equal, make_tiny_config

_CAUSAL_LM_CLASSES = {
    "dense": DenseForCausalLM,
    "loop": LoopForCausalLM,
    "linear": LinearForCausalLM,
}

# Fixed deterministic inputs (token ids well below vocab_size=256).
INPUT_IDS = torch.tensor(
    [[3, 11, 17, 5, 23, 7, 29, 13], [4, 12, 18, 6, 24, 8, 30, 14]], dtype=torch.long
)
LABELS = INPUT_IDS.clone()
# Right-padded attention mask: row 0 keeps 6 tokens, row 1 keeps 7.
PADDED_MASK = torch.tensor(
    [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long
)
# Single-token inputs for the seq_len == 1 path.
SEQ1_INPUT = torch.tensor([[3], [11]], dtype=torch.long)
SEQ1_LABELS = SEQ1_INPUT.clone()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recompute_layer_types(config):
    """Rebuild ``layer_types`` after overriding ``full_attention_interval``.

    ``InstinctConfig.__init__`` computes ``layer_types`` once from the
    interval; mutating the attribute afterwards does not recompute it (T9
    pattern, mirrors ``test_linear_model.py``).
    """
    config.layer_types = []
    for i in range(config.num_hidden_layers):
        if (i + 1) % config.full_attention_interval == 0:
            config.layer_types.append("full_attention")
        else:
            config.layer_types.append("linear_attention")
    return config


def _make_causal_lm(variant, use_moe, mode, seed=0, flash_attn=False):
    """Deterministic tiny CausalLM with ``use_grad_checkpoint=mode``.

    Same seed before construction => bitwise-identical init across modes.
    """
    torch.manual_seed(seed)
    config = make_tiny_config(use_moe=use_moe, variant=variant)
    config.flash_attn = flash_attn
    config.use_grad_checkpoint = mode
    if variant == "loop":
        config.loop_iters = 2  # total effective depth = prelude 1 + 2 + coda 1 = 4
    elif variant == "linear":
        # full_attention_interval=1 => every layer is a standard Attention block
        # (no GatedDeltaNet), so the same (k, v) cache format works for all layers.
        config.full_attention_interval = 1
        _recompute_layer_types(config)
    model = _CAUSAL_LM_CLASSES[variant](config)
    model.train()
    return model


def _forward_backward(model, input_ids=INPUT_IDS, labels=LABELS,
                      attention_mask=None, past_key_values=None, use_cache=False):
    """Forward with labels -> MoeCausalLMOutputWithPast; backward loss + aux.

    ``loss + aux_loss`` mirrors the trainer objective
    (``trainer/train_pretrain.py:37``), so the MoE router gradient flows
    through the aux term exactly as in training.
    """
    out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=use_cache)
    (out.loss + out.aux_loss).backward()
    return out


def _make_kv_cache(past_len=4, bs=2, kv_heads=2, head_dim=16, seed=0):
    """Random (k, v) cache tuple: [bs, past_len, kv_heads, head_dim].

    Layout matches what ``Attention.forward`` concatenates with the fresh
    ``xk``/``xv`` (``[bs, seq, kv_heads, head_dim]``) along ``dim=1``
    (``model/model_instinct.py:132-137``).
    """
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(bs, past_len, kv_heads, head_dim, generator=g)
    v = torch.randn(bs, past_len, kv_heads, head_dim, generator=g)
    return (k, v)


def _assert_modes_match(models, outputs):
    """Mode 1 and Mode 2 must match Mode 0 bitwise: loss, aux_loss, grads.

    ``models`` is [mode0, mode1, mode2]; each element of ``outputs`` is the
    corresponding forward output (already backpropagated).
    """
    for i in (1, 2):
        assert torch.equal(outputs[i].loss, outputs[0].loss), (
            f"Mode {i}: forward loss != Mode 0 (bitwise)"
        )
        assert torch.equal(outputs[i].aux_loss, outputs[0].aux_loss), (
            f"Mode {i}: aux_loss != Mode 0 (bitwise)"
        )
        assert_grads_equal(models[i], models[0])


# ---------------------------------------------------------------------------
# 4.1 Gradient matrix: {dense, moe, loop, linear} x {Mode 1, Mode 2}
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("variant,use_moe", [
    pytest.param("dense", False, id="dense"),
    pytest.param("dense", True, id="moe"),
    pytest.param("loop", False, id="loop"),
    pytest.param("linear", False, id="linear"),
])
def test_grad_matrix(variant, use_moe, mode):
    """Mode 0 vs {1, 2} on CPU fp32: loss / aux_loss / every param grad bitwise equal.

    ``use_grad_checkpoint`` in {1, 2} must not perturb the training objective:
    forward loss, aux_loss and every parameter gradient are ``torch.equal``
    to Mode 0. loop runs with ``loop_iters=2``; linear with
    ``full_attention_interval=1`` (all standard Attention layers).
    """
    m0 = _make_causal_lm(variant, use_moe, 0)
    m1 = _make_causal_lm(variant, use_moe, mode)
    o0 = _forward_backward(m0)
    o1 = _forward_backward(m1)

    assert torch.equal(o1.loss, o0.loss), (
        f"[{variant}, moe={use_moe}, mode={mode}] forward loss != Mode 0 (bitwise)"
    )
    assert torch.equal(o1.aux_loss, o0.aux_loss), (
        f"[{variant}, moe={use_moe}, mode={mode}] aux_loss != Mode 0 (bitwise)"
    )
    assert_grads_equal(m1, m0)


# ---------------------------------------------------------------------------
# 4.2 Step equality: 3 AdamW steps, seed 42, Mode 0 vs 1 (anti-silent-drift)
# ---------------------------------------------------------------------------

def test_step_equality():
    """3 training steps (AdamW) with identical seed: Mode 0/1 loss sequence,
    weights and final grads stay bitwise identical (``torch.equal``).

    Mirrors ``trainer/train_pretrain.py:35-49``: forward ->
    ``(loss + aux_loss).backward()`` -> ``optimizer.step()`` ->
    ``optimizer.zero_grad(set_to_none=True)``. AdamW momentum/variance amplify
    any sub-bitwise gradient divergence, so a locked 3-step loss sequence is a
    strong no-silent-drift guarantee for the Mode 1 recompute path.
    """
    torch.manual_seed(42)
    m0 = _make_causal_lm("dense", False, 0, seed=42)
    m1 = _make_causal_lm("dense", False, 1, seed=42)
    opt0 = torch.optim.AdamW(m0.parameters(), lr=1e-3)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)

    losses0, losses1 = [], []
    for _ in range(3):
        for m, opt, losses in ((m0, opt0, losses0), (m1, opt1, losses1)):
            out = m(input_ids=INPUT_IDS, labels=LABELS)
            loss = out.loss + out.aux_loss
            losses.append(loss.detach().clone())
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    for step in range(3):
        assert torch.equal(losses0[step], losses1[step]), (
            f"step {step + 1}: Mode 0/1 loss diverged (silent drift detected)"
        )

    # Weights after 3 optimizer steps: bitwise identical.
    p0 = dict(m0.named_parameters())
    p1 = dict(m1.named_parameters())
    for name in p0:
        assert torch.equal(p0[name], p1[name]), (
            f"param '{name}' diverged after 3 AdamW steps"
        )

    # One final forward+backward to compare gradients (zero_grad cleared them).
    o0 = _forward_backward(m0)
    o1 = _forward_backward(m1)
    assert torch.equal(o1.loss, o0.loss), "post-step forward loss != Mode 0"
    assert_grads_equal(m1, m0)


# ---------------------------------------------------------------------------
# 4.3 Path coverage: flash / eager+padding mask / KV cache / seq_len==1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_moe", [False, True])
def test_path_flash(use_moe):
    """Path 1 — flash: ``flash_attn=True``, seq>1, no padding mask, no KV cache.

    The flash branch (``model/model_instinct.py:139-142``) runs *before* the
    mode check, so Mode 0/1/2 all execute the identical
    ``flash_attention()`` call; Mode 2 additionally checkpoints the whole
    block and Mode 1 checkpoints the FFN. Asserting bitwise equality here
    verifies the flash path is mode-independent and the checkpointed backward
    reproduces SDPA grads exactly.

    Skipped (with reason) only if SDPA is unavailable: the flash condition
    ``hasattr(F, 'scaled_dot_product_attention') and config.flash_attn``
    cannot be satisfied on such a torch build.
    """
    if not hasattr(F, "scaled_dot_product_attention"):
        pytest.skip("F.scaled_dot_product_attention unavailable: flash branch cannot run on this torch build")
    models = [_make_causal_lm("dense", use_moe, mode, flash_attn=True) for mode in (0, 1, 2)]
    # Precondition: the flash branch is actually armed (config + SDPA present).
    assert all(m.model.layers[0].self_attn.flash for m in models), (
        "flash branch not armed — test would silently cover the eager path"
    )
    outputs = [_forward_backward(m) for m in models]
    _assert_modes_match(models, outputs)


@pytest.mark.parametrize("use_moe", [False, True])
def test_path_eager_padding_mask(use_moe):
    """Path 2 — eager + padding mask: ``flash_attn=False``, mask with 0s.

    Mode 1 recompute must apply the same ``(1.0 - mask) * -1e9`` term with the
    identical operation order (causal triu first, mask second) so padded rows
    stay bitwise equal to Mode 0's eager math attention.
    """
    models = [_make_causal_lm("dense", use_moe, mode, flash_attn=False) for mode in (0, 1, 2)]
    outputs = [_forward_backward(m, attention_mask=PADDED_MASK) for m in models]
    _assert_modes_match(models, outputs)


@pytest.mark.parametrize("use_moe", [False, True])
def test_path_kv_cache(use_moe):
    """Path 3 — KV cache: ``past_key_values`` is not None -> Mode 1 falls back
    to eager (guard ``use_grad_checkpoint == 1 and ... past_key_value is None``
    is False), which is exactly the behaviour under test: cached forward must
    stay bitwise equal to Mode 0. Mode 2 checkpoints the whole block with the
    cache attached. One (k, v) cache per layer (2 tiny layers).
    """
    pkv = _make_kv_cache()
    models = [_make_causal_lm("dense", use_moe, mode, flash_attn=False) for mode in (0, 1, 2)]
    outputs = [_forward_backward(m, past_key_values=[pkv, pkv]) for m in models]
    _assert_modes_match(models, outputs)


def test_path_kv_cache_loop():
    """Path 3 (loop variant) — KV cache into every prelude / loop / coda slot.

    ``total_effective_layers = prelude(1) + loop_iters(2) + coda(1) = 4``; the
    loop body applies the shared block ``loop_iters`` times, each with the
    cache attached. Mode 1's ``past_key_value is None`` guard is exercised per
    slot; all blocks must stay bitwise equal to Mode 0.
    """
    pkv = _make_kv_cache()
    models = [_make_causal_lm("loop", False, mode, flash_attn=False) for mode in (0, 1, 2)]
    n_slots = models[0].model.total_effective_layers
    outputs = [_forward_backward(m, past_key_values=[pkv] * n_slots) for m in models]
    _assert_modes_match(models, outputs)


def test_path_kv_cache_linear():
    """Path 3 (linear variant) — KV cache with ``full_attention_interval=1``.

    Every layer is a standard full-attention block, so all layers accept the
    (k, v) cache and the model-side ``start_pos`` scan finds it on the first
    layer. (Mixed layer_types would need GatedDeltaNet (conv, recurrent) state
    tuples per linear-attention slot — out of scope for the attention-path
    guard coverage; the pure-full-attention config exercises the identical
    ``Attention.forward`` cache branch.)
    """
    pkv = _make_kv_cache()
    models = [_make_causal_lm("linear", False, mode, flash_attn=False) for mode in (0, 1, 2)]
    n_layers = len(models[0].model.layers)
    assert all(l.layer_type == "full_attention" for l in models[0].model.layers)
    outputs = [_forward_backward(m, past_key_values=[pkv] * n_layers) for m in models]
    _assert_modes_match(models, outputs)


@pytest.mark.parametrize("use_moe", [False, True])
def test_path_seq_len_1(use_moe):
    """Path 4 — ``seq_len == 1`` with ``flash_attn=True``: the flash condition
    requires ``seq_len > 1``, so even with flash armed the forward falls to
    eager — and Mode 1's recompute runs with 1x1 score matrices.

    The model's own label loss is empty at seq_len==1 (``logits[..., :-1, :]``
    vs ``labels[..., 1:]`` span zero tokens -> NaN), which ``torch.equal``
    cannot compare, so a per-token CE surrogate on the last logits
    (``F.cross_entropy(logits[:, -1, :], labels[:, -1])``) is used as the
    backprop target.
    """
    models = [_make_causal_lm("dense", use_moe, mode, flash_attn=True) for mode in (0, 1, 2)]
    losses = []
    for m in models:
        out = m(input_ids=SEQ1_INPUT, labels=SEQ1_LABELS)
        loss = F.cross_entropy(out.logits[:, -1, :], SEQ1_LABELS[:, -1])
        losses.append(loss)
        loss.backward()
    assert torch.equal(losses[1], losses[0]), "seq_len==1: Mode 1 loss != Mode 0"
    assert torch.equal(losses[2], losses[0]), "seq_len==1: Mode 2 loss != Mode 0"
    assert_grads_equal(models[1], models[0])
    assert_grads_equal(models[2], models[0])
