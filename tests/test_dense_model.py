"""TDD tests for wiring gradient checkpointing into the Dense model (plan T7).

Written before ``model/model_instinct.py`` wires ``config.use_grad_checkpoint``
(0/1/2) into ``Attention`` / ``InstinctBlock`` / ``InstinctModel``. After the
wiring, Mode 1 (selective attention + FFN recompute) and Mode 2 (full-block
``torch.utils.checkpoint``) must be bitwise identical (``torch.equal``) to
Mode 0 in loss, aux_loss and every parameter gradient on CPU fp32.

Verified here:

1. **Mode 0 baseline.** A plain ``use_grad_checkpoint=0`` forward produces a
   finite positive loss and is deterministic under a fixed seed (the reference
   behaviour the checkpointed modes must reproduce).
2. **Mode 0/1/2 equivalence.** Three independently-initialised (same seed)
   models, forward with labels -> losses and aux losses are ``torch.equal``;
   backward -> every parameter gradient ``torch.equal``
   (``tests.helpers.assert_grads_equal``). head_dim=16 makes ``* (1/sqrt(hd))``
   bitwise equal to the eager ``/ sqrt(hd)``.
3. **MoE router gradient.** With ``use_moe=True``, Mode 1 and Mode 2 both give a
   non-zero ``gate.weight.grad`` after backward, and the returned ``aux_loss``
   is a real graph node (grad_fn attached) — the aux-loss-return fix keeps the
   router gradient alive through the checkpointed paths.
4. **Eval-mode bypass.** ``model.eval()`` skips all checkpointing
   (``self.training`` guard): Mode 1/2 forward matches Mode 0 bitwise and
   doesn't crash.
"""

import torch

from model.model_instinct import InstinctForCausalLM
from tests.helpers import assert_grads_equal, make_tiny_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(use_moe, mode, seed=0):
    """Deterministic tiny CausalLM with ``use_grad_checkpoint=mode``."""
    config = make_tiny_config(use_moe=use_moe)
    config.use_grad_checkpoint = mode
    torch.manual_seed(seed)
    return InstinctForCausalLM(config)


def _make_inputs(bs=2, seq=8, vocab=256, seed=0):
    """Random token ids + matching labels (all positions supervised)."""
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab - 1, (bs, seq), generator=g)
    return input_ids, input_ids.clone()


def _assert_nonzero_grad(param, name):
    assert param.grad is not None, f"{name}.grad is None"
    assert param.grad.abs().sum().item() > 0, f"{name}.grad is all-zero"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mode0_unchanged():
    """Mode 0 baseline: finite positive loss, deterministic under fixed seed."""
    input_ids, labels = _make_inputs()
    losses = []
    for _ in range(2):
        model = _make_model(use_moe=False, mode=0, seed=0)
        model.train()
        loss = model(input_ids=input_ids, labels=labels).loss
        losses.append(loss)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0
    assert torch.equal(losses[0], losses[1])


def test_mode012_loss_equal():
    """Mode 0/1/2: loss and aux_loss torch.equal; all parameter grads torch.equal."""
    models = [_make_model(use_moe=False, mode=mode, seed=0) for mode in (0, 1, 2)]
    for model in models:
        model.train()
    input_ids, labels = _make_inputs()

    losses, auxes = [], []
    for model in models:
        out = model(input_ids=input_ids, labels=labels)
        losses.append(out.loss)
        auxes.append(out.aux_loss)
        out.loss.backward()

    assert torch.equal(losses[0], losses[1])
    assert torch.equal(losses[0], losses[2])
    assert torch.equal(auxes[0], auxes[1])
    assert torch.equal(auxes[0], auxes[2])
    assert_grads_equal(models[0], models[1])
    assert_grads_equal(models[0], models[2])


def test_moe_router_grad():
    """MoE Mode 1 & 2: aux_loss flows (grad_fn present) and gate.weight.grad is
    non-zero after backward — the aux-loss-return fix (plan T7) works end-to-end."""
    input_ids, labels = _make_inputs()
    for mode in (1, 2):
        model = _make_model(use_moe=True, mode=mode, seed=0)
        model.train()
        out = model(input_ids=input_ids, labels=labels)
        assert out.aux_loss.grad_fn is not None, (
            f"mode {mode}: aux_loss is not a real graph node — router gradient lost"
        )
        (out.loss + out.aux_loss).backward()
        gate = model.model.layers[0].mlp.gate.weight
        _assert_nonzero_grad(gate, f"mode {mode}: model.model.layers[0].mlp.gate.weight")


def test_eval_no_checkpoint():
    """eval() bypasses checkpointing (self.training guard): Mode 1/2 == Mode 0."""
    models = [_make_model(use_moe=False, mode=mode, seed=0) for mode in (0, 1, 2)]
    for model in models:
        model.eval()
    input_ids, labels = _make_inputs()
    losses = [model(input_ids=input_ids, labels=labels).loss for model in models]
    assert torch.equal(losses[0], losses[1])
    assert torch.equal(losses[0], losses[2])
