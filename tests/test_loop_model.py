"""TDD tests for Loop-variant gradient checkpointing (plan T8).

RED first: this file is written before ``model/model_instinct_loop.py`` has the
Mode 1 (selective attention/FFN recompute) and Mode 2 (loop-body whole-block
checkpoint) wiring, so ``use_grad_checkpoint=1/2`` are silently ignored and the
mode-equality assertions fail. After wiring, the suite must go fully green with
every equality asserted via ``torch.equal`` (bitwise, CPU fp32).

Coverage:

1. **test_mode012_loss_equal** — three independent models with identical init
   seed and ``use_grad_checkpoint`` in {0, 1, 2}: forward ``loss`` (with labels)
   and ``aux_loss`` (MoE) are bitwise equal, and parameter gradients are bitwise
   equal after ``(loss + aux_loss).backward()`` (``assert_grads_equal``). The
   ``aux_loss`` term is added to the backward loss because the trainers train
   with ``loss = outputs.loss + outputs.aux_loss``; Mode 1/2 surface the
   loop-block aux *through the checkpoint return value* (grad-tracked), Mode 0
   through the module side-channel — all three must produce identical grads.

2. **test_moe_aux_loss_grad** — Mode 2 with MoE: the shared ``loop_block`` is
   checkpointed once per loop iteration; the router ``gate.weight.grad`` must be
   non-zero, proving the aux gradient survives the checkpointed recompute and
   flows back into the router.

3. **test_eval_no_checkpoint** — ``model.eval()`` disables all checkpoint paths
   (pure overhead on inference); Mode 1/2 forward must not crash and the loss
   must equal Mode 0.
"""

import torch

from model import model_instinct_loop
from tests.helpers import make_tiny_config, assert_grads_equal

# Fixed deterministic inputs: token ids well below vocab_size=256, seq=8.
INPUT_IDS = torch.tensor([[3, 11, 17, 5, 23, 7, 29, 13]], dtype=torch.long)
LABELS = INPUT_IDS.clone()


def _make_loop_causal_lm(use_moe, mode, seed=0, loop_iters=2):
    """Fresh loop InstinctForCausalLM with a tiny config, fixed seed, small loop."""
    torch.manual_seed(seed)
    config = make_tiny_config(use_moe=use_moe, variant="loop")
    config.loop_iters = loop_iters
    config.use_grad_checkpoint = mode
    model = model_instinct_loop.InstinctForCausalLM(config)
    model.train()
    return model


def _forward_backward(model, input_ids=INPUT_IDS, labels=LABELS):
    """Forward with labels, return (loss, aux_loss) and backprop loss + aux."""
    model.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    loss, aux = out.loss, out.aux_loss
    (loss + aux).backward()
    return loss, aux


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mode012_loss_equal():
    """Mode 0/1/2: forward loss (+ aux_loss) and parameter grads bitwise equal.

    Runs for both dense and MoE. Grays are compared with ``assert_grads_equal``
    (``torch.equal`` for fp32).
    """
    for use_moe in (False, True):
        models = [_make_loop_causal_lm(use_moe, mode) for mode in (0, 1, 2)]
        losses, auxes = [], []
        for m in models:
            loss, aux = _forward_backward(m)
            losses.append(loss)
            auxes.append(aux)

        # forward loss identical across modes
        assert torch.equal(losses[0], losses[1]), "Mode 0 vs 1: loss not bitwise equal"
        assert torch.equal(losses[0], losses[2]), "Mode 0 vs 2: loss not bitwise equal"
        # aux_loss identical across modes (MoE; dense aux is always the 0 scalar)
        assert torch.equal(auxes[0], auxes[1]), "Mode 0 vs 1: aux_loss not bitwise equal"
        assert torch.equal(auxes[0], auxes[2]), "Mode 0 vs 2: aux_loss not bitwise equal"

        # parameter gradients bitwise equal across modes
        assert_grads_equal(models[0], models[1])
        assert_grads_equal(models[0], models[2])


def test_mode1_engages_recompute(monkeypatch):
    """Mode 1 wiring is real: recompute_attention is invoked in the eager
    attention path during training (prelude + coda + loop_iters loop blocks)."""
    m = _make_loop_causal_lm(use_moe=False, mode=1, seed=0)
    calls = {"n": 0}
    orig = model_instinct_loop.recompute_attention

    def spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(model_instinct_loop, "recompute_attention", spy)
    out = m(input_ids=INPUT_IDS, labels=LABELS)
    assert out.loss is not None
    assert calls["n"] > 0, "Mode 1: recompute_attention never called in training"


def test_mode2_engages_checkpoint(monkeypatch):
    """Mode 2 wiring is real: torch.utils.checkpoint wraps every loop_block
    application (once per loop iteration)."""
    m = _make_loop_causal_lm(use_moe=False, mode=2, seed=0)
    calls = {"n": 0}
    orig = torch.utils.checkpoint.checkpoint

    def spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", spy)
    out = m(input_ids=INPUT_IDS, labels=LABELS)
    assert out.loss is not None
    assert calls["n"] == m.model.loop_iters, (
        f"Mode 2: checkpoint called {calls['n']}x, expected {m.model.loop_iters}x"
    )


def test_moe_aux_loss_grad():
    """Mode 2 + MoE: aux gradient flows back to the shared loop_block router."""
    for mode in (1, 2):
        m = _make_loop_causal_lm(use_moe=True, mode=mode, seed=0)
        loss, aux = _forward_backward(m)
        assert aux is not None
        gate = m.model.loop_block.mlp.gate
        assert gate.weight.grad is not None, f"Mode {mode}: loop_block gate.weight.grad is None"
        assert gate.weight.grad.abs().sum().item() > 0, (
            f"Mode {mode}: loop_block gate.weight.grad is all-zero"
        )
        # the returned aux genuinely contributes to the router gradient
        assert aux.grad_fn is not None, f"Mode {mode}: aux_loss is detached (no grad_fn)"


def test_eval_no_checkpoint():
    """eval(): Mode 1/2 forward uses the eager path (no checkpoint) and loss
    equals Mode 0."""
    models = [_make_loop_causal_lm(use_moe=False, mode=mode) for mode in (0, 1, 2)]
    losses = []
    for m in models:
        m.eval()
        out = m(input_ids=INPUT_IDS, labels=LABELS)
        losses.append(out.loss)
    assert torch.equal(losses[0], losses[1]), "eval: Mode 0 vs 1 loss differ"
    assert torch.equal(losses[0], losses[2]), "eval: Mode 0 vs 2 loss differ"
