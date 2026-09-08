"""TDD tests for ``checkpoint_ffn`` (FFN/MoE recompute + aux_loss return, plan T3).

RED first: this file is written before ``checkpoint_ffn`` exists in
``model/checkpointing.py``, so the import below fails to collect → the suite is
red. After implementing ``checkpoint_ffn`` the suite must go fully green.

Two things are verified:

1. **Gradient correctness of the FFN recompute.** ``checkpoint_ffn`` wraps the
   real ``FeedForward`` (SwiGLU) / ``MOEFeedForward`` (top-1 routing) modules
   with ``torch.utils.checkpoint`` (``use_reentrant=False``). Parameter
   gradients (``gate_proj``/``up_proj``/``down_proj``, the MoE router ``gate``,
   and expert weights) must be nonzero and bitwise identical
   (``torch.equal``) to a hand-written forward+backward reference on an
   identical-weight module. A custom ``autograd.Function`` would drop parameter
   gradients (its forward runs under ``no_grad``), which is exactly the bug
   this design avoids by delegating to ``torch.utils.checkpoint``.

2. **The MoE aux loss flows through the return value.** The router auxiliary
   loss is *not* read from the module side-channel attribute ``moe.aux_loss``
   (set during the no_grad forward, its gradient would be silently lost); it is
   returned as the second element so the ``use_reentrant=False`` backward, which
   runs ``autograd.backward`` on *all* returned outputs, carries the gradient
   back to ``gate.weight``.

All tests run on CPU fp32 with the tiny config (hidden=64, intermediate=256,
4 experts for MoE). Modules are put in ``.train()`` so the MoE aux loss is
active (``router_aux_loss_coef=5e-4 > 0``).
"""

import torch
import torch.nn as nn

from model.checkpointing import checkpoint_ffn
from model.model_instinct import FeedForward, MOEFeedForward
from tests.helpers import make_tiny_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_h(bs=2, seq=8, hidden=64, seed=0):
    """Deterministic random hidden states."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(bs, seq, hidden, generator=g)


def _assert_nonzero_grad(param, name):
    """``param.grad`` exists and is not all-zero."""
    assert param.grad is not None, f"{name}.grad is None"
    assert param.grad.abs().sum().item() > 0, f"{name}.grad is all-zero"


def _assert_all_grads_equal(m0, m1, names):
    """Each named parameter gradient is non-zero and bitwise equal across modules."""
    for name in names:
        p0 = m0.get_parameter(name)
        p1 = m1.get_parameter(name)
        _assert_nonzero_grad(p0, name)
        _assert_nonzero_grad(p1, name)
        assert torch.equal(p0.grad, p1.grad), f"{name} grad not bitwise equal"


def _fresh_dense(seed=0):
    torch.manual_seed(seed)
    return FeedForward(make_tiny_config(use_moe=False))


def _fresh_moe(seed=0):
    torch.manual_seed(seed)
    return MOEFeedForward(make_tiny_config(use_moe=True))


class DropoutFFN(nn.Module):
    """Synthetic FFN with an explicit dropout — tests RNG preservation."""

    def __init__(self, config):
        super().__init__()
        self.ffn = FeedForward(config)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.drop(self.ffn(x))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ffn_param_grads():
    """Dense SwiGLU: checkpoint_ffn grads non-zero and torch.equal to a direct
    forward+backward reference; aux must be None (FeedForward has no aux_loss)."""
    ffn = _fresh_dense(seed=0)
    ref = _fresh_dense(seed=0)  # identical weights via same init seed
    h = _make_h()

    out, aux = checkpoint_ffn(ffn, h)
    assert aux is None  # dense path: no aux_loss attribute -> getattr default
    out.sum().backward()

    ref(h).sum().backward()

    _assert_all_grads_equal(
        ffn, ref, ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
    )


def test_moe_aux_loss_grad():
    """MoE top-1: the returned aux_loss has a grad_fn and its gradient flows
    back into gate.weight; experts that routed tokens carry nonzero grads."""
    moe = _fresh_moe(seed=0)
    h = _make_h()

    out, aux = checkpoint_ffn(moe, h)
    assert aux is not None
    assert aux.grad_fn is not None  # aux is a real graph node, not a constant
    loss = out.sum() + aux
    loss.backward()

    _assert_nonzero_grad(moe.gate.weight, "moe.gate.weight")
    # top-1 routing over 16 tokens / 4 experts: at least one expert got tokens.
    expert_grads = [e.down_proj.weight.grad for e in moe.experts]
    assert any(g is not None and g.abs().sum().item() > 0 for g in expert_grads), (
        "no expert received routed tokens / nonzero down_proj gradient"
    )

    # Prove the aux term genuinely contributes: dropping it from the loss must
    # change the router gradient (identical weights + identical input).
    moe2 = _fresh_moe(seed=0)
    out2, _aux2 = checkpoint_ffn(moe2, h)
    out2.sum().backward()
    assert not torch.equal(moe.gate.weight.grad, moe2.gate.weight.grad), (
        "router gradient unchanged when aux_loss is removed — aux does not "
        "flow back to gate.weight"
    )


def test_moe_param_grads_equal():
    """MoE: gate + expert grads via checkpoint_ffn torch.equal a hand-written
    forward+backward reference (loss = out.sum() + aux on both sides)."""
    moe = _fresh_moe(seed=0)
    ref = _fresh_moe(seed=0)
    h = _make_h()

    out, aux = checkpoint_ffn(moe, h)
    (out.sum() + aux).backward()

    ref_out = ref(h)
    (ref_out.sum() + ref.aux_loss).backward()

    names = ["gate.weight"] + [
        f"experts.{i}.gate_proj.weight"
        for i in range(len(moe.experts))
    ] + [
        f"experts.{i}.up_proj.weight"
        for i in range(len(moe.experts))
    ] + [
        f"experts.{i}.down_proj.weight"
        for i in range(len(moe.experts))
    ]
    _assert_all_grads_equal(moe, ref, names)


def test_dropout_rng():
    """Dropout (p=0.1) RNG consistency:
    (a) two checkpoint_ffn forwards with the same seed are bitwise equal;
    (b) backward grads equal a same-seed direct reference — proves the
        backward recompute replayed the *same* dropout mask as forward
        (preserve_rng_state=True)."""
    config = make_tiny_config(use_moe=False)
    h = _make_h(seed=123)

    torch.manual_seed(0)
    m = DropoutFFN(config)
    torch.manual_seed(0)
    ref_m = DropoutFFN(config)
    # m/ref_m have identical weights (same init seed)…
    assert torch.equal(m.ffn.gate_proj.weight, ref_m.ffn.gate_proj.weight)

    torch.manual_seed(7)
    out1, aux1 = checkpoint_ffn(m, h)
    torch.manual_seed(7)
    out2, aux2 = checkpoint_ffn(m, h)
    assert torch.equal(out1, out2)
    assert aux1 is None and aux2 is None

    # backward grads equal same-seed hand-written reference
    torch.manual_seed(7)
    out_a, _ = checkpoint_ffn(m, h)
    out_a.sum().backward()

    torch.manual_seed(7)
    ref_m(h).sum().backward()

    for (na, pa), (nb, pb) in zip(m.named_parameters(), ref_m.named_parameters()):
        assert na == nb
        _assert_nonzero_grad(pa, na)
        assert torch.equal(pa.grad, pb.grad), f"{na} grad not bitwise equal"
