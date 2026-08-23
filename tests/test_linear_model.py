"""TDD tests for gradient-checkpoint wiring in the linear variant (plan T9).

RED first: before ``model_instinct_linear.py`` is wired, ``test_wiring_attributes``
raises ``AttributeError`` (``use_grad_checkpoint`` does not exist on ``Attention``
/ ``InstinctBlock`` yet) so the suite is genuinely red. After wiring, all
assertions must pass with ``torch.equal`` (bitwise) on CPU fp32.

What is covered:

1. ``test_mode012_loss_equal_full_attn`` — ``full_attention_interval=1`` turns
   every layer into the standard eager ``Attention`` (no GatedDeltaNet). Mode 1
   (selective recompute of the attention core + ``checkpoint_ffn``) and Mode 2
   (whole-block ``torch.utils.checkpoint``) must be bitwise identical to Mode 0
   in forward loss/aux_loss and in every parameter gradient.
2. ``test_mode2_linear_attn_loss_equal`` — ``full_attention_interval=4`` with 2
   layers leaves both layers as GatedDeltaNet. Mode 2 whole-block checkpoint
   must cover them with bitwise-identical loss/grads vs Mode 0 (we never touch
   the FLA/Triton kernel internals — ``torch.utils.checkpoint`` just re-runs
   the whole block forward).
3. ``test_moe_router_grad`` — MoE top-1 normalizes the router weight to 1.0, so
   ``gate.weight`` only receives gradient through the aux loss. Under Mode 1/2
   the checkpointed aux loss must flow back (trainers combine ``loss + aux_loss``).
4. ``test_gated_delta_net_untouched`` — the ``git diff`` vs HEAD must not touch
   the ``GatedDeltaNet`` class body (model lines 155-247): changes are limited
   to the standard ``Attention`` / ``InstinctBlock`` / ``InstinctModel``
   integration points.
"""

import re
import subprocess
from pathlib import Path

import torch

from model.model_instinct_linear import InstinctForCausalLM
from tests.helpers import assert_grads_equal, make_tiny_config

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recompute_layer_types(config):
    """Mirror ``InstinctConfig.__init__`` after overriding full_attention_interval."""
    config.layer_types = []
    for i in range(config.num_hidden_layers):
        if (i + 1) % config.full_attention_interval == 0:
            config.layer_types.append("full_attention")
        else:
            config.layer_types.append("linear_attention")
    return config


def _make_model(use_grad_checkpoint, use_moe=False, full_attention_interval=4, seed=0):
    config = make_tiny_config(use_moe=use_moe, variant="linear")
    config.full_attention_interval = full_attention_interval
    _recompute_layer_types(config)
    config.use_grad_checkpoint = use_grad_checkpoint
    torch.manual_seed(seed)  # identical init across modes (same seed)
    model = InstinctForCausalLM(config)
    model.train()
    return model


def _make_inputs(seed=0, bs=2, seq=8):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 256, (bs, seq), generator=g)
    return input_ids, input_ids.clone()


def _gated_delta_net_span(head_source):
    """Line span of the ``class GatedDeltaNet`` body in the HEAD blob of
    ``model/model_instinct_linear.py`` (1-based, exclusive end)."""
    lines = head_source.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("class GatedDeltaNet"):
            start = i + 1
            break
    assert start is not None, "class GatedDeltaNet not found in HEAD blob"
    end = len(lines) + 1
    for i in range(start, len(lines)):
        if lines[i].startswith(("class ", "def ")):
            end = i + 1
            break
    return start, end


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_wiring_attributes():
    """Mode 1 wires ``use_grad_checkpoint`` into the standard ``Attention`` and
    ``InstinctBlock``; Mode 2 wires the whole-block flag into ``InstinctBlock``."""
    m0 = _make_model(0, full_attention_interval=1)
    m1 = _make_model(1, full_attention_interval=1)
    m2 = _make_model(2, full_attention_interval=1)
    for layer in m0.model.layers:
        assert layer.use_grad_checkpoint == 0
        assert layer.self_attn.use_grad_checkpoint == 0
    for layer in m1.model.layers:
        assert layer.use_grad_checkpoint == 1
        assert layer.self_attn.use_grad_checkpoint == 1
    for layer in m2.model.layers:
        assert layer.use_grad_checkpoint == 2


def test_mode012_loss_equal_full_attn():
    """full_attention_interval=1 → all layers use the standard eager Attention.
    Mode 1/2 forward loss + aux_loss and every parameter gradient are bitwise
    equal to Mode 0 (CPU fp32, ``torch.equal``)."""
    input_ids, labels = _make_inputs()
    m0 = _make_model(0, full_attention_interval=1)
    m1 = _make_model(1, full_attention_interval=1)
    m2 = _make_model(2, full_attention_interval=1)

    o0 = m0(input_ids, labels=labels)
    o1 = m1(input_ids, labels=labels)
    o2 = m2(input_ids, labels=labels)
    assert torch.equal(o1.loss, o0.loss), "Mode 1 forward loss != Mode 0"
    assert torch.equal(o2.loss, o0.loss), "Mode 2 forward loss != Mode 0"
    assert torch.equal(o1.aux_loss, o0.aux_loss), "Mode 1 aux_loss != Mode 0"
    assert torch.equal(o2.aux_loss, o0.aux_loss), "Mode 2 aux_loss != Mode 0"

    o0.loss.backward()
    o1.loss.backward()
    assert_grads_equal(m1, m0)
    o2.loss.backward()
    assert_grads_equal(m2, m0)


def test_mode2_linear_attn_loss_equal():
    """full_attention_interval=4 with 2 layers → both layers are GatedDeltaNet.
    Mode 2 whole-block checkpoint must cover the linear-attention layers and be
    bitwise identical to Mode 0."""
    input_ids, labels = _make_inputs()
    m0 = _make_model(0)  # default interval 4 → both layers linear_attention
    m2 = _make_model(2)
    assert all(layer.layer_type == "linear_attention" for layer in m0.model.layers)

    o0 = m0(input_ids, labels=labels)
    o2 = m2(input_ids, labels=labels)
    assert torch.equal(o2.loss, o0.loss), "Mode 2 (GatedDeltaNet) forward loss != Mode 0"

    o0.loss.backward()
    o2.loss.backward()
    assert_grads_equal(m2, m0)


def test_moe_router_grad():
    """MoE top-1 normalization makes ``gate.weight`` depend only on the aux
    loss. Under Mode 1/2 the checkpointed aux_loss must flow back so
    ``layers[0].mlp.gate.weight.grad`` is nonzero after ``(loss + aux_loss)``."""
    input_ids, labels = _make_inputs()
    for mode in (1, 2):
        model = _make_model(mode, use_moe=True, full_attention_interval=1)
        out = model(input_ids, labels=labels)
        (out.loss + out.aux_loss).backward()
        gate = model.model.layers[0].mlp.gate.weight
        assert gate.grad is not None, f"Mode {mode}: gate.weight.grad is None"
        assert gate.grad.abs().sum().item() > 0, f"Mode {mode}: gate.weight.grad is all-zero"


def test_gated_delta_net_untouched():
    """The working-tree diff vs HEAD must not modify the GatedDeltaNet class
    body — only the standard Attention/InstinctBlock/InstinctModel integration
    points are allowed to change."""
    proc = subprocess.run(
        ["git", "diff", "-U0", "--", "model/model_instinct_linear.py"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr

    show = subprocess.run(
        ["git", "show", "HEAD:model/model_instinct_linear.py"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert show.returncode == 0, show.stderr
    span_start, span_end = _gated_delta_net_span(show.stdout)

    for line in proc.stdout.splitlines():
        m = re.match(r"^@@ -(\d+)(?:,(\d+))?", line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) else 1
        changed = range(start, start + count)
        if changed.start < span_end and changed.stop > span_start:
            raise AssertionError(
                f"git diff touches GatedDeltaNet class body "
                f"(class lines {span_start}-{span_end - 1}) at original lines "
                f"{list(changed)}"
            )
