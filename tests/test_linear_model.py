"""TDD tests for gradient-checkpoint wiring in the linear variant (plan T9).

RED first: before ``model_instinct_linear.py`` is wired, ``test_wiring_attributes``
raises ``AttributeError`` (``use_grad_checkpoint`` does not exist on ``Attention``
/ ``InstinctBlock`` yet) so the suite is genuinely red. After wiring, all
assertions must pass with ``torch.equal`` (bitwise) on CPU fp32.

What is covered:

1. ``test_mode012_loss_equal_full_attn`` — ``full_attention_interval=1`` turns
   every layer into the standard eager ``Attention`` (no GatedDeltaNet). Mode 1
   (Attention-only recompute of the attention core) and Mode 2
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
4. Mode 1 also checkpoints the parameter-free GatedDeltaNet core, retaining
   the same loss/gradients while leaving projections and FFNs outside replay.
"""

import torch

from model.model_instinct_linear import InstinctForCausalLM
from tests.helpers import assert_grads_equal, make_tiny_config

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


def test_packed_linear_segment_matches_standalone_forward():
    """Packing resets GatedDeltaNet convolution and recurrent state per segment."""
    model = _make_model(0)
    model.eval()
    second = torch.tensor([[20, 21, 22]])
    packed = torch.tensor([[5, 6, 7, 20, 21, 22]])
    sequence_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

    with torch.no_grad():
        standalone_logits = model(second).logits
        packed_logits = model(packed, sequence_ids=sequence_ids).logits[:, 3:]

    # Batched fallback kernels can differ from batch-size-1 by a few fp32 ulps.
    torch.testing.assert_close(packed_logits, standalone_logits, rtol=2e-4, atol=2e-6)


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


def test_mode1_linear_core_loss_and_gradients(monkeypatch):
    from model.model_instinct_linear import GatedDeltaNet
    input_ids, labels = _make_inputs()
    reference, actual = _make_model(0), _make_model(1)
    calls = {"core": 0, "projection": 0, "ffn": 0}
    original = GatedDeltaNet._chunk_core

    def core(*args):
        calls["core"] += 1
        return original(*args)

    monkeypatch.setattr(GatedDeltaNet, '_chunk_core', staticmethod(core))
    expected = reference(input_ids, labels=labels)
    expected.loss.backward()
    calls["core"] = 0
    for name, child in actual.named_modules():
        kind = 'ffn' if name.endswith('.mlp') else (
            'projection' if name.endswith(('.in_proj_qkv', '.out_proj')) else None)
        if kind:
            def count(_module, _args, kind=kind):
                calls[kind] += 1
            child.register_forward_pre_hook(count)
    got = actual(input_ids, labels=labels)
    forward = dict(calls)
    got.loss.backward()
    assert torch.equal(got.loss, expected.loss)
    assert_grads_equal(actual, reference)
    assert calls['core'] == forward['core'] * 2
    assert calls['projection'] == forward['projection']
    assert calls['ffn'] == forward['ffn']
