"""Mode 3 must reduce real replay GEMMs without changing the objective."""
import copy
from contextlib import nullcontext

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from model.model_instinct import InstinctConfig, InstinctForCausalLM


def config(mode=3, **kwargs):
    return InstinctConfig(hidden_size=32, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64,
        max_position_embeddings=64, use_grad_checkpoint=mode, **kwargs)


def set_mode(model, mode):
    model.config.use_grad_checkpoint = mode
    for child in model.modules():
        if hasattr(child, 'use_grad_checkpoint'):
            child.use_grad_checkpoint = mode


def assert_grads(actual, expected):
    for (name, a), (other, e) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == other
        assert (a.grad is None) == (e.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, e.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize('mode', [3, 4])
@pytest.mark.parametrize('flash', [False, True])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('dropout', [0., .2])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_loss_logits_gradients_and_rng(mode, flash, packed, dropout, dtype):
    torch.manual_seed(104)
    actual = InstinctForCausalLM(config(mode, flash_attn=flash, dropout=dropout)).train()
    reference = copy.deepcopy(actual)
    set_mode(reference, 0)
    tokens = torch.randint(1, 64, (2, 12))
    labels = tokens.clone()
    labels[:, -2:] = -100
    ids = None
    if packed:
        ids = (torch.arange(12) // 5).expand(2, -1).clone()
        ids[:, -2:] = -1
        labels[:, 5::5] = -100
    outputs, states = [], []
    # Two independent forward/backwards also exercise gradient accumulation
    # and ensure SAC queues are not reused across micro-batches.
    for model in (reference, actual):
        torch.manual_seed(208)
        for _ in range(2):
            ctx = torch.autocast('cpu', dtype=dtype) if dtype != torch.float32 else nullcontext()
            with ctx:
                out = model(tokens, labels=labels, sequence_ids=ids)
            out.loss.backward()
        outputs.append(out)
        states.append(torch.get_rng_state())
    torch.testing.assert_close(outputs[0].logits, outputs[1].logits, atol=0, rtol=0)
    torch.testing.assert_close(outputs[0].loss, outputs[1].loss, atol=0, rtol=0)
    assert torch.equal(states[0], states[1])
    assert_grads(actual, reference)


@pytest.mark.parametrize('mode', [3, 4])
def test_lora_with_frozen_base_and_logits_only_objective(mode):
    from model.model_lora import apply_lora
    torch.manual_seed(105)
    actual = InstinctForCausalLM(config(mode, flash_attn=False)).train()
    reference = copy.deepcopy(actual)
    set_mode(reference, 0)
    # LoRA forward closures bind their modules. Copy before injection so that
    # the reference does not accidentally call the candidate's LoRA weights.
    for model in (actual, reference):
        apply_lora(model, rank=4)
        for name, param in model.named_parameters():
            param.requires_grad_('lora' in name)
            if '.lora.B.' in name:
                with torch.no_grad():
                    param.normal_(std=.02)
    reference.load_state_dict(actual.state_dict())
    tokens = torch.randint(1, 64, (1, 12))
    for model in (reference, actual):
        model(tokens).logits.float().square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in actual.parameters() if p.requires_grad)
    assert_grads(actual, reference)


def test_mode3_eliminates_dense_gemm_replay():
    torch.manual_seed(106)
    model = InstinctForCausalLM(config(flash_attn=False)).train()
    tokens = torch.randint(1, 64, (1, 16))
    counts = {}
    for mode in (0, 2, 3, 4):
        set_mode(model, mode)
        model.zero_grad(set_to_none=True)
        with FlopCounterMode(display=False) as counter:
            model(tokens, labels=tokens).loss.backward()
        counts[mode] = counter.get_flop_counts()['Global']
    # Count executed forward+backward GEMMs; invoking an nn.Linear again is
    # not evidence of GEMM replay because SAC can serve the saved result.
    mm = torch.ops.aten.mm
    assert counts[3][mm] == counts[0][mm]
    assert counts[3][mm] < counts[2][mm]
    assert sum(counts[3].values()) < sum(counts[2].values())
    assert counts[3][mm] < counts[4][mm] < counts[2][mm]


def test_full_layer_early_stop_already_skips_down_projection_replay():
    from torch.utils._python_dispatch import TorchDispatchMode

    class CountDownShape(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.count = 0

        def __torch_dispatch__(self, op, types, args=(), kwargs=None):
            if op == torch.ops.aten.mm.default and args[0].shape == (16,96) and args[1].shape == (96,32):
                self.count += 1
            return op(*args, **(kwargs or {}))

    model = InstinctForCausalLM(config(0, flash_attn=False)).train()
    tokens = torch.ones(1,16,dtype=torch.long)
    counts = []
    for mode in (0,2):
        set_mode(model, mode)
        model.zero_grad(set_to_none=True)
        loss = model(tokens, labels=tokens).loss
        with CountDownShape() as counter:
            loss.backward()
        counts.append(counter.count)
    # Two gate/up dX GEMMs per layer share the down-forward shape. Mode 2
    # introduces no additional GEMM of this shape during its replay.
    assert counts == [4,4]


def test_mode4_bounds_gate_up_and_replays_attention():
    from model.checkpointing import dense_recipe_checkpoint_contexts, dense_checkpoint_options
    from torch.utils.checkpoint import CheckpointPolicy, SelectiveCheckpointContext
    contexts = dense_recipe_checkpoint_contexts(hidden_size=32, query_dim=32, kv_dim=16, ffn_dim=96)
    policy = contexts[0].policy_fn
    ctx = SelectiveCheckpointContext(is_recompute=False)
    assert policy(ctx, torch.ops.aten.mm.default, torch.empty(8, 32), torch.empty(32, 16)) == CheckpointPolicy.MUST_SAVE
    assert policy(ctx, torch.ops.aten.mm.default, torch.empty(8, 32), torch.empty(32, 32)) == CheckpointPolicy.MUST_RECOMPUTE
    assert policy(ctx, torch.ops.aten.mm.default, torch.empty(8, 32), torch.empty(32, 96)) == CheckpointPolicy.MUST_RECOMPUTE
    assert policy(ctx, torch.ops.aten.mm.default, torch.empty(8, 96), torch.empty(96, 32)) == CheckpointPolicy.MUST_RECOMPUTE
    assert policy(ctx, torch.ops.aten._scaled_dot_product_flash_attention.default) == CheckpointPolicy.MUST_RECOMPUTE
    cfg = config(4)
    cfg.num_hidden_layers = 8
    for layer_index in range(8):
        options = dense_checkpoint_options(cfg, None, None, layer_index, mode4_policy='kv_gate', mode4_gate_layers=4)
        policy = options['context_fn']()[0].policy_fn
        expected = CheckpointPolicy.MUST_SAVE if layer_index >= 4 else CheckpointPolicy.MUST_RECOMPUTE
        assert policy(ctx, torch.ops.aten.mm.default, torch.empty(8, 32), torch.empty(32, 96)) == expected


@pytest.mark.parametrize('shape,attention,gate_layers', [((4,3072,1536), False, 4), ((3,4096,1536), True, 2)])
def test_mode4_recipe_plan_matches_actual_buckets(shape, attention, gate_layers):
    from model.checkpointing import dense_checkpoint_options
    cfg = InstinctConfig(hidden_size=1536, intermediate_size=5504, num_hidden_layers=32,
        num_attention_heads=16, num_key_value_heads=4, head_dim=96, use_grad_checkpoint=4)
    hidden = torch.empty(shape, dtype=torch.bfloat16)
    for index in range(32):
        factory = dense_checkpoint_options(cfg, hidden, None, index)['context_fn']
        assert factory.keywords['save_attention'] == attention
        assert factory.keywords['save_gate_up'] == (index >= 32-gate_layers)


@pytest.mark.parametrize('batch,seq,dtype', [(16,4096,torch.bfloat16), (4,3072,torch.float32), (3,4096,torch.float32)])
def test_mode4_limits_selected_outputs_when_batch_or_dtype_grows(batch, seq, dtype):
    from types import SimpleNamespace
    from model.checkpointing import dense_checkpoint_options, DENSE_RECIPE_CACHE_BYTES
    cfg = InstinctConfig(hidden_size=1536, intermediate_size=5504, num_hidden_layers=32,
        num_attention_heads=16, num_key_value_heads=4, head_dim=96, use_grad_checkpoint=4)
    hidden = SimpleNamespace(shape=(batch,seq,1536), dtype=dtype, device=torch.device('cpu'))
    total = 0
    size = 4 if dtype == torch.float32 else 2
    for index in range(32):
        policy = dense_checkpoint_options(cfg, hidden, None, index)['context_fn'].keywords
        if policy['save_attention']:
            total += batch*seq*(1536*size+4*16)
        if policy['projections'] == 'kv':
            total += batch*seq*2*384*size
        if policy['save_gate_up']:
            total += batch*seq*2*5504*size
    assert 0 < total <= DENSE_RECIPE_CACHE_BYTES


@pytest.mark.parametrize('mode', [3, 4])
def test_compiled_selective_has_one_trunk_and_correct_backward(mode):
    from torch._dynamo.backends.common import aot_autograd
    from functorch.compile import make_boxed_func
    torch.manual_seed(107)
    actual = InstinctForCausalLM(config(mode, flash_attn=False)).train()
    reference = copy.deepcopy(actual)
    set_mode(reference, 0)
    graphs = []

    def compiler(gm, inputs):
        return make_boxed_func(gm.forward)

    aot = aot_autograd(fw_compiler=compiler, bw_compiler=compiler)

    def backend(gm, inputs):
        graphs.append(gm)
        return aot(gm, inputs)

    tokens = torch.randint(1, 64, (1, 12))
    expected = reference(tokens, labels=tokens).loss
    actual_loss = torch.compile(actual, backend=backend, fullgraph=True)(tokens, labels=tokens).loss
    expected.backward()
    actual_loss.backward()
    assert len(graphs) == 1
    assert graphs[0].code.count('torch.ops.higher_order.tag_activation_checkpoint(') == 2
    torch.testing.assert_close(actual_loss, expected)
    # AOT fusion/partitioning can change reduction order.
    for a, e in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, e.grad, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize('mode', [3, 4])
@pytest.mark.parametrize('kwargs', [{'use_moe': True}, {'residual_type': 'mhc'},
                                   {'residual_type': 'attnres'}])
def test_unsupported_dense_topology_is_explicit(mode, kwargs):
    with pytest.raises(ValueError, match='requires standard Dense'):
        config(mode, **kwargs)


@pytest.mark.parametrize('mode', [3, 4])
@pytest.mark.parametrize('architecture', ['linear', 'recurrsive_instinct'])
def test_unsupported_backbone_is_explicit(mode, architecture):
    from importlib import import_module
    module = import_module('model.model_instinct_linear' if architecture == 'linear'
                           else 'model.model_recurrsive_instinct')
    with pytest.raises(ValueError, match='requires standard Dense'):
        module.InstinctConfig(use_grad_checkpoint=mode)


@pytest.mark.parametrize('mode', [3, 4])
def test_eval_bypasses_checkpoint_and_training_cache_is_rejected(monkeypatch, mode):
    actual = InstinctForCausalLM(config(mode, flash_attn=False)).eval()
    tokens = torch.randint(1, 64, (1, 12))
    with monkeypatch.context() as patch:
        def forbidden(*a, **kw):
            raise AssertionError('checkpoint called in inference')
        patch.setattr(torch.utils.checkpoint, 'checkpoint', forbidden)
        actual(tokens, use_cache=True)
        actual.train()
        with torch.no_grad():
            actual(tokens, use_cache=True)
    with pytest.raises(ValueError, match='use_cache=False'):
        actual(tokens, use_cache=True)
