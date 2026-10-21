"""Mode 1 replays attention cores, never projections or FFN modules."""
import copy
import importlib

import pytest
import torch

from tests.helpers import make_tiny_config


@pytest.mark.parametrize('variant,module_name', [
    ('dense', 'model.model_instinct'),
    ('loop', 'model.model_recurrsive_instinct'),
    ('linear', 'model.model_instinct_linear'),
])
@pytest.mark.parametrize('moe', [False, True])
def test_mode1_replays_fused_core_but_not_projections_or_ffn(monkeypatch, variant, module_name, moe):
    module = importlib.import_module(module_name)
    config = make_tiny_config(use_moe=moe, variant=variant)
    config.use_grad_checkpoint = 1
    config.flash_attn = True
    config.dropout = 0.0
    if variant == 'linear':
        config.layer_types = ['full_attention'] * config.num_hidden_layers
    if variant == 'loop':
        config.loop_iters = config.mean_recurrence = 2
        config.mean_backprop_depth = 2
        config.recurrence_sampling = 'fixed'
        config.state_init_std = 0.0
        config.recurrent_layers = 1
    model = module.InstinctForCausalLM(config).train()
    original = module.flash_attention
    calls = {'attention': 0, 'projection': 0, 'ffn': 0}

    def attention(*args, **kwargs):
        calls['attention'] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'flash_attention', attention)
    handles = []
    for name, child in model.named_modules():
        kind = 'ffn' if name.endswith('.mlp') else (
            'projection' if name.endswith(('.q_proj', '.k_proj', '.v_proj', '.o_proj')) else None)
        if kind:
            def count(_module, _args, kind=kind):
                calls[kind] += 1
            handles.append(child.register_forward_pre_hook(count))
    tokens = torch.randint(1, config.vocab_size, (2, 11))
    output = model(tokens, labels=tokens)
    forward = dict(calls)
    (output.loss + output.aux_loss).backward()
    assert forward['attention'] > 0 and forward['ffn'] > 0
    assert calls['attention'] == 2 * forward['attention']
    assert calls['projection'] == forward['projection']
    assert calls['ffn'] == forward['ffn']
    for handle in handles:
        handle.remove()


def test_mode1_compile_contains_only_attention_checkpoint_regions():
    from model.model_instinct import InstinctConfig, InstinctForCausalLM
    from model.flash_attn_4 import resolve_flash_attn
    resolve_flash_attn()
    cfg = InstinctConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=32,
        max_position_embeddings=32, use_grad_checkpoint=1, flash_attn=True)
    regions = []

    def backend(gm, inputs):
        for node in gm.graph.nodes:
            if node.op == 'call_function' and 'tag_activation_checkpoint' in str(node.target):
                regions.append(gm.get_submodule(node.args[0].target).code)
        return gm.forward

    tokens = torch.randint(1, 32, (1, 13))
    torch.compile(InstinctForCausalLM(cfg).train(), backend=backend)(tokens, labels=tokens).loss.backward()
    assert len(regions) == 2
    for code in regions:
        assert 'scaled_dot_product_attention' in code
        assert 'linear' not in code and 'silu' not in code


def test_fa4_checkpoint_early_stop_does_not_switch_backend(monkeypatch):
    from types import SimpleNamespace
    from torch.utils.checkpoint import _StopRecomputationError
    module = importlib.import_module('model.flash_attn_4')
    monkeypatch.setenv('INSTINCT_CAUSAL_ATTENTION_BACKEND', 'sdpa')

    def replay_complete(*args, **kwargs):
        raise _StopRecomputationError()

    monkeypatch.setattr(module, '_get_fa4', lambda: replay_complete)
    shape = (1, 128, 2, 8)
    q = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, shape=shape,
                        is_contiguous=lambda: True, size=lambda i: shape[i])
    with pytest.raises(_StopRecomputationError):
        module.flash_attention(q, q, q, is_causal=True)


def test_fused_attention_only_preserves_dropout_gradients_and_rng():
    from model.model_instinct import InstinctForCausalLM
    from tests.helpers import assert_grads_equal
    config = make_tiny_config()
    config.flash_attn, config.dropout = True, 0.2
    reference = InstinctForCausalLM(config).train()
    actual = copy.deepcopy(reference)
    actual.config.use_grad_checkpoint = 1
    for child in actual.modules():
        if hasattr(child, 'use_grad_checkpoint'):
            child.use_grad_checkpoint = 1
    tokens = torch.arange(1, 12)[None]
    torch.manual_seed(123)
    expected = reference(tokens, labels=tokens).loss
    expected.backward()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(123)
    got = actual(tokens, labels=tokens).loss
    got.backward()
    assert torch.equal(got, expected)
    assert_grads_equal(actual, reference)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.gpu
@pytest.mark.parametrize('fp8', [False, True])
def test_v2_compiled_attention_only_loss_and_gradients(fp8, monkeypatch):
    from model.model_instinct import InstinctConfig, InstinctForCausalLM
    from model.flash_attn_4 import resolve_flash_attn
    from trainer.trainer_utils import cast_training_parameters, apply_torchao_fp8_training
    from types import SimpleNamespace
    resolve_flash_attn()
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', '1')
    torch.manual_seed(307)
    cfg = InstinctConfig(hidden_size=1536, intermediate_size=5504, num_hidden_layers=2,
        num_attention_heads=16, num_key_value_heads=4, head_dim=96, vocab_size=128,
        max_position_embeddings=256, use_grad_checkpoint=0)
    reference = cast_training_parameters(InstinctForCausalLM(cfg), 'bf16').cuda().train()
    if fp8:
        apply_torchao_fp8_training(reference, SimpleNamespace(fp8_training='tensorwise',
            fp8_filter='auto', dtype='bfloat16', device='cuda', use_compile=1))
    actual = copy.deepcopy(reference)
    actual.config.use_grad_checkpoint = 1
    for child in actual.modules():
        if hasattr(child, 'use_grad_checkpoint'):
            child.use_grad_checkpoint = 1
    expected_model, actual_model = torch.compile(reference), torch.compile(actual)
    for packed in (False, True):
        tokens = torch.randint(1, 128, (1, 137), device='cuda')
        labels = tokens.clone()
        labels[:, -7:] = -100
        ids = None
        if packed:
            ids = (torch.arange(137, device='cuda') // 43)[None]
            ids[:, -7:] = -1
            labels[:, 43::43] = -100
        reference.zero_grad(set_to_none=True)
        actual.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = expected_model(tokens, labels=labels, sequence_ids=ids).loss
        expected.backward()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            got = actual_model(tokens, labels=labels, sequence_ids=ids).loss
        got.backward()
        torch.testing.assert_close(got, expected, atol=.003, rtol=.002)
        error = norm_a = norm_e = dot = 0.
        for a, e in zip(actual.parameters(), reference.parameters()):
            assert (a.grad is None) == (e.grad is None)
            if a.grad is not None:
                ga, ge = a.grad.float(), e.grad.float()
                assert torch.isfinite(ga).all()
                error += float((ga-ge).square().sum())
                norm_a += float(ga.square().sum())
                norm_e += float(ge.square().sum())
                dot += float((ga*ge).sum())
        if fp8:
            assert dot / (norm_a*norm_e)**.5 > .99
            assert .95 < (norm_a/norm_e)**.5 < 1.05
        else:
            assert (error/norm_e)**.5 < .02
        if packed and fp8 and torch.cuda.get_device_capability()[0] == 12:
            actual.zero_grad(set_to_none=True)
            # A correct loss alone would not catch checkpointing silently
            # decomposing the fused attention into dense matrix operations.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as trace:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    actual_model(tokens, labels=labels, sequence_ids=ids).loss.backward()
                torch.cuda.synchronize()
            kernels = [event.name for event in trace.events()
                       if event.device_type == torch.autograd.DeviceType.CUDA]
            assert any('flex_attention' in name for name in kernels), kernels
