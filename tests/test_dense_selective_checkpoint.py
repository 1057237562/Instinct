"""Keep V2 selective checkpointing inside the compiled transformer graph."""
import copy
from types import SimpleNamespace

import pytest
import torch

from model.checkpointing import dense_checkpoint_options, dense_attention_checkpoint_contexts


def test_selector_is_opt_in_and_requires_v2_flex(monkeypatch):
    config = SimpleNamespace(use_moe=False, residual_type='standard', hidden_size=1536,
                             head_dim=96, num_attention_heads=16, num_key_value_heads=4)
    reference = torch.empty(1, 8, 1536)
    monkeypatch.delenv('INSTINCT_DENSE_SELECTIVE_CHECKPOINT', raising=False)
    assert dense_checkpoint_options(config, reference, None) == {}
    monkeypatch.setenv('INSTINCT_DENSE_SELECTIVE_CHECKPOINT', 'attention')
    assert dense_checkpoint_options(config, reference, None) == {}
    monkeypatch.setenv('INSTINCT_DENSE_SELECTIVE_CHECKPOINT', 'invalid')
    with pytest.raises(ValueError, match='INSTINCT_DENSE_SELECTIVE_CHECKPOINT'):
        dense_checkpoint_options(config, reference, None)


def test_context_keeps_all_checkpoint_regions_in_one_trunk(monkeypatch):
    import model.model_instinct as module
    config = module.InstinctConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=32,
        max_position_embeddings=32, use_grad_checkpoint=2, flash_attn=False)
    monkeypatch.setattr(module, 'dense_checkpoint_options',
                        lambda *a: {'context_fn': dense_attention_checkpoint_contexts})
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    model = module.InstinctForCausalLM(config).train()
    tokens = torch.randint(1, 32, (1, 16))
    torch.compile(model, backend=backend)(tokens, labels=tokens).loss.backward()
    trunk = [gm for gm in graphs if gm.code.count('torch.ops.higher_order.tag_activation_checkpoint(') == 2]
    assert len(trunk) == 1, 'Wrapping checkpoint itself can silently leave every layer eager'


@pytest.mark.gpu
@pytest.mark.parametrize('fp8', [False, True])
@pytest.mark.parametrize('mode,sequence_length', [(2,137), (3,137), (4,137), (4,4096)])
def test_compiled_selective_attention_loss_and_gradients(monkeypatch, fp8, mode, sequence_length):
    from model.model_instinct import InstinctConfig, InstinctForCausalLM
    from model.flash_attn_4 import resolve_flash_attn
    from trainer.trainer_utils import cast_training_parameters, apply_torchao_fp8_training

    if mode == 4:
        # Exercise both a replayed FFN and a cached gate/up layer in the small
        # GPU model; the production 32-layer bucket plans are checked on CPU.
        from functools import partial
        import model.model_instinct as model_module
        from model.checkpointing import dense_checkpoint_options
        monkeypatch.setattr(model_module, 'dense_checkpoint_options',
                            partial(dense_checkpoint_options, mode4_gate_layers=1))

    resolve_flash_attn()
    torch.manual_seed(307)
    monkeypatch.setenv('INSTINCT_MUON_MANTISSA', '1')
    config = InstinctConfig(hidden_size=1536, intermediate_size=5504, num_hidden_layers=2,
        num_attention_heads=16, num_key_value_heads=4, head_dim=96, vocab_size=128,
        max_position_embeddings=max(256, sequence_length), use_grad_checkpoint=2)
    model = cast_training_parameters(InstinctForCausalLM(config), 'bf16').cuda().train()
    if fp8:
        apply_torchao_fp8_training(model, SimpleNamespace(fp8_training='tensorwise',
            fp8_filter='auto', dtype='bfloat16', device='cuda', use_compile=1))
    reference = copy.deepcopy(model)
    model.config.use_grad_checkpoint = mode
    actual_model, expected_model = torch.compile(model), torch.compile(reference)
    for packed in (False, True):
        tokens = torch.randint(1, 128, (1, sequence_length), device='cuda')
        labels = tokens.clone()
        labels[:, -7:] = -100
        ids = None
        if packed:
            ids = (torch.arange(sequence_length, device='cuda') // 43)[None]
            ids[:, -7:] = -1
            labels[:, 43::43] = -100
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        monkeypatch.setenv('INSTINCT_DENSE_SELECTIVE_CHECKPOINT', '0')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = expected_model(tokens, labels=labels, sequence_ids=ids).loss
        expected.backward()
        monkeypatch.setenv('INSTINCT_DENSE_SELECTIVE_CHECKPOINT', 'attention')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = actual_model(tokens, labels=labels, sequence_ids=ids).loss
        actual.backward()
        torch.testing.assert_close(actual, expected, atol=.003, rtol=.002)
        error = norm_a = norm_e = dot = 0.
        for a, e in zip(model.parameters(), reference.parameters()):
            assert (a.grad is None) == (e.grad is None)
            if a.grad is not None:
                ga, ge = a.grad.float(), e.grad.float()
                assert torch.isfinite(ga).all()
                error += float((ga - ge).square().sum())
                norm_a += float(ga.square().sum())
                norm_e += float(ge.square().sum())
                dot += float((ga * ge).sum())
        if fp8:
            assert dot / (norm_a * norm_e)**.5 > .99
            assert .95 < (norm_a / norm_e)**.5 < 1.05
        else:
            assert (error / norm_e)**.5 < .02
