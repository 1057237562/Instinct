"""CPU regressions for task gradients and explicit legacy checkpoint migration."""

import copy
import json

import pytest
import torch

from model.model_instinct import InstinctForCausalLM, MOEFeedForward
from tests.helpers import make_tiny_config
from trainer import trainer_utils
from trainer.trainer_cli import (
    add_moe_router_migration_arg, build_trainer_parser,
    validate_moe_router_migration,
)


def test_top1_task_gradient_and_legacy_forward_compatibility():
    torch.manual_seed(42)
    config = make_tiny_config(use_moe=True)
    config.router_aux_loss_coef = 0.0
    legacy = MOEFeedForward(config)
    fixed = copy.deepcopy(legacy)
    fixed.config.norm_topk_prob = False
    x = torch.randn(2, 8, config.hidden_size)
    old_output = legacy(x)
    output = fixed(x)
    old_output.square().mean().backward()
    output.square().mean().backward()
    assert fixed.gate.weight.grad.norm() > 1e-6
    assert legacy.gate.weight.grad.norm() < fixed.gate.weight.grad.norm() * 1e-4
    # Switch routing keeps the selected probability, including at inference.
    probability = fixed.gate(x).softmax(-1).amax(-1, keepdim=True)
    torch.testing.assert_close(output, old_output * probability)
    fixed.eval()
    with torch.no_grad():
        torch.testing.assert_close(fixed(x), output)


@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('top_k', [2, 4, 8])
def test_multiexpert_task_gradient_is_preserved(normalize, top_k):
    torch.manual_seed(43)
    config = make_tiny_config(use_moe=True)
    config.num_experts = 8
    config.num_experts_per_tok = top_k
    config.norm_topk_prob = normalize
    config.router_aux_loss_coef = 0.0
    module = MOEFeedForward(config)
    module(torch.randn(2, 8, config.hidden_size)).square().mean().backward()
    assert module.gate.weight.grad.norm() > 1e-6


@pytest.mark.parametrize('top_k,normalize', [(1, False), (2, True), (4, True)])
def test_migration_applies_after_base_config_and_is_serializable(tmp_path, monkeypatch, top_k, normalize):
    config = make_tiny_config(use_moe=True)
    config.norm_topk_prob = True
    weight = tmp_path / 'legacy.pth'
    torch.save(InstinctForCausalLM(config).state_dict(), weight)
    weight.with_suffix('.json').write_text(json.dumps(config.to_dict()), encoding='utf-8')
    monkeypatch.setattr(trainer_utils.AutoTokenizer, 'from_pretrained', lambda *_a: object())

    current = make_tiny_config(use_moe=True)
    current.norm_topk_prob = False  # changing the preset alone must not migrate
    legacy, _ = trainer_utils.init_model(current, str(weight), device='cpu')
    assert current.norm_topk_prob is True
    migrated_config = make_tiny_config(use_moe=True)
    migrated, _ = trainer_utils.init_model(
        migrated_config, str(weight), device='cpu', router_norm_topk_prob=int(normalize),
        router_top_k=top_k,
    )
    assert migrated.config is migrated_config
    assert migrated_config.to_dict()['norm_topk_prob'] is normalize
    assert migrated_config.num_experts_per_tok == top_k
    assert set(legacy.state_dict()) == set(migrated.state_dict())
    restored = trainer_utils.restore_config_from_checkpoint(
        current, {'config': migrated_config.to_dict()},
    )
    assert restored.norm_topk_prob is normalize
    assert restored.num_experts_per_tok == top_k


def test_cli_migration_requires_new_stage():
    parser = build_trainer_parser('router-test')
    add_moe_router_migration_arg(parser)
    args = parser.parse_args(['--moe_router_norm_topk_prob', '0'])
    validate_moe_router_migration(args)
    args.from_resume = 1
    with pytest.raises(ValueError, match='new stage'):
        validate_moe_router_migration(args)
    args.moe_router_norm_topk_prob = None
    validate_moe_router_migration(args)
    args.moe_router_top_k = 8
    with pytest.raises(ValueError, match='new stage'):
        validate_moe_router_migration(args)


@pytest.mark.parametrize('top_k', [0, -1])
def test_cli_migration_rejects_nonpositive_topk(top_k):
    parser = build_trainer_parser('router-test')
    add_moe_router_migration_arg(parser)
    args = parser.parse_args(['--moe_router_top_k', str(top_k)])
    with pytest.raises(ValueError, match='positive integer'):
        validate_moe_router_migration(args)


def test_topk_migration_validated_against_restored_expert_count():
    config = make_tiny_config(use_moe=True)
    config.num_experts = 1
    with pytest.raises(ValueError, match='number of experts'):
        trainer_utils.configure_moe_training_router(config, top_k=2, norm_topk_prob=False)
    assert config.num_experts_per_tok == 1
    assert config.norm_topk_prob is True


def test_fresh_cli_moe_uses_trainable_top1():
    args = build_trainer_parser('router-test').parse_args(['--use_moe', '1'])
    assert trainer_utils.config_from_args(args).norm_topk_prob is False
