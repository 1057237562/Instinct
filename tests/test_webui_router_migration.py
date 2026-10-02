"""WebUI migration selection must survive CLI parsing and pipeline snapshots."""

import pytest

from scripts.moe_router_controls import migration_args
from scripts.pipeline_panel import snapshot
from trainer.trainer_cli import add_moe_router_migration_arg, build_trainer_parser


@pytest.mark.parametrize('top_k,normalize', [(1, 0), (2, 1), (4, 1), (8, 1)])
@pytest.mark.parametrize('trainer', ['pretrain', 'cpt', 'full_sft'])
def test_ui_selection_becomes_explicit_cli_arguments(top_k, normalize, trainer):
    state = {'moe_router_migration': True, 'moe_router_migration_mode': 'top2'}
    # Current model settings win over obsolete migration radio state.
    model = {'use_moe': True, 'num_experts': 8,
             'num_experts_per_tok': top_k, 'norm_topk_prob': True}
    options = migration_args(state, model, trainer)
    parser = build_trainer_parser('webui-test')
    add_moe_router_migration_arg(parser)
    args = parser.parse_args([arg for key, value in options.items() for arg in (f'--{key}', str(value))])
    assert args.moe_router_top_k == top_k
    assert args.moe_router_norm_topk_prob == normalize


def test_disabled_stale_or_unsupported_controls_do_not_migrate():
    state = {'moe_router_migration': True, 'moe_router_migration_mode': 'top2'}
    assert migration_args(state, {'use_moe': True}, 'full_sft', from_resume=True) == {}
    assert migration_args(state, {'use_moe': False}, 'pretrain') == {}
    assert migration_args(state, {'use_moe': True}, 'lora') == {}
    assert migration_args({}, {'use_moe': True}, 'pretrain') == {}


def test_pipeline_snapshot_retains_explicit_migration():
    state = {'moe_router_migration': True,
             'data_path_full_sft': 'identity_clean.jsonl'}
    model = {'hidden_size': 512, 'num_hidden_layers': 32, 'use_moe': True,
             'num_experts': 8, 'num_experts_per_tok': 4, 'norm_topk_prob': True}
    stage = snapshot(state, model, 'full_sft')
    assert stage['args']['moe_router_top_k'] == 4
    assert stage['args']['moe_router_norm_topk_prob'] == 1
    model['num_experts_per_tok'] = 1
    assert stage['args']['moe_router_top_k'] == 4


def test_multiexpert_migration_preserves_selected_normalization():
    state = {'moe_router_migration': True}
    model = {'use_moe': True, 'num_experts': 8, 'num_experts_per_tok': 3,
             'norm_topk_prob': False}
    assert migration_args(state, model, 'cpt') == {
        'moe_router_top_k': 3, 'moe_router_norm_topk_prob': 0,
    }


@pytest.mark.parametrize('top_k', [0, -1, 9])
def test_invalid_migration_rejected_by_configured_expert_count(top_k):
    with pytest.raises(ValueError, match='top-k'):
        migration_args({'moe_router_migration': True},
                       {'use_moe': True, 'num_experts': 8, 'num_experts_per_tok': top_k}, 'cpt')
