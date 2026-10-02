from scripts.cpt_defaults import apply_cpt_defaults
from scripts.moe_router_controls import migration_args


def test_upgrade_to_low_budget_cpt_preserves_selected_device_and_data():
    state = {'cpt_defaults_version': 3, 'use_moe': True,
             'bucket_gpu_memory_gb': 28.0, 'data_path_cpt': 'reviewed_identity_clean.jsonl',
             'base_weight': 'checkpoints/pretrain_20260925_121129_512_moe.pth'}
    assert apply_cpt_defaults(state)
    assert state['optimizer'] == 'muon'
    assert state['learning_rate_cpt'] == 3e-5
    assert state['warmup_ratio_cpt'] == .03
    assert state['sequence_packing_mode'] == 'bucket'
    assert state['bucket_max_seq_len'] == 4096
    assert state['data_cache_max_gb'] == 20.0
    assert state['dataset_streaming'] == 'on'
    assert state['bucket_gpu_memory_gb'] == 28.0
    assert state['data_path_cpt'] == 'reviewed_identity_clean.jsonl'
    assert state['base_weight'].endswith('_512_moe.pth')
    assert migration_args(state, {'use_moe': True}, 'cpt') == {
        'moe_router_top_k': 1, 'moe_router_norm_topk_prob': 0,
    }


def test_defaults_apply_once_but_explicit_button_can_restore_them():
    state = {'use_moe': True}
    apply_cpt_defaults(state)
    state['learning_rate_cpt'] = 2e-5
    assert not apply_cpt_defaults(state)
    assert state['learning_rate_cpt'] == 2e-5
    assert apply_cpt_defaults(state, force=True)
    assert state['learning_rate_cpt'] == 3e-5


def test_resume_and_live_runs_are_never_reconfigured():
    for state in ({'from_resume': True}, {'train_status': 'running'}, {'train_status': 'paused'}):
        before = dict(state)
        assert not apply_cpt_defaults(state, force=True)
        assert state == before


def test_dense_cpt_does_not_request_moe_migration():
    state = {'use_moe': False}
    apply_cpt_defaults(state)
    assert not state['moe_router_migration']
