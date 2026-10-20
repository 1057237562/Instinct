"""Configuration templates preserve recipes without replaying live actions."""

import json

import pytest

from scripts.config_templates import (
    delete_template, load_templates, restore_config, save_template, snapshot_config,
    initialize_default_templates,
)


def test_complete_recipe_roundtrip_and_runtime_isolation(tmp_path):
    path = tmp_path / 'templates.json'
    recipe = {
        'hidden_size': 512, 'use_moe': True, 'moe_intermediate_size': 1728,
        '_override_moe_int': True, 'train_type': 'full_sft', 'sft_start_mode': 'continue',
        '_sft_active_profile': 'continue', '_sft_last_train_type': 'full_sft',
        '_panel_synced_base': 'full_sft_123', 'epochs_pretrain': 3, 'epochs_full_sft': 2,
        'learning_rate_full_sft': 3e-6, 'data_path_pretrain': 'pretrain_clean.parquet',
        'data_path_full_sft': 'sft_clean.jsonl', 'from_resume': False,
        'batch_size': 3, 'optimizer': 'muon', 'activation_dtype': 'bfloat16',
        'fp8_training': 'tensorwise', 'kv_cache_dtype': 'bf16',
        'dataset_streaming': 'on', 'streaming_chunk_mb': 512,
        'sft_mode_profiles': {'base': {'epochs_full_sft': 5}},
        'pipeline_stages': {'pretrain': {'args': {'epochs': 3}}},
        'autotune_profile_path': 'profile.json', 'shutdown_after_train': True,
        'shutdown_delay_seconds': 90,
    }
    state = dict(recipe, train_status='running', train_triggered=True,
                 train_proc=object(), shutdown_deadline=123,
                 pipeline_run_dir='old_run', autotune_run_dir='old_tuning',
                 pipeline_upload={'upload': 'widget'}, btn_start_train=True,
                 save_prefix_full_sft='old_job', _arch_radio='stale')
    save_template('完整训练', state, path=path)
    saved = load_templates(path)['完整训练']
    assert saved == recipe
    recipe['pipeline_stages']['pretrain']['args']['epochs'] = 99
    assert saved['pipeline_stages']['pretrain']['args']['epochs'] == 3
    live_process = object()
    target = {'train_proc': live_process, 'train_status': 'running',
              'old_optional_setting': True, '_override_lin_kd': True,
              '_pending_widget_batch_size': 100, '_early_exit_layers_str': '1,2',
              '_arch_radio': 'stale', '_residual_radio': 'stale'}
    restore_config(target, saved)
    assert target['train_proc'] is live_process
    assert target['train_status'] == 'running'
    assert 'old_optional_setting' not in target and '_override_lin_kd' not in target
    assert '_pending_widget_batch_size' not in target and '_early_exit_layers_str' not in target
    assert snapshot_config(target) == dict(saved, preset='Custom')


def test_library_duplicate_update_delete_and_corruption(tmp_path):
    path = tmp_path / 'templates.json'
    assert load_templates(path) == {}
    with pytest.raises(ValueError, match='名称'):
        save_template('  ', {}, path=path)
    save_template('first', {'batch_size': 1}, path=path)
    save_template('second', {'batch_size': 2}, path=path)
    with pytest.raises(ValueError, match='同名'):
        save_template('first', {'batch_size': 9}, path=path)
    assert load_templates(path)['first']['batch_size'] == 1
    save_template('first', {'batch_size': 4}, overwrite=True, path=path)
    delete_template('second', path)
    assert load_templates(path) == {'first': {'batch_size': 4}}
    path.write_text('{broken', encoding='utf-8')
    with pytest.raises(ValueError):
        save_template('third', {}, path=path)
    assert path.read_text(encoding='utf-8') == '{broken'


def test_defaults_seed_once_without_overwriting_custom_templates(tmp_path):
    path = tmp_path / 'templates.json'
    save_template('Instinct V1', {'batch_size': 7}, path=path)
    save_template('custom', {'batch_size': 9}, path=path)
    initialize_default_templates(path)
    templates = load_templates(path)
    assert len(templates) == 8
    assert templates['Instinct V1'] == {'batch_size': 7}
    assert templates['custom'] == {'batch_size': 9}
    moe = templates['Instinct V1 MoE']
    assert moe['batch_size'] == 1 and moe['optimizer'] == 'muon'
    assert moe['num_experts_per_tok'] == 2 and moe['max_seq_len'] == 4096
    assert moe['activation_dtype'] == 'bfloat16' and moe['param_dtype'] == 'fp32'
    assert all(key in moe for key in ('data_path_pretrain', 'learning_rate_pretrain',
                                      'epochs_full_sft', 'pipeline_stages'))
    save_template('Instinct V1 MoE', {'batch_size': 3}, overwrite=True, path=path)
    delete_template('RecurrsiveInstinct', path)
    initialize_default_templates(path)
    templates = load_templates(path)
    assert 'RecurrsiveInstinct' not in templates
    assert templates['Instinct V1 MoE'] == {'batch_size': 3}


def test_legacy_library_receives_v2_without_restoring_deleted_defaults(tmp_path):
    path = tmp_path / 'templates.json'
    path.write_text(json.dumps({
        'version': 1, 'defaults_initialized': True,
        'templates': {'Instinct V1': {'batch_size': 7}, 'custom': {'batch_size': 9}},
    }), encoding='utf-8')

    initialize_default_templates(path)
    templates = load_templates(path)
    assert set(templates) == {'Instinct V1', 'custom', 'Instinct V2'}
    assert templates['Instinct V1'] == {'batch_size': 7}
    assert templates['custom'] == {'batch_size': 9}
    v2 = templates['Instinct V2']
    assert (v2['hidden_size'], v2['num_hidden_layers'], v2['intermediate_size']) == (1536, 32, 5504)
    assert v2['param_dtype'] == 'bf16' and v2['muon_mantissa'] is True
    assert v2['muon_momentum_dtype'] == 'bf16' and v2['use_grad_checkpoint'] == 2
    delete_template('Instinct V2', path)
    initialize_default_templates(path)
    assert 'Instinct V2' not in load_templates(path)


def test_default_updates_preserve_customizations_and_deletions(tmp_path, monkeypatch):
    from scripts import config_templates

    defaults_path = tmp_path / 'defaults.json'
    monkeypatch.setattr(config_templates, 'DEFAULT_TEMPLATES_FILE', defaults_path)
    defaults = {name: {'hidden_size': 512} for name in ('untouched', 'edited', 'deleted')}
    defaults_path.write_text(json.dumps({'version': 1, 'templates': defaults}), encoding='utf-8')
    path = tmp_path / 'templates.json'
    initialize_default_templates(path)
    save_template('edited', {'hidden_size': 640}, overwrite=True, path=path)
    delete_template('deleted', path)
    for config in defaults.values():
        config['hidden_size'] = 768
    defaults['new'] = {'hidden_size': 1024}
    defaults_path.write_text(json.dumps({'version': 1, 'templates': defaults}), encoding='utf-8')

    initialize_default_templates(path)
    assert load_templates(path) == {
        'untouched': {'hidden_size': 768}, 'edited': {'hidden_size': 640},
        'new': {'hidden_size': 1024},
    }
    previous = path.read_bytes()
    initialize_default_templates(path)
    assert path.read_bytes() == previous


def test_catalog_change_reaches_templates_presets_and_relay_without_python_edits(tmp_path, monkeypatch):
    from scripts import config_templates
    from data_relay.generate_training_schema import presets as relay_presets

    catalog = json.loads(config_templates.DEFAULT_TEMPLATES_FILE.read_text(encoding='utf-8'))
    catalog['presets']['Instinct V2']['intermediate_size'] = 5632
    catalog['presets']['New Dense'] = dict(catalog['presets']['Instinct V2'], num_hidden_layers=16)
    catalog['templates']['New Dense'] = {'batch_size': 2}
    catalog['preset_training_keys']['New Dense'] = ['batch_size']
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(catalog), encoding='utf-8')
    monkeypatch.setattr(config_templates, 'DEFAULT_TEMPLATES_FILE', path)

    assert config_templates.load_model_presets()['Instinct V2']['intermediate_size'] == 5632
    templates = config_templates.load_default_templates()
    assert templates['Instinct V2']['intermediate_size'] == 5632
    assert templates['New Dense']['num_hidden_layers'] == 16
    assert templates['New Dense']['batch_size'] == 2
    initial = config_templates.preset_initial_state('New Dense')
    assert initial['num_hidden_layers'] == 16 and initial['batch_size'] == 2
    relay = relay_presets()
    assert relay['Instinct V2']['intermediate_size'] == 5632
    assert relay['New Dense']['num_hidden_layers'] == 16
    assert relay['New Dense']['batch_size'] == 2
    initialize_default_templates(tmp_path / 'library.json')
    assert 'New Dense' in load_templates(tmp_path / 'library.json')
