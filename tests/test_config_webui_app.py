"""End-to-end smoke tests for the Config WebUI script via Streamlit AppTest."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from scripts.config_templates import load_model_presets, load_legacy_preset_names


SOURCE = Path(__file__).parents[1] / "scripts" / "config_webui.py"
STATE_FILE = Path(__file__).parents[1] / "trainer" / "webui_state.json"


@pytest.fixture()
def guard_state_file(tmp_path, monkeypatch):
    """AppTest 运行会真实读写 trainer/webui_state.json；测试前后保护用户存档。"""
    from scripts import config_templates
    monkeypatch.setattr(config_templates, 'TEMPLATES_FILE', tmp_path / 'templates.json')
    backup = STATE_FILE.read_bytes() if STATE_FILE.exists() else None
    STATE_FILE.unlink(missing_ok=True)
    yield
    if backup is None:
        STATE_FILE.unlink(missing_ok=True)
    else:
        STATE_FILE.write_bytes(backup)


def _preset_fingerprint_fn():
    """Extract _preset_fingerprint without importing the Streamlit script."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_preset_fingerprint":
            selected.append(node)
        elif isinstance(node, ast.Assign) and {
            target.id for target in node.targets if isinstance(target, ast.Name)
        } & {"PRESETS", "_LEGACY_PRESET_NAMES"}:
            selected.append(node)
    namespace = {"json": json, "hashlib": hashlib,
                 "load_model_presets": load_model_presets,
                 "load_legacy_preset_names": load_legacy_preset_names}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_preset_fingerprint"]


def _run_app():
    app = AppTest.from_file(str(SOURCE), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def test_app_boots_and_autosaves_state(guard_state_file):
    app = _run_app()
    # 页面正常渲染，且本轮交互结束时自动保存已写入状态文件。
    assert STATE_FILE.exists()
    saved = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    assert saved["preset"] == app.session_state["preset"]
    assert saved["preset_fingerprint"] == _preset_fingerprint_fn()(saved["preset"])


def test_restore_keeps_user_tweaks_when_fingerprint_matches(guard_state_file):
    fingerprint = _preset_fingerprint_fn()("Instinct V1 MoE")
    STATE_FILE.write_text(json.dumps({
        "preset": "Instinct V1 MoE",
        "preset_fingerprint": fingerprint,
        "batch_size": 5,
        "max_seq_len": 2048,
    }), encoding="utf-8")
    app = _run_app()
    # 指纹一致：预设定义未变，用户在预设之上的修改必须原样保留。
    assert app.session_state["preset"] == "Instinct V1 MoE"
    assert app.session_state["batch_size"] == 5
    assert app.session_state["max_seq_len"] == 2048


def test_stale_state_without_fingerprint_reapplies_preset(guard_state_file):
    STATE_FILE.write_text(json.dumps({
        "preset": "Instinct V1 MoE",
        "batch_size": 5,
        "max_seq_len": 2048,
    }), encoding="utf-8")
    app = _run_app()
    # 无指纹 = 旧版本存档：重新套用当前预设定义（V1 MoE 强制 batch_size=1）。
    assert app.session_state["batch_size"] == 1
    assert app.session_state["max_seq_len"] == 4096


def test_shutdown_toggle_renders_countdown_setting(guard_state_file):
    app = AppTest.from_file(str(SOURCE), default_timeout=60)
    app.session_state["shutdown_after_train"] = True
    app.run()
    assert not app.exception
    assert any(
        checkbox.key == "shutdown_after_train" and checkbox.value
        for checkbox in app.checkbox
    )
    assert any(
        number_input.key == "shutdown_delay_seconds" and number_input.value == 60
        for number_input in app.number_input
    )


def test_launch_command_preview_rendered(guard_state_file):
    app = _run_app()
    snippets = [c.value for c in app.code]
    # 命令预览与 Start Training 的实际启动共用 _build_training_command：
    # 预览里必须出现训练脚本、保存前缀与所选数据集。
    preview = next(s for s in snippets if "train_pretrain.py" in s)
    assert "--save_weight" in preview
    assert "--epochs" in preview
    assert "--data_path" in preview
    assert preview.startswith('"') or preview.startswith("'") or "python" in preview


def test_autotune_plan_freezes_current_model_and_shared_cli(guard_state_file, tmp_path, monkeypatch):
    from scripts import autotune_panel
    captured = []
    def launch(run_dir, command):
        captured.append((run_dir, command))
        autotune_panel.write_json(run_dir / 'run.json', {'status': 'success'})
    monkeypatch.setattr(autotune_panel, 'launch', launch)
    STATE_FILE.write_text(json.dumps(dict(preset='Instinct V1 MoE',
        sequence_packing=True, sequence_packing_mode='bucket', dataset_streaming='on',
        cache_build_mode='inline', autotune_output=str(tmp_path),
        autotune_audit='obsolete.report.json', autotune_provenance='obsolete.provenance.json')),
        encoding='utf-8')
    app = AppTest.from_file(str(SOURCE), default_timeout=60)
    app.run()
    assert not app.exception
    assert any('二分搜索' in exp.label for exp in app.expander)
    plan = next(button for button in app.button if button.key == 'btn_autotune_plan')
    assert not plan.disabled
    plan.click().run()
    assert not app.exception
    assert len(captured) == 1
    directory, command = captured[0]
    spec = autotune_panel.read_json(directory / 'spec.json')
    model = autotune_panel.read_json(directory / 'model.json')
    assert model['num_experts_per_tok'] == 2 and model['norm_topk_prob'] is True
    assert spec['args']['config_path'] == str(directory / 'model.json')
    assert 'audit' not in spec and 'provenance' not in spec
    assert not any(widget.key in ('autotune_audit', 'autotune_provenance') for widget in app.text_input)
    assert not any('自动填入路径不代表审核通过' in caption.value for caption in app.caption)
    assert spec['args']['dtype'] == app.session_state['activation_dtype']
    assert spec['args']['num_workers'] == 0
    assert spec['args']['use_grad_checkpoint'] in (1, 2)
    assert '--plan-only' in command
    assert '--target-tokens-per-second' not in command
    assert 'search' not in spec
    saved = json.loads(STATE_FILE.read_text(encoding='utf-8'))
    assert saved['autotune_run_dir'] == str(directory)


def test_autotune_profile_appears_in_real_training_preview(guard_state_file, tmp_path, monkeypatch):
    from scripts import autotune_panel
    from trainer import autotune_config
    monkeypatch.setattr(autotune_config, 'source_fingerprint', lambda: 'current')
    profile = tmp_path / 'profile.json'
    autotune_panel.write_json(profile, dict(schema=1, status='validated', hardware={'source': 'current'},
        settings={'args': {'bucket_token_budget': 16384, 'use_grad_checkpoint': 2,
                           'bucket_gpu_memory_gb': 15.89, 'use_compile': 1}, 'env': {}}))
    app = AppTest.from_file(str(SOURCE), default_timeout=60)
    app.session_state['preset'] = 'Instinct V1 MoE'
    app.session_state['autotune_profile_path'] = str(profile)
    app.session_state['autotune_use_profile'] = True
    app.run()
    assert not app.exception
    preview = next(code.value for code in app.code if 'train_pretrain.py' in code.value)
    assert '--autotune_profile' in preview
    assert '--use_grad_checkpoint' not in preview
    assert '--bucket_gpu_memory_gb' not in preview
    assert not next(button for button in app.button if button.key == 'btn_start_train').disabled


def test_workspace_groups_controls_and_preparation_preserves_training_settings(guard_state_file, monkeypatch):
    from scripts import autotune_panel

    def unexpected_launch(*args, **kwargs):
        pytest.fail('Preparing UI settings must not start an experiment')

    monkeypatch.setattr(autotune_panel, 'launch', unexpected_launch)
    STATE_FILE.write_text(json.dumps(dict(
        preset='Instinct V1 MoE', preset_fingerprint=_preset_fingerprint_fn()('Instinct V1 MoE'),
        hidden_size=512, num_hidden_layers=32, use_moe=True, model_architecture='standard',
        sequence_packing=False, sequence_packing_mode='fixed', dataset_streaming='off',
        cache_build_mode='spawn', use_grad_checkpoint=0, learning_rate_pretrain=0.0003,
        batch_size=3, param_dtype='fp32', activation_dtype='bfloat16',
    )), encoding='utf-8')
    app = _run_app()
    assert [tab.label for tab in app.tabs] == [
        '训练配置', '自动调优', '模型结构', '训练流水线', '训练监控', '配置导出',
    ]
    assert not any(button.key == 'btn_start_train' for button in app.sidebar.button)
    keys = ('hidden_size', 'num_hidden_layers', 'learning_rate_pretrain', 'batch_size',
            'param_dtype', 'activation_dtype', 'data_path_pretrain', 'from_resume')
    before = {key: app.session_state[key] for key in keys}
    assert next(b for b in app.button if b.key == 'btn_autotune_plan').disabled
    next(b for b in app.button if b.key == 'btn_autotune_prepare').click().run()
    assert not app.exception
    assert {key: app.session_state[key] for key in keys} == before
    for key, value in dict(sequence_packing=True, sequence_packing_mode='bucket',
                           dataset_streaming='on', cache_build_mode='inline', use_grad_checkpoint=1).items():
        assert app.session_state[key] == value
    assert not next(b for b in app.button if b.key == 'btn_autotune_plan').disabled
    assert next(n for n in app.number_input if n.key == 'batch_size').disabled
    # A subsequent unrelated rerun must retain widgets in every workspace.
    next(n for n in app.number_input if n.key == 'autotune_max_trials').set_value(6).run()
    assert not app.exception
    saved = json.loads(STATE_FILE.read_text(encoding='utf-8'))
    assert saved['sequence_packing_mode'] == 'bucket' and saved['autotune_max_trials'] == 6
    assert saved['learning_rate_pretrain'] == before['learning_rate_pretrain']


def test_preparation_does_not_offer_to_change_training_stage(guard_state_file):
    app = _run_app()
    next(select for select in app.selectbox if select.key == 'train_type').set_value('cpt').run()
    assert not app.exception
    assert next(b for b in app.button if b.key == 'btn_autotune_prepare').disabled
    assert next(b for b in app.button if b.key == 'btn_autotune_start').disabled


def _button(app, key):
    return next(button for button in app.button if button.key == key)


def _input(app, kind, key):
    return next(widget for widget in getattr(app, kind) if widget.key == key)


def test_instinct_v2_is_added_to_existing_library_and_loads_complete_recipe(guard_state_file):
    from scripts import config_templates

    config_templates.TEMPLATES_FILE.write_text(json.dumps({
        'version': 1, 'defaults_initialized': True,
        'templates': {'custom': {'batch_size': 7}},
    }), encoding='utf-8')
    app = _run_app()
    assert 'Instinct V2' in _input(app, 'selectbox', '_template_select').options
    _input(app, 'selectbox', '_template_select').set_value('Instinct V2').run()
    _button(app, 'btn_template_load').click().run()
    assert not app.exception
    expected = dict(
        config_template='Instinct V2', preset='Instinct V2',
        model_architecture='standard', use_moe=False, hidden_size=1536,
        intermediate_size=5504, num_hidden_layers=32, num_attention_heads=16,
        num_key_value_heads=4, max_position_embeddings=4096, inference_rope_scaling=False,
        param_dtype='bf16', activation_dtype='bfloat16', optimizer='muon',
        muon_mantissa=True, muon_momentum_dtype='bf16', muon_workspace_mb=128.0,
        use_grad_checkpoint=2, max_seq_len=2048, batch_size=1, accumulation_steps=8,
        use_compile=False, fp8_training='off', epochs_pretrain=1,
        learning_rate_pretrain=0.0003, warmup_ratio_pretrain=0.02,
    )
    assert {key: app.session_state[key] for key in expected} == expected
    preview = next(code.value for code in app.code if 'train_pretrain.py' in code.value)
    for flag in ('--hidden_size 1536', '--num_hidden_layers 32', '--param_dtype bf16',
                 '--optimizer muon', '--muon_mantissa 1', '--muon_momentum_dtype bf16',
                 '--use_grad_checkpoint 2', '--max_seq_len 2048', '--accumulation_steps 8'):
        assert flag in preview, flag
    assert '--use_compile 1' not in preview
    reloaded = _run_app()
    assert {key: reloaded.session_state[key] for key in expected} == expected
    assert config_templates.load_templates()['custom'] == {'batch_size': 7}


def test_custom_template_lifecycle_restores_full_recipe_and_survives_reload(guard_state_file):
    from scripts.config_templates import load_templates

    app = _run_app()
    assert not any(radio.key == '_preset_radio' for radio in app.sidebar.radio)
    default_names = ['RecurrsiveInstinct', 'RecurrsiveInstinct MoE', 'RecurrsiveInstinct Small',
                     'Instinct V1', 'Instinct V1 MoE', 'Instinct V2', 'instinct-linear']
    assert _input(app, 'selectbox', '_template_select').options == ['选择配置模板…'] + default_names
    _input(app, 'number_input', 'batch_size').set_value(5).run()
    _input(app, 'number_input', 'learning_rate_pretrain').set_value(0.000321).run()
    app.session_state['learning_rate_dpo'] = 0.000007
    app.session_state['pipeline_stages'] = {
        'pretrain': {'trainer': 'pretrain', 'args': {'epochs': 3, 'data_path': 'pretrain_clean.parquet'}},
    }
    _input(app, 'text_input', '_template_name').set_value('完整训练 A').run()
    _button(app, 'btn_template_add').click().run()
    assert not app.exception
    saved = load_templates()['完整训练 A']
    for key in ('batch_size', 'learning_rate_pretrain', 'activation_dtype', 'param_dtype',
                'optimizer', 'compile_mode', 'data_path_pretrain', 'pipeline_stages'):
        assert saved[key] == app.session_state[key]
    assert saved['learning_rate_dpo'] == 0.000007
    assert 'btn_template_add' not in saved and 'train_triggered' not in saved

    _input(app, 'number_input', 'batch_size').set_value(7).run()
    _input(app, 'checkbox', 'early_exit_enabled').check().run()
    _input(app, 'text_input', '_early_exit_layers_str').set_value('1,2').run()
    _input(app, 'text_input', '_template_name').set_value('训练 B').run()
    _button(app, 'btn_template_add').click().run()
    _input(app, 'selectbox', '_template_select').set_value('完整训练 A').run()
    _button(app, 'btn_template_load').click().run()
    assert not app.exception
    assert app.session_state['batch_size'] == 5
    assert app.session_state['learning_rate_pretrain'] == 0.000321
    assert not app.session_state['early_exit_enabled']
    assert app.session_state['learning_rate_dpo'] == 0.000007
    assert app.session_state['pipeline_stages'] == saved['pipeline_stages']
    preview = next(code.value for code in app.code if 'train_pretrain.py' in code.value)
    assert '--batch_size 5' in preview and '--learning_rate 0.000321' in preview

    app = _run_app()
    assert app.session_state['batch_size'] == 5
    assert app.session_state['learning_rate_dpo'] == 0.000007
    assert _input(app, 'selectbox', '_template_select').value == '完整训练 A'
    _input(app, 'number_input', 'batch_size').set_value(9).run()
    _button(app, 'btn_template_update').click().run()
    assert not app.exception
    assert load_templates()['完整训练 A']['batch_size'] == 9
    _input(app, 'text_input', '_template_name').set_value('完整训练 A').run()
    _button(app, 'btn_template_add').click().run()
    assert not app.exception
    assert any('同名模板' in error.value for error in app.error)
    _button(app, 'btn_template_delete').click().run()
    assert not app.exception
    assert list(load_templates()) == default_names + ['训练 B']


@pytest.mark.parametrize('mode', [3, 4])
def test_dense_selective_checkpoint_selection_preview_and_reload(guard_state_file, mode):
    app = _run_app()
    _input(app, 'selectbox', '_template_select').set_value('Instinct V2').run()
    _button(app, 'btn_template_load').click().run()
    _input(app, 'selectbox', 'use_grad_checkpoint').set_value(mode).run()
    assert not app.exception
    preview = next(code.value for code in app.code if 'train_pretrain.py' in code.value)
    assert f'--use_grad_checkpoint {mode}' in preview
    assert app.session_state['hidden_size'] == 1536
    assert _run_app().session_state['use_grad_checkpoint'] == mode


@pytest.mark.parametrize('mode', [3, 4])
def test_dense_selective_autotune_preparation_and_plan(guard_state_file, tmp_path, monkeypatch, mode):
    from scripts import autotune_panel
    from trainer.autotune import search_space
    captured = []

    def launch(directory, command):
        captured.append((directory, command))
        autotune_panel.write_json(directory / 'run.json', {'status': 'success'})

    monkeypatch.setattr(autotune_panel, 'launch', launch)
    STATE_FILE.write_text(json.dumps(dict(
        preset='Instinct V2', preset_fingerprint=_preset_fingerprint_fn()('Instinct V2'),
        hidden_size=1536, num_hidden_layers=32, num_attention_heads=16, num_key_value_heads=4,
        use_moe=False, model_architecture='standard', residual_type='standard',
        use_grad_checkpoint=mode, sequence_packing=False, dataset_streaming='off',
        data_path_pretrain='dataset/pretrain_clean.parquet', autotune_output=str(tmp_path),
    )), encoding='utf-8')
    app = _run_app()
    assert _input(app, 'selectbox', 'use_grad_checkpoint').value == mode
    _button(app, 'btn_autotune_prepare').click().run()
    assert not app.exception
    assert app.session_state['use_grad_checkpoint'] == mode
    assert not _button(app, 'btn_autotune_plan').disabled
    _button(app, 'btn_autotune_plan').click().run()
    assert not app.exception
    assert len(captured) == 1
    directory, command = captured[0]
    assert '--plan-only' in command
    spec = autotune_panel.read_json(directory / 'spec.json')
    assert spec['args']['use_grad_checkpoint'] == mode
    assert search_space(spec, {'resolved_token_budget': 2048})['args.use_grad_checkpoint'] == [1, 2, 3, 4]


def test_sft_template_load_does_not_reapply_profile_defaults(guard_state_file):
    app = _run_app()
    _input(app, 'selectbox', 'train_type').set_value('full_sft').run()
    _input(app, 'radio', 'sft_start_mode').set_value('continue').run()
    _input(app, 'number_input', 'batch_size').set_value(6).run()
    _input(app, 'number_input', 'learning_rate_full_sft').set_value(0.000009).run()
    _input(app, 'text_input', '_template_name').set_value('SFT continue').run()
    _button(app, 'btn_template_add').click().run()
    assert not app.exception
    _input(app, 'selectbox', 'train_type').set_value('pretrain').run()
    _input(app, 'number_input', 'batch_size').set_value(2).run()
    _button(app, 'btn_template_load').click().run()
    assert not app.exception
    assert app.session_state['train_type'] == 'full_sft'
    assert app.session_state['sft_start_mode'] == 'continue'
    assert app.session_state['batch_size'] == 6
    assert app.session_state['learning_rate_full_sft'] == 0.000009
    app = _run_app()
    assert app.session_state['batch_size'] == 6
    assert app.session_state['learning_rate_full_sft'] == 0.000009


def test_default_templates_load_their_model_and_training_settings(guard_state_file):
    from scripts.config_templates import load_templates

    app = _run_app()
    for name, config in load_templates().items():
        _input(app, 'selectbox', '_template_select').set_value(name).run()
        _button(app, 'btn_template_load').click().run()
        assert not app.exception
        for key in ('hidden_size', 'num_hidden_layers', 'model_architecture', 'use_moe',
                    'batch_size', 'max_seq_len', 'optimizer', 'use_compile',
                    'learning_rate_pretrain', 'activation_dtype', 'param_dtype'):
            assert app.session_state[key] == config[key], (name, key)
