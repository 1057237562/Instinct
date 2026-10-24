"""Keep the new backend's scope tied to actual config_webui exports/choices."""
import ast
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from model.dense_cublas.config import (ARCHITECTURES, EXCLUDED, FEATURES, MODEL_FIELDS,
                                     OPTIMIZERS, TRAIN_TYPES, WebUIRequest)


SOURCE = ast.parse(Path('scripts/config_webui.py').read_text(encoding='utf-8'))


def exported_config(state):
    # Execute only pure config builders, never the Streamlit page or trainers.
    names = {'build_config_dict', 'gen_config_json', 'compute_intermediate_size', 'compute_head_dim'}
    nodes = [node for node in SOURCE.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {'st': SimpleNamespace(session_state=state), 'math': math, 'json': json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<webui-config-builders>', 'exec'), scope)
    return json.loads(scope['gen_config_json'](scope['build_config_dict']()))


def test_every_exported_field_has_an_owner_and_values_survive():
    seen = set()
    for arch in ARCHITECTURES:
        for moe in (False, True):
            for extra in (False, True):
                config = exported_config({'model_architecture': arch, 'use_moe': moe,
                                          'early_exit_enabled': extra, 'inference_rope_scaling': extra})
                request = WebUIRequest.from_dicts(config)
                for key, value in config.items():
                    assert request.model[key] == (True if key == 'use_moe' and arch == 'moe' else value)
                seen.update(config)
    owned = {key for keys in MODEL_FIELDS.values() for key in keys}
    assert seen == owned


def test_choices_match_real_webui_and_all_training_modes_are_requirements():
    choices = {}
    for node in ast.walk(SOURCE):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'selectbox':
            key = next((item.value.value for item in node.keywords if item.arg == 'key' and isinstance(item.value, ast.Constant)), None)
            if len(node.args) > 1 and isinstance(node.args[1], ast.List):
                choices[key] = ast.literal_eval(node.args[1])
    assert tuple(choices['train_type']) == TRAIN_TYPES
    assert tuple(choices['optimizer']) == OPTIMIZERS
    for mode in TRAIN_TYPES:
        for optimizer in OPTIMIZERS:
            for dtype in choices['kv_cache_dtype']:
                r = WebUIRequest.from_dicts({}, {'train_type': mode, 'optimizer': optimizer, 'kv_cache_dtype': dtype})
                assert r.training['optimizer'] == optimizer and r.training['kv_cache_dtype'] == dtype
                if mode not in ('pretrain', 'full_sft'):
                    assert mode in r.required_features()
                assert r.report()['implementation_ready'] and not r.report()['training_ready']
                with pytest.raises(RuntimeError):
                    r.require_training_ready()


def test_compile_is_only_exclusion_and_unsupported_features_are_visible():
    config = exported_config({'model_architecture': 'looped', 'use_moe': True,
                              'early_exit_enabled': True, 'inference_rope_scaling': True,
                              'dropout': 0.2, 'hidden_act': 'gelu', 'use_grad_checkpoint': 2})
    request = WebUIRequest.from_dicts(config, {'use_compile': True, 'compile_mode': 'max-autotune', 'train_type': 'agent'})
    assert EXCLUDED == {'use_compile', 'compile_mode'}
    assert request.excluded == {'training.use_compile': True, 'training.compile_mode': 'max-autotune'}
    assert {'looped', 'moe', 'early_exit', 'dropout', 'recompute', 'agent', 'grpo', 'kv_cache'} <= set(request.required_features())
    assert request.model['hidden_act'] == 'gelu'
    for key, feature in FEATURES.items():
        assert all(dep in FEATURES for dep in feature.depends_on), key


@pytest.mark.parametrize('config,training', [
    ({'typo': 1}, {}), ({}, {'optimizer': 'sgd'}), ({'hidden_size': 0}, {}),
    ({'num_attention_heads': 3, 'num_key_value_heads': 2}, {}),
    ({'head_dim': 7}, {}), ({'early_exit_layers': [9]}, {}),
    ({'rope_scaling': {'factor': -2}}, {}), ({}, {'batch_size': 1.5}),
    ({'hidden_act': 'bad'}, {}), ({'use_grad_checkpoint': 3}, {}),
])
def test_invalid_settings_are_not_silently_downgraded(config, training):
    with pytest.raises(ValueError):
        WebUIRequest.from_dicts(config, training)


def test_defaults_match_webui_dimension_rules():
    for h in (128, 512, 768, 2048):
        r = WebUIRequest.from_dicts({'hidden_size': h})
        assert r.model['intermediate_size'] == exported_config({'hidden_size': h})['intermediate_size']


def test_planning_needs_neither_gpu_nor_framework_and_ready_check_fails():
    script = '''
import sys
from experiments.smoke_dense_triton import RejectTorch
sys.meta_path.insert(0, RejectTorch())
from experiments.plan_cublas import main
sys.argv = ['plan', '--config_path', 'trainer/config_pretrain.json', '--require_ready']
result = main()
assert not any(n in sys.modules for n in ('torch', 'torchao', 'triton'))
raise SystemExit(result)
'''
    result = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True)
    assert result.returncode == 2, result.stderr
    report = json.loads(result.stdout)
    assert not report['training_ready'] and 'recompute' in report['blockers']


def test_webui_recovers_native_mode_without_invalid_selectbox_value():
    import re
    node=next(n for n in SOURCE.body if isinstance(n,ast.FunctionDef) and n.name=='_training_type_from_command')
    scope={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),'<recovery>','exec'),scope)
    parse=scope['_training_type_from_command']
    assert parse('python -u trainer/train_cublas.py --train_type grpo --fp8 auto')=='cublas:grpo'
    assert parse('python trainer/train_full_sft.py --batch_size 4')=='full_sft'
    assert parse('python run_linear.py trainer/train_pretrain.py')=='pretrain'


def test_native_resume_discovers_old_pth_and_native_npz(tmp_path):
    import os
    repo=tmp_path/'repo';(repo/'scripts').mkdir(parents=True);(repo/'checkpoints').mkdir()
    pth=repo/'checkpoints'/'pretrain_20260101_8_resume.pth';pth.touch()
    names={'_latest_checkpoint_prefix','_default_weight_prefix'}
    nodes=[n for n in SOURCE.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'os':os,'__file__':str(repo/'scripts'/'config_webui.py'),
        'st':SimpleNamespace(session_state={'runtime_backend':'cublas'}),'_DEFAULT_WEIGHT_PREFIX':{'pretrain':'pretrain'}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<checkpoint-discovery>','exec'),scope)
    discover=scope['_latest_checkpoint_prefix']
    assert discover('pretrain',8,False)=='pretrain_20260101'
    native=repo/'checkpoints'/'pretrain_20260102_8_resume.npz';native.touch();os.utime(native,(pth.stat().st_mtime+10,)*2)
    assert discover('pretrain',8,False)=='pretrain_20260102'
    scope['st'].session_state['runtime_backend']='pytorch'
    assert discover('pretrain',8,False)=='pretrain_20260101'
