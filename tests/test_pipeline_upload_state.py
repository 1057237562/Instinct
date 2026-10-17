import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest


def test_persistence_excludes_upload_but_keeps_pipeline_snapshots(tmp_path):
    source = Path(__file__).parents[1] / 'scripts/config_webui.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('_panel_state_key_allowed', '_persist_panel_state',
                                   '_preset_fingerprint')]
    presets = [node for node in tree.body if isinstance(node, ast.Assign)
               and {t.id for t in node.targets if isinstance(t, ast.Name)}
               & {'PRESETS', '_LEGACY_PRESET_NAMES'}]
    state = {'pipeline_upload': None, 'pipeline_stages': {'pretrain': {'name': 'saved'}},
             'batch_size': 12, 'btn_load_config': True, 'clear_train_log': False,
             '_override_moe_int': True}
    namespace = {'st': SimpleNamespace(session_state=state), 'os': os, 'json': json,
                 'hashlib': hashlib}
    exec(compile(ast.Module(body=functions + presets, type_ignores=[]), str(source), 'exec'),
         namespace)
    namespace['_persist_panel_state'](str(tmp_path))
    saved = json.loads((tmp_path / 'webui_state.json').read_text(encoding='utf-8'))
    assert saved == {'pipeline_stages': state['pipeline_stages'], 'batch_size': 12,
                     '_override_moe_int': True, 'preset_fingerprint': ''}
    assert not namespace['_panel_state_key_allowed']('pipeline_upload')


def test_old_restored_null_no_longer_crashes_pipeline_panel():
    app = AppTest.from_string('''
import streamlit as st
from scripts.pipeline_panel import render
if 'restored' not in st.session_state:
    st.session_state['pipeline_upload'] = None
    st.session_state['restored'] = True
render(st, {'hidden_size':768, 'num_hidden_layers':8, 'use_moe':False}, False)
''', default_timeout=15).run()
    assert not app.exception
    app.run()
    assert not app.exception
