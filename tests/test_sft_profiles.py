"""Switching SFT radio modes must change launch inputs without losing edits."""

from scripts.sft_profiles import switch_sft_profile, track_sft_training_type
from scripts.pipeline_panel import snapshot


def test_new_modes_have_distinct_optimization_defaults_and_no_resume():
    state = {'param_dtype': 'fp16', 'optimizer': 'muon', 'bucket_gpu_memory_gb': 28}
    switch_sft_profile(state, 'base')
    assert (state['learning_rate_full_sft'], state['epochs_full_sft']) == (1e-5, 2)
    assert state['warmup_ratio_full_sft'] == .03
    assert state['param_dtype'] == 'fp32'
    switch_sft_profile(state, 'continue')
    assert (state['learning_rate_full_sft'], state['epochs_full_sft']) == (3e-6, 1)
    assert state['warmup_ratio_full_sft'] == .05
    assert state['optimizer'] == 'muon'
    assert state['bucket_gpu_memory_gb'] == 28
    assert not state['from_resume']
    assert not state['moe_router_migration']


def test_round_trip_restores_independent_user_edits_including_dataset_and_base():
    state = {}
    switch_sft_profile(state, 'base')
    state.update(learning_rate_full_sft=8e-6, epochs_full_sft=3,
                 data_path_full_sft='first_identity_clean.jsonl', base_weight='out/pretrain.pth',
                 max_seq_len=4096, compile_mode='max-autotune-no-cudagraphs')
    switch_sft_profile(state, 'continue')
    state.update(learning_rate_full_sft=2e-6, data_path_full_sft='next_identity_clean.jsonl',
                 base_weight='out/full_sft.pth', max_seq_len=2048, compile_mode='default')
    switch_sft_profile(state, 'base')
    assert state['learning_rate_full_sft'] == 8e-6
    assert state['epochs_full_sft'] == 3
    assert state['data_path_full_sft'] == 'first_identity_clean.jsonl'
    assert state['base_weight'] == 'out/pretrain.pth'
    assert state['max_seq_len'] == 4096
    assert state['compile_mode'] == 'max-autotune-no-cudagraphs'
    switch_sft_profile(state, 'continue')
    assert state['learning_rate_full_sft'] == 2e-6
    assert state['data_path_full_sft'] == 'next_identity_clean.jsonl'
    assert state['base_weight'] == 'out/full_sft.pth'
    assert state['max_seq_len'] == 2048


def test_resume_preserves_current_run_and_does_not_poison_new_stage_profile():
    state = {'learning_rate_full_sft': 7e-6, 'epochs_full_sft': 4,
             'optimizer': 'adamw', 'data_path_full_sft': 'same.jsonl'}
    before = dict(state)
    switch_sft_profile(state, 'resume')
    assert state['from_resume']
    assert all(state[key] == value for key, value in before.items())
    switch_sft_profile(state, 'continue')
    state['learning_rate_full_sft'] = 2e-6
    switch_sft_profile(state, 'resume')
    state['learning_rate_full_sft'] = 1e-6
    switch_sft_profile(state, 'continue')
    assert state['learning_rate_full_sft'] == 2e-6


def test_reruns_preserve_edits_and_running_mode_is_not_reconfigured():
    state = {}
    switch_sft_profile(state, 'continue')
    state['learning_rate_full_sft'] = 4e-6
    assert not switch_sft_profile(state, 'continue')
    assert state['learning_rate_full_sft'] == 4e-6
    state['train_status'] = 'running'
    before = dict(state)
    assert not switch_sft_profile(state, 'base')
    assert state == before


def test_other_stage_settings_do_not_leak_back_into_sft():
    state = {}
    track_sft_training_type(state, 'full_sft')
    switch_sft_profile(state, 'continue')
    state['learning_rate_full_sft'] = 2e-6
    track_sft_training_type(state, 'cpt')
    state.update(optimizer='adamw', compile_mode='reduce-overhead')
    entering = track_sft_training_type(state, 'full_sft')
    switch_sft_profile(state, 'continue', entering=entering)
    assert state['learning_rate_full_sft'] == 2e-6
    assert state['optimizer'] == 'muon'
    assert state['compile_mode'] == 'default'


def test_pipeline_snapshot_uses_selected_sft_profile():
    state = {'data_path_full_sft': 'identity_clean.jsonl'}
    model = {'hidden_size': 512, 'num_hidden_layers': 32, 'use_moe': True}
    switch_sft_profile(state, 'base')
    first = snapshot(state, model, 'full_sft')
    switch_sft_profile(state, 'continue')
    continued = snapshot(state, model, 'full_sft')
    assert first['args']['learning_rate'] == 1e-5
    assert first['args']['epochs'] == 2
    assert continued['args']['learning_rate'] == 3e-6
    assert continued['args']['epochs'] == 1


def test_streamlit_radio_changes_visible_values_and_restores_edits():
    # Exercise actual widget reruns: updating an already-rendered widget key
    # would raise StreamlitAPIException, so mode switching must occur first.
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('''
import streamlit as st
from scripts.sft_profiles import switch_sft_profile
mode = st.radio('SFT start mode', ['base', 'continue', 'resume'], key='sft_start_mode')
switch_sft_profile(st.session_state, mode)
st.number_input('Learning rate', min_value=1e-9, format='%.2e', key='learning_rate_full_sft')
st.number_input('Epochs', min_value=1, key='epochs_full_sft')
st.number_input('Warmup', min_value=0.0, max_value=0.2, key='warmup_ratio_full_sft')
''').run()
    assert not app.exception
    assert app.number_input[0].value == 1e-5
    app.number_input[0].set_value(8e-6).run()
    app.radio[0].set_value('continue').run()
    assert not app.exception
    assert app.number_input[0].value == 3e-6
    assert app.number_input[1].value == 1
    app.radio[0].set_value('base').run()
    assert not app.exception
    assert app.number_input[0].value == 8e-6
