"""Independent WebUI profiles for first SFT and completed-weight continuation."""

PROFILE_KEYS = (
    'learning_rate_full_sft', 'epochs_full_sft', 'warmup_ratio_full_sft',
    'data_path_full_sft', 'base_weight', 'max_seq_len', 'sequence_packing',
    'sequence_packing_mode', 'seq_bucket', 'bucket_max_seq_len',
    'bucket_large_threshold', 'batch_size', 'accumulation_steps', 'optimizer',
    'param_dtype', 'activation_dtype', 'fp8_training', 'fp8_filter',
    'use_compile', 'compile_mode', 'use_grad_checkpoint', 'data_cache_max_gb',
    'packing_batch_size', 'packing_num_proc', 'bucket_loader_workers',
    'moe_router_migration',
)
SFT_DEFAULTS = {
    'base': {
        'learning_rate_full_sft': 1e-5, 'epochs_full_sft': 2,
        'warmup_ratio_full_sft': 0.03,
    },
    'continue': {
        'learning_rate_full_sft': 3e-6, 'epochs_full_sft': 1,
        'warmup_ratio_full_sft': 0.05,
    },
}


def save_sft_profile(state):
    active = state.get('_sft_active_profile')
    if active in SFT_DEFAULTS:
        profiles = dict(state.get('sft_mode_profiles', {}))
        profiles[active] = {key: state[key] for key in PROFILE_KEYS if key in state}
        state['sft_mode_profiles'] = profiles


def track_sft_training_type(state, train_type):
    """Save SFT before another stage overwrites shared widgets; detect entry."""
    previous = state.get('_sft_last_train_type')
    if previous == 'full_sft' and train_type != previous:
        save_sft_profile(state)
    state['_sft_last_train_type'] = train_type
    return train_type == 'full_sft' and previous != train_type


def switch_sft_profile(state, mode, *, entering=False):
    """Call before dataset/weight/training widgets render on the radio rerun."""
    if mode not in ('base', 'continue', 'resume'):
        raise ValueError(f'Unknown SFT start mode: {mode}')
    if state.get('train_status') == 'running':
        return False
    previous = state.get('_sft_active_profile')
    if previous == mode and not entering:
        return False
    if not entering:
        save_sft_profile(state)
    if mode != 'resume':
        profiles = state.get('sft_mode_profiles', {})
        if mode in profiles:
            state.update({key: value for key, value in profiles[mode].items() if key in PROFILE_KEYS})
        else:
            # Retain runtime/VRAM choices, while establishing stage-specific
            # optimization defaults and a fresh base-weight selection.
            state.update(SFT_DEFAULTS[mode])
            state['base_weight'] = 'auto (newest matching base)'
            state['param_dtype'] = 'fp32'
            state['activation_dtype'] = 'bfloat16'
            state['moe_router_migration'] = False
            for key, value in {
                'max_seq_len': 4096, 'bucket_max_seq_len': 4096,
                'sequence_packing': True, 'sequence_packing_mode': 'bucket',
                'seq_bucket': 2, 'bucket_large_threshold': 4096,
                'batch_size': 1, 'accumulation_steps': 1,
                'optimizer': 'muon', 'compile_mode': 'default',
                'use_compile': True, 'use_grad_checkpoint': 1,
            }.items():
                state.setdefault(key, value)
    # Resume restores no new-stage defaults or profile values. The trainer
    # continues loading full model/optimizer/scaler/data cursor checkpoint state.
    state['from_resume'] = mode == 'resume'
    state['_sft_active_profile'] = mode
    return True
