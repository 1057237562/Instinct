"""Conservative V1 MoE CPT starting point, not an empirically optimal recipe."""

CPT_DEFAULTS_VERSION = 4
CPT_DEFAULTS = {
    'epochs_cpt': 1,
    'learning_rate_cpt': 3e-5,
    'warmup_ratio_cpt': 0.03,
    'max_seq_len': 4096,
    'sequence_packing': True,
    'sequence_packing_mode': 'bucket',
    'seq_bucket': 2,
    'bucket_max_seq_len': 4096,
    'bucket_large_threshold': 4096,
    'batch_size': 1,  # bucket mode derives the real batch from the VRAM budget
    'accumulation_steps': 1,
    'optimizer': 'muon',
    'param_dtype': 'fp32',
    'activation_dtype': 'bfloat16',
    'fp8_training': 'tensorwise',
    'fp8_filter': 'auto',
    'use_grad_checkpoint': 1,
    'use_compile': True,
    'compile_mode': 'default',
    'data_cache_max_gb': 20.0,
    'dataset_streaming': 'on',
    'streaming_chunk_mb': 1024,
    'streaming_prefetch_chunks': True,
    'cache_build_mode': 'inline',
    'bucket_loader_workers': 0,
}


def apply_cpt_defaults(state, *, force=False):
    """Upgrade once before widgets render; never alter an active/resumed run."""
    if state.get('from_resume', False) or state.get('train_status') in ('running', 'paused'):
        return False
    if not force and int(state.get('cpt_defaults_version', 0)) >= CPT_DEFAULTS_VERSION:
        return False
    state.update(CPT_DEFAULTS)
    # Keep the user's device budget: local 16GB and remote 32GB are both used.
    state.setdefault('bucket_gpu_memory_gb', 15.5)
    state['moe_router_migration'] = bool(state.get('use_moe', False))
    state['cpt_defaults_initialized'] = True
    state['cpt_defaults_version'] = CPT_DEFAULTS_VERSION
    return True
