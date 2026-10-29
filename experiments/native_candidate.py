"""Explicit, recorded backend overrides for bounded native experiments."""
import json
from pathlib import Path


ALLOWED={
    'attention_tune','attention_tune_workspace_mib','attention_plan_quantum','attention_plan_limit',
    'attention_docs_per_plan','attention_group_penalty','attention_local_maxima',
    'ffn_statistics_block','ffn_quant_tile','ffn_quant_col_tile','ffn_col_major_grid',
    'ffn_materialize_forward','ffn_materialize_backward','norm_groups','norm_rope_rows',
    'muon_fused_prepare','muon_tiled_io','muon_symmetric','quant_tile','kernel_options','aligned_norm_backward',
    'fused_input_norm_amax','norm_forward_rows',
    'fp8_fast_accum_large','bounded_tuning','attention_prewarm','selective_aux_kv',
    'fused_activation_layouts','attention_cost_model',
    'attention_frost_backward',
    'muon_blocked_gram',
    'fused_boundary_norm',
    'release_checkpoint_cache',
    'muon_gram_tile',
    'muon_blocked_polynomial',
    'ffn_backward_stage',
}


def apply(backend,path):
    if not path:return {}
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    unknown=config.keys()-ALLOWED
    if unknown:raise ValueError(f'unknown candidate settings: {sorted(unknown)}')
    for key,value in config.items():
        if key=='kernel_options':backend.kernel_options.update(value)
        else:setattr(backend,key,value)
    return config
