"""Validated native Dense fusions and bounded packed-attention scheduling."""


def enable_fused_ops(backend):
    backend.pack_tile=32
    backend.compact_gradients=True
    backend.fused_norm_cast=True
    backend.fused_swiglu_amax=True
    backend.fused_quant_pack=True
    backend.terminal_replay=True
    backend.fused_ffn=True
    backend.selective_attention_min_seq=1
    backend.norm_groups=1024
    backend.elementwise_block=1024
    # Explicit probe overrides survive policy application. Normal trainers use
    # the production-validated settings without importing experiment modules.
    tune=bool(getattr(getattr(backend,'config',None),'tune',False))
    defaults=dict(fused_residual_norm=True,fused_norm_backward=True,fused_scale_inverse=True,
                  fused_residual_backward=True,fused_cast_amax=True,cache_weight_scales=True,
                  norm_rope_rows=8,ffn_flat_forward=True,ffn_quant_tile=64,
                  async_attention_metadata=True,attention_plan_quantum=512,
                  attention_plan_limit=64,attention_docs_per_plan=4,attention_local_maxima=True,
                  attention_group_penalty=4096,attention_tune=tune,attention_prewarm=tune,
                  muon_tiled_io=True,muon_symmetric=True,quant_tile=64,selective_aux_kv=True)
    for name,value in defaults.items():
        if not hasattr(backend,name):setattr(backend,name,value)
    if not hasattr(backend,'aligned_pointwise_only'):
        backend.aligned_pointwise_only=True
        backend.aligned_pointers=True
