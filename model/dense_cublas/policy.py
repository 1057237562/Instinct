"""Validated native Dense fusions; independent of experimental attention schedules."""


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
