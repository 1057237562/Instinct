"""PyTorch Dense activation checkpoint profiles, ordered by decreasing memory.

The legacy use_grad_checkpoint=0/1/2 contract is intentionally unchanged.
New profiles use non-reentrant SAC and never cache quadratic math-attention
matrices. Profile costs must be measured for the actual shape/backend.
"""
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint


CHECKPOINT_PROFILES = ("legacy", "off", "ckpt1", "ckpt2", "ckpt3", "ckpt4")


def validate_checkpoint_profile(profile, *, supported=True):
    profile = profile or "legacy"
    if profile not in CHECKPOINT_PROFILES:
        raise ValueError(f"Unknown checkpoint_profile {profile!r}; choose {CHECKPOINT_PROFILES}")
    if profile.startswith("ckpt") and not supported:
        raise ValueError("ckpt1–ckpt4 currently support PyTorch Dense only; use checkpoint_profile=legacy for this architecture")
    return profile


def selective_policy(profile, hidden_size, ctx, op, *args, **kwargs):
    from torch.utils.checkpoint import CheckpointPolicy

    # SDPA backends expose output + logsumexp, not the quadratic attention map.
    fused_attention = op in {
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
    }
    save = fused_attention and profile != "ckpt4"
    if op == torch.ops.aten.mm.default:
        # Non-reentrant checkpoint stops before the final FFN down projection;
        # caching its output buys no GEMM FLOPs. Keep Q/K/V/O in ckpt2, and
        # additionally the large gate/up projections in ckpt1.
        reduction, width = args[0].shape[-1], args[1].shape[-1]
        save = reduction <= hidden_size and (
            profile == "ckpt1" or (profile == "ckpt2" and width <= hidden_size))
    return CheckpointPolicy.MUST_SAVE if save else CheckpointPolicy.PREFER_RECOMPUTE


def checkpoint_layer(layer, hidden_states, position_embeddings, attention_mask, profile):
    options = {}
    if profile != "ckpt4":
        try:
            from torch.utils.checkpoint import create_selective_checkpoint_contexts
        except ImportError as exc:
            raise RuntimeError("ckpt1–ckpt3 require PyTorch selective checkpointing (tested with 2.8)") from exc
        options["context_fn"] = partial(
            create_selective_checkpoint_contexts,
            partial(selective_policy, profile, layer.input_layernorm.weight.numel()),
        )
    return checkpoint(
        layer, hidden_states, position_embeddings, None, False, attention_mask,
        use_reentrant=False, preserve_rng_state=True, **options,
    )
