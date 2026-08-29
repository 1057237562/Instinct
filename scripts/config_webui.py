"""
Instinct Config WebUI
=======================
Interactive configuration tool for Instinct model parameters.
Streamlit-based UI with real-time parameter count estimation
and architecture visualization. Works standalone — no model
weights required.

Run from the scripts/ directory:
    streamlit run config_webui.py
"""

import streamlit as st
import math
import json
import os
import re
import sys
import subprocess
import time

# ═══════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Instinct Config", layout="wide")

# ═══════════════════════════════════════════════════════════════
# Preset definitions
# ═══════════════════════════════════════════════════════════════
PRESETS = {
    "instinct-3": {
        "hidden_size": 768,
        "num_hidden_layers": 8,
        "vocab_size": 6400,
        "dropout": 0.0,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 32768,
        "rope_theta": 1e6,
        "inference_rope_scaling": False,
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 2048,
        "attention_factor": 1.0,
        "use_moe": False,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "norm_topk_prob": True,
        "router_aux_loss_coef": 5e-4,
        "rms_norm_eps": 1e-6,
        "flash_attn": True,
        "model_architecture": "standard",
    },
    "instinct-3-moe": {
        "hidden_size": 768,
        "num_hidden_layers": 8,
        "vocab_size": 6400,
        "dropout": 0.0,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 32768,
        "rope_theta": 1e6,
        "inference_rope_scaling": False,
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 2048,
        "attention_factor": 1.0,
        "use_moe": True,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "norm_topk_prob": True,
        "router_aux_loss_coef": 5e-4,
        "rms_norm_eps": 1e-6,
        "flash_attn": True,
        "model_architecture": "standard",
    },
    "instinct2-small": {
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "vocab_size": 6400,
        "dropout": 0.0,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32768,
        "rope_theta": 1e6,
        "inference_rope_scaling": False,
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 2048,
        "attention_factor": 1.0,
        "use_moe": False,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "norm_topk_prob": True,
        "router_aux_loss_coef": 5e-4,
        "rms_norm_eps": 1e-6,
        "flash_attn": True,
        "model_architecture": "standard",
    },
    "instinct2": {
        "hidden_size": 768,
        "num_hidden_layers": 16,
        "vocab_size": 6400,
        "dropout": 0.0,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32768,
        "rope_theta": 1e6,
        "inference_rope_scaling": False,
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 2048,
        "attention_factor": 1.0,
        "use_moe": False,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "norm_topk_prob": True,
        "router_aux_loss_coef": 5e-4,
        "rms_norm_eps": 1e-6,
        "flash_attn": True,
        "model_architecture": "standard",
    },
    "instinct-linear": {
        "hidden_size": 768,
        "num_hidden_layers": 8,
        "vocab_size": 6400,
        "dropout": 0.0,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 32768,
        "rope_theta": 1e6,
        "inference_rope_scaling": False,
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 2048,
        "attention_factor": 1.0,
        "use_moe": False,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "norm_topk_prob": True,
        "router_aux_loss_coef": 5e-4,
        "rms_norm_eps": 1e-6,
        "flash_attn": True,
        "model_architecture": "linear",
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 96,
        "linear_value_head_dim": 96,
        "linear_num_key_heads": 8,
        "linear_num_value_heads": 8,
    },
}

# ═══════════════════════════════════════════════════════════════
# CSS — dark theme, clean modern look
# ═══════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
    [data-testid="stAppDeployButton"] { display: none; }

    /* Architecture diagram blocks */
    .arch-wrap {
        display: flex; flex-direction: column; align-items: center;
        width: 100%;
    }
    .arch-block {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center;
        padding: 10px 20px; border-radius: 10px; margin: 3px 0;
        font-family: 'Courier New', monospace; font-size: 14px;
        font-weight: 700; letter-spacing: 0.5px;
        border: 1px solid rgba(255,255,255,0.08);
        width: 100%; max-width: 360px;
        transition: transform 0.15s;
    }
    .arch-block:hover { transform: translateX(4px); }
    .arch-arrow {
        color: #64748b; text-align: center; font-size: 16px;
        line-height: 1; margin: 1px 0; font-weight: 300;
    }
    .arch-label {
        font-size: 10px; opacity: 0.65; font-weight: 400;
        margin-top: 2px; font-family: -apple-system, sans-serif;
        letter-spacing: 0.3px;
    }

    /* Parameter cards */
    .param-card {
        background: linear-gradient(135deg, #1e293b 0%, #1a2234 100%);
        border: 1px solid #2d3a4e; border-radius: 10px;
        padding: 12px 16px; margin: 6px 0;
    }
    .param-label {
        font-size: 11px; color: #94a3b8; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.6px;
    }
    .param-value {
        font-size: 20px; font-weight: 700;
        font-family: 'Courier New', monospace;
        color: #f1f5f9; margin-top: 2px;
    }

    /* Badge row */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        color: white; padding: 4px 14px; border-radius: 20px;
        font-size: 12px; font-weight: 600; margin: 2px 4px;
        white-space: nowrap; font-family: 'Courier New', monospace;
    }
    .badge-moe {
        background: linear-gradient(135deg, #5b21b6 0%, #7c3aed 100%);
    }
    .badge-green {
        background: linear-gradient(135deg, #065f46 0%, #10b981 100%);
    }

    /* Section header */
    .section-title {
        font-size: 11px; font-weight: 600; letter-spacing: 1px;
        color: #64748b; text-transform: uppercase; margin: 20px 0 8px 0;
        border-bottom: 1px solid #1e293b; padding-bottom: 6px;
    }

    /* Streamlit dark overrides */
    .stApp { background: #0f172a; }
    .st-emotion-cache-1kyxreq { color: #e2e8f0; }
    .st-emotion-cache-1r4qj8v { background: #1a2332; }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 14px; font-weight: 600;
    }
    .st-emotion-cache-1gulkj5 { color: #94a3b8; }
    .stDataFrame { font-size: 13px; }
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════


def fmt_num(n: int) -> str:
    """Format integer with thousands separators."""
    return f"{n:,}"


def compute_intermediate_size(hidden_size: int) -> int:
    """Compute the FFN intermediate size the same way InstinctConfig does."""
    return math.ceil(hidden_size * math.pi / 64) * 64


def compute_head_dim(hidden_size: int, num_attention_heads: int) -> int:
    """Auto-compute head dimension."""
    if num_attention_heads <= 0:
        return 0
    return hidden_size // num_attention_heads


def calc_params(config: dict) -> dict:
    """Calculate full parameter breakdown from a config dict.

    All formulas are implemented in pure Python — no model imports.
    """
    h = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    q_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", compute_head_dim(h, q_heads))
    int_size = config.get("intermediate_size", compute_intermediate_size(h))
    tie_word = config.get("tie_word_embeddings", False)
    use_moe = config.get("use_moe", False)
    num_experts = config.get("num_experts", 4)
    num_experts_per_tok = config.get("num_experts_per_tok", 1)
    moe_int_size = config.get("moe_intermediate_size", int_size)
    arch = config.get("model_architecture", "standard")
    residual_type = config.get("residual_type", "standard")

    # ---- Embedding ----
    embedding = vocab * h

    # ---- Per-layer Full Attention ----
    q_proj = h * (q_heads * head_dim)
    k_proj = h * (kv_heads * head_dim)
    v_proj = h * (kv_heads * head_dim)
    o_proj = (q_heads * head_dim) * h
    q_norm = head_dim
    k_norm = head_dim
    full_attn_per_layer = q_proj + k_proj + v_proj + o_proj + q_norm + k_norm

    q_proj_str = f"{h} x ({q_heads} x {head_dim})"
    k_proj_str = f"{h} x ({kv_heads} x {head_dim})"
    v_proj_str = f"{h} x ({kv_heads} x {head_dim})"
    o_proj_str = f"({q_heads} x {head_dim}) x {h}"

    # ---- GatedDeltaNet (Linear) Attention params ----
    if arch == "linear":
        lkhd = config.get("linear_key_head_dim", head_dim)
        lvhd = config.get("linear_value_head_dim", head_dim)
        lnkh = config.get("linear_num_key_heads", q_heads)
        lnvh = config.get("linear_num_value_heads", q_heads)
        lkd = config.get("linear_conv_kernel_dim", 4)

        key_dim = lkhd * lnkh
        value_dim = lvhd * lnvh
        conv_dim = key_dim * 2 + value_dim

        conv1d_params = conv_dim * lkd
        dt_bias_A_log = lnvh * 2
        norm = lvhd
        out_proj = value_dim * h
        in_proj_qkv = h * (key_dim * 2 + value_dim)
        in_proj_z = h * value_dim
        in_proj_b = h * lnvh
        in_proj_a = h * lnvh

        linear_attn_per_layer = (
            conv1d_params + dt_bias_A_log + norm + out_proj
            + in_proj_qkv + in_proj_z + in_proj_b + in_proj_a
        )

        lin_attn_formula = (
            f"conv1d={conv_dim}x{lkd}, dt_bias+A_log=2x{lnvh}, "
            f"norm={lvhd}, out_proj={value_dim}x{h}, "
            f"in_proj_qkv={h}x({key_dim}x2+{value_dim}), "
            f"in_proj_z={h}x{value_dim}, "
            f"in_proj_b={h}x{lnvh}, in_proj_a={h}x{lnvh}"
        )

        interval = config.get("full_attention_interval", 4)
        num_full = n_layers // interval if interval > 0 else n_layers
        num_linear = n_layers - num_full
    else:
        linear_attn_per_layer = 0
        lin_attn_formula = ""
        num_full = n_layers
        num_linear = 0

    # ---- Per-layer FFN ----
    if use_moe:
        gate_proj = h * moe_int_size
        up_proj = h * moe_int_size
        down_proj = moe_int_size * h
        per_expert = gate_proj + up_proj + down_proj
        router = h * num_experts
        ffn_per_layer = router + num_experts * per_expert
        active_ffn_per_layer = num_experts_per_tok * per_expert
        ffn_detail = (
            f"router={h}x{num_experts} + "
            f"{num_experts} experts x ({h}x{moe_int_size} + "
            f"{h}x{moe_int_size} + {moe_int_size}x{h})"
        )
        active_ffn_detail = (
            f"{num_experts_per_tok} active x ({h}x{moe_int_size} + "
            f"{h}x{moe_int_size} + {moe_int_size}x{h})"
        )
    else:
        gate_proj = h * int_size
        up_proj = h * int_size
        down_proj = int_size * h
        ffn_per_layer = gate_proj + up_proj + down_proj
        active_ffn_per_layer = ffn_per_layer
        ffn_detail = f"gate={h}x{int_size}, up={h}x{int_size}, down={int_size}x{h}"
        active_ffn_detail = ffn_detail

    # ---- Norms per layer ----
    input_layernorm = h
    post_attention_layernorm = h
    norms_per_layer = input_layernorm + post_attention_layernorm

    # ---- Layer totals (per type) ----
    full_layer_total = full_attn_per_layer + ffn_per_layer + norms_per_layer
    full_active_layer_total = full_attn_per_layer + active_ffn_per_layer + norms_per_layer
    if arch == "linear" and num_linear > 0:
        linear_layer_total = linear_attn_per_layer + ffn_per_layer + norms_per_layer
        linear_active_layer_total = linear_attn_per_layer + active_ffn_per_layer + norms_per_layer
    else:
        linear_layer_total = 0
        linear_active_layer_total = 0

    # ---- Final norm ----
    final_norm = h

    # ---- LM Head ----
    lm_head = 0 if tie_word else vocab * h

    # ---- Looped (LoopUS) modules: gate + q_head ----
    looped_gate = 0
    looped_q_head = 0
    if arch == "looped":
        dt_rank = max(1, h // 16)
        looped_gate = (h * dt_rank) + (dt_rank * h + h) + h  # dt_input_proj + delta_proj(+bias) + A_log
        looped_q_head = 2 * h + (h * 1 + 1)  # LayerNorm + Linear(h,1)(+bias)

    # ---- Residual topology ----
    residual_params = 0
    residual_formula = ""
    residual_layers = n_layers
    if arch == "looped":
        residual_layers = (
            config.get("prelude_layers", 1)
            + (1 if config.get("loop_iters", 8) > 0 else 0)
            + config.get("coda_layers", 1)
        )
    if residual_type == "mhc":
        hc = config.get("hc_mult", 4)
        mix = (2 + hc) * hc
        per_connector = mix * (hc * h) + mix + 3
        hyper_head = hc * (hc * h) + hc + 1
        residual_params = 2 * residual_layers * per_connector + hyper_head
        residual_formula = (
            f"2x{residual_layers} connectors x [({mix}x{hc * h})+{mix}+3] "
            f"+ head [({hc}x{hc * h})+{hc}+1]"
        )
    elif residual_type == "attnres":
        residual_params = (2 * residual_layers + 1) * h
        variant = config.get("attnres_variant", "block")
        residual_formula = f"(2x{residual_layers}+1) pseudo-queries x {h} ({variant})"

    # ---- Grand totals ----
    total = (
        embedding
        + num_full * full_layer_total
        + num_linear * linear_layer_total
        + final_norm + lm_head
        + looped_gate + looped_q_head
        + residual_params
    )
    total_active = (
        embedding
        + num_full * full_active_layer_total
        + num_linear * linear_active_layer_total
        + final_norm + lm_head
        + looped_gate + looped_q_head
        + residual_params
    )

    if arch == "linear":
        breakdown = {
            "Embedding": {"value": embedding, "formula": f"{vocab} x {h}"},
            "Full Attention (per layer)": {
                "value": full_attn_per_layer,
                "formula": (
                    f"q_proj={q_proj_str}, k_proj={k_proj_str}, "
                    f"v_proj={v_proj_str}, o_proj={o_proj_str}, "
                    f"q_norm+k_norm=2x{head_dim}"
                ),
            },
            "Linear Attention (per layer)": {
                "value": linear_attn_per_layer,
                "formula": lin_attn_formula,
            },
            "Per-Layer FFN": {
                "value": ffn_per_layer,
                "formula": ffn_detail,
            },
            "Per-Layer Norms": {
                "value": norms_per_layer,
                "formula": f"input_layernorm+post_attention_layernorm = 2x{h}",
            },
            "Layer Distribution": {
                "value": 0,
                "formula": f"{num_full} full attention + {num_linear} linear attention",
            },
            "Final Norm": {
                "value": final_norm,
                "formula": f"RMSNorm({h})",
            },
            "LM Head": {
                "value": lm_head,
                "formula": "0 (tied)" if tie_word else f"{vocab} x {h}",
            },
            "Total Params": {
                "value": total,
                "formula": "",
            },
        }

    else:
        breakdown = {
            "Embedding": {"value": embedding, "formula": f"{vocab} x {h}"},
            "Per-Layer Attention": {
                "value": full_attn_per_layer,
                "formula": (
                    f"q_proj={q_proj_str}, k_proj={k_proj_str}, "
                    f"v_proj={v_proj_str}, o_proj={o_proj_str}, "
                    f"q_norm+k_norm=2x{head_dim}"
                ),
            },
            "Per-Layer FFN": {
                "value": ffn_per_layer,
                "formula": ffn_detail,
            },
            "Per-Layer Norms": {
                "value": norms_per_layer,
                "formula": f"input_layernorm+post_attention_layernorm = 2x{h}",
            },
            f"All {n_layers} Layers": {
                "value": n_layers * full_layer_total,
                "formula": f"{n_layers} x (attn + ffn + norms)",
            },
            "Final Norm": {
                "value": final_norm,
                "formula": f"RMSNorm({h})",
            },
            "LM Head": {
                "value": lm_head,
                "formula": "0 (tied)" if tie_word else f"{vocab} x {h}",
            },
            "Total Params": {
                "value": total,
                "formula": "",
            },
        }

    if residual_params:
        total_row = breakdown.pop("Total Params")
        breakdown[f"Residual ({residual_type})"] = {
            "value": residual_params,
            "formula": residual_formula,
        }
        breakdown["Total Params"] = total_row

    if arch == "looped":
        breakdown["SelectiveGate"] = {
            "value": looped_gate,
            "formula": (
                f"dt_input_proj={h}x{max(1, h // 16)}, "
                f"delta_proj={max(1, h // 16)}x{h}+{h}, A_log={h}"
            ),
        }
        breakdown["Q-Head"] = {
            "value": looped_q_head,
            "formula": f"LayerNorm(2x{h}) + Linear({h},1)+1",
        }

    if use_moe:
        if arch == "linear":
            breakdown["Per-Layer Active FFN"] = {
                "value": active_ffn_per_layer,
                "formula": active_ffn_detail,
            }
        else:
            breakdown["Per-Layer Active FFN"] = {
                "value": active_ffn_per_layer,
                "formula": active_ffn_detail,
            }
        breakdown["Active Total"] = {
            "value": total_active,
            "formula": "",
        }

    return breakdown


def fmt_table(breakdown: dict) -> list:
    """Convert breakdown dict to list-of-dicts for st.dataframe."""
    rows = []
    for name, info in breakdown.items():
        rows.append(
            {
                "Component": name,
                "Parameters": fmt_num(info["value"]),
                "Computation": info["formula"],
            }
        )
    return rows


def build_config_dict() -> dict:
    """Assemble a InstinctConfig-compatible dict from st.session_state."""
    d = {
        "hidden_size": st.session_state.get("hidden_size", 768),
        "num_hidden_layers": st.session_state.get("num_hidden_layers", 8),
        "vocab_size": st.session_state.get("vocab_size", 6400),
        "dropout": st.session_state.get("dropout", 0.0),
        "hidden_act": st.session_state.get("hidden_act", "silu"),
        "tie_word_embeddings": st.session_state.get("tie_word_embeddings", False),
        "num_attention_heads": st.session_state.get("num_attention_heads", 8),
        "num_key_value_heads": st.session_state.get("num_key_value_heads", 4),
        "max_position_embeddings": st.session_state.get("max_position_embeddings", 32768),
        "rope_theta": st.session_state.get("rope_theta", 1e6),
        "inference_rope_scaling": st.session_state.get("inference_rope_scaling", False),
        "use_moe": st.session_state.get("use_moe", False),
        "use_grad_checkpoint": st.session_state.get("use_grad_checkpoint", 0),
        "rms_norm_eps": st.session_state.get("rms_norm_eps", 1e-6),
        "flash_attn": st.session_state.get("flash_attn", True),
        "intermediate_size": compute_intermediate_size(st.session_state.get("hidden_size", 768)),
        "residual_type": st.session_state.get("residual_type", "standard"),
    }

    # head_dim
    h = d["hidden_size"]
    q_heads = d["num_attention_heads"]
    d["head_dim"] = compute_head_dim(h, q_heads)

    # YaRN rope_scaling
    if d.get("inference_rope_scaling"):
        d["rope_scaling"] = {
            "type": "yarn",
            "beta_fast": st.session_state.get("beta_fast", 32),
            "beta_slow": st.session_state.get("beta_slow", 1),
            "factor": st.session_state.get("factor", 16),
            "original_max_position_embeddings": st.session_state.get(
                "original_max_position_embeddings", 2048
            ),
            "attention_factor": st.session_state.get("attention_factor", 1.0),
        }

    # Model architecture
    d["model_architecture"] = st.session_state.get("model_architecture", "standard")

    # Early Exit
    if st.session_state.get("early_exit_enabled", False):
        d["early_exit_layers"] = st.session_state.get("early_exit_layers", [4, 5, 6, 7])
        d["early_exit_loss_weight"] = st.session_state.get("early_exit_loss_weight", 0.3)

    # Linear arch params
    if d["model_architecture"] == "linear":
        d["full_attention_interval"] = st.session_state.get("full_attention_interval", 4)
        d["linear_conv_kernel_dim"] = st.session_state.get("linear_conv_kernel_dim", 4)
        d["linear_key_head_dim"] = st.session_state.get("linear_key_head_dim", compute_head_dim(h, q_heads))
        d["linear_value_head_dim"] = st.session_state.get("linear_value_head_dim", compute_head_dim(h, q_heads))
        d["linear_num_key_heads"] = st.session_state.get("linear_num_key_heads", q_heads)
        d["linear_num_value_heads"] = st.session_state.get("linear_num_value_heads", q_heads)

    # Looped (LoopUS) arch params
    if d["model_architecture"] == "looped":
        d["loop_max_steps"] = st.session_state.get("loop_max_steps", 32)
        d["q_threshold"] = st.session_state.get("loop_q_threshold", 0.9)
        d["n_supervision"] = st.session_state.get("loop_n_supervision", 6)
        d["depth_reward"] = st.session_state.get("loop_depth_reward", 0.01)
        d["exit_in_training"] = st.session_state.get("exit_in_training", True)
        d["beta"] = st.session_state.get("loop_beta", 0.5)
        d["distill_weight"] = st.session_state.get("loop_distill_weight", 0.0)
        d["distill_temperature"] = st.session_state.get("loop_distill_temperature", 2.0)
        d["teacher_stop_grad"] = st.session_state.get("teacher_stop_grad", True)
        d["depth_gain_reward"] = st.session_state.get("loop_depth_gain_reward", 0.0)
        d["loop_encoder_layers"] = st.session_state.get("loop_encoder_layers", [0, 1])
        d["loop_body_layers"] = st.session_state.get("loop_body_layers", [2, 3, 4])
        d["loop_output_layers"] = st.session_state.get("loop_output_layers", [5, 6, 7])

    if d["residual_type"] == "mhc":
        d["hc_mult"] = st.session_state.get("hc_mult", 4)
        d["hc_sinkhorn_iters"] = st.session_state.get("hc_sinkhorn_iters", 20)
        d["hc_eps"] = st.session_state.get("hc_eps", 1e-6)
    elif d["residual_type"] == "attnres":
        d["attnres_variant"] = st.session_state.get("attnres_variant", "block")
        d["attnres_block_size"] = st.session_state.get(
            "attnres_block_size", max(1, math.ceil(2 * d["num_hidden_layers"] / 8))
        )

    # MoE
    if d["use_moe"]:
        d["num_experts"] = st.session_state.get("num_experts", 4)
        d["num_experts_per_tok"] = st.session_state.get("num_experts_per_tok", 1)
        d["moe_intermediate_size"] = st.session_state.get(
            "moe_intermediate_size",
            compute_intermediate_size(h),
        )
        d["norm_topk_prob"] = st.session_state.get("norm_topk_prob", True)
        d["router_aux_loss_coef"] = st.session_state.get("router_aux_loss_coef", 5e-4)

    return d


def gen_python_code(cfg: dict) -> str:
    """Generate InstinctConfig instantiation code."""
    is_linear = cfg.get("model_architecture") == "linear"
    is_looped = cfg.get("model_architecture") == "looped"
    if is_looped:
        module = "model.model_instinct_loop"
        cls = "InstinctConfig"
    else:
        module = "model.model_instinct_linear" if is_linear else "model.model_instinct"
        cls = "InstinctConfig"
    lines = [f"from {module} import {cls}", "", f"config = {cls}("]
    params = [
        ("hidden_size", cfg["hidden_size"]),
        ("num_hidden_layers", cfg["num_hidden_layers"]),
        ("vocab_size", cfg["vocab_size"]),
        ("dropout", cfg["dropout"]),
        ("hidden_act", f'"{cfg["hidden_act"]}"'),
        ("tie_word_embeddings", str(cfg["tie_word_embeddings"])),
        ("num_attention_heads", cfg["num_attention_heads"]),
        ("num_key_value_heads", cfg["num_key_value_heads"]),
        ("max_position_embeddings", cfg["max_position_embeddings"]),
        ("rope_theta", f'{cfg["rope_theta"]:.0f}'),
        ("rms_norm_eps", f'{cfg["rms_norm_eps"]}'),
        ("flash_attn", str(cfg["flash_attn"])),
        ("residual_type", f'"{cfg.get("residual_type", "standard")}"'),
    ]

    if cfg["use_moe"]:
        params.append(("use_moe", "True"))
        params.append(("num_experts", cfg["num_experts"]))
        params.append(("num_experts_per_tok", cfg["num_experts_per_tok"]))
        params.append(("norm_topk_prob", str(cfg["norm_topk_prob"])))
        params.append(("router_aux_loss_coef", f'{cfg["router_aux_loss_coef"]}'))
    else:
        params.append(("use_moe", "False"))

    if cfg.get("use_grad_checkpoint"):
        params.append(("use_grad_checkpoint", cfg.get("use_grad_checkpoint")))

    if cfg.get("residual_type") == "mhc":
        params.append(("hc_mult", cfg.get("hc_mult", 4)))
        params.append(("hc_sinkhorn_iters", cfg.get("hc_sinkhorn_iters", 20)))
        params.append(("hc_eps", cfg.get("hc_eps", 1e-6)))
    elif cfg.get("residual_type") == "attnres":
        params.append(("attnres_variant", f'"{cfg.get("attnres_variant", "block")}"'))
        params.append(("attnres_block_size", cfg.get("attnres_block_size", 2)))

    if cfg["inference_rope_scaling"]:
        params.append(("inference_rope_scaling", "True"))

    if cfg.get("model_architecture") == "linear":
        params.append(("full_attention_interval", cfg.get("full_attention_interval", 4)))
        params.append(("linear_conv_kernel_dim", cfg.get("linear_conv_kernel_dim", 4)))
        params.append(("linear_key_head_dim", cfg.get("linear_key_head_dim", cfg["head_dim"])))
        params.append(("linear_value_head_dim", cfg.get("linear_value_head_dim", cfg["head_dim"])))
        params.append(("linear_num_key_heads", cfg.get("linear_num_key_heads", cfg["num_attention_heads"])))
        params.append(("linear_num_value_heads", cfg.get("linear_num_value_heads", cfg["num_attention_heads"])))

    if cfg.get("model_architecture") == "looped":
        params.append(("loop_max_steps", cfg.get("loop_max_steps", 32)))
        params.append(("q_threshold", cfg.get("q_threshold", 0.9)))
        params.append(("n_supervision", cfg.get("n_supervision", 6)))
        params.append(("depth_reward", cfg.get("depth_reward", 0.01)))
        params.append(("exit_in_training", str(cfg.get("exit_in_training", True))))
        params.append(("beta", cfg.get("beta", 0.5)))
        if cfg.get("distill_weight"):
            params.append(("distill_weight", cfg.get("distill_weight", 0.0)))
        if cfg.get("distill_temperature") != 2.0:
            params.append(("distill_temperature", cfg.get("distill_temperature", 2.0)))
        if not cfg.get("teacher_stop_grad", True):
            params.append(("teacher_stop_grad", str(cfg.get("teacher_stop_grad", True))))
        if cfg.get("depth_gain_reward"):
            params.append(("depth_gain_reward", cfg.get("depth_gain_reward", 0.0)))
        params.append(("loop_encoder_layers", str(cfg.get("loop_encoder_layers", [0, 1]))))
        params.append(("loop_body_layers", str(cfg.get("loop_body_layers", [2, 3, 4]))))
        params.append(("loop_output_layers", str(cfg.get("loop_output_layers", [5, 6, 7]))))

    if cfg.get("early_exit_layers") and cfg.get("early_exit_layers") != [4, 5, 6, 7]:
        params.append(("early_exit_layers", str(cfg["early_exit_layers"])))
    if cfg.get("early_exit_loss_weight") and cfg["early_exit_loss_weight"] != 0.3:
        params.append(("early_exit_loss_weight", cfg["early_exit_loss_weight"]))

    for i, (k, v) in enumerate(params):
        comma = "," if i < len(params) - 1 else ""
        lines.append(f"    {k}={v}{comma}")

    lines.append(")")
    return "\n".join(lines)


def gen_config_json(cfg: dict) -> str:
    """Generate a clean config.json dict."""
    out = {}
    keys = [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "hidden_act",
        "dropout",
        "tie_word_embeddings",
        "rms_norm_eps",
        "flash_attn",
        "use_moe",
        "use_grad_checkpoint",
        "residual_type",
    ]
    for k in keys:
        if k == "rope_theta":
            out[k] = int(cfg[k])
        else:
            out[k] = cfg.get(k)

    if cfg.get("inference_rope_scaling") and cfg.get("rope_scaling"):
        out["inference_rope_scaling"] = True
        out["rope_scaling"] = cfg["rope_scaling"]
    else:
        out["inference_rope_scaling"] = False

    out["model_architecture"] = cfg.get("model_architecture", "standard")

    if cfg.get("residual_type") == "mhc":
        for k in ["hc_mult", "hc_sinkhorn_iters", "hc_eps"]:
            out[k] = cfg.get(k)
    elif cfg.get("residual_type") == "attnres":
        for k in ["attnres_variant", "attnres_block_size"]:
            out[k] = cfg.get(k)

    if cfg.get("use_moe"):
        for k in ["num_experts", "num_experts_per_tok", "moe_intermediate_size",
                   "norm_topk_prob", "router_aux_loss_coef"]:
            out[k] = cfg.get(k)

    if cfg.get("model_architecture") == "linear":
        for k in ["full_attention_interval", "linear_conv_kernel_dim",
                   "linear_key_head_dim", "linear_value_head_dim",
                   "linear_num_key_heads", "linear_num_value_heads"]:
            out[k] = cfg.get(k)

    if cfg.get("model_architecture") == "looped":
        for k in ["loop_max_steps", "q_threshold", "n_supervision",
                   "depth_reward", "exit_in_training", "beta",
                   "distill_weight", "distill_temperature", "teacher_stop_grad",
                   "depth_gain_reward",
                   "loop_encoder_layers", "loop_body_layers", "loop_output_layers"]:
            out[k] = cfg.get(k)

    if cfg.get("early_exit_layers"):
        out["early_exit_layers"] = cfg["early_exit_layers"]
    if cfg.get("early_exit_loss_weight"):
        out["early_exit_loss_weight"] = cfg["early_exit_loss_weight"]

    return json.dumps(out, indent=2)


def load_config_to_session(config_dict: dict):
    """Populate st.session_state from a config dict (reverse of build_config_dict).
    Detects matching preset name; falls back to 'Custom'."""
    # Detect matching preset
    matched = "Custom"
    for name, preset in PRESETS.items():
        match = True
        for k, v in preset.items():
            if config_dict.get(k) != v:
                match = False
                break
        if match:
            matched = name
            break

    st.session_state.preset = matched
    if matched != "Custom":
        init_from_preset(matched)
        return

    # Manual populate for custom configs
    st.session_state.hidden_size = config_dict.get("hidden_size", 768)
    st.session_state.num_hidden_layers = config_dict.get("num_hidden_layers", 8)
    st.session_state.vocab_size = config_dict.get("vocab_size", 6400)
    st.session_state.dropout = config_dict.get("dropout", 0.0)
    st.session_state.hidden_act = config_dict.get("hidden_act", "silu")
    st.session_state.tie_word_embeddings = config_dict.get("tie_word_embeddings", False)
    st.session_state.num_attention_heads = config_dict.get("num_attention_heads", 8)
    st.session_state.num_key_value_heads = config_dict.get("num_key_value_heads", 4)
    st.session_state.max_position_embeddings = config_dict.get("max_position_embeddings", 32768)
    st.session_state.rope_theta = float(config_dict.get("rope_theta", 1e6))
    st.session_state.inference_rope_scaling = config_dict.get("inference_rope_scaling", False)
    st.session_state.use_moe = config_dict.get("use_moe", False)
    st.session_state.use_grad_checkpoint = config_dict.get("use_grad_checkpoint", 0)
    st.session_state.rms_norm_eps = config_dict.get("rms_norm_eps", 1e-6)
    st.session_state.flash_attn = config_dict.get("flash_attn", True)
    st.session_state.residual_type = config_dict.get("residual_type", "standard")
    st.session_state.pop("_residual_radio", None)
    st.session_state.hc_mult = config_dict.get("hc_mult", 4)
    st.session_state.hc_sinkhorn_iters = config_dict.get("hc_sinkhorn_iters", 20)
    st.session_state.hc_eps = config_dict.get("hc_eps", 1e-6)
    st.session_state.attnres_variant = config_dict.get("attnres_variant", "block")
    st.session_state.attnres_block_size = config_dict.get(
        "attnres_block_size", max(1, math.ceil(2 * st.session_state.num_hidden_layers / 8))
    )

    # Model architecture
    arch = config_dict.get("model_architecture", "standard")
    st.session_state.model_architecture = arch

    # YaRN params
    rs = config_dict.get("rope_scaling") or {}
    if config_dict.get("inference_rope_scaling") and rs:
        st.session_state.beta_fast = rs.get("beta_fast", 32)
        st.session_state.beta_slow = rs.get("beta_slow", 1)
        st.session_state.factor = rs.get("factor", 16)
        st.session_state.original_max_position_embeddings = rs.get("original_max_position_embeddings", 2048)
        st.session_state.attention_factor = rs.get("attention_factor", 1.0)

    # MoE
    if config_dict.get("use_moe"):
        st.session_state.num_experts = config_dict.get("num_experts", 4)
        st.session_state.num_experts_per_tok = config_dict.get("num_experts_per_tok", 1)
        st.session_state.moe_intermediate_size = config_dict.get(
            "moe_intermediate_size",
            compute_intermediate_size(st.session_state.hidden_size),
        )
        st.session_state.norm_topk_prob = config_dict.get("norm_topk_prob", True)
        st.session_state.router_aux_loss_coef = config_dict.get("router_aux_loss_coef", 5e-4)

    # Linear arch
    if arch == "linear":
        st.session_state.full_attention_interval = config_dict.get("full_attention_interval", 4)
        st.session_state.linear_conv_kernel_dim = config_dict.get("linear_conv_kernel_dim", 4)
        st.session_state.linear_key_head_dim = config_dict.get(
            "linear_key_head_dim",
            compute_head_dim(st.session_state.hidden_size, st.session_state.num_attention_heads),
        )
        st.session_state.linear_value_head_dim = config_dict.get(
            "linear_value_head_dim",
            compute_head_dim(st.session_state.hidden_size, st.session_state.num_attention_heads),
        )
        st.session_state.linear_num_key_heads = config_dict.get("linear_num_key_heads", st.session_state.num_attention_heads)
        st.session_state.linear_num_value_heads = config_dict.get("linear_num_value_heads", st.session_state.num_attention_heads)

    # Looped arch
    if arch == "looped":
        st.session_state.loop_max_steps = config_dict.get("loop_max_steps", 32)
        st.session_state.loop_q_threshold = config_dict.get("q_threshold", 0.9)
        st.session_state.loop_n_supervision = config_dict.get("n_supervision", 6)
        st.session_state.loop_depth_reward = config_dict.get("depth_reward", 0.01)
        st.session_state.exit_in_training = config_dict.get("exit_in_training", True)
        st.session_state.loop_beta = config_dict.get("beta", 0.5)
        st.session_state.loop_distill_weight = config_dict.get("distill_weight", 0.0)
        st.session_state.loop_distill_temperature = config_dict.get("distill_temperature", 2.0)
        st.session_state.teacher_stop_grad = config_dict.get("teacher_stop_grad", True)
        st.session_state.loop_depth_gain_reward = config_dict.get("depth_gain_reward", 0.0)
        st.session_state.loop_encoder_layers = config_dict.get("loop_encoder_layers", [0, 1])
        st.session_state.loop_body_layers = config_dict.get("loop_body_layers", [2, 3, 4])
        st.session_state.loop_output_layers = config_dict.get("loop_output_layers", [5, 6, 7])

    # Early Exit
    ee_layers = config_dict.get("early_exit_layers")
    st.session_state.early_exit_enabled = bool(ee_layers)
    if ee_layers:
        st.session_state.early_exit_layers = ee_layers
        st.session_state.early_exit_loss_weight = config_dict.get("early_exit_loss_weight", 0.3)


def parse_training_metrics(log_text):
    """Extract training metrics from log for charting. Returns list of dicts."""
    rows = []
    for line in log_text.strip().split("\n"):
        ep = re.search(r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\)", line)
        if not ep:
            continue
        epoch, _epochs, epoch_step, epoch_steps = map(int, ep.groups())
        row = {
            "_epoch": epoch,
            "_epoch_step": epoch_step,
            "_global_step": (epoch - 1) * epoch_steps + epoch_step,
        }
        for m in re.finditer(r"([a-zA-Z_]\w*):\s*([\d.eE+-]+)", line):
            try:
                row[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        if row:
            rows.append(row)
    return rows


def render_log_metrics_charts(metrics):
    """Render loss, learning rate and epoch time separately for one log."""
    if not metrics:
        return

    loss_keys = sorted({
        key for metric in metrics for key in metric
        if key == "loss" or key.endswith("_loss")
    })
    lr_keys = [
        key for key in ("lr", "learning_rate")
        if any(key in metric for metric in metrics)
    ]
    time_keys = ["epoch_time"] if any("epoch_time" in metric for metric in metrics) else []
    chart_groups = [
        ("Loss", loss_keys),
        ("Learning Rate", lr_keys[:1]),
        ("Epoch Time (min)", time_keys),
    ]

    rendered = False
    for title, keys in chart_groups:
        if not keys:
            continue
        rendered = True
        st.caption(title)
        chart_data = [
            {
                "step": metric.get("_global_step", index + 1),
                **{key: metric[key] for key in keys if key in metric},
            }
            for index, metric in enumerate(metrics)
        ]
        st.line_chart(chart_data, x="step", use_container_width=True)

    if not rendered:
        st.info("This log has no plottable training metrics yet.")


# ═══════════════════════════════════════════════════════════════
# Architecture diagram (HTML)
# ═══════════════════════════════════════════════════════════════

def arch_diagram(cfg: dict) -> str:
    """Build an HTML architecture diagram string."""
    n = cfg["num_hidden_layers"]
    use_moe = cfg.get("use_moe", False)
    arch = cfg.get("model_architecture", "standard")
    residual_type = cfg.get("residual_type", "standard")

    blocks = []

    def _block(text, detail="", color_start="#1e3a5f", color_end="#2563eb"):
        return (
            f'<div class="arch-block" style="background: linear-gradient(135deg, '
            f'{color_start} 0%, {color_end} 100%);">'
            f"<div>{text}</div>"
            f'<div class="arch-label">{detail}</div>'
            f"</div>"
        )

    def _arrow():
        return '<div class="arch-arrow">&#9660;</div>'

    # Input
    blocks.append(_block("Input", "token ids"))
    blocks.append(_arrow())

    # Embedding
    emb_detail = f"vocab={cfg['vocab_size']}, dim={cfg['hidden_size']}"
    blocks.append(_block("Embedding", emb_detail, "#0f3b5e", "#1d6fa5"))
    blocks.append(_arrow())

    # Layer stack
    layer_color_s = "#1a3a5c"
    layer_color_e = "#2d6a9f"
    linear_color_s = "#3b1f6e"
    linear_color_e = "#7c3aed"
    for i in range(min(n, 32)):  # cap display at 32
        label = f"Layer {i+1}" if n <= 32 or i < 16 or i >= n - 16 else "..."

        if n > 32 and i == 16:
            blocks.append(
                _block(
                    "&#8942;",
                    f"... {n - 30} more layers ...",
                    "#1a2332",
                    "#1a2332",
                )
            )
            blocks.append(_arrow())
            continue
        elif n > 32 and i >= 16 and i < n - 16:
            continue

        if arch == "linear":
            interval = cfg.get("full_attention_interval", 4)
            is_full = interval > 0 and (i + 1) % interval == 0
            attn_type = "Full Attn" if is_full else "Linear Attn"
            cs, ce = (layer_color_s, layer_color_e) if is_full else (linear_color_s, linear_color_e)
        else:
            attn_type = f"Attn({cfg['num_attention_heads']}h/{cfg['num_key_value_heads']}kv)"
            cs, ce = layer_color_s, layer_color_e

        ffn_label = "MoE-FFN" if use_moe else "FFN"
        layer_detail = f"{attn_type} + {ffn_label}"
        if residual_type == "mhc":
            layer_detail += f" · mHC({cfg.get('hc_mult', 4)} streams)"
        elif residual_type == "attnres":
            variant = cfg.get("attnres_variant", "block").title()
            layer_detail += f" · {variant} AttnRes"
        if use_moe:
            n_exp = cfg.get("num_experts", 4)
            n_top = cfg.get("num_experts_per_tok", 1)
            layer_detail += f" ({n_exp}E, top-{n_top})"
        if cfg.get("early_exit_layers") and (i + 1) in cfg["early_exit_layers"]:
            layer_detail += " · EE"

        if arch == "looped":
            loop_color_s, loop_color_e = "#4a1d5e", "#9d4edd"
            if i in cfg.get("loop_body_layers", []):
                layer_detail += " · LOOP"
                cs, ce = loop_color_s, loop_color_e
            elif i in cfg.get("loop_encoder_layers", []):
                layer_detail += " · ENC"
                cs, ce = "#0f3b5e", "#1d6fa5"
            elif i in cfg.get("loop_output_layers", []):
                layer_detail += " · DEC"
                cs, ce = "#1b4332", "#2d8a4e"

        blocks.append(
            _block(f"Layer {i+1}", layer_detail, cs, ce)
        )
        blocks.append(_arrow())

    # Final Norm
    blocks.append(
        _block("Final Norm", f"RMSNorm({cfg['hidden_size']})", "#1b4332", "#2d8a4e")
    )
    blocks.append(_arrow())

    # LM Head
    lm_detail = f"vocab={cfg['vocab_size']}, dim={cfg['hidden_size']}"
    if cfg.get("tie_word_embeddings"):
        lm_detail += " (tied)"
    blocks.append(_block("LM Head", lm_detail, "#4a1942", "#7c3aed"))
    blocks.append(_arrow())

    # Output
    blocks.append(_block("Output", "logits", "#3b0f2e", "#6b1d5e"))

    return f'<div class="arch-wrap">{"".join(blocks)}</div>'


# ═══════════════════════════════════════════════════════════════
# Initialize session state
# ═══════════════════════════════════════════════════════════════

def init_from_preset(preset_name: str):
    """Load preset values into session_state."""
    if preset_name == "Custom":
        return
    data = PRESETS.get(preset_name)
    if data is None:
        return
    # Presets predate the optional residual topologies; selecting one resets to
    # the exact historical Standard Transformer graph.
    st.session_state.residual_type = "standard"
    st.session_state.pop("_residual_radio", None)
    for k, v in data.items():
        st.session_state[k] = v


if "preset" not in st.session_state:
    st.session_state.preset = "instinct-3"
    init_from_preset("instinct-3")

if "model_architecture" not in st.session_state:
    st.session_state.model_architecture = "standard"

# Auto-load config from file on first page load (before any user interaction)
if not st.session_state.get("_config_auto_loaded"):
    st.session_state._config_auto_loaded = True
    trainer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer")
    state_file = os.path.join(trainer_dir, "webui_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if k.startswith("btn_") or k == "clear_train_log":
                    continue  # 兼容旧 state 文件里已保存的按钮 key，赋值会报错
                st.session_state[k] = v
        st.rerun()
    else:
        default_config = os.path.join(trainer_dir, "config_pretrain.json")
        if os.path.exists(default_config):
            with open(default_config, "r", encoding="utf-8") as f:
                load_config_to_session(json.load(f))
            if st.session_state.preset != "instinct-3":
                st.rerun()

# Handle deferred config load from button click (must run before any widget with the same key)
if "_pending_config_load" in st.session_state:
    load_config_to_session(st.session_state._pending_config_load)
    del st.session_state._pending_config_load
    st.rerun()

# ═══════════════════════════════════════════════════════════════
# Refresh recovery — detect running training process on page reload
# ═══════════════════════════════════════════════════════════════
def _find_running_train_process():
    """Check if a train_*.py process is still alive. Returns (pid, script_name) or (None, None)."""
    import subprocess as _sp
    try:
        if sys.platform == "win32":
            result = _sp.run(
                ['wmic', 'process', 'where', 'name="python.exe"', 'get', 'ProcessId,CommandLine'],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if 'train_' in line and '.py' in line:
                    parts = line.strip().rsplit(None, 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        script = line.split('train_')[1].split('.py')[0] if 'train_' in line else '?'
                        return int(parts[1]), f"train_{script}"
        else:
            result = _sp.run(['pgrep', '-f', 'train_.*\\.py'], capture_output=True, text=True, timeout=5)
            pids = result.stdout.strip().split()
            if pids:
                result2 = _sp.run(['ps', '-p', pids[0], '-o', 'command='], capture_output=True, text=True, timeout=5)
                script = 'train_' + result2.stdout.split('train_')[1].split('.py')[0] if 'train_' in result2.stdout else '?'
                return int(pids[0]), f"train_{script}"
    except Exception:
        pass
    return None, None

def _latest_train_log(trainer_dir):
    """Return the most recently modified training log file (or None)."""
    candidates = []
    logs_dir = os.path.join(trainer_dir, "logs")
    if os.path.isdir(logs_dir):
        candidates += [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".log")]
    legacy = os.path.join(trainer_dir, "train_output.log")
    if os.path.exists(legacy):
        candidates.append(legacy)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


_DEFAULT_WEIGHT_PREFIX = {
    "pretrain": "pretrain",
    "full_sft": "full_sft",
    "lora": "lora",
    "dpo": "dpo",
    "ppo": "ppo_actor",
    "grpo": "grpo",
    "agent": "agent",
    "distillation": "full_dist",
}

# 暂停退出码：训练进程识别到 .pause_request 标记后保存检查点并以 42 退出，
# 与 0=成功 / 其他=失败 相区分，WebUI 轮询时据此把状态置为 "paused"。
PAUSE_EXIT_CODE = 42


def _default_weight_prefix(train_type):
    return _DEFAULT_WEIGHT_PREFIX.get(train_type, train_type)


def _arch_tag():
    architecture = st.session_state.get("model_architecture", "standard")
    tag = {"standard": "", "linear": "_linear", "looped": "_looped"}.get(
        architecture, f"_{architecture}"
    )
    residual_type = st.session_state.get("residual_type", "standard")
    if residual_type == "mhc":
        tag += "_mhc"
    elif residual_type == "attnres":
        variant = st.session_state.get("attnres_variant", "block")
        tag += f"_attnres_{variant}"
        if variant == "block":
            tag += str(st.session_state.get("attnres_block_size", 2))
    return tag


def _matches_arch_tag(prefix, arch_tag):
    """Match generated weight prefixes without mixing backbone/residual topologies."""
    if arch_tag:
        return prefix.endswith(arch_tag)
    return not (
        prefix.endswith(("_linear", "_looped", "_mhc", "_attnres_full"))
        or re.search(r"_attnres_block\d+$", prefix) is not None
    )


def _persist_panel_state(trainer_dir):
    """把训练面板参数（含训练超参）写入 webui_state.json，刷新/重启后自动恢复。"""
    def _serializable(v):
        return isinstance(v, (str, int, float, bool, list, dict, type(None)))
    state = {
        k: v for k, v in st.session_state.items()
        if not k.startswith("_")
        and not k.startswith("btn_")  # 按钮状态只读，恢复赋值会抛 StreamlitValueAssignmentNotAllowedError
        and k not in ("train_proc", "train_status", "train_log_path", "clear_train_log")
        and not k.startswith("save_prefix_")
        and _serializable(v)
    }
    with open(os.path.join(trainer_dir, "webui_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _latest_checkpoint_prefix(train_type, hidden_size, use_moe, arch_tag=""):
    """Derive the newest matching run id from checkpoints/{train_type}_*_{dim}{moe}_resume.pth."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ckpt_dir = os.path.join(repo_root, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    dim_suffix = f"_{hidden_size}{'_moe' if use_moe else ''}"
    # 按默认权重前缀匹配（如 distillation 对应 full_dist_*），而非直接用 train_type，
    # 否则蒸馏的 full_dist_*_resume.pth 无法被续训发现。
    expect_stem = _default_weight_prefix(train_type) + "_"
    matches = []
    for f in os.listdir(ckpt_dir):
        if not f.endswith("_resume.pth"):
            continue
        stem = f[: -len("_resume.pth")]
        if stem.endswith(dim_suffix):
            prefix = stem[: -len(dim_suffix)]
            if prefix.startswith(expect_stem):
                if not _matches_arch_tag(prefix, arch_tag):
                    continue
                matches.append((os.path.getmtime(os.path.join(ckpt_dir, f)), prefix))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def _resolve_save_prefix(train_type, hidden_size, use_moe, from_resume, stamp):
    arch_tag = _arch_tag()
    if from_resume:
        # 1. 暂停状态记录的 weight 优先：这正是刚刚被暂停的那一轮训练，绝不落到旧检查点。
        paused = _read_paused_state()
        if paused and paused.get("weight"):
            weight = paused["weight"]
            if _resume_checkpoint_exists(weight, hidden_size, use_moe):
                st.session_state[f"save_prefix_{train_type}"] = weight
                return weight
        # 2. 会话内已选择的 save_prefix（同会话续训无需重新扫描磁盘）。
        prefix = st.session_state.get(f"save_prefix_{train_type}")
        if prefix:
            return prefix
        # 3. 磁盘扫描最新匹配检查点（仍按默认前缀 stem 匹配，保持 train_type 隔离）。
        prefix = _latest_checkpoint_prefix(train_type, hidden_size, use_moe, arch_tag)
        if prefix:
            return prefix
        # 4. 找不到任何续训检查点：不清静合成新前缀，置标记让调用方告警（避免续训落到旧/新权重）。
        st.session_state["_resume_not_found"] = True
        return None
    prefix = f"{_default_weight_prefix(train_type)}_{stamp}{arch_tag}"
    st.session_state[f"save_prefix_{train_type}"] = prefix
    return prefix


def _checkpoints_dir():
    """与训练进程共享的 checkpoints/ 目录（训练默认 ./checkpoints，相对仓库根）。"""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    return os.path.join(repo_root, "checkpoints")


def _resume_checkpoint_exists(weight, hidden_size, use_moe):
    """续训检查点是否存在：checkpoints/{weight}_{dim}{_moe}_resume.pth。"""
    return os.path.exists(os.path.join(
        _checkpoints_dir(),
        f"{weight}_{hidden_size}{'_moe' if use_moe else ''}_resume.pth",
    ))


def _pause_flag_path():
    """暂停请求标记：训练进程在每个 step 边界检查该文件，存在则保存检查点并以 42 退出。"""
    return os.path.join(_checkpoints_dir(), ".pause_request")


def _paused_state_path():
    """WebUI 恢复用暂停状态记录；页面重载时据此把状态置为 \"paused\"。"""
    return os.path.join(_checkpoints_dir(), ".paused.json")


def _request_pause(train_type):
    """请求暂停当前训练：写暂停标记 + 状态记录。checkpoints/ 可能在首次周期保存前不存在，需先创建。"""
    os.makedirs(_checkpoints_dir(), exist_ok=True)
    with open(_pause_flag_path(), "w", encoding="utf-8"):
        pass  # 训练进程仅检查文件存在性
    state = {
        "train_type": train_type,
        "paused_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": None,  # 点击时 epoch 未知，真实进度以训练进程保存的检查点为准
        "weight": st.session_state.get(f"save_prefix_{train_type}") or None,
    }
    with open(_paused_state_path(), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _clear_pause_markers():
    """清除暂停标记与状态记录（启动 / 续训前调用，防止残留标记误触发暂停）。"""
    for path in (_pause_flag_path(), _paused_state_path()):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _read_paused_state():
    """读取暂停状态记录；文件不存在或内容损坏时返回 None（不抛出，避免拖垮整页）。"""
    path = _paused_state_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


_BASE_WEIGHT_TYPE = {
    "pretrain": "pretrain",
    "full_sft": "pretrain",
    "lora": "full_sft",
    "dpo": "full_sft",
    "ppo": "full_sft",
    "grpo": "full_sft",
    "agent": "full_sft",
    "distillation": "full_sft",
}


def _available_weight_files():
    """Scan out/ + checkpoints/ for loadable base-weight .pth files (excludes _resume.pth)."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    files = []
    for sub in ("out", "checkpoints"):
        d = os.path.join(repo_root, sub)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".pth") and "_resume.pth" not in f:
                    files.append(os.path.join(sub, f))
    return files


def _latest_weight_file(base_type, hidden_size, use_moe, arch_tag=""):
    """Newest loadable .pth of a base weight type (out/ preferred, then checkpoints/)."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dim_suffix = f"_{hidden_size}{'_moe' if use_moe else ''}"
    candidates = []
    for sub in ("out", "checkpoints"):
        d = os.path.join(repo_root, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith(".pth") or "_resume.pth" in f:
                continue
            stem = f[: -len(".pth")]
            if stem.endswith(dim_suffix) and stem[: -len(dim_suffix)].startswith(f"{base_type}_"):
                type_part = stem[: -len(dim_suffix)]
                if not _matches_arch_tag(type_part, arch_tag):
                    continue
                candidates.append((os.path.getmtime(os.path.join(d, f)), os.path.join(sub, f)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _try_recover_training_state():
    if "train_status" in st.session_state:
        return

    trainer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer")
    log_path = _latest_train_log(trainer_dir)

    if log_path is None:
        return

    mtime = os.path.getmtime(log_path)
    recently_modified = (time.time() - mtime) < 60

    pid, script = _find_running_train_process()
    paused_state = _read_paused_state()

    if pid is not None:
        st.session_state.train_status = "running"
        st.session_state.train_log_path = log_path
        if pid and script:
            st.session_state.train_type = script
    elif paused_state is not None:
        # 必须先于 recently_modified 检查：刚暂停的日志 (<60s 新) 否则会被误判为 running。
        st.session_state.train_status = "paused"
        st.session_state.train_log_path = log_path
        if paused_state.get("train_type"):
            train_type = paused_state["train_type"]
            st.session_state.train_type = train_type
        else:
            train_type = st.session_state.get("train_type", "pretrain")
        if paused_state.get("weight"):
            st.session_state[f"save_prefix_{train_type}"] = paused_state.get("weight")
    elif recently_modified:
        st.session_state.train_status = "running"
        st.session_state.train_log_path = log_path
        if pid and script:
            st.session_state.train_type = script
    else:
        with open(log_path, "r", encoding="utf-8") as f:
            log = f.read()
        if log.strip():
            st.session_state.train_status = "success"
            st.session_state.train_log_path = log_path

_try_recover_training_state()

# ═══════════════════════════════════════════════════════════════
# ═══ SIDEBAR ═══
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="font-size:22px; font-weight:700; letter-spacing:-0.5px; '
        'background: linear-gradient(135deg, #60a5fa, #a78bfa); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent; '
        'margin-bottom: 4px;">instinct</div>',
        unsafe_allow_html=True,
    )
    st.caption("Interactive Model Configurator")
    st.markdown(
        '<hr style="margin: 8px 0 16px 0; border-color: #1e293b;">',
        unsafe_allow_html=True,
    )

    # ── Model Preset ──
    with st.expander("Model Preset", expanded=True):
        preset_options = [
            "instinct-3",
            "instinct-3-moe",
            "instinct2-small",
            "instinct2",
            "instinct-linear",
            "Custom",
        ]
        current_preset = st.session_state.preset
        default_idx = (
            preset_options.index(current_preset)
            if current_preset in preset_options
            else 5
        )
        chosen = st.radio(
            "Quick-select preset",
            preset_options,
            index=default_idx,
            key="_preset_radio",
            label_visibility="collapsed",
            horizontal=False,
        )
        if chosen != st.session_state.preset:
            st.session_state.preset = chosen
            if chosen != "Custom":
                init_from_preset(chosen)
            st.rerun()

    # ── Model Architecture ──
    with st.expander("Model Architecture", expanded=True):
        arch_options = ["Standard Transformer", "MoE Transformer", "GatedDeltaNet (Linear)", "Looped Transformer"]
        current_arch = st.session_state.get("model_architecture", "standard")
        arch_index = {"standard": 0, "moe": 1, "linear": 2, "looped": 3}.get(current_arch, 0)
        chosen_arch = st.radio(
            "Architecture type",
            arch_options,
            index=arch_index,
            key="_arch_radio",
            label_visibility="collapsed",
            horizontal=False,
        )
        arch_map = {
            "Standard Transformer": "standard",
            "MoE Transformer": "moe",
            "GatedDeltaNet (Linear)": "linear",
            "Looped Transformer": "looped",
        }
        mapped = arch_map[chosen_arch]
        if mapped != st.session_state.get("model_architecture", "standard"):
            st.session_state.model_architecture = mapped
            if mapped == "moe":
                st.session_state.use_moe = True
            st.rerun()

    # ── Core Architecture ──
    with st.expander("Core Architecture", expanded=True):
        st.slider(
            "hidden_size",
            128,
            2048,
            st.session_state.get("hidden_size", 768),
            step=64,
            key="hidden_size",
        )
        st.slider(
            "num_hidden_layers",
            1,
            64,
            st.session_state.get("num_hidden_layers", 8),
            key="num_hidden_layers",
        )
        st.number_input(
            "vocab_size",
            1000,
            100000,
            st.session_state.get("vocab_size", 6400),
            step=100,
            key="vocab_size",
        )
        st.slider(
            "dropout",
            0.0,
            0.5,
            st.session_state.get("dropout", 0.0),
            step=0.05,
            key="dropout",
        )
        st.selectbox(
            "hidden_act",
            ["silu", "gelu", "relu"],
            index=["silu", "gelu", "relu"].index(
                st.session_state.get("hidden_act", "silu")
            ),
            key="hidden_act",
        )
        st.checkbox(
            "tie_word_embeddings",
            value=st.session_state.get("tie_word_embeddings", False),
            key="tie_word_embeddings",
        )

    # ── Residual Connections ──
    with st.expander("Residual Connections (Experimental)", expanded=True):
        _residual_labels = {
            "standard": "Standard residual",
            "mhc": "mHC (Manifold-Constrained Hyper-Connections)",
            "attnres": "AttnRes (Attention Residuals)",
        }
        _residual_options = list(_residual_labels)
        _current_residual = st.session_state.get("residual_type", "standard")
        _chosen_residual = st.radio(
            "Residual topology",
            _residual_options,
            index=_residual_options.index(_current_residual),
            format_func=lambda value: _residual_labels[value],
            key="_residual_radio",
        )
        st.session_state.residual_type = _chosen_residual

        if _chosen_residual == "mhc":
            st.caption("Keeps parallel residual streams and projects their mixer onto a doubly-stochastic manifold.")
            st.slider(
                "hc_mult (parallel streams)", 1, 8,
                st.session_state.get("hc_mult", 4), key="hc_mult",
            )
            st.slider(
                "hc_sinkhorn_iters", 1, 50,
                st.session_state.get("hc_sinkhorn_iters", 20), key="hc_sinkhorn_iters",
                help="Alternating row/column normalizations used by the Sinkhorn-Knopp projection.",
            )
            st.number_input(
                "hc_eps", min_value=1e-9, max_value=1e-3,
                value=float(st.session_state.get("hc_eps", 1e-6)),
                format="%.1e", key="hc_eps",
            )
        elif _chosen_residual == "attnres":
            st.caption("Uses learned pseudo-queries to select residual sources along model depth.")
            st.radio(
                "AttnRes variant", ["block", "full"],
                index=["block", "full"].index(st.session_state.get("attnres_variant", "block")),
                format_func=lambda value: "Block AttnRes (recommended)" if value == "block" else "Full AttnRes",
                key="attnres_variant",
                horizontal=True,
            )
            if st.session_state.get("attnres_variant", "block") == "block":
                _default_attnres_block = max(
                    1, math.ceil(2 * st.session_state.get("num_hidden_layers", 8) / 8)
                )
                _max_attnres_block = max(2, 2 * st.session_state.get("num_hidden_layers", 8))
                st.session_state.attnres_block_size = min(
                    _max_attnres_block,
                    max(1, st.session_state.get("attnres_block_size", _default_attnres_block)),
                )
                st.number_input(
                    "attnres_block_size (sublayers)", min_value=1,
                    max_value=_max_attnres_block,
                    value=st.session_state.get("attnres_block_size", _default_attnres_block),
                    step=1, key="attnres_block_size",
                    help="Counts attention and MLP sublayers; choose about total_sublayers / 8 for ~8 blocks.",
                )

    # ── Attention ──
    with st.expander("Attention", expanded=True):
        st.slider(
            "num_attention_heads",
            1,
            32,
            st.session_state.get("num_attention_heads", 8),
            key="num_attention_heads",
        )
        st.slider(
            "num_key_value_heads",
            1,
            32,
            st.session_state.get("num_key_value_heads", 4),
            key="num_key_value_heads",
        )
        _h = st.session_state.get("hidden_size", 768)
        _q = st.session_state.get("num_attention_heads", 8)
        _hd = compute_head_dim(_h, _q)
        st.markdown(
            f'<div style="background:#1a2332;border:1px solid #2d3a4e;'
            f'border-radius:8px;padding:8px 12px;margin-top:4px;">'
            f'<span style="color:#94a3b8;font-size:12px;">head_dim (auto) </span>'
            f'<span style="font-family:Courier New;font-weight:700;font-size:18px;'
            f'color:#e2e8f0;">{_hd}</span>'
            f'<span style="color:#64748b;font-size:12px;margin-left:8px;">'
            f'= {_h} // {_q}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Linear Attention Config ──
    if st.session_state.get("model_architecture") == "linear":
        with st.expander("Linear Attention Config", expanded=True):
            _h = st.session_state.get("hidden_size", 768)
            _q = st.session_state.get("num_attention_heads", 8)
            _hd = compute_head_dim(_h, _q)
            st.slider(
                "full_attention_interval",
                1, 16,
                st.session_state.get("full_attention_interval", 4),
                key="full_attention_interval",
                help="Every Nth layer uses standard full attention",
            )
            st.slider(
                "linear_conv_kernel_dim",
                1, 8,
                st.session_state.get("linear_conv_kernel_dim", 4),
                key="linear_conv_kernel_dim",
                help="Conv1d kernel size for GatedDeltaNet",
            )
            override_lin_kd = st.checkbox(
                "Override linear_key_head_dim",
                value=st.session_state.get("_override_lin_kd", False),
                key="_override_lin_kd",
            )
            if override_lin_kd:
                st.number_input(
                    "linear_key_head_dim",
                    value=st.session_state.get("linear_key_head_dim", _hd),
                    step=8, key="linear_key_head_dim",
                )
            else:
                st.session_state["linear_key_head_dim"] = _hd
                st.markdown(
                    f'<div style="background:#1a2332;border:1px solid #2d3a4e;'
                    f'border-radius:8px;padding:8px 12px;margin-top:4px;">'
                    f'<span style="color:#94a3b8;font-size:12px;">linear_key_head_dim (auto) </span>'
                    f'<span style="font-family:Courier New;font-weight:700;'
                    f'font-size:18px;color:#e2e8f0;">{_hd}</span>'
                    f"</div>", unsafe_allow_html=True,
                )
            override_lin_vd = st.checkbox(
                "Override linear_value_head_dim",
                value=st.session_state.get("_override_lin_vd", False),
                key="_override_lin_vd",
            )
            if override_lin_vd:
                st.number_input(
                    "linear_value_head_dim",
                    value=st.session_state.get("linear_value_head_dim", _hd),
                    step=8, key="linear_value_head_dim",
                )
            else:
                st.session_state["linear_value_head_dim"] = _hd
                st.markdown(
                    f'<div style="background:#1a2332;border:1px solid #2d3a4e;'
                    f'border-radius:8px;padding:8px 12px;margin-top:4px;">'
                    f'<span style="color:#94a3b8;font-size:12px;">linear_value_head_dim (auto) </span>'
                    f'<span style="font-family:Courier New;font-weight:700;'
                    f'font-size:18px;color:#e2e8f0;">{_hd}</span>'
                    f"</div>", unsafe_allow_html=True,
                )
            st.number_input(
                "linear_num_key_heads",
                value=st.session_state.get("linear_num_key_heads", _q),
                step=1, key="linear_num_key_heads",
            )
            st.number_input(
                "linear_num_value_heads",
                value=st.session_state.get("linear_num_value_heads", _q),
                step=1, key="linear_num_value_heads",
            )

    # ── Looped (LoopUS) Config ──
    if st.session_state.get("model_architecture") == "looped":
        with st.expander("Looped (LoopUS) Config", expanded=True):
            _n_layers = st.session_state.get("num_hidden_layers", 8)
            st.number_input(
                "loop_max_steps (safety cap)",
                min_value=1, max_value=128,
                value=st.session_state.get("loop_max_steps", 32),
                key="loop_max_steps",
                help="Dynamic loop safety cap. The loop exits when q >= threshold; "
                     "this only bounds worst-case (effectively infinite for trained models).",
            )
            st.slider(
                "loop_q_threshold",
                0.1, 1.0,
                st.session_state.get("loop_q_threshold", 0.9),
                step=0.05,
                key="loop_q_threshold",
                help="Confidence threshold for early exit (q > threshold halts)",
            )
            st.number_input(
                "loop_n_supervision",
                min_value=1, max_value=128,
                value=st.session_state.get("loop_n_supervision", 6),
                key="loop_n_supervision",
                help="How many of the loop steps get gradients (random deep supervision)",
            )
            st.slider(
                "loop_depth_reward (λ)",
                0.0, 0.5,
                st.session_state.get("loop_depth_reward", 0.01),
                step=0.005,
                key="loop_depth_reward",
                help="Reward weight on expected loop depth λ·E[steps]. "
                     "Higher λ → stronger incentive to exit early. Anneal upward for faster exit.",
            )
            st.checkbox(
                "exit_in_training",
                value=st.session_state.get("exit_in_training", True),
                key="exit_in_training",
                help="Allow per-sample early exit during training (q > threshold stops the loop). "
                     "Disable to always run the full cap and only reward via λ.",
            )
            st.slider(
                "loop_beta (monotonicity weight)",
                0.0, 2.0,
                st.session_state.get("loop_beta", 0.5),
                step=0.05,
                key="loop_beta",
            )
            st.markdown("#### Deep-Thinking Rewards")
            st.caption("Self-distillation + depth-gain reward. Train with "
                       "exit_in_training OFF so every loop state is visited.")
            st.slider(
                "loop_distill_weight",
                0.0, 2.0,
                st.session_state.get("loop_distill_weight", 0.0),
                step=0.05,
                key="loop_distill_weight",
                help="Self-distillation weight: shallower loop depths imitate the "
                     "final depth's output distribution (KL), forcing deeper states "
                     "to carry richer representation. 0 = off.",
            )
            st.slider(
                "loop_distill_temperature",
                0.5, 5.0,
                st.session_state.get("loop_distill_temperature", 2.0),
                step=0.1,
                key="loop_distill_temperature",
                help="Self-distillation temperature T (soften teacher/student "
                     "distributions; KL scaled by T²)",
            )
            st.checkbox(
                "teacher_stop_grad",
                value=st.session_state.get("teacher_stop_grad", True),
                key="teacher_stop_grad",
                help="Stop gradient on teacher (final-depth) logits to avoid the "
                     "trivial self-KL solution. 1 = recommended.",
            )
            st.slider(
                "loop_depth_gain_reward",
                0.0, 1.0,
                st.session_state.get("loop_depth_gain_reward", 0.0),
                step=0.01,
                key="loop_depth_gain_reward",
                help="Depth-gain reward: reward a deeper step only when it reduces "
                     "LM loss vs the shallowest-supervised-depth baseline "
                     "(L1 − Lb)_+. 0 = off.",
            )
            st.caption(f"Layer partition (total: {_n_layers})")
            enc_s = st.text_input(
                "Encoder layers (comma-separated)",
                value=",".join(str(x) for x in st.session_state.get("loop_encoder_layers", [0, 1])),
                key="_loop_encoder_layers_str",
            )
            body_s = st.text_input(
                "Loop body layers (comma-separated)",
                value=",".join(str(x) for x in st.session_state.get("loop_body_layers", [2, 3, 4])),
                key="_loop_body_layers_str",
            )
            out_s = st.text_input(
                "Output layers (comma-separated)",
                value=",".join(str(x) for x in st.session_state.get("loop_output_layers", [5, 6, 7])),
                key="_loop_output_layers_str",
            )
            try:
                st.session_state.loop_encoder_layers = [int(x.strip()) for x in enc_s.split(",") if x.strip()]
                st.session_state.loop_body_layers = [int(x.strip()) for x in body_s.split(",") if x.strip()]
                st.session_state.loop_output_layers = [int(x.strip()) for x in out_s.split(",") if x.strip()]
            except ValueError:
                pass

    # ── Position Encoding ──
    with st.expander("Position Encoding", expanded=True):
        st.number_input(
            "max_position_embeddings",
            value=st.session_state.get("max_position_embeddings", 32768),
            step=1024,
            key="max_position_embeddings",
            format="%d",
        )
        st.number_input(
            "rope_theta",
            10000.0,
            10000000.0,
            value=st.session_state.get("rope_theta", 1e6),
            format="%.0e",
            key="rope_theta",
        )
        st.checkbox(
            "YaRN (inference_rope_scaling)",
            value=st.session_state.get("inference_rope_scaling", False),
            key="inference_rope_scaling",
        )
        if st.session_state.get("inference_rope_scaling", False):
            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    "beta_fast",
                    value=st.session_state.get("beta_fast", 32),
                    key="beta_fast",
                )
                st.number_input(
                    "beta_slow",
                    value=st.session_state.get("beta_slow", 1),
                    key="beta_slow",
                )
                st.number_input(
                    "factor",
                    value=st.session_state.get("factor", 16),
                    key="factor",
                )
            with c2:
                st.number_input(
                    "original_max_pos",
                    value=st.session_state.get(
                        "original_max_position_embeddings", 2048
                    ),
                    key="original_max_position_embeddings",
                )
                st.number_input(
                    "attention_factor",
                    value=st.session_state.get("attention_factor", 1.0),
                    step=0.1,
                    key="attention_factor",
                    format="%.1f",
                )

    # ── MoE ──
    with st.expander("MoE (Experimental)", expanded=True):
        st.checkbox(
            "use_moe",
            value=st.session_state.get("use_moe", False),
            key="use_moe",
        )
        if st.session_state.get("use_moe", False):
            st.slider(
                "num_experts",
                2,
                16,
                st.session_state.get("num_experts", 4),
                key="num_experts",
            )
            st.slider(
                "num_experts_per_tok",
                1,
                4,
                st.session_state.get("num_experts_per_tok", 1),
                key="num_experts_per_tok",
            )
            _moe_int = st.session_state.get(
                "moe_intermediate_size", compute_intermediate_size(_h)
            )
            override_moe = st.checkbox(
                "Override moe_intermediate_size",
                value=st.session_state.get("_override_moe_int", False),
                key="_override_moe_int",
            )
            if override_moe:
                st.number_input(
                    "moe_intermediate_size",
                    value=_moe_int,
                    step=64,
                    key="moe_intermediate_size",
                )
            else:
                _auto_moe_int = compute_intermediate_size(
                    st.session_state.get("hidden_size", 768)
                )
                st.session_state["moe_intermediate_size"] = _auto_moe_int
                st.markdown(
                    f'<div style="background:#1a2332;border:1px solid #2d3a4e;'
                    f'border-radius:8px;padding:8px 12px;margin-top:4px;">'
                    f'<span style="color:#94a3b8;font-size:12px;">'
                    f"moe_intermediate_size (auto) </span>"
                    f'<span style="font-family:Courier New;font-weight:700;'
                    f'font-size:18px;color:#e2e8f0;">{_auto_moe_int}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.checkbox(
                "norm_topk_prob",
                value=st.session_state.get("norm_topk_prob", True),
                key="norm_topk_prob",
            )
            st.number_input(
                "router_aux_loss_coef",
                0.0,
                0.01,
                value=st.session_state.get("router_aux_loss_coef", 5e-4),
                format="%.1e",
                key="router_aux_loss_coef",
            )

    # ── Early Exit (training only) ──
    with st.expander("Early Exit (Training)", expanded=False):
        st.checkbox(
            "Enable Early Exit Loss",
            value=st.session_state.get("early_exit_enabled", False),
            key="early_exit_enabled",
            help="Add shared-LM-head CE loss at intermediate layers during training",
        )
        if st.session_state.get("early_exit_enabled", False):
            ee_default = st.session_state.get("early_exit_layers", [4, 5, 6, 7])
            layers_str = st.text_input(
                "early_exit_layers (comma-separated)",
                value=",".join(str(x) for x in ee_default),
                key="_early_exit_layers_str",
            )
            try:
                st.session_state.early_exit_layers = [int(x.strip()) for x in layers_str.split(",") if x.strip()]
            except ValueError:
                st.session_state.early_exit_layers = [4, 5, 6, 7]
            st.slider(
                "early_exit_loss_weight",
                0.0, 1.0,
                st.session_state.get("early_exit_loss_weight", 0.3),
                0.05,
                key="early_exit_loss_weight",
            )

    # ── Misc ──
    with st.expander("Misc"):
        st.number_input(
            "rms_norm_eps",
            value=st.session_state.get("rms_norm_eps", 1e-6),
            format="%.0e",
            key="rms_norm_eps",
        )
        st.checkbox(
            "flash_attn",
            value=st.session_state.get("flash_attn", True),
            key="flash_attn",
        )

    # ── Training ──
    with st.expander("🚀 Training", expanded=(st.session_state.get("train_status") == "running")):
        train_type = st.selectbox(
            "Training type",
            ["pretrain", "full_sft", "lora", "dpo", "ppo", "grpo", "agent", "distillation"],
            key="train_type",
        )
        if train_type in ("pretrain", "full_sft", "distillation"):
            st.radio("Dataset size", ["mini", "normal"], key="dataset_size", horizontal=True)
        config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "trainer", f"config_{train_type}.json"
        )
        if os.path.exists(config_file):
            if st.button("📂 Load config from file", use_container_width=True, key="btn_load_config",
                         help=f"Load model architecture from {config_file}"):
                with open(config_file, "r", encoding="utf-8") as f:
                    st.session_state._pending_config_load = json.load(f)
                st.rerun()
        st.checkbox("Resume from checkpoint (--from_resume)", value=False, key="from_resume",
                    help="Auto-detect and resume from checkpoints/{weight}_{dim}{_moe}_resume.pth")
        _weight_files = _available_weight_files()
        _auto_label = "auto (newest matching base)"
        _base_options = ["none (from scratch)", _auto_label] + _weight_files
        if "base_weight" not in st.session_state:
            st.session_state.base_weight = "none" if train_type == "pretrain" else _auto_label
        elif st.session_state.base_weight not in _base_options:
            st.session_state.base_weight = _auto_label
        st.selectbox(
            "Base weights (--from_weight)",
            _base_options,
            key="base_weight",
            help="SFT 基于 pretrain、LoRA/DPO/PPO/GRPO/Agent/蒸馏基于 full_sft 启动。"
                 "可选 out/ 与 checkpoints/ 下的 .pth（自动排除 _resume 检查点），"
                 "或 'auto' 自动选择最新匹配权重；'none' 从随机初始化开始。"
                 "选中 Resume 且检查点含完整状态时，基础权重会自动跳过。",
        )
        st.number_input(
            "batch_size",
            min_value=1, max_value=512,
            value=st.session_state.get("batch_size", 32),
            step=8, key="batch_size",
            help="批次大小。小模型 GPU 利用率低时，可提升到 64/128 放大单步 GEMM "
                 "(注意：batch×seq 与激活显存成正比，过高会 OOM)",
        )
        st.number_input(
            "max_seq_len (训练截断长度)",
            min_value=64, max_value=8192,
            value=st.session_state.get("max_seq_len", 768),
            step=32, key="max_seq_len",
            help="训练最大截断长度（token 数）。mini 数据建议 768（旧默认 340 过低，"
                 "导致 GEMM 尺寸小、GPU 利用率低）",
        )
        _packing_supported = train_type in ("pretrain", "full_sft", "lora", "distillation")
        st.checkbox(
            "Sequence packing",
            value=st.session_state.get("sequence_packing", False),
            key="sequence_packing",
            disabled=not _packing_supported,
            help="把多个完整 Pretrain/SFT 样本装入固定长度 block，保留 BOS/EOS 与 SFT loss mask，"
                 "显著减少 padding。首次启用会构建并缓存 Arrow 数据集；packing 与非 packing "
                 "的 step 坐标不同时，resume 会在当前 epoch 内先训练未对齐的原始数据行，"
                 "到达 packing 分组边界后再切换 packed blocks。",
        )
        st.number_input(
            "Packing cache batch size",
            min_value=32, max_value=10000,
            value=st.session_state.get("packing_batch_size", 1000),
            step=100, key="packing_batch_size",
            disabled=not (_packing_supported and st.session_state.get("sequence_packing", False)),
            help="首次构建 Arrow cache 时每批处理的原始样本数；越大通常填充率越高，但占用更多 CPU 内存。",
        )
        if st.session_state.get("sequence_packing", False) and st.session_state.get("from_resume", False):
            st.info(
                "从非 packing checkpoint 切换时会保留模型、优化器与 GradScaler，并还原当前 "
                "epoch 的 shuffle 顺序。未对齐的数据行继续使用原始 batch，抵达下一个 packing "
                "分组边界后，同一 epoch 的剩余数据立即切换为 packed blocks。"
            )
        st.number_input(
            "Gradient accumulation steps",
            min_value=1, max_value=128,
            value=st.session_state.get("accumulation_steps", 1),
            step=1, key="accumulation_steps",
            help="1 = 每个 batch 立即更新（不做梯度累积）。大于 1 会放大有效 batch，"
                 "但与 reduce-overhead 的 CUDA Graph 梯度缓冲不兼容，建议改用 default compile mode。",
        )
        if (
            st.session_state.get("residual_type") == "attnres"
            and st.session_state.get("batch_size", 32)
            * st.session_state.get("max_seq_len", 768) > 8192
        ):
            st.warning(
                "AttnRes 会保留残差 source bank；当前 batch×seq 偏高。"
                "16 GB GPU 建议先用 batch_size=4–8，再通过梯度累积放大有效批量。"
            )
        st.selectbox(
            "Optimizer",
            ["adamw", "adafactor", "muon"],
            index=["adamw", "adafactor", "muon"].index(
                st.session_state.get("optimizer", "adamw")
            ),
            key="optimizer",
            help="AdamW (默认) / Adafactor (torch 内置) / Muon (torch>=2.10 内置，"
                 "否则自动回退到原生纯 PyTorch 实现)",
        )
        st.checkbox(
            "Use torch.compile (Triton)",
            value=st.session_state.get("use_compile", True),
            key="use_compile",
            help="启用 torch.compile (Triton 后端)，实测约 40% 提速；MoE/Loop 变体兼容性请自行验证",
        )
        st.selectbox(
            "torch.compile mode",
            ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
            index=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"].index(
                st.session_state.get("compile_mode", "reduce-overhead")
            ),
            key="compile_mode",
            disabled=not st.session_state.get("use_compile", True),
            help="default=Triton 编译（现状）；reduce-overhead=叠加 CUDA graph，"
                 "消除 kernel launch 间隙，小模型首选（MoE 动态路由可能部分 fallback，无碍）；"
                 "max-autotune=搜索更多 kernel，首次编译较慢；"
                 "max-autotune-no-cudagraphs=保留 autotune 但关闭 CUDA Graph",
        )
        if (
            st.session_state.get("use_compile", True)
            and st.session_state.get("compile_mode", "reduce-overhead")
            in ("reduce-overhead", "max-autotune")
            and st.session_state.get("accumulation_steps", 1) > 1
        ):
            st.error(
                "当前 compile mode 使用 CUDA Graph，不能安全地跨 step 累积梯度。"
                "请把 Gradient accumulation steps 设为 1，或改用 default / "
                "max-autotune-no-cudagraphs。"
            )
        st.selectbox(
            "Training profiler",
            ["off", "timing", "torch"],
            index=["off", "timing", "torch"].index(
                st.session_state.get("profile", "off")
            ),
            key="profile",
            help="timing：低开销统计真实 step、tokens/s、前向/反向/优化器耗时和峰值显存；"
                 "torch：额外采集一个短窗口的算子/kernel trace，文件写入 profiler_traces。",
        )
        _profile_enabled = st.session_state.get("profile", "off") != "off"
        _profile_col1, _profile_col2 = st.columns(2)
        with _profile_col1:
            st.number_input(
                "Profiler warmup steps", min_value=0, max_value=1000,
                value=st.session_state.get("profile_warmup", 10), step=1,
                key="profile_warmup", disabled=not _profile_enabled,
                help="每次启动或 resume 后先跳过这些编译/预热 step。",
            )
            st.number_input(
                "Profiler report interval", min_value=1, max_value=10000,
                value=st.session_state.get("profile_interval", 100), step=10,
                key="profile_interval", disabled=not _profile_enabled,
                help="每隔多少个稳定 step 输出一次 [PROFILE] 汇总。",
            )
        with _profile_col2:
            st.number_input(
                "Trace active steps", min_value=1, max_value=100,
                value=st.session_state.get("profile_active_steps", 5), step=1,
                key="profile_active_steps",
                disabled=st.session_state.get("profile", "off") != "torch",
                help="torch 模式实际记录到 trace 的 step 数；建议 3–10。",
            )
        if st.session_state.get("profile", "off") == "torch":
            st.warning("算子 trace 会明显拖慢采集窗口，只建议短时诊断；其余训练会继续正常运行。")
        st.selectbox(
            "梯度检查点模式（0关闭/1选择性/2整层）",
            [0, 1, 2],
            index=st.session_state.get("use_grad_checkpoint", 0),
            key="use_grad_checkpoint",
            help="Gradient checkpointing mode: 0 = off, 1 = selectively recompute "
                 "attention/FFN activations, 2 = full-layer checkpointing.",
        )
        st.selectbox(
            "参数精度 (param_dtype)",
            ["fp32", "bf16", "fp16"],
            index=["fp32", "bf16", "fp16"].index(st.session_state.get("param_dtype", "fp32")),
            key="param_dtype",
            help="模型参数精度：fp32=主权重(推荐)；bf16/fp16=训练时权重直接 cast",
        )
        st.selectbox(
            "激活层精度 (dtype)",
            ["bfloat16", "float16", "fp32"],
            index=["bfloat16", "float16", "fp32"].index(st.session_state.get("activation_dtype", "bfloat16")),
            key="activation_dtype",
            help="激活层计算精度：bfloat16(推荐) / float16 / fp32(纯精度，慢)",
        )
        st.selectbox(
            "TorchAO FP8 training",
            ["off", "tensorwise", "rowwise", "rowwise_with_gw_hp"],
            index=["off", "tensorwise", "rowwise", "rowwise_with_gw_hp"].index(
                st.session_state.get("fp8_training", "off")
            ),
            key="fp8_training",
            help="仅量化兼容 Linear 的前向/反向 GEMM，权重与优化器状态仍保持 BF16/FP32。"
                 "tensorwise 最快；rowwise 数值更稳健，但部分消费级 GPU 暂不支持。",
        )
        st.selectbox(
            "TorchAO FP8 Linear filter",
            ["auto", "eligible"],
            index=["auto", "eligible"].index(st.session_state.get("fp8_filter", "auto")),
            key="fp8_filter",
            disabled=st.session_state.get("fp8_training", "off") == "off",
            help="auto 跳过预计量化开销大于收益的小 GEMM；eligible 转换全部尺寸可被 FP8 支持的 Linear。",
        )
        if (
            st.session_state.get("fp8_training", "off") != "off"
            and st.session_state.get("activation_dtype", "bfloat16") != "bfloat16"
        ):
            st.error("TorchAO FP8 training 需要将激活层精度设为 bfloat16。")
        if st.session_state.get("fp8_training", "off").startswith("rowwise"):
            st.info("部分消费级 Blackwell GPU 暂不支持 TorchAO rowwise；启动时会先实测，不支持则降级为 tensorwise。")
        st.selectbox(
            "KV Cache 精度 (kv_cache_dtype)",
            ["fp32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2"],
            index=["fp32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2"].index(st.session_state.get("kv_cache_dtype", "fp32")),
            key="kv_cache_dtype",
            help="KV Cache 精度：fp8 量化缓存，decode 带宽减半(影响 RL rollouts 与推理)",
        )
        if st.button("Start Training", use_container_width=True, key="btn_start_train"):
            st.session_state.train_triggered = True
        if st.session_state.get("train_status") == "running":
            if st.button("⏸ Pause Training", use_container_width=True, key="btn_pause_train",
                         help="写入 checkpoints/.pause_request，训练进程在下一个 step 边界保存检查点并退出(码42)"):
                _request_pause(st.session_state.get("train_type", "pretrain"))
                st.rerun()
            log_path = st.session_state.get("train_log_path")
            if log_path and os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    log = f.read()
                pi = re.findall(r'Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\)', log)
                if pi:
                    ep, te, stp, tst = map(int, pi[-1])
                    pct = min(((ep - 1) + stp / tst) / te, 1.0)
                    st.progress(pct)
                    st.caption(f"Epoch {ep}/{te} — Step {stp}/{tst} ({pct * 100:.1f}%)")
                else:
                    st.info("Training in progress...")
            else:
                st.info("Training in progress...")
        elif st.session_state.get("train_status") == "success":
            st.success("Training completed")
        elif st.session_state.get("train_status") == "failed":
            st.error("Training process exited with an error")
            _failed_log_path = st.session_state.get("train_log_path")
            if _failed_log_path and os.path.exists(_failed_log_path):
                with open(_failed_log_path, "r", encoding="utf-8", errors="replace") as _failed_log:
                    _failed_tail = "".join(_failed_log.readlines()[-80:])
                if "out of memory" in _failed_tail.lower() or "CUBLAS_STATUS_INTERNAL_ERROR" in _failed_tail:
                    st.warning(
                        "检测到 CUDA OOM。请降低 batch_size；AttnRes + 1024 tokens "
                        "在 16 GB GPU 上建议从 batch_size=4 开始。"
                    )
                if "gradient tensor output of CUDAGraphs" in _failed_tail:
                    st.warning(
                        "检测到 CUDA Graph 梯度缓冲被覆盖。无需梯度累积时请将 "
                        "Gradient accumulation steps 设为 1；需要累积时请改用 default "
                        "或 max-autotune-no-cudagraphs。"
                    )
                with st.expander("Failure log (last 80 lines)", expanded=True):
                    st.code(_failed_tail, language="text")
        elif st.session_state.get("train_status") == "paused":
            st.warning("⏸ Training paused — model state saved. Tick 'Resume from checkpoint' and press Start Training to continue.")
            st.caption("Pause saves out/ + checkpoints/*_resume.pth so training can continue later.")

# ═══════════════════════════════════════════════════════════════
# ═══ MAIN AREA ═══
# ═══════════════════════════════════════════════════════════════

if st.session_state.get("train_triggered", False):
    st.session_state.train_triggered = False
    proc = st.session_state.get("train_proc")
    if proc is not None:
        if proc.poll() is None:
            st.session_state.train_status = "running"
        else:
            st.session_state.train_proc = None
    if st.session_state.get("train_proc") is None:
        _clear_pause_markers()  # 清掉残留的暂停标记/状态，冷启动与续训都从干净状态开始
        try:
            trainer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer")
            train_type = st.session_state.get("train_type", "pretrain")
            script_name = f"train_{train_type}.py"
            script_path = os.path.join(trainer_dir, script_name)
            if not os.path.exists(script_path):
                st.session_state.train_status = "failed"
            else:
                cfg = build_config_dict()
                config_path = os.path.join(trainer_dir, f"config_{train_type}.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                from_resume = st.session_state.get("from_resume", False)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                save_prefix = _resolve_save_prefix(train_type, cfg["hidden_size"], cfg["use_moe"],
                                                   from_resume, stamp)
                _launch_ok = True
                if from_resume and (save_prefix is None or not _resume_checkpoint_exists(
                        save_prefix, cfg["hidden_size"], cfg["use_moe"])):
                    # 续训找不到/缺失对应 *_resume.pth：不清静启动（绝不落到旧权重或新一轮），置 failed 并告警。
                    _launch_ok = False
                    st.session_state.train_status = "failed"
                    logs_dir = os.path.join(trainer_dir, "logs")
                    os.makedirs(logs_dir, exist_ok=True)
                    warn_path = os.path.join(logs_dir, f"train_{_default_weight_prefix(train_type)}.log")
                    warn_msg = (f"⚠️ Resume failed: no matching checkpoints/{train_type}_*_resume.pth "
                                f"for hidden_size={cfg['hidden_size']}. Untick 'Resume from checkpoint' to start "
                                "fresh, or tick it only when a paused/periodic resume checkpoint exists.")
                    with open(warn_path, "a", encoding="utf-8") as _log:
                        _log.write(warn_msg + "\n")
                    st.session_state.train_log_path = warn_path
                    st.warning(warn_msg + " — 本次未启动训练进程。")
                else:
                    st.session_state.pop("_resume_not_found", None)
                from_weight = st.session_state.get("base_weight", "auto")
                # selectbox 显示文案(见 _base_options)需映射回 CLI 值，否则会去加载 out/none (from scratch)_768.pth
                if from_weight == "none (from scratch)":
                    from_weight = "none"
                elif from_weight == "auto (newest matching base)":
                    from_weight = "auto"
                if from_weight == "auto":
                    base_type = _BASE_WEIGHT_TYPE.get(train_type, train_type)
                    auto_path = _latest_weight_file(base_type, cfg["hidden_size"], cfg["use_moe"], _arch_tag())
                    from_weight = auto_path if auto_path else "none"
                elif from_weight.startswith(("out/", "checkpoints/")):
                    from_weight = os.path.join(os.path.dirname(trainer_dir), from_weight)
                if cfg.get("model_architecture") == "linear":
                    # linear 架构必须经 run_linear.py 的 sys.modules 劫持启动，直接跑 trainer 会静默回退到标准 Transformer
                    runner = os.path.join(os.path.dirname(trainer_dir), "run_linear.py")
                    cmd = [sys.executable, "-u", runner, script_path]
                else:
                    cmd = [sys.executable, "-u", script_path]
                cmd.extend([
                    "--config_path", config_path,
                    "--hidden_size", str(cfg["hidden_size"]),
                    "--num_hidden_layers", str(cfg["num_hidden_layers"]),
                    "--use_moe", "1" if cfg["use_moe"] else "0",
                ])
                if train_type == "distillation":
                    if from_weight != "none":
                        cmd.extend(["--from_student_weight", from_weight])
                        cmd.extend(["--from_teacher_weight", from_weight])
                else:
                    cmd.extend(["--from_weight", from_weight])
                cmd.extend(["--batch_size", str(st.session_state.get("batch_size", 32))])
                cmd.extend(["--max_seq_len", str(st.session_state.get("max_seq_len", 768))])
                cmd.extend(["--sequence_packing", "1" if st.session_state.get("sequence_packing", False) else "0"])
                cmd.extend(["--packing_batch_size", str(st.session_state.get("packing_batch_size", 1000))])
                cmd.extend(["--accumulation_steps", str(st.session_state.get("accumulation_steps", 1))])
                cmd.extend(["--optimizer", st.session_state.get("optimizer", "adamw")])
                cmd.extend(["--dtype", st.session_state.get("activation_dtype", "bfloat16")])
                cmd.extend(["--param_dtype", st.session_state.get("param_dtype", "fp32")])
                cmd.extend(["--kv_cache_dtype", st.session_state.get("kv_cache_dtype", "fp32")])
                cmd.extend(["--fp8_training", st.session_state.get("fp8_training", "off")])
                cmd.extend(["--fp8_filter", st.session_state.get("fp8_filter", "auto")])
                cmd.extend(["--profile", st.session_state.get("profile", "off")])
                cmd.extend(["--profile_warmup", str(st.session_state.get("profile_warmup", 10))])
                cmd.extend(["--profile_interval", str(st.session_state.get("profile_interval", 100))])
                cmd.extend(["--profile_active_steps", str(st.session_state.get("profile_active_steps", 5))])
                if st.session_state.get("use_compile", True):
                    cmd.extend(["--use_compile", "1"])
                    cmd.extend(["--compile_mode", st.session_state.get("compile_mode", "reduce-overhead")])
                grad_ckpt = st.session_state.get("use_grad_checkpoint", 0)
                if grad_ckpt:
                    cmd.extend(["--use_grad_checkpoint", str(grad_ckpt)])
                if train_type == "lora":
                    cmd.extend(["--lora_name", save_prefix])
                else:
                    cmd.extend(["--save_weight", save_prefix])
                if cfg.get("model_architecture") == "looped":
                    cmd.extend(["--use_looped", "1"])
                if cfg.get("early_exit_layers") and st.session_state.get("early_exit_enabled", False):
                    cmd.extend(["--early_exit", "1"])
                if from_resume:
                    cmd.extend(["--from_resume", "1"])
                if train_type == "pretrain":
                    suffix = "_mini" if st.session_state.get("dataset_size", "mini") == "mini" else ""
                    cmd.extend(["--data_path", f"./dataset/pretrain_t2t{suffix}.jsonl"])
                elif train_type in ("full_sft", "distillation"):
                    suffix = "_mini" if st.session_state.get("dataset_size", "mini") == "mini" else ""
                    cmd.extend(["--data_path", f"./dataset/sft_t2t{suffix}.jsonl"])
                if _launch_ok:
                    logs_dir = os.path.join(trainer_dir, "logs")
                    os.makedirs(logs_dir, exist_ok=True)
                    log_path = os.path.join(logs_dir, f"train_{save_prefix}.log")
                    if not from_resume:
                        seq = 1
                        while os.path.exists(log_path):
                            seq += 1
                            log_path = os.path.join(logs_dir, f"train_{save_prefix}_{seq}.log")
                    log_file = open(log_path, "a" if from_resume else "w", encoding="utf-8")
                    log_file.write(f"# torch.compile (Triton): {'ON' if st.session_state.get('use_compile', True) else 'OFF'}\n")
                    log_file.write(f"# gradient accumulation steps: {st.session_state.get('accumulation_steps', 1)}\n")
                    log_file.write(
                        f"# Sequence packing: {'ON' if st.session_state.get('sequence_packing', False) else 'OFF'} "
                        f"(cache_batch={st.session_state.get('packing_batch_size', 1000)})\n"
                    )
                    log_file.write(
                        f"# TorchAO FP8 training: {st.session_state.get('fp8_training', 'off')} "
                        f"(filter={st.session_state.get('fp8_filter', 'auto')})\n"
                    )
                    log_file.write(
                        f"# Training profiler: {st.session_state.get('profile', 'off')} "
                        f"(warmup={st.session_state.get('profile_warmup', 10)}, "
                        f"interval={st.session_state.get('profile_interval', 100)}, "
                        f"trace_steps={st.session_state.get('profile_active_steps', 5)})\n"
                    )
                    log_file.flush()
                    st.session_state.train_log_path = log_path
                    st.session_state.train_proc = subprocess.Popen(
                        cmd,
                        cwd=os.path.dirname(trainer_dir),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONUTF8": "1"},  # Windows: torch.compile 需 UTF-8 模式，否则 gbk 解码崩溃
                    )
                    st.session_state.train_status = "running"
        except Exception:
            st.session_state.train_status = "failed"

if st.session_state.get("train_status") == "running":
    proc = st.session_state.get("train_proc")
    if proc is not None and proc.poll() is not None:
        rc = proc.poll()
        if rc == PAUSE_EXIT_CODE:
            st.session_state.train_status = "paused"
        elif rc == 0:
            st.session_state.train_status = "success"
        else:
            st.session_state.train_status = "failed"

cfg = build_config_dict()
breakdown = calc_params(cfg)
total_params = breakdown["Total Params"]["value"]

# ── Header row ──
name = st.session_state.preset
if name == "Custom":
    name = "Custom Model"

suffix = ""
if cfg["use_moe"]:
    active_total = breakdown["Active Total"]["value"]
    suffix = " (MoE)"
elif cfg.get("model_architecture") == "linear":
    active_total = total_params
    suffix = " (Linear)"
elif cfg.get("model_architecture") == "looped":
    active_total = total_params
    suffix = " (Looped)"
else:
    active_total = total_params
    suffix = ""
if cfg.get("residual_type", "standard") == "mhc":
    suffix = f"{suffix} + mHC" if suffix else " (mHC)"
elif cfg.get("residual_type", "standard") == "attnres":
    _attnres_label = f"{cfg.get('attnres_variant', 'block').title()} AttnRes"
    suffix = f"{suffix} + {_attnres_label}" if suffix else f" ({_attnres_label})"
name_label = f"{name} &nbsp;{suffix}" if suffix else name

col_title, col_badges = st.columns([1.2, 2])
with col_title:
    st.markdown(
        f'<div style="font-size: 26px; font-weight: 700; '
        f'letter-spacing: -0.5px; color: #f1f5f9;">{name_label}</div>',
        unsafe_allow_html=True,
    )

with col_badges:
    badge_html = (
        f'<span class="badge">{fmt_num(total_params)} params</span>'
    )
    if cfg["use_moe"]:
        badge_html += (
            f'<span class="badge badge-moe">{fmt_num(active_total)} active</span>'
        )
    badge_html += (
        f'<span class="badge badge-green">{cfg["num_hidden_layers"]} layers</span>'
    )
    badge_html += (
        f'<span class="badge">dim={cfg["hidden_size"]}</span>'
    )
    badge_html += (
        f'<span class="badge">vocab={cfg["vocab_size"]}</span>'
    )
    if cfg.get("residual_type", "standard") != "standard":
        badge_html += (
            f'<span class="badge badge-moe">residual={cfg["residual_type"]}</span>'
        )
    st.markdown(
        f'<div style="margin-top: 6px;">{badge_html}</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<hr style="margin: 4px 0 16px 0; border-color: #1e293b;">',
    unsafe_allow_html=True,
)

# ── Training progress bar (prominent, top of page) ──
if st.session_state.get("train_status") == "running":
    log_path = st.session_state.get("train_log_path")
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = f.read()
        pi = re.findall(r'Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\)', log)
        if pi:
            ep, te, stp, tst = map(int, pi[-1])
            pct = min(((ep - 1) + stp / tst) / te, 1.0)
            st.progress(pct, text=f"⏳ Training — Epoch {ep}/{te}, Step {stp}/{tst} ({pct*100:.1f}%)")
        else:
            st.info("⏳ Training in progress... (waiting for first epoch output)")
    else:
        st.info("⏳ Training started...")
elif st.session_state.get("train_status") == "paused":
    tt = st.session_state.get("train_type", "pretrain")
    cfg_local = build_config_dict()
    ckpt_prefix = _latest_checkpoint_prefix(tt, cfg_local["hidden_size"], cfg_local["use_moe"], _arch_tag())
    st.warning(f"⏸ Training paused. Resume checkpoint: checkpoints/{ckpt_prefix}_{cfg_local['hidden_size']}_resume.pth" if ckpt_prefix else "⏸ Training paused. Resume via 'Resume from checkpoint' + Start Training.")

# ── Main content (single column) ──

# ═══ Architecture diagram ──
st.markdown(
    '<div style="font-size: 13px; font-weight: 600; letter-spacing: 0.8px; '
    'color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;">'
    "Architecture</div>",
    unsafe_allow_html=True,
)

st.markdown(arch_diagram(cfg), unsafe_allow_html=True)

# Additional architecture info
int_size = cfg.get("intermediate_size", compute_intermediate_size(cfg["hidden_size"]))
st.markdown(
    f'<div style="margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px;">'
    f'<span class="badge" style="background: #1e293b; border: 1px solid #334155; '
    f'background: none; -webkit-text-fill-color: #e2e8f0; color: #e2e8f0;">'
    f"FFN intermediate: {fmt_num(int_size)}</span>"
    f'<span class="badge" style="background: #1e293b; border: 1px solid #334155; '
    f'background: none; -webkit-text-fill-color: #e2e8f0; color: #e2e8f0;">'
    f"head_dim: {cfg['head_dim']}</span>"
    f'<span class="badge" style="background: #1e293b; border: 1px solid #334155; '
    f'background: none; -webkit-text-fill-color: #e2e8f0; color: #e2e8f0;">'
    f"max_pos: {cfg['max_position_embeddings']}</span>"
    f'<span class="badge" style="background: #1e293b; border: 1px solid #334155; '
    f'background: none; -webkit-text-fill-color: #e2e8f0; color: #e2e8f0;">'
    f"rope_theta: {cfg['rope_theta']:.0e}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ═══ Breakdown + JSON + Code ═══
# ── Parameter Breakdown ──
st.markdown(
    '<div class="section-title">Parameter Breakdown</div>',
    unsafe_allow_html=True,
)

table_data = []
for name, info in breakdown.items():
    table_data.append(
        {
            "Component": name,
            "Params": fmt_num(info["value"]),
        }
    )
st.dataframe(
    table_data,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Component": st.column_config.TextColumn("Component", width="medium"),
        "Params": st.column_config.TextColumn("Parameters", width="small"),
    },
)

# Show percentage of total
pct_rows = []
for name, info in breakdown.items():
    if name in ("Total Params", "Active Total", "Layer Distribution"):
        continue
    val = info["value"]
    pct = (val / total_params * 100) if total_params > 0 else 0
    pct_rows.append(
        {
            "Component": name,
            "% of Total": f"{pct:.1f}%",
        }
    )
st.dataframe(
    pct_rows,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Component": st.column_config.TextColumn("Component", width="medium"),
        "% of Total": st.column_config.TextColumn("% of Total", width="small"),
    },
)

# ── Config JSON ──
st.markdown(
    '<div class="section-title">Config JSON</div>',
    unsafe_allow_html=True,
)

json_str = gen_config_json(cfg)
st.code(json_str, language="json", line_numbers=False)

# ── Python Code ──
st.markdown(
    '<div class="section-title">Python Code</div>',
    unsafe_allow_html=True,
)

py_code = gen_python_code(cfg)
st.code(py_code, language="python", line_numbers=False)

# ── Training Log ──
_auto_refresh_training_log = False
if st.session_state.get("train_status") == "running":
    st.markdown('<div class="section-title">📊 Training Monitor</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        _auto_refresh_training_log = st.checkbox(
            "Auto-refresh 2s", value=True, key="auto_refresh_log"
        )
    with c2:
        if st.button("Clear Log", key="btn_clear_train_log"):
            log_path = st.session_state.get("train_log_path")
            if log_path and os.path.exists(log_path):
                open(log_path, "w").close()
                st.rerun()
    with c3:
        current_log = st.session_state.get("train_log_path")
        st.caption(f"log: {os.path.basename(current_log) if current_log else '-'}")
    log_path = st.session_state.get("train_log_path")
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = f.read()
        metrics = parse_training_metrics(log)
        with st.expander("📄 Raw Log", expanded=len(metrics) == 0):
            st.code(log[-10000:] if len(log) > 10000 else log or "(empty)", language="text", line_numbers=False)
elif st.session_state.get("train_status") == "paused":
    tt = st.session_state.get("train_type", "pretrain")
    cfg_local = build_config_dict()
    ckpt_prefix = _latest_checkpoint_prefix(tt, cfg_local["hidden_size"], cfg_local["use_moe"], _arch_tag())
    st.warning(f"⏸ Training paused. Resume checkpoint: checkpoints/{ckpt_prefix}_{cfg_local['hidden_size']}_resume.pth" if ckpt_prefix else "⏸ Training paused. Resume via 'Resume from checkpoint' + Start Training.")
    st.caption("Pause saves out/ + checkpoints/*_resume.pth so training can continue later.")
elif st.session_state.get("train_status") == "success":
    st.success("Training completed successfully")
    log_path = st.session_state.get("train_log_path")
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = f.read()
        with st.expander("📄 Training Log", expanded=False):
            st.code(log[-5000:] if len(log) > 5000 else log, language="text", line_numbers=False)
elif st.session_state.get("train_status") == "failed":
    st.error("Training failed to start")
    log_path = st.session_state.get("train_log_path")
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            log = f.read()
        if log.strip():
            with st.expander("📄 Error Log", expanded=True):
                st.code(log[-5000:] if len(log) > 5000 else log, language="text", line_numbers=False)

# ── Training Data: exactly one historical log + the current log ──
trainer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer")
logs_dir = os.path.join(trainer_dir, "logs")
current_log_path = st.session_state.get("train_log_path")
if not current_log_path or not os.path.exists(current_log_path):
    current_log_path = None

log_candidates = []
if os.path.isdir(logs_dir):
    log_candidates.extend(
        os.path.join(logs_dir, filename)
        for filename in os.listdir(logs_dir)
        if filename.endswith(".log")
    )
legacy_log = os.path.join(trainer_dir, "train_output.log")
if os.path.exists(legacy_log):
    log_candidates.append(legacy_log)

current_log_key = (
    os.path.normcase(os.path.abspath(current_log_path)) if current_log_path else None
)
history_logs = sorted(
    (
        path for path in log_candidates
        if os.path.normcase(os.path.abspath(path)) != current_log_key
    ),
    key=os.path.getmtime,
    reverse=True,
)

# Failed/compile-only logs have no Epoch metrics and should not occupy the one
# historical comparison slot.
history_data = {}
for history_path in history_logs:
    with open(history_path, "r", encoding="utf-8", errors="replace") as history_file:
        history_content = history_file.read()
    history_metrics = parse_training_metrics(history_content)
    if history_metrics:
        history_data[history_path] = (history_content, history_metrics)

current_content, current_metrics = "", []
if current_log_path:
    with open(current_log_path, "r", encoding="utf-8", errors="replace") as current_file:
        current_content = current_file.read()
    current_metrics = parse_training_metrics(current_content)

if history_data or current_log_path:
    st.markdown('<div class="section-title">📊 Training Data</div>', unsafe_allow_html=True)
    history_column, current_column = st.columns(2)

    with history_column:
        st.markdown("#### Historical log")
        if history_data:
            if st.session_state.get("history_log_select") not in history_data:
                st.session_state.history_log_select = next(iter(history_data))
            selected_history = st.selectbox(
                "Compare against",
                list(history_data),
                format_func=os.path.basename,
                key="history_log_select",
            )
            historical_content, historical_metrics = history_data[selected_history]
            st.caption(os.path.basename(selected_history))
            render_log_metrics_charts(historical_metrics)
            with st.expander("📄 Historical raw log", expanded=False):
                st.code(
                    historical_content[-5000:] if len(historical_content) > 5000 else historical_content,
                    language="text", line_numbers=False,
                )
        else:
            st.info("No earlier log with training metrics is available.")

    with current_column:
        st.markdown("#### Current training")
        if current_log_path:
            st.caption(os.path.basename(current_log_path))
            render_log_metrics_charts(current_metrics)
        else:
            st.info("No current training log is selected.")

# Footer
st.markdown(
'<div style="margin-top: 24px; font-size: 11px; color: #475569; '
'text-align: center; border-top: 1px solid #1e293b; padding-top: 12px;">'
"Instinct Config WebUI &mdash; standalone, no model weights required"
"</div>",
unsafe_allow_html=True,
)

# 每次交互后持久化面板参数（widget 变更触发 rerun，此时 session_state 即当前值）
_trainer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainer")
_persist_panel_state(_trainer_dir)

if _auto_refresh_training_log:
    time.sleep(2)
    st.rerun()
