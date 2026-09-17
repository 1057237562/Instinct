"""Instinct V2 latent recurrent-depth Transformer.

This backbone follows Geiping et al., *Scaling up Test-Time Compute with Latent
Reasoning* (arXiv:2502.05171):

    tokens -> Prelude -> [state || embedded input] adapter
           -> shared multi-layer recurrent core x r -> Coda -> LM head

The recurrent state starts from truncated Gaussian noise.  During training one
depth is sampled for the complete micro-batch from a log-normal Poisson
distribution and only the final ``mean_backprop_depth`` recurrences retain an
autograd graph.  At inference ``num_steps`` directly scales latent compute.

The public Instinct config/model/generation interfaces stay compatible with the
other backbones so all trainers can select this implementation with
``--use_looped 1`` or ``model_architecture=looped``.
"""
import math
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from model.flash_attn_4 import flash_attention
from model.kv_cache_quant import parse_cache, make_cache
from model.checkpointing import recompute_attention, checkpoint_ffn
from model.attention_mask import apply_attention_mask
from model.sequence_packing import merge_packed_attention_mask, positions_from_sequence_ids
from model.model_instinct import AttentionResidual, ManifoldHyperConnection, ManifoldHyperHead

# ═══════════════════════════════════════════════════════════════
# Instinct Loop Config
# ═══════════════════════════════════════════════════════════════
class InstinctConfig(PretrainedConfig):
    """Configuration for the Instinct V2 recurrent-depth backbone."""
    model_type = "instinct"
    def __init__(self, hidden_size: int = 768, num_hidden_layers: int = 8, use_moe: bool = False, **kwargs):
        """初始化配置:全部可选字段经 kwargs 传入。"""
        saved_rope_scaling = kwargs.pop("rope_scaling", None)
        saved_rope_parameters = kwargs.pop("rope_parameters", None)
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.model_architecture = kwargs.get("model_architecture", "looped")
        self.recurrent_architecture_version = 2
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.param_dtype = kwargs.get("param_dtype", "fp32")
        self.kv_cache_dtype = kwargs.get("kv_cache_dtype", "fp32")
        # Gradient Checkpointing configs
        self.use_grad_checkpoint = kwargs.get("use_grad_checkpoint", 0)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        if self.num_attention_heads < 1 or self.num_key_value_heads < 1:
            raise ValueError("attention head counts must be >= 1")
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads must not exceed num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.initializer_range = kwargs.get(
            "initializer_range", math.sqrt(2.0 / (5.0 * hidden_size))
        )
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        default_rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.rope_scaling = saved_rope_scaling or saved_rope_parameters or default_rope_scaling
        # MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        self.residual_type = kwargs.get("residual_type", "standard")
        if self.residual_type not in {"standard", "mhc", "attnres"}:
            raise ValueError("residual_type must be one of: standard, mhc, attnres")
        self.hc_mult = int(kwargs.get("hc_mult", 4))
        self.hc_sinkhorn_iters = int(kwargs.get("hc_sinkhorn_iters", 20))
        self.hc_eps = float(kwargs.get("hc_eps", 1e-6))
        self.mhc_init_std = float(kwargs.get("mhc_init_std", 0.02))
        if self.hc_mult < 1 or self.hc_sinkhorn_iters < 1:
            raise ValueError("hc_mult and hc_sinkhorn_iters must be >= 1")
        self.attnres_variant = kwargs.get("attnres_variant", "block")
        if self.attnres_variant not in {"full", "block"}:
            raise ValueError("attnres_variant must be one of: full, block")
        # Latent recurrent-depth settings.  ``loop_iters`` remains the public
        # compatibility alias for the paper's mean recurrence r-bar.
        default_side = 2 if num_hidden_layers >= 8 else 1
        self.prelude_layers = int(kwargs.get("prelude_layers", default_side))
        self.coda_layers = int(kwargs.get("coda_layers", default_side))
        default_recurrent = max(
            1, num_hidden_layers - self.prelude_layers - self.coda_layers
        )
        self.recurrent_layers = int(kwargs.get(
            "recurrent_layers",
            kwargs.get("recurrent_block_layers", default_recurrent),
        ))
        self.loop_iters = int(kwargs.get(
            "loop_iters", kwargs.get("mean_recurrence", 8)
        ))
        self.mean_recurrence = self.loop_iters
        self.mean_backprop_depth = int(kwargs.get(
            "mean_backprop_depth", min(4, self.loop_iters)
        ))
        self.recurrence_sampling = kwargs.get(
            "recurrence_sampling", "lognormal_poisson"
        )
        self.recurrence_log_normal_sigma = float(kwargs.get(
            "recurrence_log_normal_sigma", 1.0
        ))
        self.max_recurrence = int(kwargs.get(
            "max_recurrence", max(self.loop_iters, 4 * self.loop_iters)
        ))
        self.state_init_std = float(kwargs.get(
            "state_init_std", math.sqrt(2.0 / 5.0)
        ))
        self.embedding_scale = float(kwargs.get(
            "embedding_scale", math.sqrt(hidden_size)
        ))
        self.use_input_injection = bool(kwargs.get("use_input_injection", True))
        # Huginn uses learnable Q/K biases and no per-head QK normalization.
        self.qk_bias = bool(kwargs.get("qk_bias", True))
        self.qk_norm = bool(kwargs.get("qk_norm", False))
        if min(self.prelude_layers, self.coda_layers) < 0:
            raise ValueError("prelude_layers and coda_layers must be >= 0")
        if self.recurrent_layers < 1 or self.loop_iters < 1:
            raise ValueError("recurrent_layers and loop_iters must be >= 1")
        if not 1 <= self.mean_backprop_depth <= self.max_recurrence:
            raise ValueError("mean_backprop_depth must be in [1, max_recurrence]")
        if self.max_recurrence < self.loop_iters:
            raise ValueError("max_recurrence must be >= loop_iters")
        if self.recurrence_sampling not in {"lognormal_poisson", "fixed"}:
            raise ValueError("recurrence_sampling must be lognormal_poisson or fixed")
        if self.recurrence_log_normal_sigma < 0 or self.state_init_std < 0:
            raise ValueError("recurrence sigma and state_init_std must be non-negative")
        default_block_size = max(1, math.ceil(2 * (
            self.prelude_layers
            + self.loop_iters * self.recurrent_layers
            + self.coda_layers
        ) / 8))
        self.attnres_block_size = int(kwargs.get("attnres_block_size", default_block_size))
        if self.attnres_block_size < 1:
            raise ValueError("attnres_block_size must be >= 1")

# ═══════════════════════════════════════════════════════════════
# Instinct Loop Model
# ═══════════════════════════════════════════════════════════════
class RMSNorm(torch.nn.Module):
    """RMS 归一化层:对最后一维做均方根归一化,再乘可学习权重逐元素缩放。

    计算在 fp32 下进行（与 HF 实现一致）,避免低精度数值不稳定。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        """初始化归一化权重（全 1）与 eps。"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        """归一化核心:y = x / sqrt(mean(x^2) + eps)。"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """前向:归一化后乘权重,并恢复输入 dtype。"""
        if not self.training and getattr(self, '_inference_norm', None) is not None:
            return self._inference_norm(x, self.weight, self.eps)
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """预计算 RoPE 的 cos/sin 频率表,返回 (freqs_cos, freqs_sin),形状 (end, dim)。

    rope_scaling 非空时应用 YaRN 缩放:f'(i) = f(i)((1-γ) + γ/s),γ 为线性 ramp。
    """
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """对 q/k 施加旋转位置编码（rotate_half 拼接实现）。"""
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    if cos.ndim == 3:
        unsqueeze_dim = 2
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA 头扩展:把 KV 头沿新维重复 n_rep 次并展平,使其与 Q 头数一致。"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: InstinctConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.qk_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.qk_bias,
        )
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.kv_cache_dtype = getattr(config, 'kv_cache_dtype', 'fp32')
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
        self.use_grad_checkpoint = getattr(config, "use_grad_checkpoint", 0)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            k_past, v_past = parse_cache(past_key_value)
            k_past = k_past.to(xq.dtype)
            v_past = v_past.to(xq.dtype)
            xk = torch.cat([k_past, xk], dim=1)
            xv = torch.cat([v_past, xv], dim=1)
        past_kv = make_cache(xk, xv, self.kv_cache_dtype) if use_cache else None
        if self.flash and not self.training and seq_len == 1:
            output = flash_attention(xq, xk, xv, dropout_p=0.0,
                                     is_causal=False, attention_mask=attention_mask)
            output = output.reshape(bsz, seq_len, -1)
        elif (self.flash and (seq_len > 1)
                and (not self.is_causal or past_key_value is None)):
            # Keep packed attention fused; mode 1 checkpoints the FFN only.
            output = flash_attention(
                xq, xk, xv, dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal, attention_mask=attention_mask,
            )
            output = output.reshape(bsz, seq_len, -1)
        else:
            if self.use_grad_checkpoint == 1 and self.training and past_key_value is None:
                # Mode 1: selective attention recompute — only pre-transpose Q/K/V is kept,
                # QK^T/softmax/dropout/@V are recomputed in backward to save O(seq^2) memory
                output = recompute_attention(xq, xk, xv, attention_mask, self.is_causal, self.attn_dropout.p, self.head_dim)
            else:
                xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
                scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
                if attention_mask is not None: scores = apply_attention_mask(scores, attention_mask)
                output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
                output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: InstinctConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        if not self.training and getattr(self, '_inference_gate', None) is not None:
            return self.down_proj(self._inference_gate(gate, up))
        return self.down_proj(self.act_fn(gate) * up)

class MOEFeedForward(nn.Module):
    def __init__(self, config: InstinctConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class InstinctBlock(nn.Module):
    """Transformer layer with the paper's four-norm sandwich layout."""
    def __init__(self, layer_id: int, config: InstinctConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_residual_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_mlp_residual_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.use_grad_checkpoint = getattr(config, "use_grad_checkpoint", 0)
        self.layer_id = layer_id
        self.residual_type = config.residual_type
        if self.residual_type == "mhc":
            self.attn_hc = ManifoldHyperConnection(config)
            self.ffn_hc = ManifoldHyperConnection(config)
        elif self.residual_type == "attnres":
            self.attn_residual = AttentionResidual(config)
            self.mlp_residual = AttentionResidual(config)
            self.attnres_block_size = config.attnres_block_size

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        if self.residual_type == "mhc":
            return self._forward_mhc(
                hidden_states, position_embeddings, past_key_value, use_cache, attention_mask
            )
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states = self.post_attention_residual_layernorm(
            hidden_states + residual
        )
        normed = self.post_attention_layernorm(hidden_states)
        if self.use_grad_checkpoint == 1 and self.training:
            ffn_out, aux = checkpoint_ffn(self.mlp, normed)
            if aux is not None:
                self.mlp.aux_loss = aux
            hidden_states = self.post_mlp_residual_layernorm(
                hidden_states + ffn_out
            )
        else:
            hidden_states = self.post_mlp_residual_layernorm(
                hidden_states + self.mlp(normed)
            )
        return hidden_states, present_key_value

    def _run_mlp(self, hidden_states):
        if self.use_grad_checkpoint == 1 and self.training:
            ffn_out, aux = checkpoint_ffn(self.mlp, hidden_states)
            if aux is not None:
                self.mlp.aux_loss = aux
            return ffn_out
        return self.mlp(hidden_states)

    def _forward_mhc(self, hidden_states, position_embeddings, past_key_value=None,
                     use_cache=False, attention_mask=None):
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(collapsed), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states = self.attn_hc.merge(hidden_states, attn_output, post, comb)
        hidden_states = self.post_attention_residual_layernorm(hidden_states)
        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self._run_mlp(self.post_attention_layernorm(collapsed))
        hidden_states = self.ffn_hc.merge(hidden_states, mlp_output, post, comb)
        return self.post_mlp_residual_layernorm(hidden_states), present_key_value

    def forward_attnres_full(self, source_bank, position_embeddings, past_key_value=None,
                             use_cache=False, attention_mask=None, depth_index=None,
                             input_injection=None):
        hidden_states = self.attn_residual(source_bank)
        if input_injection is not None:
            hidden_states = hidden_states + input_injection
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        source_bank = torch.cat((source_bank, attn_output.unsqueeze(0)), dim=0)
        hidden_states = self.mlp_residual(source_bank)
        mlp_output = self._run_mlp(self.post_attention_layernorm(hidden_states))
        return torch.cat((source_bank, mlp_output.unsqueeze(0)), dim=0), present_key_value

    def _block_residual(self, source_bank, partial_count, residual):
        partial = source_bank[-1] + residual
        if partial_count + 1 == self.attnres_block_size:
            return torch.cat((source_bank[:-1], partial.unsqueeze(0), torch.zeros_like(partial).unsqueeze(0)), dim=0)
        return torch.cat((source_bank[:-1], partial.unsqueeze(0)), dim=0)

    @staticmethod
    def _block_sources(source_bank, partial_count):
        return source_bank[:-1] if partial_count == 0 else source_bank

    def forward_attnres_block(self, source_bank, position_embeddings, past_key_value=None,
                              use_cache=False, attention_mask=None, depth_index=None,
                              input_injection=None):
        depth_index = self.layer_id if depth_index is None else depth_index
        partial_count = (2 * depth_index) % self.attnres_block_size
        hidden_states = self.attn_residual(self._block_sources(source_bank, partial_count))
        if input_injection is not None:
            hidden_states = hidden_states + input_injection
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        source_bank = self._block_residual(source_bank, partial_count, attn_output)
        partial_count = (partial_count + 1) % self.attnres_block_size
        hidden_states = self.mlp_residual(self._block_sources(source_bank, partial_count))
        mlp_output = self._run_mlp(self.post_attention_layernorm(hidden_states))
        return self._block_residual(source_bank, partial_count, mlp_output), present_key_value

class _LegacyInstinctLoopModel(nn.Module):
    """
    Loop Transformer model with prelude → shared loop block → coda architecture.

    Architecture:
      Embedding → Prelude (unique blocks, once) → Loop (1 shared block × N iters) → Coda (unique blocks, once) → Norm → Output

    Each loop iteration injects the frozen prelude output (input injection) and a learnable
    loop-position embedding to maintain stability and provide temporal awareness.

    Config parameters:
      prelude_layers: number of unique blocks before the loop (default 1)
      loop_iters: number of times the shared block is applied (default 8)
      coda_layers: number of unique blocks after the loop (default 1)
      use_input_injection: whether to add frozen prelude output each iteration (default True)

    Total effective depth = prelude_layers + loop_iters + coda_layers
    Unique transformer blocks = prelude_layers + 1 + coda_layers
    """
    def __init__(self, config: InstinctConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.prelude_layers = config.prelude_layers
        self.loop_iters = config.loop_iters
        self.coda_layers = config.coda_layers
        self.total_effective_layers = self.prelude_layers + self.loop_iters + self.coda_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Prelude: unique transformer blocks (run once, freeze output for injection)
        self.prelude = nn.ModuleList([
            InstinctBlock(l, config) for l in range(self.prelude_layers)
        ]) if self.prelude_layers > 0 else nn.ModuleList()

        # Loop: one shared transformer block, applied loop_iters times
        self.loop_block = InstinctBlock(0, config) if self.loop_iters > 0 else None

        # Coda: unique transformer blocks (run once after loop)
        self.coda = nn.ModuleList([
            InstinctBlock(self.prelude_layers + 1 + l, config)
            for l in range(self.coda_layers)
        ]) if self.coda_layers > 0 else nn.ModuleList()

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_type = config.residual_type
        if self.residual_type == "mhc":
            self.hc_head = ManifoldHyperHead(config)
        elif self.residual_type == "attnres":
            self.output_residual = AttentionResidual(config)

        # Learnable loop-position embedding (initialized to zeros for gradual activation)
        if self.loop_iters > 0:
            self.loop_pos_embed = nn.Embedding(self.loop_iters, config.hidden_size)
            nn.init.zeros_(self.loop_pos_embed.weight)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, end=config.max_position_embeddings,
            rope_base=config.rope_theta, rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    @property
    def layers(self):
        """Return all unique blocks for compatibility with parameter counting / checkpoint helpers."""
        blocks = list(self.prelude)
        if self.loop_block is not None:
            blocks.append(self.loop_block)
        blocks.extend(self.coda)
        return blocks

    def _run_block(self, layer, hidden_states, position_embeddings, past_key_value,
                   use_cache, attention_mask, depth_index, input_injection=None):
        if self.residual_type != "attnres":
            return layer(
                hidden_states, position_embeddings, past_key_value=past_key_value,
                use_cache=use_cache, attention_mask=attention_mask
            )
        layer_forward = (layer.forward_attnres_full if self.config.attnres_variant == "full"
                         else layer.forward_attnres_block)
        return layer_forward(
            hidden_states, position_embeddings, past_key_value=past_key_value,
            use_cache=use_cache, attention_mask=attention_mask,
            depth_index=depth_index, input_injection=input_injection
        )

    def _readout(self, hidden_states, completed_layers):
        if self.residual_type == "mhc":
            return self.hc_head(hidden_states)
        if self.residual_type != "attnres":
            return hidden_states
        if self.config.attnres_variant == "block":
            partial_count = (2 * completed_layers) % self.config.attnres_block_size
            hidden_states = hidden_states[:-1] if partial_count == 0 else hidden_states
        return self.output_residual(hidden_states)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                return_intermediate=False, layer_callback=None, sequence_ids=None,
                position_ids=None, **kwargs):
        batch_size, seq_length = input_ids.shape

        # Normalize past_key_values: must match total effective layers (each loop iter = 1 slot)
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * self.total_effective_layers

        # Determine start position for RoPE slicing (from first available KV cache entry)
        start_pos = 0
        for pkv in past_key_values:
            if pkv is not None:
                start_pos = pkv[0].shape[1]
                break

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if self.residual_type == "mhc":
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        elif self.residual_type == "attnres":
            if self.config.attnres_variant == "full":
                hidden_states = hidden_states.unsqueeze(0)
            else:
                hidden_states = torch.stack((hidden_states, torch.zeros_like(hidden_states)), dim=0)

        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim, end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling
            )
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)

        if sequence_ids is not None:
            if any(pkv is not None for pkv in past_key_values):
                raise ValueError("sequence_ids cannot be used together with a KV cache")
            if position_ids is None:
                position_ids = positions_from_sequence_ids(sequence_ids)
            attention_mask = merge_packed_attention_mask(sequence_ids, attention_mask)
        if position_ids is None:
            position_embeddings = (
                self.freqs_cos[start_pos:start_pos + seq_length],
                self.freqs_sin[start_pos:start_pos + seq_length]
            )
        else:
            position_embeddings = (self.freqs_cos[position_ids], self.freqs_sin[position_ids])

        presents = []
        intermediates = []
        kv_idx = 0

        # ── Prelude ──
        for layer in self.prelude:
            hidden_states, present = self._run_block(
                layer, hidden_states, position_embeddings, past_key_values[kv_idx],
                use_cache, attention_mask, depth_index=kv_idx
            )
            presents.append(present)
            kv_idx += 1
            readout = self._readout(hidden_states, kv_idx)
            if layer_callback is not None:
                layer_callback(kv_idx, self.norm(readout))
            if return_intermediate:
                intermediates.append(self.norm(readout))

        # ── Freeze prelude output for input injection ──
        frozen_input = None
        if self.config.use_input_injection:
            frozen_input = (self._readout(hidden_states, kv_idx)
                            if self.residual_type == "attnres" else hidden_states)

        # ── Loop ──
        if self.loop_iters > 0:
            loop_device = hidden_states.device
            for t in range(self.loop_iters):
                # Inject learnable loop-position signal
                loop_signal = self.loop_pos_embed(torch.tensor([t], device=loop_device))
                input_injection = None
                if self.residual_type == "attnres":
                    input_injection = loop_signal
                    if self.config.use_input_injection:
                        input_injection = input_injection + frozen_input
                    updated = hidden_states
                else:
                    updated = hidden_states + loop_signal
                    if self.config.use_input_injection:
                        updated = updated + frozen_input

                if self.config.use_grad_checkpoint == 2 and self.training:
                    # Mode 2: checkpoint the whole shared loop_block (the old loop_grad_checkpoint
                    # promise); MoE aux_loss returns through the closure so its gradient reaches the router
                    def _loop_blk(h, pkv, depth_index=kv_idx, injection=input_injection):
                        hs, present = self._run_block(
                            self.loop_block, h, position_embeddings, pkv,
                            use_cache, attention_mask, depth_index=depth_index,
                            input_injection=injection
                        )
                        aux = self.loop_block.mlp.aux_loss if isinstance(self.loop_block.mlp, MOEFeedForward) else None
                        return hs, present, aux
                    hidden_states, present, loop_aux = torch.utils.checkpoint.checkpoint(
                        _loop_blk, updated, past_key_values[kv_idx], use_reentrant=False, preserve_rng_state=True)
                    if loop_aux is not None:
                        self.loop_block.mlp.aux_loss = loop_aux
                else:
                    hidden_states, present = self._run_block(
                        self.loop_block, updated, position_embeddings, past_key_values[kv_idx],
                        use_cache, attention_mask, depth_index=kv_idx,
                        input_injection=input_injection
                    )
                presents.append(present)
                kv_idx += 1
                readout = self._readout(hidden_states, kv_idx)
                if layer_callback is not None:
                    layer_callback(kv_idx, self.norm(readout))
                if return_intermediate:
                    intermediates.append(self.norm(readout))

        # ── Coda ──
        for layer in self.coda:
            hidden_states, present = self._run_block(
                layer, hidden_states, position_embeddings, past_key_values[kv_idx],
                use_cache, attention_mask, depth_index=kv_idx
            )
            presents.append(present)
            kv_idx += 1
            readout = self._readout(hidden_states, kv_idx)
            if layer_callback is not None:
                layer_callback(kv_idx, self.norm(readout))
            if return_intermediate:
                intermediates.append(self.norm(readout))

        hidden_states = self.norm(self._readout(hidden_states, kv_idx))

        # Collect aux_loss from MoE layers
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )
        if return_intermediate:
            return hidden_states, presents, aux_loss, intermediates
        return hidden_states, presents, aux_loss

class InstinctLoopModel(nn.Module):
    """Prelude -> recurrent latent core -> coda implementation for Instinct V2."""

    def __init__(self, config: InstinctConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.prelude_layers = config.prelude_layers
        self.recurrent_layers = config.recurrent_layers
        self.loop_iters = config.loop_iters
        self.coda_layers = config.coda_layers
        self.total_effective_layers = self.effective_layers(self.loop_iters)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.prelude = nn.ModuleList(
            InstinctBlock(i, config) for i in range(self.prelude_layers)
        )
        self.input_adapter = (
            nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
            if config.use_input_injection else nn.Identity()
        )
        self.recurrent_block = nn.ModuleList(
            InstinctBlock(self.prelude_layers + i, config)
            for i in range(self.recurrent_layers)
        )
        coda_offset = self.prelude_layers + self.recurrent_layers * self.loop_iters
        self.coda = nn.ModuleList(
            InstinctBlock(coda_offset + i, config)
            for i in range(self.coda_layers)
        )
        self.recurrent_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_type = config.residual_type
        if self.residual_type == "mhc":
            self.hc_head = ManifoldHyperHead(config)
        elif self.residual_type == "attnres":
            self.output_residual = AttentionResidual(config)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.last_num_steps = self.loop_iters
        self.last_num_steps_no_grad = 0
        self.last_num_steps_with_grad = self.loop_iters

    def effective_layers(self, num_steps):
        return (
            self.prelude_layers
            + int(num_steps) * self.recurrent_layers
            + self.coda_layers
        )

    @property
    def loop_block(self):
        """Compatibility alias for code that inspected the former one-layer core."""
        return self.recurrent_block[0]

    @property
    def layers(self):
        """All materialized (unique-parameter) transformer layers."""
        return list(self.prelude) + list(self.recurrent_block) + list(self.coda)

    def _run_block(self, layer, hidden_states, position_embeddings, past_key_value,
                   use_cache, attention_mask, depth_index):
        if self.residual_type != "attnres":
            return layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
        layer_forward = (
            layer.forward_attnres_full
            if self.config.attnres_variant == "full"
            else layer.forward_attnres_block
        )
        return layer_forward(
            hidden_states,
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            depth_index=depth_index,
        )

    def _readout(self, hidden_states, completed_layers):
        if self.residual_type == "mhc":
            return self.hc_head(hidden_states)
        if self.residual_type != "attnres":
            return hidden_states
        if self.config.attnres_variant == "block":
            partial_count = (2 * completed_layers) % self.config.attnres_block_size
            hidden_states = hidden_states[:-1] if partial_count == 0 else hidden_states
        return self.output_residual(hidden_states)

    def _sample_total_steps(self, device):
        if self.config.recurrence_sampling == "fixed":
            sampled = self.loop_iters
        else:
            sampled_tensor = torch.zeros(1, dtype=torch.long, device=device)
            should_sample = not dist.is_initialized() or dist.get_rank() == 0
            if should_sample:
                sigma = self.config.recurrence_log_normal_sigma
                mu = math.log(max(float(self.loop_iters), 1.0)) - 0.5 * sigma**2
                rate = torch.empty(1, device=device).log_normal_(mean=mu, std=sigma)
                sampled_tensor.copy_(torch.poisson(rate).to(torch.long) + 1)
            if dist.is_initialized():
                dist.broadcast(sampled_tensor, src=0)
            sampled = int(sampled_tensor.item())
        return max(1, min(sampled, self.config.max_recurrence))

    def _resolve_steps(self, num_steps, device):
        if num_steps is None:
            total = self._sample_total_steps(device) if self.training else self.loop_iters
            with_grad = (
                min(self.config.mean_backprop_depth, total)
                if self.training else total
            )
            return total - with_grad, with_grad
        if isinstance(num_steps, torch.Tensor):
            values = num_steps.flatten().tolist()
        elif isinstance(num_steps, (tuple, list)):
            values = list(num_steps)
        else:
            values = [num_steps]
        if len(values) == 2:
            no_grad, with_grad = int(values[0]), int(values[1])
            if no_grad < 0 or with_grad < 0 or no_grad + with_grad < 1:
                raise ValueError(
                    "num_steps pair must contain non-negative values with positive sum"
                )
            return no_grad, with_grad
        total = int(values[0])
        if not 1 <= total <= self.config.max_recurrence:
            raise ValueError(
                f"num_steps must be in [1, {self.config.max_recurrence}], got {total}"
            )
        with_grad = (
            min(self.config.mean_backprop_depth, total)
            if self.training else total
        )
        return total - with_grad, with_grad

    def _initialize_state(self, embedded_input):
        if not self.config.use_input_injection:
            return embedded_input.clone()
        std = self.config.state_init_std
        state = torch.empty_like(embedded_input)
        if std == 0:
            return state.zero_()
        return nn.init.trunc_normal_(
            state, mean=0.0, std=std, a=-3 * std, b=3 * std
        )

    def _inject_input(self, state, embedded_input, completed_layers):
        if not self.config.use_input_injection:
            return state
        if self.residual_type != "attnres":
            return self.input_adapter(torch.cat((state, embedded_input), dim=-1))
        # Attention Residuals maintains a source bank.  Apply the paper adapter
        # to its readout, then inject the delta into the bank.  The standard
        # residual path above is the exact architecture from the paper.
        readout = self._readout(state, completed_layers)
        embedded_readout = self._readout(embedded_input, self.prelude_layers)
        adapted = self.input_adapter(
            torch.cat((readout, embedded_readout), dim=-1)
        )
        return state + (adapted - readout).unsqueeze(0)

    def _core_iteration(self, state, embedded_input, position_embeddings,
                        past_key_values, use_cache, attention_mask, kv_idx):
        state = self._inject_input(state, embedded_input, kv_idx)
        presents = []
        aux_loss = state.new_zeros(1).squeeze()
        for offset, layer in enumerate(self.recurrent_block):
            state, present = self._run_block(
                layer,
                state,
                position_embeddings,
                past_key_values[offset],
                use_cache,
                attention_mask,
                depth_index=kv_idx + offset,
            )
            presents.append(present)
            if isinstance(layer.mlp, MOEFeedForward):
                aux_loss = aux_loss + layer.mlp.aux_loss
        return state, presents, aux_loss

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, return_intermediate=False, layer_callback=None,
                sequence_ids=None, position_ids=None, num_steps=None,
                input_states=None, **kwargs):
        _, seq_length = input_ids.shape
        no_grad_steps, grad_steps = self._resolve_steps(num_steps, input_ids.device)
        total_steps = no_grad_steps + grad_steps
        self.last_num_steps = total_steps
        self.last_num_steps_no_grad = no_grad_steps
        self.last_num_steps_with_grad = grad_steps
        expected_cache_slots = self.effective_layers(total_steps)

        if hasattr(past_key_values, "layers"):
            past_key_values = None
        if past_key_values is None:
            past_key_values = [None] * expected_cache_slots
        elif len(past_key_values) != expected_cache_slots:
            raise ValueError(
                "KV cache depth does not match num_steps: "
                f"expected {expected_cache_slots}, got {len(past_key_values)}"
            )

        start_pos = 0
        for pkv in past_key_values:
            if pkv is not None:
                start_pos = pkv[0].shape[1]
                break

        hidden_states = (
            self.dropout(self.embed_tokens(input_ids)) * self.config.embedding_scale
        )
        if self.residual_type == "mhc":
            hidden_states = hidden_states.unsqueeze(2).expand(
                -1, -1, self.config.hc_mult, -1
            ).contiguous()
        elif self.residual_type == "attnres":
            if self.config.attnres_variant == "full":
                hidden_states = hidden_states.unsqueeze(0)
            else:
                hidden_states = torch.stack(
                    (hidden_states, torch.zeros_like(hidden_states)), dim=0
                )

        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.freqs_cos = freqs_cos.to(hidden_states.device)
            self.freqs_sin = freqs_sin.to(hidden_states.device)

        if sequence_ids is not None:
            if any(pkv is not None for pkv in past_key_values):
                raise ValueError("sequence_ids cannot be used together with a KV cache")
            if position_ids is None:
                position_ids = positions_from_sequence_ids(sequence_ids)
            attention_mask = merge_packed_attention_mask(sequence_ids, attention_mask)
        if position_ids is None:
            position_embeddings = (
                self.freqs_cos[start_pos:start_pos + seq_length],
                self.freqs_sin[start_pos:start_pos + seq_length],
            )
        else:
            position_embeddings = (
                self.freqs_cos[position_ids], self.freqs_sin[position_ids]
            )

        presents, intermediates = [], []
        aux_loss = hidden_states.new_zeros(1).squeeze()
        kv_idx = 0

        def record(state, completed_layers):
            if layer_callback is None and not return_intermediate:
                return
            readout = self.norm(self._readout(state, completed_layers))
            if layer_callback is not None:
                layer_callback(completed_layers, readout)
            if return_intermediate:
                intermediates.append(readout)

        for layer in self.prelude:
            hidden_states, present = self._run_block(
                layer,
                hidden_states,
                position_embeddings,
                past_key_values[kv_idx],
                use_cache,
                attention_mask,
                depth_index=kv_idx,
            )
            presents.append(present)
            if isinstance(layer.mlp, MOEFeedForward):
                aux_loss = aux_loss + layer.mlp.aux_loss
            kv_idx += 1
            record(hidden_states, kv_idx)

        embedded_input = hidden_states
        state = (
            self._initialize_state(embedded_input)
            if input_states is None else input_states.clone()
        )

        with torch.no_grad():
            for _ in range(no_grad_steps):
                state, iteration_presents, _ = self._core_iteration(
                    state,
                    embedded_input,
                    position_embeddings,
                    past_key_values[kv_idx:kv_idx + self.recurrent_layers],
                    use_cache,
                    attention_mask,
                    kv_idx,
                )
                presents.extend(iteration_presents)
                kv_idx += self.recurrent_layers
                record(state, kv_idx)

        for _ in range(grad_steps):
            cache_slice = past_key_values[kv_idx:kv_idx + self.recurrent_layers]
            can_checkpoint = (
                self.config.use_grad_checkpoint == 2
                and self.training
                and not use_cache
                and all(cache is None for cache in cache_slice)
            )
            if can_checkpoint:
                depth_index = kv_idx

                def checkpointed_iteration(current_state, recurrent_input):
                    output, _, iteration_aux = self._core_iteration(
                        current_state,
                        recurrent_input,
                        position_embeddings,
                        [None] * self.recurrent_layers,
                        False,
                        attention_mask,
                        depth_index,
                    )
                    return output, iteration_aux

                state, iteration_aux = torch.utils.checkpoint.checkpoint(
                    checkpointed_iteration,
                    state,
                    embedded_input,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                iteration_presents = [None] * self.recurrent_layers
            else:
                state, iteration_presents, iteration_aux = self._core_iteration(
                    state,
                    embedded_input,
                    position_embeddings,
                    cache_slice,
                    use_cache,
                    attention_mask,
                    kv_idx,
                )
            aux_loss = aux_loss + iteration_aux
            presents.extend(iteration_presents)
            kv_idx += self.recurrent_layers
            record(state, kv_idx)

        hidden_states = self.recurrent_norm(state)
        for layer in self.coda:
            hidden_states, present = self._run_block(
                layer,
                hidden_states,
                position_embeddings,
                past_key_values[kv_idx],
                use_cache,
                attention_mask,
                depth_index=kv_idx,
            )
            presents.append(present)
            if isinstance(layer.mlp, MOEFeedForward):
                aux_loss = aux_loss + layer.mlp.aux_loss
            kv_idx += 1
            record(hidden_states, kv_idx)

        hidden_states = self.norm(self._readout(hidden_states, kv_idx))
        if return_intermediate:
            return hidden_states, presents, aux_loss, intermediates
        return hidden_states, presents, aux_loss


def _compute_lm_loss(logits: torch.Tensor, labels):
    """Next-token cross-entropy over the last hidden state; None when labels are absent."""
    if labels is None:
        return None
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    return F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

class InstinctForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = InstinctConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def _init_weights(self, module):
        """Takase-style truncated initialization used by the recurrent model."""
        if isinstance(module, nn.Linear):
            std = self.config.initializer_range
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            std = self.config.initializer_range
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif "RMSNorm" in module.__class__.__name__:
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)


    def __init__(self, config: InstinctConfig = None):
        self.config = config or InstinctConfig()
        super().__init__(self.config)
        self.model = InstinctLoopModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()
        effective_depth = (
            self.config.prelude_layers
            + self.config.loop_iters * self.config.recurrent_layers
            + self.config.coda_layers
        )
        output_std = math.sqrt(
            1.0 / (5.0 * self.config.hidden_size * effective_depth)
        )
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(
                ("self_attn.o_proj", "mlp.down_proj")
            ):
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=output_std,
                    a=-3 * output_std,
                    b=3 * output_std,
                )

    def load_pretrained_weights(self, weights):
        """Load V2 weights or migrate Dense/legacy-loop weights as initialization.

        Dense layers are partitioned in order into Prelude, recurrent core and
        Coda.  A former one-layer LoopUS core is copied into every layer of the
        new recurrent block.  New adapter/sandwich-norm parameters retain their
        paper initialization.
        """
        state = dict(weights)
        mapped = {}
        dense_prefix = "model.layers."
        has_dense_layers = any(key.startswith(dense_prefix) for key in state)
        has_legacy_loop = any(key.startswith("model.loop_block.") for key in state)
        if has_dense_layers:
            layer_targets = (
                [f"model.prelude.{i}" for i in range(self.config.prelude_layers)]
                + [f"model.recurrent_block.{i}" for i in range(self.config.recurrent_layers)]
                + [f"model.coda.{i}" for i in range(self.config.coda_layers)]
            )
            for key, value in state.items():
                if not key.startswith(dense_prefix):
                    mapped[key] = value
                    continue
                suffix = key[len(dense_prefix):]
                index_text, separator, parameter = suffix.partition(".")
                if separator and index_text.isdigit():
                    index = int(index_text)
                    if index < len(layer_targets):
                        mapped[f"{layer_targets[index]}.{parameter}"] = value
        elif has_legacy_loop:
            for key, value in state.items():
                if key.startswith("model.loop_pos_embed."):
                    continue
                if key.startswith("model.loop_block."):
                    suffix = key[len("model.loop_block."):]
                    for index in range(self.config.recurrent_layers):
                        mapped[f"model.recurrent_block.{index}.{suffix}"] = value
                else:
                    mapped[key] = value
        else:
            mapped = state

        incompatible = self.load_state_dict(mapped, strict=False)
        allowed_missing_fragments = (
            "input_adapter", "recurrent_norm",
            "post_attention_residual_layernorm", "post_mlp_residual_layernorm",
            "self_attn.q_proj.bias", "self_attn.k_proj.bias",
        )
        significant_missing = [
            key for key in incompatible.missing_keys
            if not any(fragment in key for fragment in allowed_missing_fragments)
        ]
        significant_unexpected = [
            key for key in incompatible.unexpected_keys
            if "loop_pos_embed" not in key
            and "self_attn.q_norm.weight" not in key
            and "self_attn.k_norm.weight" not in key
        ]
        if significant_missing or significant_unexpected:
            warnings.warn(
                "Partial Instinct V2 weight migration; missing="
                f"{significant_missing[:8]}, unexpected={significant_unexpected[:8]}",
                RuntimeWarning,
            )
        return incompatible

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, labels=None, logit_lens=False, **kwargs):
        def _compute_logits(hidden_states):
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            return self.lm_head(hidden_states[:, slice_indices, :])

        # Logit lens: unembed each effective layer's normed hidden state via the shared LM head (inference-only probing, no training)
        if logit_lens:
            hidden_states, past_key_values, aux_loss, intermediates = self.model(
                input_ids, attention_mask, past_key_values, use_cache,
                return_intermediate=True, **kwargs
            )
            layer_logits = [_compute_logits(h) for h in intermediates]
            logits = _compute_logits(hidden_states)
            loss = _compute_lm_loss(logits, labels)
            output = MoeCausalLMOutputWithPast(
                loss=loss, aux_loss=aux_loss, logits=logits,
                past_key_values=past_key_values, hidden_states=hidden_states
            )
            output.layer_logits = layer_logits  # list[Tensor]: one [bs, seq, vocab] per effective layer
            output.recurrent_steps = self.model.last_num_steps
            return output

        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )
        logits = _compute_logits(hidden_states)
        loss = _compute_lm_loss(logits, labels)
        output = MoeCausalLMOutputWithPast(
            loss=loss, aux_loss=aux_loss, logits=logits,
            past_key_values=past_key_values, hidden_states=hidden_states
        )
        output.recurrent_steps = self.model.last_num_steps
        return output

    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85,
                 top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True,
                 num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        initial_length = input_ids.shape[1]
        input_storage = input_ids.new_empty((input_ids.shape[0], initial_length + max_new_tokens))
        input_storage[:, :initial_length].copy_(input_ids)
        input_ids = input_storage[:, :initial_length]
        attention_storage = None
        if attention_mask is not None:
            attention_storage = attention_mask.new_ones((attention_mask.shape[0], initial_length + max_new_tokens))
            attention_storage[:, :initial_length].copy_(attention_mask)
            attention_mask = attention_storage[:, :initial_length]
        past_key_values = kwargs.pop("past_key_values", None)
        from model.generation_stream import TokenChunkBuffer
        chunk_size = int(kwargs.pop('stream_chunk_size', 16))
        # Cache-returning callers need the cache aligned to the exact stop token.
        if kwargs.get('return_kv'):
            chunk_size = 1
        chunks = TokenChunkBuffer(streamer, chunk_size, eos_token_id)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())
        forward_kwargs = {'logits_to_keep': 1, **kwargs}
        for step in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(
                input_ids[:, past_len:], attention_mask, past_key_values,
                use_cache=use_cache, **forward_kwargs
            )
            attention_mask = attention_storage[:, :initial_length + step + 1] if attention_storage is not None else None
            logits = outputs.logits[:, -1, :]
            if do_sample:
                logits = logits / temperature
            if repetition_penalty != 1.0:
                seen = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, input_ids, True)
                penalized = torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
                logits = torch.where(seen, penalized, logits)
            if do_sample and top_k > 0:
                logits.masked_fill_(logits < torch.topk(logits, top_k)[0][..., -1, None], -float('inf'))
            if do_sample and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits.masked_fill_(mask.scatter(1, sorted_indices, mask), -float('inf'))
            next_token = (
                torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token
                )
            input_storage[:, initial_length + step:initial_length + step + 1].copy_(next_token)
            input_ids = input_storage[:, :initial_length + step + 1]
            past_key_values = outputs.past_key_values if use_cache else None
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
            if chunks.push(next_token, finished, final=step + 1 == max_new_tokens):
                break
        if chunks.overshoot:
            input_ids = input_ids[:, :-chunks.overshoot]
        if streamer:
            streamer.end()
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
