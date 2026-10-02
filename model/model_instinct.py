"""
Instinct 模型定义:配置类(InstinctConfig)与 Dense Transformer 主干。

包含 RMSNorm、RoPE(支持 YaRN / LongRoPE 外推)、GQA 注意力(flash / math / 梯度检查点
三路实现)、SwiGLU FFN、MoE 路由、mHC / Attention Residuals 可选残差拓扑、
Early Exit / logit lens,以及带完整采样参数(temperature / top_k / top_p /
repetition_penalty)的自定义 generate 循环。
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from model.flash_attn_4 import flash_attention
from model.kv_cache_quant import is_fp8, make_cache, parse_cache
from model.static_cache import (
    DecodeState,
    StaticKVCache,
    StaticKVCacheLayer,
    STATIC_CACHE_INITIAL_DECODE_TOKENS,
    bucket_capacity,
    layer_specs,
)
from model.checkpointing import recompute_attention, checkpoint_ffn
from model.attention_mask import apply_attention_mask, prepare_sdpa_attention_bias
from model.packed_attention import maybe_prepare_flex_mask
from model.sequence_packing import (
    merge_packed_attention_mask, positions_from_sequence_ids,
)
from model.rope import (
    build_rope_caches, precompute_freqs_cis, select_rope_cache,
    validate_rope_scaling,
)
from model.moe_dispatch import refresh_inference_stacks, routed_moe_forward

# ═══════════════════════════════════════════════════════════════
# InstinctConfig
# ═══════════════════════════════════════════════════════════════
class InstinctConfig(PretrainedConfig):
    """模型超参配置(对齐 Qwen3 生态)。kwargs 中未给出的字段取默认值:
    vocab 6400、8 层、dim 768、GQA 8/4 头、SwiGLU、RoPE θ=1e6、max_pos 32768;
    use_moe=True 时启用 MoE 路由相关字段。"""
    model_type = "instinct"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        # Transformers 5 standardizes legacy ``rope_scaling`` during the base
        # config init, before old-style custom configs have assigned model
        # dimensions. Consume both serialized spellings here and restore the
        # value after our dimensions exist.
        saved_rope_scaling = kwargs.pop("rope_scaling", None)
        saved_rope_parameters = kwargs.pop("rope_parameters", None)
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.model_architecture = kwargs.get("model_architecture", "standard")
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
        self.initializer_range = kwargs.get("initializer_range", 0.02)
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
        self.rope_scaling = validate_rope_scaling(
            saved_rope_scaling or saved_rope_parameters or default_rope_scaling,
            self.head_dim,
            self.max_position_embeddings,
        )
        # Early Exit configs (LayerSkip-style: shared LM head, no auxiliary classifiers)
        self.early_exit_layers = kwargs.get("early_exit_layers", [4, 5, 6, 7])
        self.early_exit_loss_weight = kwargs.get("early_exit_loss_weight", 0.3)
        # MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        # Residual topology. ``standard`` keeps old configs/checkpoints exact.
        self.residual_type = kwargs.get("residual_type", "standard")
        if self.residual_type not in {"standard", "mhc", "attnres"}:
            raise ValueError("residual_type must be one of: standard, mhc, attnres")
        self.hc_mult = int(kwargs.get("hc_mult", 4))
        self.hc_sinkhorn_iters = int(kwargs.get("hc_sinkhorn_iters", 20))
        self.hc_eps = float(kwargs.get("hc_eps", 1e-6))
        self.mhc_init_std = float(kwargs.get("mhc_init_std", 0.02))
        if self.hc_mult < 1:
            raise ValueError("hc_mult must be >= 1")
        if self.hc_sinkhorn_iters < 1:
            raise ValueError("hc_sinkhorn_iters must be >= 1")
        self.attnres_variant = kwargs.get("attnres_variant", "block")
        if self.attnres_variant not in {"full", "block"}:
            raise ValueError("attnres_variant must be one of: full, block")
        default_block_size = max(1, math.ceil(2 * self.num_hidden_layers / 8))
        self.attnres_block_size = int(kwargs.get("attnres_block_size", default_block_size))
        if self.attnres_block_size < 1:
            raise ValueError("attnres_block_size must be >= 1")

# ═══════════════════════════════════════════════════════════════
# Instinct Model
# ═══════════════════════════════════════════════════════════════
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        """RMS 归一化核心:x / sqrt(mean(x²) + eps),按最后一维计算。"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """在 fp32 下归一化避免低精度下平方和溢出,再乘权重并恢复原 dtype。"""
        if not self.training and getattr(self, '_inference_norm', None) is not None:
            return self._inference_norm(x, self.weight, self.eps)
        return (self.weight * self.norm(x.float())).type_as(x)


class UnweightedRMSNorm(nn.Module):
    """Parameter-free RMSNorm used to normalize residual-routing keys."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x_float = x.float()
        return (x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class AttentionResidual(nn.Module):
    """Depth-wise softmax attention over ``[depth, batch, seq, hidden]`` sources."""

    def __init__(self, config: InstinctConfig):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(config.hidden_size))
        self.key_norm = UnweightedRMSNorm(config.rms_norm_eps)

    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        keys = self.key_norm(sources)
        logits = torch.einsum("d,nbtd->nbt", self.query.float(), keys.float())
        weights = torch.softmax(logits, dim=0).to(sources.dtype)
        return torch.einsum("nbt,nbtd->btd", weights, sources)


class ManifoldHyperConnection(nn.Module):
    """Pure-PyTorch mHC pre/post mixer with a Sinkhorn manifold projection."""

    def __init__(self, config: InstinctConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = UnweightedRMSNorm(config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.zeros(mix))
        self.scale = nn.Parameter(torch.ones(3))
        nn.init.normal_(self.fn, mean=0.0, std=config.mhc_init_std)

    def forward(self, hidden_streams: torch.Tensor):
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2)).float()
        logits = F.linear(flat, self.fn.float())
        pre_w, post_w, comb_w = logits.split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale
        comb_logits = comb_logits + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)
        return post, comb, collapsed.to(hidden_streams.dtype)

    @staticmethod
    def merge(hidden_streams, sublayer_output, post, comb):
        dtype = hidden_streams.dtype
        mixed = torch.matmul(comb.to(dtype).transpose(-1, -2), hidden_streams)
        return mixed + post.to(dtype).unsqueeze(-1) * sublayer_output.unsqueeze(-2)


class ManifoldHyperHead(nn.Module):
    """Collapse the final mHC stream bundle back to one hidden sequence."""

    def __init__(self, config: InstinctConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.input_norm = UnweightedRMSNorm(config.rms_norm_eps)
        self.fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.zeros(self.hc_mult))
        self.scale = nn.Parameter(torch.ones(1))
        nn.init.normal_(self.fn, mean=0.0, std=config.mhc_init_std)

    def forward(self, hidden_streams):
        flat = self.input_norm(hidden_streams.flatten(start_dim=2)).float()
        logits = F.linear(flat, self.fn.float())
        pre = torch.sigmoid(logits * self.scale.float() + self.base.float()) + self.hc_eps
        return (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    if cos.ndim == 3:
        unsqueeze_dim = 2
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
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
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.kv_cache_dtype = getattr(config, 'kv_cache_dtype', 'fp32')
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
        self.use_grad_checkpoint = getattr(config, "use_grad_checkpoint", 0)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None,
                cache_positions=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        static = isinstance(past_key_value, StaticKVCacheLayer)
        if static:
            # Preallocated buffer: this step's slice is written in place and the
            # whole buffer is attended through the mask the trunk built, so the
            # shapes never change and the step stays capturable.
            xk, xv = past_key_value.append(xk, xv, cache_positions)
            past_kv = past_key_value if use_cache else None
        else:
            if past_key_value is not None:
                k_past, v_past = parse_cache(past_key_value)
                k_past = k_past.to(xq.dtype)
                v_past = v_past.to(xq.dtype)
                xk = torch.cat([k_past, xk], dim=1)
                xv = torch.cat([v_past, xv], dim=1)
            past_kv = make_cache(xk, xv, self.kv_cache_dtype) if use_cache else None
        if self.flash and not self.training and static:
            # The mask carries causality already (see StaticKVCache.mask_for).
            output = flash_attention(xq, xk, xv, dropout_p=0.0,
                                     is_causal=False, attention_mask=attention_mask)
            output = output.reshape(bsz, seq_len, -1)
        elif self.flash and not self.training and seq_len == 1:
            # A decode query can attend to every cached key; a top-left causal
            # mask would incorrectly restrict it to the first cached position.
            output = flash_attention(xq, xk, xv, dropout_p=0.0,
                                     is_causal=False, attention_mask=attention_mask)
            output = output.reshape(bsz, seq_len, -1)
        elif (self.flash and (seq_len > 1)
                and (not self.is_causal or past_key_value is None)):
            # Packed masks stay on memory-efficient SDPA. Mode 1 checkpoints
            # the FFN only on this fast path; recomputing attention here was
            # slower and caused torch.compile to decompose SDPA into dense BMMs.
            output = flash_attention(
                xq, xk, xv, dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal, attention_mask=attention_mask,
            )
            output = output.reshape(bsz, seq_len, -1)
        else:
            if self.use_grad_checkpoint == 1 and self.training and past_key_value is None:
                # Mode 1: recompute attention core in backward; needs pre-transpose q/k/v and q/k same-seq (no cache).
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
        # Detached, tiny [num_experts] snapshot consumed only at log intervals.
        # It is deliberately not a buffer, so checkpoints remain unchanged.
        self.router_load = None
        # Grouped-GEMM backend and prebuilt weight stacks for the traceable
        # inference dispatch, installed by optimize_inference. None keeps the
        # training/autograd forward.
        self._inference_backend = None
        self._inference_stacked = None

    def forward(self, x):
        y, scores, topk_idx = routed_moe_forward(
            x, self.gate, self.experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
            norm_topk_prob=self.config.norm_topk_prob,
            act_fn=self.act_fn,
            inference_backend=self._inference_backend,
            inference_stacks=self._inference_stacked,
        )
        if self.training:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.router_load = load.mean(dim=0).detach()
            if self.config.router_aux_loss_coef > 0:
                self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
            else:
                self.aux_loss = scores.new_zeros(1).squeeze()
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y

class InstinctBlock(nn.Module):
    def __init__(self, layer_id: int, config: InstinctConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
            # Cache the modulo offset rather than reading layer_id in forward.
            # torch.compile otherwise specializes on every layer_id and hits its
            # recompilation limit on models deeper than eight layers.
            self.attnres_partial_count = (2 * layer_id) % self.attnres_block_size

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False,
                attention_mask=None, cache_positions=None):
        if self.residual_type == "mhc":
            return self._forward_mhc(
                hidden_states, position_embeddings, past_key_value, use_cache, attention_mask,
                cache_positions,
            )
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask, cache_positions
        )
        hidden_states = hidden_states + residual
        normed = self.post_attention_layernorm(hidden_states)
        if self.use_grad_checkpoint == 1 and self.training:
            ffn_out, aux = checkpoint_ffn(self.mlp, normed)
            if aux is not None:
                self.mlp.aux_loss = aux   # returned graph node replaces the no_grad side-channel attribute
            hidden_states = hidden_states + ffn_out
        else:
            hidden_states = hidden_states + self.mlp(normed)
        return hidden_states, present_key_value

    def _run_mlp(self, hidden_states):
        if self.use_grad_checkpoint == 1 and self.training:
            ffn_out, aux = checkpoint_ffn(self.mlp, hidden_states)
            if aux is not None:
                self.mlp.aux_loss = aux
            return ffn_out
        return self.mlp(hidden_states)

    def _forward_mhc(self, hidden_states, position_embeddings, past_key_value=None,
                     use_cache=False, attention_mask=None, cache_positions=None):
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(collapsed), position_embeddings,
            past_key_value, use_cache, attention_mask, cache_positions
        )
        hidden_states = self.attn_hc.merge(hidden_states, attn_output, post, comb)
        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self._run_mlp(self.post_attention_layernorm(collapsed))
        hidden_states = self.ffn_hc.merge(hidden_states, mlp_output, post, comb)
        return hidden_states, present_key_value

    def forward_attnres_full(self, source_bank, position_embeddings, past_key_value=None,
                             use_cache=False, attention_mask=None, cache_positions=None):
        hidden_states = self.attn_residual(source_bank)
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask, cache_positions
        )
        source_bank = torch.cat((source_bank, attn_output.unsqueeze(0)), dim=0)
        hidden_states = self.mlp_residual(source_bank)
        mlp_output = self._run_mlp(self.post_attention_layernorm(hidden_states))
        source_bank = torch.cat((source_bank, mlp_output.unsqueeze(0)), dim=0)
        return source_bank, present_key_value

    def _block_residual(self, source_bank, partial_count, residual):
        partial = source_bank[-1] + residual
        if partial_count + 1 == self.attnres_block_size:
            return torch.cat((source_bank[:-1], partial.unsqueeze(0), torch.zeros_like(partial).unsqueeze(0)), dim=0)
        return torch.cat((source_bank[:-1], partial.unsqueeze(0)), dim=0)

    @staticmethod
    def _block_sources(source_bank, partial_count):
        return source_bank[:-1] if partial_count == 0 else source_bank

    def forward_attnres_block(self, source_bank, position_embeddings, past_key_value=None,
                              use_cache=False, attention_mask=None, cache_positions=None):
        partial_count = self.attnres_partial_count
        hidden_states = self.attn_residual(self._block_sources(source_bank, partial_count))
        attn_output, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask, cache_positions
        )
        source_bank = self._block_residual(source_bank, partial_count, attn_output)
        partial_count = (partial_count + 1) % self.attnres_block_size
        hidden_states = self.mlp_residual(self._block_sources(source_bank, partial_count))
        mlp_output = self._run_mlp(self.post_attention_layernorm(hidden_states))
        source_bank = self._block_residual(source_bank, partial_count, mlp_output)
        return source_bank, present_key_value

class InstinctModel(nn.Module):
    def __init__(self, config: InstinctConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([InstinctBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_type = config.residual_type
        if self.residual_type == "mhc":
            self.hc_head = ManifoldHyperHead(config)
        elif self.residual_type == "attnres":
            self.output_residual = AttentionResidual(config)
        freqs_cos, freqs_sin, short_cos, short_sin = build_rope_caches(
            config.head_dim, config.max_position_embeddings,
            config.rope_theta, config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.register_buffer("freqs_cos_short", short_cos, persistent=False)
        self.register_buffer("freqs_sin_short", short_sin, persistent=False)
        self._rope_ready = True

    def ensure_rope_caches(self, device=None) -> bool:
        """Rebuild the RoPE caches if they were lost during meta-device init.

        A Python flag rather than ``freqs_cos[0, 0] == 0``: a tensor comparison
        used as a branch condition is a data-dependent jump, which costs one
        graph break per compiled forward. Inference also calls this before
        compiling (see ``optimize_inference``), so the check settles once.
        """
        if self._rope_ready and self.freqs_cos[0, 0] != 0:
            return False
        freqs_cos, freqs_sin, short_cos, short_sin = build_rope_caches(
            self.config.head_dim, self.config.max_position_embeddings,
            self.config.rope_theta, self.config.rope_scaling,
        )
        target = device if device is not None else self.freqs_cos.device
        self.freqs_cos, self.freqs_sin = freqs_cos.to(target), freqs_sin.to(target)
        if short_cos is not None:
            self.freqs_cos_short = short_cos.to(target)
            self.freqs_sin_short = short_sin.to(target)
        self._rope_ready = True
        return True

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                sequence_ids=None, position_ids=None, **kwargs):
        batch_size, seq_length = input_ids.shape
        if isinstance(past_key_values, StaticKVCache):
            # Preallocated buffer: the slots to write are the logical positions,
            # so they must come from the caller instead of from cache shapes.
            if position_ids is None:
                raise ValueError("a StaticKVCache needs explicit position_ids")
            start_pos = 0
            # The cache owns the mask: it hides the buffer tail and, on a
            # prompt, everything after each query. Building it here keeps every
            # layer's SDPA the same shape on every step.
            attention_mask = past_key_values.mask_for(position_ids)
        else:
            if hasattr(past_key_values, 'layers'):
                past_key_values = None  # legacy HF cache objects are not ours
            past_key_values = past_key_values or [None] * len(self.layers)
            start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if self.residual_type == "mhc":
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        elif self.residual_type == "attnres":
            if self.config.attnres_variant == "full":
                hidden_states = hidden_states.unsqueeze(0)
            else:
                # [embedding, completed blocks..., current partial]
                hidden_states = torch.stack((hidden_states, torch.zeros_like(hidden_states)), dim=0)
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if not self._rope_ready:
            self.ensure_rope_caches(hidden_states.device)
        if sequence_ids is not None:
            if past_key_values[0] is not None:
                raise ValueError("sequence_ids cannot be used together with a KV cache")
            if position_ids is None:
                position_ids = positions_from_sequence_ids(sequence_ids)
            flex_mask = maybe_prepare_flex_mask(
                sequence_ids, hidden_states, attention_mask=attention_mask,
                enabled=self.config.flash_attn,
                dropout_p=self.config.dropout if self.training else 0.0,
                head_dim=self.config.head_dim,
            )
            if flex_mask is not None:
                attention_mask = flex_mask
            else:
                attention_mask = merge_packed_attention_mask(sequence_ids, attention_mask)
                if self.config.flash_attn and seq_length > 1:
                    attention_mask = prepare_sdpa_attention_bias(
                        attention_mask, hidden_states, query_length=seq_length,
                    )
        position_embeddings = select_rope_cache(
            self.freqs_cos, self.freqs_sin,
            self.freqs_cos_short, self.freqs_sin_short,
            self.config.rope_scaling,
            start_pos=start_pos, seq_length=seq_length,
            position_ids=position_ids,
        )
        presents, intermediates = [], []
        aux_loss = hidden_states.new_zeros(1).squeeze()
        return_intermediate = kwargs.pop("return_intermediate", False)
        exit_check_fn = kwargs.pop("exit_check_fn", None)
        layer_callback = kwargs.pop("layer_callback", None)
        for layer, past_key_value in zip(self.layers, past_key_values):
            layer_forward = layer
            if self.residual_type == "attnres":
                layer_forward = (layer.forward_attnres_full if self.config.attnres_variant == "full"
                                 else layer.forward_attnres_block)
            if self.config.use_grad_checkpoint == 2 and self.training:
                # Pass the module directly, not a closure over it: torch's non-reentrant
                # checkpoint mis-differentiates parameterized closures composed over 2+
                # layers (wrong grads in the first block). The MoE aux_loss attribute
                # stays graph-connected (grad-enabled forward) and is recompute-regenerated.
                hidden_states, present = torch.utils.checkpoint.checkpoint(
                    layer_forward, hidden_states, position_embeddings, past_key_value,
                    use_cache, attention_mask,
                    use_reentrant=False, preserve_rng_state=True)
            else:
                hidden_states, present = layer_forward(
                    hidden_states,
                    position_embeddings,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    cache_positions=position_ids
                )
            presents.append(present)
            if isinstance(layer.mlp, MOEFeedForward):
                aux_loss = aux_loss + layer.mlp.aux_loss
            readout = self._readout(hidden_states, len(presents))
            if layer_callback is not None:
                # Per-layer streaming hook (logit lens etc.): fires with the normed
                # hidden state right after each layer computes it.
                layer_callback(len(presents), self.norm(readout))
            if return_intermediate:
                intermediates.append(self.norm(readout))
            if exit_check_fn is not None:
                normed = self.norm(readout)
                if exit_check_fn(len(presents), normed):
                    presents.extend([None] * (len(self.layers) - len(presents)))
                    if return_intermediate:
                        return normed, presents, aux_loss, intermediates
                    return normed, presents, aux_loss
        hidden_states = self.norm(self._readout(hidden_states, len(presents), final=True))
        if return_intermediate:
            return hidden_states, presents, aux_loss, intermediates
        return hidden_states, presents, aux_loss

    def _readout(self, hidden_states, completed_layers, final=False):
        if self.residual_type == "mhc":
            return self.hc_head(hidden_states)
        if self.residual_type != "attnres":
            return hidden_states
        if self.config.attnres_variant == "block":
            partial_count = (2 * completed_layers) % self.config.attnres_block_size
            hidden_states = hidden_states[:-1] if partial_count == 0 else hidden_states
        return self.output_residual(hidden_states)

class InstinctForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = InstinctConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def _init_weights(self, module):
        """Keep Instinct's initialization stable across Transformers releases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif "RMSNorm" in module.__class__.__name__:
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)

    def __init__(self, config: InstinctConfig = None):
        self.config = config or InstinctConfig()
        super().__init__(self.config)
        self.model = InstinctModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        # Decode resources reused across generate() calls, keyed by capacity.
        # Populated lazily by _decode_state_for once optimize_inference allowed it.
        self._decode_states = {}
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, labels=None, early_exit=False, logit_lens=False, **kwargs):
        def _cross_entropy_loss(logits, labels):
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            return F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        def _compute_logits(hidden_states):
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            return self.lm_head(hidden_states[:, slice_indices, :])

        # Refresh the inference MoE weight stacks here, in eager Python: the
        # version-counter checks that keep the stacks valid cannot run inside the
        # compiled trunk (see moe_dispatch.refresh_inference_stacks).
        stack_modules = getattr(self, '_inference_stack_modules', None)
        if stack_modules is not None and not torch.compiler.is_compiling():
            refresh_inference_stacks(stack_modules)

        # Logit lens: unembed each layer's normed hidden state via the shared LM head (inference-only probing, no training)
        if logit_lens:
            hidden_states, pkv, aux_loss, intermediates = self.model(
                input_ids, attention_mask, past_key_values, use_cache,
                return_intermediate=True, **kwargs
            )
            layer_logits = [_compute_logits(h) for h in intermediates]
            logits = _compute_logits(hidden_states)
            loss = _cross_entropy_loss(logits, labels) if labels is not None else None
            output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                               past_key_values=pkv, hidden_states=hidden_states)
            output.layer_logits = layer_logits  # list[Tensor]: one [bs, seq, vocab] per layer
            return output

        # Early exit training: add CE loss at intermediate layers via shared LM head
        if early_exit and labels is not None:
            hidden_states, pkv, aux_loss, intermediates = self.model(
                input_ids, attention_mask, past_key_values, use_cache,
                return_intermediate=True, **kwargs
            )
            logits = _compute_logits(hidden_states)
            loss = _cross_entropy_loss(logits, labels)

            ee_losses = []
            for i, h in enumerate(intermediates):
                if (i + 1) in self.config.early_exit_layers:
                    ee_logits = _compute_logits(h)
                    ee_losses.append(_cross_entropy_loss(ee_logits, labels))

            if ee_losses:
                loss = loss + self.config.early_exit_loss_weight * sum(ee_losses) / len(ee_losses)

            return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                             past_key_values=pkv, hidden_states=hidden_states)

        if early_exit and labels is None:
            exit_threshold = kwargs.pop("exit_threshold", 0.9)
            def _ee_check(layer_idx, normed_h):
                if layer_idx not in self.config.early_exit_layers:
                    return False
                logits = _compute_logits(normed_h)
                return F.softmax(logits.float(), dim=-1).max(dim=-1).values.mean().item() >= exit_threshold
            hidden_states, pkv, aux_loss = self.model(
                input_ids, attention_mask, past_key_values, use_cache,
                exit_check_fn=_ee_check, **kwargs
            )
            logits = _compute_logits(hidden_states)
            loss = _cross_entropy_loss(logits, labels) if labels is not None else None
            return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                             past_key_values=pkv, hidden_states=hidden_states)

        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        logits = _compute_logits(hidden_states)
        loss = _cross_entropy_loss(logits, labels) if labels is not None else None
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

    # https://github.com/1057237562/Instinct/discussions/611
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
        early_exit = kwargs.pop("early_exit", False)
        exit_threshold = kwargs.pop("exit_threshold", 0.9)
        # Generation metadata, not a Transformer input. The native loop uses
        # EOS for finished rows; padding is represented by attention_mask.
        # Leaving this in forward_kwargs silently disables CUDA graph replay
        # for HF-compatible callers such as the Chat WebUI.
        kwargs.pop("pad_token_id", None)
        from model.generation_stream import TokenChunkBuffer
        chunk_size = int(kwargs.pop('stream_chunk_size', 16))
        # Cache-returning callers need the cache aligned to the exact stop token.
        if kwargs.get('return_kv'):
            chunk_size = 1
        chunks = TokenChunkBuffer(streamer, chunk_size, eos_token_id)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        forward_kwargs = dict(kwargs)
        forward_kwargs.setdefault('logits_to_keep', 1)
        if early_exit:
            forward_kwargs['early_exit'] = True
            forward_kwargs['exit_threshold'] = exit_threshold
        sample = dict(temperature=temperature, top_p=top_p, top_k=top_k, do_sample=do_sample,
                      repetition_penalty=repetition_penalty, eos_token_id=eos_token_id)
        # max_new_tokens is an output limit, not a prediction that every answer
        # will reach it.  Planning the static cache for the whole allowance
        # makes every attention step scan thousands of untouched slots (the
        # WebUI permits 16K).  Start small and grow at a bucket boundary while
        # preserving the original output limit.
        decode_headroom = max(1, int(getattr(
            self, '_static_cache_decode_headroom', STATIC_CACHE_INITIAL_DECODE_TOKENS)))
        planned_new_tokens = min(max_new_tokens, decode_headroom)
        state = self._decode_state_for(input_ids, attention_mask, planned_new_tokens,
                                      use_cache=use_cache, early_exit=early_exit, kwargs=kwargs)
        if state is not None:
            written = self._decode_with_static_cache(
                state, input_storage, finished, max_new_tokens, chunks, sample, forward_kwargs)
            input_ids = input_storage[:, :initial_length + written]
        else:
            for step in range(max_new_tokens):
                past_len = past_key_values[0][0].shape[1] if past_key_values else 0
                outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **forward_kwargs)
                if step == 0:
                    chunks.start_decode()
                attention_mask = attention_storage[:, :initial_length + step + 1] if attention_storage is not None else None
                next_token = _sample_next_token(outputs.logits[:, -1, :], input_ids, finished, **sample)
                input_storage[:, initial_length + step:initial_length + step + 1].copy_(next_token)
                input_ids = input_storage[:, :initial_length + step + 1]
                past_key_values = outputs.past_key_values if use_cache else None
                if eos_token_id is not None:
                    finished |= next_token.squeeze(-1).eq(eos_token_id)
                final_step = step + 1 == max_new_tokens
                if chunks.push(next_token, finished, final=final_step) or final_step:
                    break
        if chunks.overshoot:
            input_ids = input_ids[:, :-chunks.overshoot]
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids

    def _decode_state_for(self, input_ids, attention_mask, max_new_tokens, *, use_cache, early_exit, kwargs):
        """A reusable decode state for this generation, or None for the growing cache.

        Only single unpadded sequences qualify: the in-place append writes one
        position per step, and padded rows would need per-row slots. The legacy
        cache stays the default everywhere else.

        States are kept between calls and picked smallest-first, so a chat turn
        usually reuses the previous turn's buffers -- and with them the compiled
        graph and the recorded CUDA graph, both of which are keyed on the buffer
        addresses and shapes.
        """
        if not use_cache or not getattr(self, '_static_cache_ok', False):
            return None
        if input_ids.shape[0] != 1 or early_exit or kwargs.get('return_kv'):
            return None
        if attention_mask is not None and not bool(attention_mask.all()):
            return None
        needed = input_ids.shape[1] + max_new_tokens
        states = self._decode_states
        for state in sorted(states.values(), key=lambda item: item.capacity):
            if state.accepts(needed):
                return state
        capacity = bucket_capacity(needed, self.config.max_position_embeddings)
        if capacity is None:
            print(f'[Inference] sequence of {needed} tokens exceeds the bucket ladder; '
                  f'using the growing cache.', flush=True)
            return None
        if is_fp8(self.config.kv_cache_dtype) and not getattr(self, '_static_cache_fp8_noted', False):
            self._static_cache_fp8_noted = True
            print(f'[Inference] static KV cache stores keys/values in {next(self.parameters()).dtype} '
                  f'rather than {self.config.kv_cache_dtype}: a quantized buffer would have to be '
                  f'dequantized and re-quantized on every step.', flush=True)
        parameters = next(self.parameters())
        state = DecodeState(capacity, layer_specs(self), parameters.dtype, input_ids.device)
        bytes_needed = sum(layer.key.numel() * layer.key.element_size() * 2 for layer in state.cache)
        if input_ids.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(input_ids.device)
            if bytes_needed > 0.25 * free:
                print(f'[Inference] static KV cache needs {bytes_needed / 2 ** 20:.0f} MiB of '
                      f'{free / 2 ** 20:.0f} MiB free for max_new_tokens={max_new_tokens}; '
                      f'using the growing cache.', flush=True)
                return None
        # Keep the smallest state as the fast path for normal chat turns and one
        # larger state for a response that crossed a capacity boundary.  When a
        # response grows again, replace the larger state rather than evicting
        # the small state and permanently slowing every later short answer.
        while len(states) >= 2:
            states.pop(max(states))
        states[capacity] = state
        return state

    def _decode_with_static_cache(self, state, input_storage, finished, max_new_tokens,
                                  chunks, sample, forward_kwargs):
        """Decode into a fixed-shape cache; returns the number of tokens written.

        Token/EOS bookkeeping matches the growing-cache loop exactly, so the two
        paths emit identical sequences. Every decode step has identical shapes
        and no host-derived offset, which is what lets this step be captured as
        one CUDA graph. Prefill keeps the prompt's shape and is not captured.
        """
        initial_length = input_storage.shape[1] - max_new_tokens
        device = input_storage.device
        prompt = input_storage[:, :initial_length]
        eos_token_id = sample['eos_token_id']
        cache = state.cache
        position_ids = torch.arange(initial_length, device=device).unsqueeze(0)
        outputs = self.forward(prompt, None, cache, use_cache=True, position_ids=position_ids,
                               **forward_kwargs)
        chunks.start_decode()
        next_token = _sample_next_token(outputs.logits[:, -1, :], prompt, finished, **sample)
        written = 0
        # The state owns these buffers, so the graph recorded on the first turn
        # stays valid for every later turn that reuses the state.
        step_ids, step_position = state.step_ids, state.step_position
        graph_eligible = self._decode_kwargs_capturable(forward_kwargs)
        for step in range(max_new_tokens):
            position = initial_length + step
            input_storage[:, position:position + 1].copy_(next_token)
            written = step + 1
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
            final_step = step + 1 == max_new_tokens
            if chunks.push(next_token, finished, final=final_step) or final_step:
                break  # stop before the forward whose logits would be discarded
            # The answer reached this cache's real capacity.  Allocate the next
            # bucket and prefill it from the tokens already produced.  This is
            # infrequent (at 512/1024/2048/...) and avoids paying the largest
            # bucket's attention cost for every short answer.
            if position >= state.capacity:
                context = input_storage[:, :position + 1]
                state = self._decode_state_for(
                    context, None, 1, use_cache=True, early_exit=False, kwargs={})
                if state is None:
                    raise RuntimeError(
                        'static KV cache could not grow at capacity boundary; '
                        'reduce the context length or free GPU memory')
                cache = state.cache
                step_ids, step_position = state.step_ids, state.step_position
                position_ids = torch.arange(position + 1, device=device).unsqueeze(0)
                outputs = self.forward(
                    context, None, cache, use_cache=True,
                    position_ids=position_ids, **forward_kwargs)
                next_token = _sample_next_token(
                    outputs.logits[:, -1, :], context, finished, **sample)
                continue

            step_ids.copy_(next_token.view(1, 1))
            step_position.fill_(position)
            # A graph recorded for plain decoding must never serve a call with
            # Python callbacks (logit lens / early-exit diagnostics).
            if state.graph is not None and graph_eligible:
                graph, outputs = state.graph
                graph.replay()
            elif state.capture_failed or not graph_eligible:
                outputs = self.forward(step_ids, None, cache, use_cache=True,
                                       position_ids=step_position, **forward_kwargs)
            else:
                recorded = self._capture_decode_step(cache, step_ids, step_position, forward_kwargs)
                if recorded is None:
                    state.capture_failed = True
                    outputs = self.forward(step_ids, None, cache, use_cache=True,
                                           position_ids=step_position, **forward_kwargs)
                else:
                    state.graph = recorded
                    graph, outputs = recorded
                    graph.replay()
            next_token = _sample_next_token(outputs.logits[:, -1, :],
                                            input_storage[:, :position + 1], finished, **sample)
        return written

    @staticmethod
    def _decode_kwargs_capturable(forward_kwargs) -> bool:
        """Whether a graph recorded for these kwargs stays faithful.

        Anything that changes the trunk's behaviour per call (the logit-lens
        callback, the early-exit hook) routes to the eager module and must not be
        served from a graph recorded for a plain call.
        """
        if any(key in forward_kwargs for key in ('layer_callback', 'exit_check_fn', 'return_intermediate')):
            return False
        return set(forward_kwargs) <= {'logits_to_keep', 'use_cache'}

    def _capture_decode_step(self, cache, step_ids, step_position, forward_kwargs):
        """Record one decode step as a CUDA graph; None keeps decoding eagerly.

        A single token costs ~1000 kernel launches and as many compiler guard
        evaluations, and the decode loop is host-bound rather than GPU-bound, so
        replaying one captured graph is a large win. Everything the step reads is
        already fixed-shape: the trunk is compiled, the KV cache is preallocated
        and the two inputs are the caller's buffers. Sampling stays outside the
        graph, so an EOS check can still stop the loop mid-flight.
        """
        if not torch.cuda.is_available():
            return None
        if any(key in forward_kwargs for key in ('layer_callback', 'exit_check_fn', 'return_intermediate')):
            return None  # those route to the eager trunk (logit lens et al.)
        if not self._decode_is_capturable():
            return None
        try:
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(2):  # required before capture: warm every kernel
                    self.forward(step_ids, None, cache, use_cache=True,
                                 position_ids=step_position, **forward_kwargs)
            torch.cuda.current_stream().wait_stream(side_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self.forward(step_ids, None, cache, use_cache=True,
                                       position_ids=step_position, **forward_kwargs)
            return graph, outputs
        except Exception as exc:
            # A failed capture invalidates its stream and leaves the error latched
            # for the next launch; clear it so the eager fallback can proceed.
            torch.cuda.synchronize()
            self._capture_ok = False
            print(f'[Inference] CUDA graph capture unavailable ({type(exc).__name__}: '
                  f'{str(exc)[:140]}); decoding eagerly.', flush=True)
            return None

    def _decode_is_capturable(self) -> bool:
        """Whether every expert dispatch in this model is free of host syncs.

        ``cudaStreamCapture`` aborts on any synchronizing op, and two MoE paths
        still synchronize: the per-expert loop (``nonzero`` per expert) and the
        ``cached`` grouped backend (``offsets.cpu()``). Decided once from the
        installed backends; models that never went through ``optimize_inference``
        keep the proven eager loop.
        """
        cached = getattr(self, '_capture_ok', None)
        if cached is None:
            trunk = getattr(self.model, 'original', self.model)
            cached = all(
                getattr(layer.mlp, '_inference_backend', None) in ('native', 'triton')
                for layer in trunk.layers
                if isinstance(layer.mlp, MOEFeedForward)
            )
            self._capture_ok = cached
        return cached


def _sample_next_token(logits, input_ids, finished, *, temperature, top_p, top_k, do_sample,
                       repetition_penalty, eos_token_id):
    """Sample or take the argmax of ``logits``; shared by both decode loops."""
    logits = logits[:, -1, :] if logits.dim() == 3 else logits
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
    next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
    if eos_token_id is not None:
        next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
    return next_token
