"""
线性注意力（Linear Attention）变体:Instinct 的 GatedDeltaNet 混合架构。

与主线 model/model_instinct.py（Dense GQA 多头注意力）不同,本文件把主干的一部分层
替换为 GatedDeltaNet——一种门控 delta rule 线性注意力:通过因果卷积（depthwise conv1d）
做局部建模,再用线性递归状态传递全局信息,使注意力复杂度从 O(T^2) 降到 O(T),
适合超长序列的预训练与推理。层间按 config.full_attention_interval 交替
「linear_attention / full_attention」,兼顾线性注意力的效率与全注意力的表达能力。

本文件自包含:定义了独立的 InstinctConfig / InstinctModel / InstinctForCausalLM,
对外接口与 Dense 主线完全一致（同为 Qwen3 风格,支持 KV cache 量化、梯度检查点、
Logit Lens 与 Early Exit 动态推理）,训练脚本与模型转换工具可直接复用。
"""
import math
import torch
import torch.nn.functional as F
import logging
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from model.flash_attn_4 import flash_attention
from model.kv_cache_quant import parse_cache, make_cache
from model.checkpointing import recompute_attention, checkpoint_ffn

logger = logging.getLogger(__name__)

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    logger.info('causal-conv1d detected, CUDA conv acceleration enabled')
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
    logger.warning('causal-conv1d not available, falling back to PyTorch native conv')

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    logger.info('flash-linear-attention (FLA) detected, Triton linear attention acceleration enabled')
except ImportError:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    logger.warning('flash-linear-attention (FLA) not available, falling back to PyTorch native linear attention')

# ═══════════════════════════════════════════════════════════════
# Instinct Config
# ═══════════════════════════════════════════════════════════════
class InstinctConfig(PretrainedConfig):
    """线性注意力变体（GatedDeltaNet 混合架构）的配置类。

    继承 PretrainedConfig,与 Dense 主线共享大部分字段;线性注意力专属的超参包括:
    - full_attention_interval:每 N 层插入一层标准全注意力
    - linear_conv_kernel_dim:因果卷积核大小
    - linear_key_head_dim / linear_value_head_dim / linear_num_key_heads / linear_num_value_heads:线性注意力头维度
    - layer_types:按 full_attention_interval 自动生成的逐层类型列表（"linear_attention" / "full_attention"）
    """
    model_type = "instinct"
    def __init__(self, hidden_size: int = 768, num_hidden_layers: int = 8, use_moe: bool = False, **kwargs):
        """初始化配置:全部可选字段经 kwargs 传入,并据此生成 layer_types。"""
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.param_dtype = kwargs.get("param_dtype", "fp32")
        self.kv_cache_dtype = kwargs.get("kv_cache_dtype", "fp32")
        # Gradient Checkpointing configs
        self.use_grad_checkpoint = kwargs.get("use_grad_checkpoint", 0)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        # Early Exit configs (LayerSkip-style: shared LM head, no auxiliary classifiers)
        self.early_exit_layers = kwargs.get("early_exit_layers", [4, 5, 6, 7])
        self.early_exit_loss_weight = kwargs.get("early_exit_loss_weight", 0.3)
        # MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        # GatedDeltaNet configs
        self.full_attention_interval = kwargs.get("full_attention_interval", 4)
        self.linear_conv_kernel_dim = kwargs.get("linear_conv_kernel_dim", 4)
        self.linear_key_head_dim = kwargs.get("linear_key_head_dim", self.head_dim)
        self.linear_value_head_dim = kwargs.get("linear_value_head_dim", self.head_dim)
        self.linear_num_key_heads = kwargs.get("linear_num_key_heads", self.num_attention_heads)
        self.linear_num_value_heads = kwargs.get("linear_num_value_heads", self.num_attention_heads)
        self.layer_types = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.full_attention_interval == 0:
                self.layer_types.append("full_attention")
            else:
                self.layer_types.append("linear_attention")

# ═══════════════════════════════════════════════════════════════
# Instinct Model
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
        return (self.weight * self.norm(x.float())).type_as(x)

class RMSNormGated(nn.Module):
    """带门控的 RMS 归一化:归一化结果再乘以 SiLU(gate),用于 GatedDeltaNet 的 z 输出门。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        """初始化归一化权重（全 1）与 eps。"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate=None):
        """前向:fp32 归一化 + 权重缩放,再乘 SiLU 门控。"""
        x_float = x.float()
        x_normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = (self.weight * x_normed.type_as(x))
        return x_normed * F.silu(gate.float()).type_as(x)

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    """沿指定维度做 L2 归一化（保留方向、归一化幅度）。"""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

def torch_chunk_gated_delta_rule(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
    chunk_size: int = 128, initial_state: torch.Tensor = None, output_final_state: bool = False,
):
    """chunk_gated_delta_rule 的纯 PyTorch 参考实现（FLA Triton 内核缺失时的回退）。

    算法分两层:
    - 块内（intra-chunk）:构造严格下三角的关联矩阵 B = -(k_beta @ k^T) ⊙ decay_mask
      （decay_mask = exp(g_i - g_j),g 为累积对数衰减）,再逐行扫描
      A_i = B_i + B_i @ A_{:i} 计算 (I - B)^{-1} - I（B 幂零,无需显式求逆）;
      得到 A 后,v = A @ v_beta、k_cumdecay = A @ (k_beta * exp(g));
    - 块间（inter-chunk）:用循环状态 S(B, H, Dk, Dv) 跨块传递:
      v_new = v_i - k_cumdecay @ S（delta rule 残差更新）,
      out = (q * exp(g)) @ S + attn_intra @ v_new,
      S = S * exp(g_last) + (k * exp(g_last - g))^T @ v_new。

    输出形状 (B, T, H, Dv)（去掉 padding）;output_final_state=False 时返回的 S 为 None。
    """
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    q, k, v = [F.pad(x, (0, 0, 0, pad)) for x in (q, k, v)]
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    T_pad = T + pad
    scale = Dk ** -0.5
    q = q * scale
    v_beta, k_beta = v * beta.unsqueeze(-1), k * beta.unsqueeze(-1)
    q, k, v, k_beta, v_beta = [x.reshape(B, H, -1, chunk_size, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)]
    g = g.reshape(B, H, -1, chunk_size)
    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(mask_upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, device=attn.device, dtype=attn.dtype)
    v = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    S = torch.zeros(B, H, Dk, Dv, device=v.device, dtype=v.dtype) if initial_state is None else initial_state.float()
    out = torch.zeros_like(v)
    mask_causal = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    num_chunks = T_pad // chunk_size
    for i in range(num_chunks):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn_intra = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_causal, 0)
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ S
        out[:, :, i] = attn_inter + attn_intra @ v_new
        S = S * g[:, :, i, -1, None, None].exp() + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    if not output_final_state:
        S = None
    out = out.reshape(B, H, -1, Dv)[:, :, :T].transpose(1, 2).contiguous().to(q.dtype)
    return out, S

class GatedDeltaNet(nn.Module):
    def __init__(self, config: InstinctConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias=False, kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads).uniform_(0, 16).log_())
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(self, x, conv_state=None, recurrent_state=None, use_cache=False):
        input_dtype = x.dtype
        # 参数可能被 param_dtype 转成 bf16/fp16：输入跟随参数 dtype，否则 conv1d 等会
        # 因 fp32 输入 vs bf16 权重在 torch.compile 的 fake tensor 阶段直接报 dtype 不匹配
        param_dtype = next(self.parameters()).dtype
        x = x.to(param_dtype)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            return self._forward(x, conv_state, recurrent_state, use_cache, input_dtype)

    def _forward(self, x, conv_state, recurrent_state, use_cache, input_dtype):
        B, T, _ = x.shape
        w, b = self.conv1d.weight.squeeze(1), self.conv1d.bias
        use_recurrent = conv_state is not None and T == 1
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(B, T, -1, self.head_v_dim)
        beta, a = self.in_proj_b(x).sigmoid(), self.in_proj_a(x)
        if use_recurrent:
            if causal_conv1d_update is not None and x.is_cuda:
                mixed_qkv = causal_conv1d_update(mixed_qkv, conv_state, w, b, "silu")
            else:
                xc = torch.cat([conv_state, mixed_qkv], dim=-1)
                conv_state.copy_(xc[:, :, 1:])
                mixed_qkv = (xc * w).sum(-1, keepdim=True)
                if b is not None: mixed_qkv = mixed_qkv + b.unsqueeze(-1)
                mixed_qkv = F.silu(mixed_qkv)
        else:
            if use_cache:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - 1 - mixed_qkv.shape[-1], 0))[:, :, -(self.conv_kernel_size - 1):]
            if causal_conv1d_fn is not None and mixed_qkv.is_cuda:
                mixed_qkv = causal_conv1d_fn(x=mixed_qkv, weight=w, bias=b, activation="silu")
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :T])
        mixed_qkv = mixed_qkv.transpose(1, 2)
        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = l2norm(q.reshape(B, T, -1, self.head_k_dim))
        k = l2norm(k.reshape(B, T, -1, self.head_k_dim))
        v = v.reshape(B, T, -1, self.head_v_dim)
        g = (-self.A_log.exp() * F.softplus(a + self.dt_bias))
        if self.num_v_heads // self.num_k_heads > 1:
            q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        if use_recurrent:
            if fused_recurrent_gated_delta_rule is not None and q.is_cuda:
                try:
                    out, recurrent_state = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=recurrent_state, output_final_state=use_cache)
                except Exception as e:
                    logger.warning_once(f"FLA fused_recurrent kernel failed: {e}, falling back to torch.")
                    out = None
            else:
                out = None
            if out is None:
                scale = q.shape[-1] ** -0.5
                q_t, k_t, v_t = q.squeeze(1) * scale, k.squeeze(1), v.squeeze(1)
                recurrent_state = recurrent_state * g.squeeze(1).exp().unsqueeze(-1).unsqueeze(-1)
                kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
                recurrent_state = recurrent_state + k_t.unsqueeze(-1) * ((v_t - kv_mem) * beta.squeeze(1).unsqueeze(-1)).unsqueeze(-2)
                out = (recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2).unsqueeze(1)
        else:
            if chunk_gated_delta_rule is not None and q.is_cuda:
                try:
                    out, recurrent_state = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=use_cache)
                except Exception as e:
                    logger.warning_once(f"FLA chunk kernel failed: {e}, falling back to torch.")
                    out, recurrent_state = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=use_cache)
            else:
                out, recurrent_state = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=use_cache)
        out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        out = self.out_proj(out.reshape(B, T, -1))
        if out.dtype != input_dtype: out = out.to(input_dtype)
        return out, (conv_state, recurrent_state) if use_cache else None

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
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
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA 头扩展:把 KV 头沿新维重复 n_rep 次并展平,使其与 Q 头数一致。"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    """标准 GQA 因果注意力（Qwen3 风格）:q/k 先 RMSNorm 归一,再施加 RoPE。

    支持 SDPA/FA4 flash 快速路径与 eager 路径（含选择性重算 recompute_attention）;
    KV cache 经 make_cache/parse_cache 支持量化（fp8 等）与 dtype 回读。
    """
    def __init__(self, config: InstinctConfig):
        """初始化 QKV/输出投影、q/k 归一化、dropout、KV cache dtype 与 flash 开关。"""
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
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

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """前向:投影 → q/k 归一化 → RoPE → 拼接历史 KV → flash/eager 注意力 → 输出投影。"""
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
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # FA4 / SDPA fused fast path: 输入输出布局 (bs, seq, heads, hd),kernel 内部处理 GQA
            output = flash_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            output = output.reshape(bsz, seq_len, -1)
        else:
            if self.use_grad_checkpoint == 1 and self.training and past_key_value is None:
                # Mode 1 selective recompute: recompute the eager attention core
                # (QK^T, causal mask, softmax, dropout, @V) in backward instead of
                # keeping the O(seq^2) scores alive. q/k/v are pre-transpose and
                # pre-repeat_kv; the linear standard Attention is always causal.
                output = recompute_attention(xq, xk, xv, attention_mask, True, self.attn_dropout.p, self.head_dim)
            else:
                xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
                scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
                if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
                output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
                output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    """SwiGLU FFN:gate_proj(x) * up_proj(x) 经激活后由 down_proj 投影回 hidden_size。"""
    def __init__(self, config: InstinctConfig, intermediate_size: int = None):
        """初始化 gate/down/up 三个线性层与激活函数;intermediate_size 可覆盖默认值。"""
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """前向:SwiGLU 门控激活（act(gate(x)) * up(x)）→ down 投影。"""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    """Top-k 路由 MoE FFN:softmax 门控选 top-k 专家,按权重聚合专家输出。

    训练时累计 router aux loss（负载均衡正则,通过 aux_loss 属性暴露给外部累加）。
    """
    def __init__(self, config: InstinctConfig):
        """初始化路由器与专家列表（每个专家是一个 FeedForward）。"""
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """前向:门控打分 → top-k 选择 → 逐专家 index_add 聚合;附带 aux_loss 计算。"""
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
    """混合注意力块:按 config.layer_types 分派——linear_attention 层走 GatedDeltaNet,
    full_attention 层走标准 Attention;Pre-Norm 残差结构,FFN 支持 mode-1 选择性重算（checkpoint_ffn）。"""
    def __init__(self, layer_id: int, config: InstinctConfig):
        """按 layer_type 构造注意力子模块、两个 RMSNorm 与 MLP（可切 MoE）。"""
        super().__init__()
        self.layer_type = config.layer_types[layer_id]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_id)
        else:
            self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.use_grad_checkpoint = getattr(config, "use_grad_checkpoint", 0)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """前向:残差 + 注意力（GatedDeltaNet 或 Attention）→ post-norm → MLP;返回 (hidden_states, present_key_value)。"""
        residual = hidden_states
        if self.layer_type == "linear_attention":
            conv_state = past_key_value[0] if past_key_value is not None else None
            recurrent_state = past_key_value[1] if past_key_value is not None else None
            hidden_states, present_key_value = self.linear_attn(
                self.input_layernorm(hidden_states), conv_state, recurrent_state, use_cache
            )
        else:
            hidden_states, present_key_value = self.self_attn(
                self.input_layernorm(hidden_states), position_embeddings,
                past_key_value, use_cache, attention_mask
            )
        hidden_states += residual
        normed = self.post_attention_layernorm(hidden_states)
        if self.use_grad_checkpoint == 1 and self.training:
            # Mode 1 FFN recompute (all layer types, incl. GatedDeltaNet layers):
            # the MoE aux_loss is surfaced through checkpoint_ffn's return value
            # (it runs under no_grad forward, so the module attribute side-channel
            # would carry a gradient-less constant); storing it back preserves the
            # graph node so the model-side accumulation keeps the router gradient.
            ffn_out, aux = checkpoint_ffn(self.mlp, normed)
            if aux is not None:
                self.mlp.aux_loss = aux
            hidden_states = hidden_states + ffn_out
        else:
            hidden_states = hidden_states + self.mlp(normed)
        return hidden_states, present_key_value

class InstinctModel(nn.Module):
    """混合架构主干:Embedding → N 个 InstinctBlock → 最终 RMSNorm。

    预计算 RoPE 频率表并注册为非持久 buffer;支持 mode-2 整层梯度检查点、
    per-layer callback（logit lens 等流式钩子）、中间态收集与 early-exit 早退钩子。
    """
    def __init__(self, config: InstinctConfig):
        """初始化 embedding、dropout、层列表、最终 norm 与 RoPE buffer。"""
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([InstinctBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, end=config.max_position_embeddings,
            rope_base=config.rope_theta, rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """前向:推导 start_pos（首个有 KV 的全注意力层）→ embedding → 逐层前向。

        逐层收集 presents / intermediates,累加 MoE aux_loss;支持 mode-2 整层 checkpoint、
        layer_callback 流式钩子与 exit_check_fn 早退（早退时用 None 补齐剩余层 KV 槽位）。
        """
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = 0
        for i, lt in enumerate(self.config.layer_types):
            if lt == "full_attention" and past_key_values[i] is not None:
                start_pos = past_key_values[i][0].shape[1]
                break
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        presents, intermediates = [], []
        aux_loss = hidden_states.new_zeros(1).squeeze()
        return_intermediate = kwargs.pop("return_intermediate", False)
        exit_check_fn = kwargs.pop("exit_check_fn", None)
        layer_callback = kwargs.pop("layer_callback", None)
        for layer, past_key_value in zip(self.layers, past_key_values):
            if self.config.use_grad_checkpoint == 2 and self.training:
                # Mode 2 whole-layer checkpoint: re-run the entire block forward
                # (standard Attention and GatedDeltaNet alike) in backward via
                # torch.utils.checkpoint — we never touch the FLA/Triton kernel
                # internals. `layer=layer` binds the loop variable so backward's
                # re-run uses the *same* block that ran in forward.
                def _blk(h, pkv, layer=layer):
                    hs, present = layer(h, position_embeddings, past_key_value=pkv,
                                        use_cache=use_cache, attention_mask=attention_mask)
                    aux = layer.mlp.aux_loss if isinstance(layer.mlp, MOEFeedForward) else None
                    return hs, present, aux
                hidden_states, present, layer_aux = torch.utils.checkpoint.checkpoint(
                    _blk, hidden_states, past_key_value, use_reentrant=False, preserve_rng_state=True)
                presents.append(present)
                if layer_aux is not None:
                    aux_loss = aux_loss + layer_aux
            else:
                hidden_states, present = layer(
                    hidden_states,
                    position_embeddings,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    attention_mask=attention_mask
                )
                presents.append(present)
                if isinstance(layer.mlp, MOEFeedForward):
                    aux_loss = aux_loss + layer.mlp.aux_loss
            if layer_callback is not None:
                # Per-layer streaming hook (logit lens etc.): fires with the normed
                # hidden state right after each layer computes it.
                layer_callback(len(presents), self.norm(hidden_states))
            if return_intermediate:
                intermediates.append(self.norm(hidden_states))
            if exit_check_fn is not None:
                normed = self.norm(hidden_states)
                if exit_check_fn(len(presents), normed):
                    presents.extend([None] * (len(self.layers) - len(presents)))
                    if return_intermediate:
                        return normed, presents, aux_loss, intermediates
                    return normed, presents, aux_loss
        hidden_states = self.norm(hidden_states)
        if return_intermediate:
            return hidden_states, presents, aux_loss, intermediates
        return hidden_states, presents, aux_loss

class InstinctForCausalLM(PreTrainedModel, GenerationMixin):
    """线性注意力模型的语言模型头:共享 embedding/lm_head 权重（tied weights）。

    支持 Logit Lens（logit_lens 逐层反嵌入探测）、Early Exit（训练时加权早退损失 /
    推理时置信度阈值动态退出）与 generate 采样解码;统一返回 MoeCausalLMOutputWithPast。
    """
    config_class = InstinctConfig
    def __init__(self, config: InstinctConfig = None):
        """初始化主干模型与 lm_head,并绑定 embedding 权重（tied weights）。"""
        self.config = config or InstinctConfig()
        super().__init__(self.config)
        self.model = InstinctModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, labels=None, early_exit=False, logit_lens=False, **kwargs):
        """前向:按 logit_lens / early_exit / 普通三条路径计算 logits 与损失,统一返回 MoeCausalLMOutputWithPast。"""
        def _cross_entropy_loss(logits, labels):
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            return F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        def _compute_logits(hidden_states):
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            return self.lm_head(hidden_states[:, slice_indices, :])

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
    
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85,
                 top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True,
                 num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """自回归解码循环:top-k / top-p / temperature 采样,支持 repetition penalty、
        streamer、early exit 与 KV cache 复用;return_kv=True 时额外返回 past_kv。"""
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        early_exit = kwargs.pop("early_exit", False)
        exit_threshold = kwargs.pop("exit_threshold", 0.9)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = 0
            if past_key_values:
                for i, lt in enumerate(self.model.config.layer_types):
                    if lt == "full_attention" and past_key_values[i] is not None:
                        past_len = past_key_values[i][0].shape[1]
                        break
            forward_kwargs = dict(kwargs)
            if early_exit:
                forward_kwargs['early_exit'] = True
                forward_kwargs['exit_threshold'] = exit_threshold
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **forward_kwargs)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1
            ) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids