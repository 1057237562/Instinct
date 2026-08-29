"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import math
import inspect
import importlib.metadata
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel
from model.model_instinct import InstinctForCausalLM, InstinctConfig


def _architecture_classes(architecture):
    """Import optional backbones only when selected (important for Windows workers)."""
    if architecture == 'linear':
        from model.model_instinct_linear import InstinctConfig as Config, InstinctForCausalLM as Model
        return Config, Model
    if architecture == 'looped':
        from model.model_instinct_loop import InstinctConfig as Config, InstinctForCausalLM as Model
        return Config, Model
    return InstinctConfig, InstinctForCausalLM

def get_model_params(model: torch.nn.Module, config) -> None:
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


_TORCHAO_FP8_RECIPES = {"tensorwise", "rowwise", "rowwise_with_gw_hp"}
_TORCHAO_FP8_PROBE_CACHE = {}


def _fp8_linear_is_eligible(module: torch.nn.Module, fqn: str) -> bool:
    """Return whether a Linear has hardware-compatible FP8 GEMM dimensions.

    The language-model output projection stays in high precision for numerical
    stability (and because it shares its weight with the token embedding).
    LoRA adapters also stay high precision; only their base Linear is eligible.
    """
    if not isinstance(module, torch.nn.Linear):
        return False
    if fqn == "lm_head" or fqn.endswith(".lm_head") or ".lora." in fqn:
        return False
    return module.in_features % 16 == 0 and module.out_features % 16 == 0


def _probe_torchao_fp8_recipe(recipe: str, device: torch.device, config_cls, convert_fn):
    """Exercise one real FP8 forward/backward before converting the full model."""
    cache_key = (recipe, device.type, device.index)
    cached = _TORCHAO_FP8_PROBE_CACHE.get(cache_key)
    if cached is not None:
        if isinstance(cached, Exception):
            raise cached
        return

    try:
        with torch.cuda.device(device):
            probe = torch.nn.Sequential(
                torch.nn.Linear(64, 64, bias=False, device=device, dtype=torch.bfloat16)
            )
            convert_fn(probe, config=config_cls.from_recipe_name(recipe))
            x = torch.randn(32, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
            probe(x).float().square().mean().backward()
            torch.cuda.synchronize(device)
            del probe, x
        _TORCHAO_FP8_PROBE_CACHE[cache_key] = True
    except Exception as exc:
        _TORCHAO_FP8_PROBE_CACHE[cache_key] = exc
        raise


def apply_torchao_fp8_training(model: torch.nn.Module, args, *, label: str = "model") -> torch.nn.Module:
    """Convert eligible Linear layers to TorchAO Float8Linear for training.

    Conversion is state-dict compatible: Float8Linear reuses the original
    Parameters and only changes the forward/backward GEMMs.  Call this after
    loading model/optimizer resume state and before DDP/torch.compile wrapping.
    """
    requested_recipe = getattr(args, "fp8_training", "off")
    if requested_recipe == "off":
        return model
    if requested_recipe not in _TORCHAO_FP8_RECIPES:
        raise ValueError(f"Unknown --fp8_training recipe: {requested_recipe}")
    if getattr(args, "dtype", "bfloat16") != "bfloat16":
        raise ValueError("TorchAO FP8 training requires --dtype bfloat16")
    if not torch.cuda.is_available() or not str(getattr(args, "device", "cuda")).startswith("cuda"):
        raise RuntimeError("TorchAO FP8 training requires a CUDA GPU")

    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"TorchAO FP8 training requires a CUDA model, got {device}")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (8, 9):
        raise RuntimeError(
            f"TorchAO FP8 training requires NVIDIA compute capability >= 8.9, got {major}.{minor}"
        )

    try:
        import torchao
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as exc:
        raise ImportError(
            "--fp8_training requires TorchAO. Install the project FP8 dependencies "
            "with: python -m pip install -r requirements-fp8.txt"
        ) from exc

    active_recipe = requested_recipe
    try:
        _probe_torchao_fp8_recipe(
            active_recipe, device, Float8LinearConfig, convert_to_float8_training
        )
    except Exception as exc:
        if active_recipe.startswith("rowwise"):
            Logger(
                f"[TorchAO FP8] {active_recipe} is unavailable on "
                f"{torch.cuda.get_device_name(device)} ({exc}); falling back to tensorwise"
            )
            active_recipe = "tensorwise"
            _probe_torchao_fp8_recipe(
                active_recipe, device, Float8LinearConfig, convert_to_float8_training
            )
        else:
            raise RuntimeError(f"TorchAO FP8 tensorwise preflight failed: {exc}") from exc

    filter_mode = getattr(args, "fp8_filter", "auto")
    auto_filter = None
    if filter_mode == "auto":
        try:
            from torchao.float8 import _auto_filter_for_recipe
            auto_filter = _auto_filter_for_recipe(active_recipe, filter_fqns=["lm_head"])
        except (ImportError, AttributeError):
            Logger("[TorchAO FP8] auto filter unavailable; using all eligible Linear layers")
    elif filter_mode != "eligible":
        raise ValueError(f"Unknown --fp8_filter mode: {filter_mode}")

    def module_filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        if not _fp8_linear_is_eligible(module, fqn):
            return False
        return auto_filter(module, fqn) if auto_filter is not None else True

    converted_names = [
        name for name, module in model.named_modules() if module_filter_fn(module, name)
    ]
    if not converted_names and auto_filter is not None:
        Logger(
            f"[TorchAO FP8] auto filter selected no layers for {label}; "
            "falling back to all eligible Linear layers"
        )
        auto_filter = None
        converted_names = [
            name for name, module in model.named_modules() if module_filter_fn(module, name)
        ]
    if not converted_names:
        raise RuntimeError(
            f"TorchAO FP8 filter {filter_mode!r} selected no Linear layers for {label}; "
            "use --fp8_filter eligible or disable FP8"
        )

    config = Float8LinearConfig.from_recipe_name(active_recipe)
    convert_to_float8_training(model, module_filter_fn=module_filter_fn, config=config)
    version = getattr(torchao, "__version__", importlib.metadata.version("torchao"))
    if getattr(args, "use_compile", 0) != 1:
        Logger("[TorchAO FP8] warning: torch.compile is disabled; FP8 may be slower than BF16")
    Logger(
        f"[TorchAO FP8] enabled for {label}: requested={requested_recipe}, "
        f"active={active_recipe}, filter={filter_mode}, linear_layers={len(converted_names)}, "
        f"torchao={version}"
    )
    setattr(args, "fp8_training_active", active_recipe)
    return model


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def config_from_args(args, **overrides):
    """从命令行参数（可选叠加 JSON 配置）构建模型配置。

    参数:
        args: argparse 解析出的命令行参数对象
        **overrides: 显式覆盖项（hidden_size / num_hidden_layers / use_moe / use_looped 等）

    说明:
        - 指定 config_path 时，以 JSON 为基底再叠加命令行覆盖；
        - 按 model_architecture（兼容 use_looped）选择 Dense / Linear / Looped 配置类。
    """
    overrides.setdefault('param_dtype', getattr(args, 'param_dtype', 'fp32'))
    overrides.setdefault('kv_cache_dtype', getattr(args, 'kv_cache_dtype', 'fp32'))
    overrides.setdefault("use_grad_checkpoint", int(getattr(args, "use_grad_checkpoint", 0)))
    for field in (
        "residual_type", "hc_mult", "hc_sinkhorn_iters", "hc_eps",
        "attnres_variant", "attnres_block_size",
    ):
        value = getattr(args, field, None)
        if value is not None:
            overrides.setdefault(field, value)
    hidden_size = overrides.pop('hidden_size', getattr(args, 'hidden_size', 768))
    num_hidden_layers = overrides.pop('num_hidden_layers', getattr(args, 'num_hidden_layers', 8))
    use_moe = overrides.pop('use_moe', bool(getattr(args, 'use_moe', 0)))
    use_looped = overrides.pop('use_looped', bool(getattr(args, 'use_looped', 0)))
    architecture = getattr(args, 'model_architecture', None)
    if use_looped:
        architecture = 'looped'
    config_path = getattr(args, 'config_path', None)
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg_dict = json.load(f)
        cfg_dict['hidden_size'] = hidden_size
        cfg_dict['num_hidden_layers'] = num_hidden_layers
        cfg_dict['use_moe'] = use_moe
        cfg_dict.update(overrides)
        architecture = architecture or cfg_dict.get('model_architecture', 'standard')
        cfg_dict['model_architecture'] = architecture
        cfg_cls, _ = _architecture_classes(architecture)
        return cfg_cls(**cfg_dict)
    architecture = architecture or 'standard'
    cfg_cls, _ = _architecture_classes(architecture)
    return cfg_cls(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
        model_architecture=architecture,
        **overrides,
    )


_TOPOLOGY_CONFIG_FIELDS = (
    "model_architecture", "residual_type",
    "hc_mult", "hc_sinkhorn_iters", "hc_eps", "mhc_init_std",
    "attnres_variant", "attnres_block_size",
    "full_attention_interval", "linear_conv_kernel_dim",
    "linear_key_head_dim", "linear_value_head_dim",
    "linear_num_key_heads", "linear_num_value_heads",
    "loop_iters", "prelude_layers", "coda_layers", "use_input_injection",
)


def restore_config_from_checkpoint(current_config, checkpoint_data, *,
                                   config_key="config", fallback_topology_key=None):
    """Rebuild the model config before loading a resume checkpoint.

    Resume state is authoritative for architecture and residual topology. This
    prevents a refreshed WebUI/CLI config from constructing a Standard model
    and then trying to load mHC/AttnRes parameters into it.

    ``fallback_topology_key`` supports legacy distillation checkpoints that
    stored the student config but not a separate teacher config: teacher shape
    fields stay current while topology fields follow the saved student.
    """
    if not checkpoint_data:
        return current_config
    saved_config = checkpoint_data.get(config_key)
    if saved_config is None and fallback_topology_key:
        topology = checkpoint_data.get(fallback_topology_key)
        if topology is not None:
            saved_config = current_config.to_dict()
            for field in _TOPOLOGY_CONFIG_FIELDS:
                if field in topology:
                    saved_config[field] = topology[field]
    if saved_config is None:
        Logger(f"[Resume] checkpoint has no {config_key!r}; using current model config")
        return current_config

    saved_config = dict(saved_config)
    architecture = saved_config.get(
        "model_architecture", getattr(current_config, "model_architecture", "standard")
    )
    saved_config["model_architecture"] = architecture
    config_cls, _ = _architecture_classes(architecture)
    restored = config_cls(**saved_config)
    Logger(
        f"[Resume] restored checkpoint config: architecture={architecture}, "
        f"residual={getattr(restored, 'residual_type', 'standard')}"
    )
    return restored


def Logger(content: str) -> None:
    if is_main_process():
        print(content)


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode() -> int:
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════
# 优化器工厂: AdamW / Adafactor / Muon
# ═══════════════════════════════════════════════════════════════

def _zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz 迭代求矩阵的 0 次幂（正交化），使用全局收敛的 5 阶迭代系数。

    与 torch 官方 `torch.optim.Muon` 内部的 `_orthogonalize` 语义保持一致：
    对矩阵（ndim>=2）的梯度做正交化，只保留方向信息、去掉幅度。
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class MuonOptimizer(torch.optim.Optimizer):
    """Muon (MomentUm Orthogonalized by Newton-schulz) 的纯 PyTorch 原生实现。

    用于 torch < 2.10（此时 `torch.optim.Muon` 尚不存在）时的回退方案。
    默认超参与 `torch.optim.Muon` 完全对齐，API 兼容，升级 torch 后可直接切换内置版本：

    - lr=0.02, momentum=0.95, nesterov=True, ns_steps=5
    - orthogonalize_scale='weight_decay', weight_decay=0.01

    更新规则：
    - ndim >= 2 的权重（矩阵）使用「动量 + Newton-Schulz 正交化」的更新；
    - 其余 1D 参数退化为带 Nesterov 动量的 SGD；
    - 采用解耦权重衰减（decoupled weight decay）。
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 orthogonalize_scale='weight_decay', weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if ns_steps < 0:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
        if orthogonalize_scale not in ('lr', 'weight_decay'):
            raise ValueError(f"Invalid orthogonalize_scale value: {orthogonalize_scale}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        orthogonalize_scale=orthogonalize_scale, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            orthogonalize_scale = group['orthogonalize_scale']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')
                state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if nesterov else buf
                if p.ndim >= 2:
                    g = _zeropower_via_newtonschulz5(g, ns_steps)
                    scale = lr if orthogonalize_scale == 'lr' else lr * weight_decay
                    p.add_(g, alpha=-scale)
                else:
                    p.add_(g, alpha=-lr)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
        return loss


class CombinedOptimizer(torch.optim.Optimizer):
    """将多个 PyTorch 优化器组合为一个 Optimizer-like 接口。

    背景：`torch.optim.Muon`（torch>=2.10 内置）只接受 2D 参数矩阵，
    模型中的 1D 参数（RMSNorm 权重、bias 等）必须交给 AdamW 处理。
    为保持训练脚本的调用方式不变（step / zero_grad / param_groups /
    state_dict / load_state_dict），把两个子优化器包成一个对外兼容的对象：

    - `param_groups` 直接拼接子优化器的 param_groups（同一 dict 对象），
      训练脚本里 `param_group['lr'] = lr` 的调度方式对子优化器透明生效；
    - `step` / `zero_grad` 依次转发给所有子优化器；
    - `state_dict` / `load_state_dict` 以 `{"optimizers": [...]}` 格式存取。
    """

    def __init__(self, optimizers):
        optimizers = list(optimizers)
        if not optimizers:
            raise ValueError("optimizers 不能为空")
        self.optimizers = optimizers
        # 拼接子优化器的 param_groups（共享 dict 对象），超参由子优化器各自管理
        param_groups = [g for opt in optimizers for g in opt.param_groups]
        super().__init__(param_groups, defaults={})

    def zero_grad(self, *args, **kwargs):
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def step(self, closure=None):
        for opt in self.optimizers:
            if closure is None:
                opt.step()
            else:
                opt.step(closure)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        if "optimizers" not in state_dict:
            raise ValueError(
                "检查点中的优化器状态不是 CombinedOptimizer 格式（可能由旧版 "
                "MuonOptimizer 或其它优化器保存）。跨格式恢复 Muon 训练状态不受支持，"
                "请改用 --from_weight 从模型权重继续，而非 --from_resume。"
            )
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)
        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]


_OPTIMIZER_ALIASES = {
    'adamw': 'adamw',
    'adafactor': 'adafactor',
    'adafactory': 'adafactor',  # 兼容 "AdaFactory" 拼写
    'muon': 'muon',
}


def build_optimizer(params, lr: float, optimizer: str = 'adamw', **kwargs) -> torch.optim.Optimizer:
    """按名称统一构建优化器：AdamW / Adafactor / Muon。

    参数:
        params: 可迭代参数（model.parameters() 或 LoRA 参数列表）
        lr: 学习率
        optimizer: 优化器名（adamw / adafactor / muon）
        **kwargs: 透传给具体优化器的额外参数

    说明:
        - AdamW    -> torch.optim.AdamW（保持原有默认行为）
        - Adafactor-> torch.optim.Adafactor（显式 lr，关闭自动相对步长）
        - Muon     -> 2D 参数交给 Muon（优先 torch.optim.Muon，torch>=2.10 内置；
                      否则回退到原生 MuonOptimizer）；1D 参数（RMSNorm 权重/bias 等）
                      交给 AdamW。两类都存在时组合为 CombinedOptimizer 返回。
    """
    name = _OPTIMIZER_ALIASES.get(str(optimizer).strip().lower(), str(optimizer).strip().lower())
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, **kwargs)
    if name == 'adafactor':
        if 'relative_step' in inspect.signature(torch.optim.Adafactor.__init__).parameters:
            # 旧版 API（torch < 2.8）：关闭 relative_step / scale_parameter，使 lr 显式生效
            defaults = dict(lr=lr, relative_step=False, scale_parameter=False, warmup_init=False)
            defaults.update(kwargs)
        else:
            # 新版 API（torch >= 2.8）：lr 直接生效，无需额外开关
            defaults = dict(lr=lr)
            defaults.update(kwargs)
        return torch.optim.Adafactor(params, **defaults)
    if name == 'muon':
        params = list(params)
        matrix_params = [p for p in params if p.ndim == 2]
        other_params = [p for p in params if p.ndim != 2]
        if not matrix_params:
            return torch.optim.AdamW(other_params, lr=lr)
        if hasattr(torch.optim, 'Muon'):
            muon_opt = torch.optim.Muon(matrix_params, lr=lr, **kwargs)
        else:
            muon_opt = MuonOptimizer(matrix_params, lr=lr, **kwargs)
        if not other_params:
            return muon_opt
        return CombinedOptimizer([muon_opt, torch.optim.AdamW(other_params, lr=lr)])
    raise ValueError(f"未知优化器: {optimizer}，可选: adamw / adafactor / muon")


def lm_checkpoint(lm_config, weight: str = 'full_sft', model=None, optimizer=None, epoch: int = 0, step: int = 0, wandb=None, save_dir: str = './checkpoints', **kwargs):
    """保存或加载训练检查点。

    参数:
        lm_config: 模型配置（决定文件名中的 hidden_size / _moe 后缀）
        weight: 权重前缀名
        model: 传入时执行「保存」，为 None 时执行「加载」
        optimizer / epoch / step / wandb: 随 resume 检查点一并保存的训练状态
        save_dir: 检查点保存目录（默认 ./checkpoints）
        **kwargs: 额外随检查点保存的状态（如 scaler / ref_model / teacher_model）

    说明:
        - 保存：原子写入权重 .pth、resume 检查点与 config JSON；
        - 加载：读取 *_resume.pth，GPU 数量变化时自动按比例换算 step。
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id,
            'config': lm_config.to_dict(),
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        config_json_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.json'
        config_json_tmp = config_json_path + '.tmp'
        with open(config_json_tmp, 'w', encoding='utf-8') as f:
            json.dump(lm_config.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(config_json_tmp, config_json_path)

        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                data_config = ckp_data.get('data_config')
                if data_config and data_config.get('epoch_steps'):
                    data_config['epoch_steps'] = data_config['epoch_steps'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def pause_save_checkpoint(args, lm_config, *, weight, model, optimizer, epoch, step, scaler=None, wandb=None, lora_save=False, **lm_ckpt_kwargs) -> None:
    """暂停时保存检查点（与各训练脚本的周期保存语义完全一致）。

    参数:
        args: 训练参数（需含 save_dir，决定权重 .pth 的输出目录）
        lm_config: 模型配置（决定文件名中的 hidden_size / _moe 后缀）
        weight: 权重前缀名
        model: 当前模型（传入 lm_checkpoint 保存 resume 检查点）
        optimizer: 优化器
        epoch / step: 当前训练位置
        scaler: GradScaler（可为 None，PPO / GRPO / Agent 训练不传）
        wandb: 日志对象（可为 None）
        lora_save: 为 True 时仅保存 LoRA 分支权重（save_lora），不保存完整 state_dict
        **lm_ckpt_kwargs: 额外随 resume 检查点保存的状态（如 ref_model / teacher_model / scheduler / critic_model 等）

    说明:
        - 先写 out/ 下的权重 .pth（fp16，或 LoRA 专用权重），再经 lm_checkpoint 原子写入 resume 检查点；
        - 与周期保存语义一致：不冲刷梯度，由调用方以 is_main_process() 守护后再调用。
    """
    model.eval()
    if lora_save:
        from model.model_lora import save_lora
        moe_suffix = '_moe' if lm_config.use_moe else ''
        save_lora(model, f'{args.save_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}.pth')
    else:
        moe_suffix = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.save_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
        del state_dict
    lm_checkpoint(lm_config, weight=weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='./checkpoints', **lm_ckpt_kwargs)
    model.train()


def init_model(lm_config, from_weight: str = 'pretrain', tokenizer_path: str = './model', save_dir: str = './out', device: str = 'cuda') -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    architecture = getattr(lm_config, 'model_architecture', 'standard')
    _, model_cls = _architecture_classes(architecture)
    model = model_cls(lm_config)

    if from_weight != 'none':
        if from_weight.endswith('.pth'):
            weight_path = from_weight
        else:
            moe_suffix = '_moe' if lm_config.use_moe else ''
            weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        if architecture == 'looped':
            model.load_pretrained_weights(weights)
        else:
            model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    param_dtype = getattr(lm_config, 'param_dtype', 'fp32')
    if param_dtype != 'fp32':
        model = model.to({'bf16': torch.bfloat16, 'fp16': torch.float16}[param_dtype])
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """断点续训采样器：跳过前 skip_batches 个 batch，从上次中断处继续产出。

    参数:
        sampler: 底层采样器（DistributedSampler 或索引列表）
        batch_size: 每个 batch 的样本数
        skip_batches: 需要跳过的 batch 数（由上次中断的 step 换算而来）
    """

    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """通用对话模型奖励打分器（用于 PPO / GRPO 等 RL 阶段）。

    参数:
        model_path: HuggingFace 模型路径
        device: 运行设备
        dtype: 推理精度

    说明:
        get_score 拼接对话历史与回复后调用模型的 get_score 打分，
        并将结果裁剪到 [-3, 3] 区间。
    """

    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
