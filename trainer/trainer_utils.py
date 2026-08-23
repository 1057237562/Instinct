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
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_instinct import InstinctForCausalLM, InstinctConfig
from model.model_instinct_loop import InstinctConfig as LoopedInstinctConfig, InstinctForCausalLM as LoopedInstinctForCausalLM

def get_model_params(model, config):
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


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def config_from_args(args, **overrides):
    overrides.setdefault('param_dtype', getattr(args, 'param_dtype', 'fp32'))
    overrides.setdefault('kv_cache_dtype', getattr(args, 'kv_cache_dtype', 'fp32'))
    overrides.setdefault("use_grad_checkpoint", int(getattr(args, "use_grad_checkpoint", 0)))
    hidden_size = overrides.pop('hidden_size', getattr(args, 'hidden_size', 768))
    num_hidden_layers = overrides.pop('num_hidden_layers', getattr(args, 'num_hidden_layers', 8))
    use_moe = overrides.pop('use_moe', bool(getattr(args, 'use_moe', 0)))
    use_looped = overrides.pop('use_looped', bool(getattr(args, 'use_looped', 0)))
    config_path = getattr(args, 'config_path', None)
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg_dict = json.load(f)
        cfg_dict['hidden_size'] = hidden_size
        cfg_dict['num_hidden_layers'] = num_hidden_layers
        cfg_dict['use_moe'] = use_moe
        cfg_dict.update(overrides)
        use_looped = use_looped or cfg_dict.get('model_architecture') == 'looped'
        cfg_cls = LoopedInstinctConfig if use_looped else InstinctConfig
        return cfg_cls(**cfg_dict)
    cfg_cls = LoopedInstinctConfig if use_looped else InstinctConfig
    return cfg_cls(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
        **overrides,
    )


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
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


def build_optimizer(params, lr, optimizer='adamw', **kwargs):
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


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='./checkpoints', **kwargs):
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
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='./model', save_dir='./out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if isinstance(lm_config, LoopedInstinctConfig):
        model = LoopedInstinctForCausalLM(lm_config)
    else:
        model = InstinctForCausalLM(lm_config)

    if from_weight != 'none':
        if from_weight.endswith('.pth'):
            weight_path = from_weight
        else:
            moe_suffix = '_moe' if lm_config.use_moe else ''
            weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        if isinstance(model, LoopedInstinctForCausalLM):
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