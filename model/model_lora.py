"""LoRA(Low-Rank Adaptation)低秩适配实现。

核心思想:冻结原始权重,仅在每个 ``nn.Linear`` 层旁并联一对低秩矩阵
``A``(in_features x rank)与 ``B``(rank x out_features),前向输出变为
``y = Wx + BAx``;微调时只更新 A/B,可训练参数量大幅减少。

关键设计:
- ``LoRA`` 层:``A`` 高斯初始化(mean=0, std=0.02)、``B`` 零初始化,
  初始 ``B@A = 0``,注入后模型输出与原模型逐位一致;
- ``apply_lora``:只对方阵 Linear 层(in_features == out_features)注入;
- ``save_lora`` / ``load_lora``:LoRA state_dict 存取(键形如 ``<层名>.lora.*``),
  兼容 DDP 的 ``module.`` 前缀与 ``torch.compile`` 的 ``_orig_mod`` 包装;
- ``merge_lora``:将 LoRA 分支合并回原权重(``W += B@A``),输出完整模型权重。

相关用法见 ``trainer/train_lora.py`` 与 ``eval_llm.py --lora_weight``。
"""
import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        """构造低秩分支:``A`` 高斯初始化、``B`` 零初始化,初始 ``B@A = 0``。"""
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """低秩分支前向:``y = B(A(x))``。"""
        return self.B(self.A(x))


def apply_lora(model: nn.Module, rank: int = 16) -> None:
    """为模型中方阵 ``nn.Linear`` 层(in_features == out_features)注入 LoRA 分支。

    注入后该层 forward 替换为 ``original(x) + lora(x)``(相加要求输入输出同维,
    故仅方阵层被注入);``rank`` 为低秩维度,默认 16。
    """
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model: nn.Module, path: str) -> None:
    """从 ``path`` 加载 LoRA state_dict 并写入模型各层的 ``lora`` 模块。

    兼容 DDP 保存的 ``module.`` 前缀;仅 ``hasattr(module, 'lora')`` 的层会被写入。
    """
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model: nn.Module, path: str) -> None:
    """收集模型全部 LoRA 分支权重(state_dict 键形如 ``<层名>.lora.*``,转 fp16)保存到 ``path``。

    兼容 ``torch.compile`` 的 ``_orig_mod`` 包装与 DDP 的 ``module.`` 前缀。
    """
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model: nn.Module, lora_path: str, save_path: str) -> None:
    """加载 LoRA 权重并合并进原权重(``W += B@A``),保存不含任何 ``.lora.`` 键的完整权重。

    适用于把微调结果固化为独立模型文件(如供 ``convert_model.py`` 转换)。
    """
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
