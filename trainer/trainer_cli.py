"""
训练 CLI 公共工具：提取 8 个训练脚本中重复的参数解析 / 环境初始化 / 调度 / 更新样板代码。

集中管理训练脚本共用的 argparse 参数、分布式与随机种子初始化、混合精度上下文、
wandb（SwanLab）日志初始化、余弦 LR 调度、GradScaler 参数更新与 epoch 末尾的
残余梯度冲刷，供后续各 train_*.py 重构后统一调用，保证行为与现有内联代码完全一致。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.distributed as dist
from contextlib import nullcontext
from trainer.trainer_utils import (
    Logger, get_lr, is_main_process, init_distributed_mode, setup_seed,
)

# 暂停退出码：与 0=成功 / 非0=失败 相区分，供 WebUI 识别“已暂停”状态。
PAUSE_EXIT_CODE: int = 42


def build_trainer_parser(description: str, *, defaults: dict | None = None) -> argparse.ArgumentParser:
    """构建训练脚本共用的命令行参数解析器。

    参数:
        description: argparse 描述文本（各训练脚本传入自己的名称）
        defaults: 各训练脚本的默认值覆盖（save_weight / epochs / batch_size / learning_rate 等按脚本不同）

    返回:
        已注册全部公共参数的 ArgumentParser；调用方随后自行 add_argument 脚本专属参数
        （如 --data_path / --beta / --lora_name / --student_hidden_size 等）并 parse_args。

    说明:
        公共参数与 train_pretrain.py 的登记保持一致（参数名 / 默认值 / choices / help 逐字照搬），
        训练脚本差异仅通过 defaults 覆盖默认值，不改变参数语义。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--save_dir", type=str, default="./out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adafactor", "muon"], help="优化器类型（adamw / adafactor / muon）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="激活层计算精度（bfloat16/float16/fp32）")
    parser.add_argument("--param_dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"], help="模型参数精度（fp32=主权重，bf16/fp16=训练时权重直接 cast）")
    parser.add_argument("--kv_cache_dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2"], help="KV Cache 精度（fp8 时缓存量化，decode 带宽减半）")
    parser.add_argument(
        "--fp8_training", type=str, default="off",
        choices=["off", "tensorwise", "rowwise", "rowwise_with_gw_hp"],
        help="TorchAO FP8 Linear 训练方案（off=关闭；tensorwise=最快；rowwise=更稳健）",
    )
    parser.add_argument(
        "--fp8_filter", type=str, default="auto", choices=["auto", "eligible"],
        help="FP8 Linear 筛选（auto=跳过预计无加速的小 GEMM；eligible=转换全部尺寸兼容层）",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument(
        '--seq_bucket', default=2, type=int,
        help='Sequence packing 自动长度桶数量；使用排序 + DP + 斜率优化求桶边界',
    )
    parser.add_argument(
        '--sequence_packing_mode', default='fixed', choices=['fixed', 'bucket'],
        help='Packing 形式：fixed=原固定长度；bucket=实验性自动长度桶',
    )
    parser.add_argument(
        '--sequence_packing', '--packing', dest='sequence_packing',
        default=0, type=int, choices=[0, 1],
        help='Pretrain/SFT 序列 packing（1=把完整样本装入定长 block，显著减少 padding）',
    )
    parser.add_argument(
        '--packing_batch_size', default=1000, type=int,
        help='首次构建 packing Arrow cache 时每批处理的原始样本数',
    )
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_looped', default=0, type=int, choices=[0, 1], help="是否使用LoopUS循环架构（0=否，1=是）")
    parser.add_argument('--model_architecture', default=None, choices=['standard', 'linear', 'looped'], help="模型主干（默认读取config；--use_looped 1仍可兼容切换LoopUS）")
    parser.add_argument('--residual_type', default=None, choices=['standard', 'mhc', 'attnres'], help="残差拓扑（默认读取config或standard）")
    parser.add_argument('--hc_mult', default=None, type=int, help="mHC并行残差流数量")
    parser.add_argument('--hc_sinkhorn_iters', default=None, type=int, help="mHC Sinkhorn-Knopp迭代次数")
    parser.add_argument('--hc_eps', default=None, type=float, help="mHC数值稳定项")
    parser.add_argument('--attnres_variant', default=None, choices=['full', 'block'], help="Attention Residuals变体")
    parser.add_argument('--attnres_block_size', default=None, type=int, help="Block AttnRes块大小（按Attention/MLP子层计）")
    parser.add_argument('--depth_reward', default=-1.0, type=float, help="深度reward权重λ（>0时覆盖config，-1用config默认；可随训练退火）")
    parser.add_argument('--distill_weight', default=-1.0, type=float, help="自蒸馏权重（>0时启用：浅层循环深度向最终深度输出分布学习；建议配合exit_in_training=0跑满循环）")
    parser.add_argument('--distill_temperature', default=2.0, type=float, help="自蒸馏温度T（软化教师/学生分布）")
    parser.add_argument('--teacher_stop_grad', default=1, type=int, choices=[0, 1], help="教师logits是否stop-grad（1=是，0=否）")
    parser.add_argument('--depth_gain_reward', default=0.0, type=float, help="深度增益奖励权重（>0时启用：仅当更深步相对第1步基线降低LM损失时给予正奖励）")
    parser.add_argument('--exit_in_training', default=-1, type=int, choices=[-1, 0, 1], help="训练中是否允许早退（-1用config默认；自蒸馏建议0=跑满循环）")
    parser.add_argument('--n_supervision', default=-1, type=int, help="随机深度监督步数（-1用config默认；自蒸馏可提高以覆盖更多深度）")
    parser.add_argument('--early_exit', default=0, type=int, choices=[0, 1], help="启用Early Exit训练（0=否，1=是）")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="Instinct-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--use_grad_checkpoint", default=0, type=int, choices=[0, 1, 2], help="梯度检查点模式（0=关闭, 1=选择性重算注意力/FFN, 2=整层checkpoint）")
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], help="torch.compile 模式（default=Triton 编译；reduce-overhead=叠加 CUDA graph，小模型首选；max-autotune=极限调优，编译极慢）")
    parser.add_argument(
        "--profile", type=str, default="off", choices=["off", "timing", "torch"],
        help="训练性能分析（timing=低开销阶段计时；torch=额外导出短窗口算子 trace）",
    )
    parser.add_argument("--profile_interval", type=int, default=100, help="timing profiler 汇总间隔")
    parser.add_argument("--profile_warmup", type=int, default=10, help="启动/续训后跳过的 profiler 预热步数")
    parser.add_argument("--profile_active_steps", type=int, default=5, help="torch profiler trace 采集步数")
    parser.add_argument("--profile_dir", type=str, default="./profiler_traces", help="PyTorch profiler trace 输出目录")
    parser.add_argument('--config_path', default='', type=str, help="JSON配置文件路径")
    parser.add_argument('--pause_file', type=str, default='./checkpoints/.pause_request', help='暂停请求标记文件；存在时在下个 step 边界保存检查点并退出(码42)')
    if defaults is not None:
        parser.set_defaults(**defaults)
    return parser


def pause_requested(args) -> bool:
    """检查是否存在暂停请求标记文件（--pause_file），存在则训练应在下个 step 边界暂停。

    参数:
        args: argparse 解析出的命令行参数对象（需含 pause_file）

    返回:
        标记文件存在返回 True，否则返回 False。
    """
    return os.path.exists(args.pause_file)


def clear_pause_request(args) -> None:
    """删除暂停请求标记文件（--pause_file），防止残留标记在下次启动时误触发暂停。

    参数:
        args: argparse 解析出的命令行参数对象（需含 pause_file）

    说明:
        标记文件已不存在时静默通过（FileNotFoundError 忽略），保证重复清理 / 陈旧标记安全。
    """
    try:
        os.remove(args.pause_file)
    except FileNotFoundError:
        pass


def setup_dist_and_seed(args) -> int:
    """初始化分布式环境与随机种子，返回本地 rank（local_rank）。

    参数:
        args: argparse 解析出的命令行参数对象（DDP 时其 device 会被改写为 cuda:{local_rank}）

    返回:
        local_rank：非 DDP 模式为 0；DDP 模式下为当前进程的 LOCAL_RANK。

    说明:
        先 init_distributed_mode() 初始化进程组；已初始化时把 args.device 指向当前卡；
        随机种子 = 42 + 全局 rank，保证各进程采样一致。
    """
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    return local_rank


def build_autocast_ctx(args):
    """按 args.device / args.dtype 构建混合精度上下文管理器。

    参数:
        args: argparse 解析出的命令行参数对象（需含 device / dtype）

    返回:
        CPU 或 fp32 时返回 nullcontext()，否则返回 torch.cuda.amp.autocast(dtype=dtype)。
    """
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'fp32': torch.float32}[args.dtype]
    return nullcontext() if device_type == "cpu" or args.dtype == "fp32" else torch.cuda.amp.autocast(dtype=dtype)


def init_wandb_logger(args, ckp_data=None, *, project=None, run_name=None):
    """按需初始化 wandb（SwanLab）日志器。

    参数:
        args: argparse 解析出的命令行参数对象（需含 use_wandb / wandb_project / epochs / batch_size / learning_rate）
        ckp_data: resume 检查点数据（含 'wandb_id' 时按 must 模式续跑同一实验）
        project: 覆盖 wandb_project 的显式项目名（默认 args.wandb_project）
        run_name: 覆盖默认实验名的显式名称（默认 Instinct-Epoch-{epochs}-BatchSize-{batch_size}-LearningRate-{learning_rate}）

    返回:
        未启用 wandb 或非主进程时返回 None；否则返回已 init 的 swanlab 模块对象。
    """
    if not args.use_wandb or not is_main_process():
        return None
    import swanlab as wandb
    wandb_id = ckp_data.get('wandb_id') if ckp_data else None
    resume = 'must' if wandb_id else None
    name = run_name or f"Instinct-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    wandb.init(project=project or args.wandb_project, name=name, id=wandb_id, resume=resume)
    return wandb


def set_cosine_lr(optimizer, epoch, step, iters, args) -> None:
    """余弦退火学习率调度：按全局步数更新优化器各参数组的 lr。

    参数:
        optimizer: 优化器（遍历其 param_groups 写回 'lr'）
        epoch: 当前轮数（从 0 起）
        step: 本 epoch 内步数（从 1 起）
        iters: 本 epoch 总步数（含断点续训跳过的步数）
        args: argparse 解析出的命令行参数对象（需含 epochs / learning_rate）
    """
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def step_with_scaler(scaler, optimizer, params, grad_clip) -> None:
    """GradScaler 完整参数更新：反缩放 → 裁剪梯度 → 更新参数 → 清空梯度。

    参数:
        scaler: torch.cuda.amp.GradScaler
        optimizer: 优化器
        params: 参与梯度裁剪的参数（如 model.parameters() 或 lora_params）
        grad_clip: 梯度裁剪阈值
    """
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def flush_remaining_grad(scaler, optimizer, params, grad_clip, last_step, start_step, accumulation_steps) -> None:
    """epoch 末尾冲刷未累积满的残余梯度。

    参数:
        scaler: torch.cuda.amp.GradScaler
        optimizer: 优化器
        params: 参与梯度裁剪的参数
        grad_clip: 梯度裁剪阈值
        last_step: 本 epoch 最后一个 step
        start_step: 断点续训的起始步数（跳过前 start_step 步）
        accumulation_steps: 梯度累积步数

    说明:
        守护式调用：仅当本 epoch 实际推进了步数（last_step > start_step）且最后一步
        未凑满 accumulation_steps 时，才冲刷残余梯度并更新参数；否则为 no-op。
    """
    if last_step > start_step and last_step % accumulation_steps != 0:
        step_with_scaler(scaler, optimizer, params, grad_clip)
