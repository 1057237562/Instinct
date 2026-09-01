"""
DPO（Direct Preference Optimization）离线偏好优化脚本（Pipeline 可选阶段）。

在 dpo.jsonl 的偏好对（chosen / rejected）上优化策略模型：以冻结的参考模型为基线，
损失 = -log sigmoid(beta * (pi_logratios - ref_logratios))，拉大偏好对的似然差。
学习率建议很小（默认 4e-8）以避免遗忘既有能力，输出 {save_weight}_{hidden_size}.pth。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "huggingface"))
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, pause_save_checkpoint,
    setup_seed, init_model, SkipBatchSampler, config_from_args, build_optimizer,
    restore_config_from_checkpoint, apply_torchao_fp8_training,
)
from trainer.trainer_cli import (
    build_trainer_parser, setup_dist_and_seed, build_autocast_ctx,
    init_wandb_logger, set_cosine_lr, step_with_scaler, flush_remaining_grad,
    PAUSE_EXIT_CODE, pause_requested, clear_pause_request,
)
from trainer.training_profiler import TrainingProfiler

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """按 labels 位置从 logits 中收集逐 token 对数概率。

    参数:
        logits: 模型输出 logits，shape (batch_size, seq_len, vocab_size)
        labels: 目标 token id，shape (batch_size, seq_len)

    返回:
        逐 token 对数概率，shape (batch_size, seq_len)
    """
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs: torch.Tensor, policy_log_probs: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    """DPO 损失：基于参考模型与策略模型的偏好对对数概率差。

    参数:
        ref_log_probs: 参考模型逐 token 对数概率，(batch_size, seq_len)
        policy_log_probs: 策略模型逐 token 对数概率，(batch_size, seq_len)
        mask: 有效 token 掩码（1=参与计算）
        beta: DPO 温度系数（越大越强调与参考模型的偏离）

    返回:
        标量损失（chosen 与 rejected 各占 batch 前/后一半）
    """
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch: int, loader: DataLoader, iters: int, ref_model, lm_config, start_step: int = 0, wandb=None, beta: float = 0.1) -> None:
    """执行一个 epoch 的 DPO 优化循环。

    参数:
        epoch: 当前轮数（从 0 起）
        loader: 数据加载器
        iters: 本 epoch 总步数（含断点续训跳过的步数）
        ref_model: 冻结的参考模型（no_grad 提供基线对数概率）
        lm_config: 模型配置（用于保存时的文件名后缀）
        start_step: 断点续训的起始步数（跳过前 start_step 步）
        wandb: 日志对象（swanlab / wandb，可选）
        beta: DPO 温度系数

    说明:
        每个 step 依次完成：chosen/rejected 拼接 → 余弦 LR 调度 → 参考模型与策略
        模型前向（autocast）→ DPO 损失 → 反向（GradScaler）→ 累积后更新参数
        → 周期性打日志 / 存检查点。
    """
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        profiler.begin_step(
            tokens=batch['x_chosen'].numel() + batch['x_rejected'].numel(),
            useful_tokens=int(batch['mask_chosen'].sum().item() + batch['mask_rejected'].sum().item()),
        )
        with profiler.phase("data_transfer"):
            x_chosen = batch['x_chosen'].to(args.device)
            x_rejected = batch['x_rejected'].to(args.device)
            y_chosen = batch['y_chosen'].to(args.device)
            y_rejected = batch['y_rejected'].to(args.device)
            mask_chosen = batch['mask_chosen'].to(args.device)
            mask_rejected = batch['mask_rejected'].to(args.device)
            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        set_cosine_lr(optimizer, epoch, step, iters, args)

        with autocast_ctx:
            with profiler.phase("reference_forward"):
                with torch.no_grad():
                    ref_outputs = ref_model(x)
                    ref_logits = ref_outputs.logits
                ref_log_probs = logits_to_log_probs(ref_logits, y)

            with profiler.phase("policy_forward"):
                outputs = model(x)
                logits = outputs.logits
                policy_log_probs = logits_to_log_probs(logits, y)

                dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
                loss = dpo_loss_val + outputs.aux_loss
                loss = loss / args.accumulation_steps

        with profiler.phase("backward"):
            scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            with profiler.phase("optimizer"):
                step_with_scaler(scaler, optimizer, model.parameters(), args.grad_clip)

        profile_metrics = profiler.end_step()
        if profile_metrics and wandb:
            wandb.log(profile_metrics)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')

            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='./checkpoints', ref_model=ref_model)
            model.train()
            del state_dict

        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss

        if pause_requested(args):
            clear_pause_request(args)
            profiler.finish()
            if is_main_process():
                pause_save_checkpoint(args, lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, ref_model=ref_model)
            Logger('[PAUSED] Training paused — resume checkpoint saved.')
            sys.exit(PAUSE_EXIT_CODE)

    flush_remaining_grad(scaler, optimizer, model.parameters(), args.grad_clip, last_step, start_step, args.accumulation_steps)


if __name__ == "__main__":
    parser = build_trainer_parser(
        description="Instinct DPO (Direct Preference Optimization)",
        defaults={
            'save_weight': 'dpo',
            'epochs': 1,
            'batch_size': 4,
            'learning_rate': 4e-8,
            'accumulation_steps': 1,
            'save_interval': 100,
            'max_seq_len': 1024,
            'from_weight': 'full_sft',
            'wandb_project': 'Instinct-DPO',
        },
    )
    parser.add_argument("--data_path", type=str, default="./dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    args = parser.parse_args()

    # 1. 初始化环境和随机种子
    local_rank = setup_dist_and_seed(args)

    # 2. 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = config_from_args(args)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume==1 else None
    lm_config = restore_config_from_checkpoint(lm_config, ckp_data)

    # 3. 设置混合精度
    autocast_ctx = build_autocast_ctx(args)

    # 4. 配wandb
    wandb = init_wandb_logger(args, ckp_data, run_name=f"Instinct-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}")

    # 5. 定义模型和参考模型
    if ckp_data and 'ref_model' in ckp_data:
        base_weight = 'none'
        ref_from_ckp = True
    else:
        base_weight = args.from_weight
        ref_from_ckp = False
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 初始化参考模型（ref_model冻结）
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = build_optimizer(model.named_parameters(), lr=args.learning_rate, optimizer=args.optimizer)

    # 6. 从ckp恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        if ref_from_ckp:
            ref_model.load_state_dict(ckp_data['ref_model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    model = apply_torchao_fp8_training(model, args, label="policy")

    # 7. 编译和分布式包装
    if args.use_compile == 1:
        model = torch.compile(
            model, mode=args.compile_mode,
            dynamic=getattr(lm_config, 'residual_type', 'standard') == 'attnres',
        )
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    profiler = TrainingProfiler(args, name="dpo")

    # 8. 开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    final_profile_metrics = profiler.finish()
    if final_profile_metrics and wandb:
        wandb.log(final_profile_metrics)

    # 9. 清理分布进程
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
