"""
全量指令微调（Full SFT）脚本（Pipeline 第 2 阶段，必须）：在对话语料上微调预训练权重。

使用 sft_t2t(_mini).jsonl（OpenAI 对话格式，含工具调用样本）对
--from_weight pretrain 的基座做全参数微调，让模型学会遵循指令与工具调用，
输出 {save_weight}_{hidden_size}.pth，作为 LoRA / DPO / 蒸馏等阶段的基座。
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, pause_save_checkpoint,
    setup_seed, init_model, SkipBatchSampler, config_from_args, build_optimizer,
    restore_config_from_checkpoint, apply_torchao_fp8_training,
    prepare_lm_batch,
)
from trainer.trainer_cli import (
    build_trainer_parser, setup_dist_and_seed, build_autocast_ctx,
    init_wandb_logger, set_cosine_lr, step_with_scaler, flush_remaining_grad,
    PAUSE_EXIT_CODE, pause_requested, clear_pause_request,
)
from trainer.packing_transition import packing_data_config, SequencePackingPlan
from trainer.training_profiler import TrainingProfiler

warnings.filterwarnings('ignore')


def train_epoch(epoch: int, loader: DataLoader, iters: int, start_step: int = 0, wandb=None) -> None:
    """执行一个 epoch 的全量微调循环。

    参数:
        epoch: 当前轮数（从 0 起）
        loader: 数据加载器
        iters: 本 epoch 总步数（含断点续训跳过的步数）
        start_step: 断点续训的起始步数（跳过前 start_step 步）
        wandb: 日志对象（swanlab / wandb，可选）

    说明:
        每个 step 依次完成：余弦 LR 调度 → 前向（autocast）→ 反向（GradScaler）
        → 按 accumulation_steps 累积后裁剪梯度并更新参数 → 周期性打日志 / 存检查点。
    """
    start_time = time.time()
    data_config['epoch_steps'] = int(iters)
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        input_ids, labels = batch[:2]
        profiler.begin_step(tokens=input_ids.numel(), useful_tokens=(labels != -100).sum().item())
        with profiler.phase("data_transfer"):
            input_ids, labels, sequence_ids = prepare_lm_batch(batch, args.device)
        last_step = step
        set_cosine_lr(optimizer, epoch, step, iters, args)

        with profiler.phase("forward"):
            with autocast_ctx:
                res = model(input_ids, labels=labels, sequence_ids=sequence_ids)
                loss = res.loss + res.aux_loss
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
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='./checkpoints', scaler=scaler,
                         data_config=data_config)
            model.train()
            del state_dict

        del input_ids, labels, sequence_ids, res, loss

        if pause_requested(args):
            clear_pause_request(args)
            profiler.finish()
            if is_main_process():
                pause_save_checkpoint(args, lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, data_config=data_config)
            Logger('[PAUSED] Training paused — resume checkpoint saved.')
            sys.exit(PAUSE_EXIT_CODE)

    flush_remaining_grad(scaler, optimizer, model.parameters(), args.grad_clip, last_step, start_step, args.accumulation_steps)


if __name__ == "__main__":
    parser = build_trainer_parser(
        description="Instinct Full SFT",
        defaults={
            'save_weight': 'full_sft',
            'batch_size': 16,
            'learning_rate': 1e-5,
            'accumulation_steps': 1,
            'max_seq_len': 768,
            'from_weight': 'pretrain',
            'wandb_project': 'Instinct-Full-SFT',
        },
    )
    parser.add_argument("--data_path", type=str, default="./dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    args = parser.parse_args()

    # 1. 初始化环境和随机种子
    local_rank = setup_dist_and_seed(args)
    # 2. 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = config_from_args(args)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume==1 else None
    data_config = packing_data_config(args)
    lm_config = restore_config_from_checkpoint(lm_config, ckp_data)

    # 3. 设置混合精度
    autocast_ctx = build_autocast_ctx(args)
    # 4. 配wandb
    wandb = init_wandb_logger(args, ckp_data, run_name=f"Instinct-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}")

    # 5. 定义模型、数据、优化器
    # Resume 检查点已包含完整模型状态，无需再加载 --from_weight 基础权重
    model, tokenizer = init_model(lm_config, 'none' if ckp_data else args.from_weight, device=args.device)
    packing_plan = SequencePackingPlan(
        args, ckp_data,
        lambda packing, sample_indices=None: SFTDataset(
            args.data_path, tokenizer, max_length=args.max_seq_len,
            packing=packing,
            packing_batch_size=args.packing_batch_size,
            sample_indices=sample_indices,
        ),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = build_optimizer(model.named_parameters(), lr=args.learning_rate, optimizer=args.optimizer)

    # 6. 从ckp恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    model = apply_torchao_fp8_training(model, args)

    # 7. 编译和分布式包装
    if args.use_compile == 1:
        model = torch.compile(
            model, mode=args.compile_mode,
            dynamic=getattr(lm_config, 'residual_type', 'standard') == 'attnres',
        )
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    profiler = TrainingProfiler(args, name="full_sft")

    # 8. 开始训练
    for epoch in range(start_epoch, args.epochs):
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        train_ds, transition_batches, active_packing = packing_plan.epoch_data(
            epoch, skip, args.batch_size,
        )
        packing_plan.update_checkpoint_config(data_config, epoch=epoch, active=active_packing)
        if transition_batches is None:
            train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
            train_sampler and train_sampler.set_epoch(epoch)
            setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
            batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        else:
            batch_sampler = transition_batches
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    final_profile_metrics = profiler.finish()
    if final_profile_metrics and wandb:
        wandb.log(final_profile_metrics)

    # 9. 清理分布进程
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
