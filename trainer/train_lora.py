"""
LoRA 低秩微调脚本（Pipeline 第 3 阶段，可选）：冻结主干，仅训练低秩适配矩阵。

在垂直领域数据（如 lora_medical.jsonl）上微调 --from_weight 基座（默认 full_sft），
LoRA 参数量占比极小、CPU 亦可运行；只保存 LoRA 权重，推理时叠加到基座模型上。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trainer.compile_cache  # noqa: F401  # configure Inductor before torch import
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "huggingface"))
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import time
import warnings
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, pause_save_checkpoint,
    init_model, config_from_args, build_optimizer,
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


def train_epoch(epoch: int, loader: DataLoader, iters: int, lora_params: list, start_step: int = 0, wandb=None) -> None:
    """执行一个 epoch 的 LoRA 微调循环。

    参数:
        epoch: 当前轮数（从 0 起）
        loader: 数据加载器
        iters: 本 epoch 总步数（含断点续训跳过的步数）
        lora_params: 待训练的 LoRA 参数列表（仅用于梯度裁剪）
        start_step: 断点续训的起始步数（跳过前 start_step 步）
        wandb: 日志对象（swanlab / wandb，可选）

    说明:
        每个 step 依次完成：余弦 LR 调度 → 前向（autocast）→ 反向（GradScaler）
        → 按 accumulation_steps 累积后裁剪梯度并更新参数 → 周期性打日志 / 存 LoRA 权重。
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
                step_with_scaler(scaler, optimizer, lora_params, args.grad_clip)

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
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
            # LoRA只保存LoRA权重
            save_lora(model, lora_save_path)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='./checkpoints', data_config=data_config)
            model.train()

        del input_ids, labels, sequence_ids, res, loss

        if pause_requested(args):
            clear_pause_request(args)
            profiler.finish()
            if is_main_process():
                pause_save_checkpoint(args, lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, lora_save=True, data_config=data_config)
            Logger('[PAUSED] Training paused — LoRA checkpoint saved.')
            sys.exit(PAUSE_EXIT_CODE)

    flush_remaining_grad(scaler, optimizer, lora_params, args.grad_clip, last_step, start_step, args.accumulation_steps)

if __name__ == "__main__":
    parser = build_trainer_parser(
        description="Instinct LoRA Fine-tuning",
        defaults={
            'save_weight': 'lora',
            'epochs': 10,
            'learning_rate': 1e-4,
            'accumulation_steps': 1,
            'log_interval': 10,
            'from_weight': 'full_sft',
            'wandb_project': 'Instinct-LoRA',
        },
    )
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--data_path", type=str, default="./dataset/lora_medical.jsonl", help="LoRA训练数据路径")
    args = parser.parse_args()

    # 1. 初始化环境和随机种子
    local_rank = setup_dist_and_seed(args)

    # 2. 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = config_from_args(args)
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='./checkpoints') if args.from_resume==1 else None
    data_config = packing_data_config(args)
    lm_config = restore_config_from_checkpoint(lm_config, ckp_data)

    # 3. 设置混合精度
    autocast_ctx = build_autocast_ctx(args)

    # 4. 配wandb
    wandb_run_name = f"Instinct-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
    wandb = init_wandb_logger(args, ckp_data, run_name=wandb_run_name)

    # 5. 定义模型、应用LoRA、冻结非LoRA参数
    model, tokenizer = init_model(lm_config, 'none' if ckp_data else args.from_weight, device=args.device)
    if args.use_compile == 1:
        args.use_compile = 0
        Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
    model = apply_torchao_fp8_training(model, args, label="LoRA base")
    apply_lora(model)

    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    # 冻结非LoRA参数，收集LoRA参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    # 6. 定义数据和优化器
    packing_plan = SequencePackingPlan(
        args, ckp_data,
        lambda packing, sample_indices=None: SFTDataset(
            args.data_path, tokenizer,
            max_length=(lm_config.max_position_embeddings
                        if packing and args.sequence_packing_mode == 'bucket'
                        else args.max_seq_len),
            packing=packing,
            packing_batch_size=args.packing_batch_size,
            packing_mode=args.sequence_packing_mode,
            seq_bucket=args.seq_bucket,
            sample_indices=sample_indices,
        ),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = build_optimizer(lora_params, lr=args.learning_rate, optimizer=args.optimizer)

    # 7. 从ckp恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # 8. 分布式包装（LoRA monkey-patch forward 不使用 torch.compile）
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    profiler = TrainingProfiler(args, name="lora")

    # 9. 开始训练
    for epoch in range(start_epoch, args.epochs):
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        train_ds, transition_batches, active_packing = packing_plan.epoch_data(
            epoch, skip, args.batch_size,
        )
        packing_plan.update_checkpoint_config(data_config, epoch=epoch, active=active_packing)
        if transition_batches is None:
            batch_sampler = packing_plan.batch_sampler(
                train_ds, active_packing=active_packing, epoch=epoch,
                batch_size=args.batch_size, skip_batches=skip,
            )
        else:
            batch_sampler = transition_batches
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)

    final_profile_metrics = profiler.finish()
    if final_profile_metrics and wandb:
        wandb.log(final_profile_metrics)

    # 10. 清理分布进程
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
