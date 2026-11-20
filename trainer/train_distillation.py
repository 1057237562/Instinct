"""
知识蒸馏脚本（Pipeline 可选阶段，白盒蒸馏）：学生模型向冻结的教师模型学习。

默认用 MoE 版 full_sft 蒸馏 Dense 版，也可用更大的 teacher_hidden_size 蒸馏更小学生。
损失 = alpha * CE + (1 - alpha) * KL(T)，教师仅前向提供软标签，
输出 {save_weight}_{hidden_size}.pth。
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
from model.compile_policy import training_compile_kwargs
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from scripts.data_loader.lm_dataset import SFTDataset
from trainer.trainer_cli import (
    build_autocast_ctx, build_trainer_parser,
    init_wandb_logger, set_cosine_lr, setup_dist_and_seed, step_with_scaler,
)
from trainer.packing_transition import packing_data_config, SequencePackingPlan
from trainer.trainer_utils import (
    Logger, lm_checkpoint, init_model,
    config_from_args, build_optimizer, checkpoint_after_update, checkpoint_due, wait_for_checkpoint,
    restore_config_from_checkpoint, apply_torchao_fp8_training,
    configure_bucket_memory_budget,
    prepare_lm_batch, release_compiled_cuda_memory,
)
from trainer.training_profiler import TrainingProfiler

warnings.filterwarnings('ignore')


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0, reduction: str = 'batchmean') -> torch.Tensor:
    """KL 蒸馏损失：学生分布向教师软标签分布对齐。

    参数:
        student_logits: 学生模型 logits
        teacher_logits: 教师模型 logits（no_grad，仅作软标签）
        temperature: 蒸馏温度（软化分布）
        reduction: KL 损失的 reduction 方式

    返回:
        经 temperature^2 缩放后的 KL 损失
    """
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student,
                start_step=0, wandb=None, alpha=0.0, temperature=1.0,
                packing_plan=None):
    start_time = time.time()
    data_config['epoch_steps'] = int(iters)

    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, batch in enumerate(loader, start=start_step + 1):
        bucket_status = packing_plan.observe_batch(
            batch, epoch=epoch, step=step,
        ) if packing_plan is not None else ''
        input_ids, labels = batch[:2]
        profiler.begin_step(tokens=input_ids.numel(), useful_tokens=(labels != -100).sum().item())
        with profiler.phase("data_transfer"):
            input_ids, labels, sequence_ids = prepare_lm_batch(batch, args.device)
        loss_mask = (labels[..., 1:] != -100).float()
        set_cosine_lr(optimizer, epoch, step, iters, args)

        # 前向传播（学生模型）
        with profiler.phase("student_forward"):
            with autocast_ctx:
                res = model(input_ids, sequence_ids=sequence_ids)
                student_logits = res.logits[..., :-1, :].contiguous()

        # 教师模型前向传播（只在eval & no_grad）
        if teacher_model is not None:
            with profiler.phase("teacher_forward"):
                with torch.no_grad():
                    teacher_logits = teacher_model(
                        input_ids, sequence_ids=sequence_ids
                    ).logits[..., :-1, :].contiguous()
                    vocab_size_student = student_logits.size(-1)
                    teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== 计算损失 ==========
        # 1) Ground-Truth CE Loss
        shift_labels = labels[..., 1:].contiguous()
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
        else: ce_loss = ce_loss_raw

        # 2) Distillation Loss
        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = alpha * CE + (1-alpha) * Distill
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        with profiler.phase("backward"):
            scaler.scale(loss).backward()

        optimizer_boundary = step % args.accumulation_steps == 0 or step == iters
        if optimizer_boundary:
            with profiler.phase("optimizer"):
                step_with_scaler(scaler, optimizer, model.parameters(), args.grad_clip)

        profile_metrics = profiler.end_step()
        if profile_metrics and wandb:
            wandb.log(profile_metrics)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) / 60
            elapsed_min = spend_time / 60

            bucket_text = f', {bucket_status}' if bucket_status else ''
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}{bucket_text}, epoch_time: {eta_min:.3f}min, elapsed_time: {elapsed_min:.3f}min')

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min,
                    "elapsed_time": elapsed_min,
                })

        del input_ids, labels, sequence_ids, loss_mask, res, student_logits, ce_loss, distill_loss, loss

        if (
            args.use_compile == 1
            and packing_plan is not None
            and packing_plan.should_release_large_cuda_memory(epoch=epoch, step=step)
        ):
            release_compiled_cuda_memory(
                f'epoch={epoch + 1}, completed buckets >{args.bucket_large_threshold} tokens'
            )

        if optimizer_boundary:
            checkpoint_after_update(
                args, lm_config_student, profiler, periodic=step == iters or checkpoint_due(args, step, start_step),
                weight=args.save_weight, model=model, optimizer=optimizer,
                epoch=epoch, step=step, wandb=wandb,
                scaler=scaler, teacher_model=teacher_model,
                teacher_config=lm_config_teacher.to_dict(), data_config=data_config,
            )


if __name__ == "__main__":
    # 模拟用moe模型蒸馏dense模型，也可以用更大teacher_hidden_size模型蒸馏更小student_hidden_size的
    parser = build_trainer_parser(
        description="Instinct Knowledge Distillation",
        defaults={
            'save_weight': 'full_dist',
            'epochs': 6,
            'batch_size': 32,
            'learning_rate': 5e-6,
            'accumulation_steps': 1,
            'save_interval': 100,
            'max_seq_len': 340,
            'wandb_project': 'Instinct-Distillation',
        },
    )
    parser.add_argument("--data_path", type=str, default="./dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--student_hidden_size', default=768, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1], help="学生模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    args = parser.parse_args()

    # 1. 初始化环境和随机种子
    local_rank = setup_dist_and_seed(args)

    # 2. 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config_student = config_from_args(args, hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.student_use_moe))
    lm_config_teacher = config_from_args(args, hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.teacher_use_moe))
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume==1 else None
    data_config = packing_data_config(args)
    lm_config_student = restore_config_from_checkpoint(lm_config_student, ckp_data)
    lm_config_teacher = restore_config_from_checkpoint(
        lm_config_teacher, ckp_data,
        config_key="teacher_config", fallback_topology_key="config",
    )

    # 3. 混合精度上下文
    autocast_ctx = build_autocast_ctx(args)

    # 4. wandb 日志
    wandb = init_wandb_logger(args, ckp_data, run_name=f"Instinct-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}")
    # 5. 定义学生和教师模型
    model, tokenizer = init_model(lm_config_student, 'none' if ckp_data else args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    if ckp_data and 'teacher_model' in ckp_data:
        teacher_model, _ = init_model(lm_config_teacher, 'none', device=args.device)
    else:
        teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    configure_bucket_memory_budget(
        model, args, extra_models=(teacher_model,), checkpoint_data=ckp_data,
    )
    data_config = packing_data_config(args)
    packing_plan = SequencePackingPlan(
        args, ckp_data,
        lambda packing, sample_indices=None: SFTDataset(
            args.data_path, tokenizer,
            max_length=(min(lm_config_student.max_position_embeddings,
                            args.bucket_max_seq_len)
                        if packing and args.sequence_packing_mode == 'bucket'
                        else args.max_seq_len),
            packing=packing,
            packing_batch_size=args.packing_batch_size,
            packing_mode=args.sequence_packing_mode,
            seq_bucket=args.seq_bucket,
            packing_num_proc=args.packing_num_proc,
            bucket_gpu_memory_gb=args.bucket_gpu_memory_gb,
            bucket_token_budget_override=getattr(args, 'bucket_token_budget', None),
            sample_indices=sample_indices,
        ),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = build_optimizer(model.named_parameters(), lr=args.learning_rate, optimizer=args.optimizer)

    # 6. 从ckp恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        if 'teacher_model' in ckp_data:
            teacher_model.load_state_dict(ckp_data['teacher_model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    model = apply_torchao_fp8_training(model, args, label="student")

    # 7. 编译和分布式包装
    if args.use_compile == 1:
        model = torch.compile(
            model, **training_compile_kwargs(model, mode=args.compile_mode),
            dynamic=getattr(lm_config_student, 'residual_type', 'standard') == 'attnres',
        )
        Logger(f"torch.compile enabled; cache={os.environ['TORCHINDUCTOR_CACHE_DIR']}")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    profiler = TrainingProfiler(args, name="distillation")

    # 8. 开始训练
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
        loader = DataLoader(
            train_ds, batch_sampler=batch_sampler,
            num_workers=packing_plan.loader_num_workers(), pin_memory=True,
        )
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature, packing_plan)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature, packing_plan)
        if (
            args.use_compile == 1
            and epoch + 1 < args.epochs
            and packing_plan.has_large_phase(epoch)
        ):
            release_compiled_cuda_memory(
                f'epoch={epoch + 1} complete, preparing next long-bucket phase'
            )

    wait_for_checkpoint()
    final_profile_metrics = profiler.finish()
    if final_profile_metrics and wandb:
        wandb.log(final_profile_metrics)

    # 9. 清理分布进程
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
