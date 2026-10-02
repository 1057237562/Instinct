"""
预训练脚本（Pipeline 第 1 阶段，必须）：从零训练 Instinct 基础模型。

在纯文本语料（pretrain_t2t(_mini).jsonl）上做自回归 next-token 预测，
输出 {save_weight}_{hidden_size}.pth 权重，作为后续 SFT / LoRA / RL 等阶段的基座。
支持 DDP、梯度累积、混合精度（autocast + GradScaler）、断点续训，
以及 Looped 循环架构的可选深度奖励 / 自蒸馏。
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
from scripts.data_loader.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint,
    init_model, config_from_args, build_optimizer,
    pause_save_checkpoint, restore_config_from_checkpoint, apply_torchao_fp8_training,
    prepare_lm_batch, release_compiled_cuda_memory,
    configure_bucket_memory_budget,
)
from trainer.trainer_cli import (
    build_trainer_parser, setup_dist_and_seed, build_autocast_ctx, init_wandb_logger,
    set_cosine_lr, set_cosine_lr_progress, step_with_scaler, flush_remaining_grad,
    PAUSE_EXIT_CODE, pause_requested, clear_pause_request,
    add_moe_router_migration_arg, validate_moe_router_migration,
)
from trainer.packing_transition import packing_data_config, SequencePackingPlan
from trainer.training_profiler import TrainingProfiler
from trainer.moe_monitor import collect_moe_routing_stats
from trainer.streaming_pretrain import (
    ChunkedPackedEpochLoader,
    should_stream_pretrain,
    validate_streaming_budget,
    streaming_token_progress,
)
from scripts.data_loader.streaming_chunks import build_chunk_plan
from scripts.data_loader.sequence_bucket import packing_preprocess_workers
from scripts.data_loader.source_format import canonical_source_key

warnings.filterwarnings('ignore')


def _recompiled_resume_note(saved_data_config, streaming_plan, data_path):
    """Explain what a chunk cursor means on a rebuilt plan.

    Compiling the same corpus into another container (JSONL -> Parquet) rebuilds
    the chunk plan.  When the file was compiled with a matching
    ``--align-chunk-bytes`` its chunks cover exactly the rows the JSONL plan
    covered and the cursor keeps its meaning; otherwise the saved chunk index
    marks the same fraction of the corpus while its row range moves, so training
    replays or skips part of one chunk.  Resuming at an epoch boundary has no
    drift either way.  Returns the message to log, or ``None`` when the plan did
    not change.
    """
    saved_path = str(saved_data_config.get('data_path') or '')
    resume_chunk = int(saved_data_config.get('streaming_chunk_index', 0) or 0)
    resume_step = int(saved_data_config.get('streaming_chunk_step', 0) or 0)
    if not (resume_chunk or resume_step) or not saved_path:
        return None
    if canonical_source_key(saved_path) != canonical_source_key(data_path):
        return None  # A different corpus is rejected before training starts.
    if os.path.normcase(os.path.abspath(saved_path)) == os.path.normcase(
        os.path.abspath(data_path)
    ):
        return None
    chunks = streaming_plan['chunks']
    if resume_chunk >= len(chunks):
        return None
    saved_chunk_bytes = int(saved_data_config.get('streaming_chunk_mb', 0) or 0) * 1024 ** 2
    aligned = int((streaming_plan.get('identity') or {}).get('aligned_chunk_bytes') or 0)
    if aligned and aligned == saved_chunk_bytes:
        return (
            f'[Dataset Streaming] resuming on a recompiled corpus: the plan was '
            f'rebuilt for {os.path.basename(data_path)} ({len(chunks)} chunks) and '
            'its chunks are aligned to the JSONL chunk boundaries, so the saved '
            'cursor points at the same rows.'
        )
    row_start = sum(int(chunk['rows']) for chunk in chunks[:resume_chunk])
    return (
        f'[Dataset Streaming] resuming on a recompiled corpus: the plan was '
        f'rebuilt for {os.path.basename(data_path)} ({len(chunks)} chunks), so '
        f'saved cursor chunk {resume_chunk + 1} step {resume_step} now starts at '
        f'row {row_start:,} of {int(streaming_plan["rows"]):,}; part of one chunk '
        'is replayed or skipped.  Resuming at an epoch boundary has no drift, and '
        'compiling with --align-chunk-bytes makes the cursor exact.'
    )


def train_epoch(epoch: int, loader: DataLoader, iters: int, start_step: int = 0,
                wandb=None, packing_plan=None, token_schedule=None) -> int:
    """执行一个 epoch 的预训练循环。

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
    start_tokens = int(data_config.get('streaming_processed_tokens', 0))
    data_config['epoch_steps'] = int(iters)
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        bucket_status = packing_plan.observe_batch(
            batch, epoch=epoch, step=step,
        ) if packing_plan is not None else ''
        input_ids, labels = batch[:2]
        profiler.begin_step(
            tokens=input_ids.numel(), useful_tokens=(labels != -100).sum().item()
        )
        with profiler.phase("data_transfer"):
            input_ids, labels, sequence_ids = prepare_lm_batch(batch, args.device)
        last_step = step
        if token_schedule is None:
            set_cosine_lr(optimizer, epoch, step, iters, args)
        else:
            if len(batch) == 3:
                batch_tokens = (batch[2] >= 0).sum().to(args.device)
            else:
                batch_tokens = (batch[1] != -100).sum().to(args.device)
            if dist.is_initialized():
                dist.all_reduce(batch_tokens, op=dist.ReduceOp.SUM)
            consumed = int(batch_tokens.item())
            completed = int(data_config.get('streaming_processed_tokens', 0)) + consumed
            set_cosine_lr_progress(
                optimizer, completed, int(token_schedule['total_tokens']), args,
            )
            data_config['streaming_processed_tokens'] = completed

        with profiler.phase("forward"):
            with autocast_ctx:
                res = model(
                    input_ids, labels=labels, sequence_ids=sequence_ids,
                    early_exit=bool(args.early_exit),
                )
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

        if step % args.log_interval == 0 or (token_schedule is None and step == iters):
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) / 60
            progress_text = f'({step}/{iters})'
            if token_schedule is not None:
                epoch_tokens = int(token_schedule['total_tokens']) // args.epochs
                epoch_done, eta_min = streaming_token_progress(
                    completed, start_tokens, epoch_tokens, epoch, spend_time,
                )
                progress_text = (
                    f'({step}/~{iters}), tokens: {epoch_done}/{epoch_tokens}, '
                    f'chunk: {data_config.get("streaming_chunk_index", 0) + 1}/'
                    f'{data_config["streaming_chunks"]}'
                )
            elapsed_min = spend_time / 60
            loop_steps = getattr(res, 'recurrent_steps', None)
            loop_str = f', loop_steps: {loop_steps:.2f}' if loop_steps else ''
            bucket_text = f', {bucket_status}' if bucket_status else ''
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]{progress_text}, loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}{loop_str}, lr: {current_lr:.8f}{bucket_text}, epoch_time: {eta_min:.1f}min, elapsed_time: {elapsed_min:.1f}min')
            log_dict = {"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min, "elapsed_time": elapsed_min}
            if loop_steps: log_dict["loop_steps"] = loop_steps
            if lm_config.use_moe:
                routing_stats = collect_moe_routing_stats(model)
                if routing_stats is not None:
                    Logger(routing_stats.format_line())
                    log_dict.update(routing_stats.metrics())
            if wandb: wandb.log(log_dict)

        if (
            step % args.save_interval == 0
            or (token_schedule is None and step == iters)
        ) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='./checkpoints', data_config=data_config)
            model.train()
            del state_dict

        del input_ids, labels, sequence_ids, res, loss

        if (
            args.use_compile == 1
            and packing_plan is not None
            and packing_plan.should_release_large_cuda_memory(epoch=epoch, step=step)
        ):
            release_compiled_cuda_memory(
                f'epoch={epoch + 1}, completed buckets >{args.bucket_large_threshold} tokens'
            )

        # 暂停分支：检测到暂停请求标记时，清标记并保存检查点后以 42 退出（WebUI 据此识别“已暂停”）。
        if pause_requested(args):
            clear_pause_request(args)
            profiler.finish()
            if is_main_process():
                pause_save_checkpoint(args, lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, data_config=data_config)
            Logger('[PAUSED] Training paused — resume checkpoint saved.')
            sys.exit(PAUSE_EXIT_CODE)

    flush_remaining_grad(scaler, optimizer, model.parameters(), args.grad_clip, last_step, start_step, args.accumulation_steps)
    return last_step


if __name__ == "__main__":
    parser = build_trainer_parser(
        description="Instinct Pretraining",
        defaults={
            'save_weight': 'pretrain',
            'epochs': 2,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'num_workers': 8,
            # A physical batch is one optimizer step by default.  Apart from being
            # the least surprising behaviour, this keeps CUDA Graph gradient
            # buffers from being reused across accumulation steps.
            'accumulation_steps': 1,
            'max_seq_len': 340,
            'wandb_project': 'Instinct-Pretrain',
        },
    )
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    add_moe_router_migration_arg(parser)
    args = parser.parse_args()
    validate_moe_router_migration(args)

    # 1. 初始化环境和随机种子
    local_rank = setup_dist_and_seed(args)

    streaming_enabled = should_stream_pretrain(args)
    if streaming_enabled and args.cache_build_mode == 'inline':
        # Zero Python child processes in the complete data path. PyArrow may
        # still use native threads, which do not re-import Python/DLL modules.
        requested_packing_workers = args.packing_num_proc
        requested_loader_workers = args.bucket_loader_workers
        # Keep Arrow/Datasets in one Python process on Windows, but let the
        # FastTokenizer's Rust/Rayon batch encoder use the requested CPU
        # parallelism without importing another copy of PyTorch per worker.
        tokenizer_threads = packing_preprocess_workers(requested_packing_workers)
        os.environ['INSTINCT_TOKENIZER_THREADS'] = str(tokenizer_threads)
        args.packing_num_proc = 1
        args.bucket_loader_workers = 0
        args.num_workers = 0
        Logger(
            '[Dataset Streaming] spawn-free inline mode: '
            f'packing_num_proc={requested_packing_workers}->1, '
            f'tokenizer_threads={tokenizer_threads}, '
            f'bucket_loader_workers={requested_loader_workers}->0, '
            'num_workers=0'
        )

    # 2. 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = config_from_args(args)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume==1 else None
    data_config = packing_data_config(args)
    lm_config = restore_config_from_checkpoint(lm_config, ckp_data)

    # 3. 设置混合精度
    autocast_ctx = build_autocast_ctx(args)

    # 4. 配wandb
    wandb = init_wandb_logger(args, ckp_data, run_name=f"Instinct-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}")

    # 5. 定义模型、数据、优化器
    model, tokenizer = init_model(
        lm_config, 'none' if ckp_data else args.from_weight, device=args.device,
        router_norm_topk_prob=args.moe_router_norm_topk_prob,
        router_top_k=args.moe_router_top_k,
    )
    configure_bucket_memory_budget(model, args, checkpoint_data=ckp_data)
    data_config = packing_data_config(args)
    if getattr(lm_config, 'model_architecture', '') == 'looped':
        Logger(
            '[Instinct V2 recurrent-depth] '
            f'P/R/C=({lm_config.prelude_layers}/{lm_config.recurrent_layers}/'
            f'{lm_config.coda_layers}), mean recurrence={lm_config.loop_iters}, '
            f'backprop depth={lm_config.mean_backprop_depth}, '
            f'sampling={lm_config.recurrence_sampling}, cap={lm_config.max_recurrence}'
        )
        if any((args.depth_reward >= 0, args.distill_weight >= 0,
                args.exit_in_training >= 0, args.n_supervision > 0)):
            Logger(
                '[Instinct V2] LoopUS depth-reward/self-distillation flags are '
                'deprecated and ignored; recurrence is trained with randomized '
                'unrolling and truncated backpropagation.'
            )
    packed_max_length = (
        min(lm_config.max_position_embeddings, args.bucket_max_seq_len)
        if args.sequence_packing_mode == 'bucket' else args.max_seq_len
    )

    def make_pretrain_dataset(packing, sample_indices=None, byte_range=None):
        return PretrainDataset(
            args.data_path, tokenizer,
            max_length=packed_max_length if packing else args.max_seq_len,
            packing=packing,
            packing_batch_size=args.packing_batch_size,
            packing_mode=args.sequence_packing_mode,
            seq_bucket=args.seq_bucket,
            packing_num_proc=args.packing_num_proc,
            bucket_gpu_memory_gb=args.bucket_gpu_memory_gb,
            bucket_token_budget_override=getattr(args, 'bucket_token_budget', None),
            sample_indices=sample_indices,
            byte_range=byte_range,
            source_fingerprint=(
                streaming_plan.get('source_sha256') if streaming_plan else None
            ),
        )

    streaming_plan = None
    if streaming_enabled:
        if not args.sequence_packing:
            raise ValueError(
                'bounded dataset streaming currently requires --sequence_packing 1'
            )
        if ckp_data and int(ckp_data.get('world_size', 1)) != (
            dist.get_world_size() if dist.is_initialized() else 1
        ):
            raise ValueError(
                'streaming resume currently requires the same GPU count because '
                'the checkpoint stores a per-rank chunk cursor'
            )
        chunk_bytes = validate_streaming_budget(args)
        os.environ['INSTINCT_MANAGED_DATA_CACHE'] = '1'
        if not dist.is_initialized() or dist.get_rank() == 0:
            streaming_plan = build_chunk_plan(
                args.data_path, chunk_bytes=chunk_bytes,
                max_length=packed_max_length, tokenizer=tokenizer,
            )
        if dist.is_initialized():
            payload = [streaming_plan]
            dist.broadcast_object_list(payload, src=0)
            streaming_plan = payload[0]
        saved_data_config = (ckp_data or {}).get('data_config') or {}
        if ckp_data and ckp_data.get('step', 0) and 'streaming_chunk_index' not in saved_data_config:
            raise ValueError(
                'this checkpoint predates the streaming chunk cursor and cannot '
                'safely resume in streaming mode'
            )
        for key in (
            'streaming_processed_tokens', 'streaming_chunk_index',
            'streaming_chunk_step',
        ):
            if key in saved_data_config:
                data_config[key] = saved_data_config[key]
        note = _recompiled_resume_note(
            saved_data_config, streaming_plan, args.data_path
        )
        if note:
            Logger(note)
        data_config.update(
            streaming_enabled=True,
            streaming_chunks=len(streaming_plan['chunks']),
            streaming_tokens_per_epoch=int(streaming_plan['tokens']),
            streaming_prefetch_chunks=int(args.streaming_prefetch_chunks),
        )
        Logger(
            f"[Dataset Streaming] bounded Arrow windows enabled: "
            f"chunks={len(streaming_plan['chunks'])}, "
            f"chunk_target={chunk_bytes / 1024 ** 2:.0f}MiB, "
            f"rows={streaming_plan['rows']:,}, "
            f"tokens/epoch={streaming_plan['tokens']:,}, "
            f"cache_budget={args.data_cache_max_gb:g}GiB, "
            f"prefetch_next={int(args.streaming_prefetch_chunks)}"
        )
    else:
        data_config['streaming_enabled'] = False

    packing_plan = SequencePackingPlan(
        args, ckp_data,
        lambda packing, sample_indices=None: make_pretrain_dataset(
            packing, sample_indices,
        ),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = build_optimizer(model.named_parameters(), lr=args.learning_rate, optimizer=args.optimizer)
    # Make the first compiled backward obey the same no-accumulation invariant
    # as every backward following optimizer.step().
    optimizer.zero_grad(set_to_none=True)

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
        Logger(f"torch.compile enabled; cache={os.environ['TORCHINDUCTOR_CACHE_DIR']}")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    profiler = TrainingProfiler(args, name="pretrain")

    # 8. 开始训练
    for epoch in range(start_epoch, args.epochs):
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        if streaming_enabled:
            packing_plan.update_checkpoint_config(data_config, epoch=epoch, active=True)
            resume_config = (
                (ckp_data or {}).get('data_config') or {}
                if epoch == start_epoch and skip > 0 else {}
            )
            loader = ChunkedPackedEpochLoader(
                plan=streaming_plan,
                dataset_factory=make_pretrain_dataset,
                packing_plan=packing_plan,
                args=args,
                epoch=epoch,
                data_config=data_config,
                resume_config=resume_config,
            )
            if skip > 0 and loader.resume_chunk == len(streaming_plan['chunks']):
                Logger(f'[Resume] Epoch {epoch + 1}/{args.epochs} already complete: '
                       f'all {loader.resume_chunk} chunks consumed, {skip} steps completed.')
            elif skip > 0:
                Logger(
                    f'Epoch [{epoch + 1}/{args.epochs}]: resume global step '
                    f'{skip + 1}, chunk={resume_config.get("streaming_chunk_index", 0) + 1}, '
                    f'chunk_step={resume_config.get("streaming_chunk_step", 0)}'
                )
            last_step = train_epoch(
                epoch, loader, len(loader), skip, wandb, packing_plan,
                token_schedule={
                    'total_tokens': int(streaming_plan['tokens']) * args.epochs,
                },
            )
            # __len__ is only an ETA estimate because packing is local to each
            # window. Always persist the exact final cursor and model state.
            if is_main_process():
                pause_save_checkpoint(
                    args, lm_config, weight=args.save_weight, model=model,
                    optimizer=optimizer, scaler=scaler, epoch=epoch,
                    step=last_step, wandb=wandb, data_config=data_config,
                )
                Logger(
                    f'Epoch:[{epoch + 1}/{args.epochs}]({last_step}/{last_step}), '
                    f'chunks_complete: {len(streaming_plan["chunks"])}, '
                    f'epoch_time: 0.0min, checkpoint_saved: 1, '
                    'Streaming epoch complete (all chunks consumed).'
                )
        else:
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
                train_epoch(epoch, loader, len(loader) + skip, start_step, wandb, packing_plan)
            else:
                train_epoch(epoch, loader, len(loader), 0, wandb, packing_plan)
        if (
            args.use_compile == 1
            and epoch + 1 < args.epochs
            and packing_plan.has_large_phase(epoch)
        ):
            release_compiled_cuda_memory(
                f'epoch={epoch + 1} complete, preparing next long-bucket phase'
            )

    final_profile_metrics = profiler.finish()
    if final_profile_metrics and wandb:
        wandb.log(final_profile_metrics)

    # 9. 清理分布进程
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
