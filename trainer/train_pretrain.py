import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "huggingface"))
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler, config_from_args, build_optimizer

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, early_exit=bool(args.early_exit))
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            loop_steps = getattr(model, 'last_avg_steps', None)
            loop_str = f', loop_steps: {loop_steps:.2f}' if loop_steps else ''
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}{loop_str}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            log_dict = {"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min}
            if loop_steps: log_dict["loop_steps"] = loop_steps
            if wandb: wandb.log(log_dict)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='./checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instinct Pretraining")
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
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_looped', default=0, type=int, choices=[0, 1], help="是否使用LoopUS循环架构（0=否，1=是）")
    parser.add_argument('--depth_reward', default=-1.0, type=float, help="深度reward权重λ（>0时覆盖config，-1用config默认；可随训练退火）")
    parser.add_argument('--distill_weight', default=-1.0, type=float, help="自蒸馏权重（>0时启用：浅层循环深度向最终深度输出分布学习；建议配合exit_in_training=0跑满循环）")
    parser.add_argument('--distill_temperature', default=2.0, type=float, help="自蒸馏温度T（软化教师/学生分布）")
    parser.add_argument('--teacher_stop_grad', default=1, type=int, choices=[0, 1], help="教师logits是否stop-grad（1=是，0=否）")
    parser.add_argument('--depth_gain_reward', default=0.0, type=float, help="深度增益奖励权重（>0时启用：仅当更深步相对第1步基线降低LM损失时给予正奖励）")
    parser.add_argument('--exit_in_training', default=-1, type=int, choices=[-1, 0, 1], help="训练中是否允许早退（-1用config默认；自蒸馏建议0=跑满循环）")
    parser.add_argument('--n_supervision', default=-1, type=int, help="随机深度监督步数（-1用config默认；自蒸馏可提高以覆盖更多深度）")
    parser.add_argument('--early_exit', default=0, type=int, choices=[0, 1], help="启用Early Exit训练（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="Instinct-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--use_grad_checkpoint", default=0, type=int, choices=[0, 1, 2], help="梯度检查点模式（0=关闭, 1=选择性重算注意力/FFN, 2=整层checkpoint）")
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], help="torch.compile 模式（default=Triton 编译；reduce-overhead=叠加 CUDA graph，小模型首选；max-autotune=极限调优，编译极慢）")
    parser.add_argument('--config_path', default='', type=str, help="JSON配置文件路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = config_from_args(args)
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='./checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'fp32': torch.float32}[args.dtype]
    autocast_ctx = nullcontext() if device_type == "cpu" or args.dtype == "fp32" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"Instinct-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, 'none' if ckp_data else args.from_weight, device=args.device)
    if args.use_looped and args.depth_reward >= 0:
        model.set_depth_reward(args.depth_reward)
        Logger(f'[Looped] depth_reward λ = {args.depth_reward}')
    if args.use_looped and args.distill_weight >= 0:
        model.config.distill_weight = args.distill_weight
        model.config.distill_temperature = args.distill_temperature
        model.config.teacher_stop_grad = bool(args.teacher_stop_grad)
        model.config.depth_gain_reward = args.depth_gain_reward
        if args.exit_in_training >= 0:
            model.config.exit_in_training = bool(args.exit_in_training)
        if args.n_supervision > 0:
            model.config.n_supervision = args.n_supervision
        Logger(f'[Looped] self-distill w={args.distill_weight} T={args.distill_temperature} '
               f'stop_grad={args.teacher_stop_grad} | depth-gain reward={args.depth_gain_reward} '
               f'| exit_in_training={model.config.exit_in_training} '
               f'| n_supervision={model.config.n_supervision}')
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = build_optimizer(model.parameters(), lr=args.learning_rate, optimizer=args.optimizer)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model, mode=args.compile_mode)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()