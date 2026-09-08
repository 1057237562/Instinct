"""MoFE post-pretraining with an attention router over frozen FFN experts.

The expert manifest is the source of truth.  Each source checkpoint contributes
one FeedForward expert per Transformer layer; those parameters never receive
gradients.  Checkpoints therefore contain only the trainable router/shared
delta plus optimizer state, not another copy of every frozen source model.
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import trainer.compile_cache  # noqa: F401
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "huggingface")
)
import datasets  # noqa: F401  # must precede torch on Windows
import time
import warnings
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from dataset.lm_dataset import PretrainDataset
from model.mofe_checkpoint import (
    assemble_mofe,
    load_manifest,
    load_trainable_state_dict,
    set_mofe_train_scope,
    trainable_state_dict,
)
from trainer.packing_transition import SequencePackingPlan, packing_data_config
from trainer.trainer_cli import (
    PAUSE_EXIT_CODE,
    build_autocast_ctx,
    build_trainer_parser,
    clear_pause_request,
    flush_remaining_grad,
    init_wandb_logger,
    pause_requested,
    set_cosine_lr,
    setup_dist_and_seed,
    step_with_scaler,
)
from trainer.trainer_utils import (
    Logger,
    apply_torchao_fp8_training,
    build_optimizer,
    config_from_args,
    init_model,
    is_main_process,
    prepare_lm_batch,
)
from trainer.training_profiler import TrainingProfiler

warnings.filterwarnings("ignore")


def _raw_model(model):
    value = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(value, "_orig_mod", value)


def _resume_path(args, config):
    return os.path.join(args.checkpoint_dir, f"{args.save_weight}_{config.hidden_size}_moe_resume.pth")


def _delta_path(args, config):
    return os.path.join(args.save_dir, f"{args.save_weight}_{config.hidden_size}_mofe_delta.pth")


def _checkpoint_payload(model, optimizer, scaler, epoch, step):
    return {
        "format": "instinct-mofe-delta-v1",
        "manifest": manifest.to_dict(),
        "manifest_fingerprint": manifest.fingerprint,
        "config": lm_config.to_dict(),
        "train_scope": args.train_scope,
        "model_delta": trainable_state_dict(_raw_model(model)),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "data_config": data_config,
    }


def save_checkpoint(model, optimizer, scaler, epoch, step):
    if not is_main_process():
        return
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    payload = _checkpoint_payload(model, optimizer, scaler, epoch, step)
    torch.save(payload, _resume_path(args, lm_config))
    torch.save(
        {
            key: value
            for key, value in payload.items()
            if key not in {"optimizer", "scaler", "data_config"}
        },
        _delta_path(args, lm_config),
    )
    Logger(f"[MoFE] saved compact delta: {_delta_path(args, lm_config)}")


def load_resume(model, optimizer, scaler):
    if not args.from_resume:
        return 0, 0, None
    path = _resume_path(args, lm_config)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MoFE resume checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "instinct-mofe-delta-v1":
        raise ValueError(f"Unsupported MoFE checkpoint format in {path}")
    if payload.get("manifest_fingerprint") != manifest.fingerprint:
        raise ValueError(
            "Expert manifest changed since this checkpoint was created; refusing to resume "
            "against a different frozen expert bank."
        )
    load_trainable_state_dict(model, payload["model_delta"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    return int(payload.get("epoch", 0)), int(payload.get("step", 0)), payload


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, packing_plan=None):
    started = time.time()
    last_step = start_step
    data_config["epoch_steps"] = int(iters)
    for step, batch in enumerate(loader, start=start_step + 1):
        status = packing_plan.observe_batch(batch, epoch=epoch, step=step) if packing_plan else ""
        profiler.begin_step(
            tokens=batch[0].numel(), useful_tokens=(batch[1] != -100).sum().item()
        )
        with profiler.phase("data_transfer"):
            input_ids, labels, sequence_ids = prepare_lm_batch(batch, args.device)
        last_step = step
        set_cosine_lr(optimizer, epoch, step, iters, args)
        with profiler.phase("forward"):
            with autocast_ctx:
                result = model(input_ids, labels=labels, sequence_ids=sequence_ids)
                loss = (result.loss + result.aux_loss) / args.accumulation_steps
        with profiler.phase("backward"):
            scaler.scale(loss).backward()
        if step % args.accumulation_steps == 0:
            with profiler.phase("optimizer"):
                step_with_scaler(scaler, optimizer, trainable_parameters, args.grad_clip)
        profile_metrics = profiler.end_step()
        if profile_metrics and wandb:
            wandb.log(profile_metrics)

        if step % args.log_interval == 0 or step == iters:
            total_loss = loss.item() * args.accumulation_steps
            aux_loss = result.aux_loss.item() if result.aux_loss is not None else 0.0
            elapsed = (time.time() - started) / 60
            eta = elapsed / max(step - start_step, 1) * (iters - step)
            route = []
            for layer in _raw_model(model).model.layers:
                stats = getattr(layer.mlp, "routing_stats", None)
                if stats:
                    route.append(float(stats["load"].max().cpu()))
            max_load = sum(route) / len(route) if route else 0.0
            suffix = f", {status}" if status else ""
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss:{total_loss:.4f}, "
                f"aux:{aux_loss:.4f}, mean_layer_max_load:{max_load:.3f}, "
                f"lr:{optimizer.param_groups[-1]['lr']:.8f}{suffix}, eta:{eta:.1f}min"
            )
            if wandb:
                wandb.log({
                    "loss": total_loss,
                    "aux_loss": aux_loss,
                    "mean_layer_max_load": max_load,
                    "learning_rate": optimizer.param_groups[-1]["lr"],
                })

        if step % args.save_interval == 0 or step == iters:
            save_checkpoint(model, optimizer, scaler, epoch, step)

        if pause_requested(args):
            clear_pause_request(args)
            save_checkpoint(model, optimizer, scaler, epoch, step)
            profiler.finish()
            Logger("[PAUSED] MoFE post-pretraining paused; compact resume state saved.")
            raise SystemExit(PAUSE_EXIT_CODE)

    flush_remaining_grad(
        scaler, optimizer, trainable_parameters, args.grad_clip,
        last_step, start_step, args.accumulation_steps,
    )


if __name__ == "__main__":
    parser = build_trainer_parser(
        "Instinct MoFE post-pretraining",
        defaults={
            "save_weight": "mofe_post_pretrain",
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "accumulation_steps": 4,
            "max_seq_len": 768,
            "wandb_project": "Instinct-MoFE-PostPretrain",
        },
    )
    parser.add_argument("--data_path", required=True, help="Post-pretraining JSONL corpus")
    parser.add_argument("--expert_manifest", required=True, help="JSON manifest for base and n frozen experts")
    parser.add_argument("--checkpoint_dir", default="./checkpoints", help="Compact resume checkpoint directory")
    parser.add_argument("--train_scope", choices=["router_only", "router_shared"], default="router_shared")
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--router_type", choices=["attention", "linear"], default="attention")
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--router_aux_loss_coef", type=float, default=5e-4)
    parser.add_argument("--router_z_loss_coef", type=float, default=1e-3)
    args = parser.parse_args()

    if args.model_architecture not in (None, "standard") or args.use_looped:
        raise ValueError("MoFE post-pretraining currently defines a new standard-backbone architecture only")
    args.model_architecture = "standard"
    args.use_moe = 1
    local_rank = setup_dist_and_seed(args)
    manifest = load_manifest(args.expert_manifest)
    lm_config = config_from_args(
        args,
        use_moe=True,
        num_experts=len(manifest.experts),
        num_experts_per_tok=args.num_experts_per_tok,
        moe_expert_mode="frozen",
        router_type=args.router_type,
        router_temperature=args.router_temperature,
        router_aux_loss_coef=args.router_aux_loss_coef,
        router_z_loss_coef=args.router_z_loss_coef,
    )
    if lm_config.num_experts_per_tok > len(manifest.experts):
        raise ValueError("num_experts_per_tok cannot exceed the manifest expert count")

    os.makedirs(args.save_dir, exist_ok=True)
    autocast_ctx = build_autocast_ctx(args)
    model, tokenizer = init_model(lm_config, "none", device=args.device)
    report = assemble_mofe(model, manifest)
    trainable_names = set_mofe_train_scope(model, args.train_scope)
    model.to(args.device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    Logger(
        f"[MoFE] assembled {len(manifest.experts)} frozen experts x {lm_config.num_hidden_layers} layers; "
        f"loaded={report['copied_ffn_tensors']} FFN tensors; train_scope={args.train_scope}; "
        f"trainable={sum(p.numel() for p in trainable_parameters) / 1e6:.3f}M"
    )
    Logger(f"[MoFE] manifest fingerprint: {manifest.fingerprint}")

    scaler = torch.cuda.amp.GradScaler(enabled=args.dtype == "float16")
    optimizer = build_optimizer(
        [(name, parameter) for name, parameter in model.named_parameters() if name in trainable_names],
        lr=args.learning_rate,
        optimizer=args.optimizer,
    )
    optimizer.zero_grad(set_to_none=True)
    start_epoch, start_step, resume_data = load_resume(model, optimizer, scaler)
    data_config = packing_data_config(args)
    if resume_data and resume_data.get("data_config"):
        data_config.update(resume_data["data_config"])

    wandb = init_wandb_logger(
        args, resume_data,
        run_name=f"Instinct-MoFE-{len(manifest.experts)}E-{args.train_scope}",
    )
    packing_plan = SequencePackingPlan(
        args, resume_data,
        lambda packing, sample_indices=None: PretrainDataset(
            args.data_path,
            tokenizer,
            max_length=(
                min(lm_config.max_position_embeddings, args.bucket_max_seq_len)
                if packing and args.sequence_packing_mode == "bucket"
                else args.max_seq_len
            ),
            packing=packing,
            packing_batch_size=args.packing_batch_size,
            packing_mode=args.sequence_packing_mode,
            seq_bucket=args.seq_bucket,
            packing_num_proc=args.packing_num_proc,
            bucket_gpu_memory_gb=args.bucket_gpu_memory_gb,
            sample_indices=sample_indices,
        ),
    )

    model = apply_torchao_fp8_training(model, args)
    if args.use_compile:
        model = torch.compile(model, mode=args.compile_mode)
        Logger(f"torch.compile enabled; cache={os.environ['TORCHINDUCTOR_CACHE_DIR']}")
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    profiler = TrainingProfiler(args, name="mofe_post_pretrain")

    for epoch in range(start_epoch, args.epochs):
        skip = start_step if epoch == start_epoch else 0
        train_ds, transition_batches, active_packing = packing_plan.epoch_data(
            epoch, skip, args.batch_size,
        )
        packing_plan.update_checkpoint_config(data_config, epoch=epoch, active=active_packing)
        batch_sampler = (
            packing_plan.batch_sampler(
                train_ds, active_packing=active_packing, epoch=epoch,
                batch_size=args.batch_size, skip_batches=skip,
            )
            if transition_batches is None else transition_batches
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=packing_plan.loader_num_workers(),
            pin_memory=True,
        )
        train_epoch(epoch, loader, len(loader) + skip, skip, wandb, packing_plan)
        start_step = 0

    final_profile_metrics = profiler.finish()
    if final_profile_metrics and wandb:
        wandb.log(final_profile_metrics)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
