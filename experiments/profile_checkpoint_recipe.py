"""Finite synthetic recipe run with workspace sweep and verified save-and-pause.

Includes accumulated backward, grad clipping, and real Muon/Adam updates. Shapes,
precision and compile settings match the named V2 recipe; data loading and corpus
quality are outside this measurement. Never use while another trainer is running.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trainer.compile_cache  # noqa: F401 -- before torch
import datasets  # noqa: F401 -- Windows DLL ordering
import torch
from types import SimpleNamespace

from experiments.profile_checkpoint_policy import reserve_training_state
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from trainer.trainer_cli import step_with_scaler
from trainer.trainer_utils import cast_training_parameters, apply_torchao_fp8_training, pause_save_checkpoint


def verify_state(actual, saved):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual.detach().cpu(), saved.cpu())
    elif isinstance(actual, dict):
        assert actual.keys() == saved.keys()
        for key in actual:
            verify_state(actual[key], saved[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(saved)
        for a, b in zip(actual, saved):
            verify_state(a, b)
    else:
        assert actual == saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', type=int, choices=[2, 4], required=True)
    parser.add_argument('--workspaces', type=int, nargs='+', default=[128, 256, 512, 1024, 2048, 3072])
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seq', type=int, default=3072)
    parser.add_argument('--segments', type=int, default=3)
    parser.add_argument('--alternate-shape', type=int, nargs=2)
    parser.add_argument('--alternate-segments', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2, help='updates for every workspace')
    parser.add_argument('--steps', type=int, default=2, help='updates per measured window')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--allocator-limit-gib', type=float, default=14.37)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.workspaces + [args.batch, args.seq, args.segments, args.accumulation,
                              args.warmup, args.steps, args.repeats]) < 1:
        parser.error('counts must be positive')
    if args.segments > args.seq or (args.alternate_shape and min(args.alternate_shape) < 1):
        parser.error('invalid shape/segment count')
    os.environ.update(INSTINCT_MUON_MANTISSA='1', INSTINCT_MUON_MOMENTUM_DTYPE='bf16',
                      INSTINCT_MUON_BATCH_SIZE='16', INSTINCT_MUON_WORKSPACE_MB=str(args.workspaces[0]))
    torch.set_num_threads(1)
    torch.manual_seed(307)
    torch.cuda.set_per_process_memory_fraction(min(1., args.allocator_limit_gib * 2**30 /
        torch.cuda.get_device_properties(0).total_memory))
    config = InstinctConfig(**dict(json.loads((ROOT / 'trainer/config_instinct_v2.json').read_text()),
                                  use_grad_checkpoint=args.mode))
    raw_model = cast_training_parameters(InstinctForCausalLM(config), 'bf16').cuda().train()
    optimizer = reserve_training_state(raw_model)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    apply_torchao_fp8_training(raw_model, SimpleNamespace(fp8_training='tensorwise', fp8_filter='auto',
        dtype='bfloat16', device='cuda', use_compile=1))
    model = torch.compile(raw_model, mode='max-autotune-no-cudagraphs', dynamic=False)
    shapes = [(args.batch, args.seq)]
    if args.alternate_shape:
        shapes.append(tuple(args.alternate_shape))
    batches = []
    for shape_index, (b, s) in enumerate(shapes):
        tokens = torch.randint(3, config.vocab_size, (b, s), device='cuda')
        labels = tokens.clone()
        ids = None
        segments = args.segments if shape_index == 0 else args.alternate_segments
        if segments > 1:
            ids = (torch.arange(s, device='cuda') * segments // s).expand(b, -1).contiguous()
            labels[:, 1:][ids[:, 1:] != ids[:, :-1]] = -100
        batches.append((tokens, labels, ids, int((labels[:, 1:] != -100).sum())))
    stop = False

    def request_pause(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_pause)
    prefix = f'ckpt{args.mode}_recipe_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    pause_path = ROOT / 'checkpoints' / (prefix + '.pause_request')
    updates = microsteps = 0
    windows = {str(w): [] for w in args.workspaces}
    all_peaks = []
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', args=vars(args), config=config.to_dict(),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        scope='synthetic accumulated updates, including clipping and Muon/Adam; excludes loader/saves',
        compile_dynamic=False,
        windows=windows, pause_file=str(pause_path), checkpoint_prefix=prefix)

    def write():
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')

    def update(workspace):
        nonlocal updates, microsteps
        for opt in optimizer.optimizers:
            if hasattr(opt, 'workspace_bytes'):
                opt.workspace_bytes = workspace * 2**20
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        useful_tokens = 0
        for _ in range(args.accumulation):
            tokens, labels, ids, valid = batches[microsteps % len(batches)]
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(tokens, labels=labels, sequence_ids=ids)
                loss = out.loss / args.accumulation
                del out
            loss.backward()
            useful_tokens += valid
            microsteps += 1
        torch.cuda.synchronize()
        backward_done = time.perf_counter()
        activation_peak = torch.cuda.max_memory_allocated() / 2**30
        activation_reserved = torch.cuda.max_memory_reserved() / 2**30
        torch.cuda.reset_peak_memory_stats()
        step_with_scaler(scaler, optimizer, model.parameters(), 1.)
        torch.cuda.synchronize()
        end = time.perf_counter()
        updates += 1
        result = dict(workspace_mb=workspace, update=updates, microsteps=microsteps,
            total_ms=(end-start)*1000, forward_backward_ms=(backward_done-start)*1000,
            optimizer_ms=(end-backward_done)*1000, useful_tokens=useful_tokens,
            allocated_gib=max(activation_peak, torch.cuda.max_memory_allocated()/2**30),
            reserved_gib=max(activation_reserved, torch.cuda.max_memory_reserved()/2**30),
            activation_peak_gib=activation_peak, optimizer_peak_gib=torch.cuda.max_memory_allocated()/2**30,
            loss=float(loss.detach())*args.accumulation)
        all_peaks.append(result)
        print(json.dumps(result), flush=True)
        return result

    write()
    for workspace in args.workspaces:
        for _ in range(args.warmup):
            update(workspace)
            if stop or pause_path.exists():
                break
        if stop or pause_path.exists():
            break
    if not stop and not pause_path.exists():
        # Rotate order so workspace comparisons do not all follow warm/cold GPU order.
        for repeat in range(args.repeats):
            order = args.workspaces[repeat % len(args.workspaces):] + args.workspaces[:repeat % len(args.workspaces)]
            for workspace in order:
                window = []
                for _ in range(args.steps):
                    window.append(update(workspace))
                    if stop or pause_path.exists():
                        break
                windows[str(workspace)].append(window)
                write()
                if stop or pause_path.exists():
                    break
            if stop or pause_path.exists():
                break
    # Completion and SIGINT use the same full save after an accumulated update.
    cursor = dict(synthetic=True, seed=307, shapes=shapes, segments=[args.segments, args.alternate_segments][:len(shapes)],
                  next_microstep=microsteps, optimizer_updates=updates, accumulation=args.accumulation)
    (ROOT / 'out').mkdir(exist_ok=True)
    pause_save_checkpoint(SimpleNamespace(save_dir=str(ROOT / 'out')), config, weight=prefix,
        model=model, optimizer=optimizer, scaler=scaler, epoch=0, step=microsteps,
        data_config=cursor, rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state())
    resume_path = ROOT / 'checkpoints' / f'{prefix}_{config.hidden_size}_resume.pth'
    saved = torch.load(resume_path, map_location='cpu', mmap=True, weights_only=False)
    assert saved['step'] == microsteps and saved['data_config'] == cursor
    verify_state(raw_model.state_dict(), saved['model'])
    verify_state(optimizer.state_dict(), saved['optimizer'])
    verify_state(scaler.state_dict(), saved['scaler'])
    weight_path = ROOT / 'out' / f'{prefix}_{config.hidden_size}.pth'
    weights = torch.load(weight_path, map_location='cpu', mmap=True, weights_only=True)
    for name, value in saved['model'].items():
        assert torch.equal(weights[name], value.half())
    report.update(status='paused' if stop or pause_path.exists() else 'ok',
        saved_resume=str(resume_path), saved_weights=str(weight_path), verified_current_state=True,
        updates=updates, microsteps=microsteps, all_updates=all_peaks)
    report['summary'] = {}
    for workspace, measured in windows.items():
        if measured:
            ms = [statistics.mean(row['total_ms'] for row in window) for window in measured]
            rows = [row for window in measured for row in window]
            report['summary'][workspace] = dict(window_ms=ms, median_ms=statistics.median(ms),
                useful_tokens_per_second=statistics.mean(row['useful_tokens'] for row in rows)*1000/statistics.median(ms),
                optimizer_median_ms=statistics.median(row['optimizer_ms'] for row in rows),
                peak_allocated_gib=max(row['allocated_gib'] for row in rows),
                peak_reserved_gib=max(row['reserved_gib'] for row in rows))
    write()
    print(json.dumps(dict(status=report['status'], summary=report['summary'], saved_resume=str(resume_path))), flush=True)


if __name__ == '__main__':
    main()
