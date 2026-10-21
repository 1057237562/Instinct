"""Measure and validate fused split-weight Muon against its foreach reference.

The full-training section uses the selected model and synthetic numeric tokens,
with identical initialization, batches, LR schedule, autocast and checkpointing.
No production weights, corpora or checkpoints are read or written.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import datasets  # noqa: F401,E402 -- Windows DLL order
import trainer.compile_cache  # noqa: F401,E402
import torch  # noqa: E402
from experiments.audit_muon_precision import production_shapes, BF16MomentumMuon
from experiments.validate_full_training_kernels import fingerprint
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from trainer.batched_muon import BatchedMuon, _split_precise
from trainer.trainer_utils import (
    build_optimizer, cast_training_parameters, _is_muon_hidden_matrix,
    apply_torchao_fp8_training,
)


def set_fusion(value, compare_ns=False):
    os.environ['INSTINCT_MUON_FUSED_UPDATES'] = value
    if compare_ns:
        os.environ['INSTINCT_MUON_NS_BACKEND'] = 'torch' if value == '0' else 'auto'


def selected_shapes(args, *, full=False):
    if args.config is None and not full:
        return production_shapes()[0]
    path = args.config or ROOT / 'trainer/config_instinct_v1_moe.json'
    raw = json.loads(path.read_text(encoding='utf-8'))
    with torch.device('meta'):
        model = InstinctForCausalLM(InstinctConfig(**raw))
    return [tuple(p.shape) for n, p in model.named_parameters()
            if _is_muon_hidden_matrix(n, p)
            and (full or n.startswith(('model.layers.0.', 'model.layers.1.')))]


def isolated(args):
    shapes = selected_shapes(args, full=args.full_optimizer)
    results = []
    for trial in range(args.isolated_trials):
        # Alternate ordering to expose simple thermal/order bias.
        order = [('reference', '0'), ('fused', 'auto')]
        if trial % 2:
            order.reverse()
        for name, flag in order:
            set_fusion(flag, args.compare_ns)
            gc.collect()
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.manual_seed(73)
            params = [torch.nn.Parameter(torch.randn(s, device='cuda').bfloat16()*.02) for s in shapes]
            for p in params:
                p.grad = torch.randn_like(p)
            opt = BatchedMuon(params, use_mantissa=True, lr=5e-4,
                              adjust_lr_fn='match_rms_adamw', workspace_mb=args.workspace_mb,
                              momentum_dtype=args.momentum_dtype)
            for _ in range(args.isolated_warmup):
                opt.step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = []
            wall = []
            for _ in range(args.isolated_steps):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter()
                start.record()
                opt.step()
                end.record()
                torch.cuda.synchronize()
                wall.append((time.perf_counter()-t0)*1000)
                timings.append(start.elapsed_time(end))
            result = dict(trial=trial, mode=name, matrices=len(shapes),
                          elements=sum(p.numel() for p in params),
                          median_ms=statistics.median(timings),
                          median_wall_ms=statistics.median(wall),
                          peak_bytes=torch.cuda.max_memory_allocated()-base,
                          persistent_bytes=torch.cuda.memory_allocated()-base)
            if trial == 0:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    opt.step()
                    torch.cuda.synchronize()
                events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                result['cuda_events'] = len(events)
                result['cuda_time_us'] = sum(e.device_time_total for e in events)
                result['cpu_operator_events'] = len(prof.events()) - len(events)
                del prof, events
            results.append(result)
            print('ISOLATED', json.dumps(result), flush=True)
            del p, params, opt
    return results


def shared_gradients(args):
    shapes = selected_shapes(args)
    torch.manual_seed(79)
    values = [torch.randn(s, device='cuda')*.02 for s in shapes]
    pairs = [_split_precise(p) for p in values]
    refs = [torch.nn.Parameter(high.clone()) for high, _ in pairs]
    news = [torch.nn.Parameter(high.clone()) for high, _ in pairs]
    ref = BatchedMuon(refs, use_mantissa=True, adjust_lr_fn='match_rms_adamw',
                     momentum_dtype=args.momentum_dtype, workspace_mb=args.workspace_mb)
    new = BatchedMuon(news, use_mantissa=True, adjust_lr_fn='match_rms_adamw',
                     momentum_dtype=args.momentum_dtype, workspace_mb=args.workspace_mb)
    for p, q, (_, low) in zip(refs, news, pairs):
        ref.state[p]['mantissa'] = low.clone()
        new.state[q]['mantissa'] = low.clone()
    del values, pairs
    for step in range(args.shared_steps):
        for i, (p, q) in enumerate(zip(refs, news)):
            p.grad = None if (i+step) % 11 == 0 else torch.randn_like(p)
            q.grad = p.grad
        for opt in (ref, new):
            opt.param_groups[0]['lr'] = 5e-4 * (1-step/args.shared_steps)
        set_fusion('0', args.compare_ns)
        ref.step()
        set_fusion('auto', args.compare_ns)
        new.step()
        # Compare every high bit, low bit and FP32 momentum value at every step.
        for p, q in zip(refs, news):
            if not torch.equal(p.view(torch.int16), q.view(torch.int16)):
                raise AssertionError(f'Weight mismatch at step {step}')
            for key, value in ref.state[p].items():
                if not torch.equal(value, new.state[q][key]):
                    raise AssertionError(f'{key} mismatch at step {step}')
        if (step+1) % 20 == 0:
            print('SHARED', step+1, 'all weights/mantissas/momentum exact', flush=True)
    return dict(steps=args.shared_steps, matrices=len(shapes),
                elements=sum(p.numel() for p in refs), every_step_bitwise_equal=True)


def full_training(args):
    gc.collect()
    torch.cuda.empty_cache()
    os.environ['INSTINCT_MUON_MANTISSA'] = '1'
    torch.manual_seed(83)
    raw = json.loads((args.config or ROOT/'trainer/config_instinct_v1_moe.json').read_text(encoding='utf-8'))
    raw.update(param_dtype='bf16', use_grad_checkpoint=args.checkpoint)
    model = cast_training_parameters(InstinctForCausalLM(InstinctConfig(**raw)), 'bf16')
    initial = {k: v.detach().clone() for k,v in model.state_dict().items()}
    model = model.cuda().train()
    model = apply_torchao_fp8_training(model, SimpleNamespace(
        fp8_training=args.fp8, fp8_filter='auto', dtype='bfloat16', device='cuda',
        use_compile=int(args.compile_mode != 'off')))
    forward = (torch.compile(model, mode=args.compile_mode)
               if args.compile_mode != 'off' else model)
    generator = torch.Generator().manual_seed(89)
    # Numeric sequences only; seeds and held-out batches are distinct.
    data = torch.randint(3, raw['vocab_size'], (8, args.batch, 8), generator=generator)
    data = data.repeat(1,1,(args.seq+7)//8)[:,:,:args.seq].cuda()
    heldout = torch.randint(3, raw['vocab_size'], (2, args.batch, 8), generator=generator)
    heldout = heldout.repeat(1,1,(args.seq+7)//8)[:,:,:args.seq].cuda()
    report = dict(config=raw, batch=args.batch, seq=args.seq, steps=args.steps,
                  compiled=args.compile_mode != 'off', fp8=args.fp8,
                  input_sha256=fingerprint([('data',data),('heldout',heldout)]), phases={})
    def evaluate():
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            losses = []
            for tokens in heldout:
                output = model(tokens, labels=tokens)
                losses.append(float(output.loss+output.aux_loss))
        model.train()
        return statistics.fmean(losses)
    for phase, flag in [('reference','0'), ('repeat','0'), ('fused','auto')]:
        set_fusion(flag, args.compare_ns)
        model.load_state_dict(initial)
        opt = build_optimizer(model.named_parameters(), lr=5e-4, optimizer='muon')
        opt.zero_grad(set_to_none=True)
        row = dict(losses=[], step_ms=[], optimizer_ms=[], wall_ms=[], validation_before=evaluate())
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.steps):
            for group in opt.param_groups:
                group['lr'] = 5e-4 * (1-.9*step/max(1,args.steps-1))
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            begin, boundary, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            begin.record()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tokens = data[step % len(data)]
                output = forward(tokens, labels=tokens)
                loss = output.loss + output.aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            boundary.record()
            opt.step()
            end.record()
            torch.cuda.synchronize()
            row['wall_ms'].append((time.perf_counter()-t0)*1000)
            row['step_ms'].append(begin.elapsed_time(end))
            row['optimizer_ms'].append(boundary.elapsed_time(end))
            row['losses'].append(float(loss.detach()))
            if not torch.isfinite(loss):
                raise AssertionError(f'Nonfinite loss in {phase} step {step}')
            if (step+1) % 10 == 0:
                print('TRAIN',phase,step+1,'loss',row['losses'][-1],
                      'step_ms',row['step_ms'][-1],'opt_ms',row['optimizer_ms'][-1],flush=True)
        row['training_peak_bytes'] = torch.cuda.max_memory_allocated()
        row['validation_after'] = evaluate()
        row['model_sha256'] = fingerprint(model.named_parameters())
        states = ((f'{idx}/{name}/{key}', value.reshape(-1))
                  for idx, optimizer in enumerate(opt.optimizers)
                  for name, param in model.named_parameters()
                  for key,value in sorted(optimizer.state.get(param,{}).items())
                  if isinstance(value,torch.Tensor))
        row['optimizer_sha256'] = fingerprint(states)
        row['warm_step_median_ms'] = statistics.median(row['step_ms'][10:])
        row['warm_optimizer_median_ms'] = statistics.median(row['optimizer_ms'][10:])
        reference = report['phases'].get('reference', row)
        row['bitwise_equal'] = all(row[k] == reference[k] for k in
            ['losses', 'validation_after', 'model_sha256', 'optimizer_sha256'])
        report['phases'][phase] = row
        print('TRAIN_RESULT', phase, json.dumps({k:v for k,v in row.items() if not isinstance(v,list)}), flush=True)
        args.output.with_suffix('.training.partial.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
        del opt, output, loss, states
        for p in model.parameters():
            p.grad = None
        gc.collect()
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sections', default='isolated,shared,train')
    parser.add_argument('--config', type=Path, help='Model config; defaults to V1 MoE')
    parser.add_argument('--compare-ns', action='store_true', help='Also compare torch NS against auto NS')
    parser.add_argument('--full-optimizer', action='store_true', help='Time all model matrices in isolated mode')
    parser.add_argument('--workspace-mb', type=float, default=64)
    parser.add_argument('--isolated-trials', type=int, default=3)
    parser.add_argument('--isolated-warmup', type=int, default=20)
    parser.add_argument('--isolated-steps', type=int, default=50)
    parser.add_argument('--checkpoint', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--fp8', choices=['off', 'tensorwise'], default='off')
    parser.add_argument('--compile-mode', choices=['off', 'max-autotune-no-cudagraphs'], default='off')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--shared-steps', type=int, default=100)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seq', type=int, default=256)
    parser.add_argument('--momentum-dtype', choices=['fp32','bf16'], default='fp32')
    parser.add_argument('--performance-only', action='store_true',
                        help='Use normal nondeterministic training; record equality but do not gate on it')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 10:
        parser.error('steps must exceed the 10-step warmup window')
    if min(args.isolated_trials, args.isolated_warmup, args.isolated_steps, args.shared_steps) < 1:
        parser.error('trial, warmup and step counts must be positive')
    if args.workspace_mb <= 0:
        parser.error('workspace-mb must be positive')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['INSTINCT_MUON_NS_BACKEND'] = 'auto'
    os.environ['INSTINCT_MUON_BACKEND'] = 'batched'
    os.environ['INSTINCT_MUON_WORKSPACE_MB'] = str(args.workspace_mb)
    os.environ['INSTINCT_MUON_MOMENTUM_DTYPE'] = args.momentum_dtype
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(not args.performance_only)
    torch.backends.cuda.matmul.allow_tf32 = False
    files = ['trainer/batched_muon.py', 'trainer/muon_kernels.py', 'trainer/trainer_utils.py',
             'experiments/validate_muon_mantissa_fusion.py']
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
                  protocol=vars(args)|{'output':str(args.output),
                                       'config':str(args.config) if args.config else None},
                  deterministic=not args.performance_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for section in args.sections.split(','):
        report[section] = {'isolated':isolated,'shared':shared_gradients,'train':full_training}[section](args)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        gc.collect()
        torch.cuda.empty_cache()
    if ('train' in report and not args.performance_only
            and not all(r['bitwise_equal'] for r in report['train']['phases'].values())):
        raise SystemExit('Full-training equality gate failed; inspect the reference repeat')
    print('SAVED',args.output,flush=True)


if __name__ == '__main__':
    main()
