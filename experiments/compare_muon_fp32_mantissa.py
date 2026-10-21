"""Full V1 MoE: production FP32 master parameters versus fused BF16+uint16.

Independent subprocesses and ABBA ordering isolate allocator/graph ownership.
Both arms use BF16 activation autocast and FP32 AdamW parameters. Mantissa
momentum is selectable (FP32 by default). Each
keeps its production factory workspace default unless --workspace-mb is given.
Synthetic numeric tokens only; no production corpus, weights or checkpoints.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import datasets  # noqa: F401 -- Windows import order
    import trainer.compile_cache  # noqa: F401
    import torch
    from experiments.validate_full_training_kernels import fingerprint
    from model.model_instinct import InstinctConfig, InstinctForCausalLM
    from trainer.trainer_utils import build_optimizer, cast_training_parameters

    split = args.arm == 'mantissa'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['INSTINCT_MUON_MANTISSA'] = '1' if split else '0'
    os.environ['INSTINCT_MUON_MOMENTUM_DTYPE'] = args.momentum_dtype if split else 'fp32'
    os.environ['INSTINCT_MUON_BACKEND'] = 'batched'
    os.environ['INSTINCT_MUON_NS_BACKEND'] = 'auto'
    os.environ['INSTINCT_MUON_FUSED_UPDATES'] = 'auto'
    if args.muon_batch_size is None:
        os.environ.pop('INSTINCT_MUON_BATCH_SIZE', None)
    else:
        os.environ['INSTINCT_MUON_BATCH_SIZE'] = str(args.muon_batch_size)
    if args.workspace_mb is None:
        os.environ.pop('INSTINCT_MUON_WORKSPACE_MB', None)
    else:
        os.environ['INSTINCT_MUON_WORKSPACE_MB'] = str(args.workspace_mb)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    raw = json.loads((ROOT/'trainer/config_instinct_v1_moe.json').read_text())
    dtype = 'bf16' if split else 'fp32'
    raw.update(param_dtype=dtype, use_grad_checkpoint=1)
    model = cast_training_parameters(InstinctForCausalLM(InstinctConfig(**raw)), dtype).cuda().train()
    opt = build_optimizer(model.named_parameters(), lr=5e-4, optimizer='muon')
    generator = torch.Generator().manual_seed(89)
    def batches(count):
        tokens = torch.randint(3, raw['vocab_size'], (count,args.batch,8), generator=generator)
        return tokens.repeat(1,1,(args.seq+7)//8)[:,:,:args.seq].cuda()
    data, heldout = batches(8), batches(2)
    def evaluate():
        model.eval()
        values = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for tokens in heldout:
                out = model(tokens, labels=tokens)
                values.append(float(out.loss+out.aux_loss))
        model.train()
        return statistics.fmean(values)
    row = dict(arm=args.arm, run=args.run, seed=args.seed, steps=args.steps, batch=args.batch,
               seq=args.seq, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
               config=raw, compiled_trunk=False, autocast='bfloat16', deterministic=args.deterministic,
               muon_workspace_mb=opt.optimizers[0].workspace_bytes/2**20,
               muon_batch_size=opt.optimizers[0].batch_size,
               muon_chunks={str(tuple(p.shape)): opt.optimizers[0]._chunk_size(p)
                            for group in opt.optimizers[0].param_groups for p in group['params']},
               muon_momentum_dtype=str(opt.optimizers[0].momentum_dtype),
               input_sha256=fingerprint([('data',data),('heldout',heldout)]),
               validation_before=evaluate(), losses=[], step_ms=[], optimizer_ms=[], wall_ms=[])
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cold_peak = 0
    peak_before_probe = 0
    for step in range(args.steps):
        if step == args.warmup:
            cold_peak = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        for group in opt.param_groups:
            group['lr'] = 5e-4 * (1-.9*step/max(1,args.steps-1))
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        probe = args.memory_phases and step == args.steps-1
        if probe:
            peak_before_probe = torch.cuda.max_memory_allocated()
            row['memory_phases'] = {'before_forward': {'allocated': torch.cuda.memory_allocated()}}
            torch.cuda.reset_peak_memory_stats()
        def record_memory(phase):
            if probe:
                torch.cuda.synchronize()
                row['memory_phases'][phase] = dict(allocated=torch.cuda.memory_allocated(),
                                                       peak=torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
        start, boundary, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        wall_start = time.perf_counter()
        start.record()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            tokens = data[step % len(data)]
            out = model(tokens, labels=tokens)
            loss = out.loss + out.aux_loss
        record_memory('forward')
        loss.backward()
        record_memory('backward')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        record_memory('clipping')
        boundary.record()
        opt.step()
        end.record()
        torch.cuda.synchronize()
        record_memory('optimizer')
        row['wall_ms'].append((time.perf_counter()-wall_start)*1000)
        row['step_ms'].append(start.elapsed_time(end))
        row['optimizer_ms'].append(boundary.elapsed_time(end))
        row['losses'].append(float(loss.detach()))
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss in {args.arm}, step {step}')
        if (step+1)%10 == 0:
            print('STEP',args.run,args.arm,step+1,row['losses'][-1],row['step_ms'][-1],flush=True)
    row['warm_peak_bytes'] = max(torch.cuda.max_memory_allocated(), peak_before_probe,
                                *(p.get('peak',0) for p in row.get('memory_phases',{}).values()))
    row['cold_peak_bytes'] = cold_peak
    row['peak_bytes'] = max(cold_peak,row['warm_peak_bytes'])
    row['warm_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    muon = opt.optimizers[0]
    muon_params = [p for g in muon.param_groups for p in g['params']]
    owned = sum(p.numel()*p.element_size() + (p.grad.numel()*p.grad.element_size() if p.grad is not None else 0)
                for p in muon_params)
    owned += sum(t.numel()*t.element_size() for s in muon.state.values() for t in s.values()
                 if isinstance(t,torch.Tensor))
    row['muon_bytes_per_element_including_gradients'] = owned/sum(p.numel() for p in muon_params)
    row['actual_momentum_dtypes'] = sorted({str(s['momentum_buffer'].dtype) for s in muon.state.values()})
    if args.memory_phases:
        row['owned_tensor_bytes'] = {
            'parameters': sum(p.numel()*p.element_size() for p in model.parameters()),
            'gradients': sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None),
            'optimizer': sum(v.numel()*v.element_size() for o in opt.optimizers for s in o.state.values()
                             for v in s.values() if isinstance(v,torch.Tensor))}
    row['validation_after'] = evaluate()
    row['model_sha256'] = fingerprint(model.named_parameters())
    states = ((f'{idx}/{name}/{key}',value.reshape(-1))
              for idx,optimizer in enumerate(opt.optimizers)
              for name,p in model.named_parameters()
              for key,value in sorted(optimizer.state.get(p,{}).items())
              if isinstance(value,torch.Tensor))
    row['optimizer_sha256'] = fingerprint(states)
    row['median_step_ms'] = statistics.median(row['step_ms'][args.warmup:])
    row['median_optimizer_ms'] = statistics.median(row['optimizer_ms'][args.warmup:])
    row['median_wall_ms'] = statistics.median(row['wall_ms'][args.warmup:])
    row['train_last10_mean'] = statistics.fmean(row['losses'][-10:])
    args.output.write_text(json.dumps(row,indent=2),encoding='utf-8')
    print('RESULT',json.dumps({k:v for k,v in row.items() if k not in
                              ('config','losses','step_ms','optimizer_ms','wall_ms')}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['fp32','mantissa'])
    parser.add_argument('--run',type=int,default=0)
    parser.add_argument('--seed',type=int,default=83)
    parser.add_argument('--steps',type=int,default=30)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--seq',type=int,default=2048)
    parser.add_argument('--workspace-mb',type=int)
    parser.add_argument('--muon-batch-size',type=int)
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--momentum-dtype',choices=['fp32','bf16'],default='fp32')
    parser.add_argument('--memory-phases',action='store_true',help='Diagnostic phase peaks on the last step; not a timing run')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.steps <= args.warmup:
        parser.error('steps must exceed warmup')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.arm:
        worker(args)
        return
    files = ['trainer/batched_muon.py','trainer/muon_kernels.py','trainer/trainer_utils.py',
             'experiments/compare_muon_fp32_mantissa.py']
    report = dict(protocol=vars(args)|{'output':str(args.output)},
                  revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
                  description='FP32 master vs fused mantissa, factory workspace defaults, ABBA fresh processes',
                  runs=[])
    for run,arm in enumerate(('fp32','mantissa','mantissa','fp32')):
        path = args.output.with_name(args.output.stem+f'.run{run}.json')
        command = [sys.executable,'-u',str(Path(__file__).resolve()),'--arm',arm,'--run',str(run),
                   '--steps',str(args.steps),'--warmup',str(args.warmup),'--batch',str(args.batch),
                   '--seq',str(args.seq),'--seed',str(args.seed),'--output',str(path)]
        command += ['--momentum-dtype',args.momentum_dtype]
        if args.workspace_mb is not None:
            command += ['--workspace-mb',str(args.workspace_mb)]
        if args.muon_batch_size is not None:
            command += ['--muon-batch-size',str(args.muon_batch_size)]
        if args.deterministic:
            command.append('--deterministic')
        if args.memory_phases:
            command.append('--memory-phases')
        subprocess.run(command,cwd=ROOT,check=True)
        report['runs'].append(json.loads(path.read_text(encoding='utf-8')))
        # Persist each successful phase in the aggregate before dropping its temporary copy.
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        path.unlink()
    for arm in ('fp32','mantissa'):
        rows = [r for r in report['runs'] if r['arm']==arm]
        report[arm] = dict(median_step_ms=statistics.median(r['median_step_ms'] for r in rows),
                          median_optimizer_ms=statistics.median(r['median_optimizer_ms'] for r in rows),
                          peak_bytes=max(r['peak_bytes'] for r in rows),
                          validation_after=[r['validation_after'] for r in rows],
                          repeats_exact=all(rows[0][k]==rows[1][k] for k in
                              ('losses','model_sha256','optimizer_sha256')))
    assert len({r['input_sha256'] for r in report['runs']}) == 1
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMPARISON',json.dumps({k:report[k] for k in ('fp32','mantissa')}),flush=True)


if __name__ == '__main__':
    main()
