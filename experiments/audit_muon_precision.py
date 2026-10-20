"""Audit split-weight Muon; BF16 momentum remains an experiment only.

Run from the repo root. Uses synthetic numeric token IDs (no external corpus),
never loads/saves training weights and never changes production defaults.
Storage measurements include real tensor bytes, steady CUDA allocation and step
peaks separately. Train probes use the same BF16 autocast, Muon/AdamW partition,
normalized top-2 MoE and held-out synthetic batches for all arms. Small-model
results do not establish convergence on V1 or long pretraining runs.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import datasets  # noqa: F401,E402 -- Windows DLL order
import trainer.compile_cache  # noqa: F401,E402
import torch  # noqa: E402
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from trainer.batched_muon import BatchedMuon, _reconstruct_precise, _split_precise
from trainer.trainer_utils import (
    CombinedOptimizer, _is_muon_hidden_matrix, build_optimizer, cast_training_parameters,
)


class BF16MomentumMuon(BatchedMuon):
    """Experimental actual 2-byte momentum storage, FP32 arithmetic per chunk.

    Quantize AFTER Nesterov/NS consumes the updated FP32 momentum, matching the
    old delta script's simulation. Super promotes the stored BF16 next step.
    This is not the same arithmetic as directly performing lerp in BF16.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_mantissa=True, **kwargs)

    def _step_chunk(self, params, group):
        super()._step_chunk(params, group)
        for p in params:
            self.state[p]['momentum_buffer'] = self.state[p]['momentum_buffer'].to(torch.bfloat16)

    def state_dict(self):
        # Prevent accidental use of this research state by production trainers.
        state = super().state_dict()
        state['muon_precision'] = 'experimental_bf16_momentum_v1'
        return state


def make_opt(params, arm, **kwargs):
    if arm == 'mantissa_bf16mom':
        return BF16MomentumMuon(params, **kwargs)
    return BatchedMuon(params, use_mantissa=arm == 'mantissa_fp32mom', **kwargs)


def tensor_bytes(params, opt):
    state = [t for entry in opt.state.values() for t in entry.values() if isinstance(t, torch.Tensor)]
    groups = {'parameters': list(params), 'gradients': [p.grad for p in params if p.grad is not None],
              'optimizer': state}
    return {k: sum(t.numel() * t.element_size() for t in ts) for k, ts in groups.items()}


def precise(p, opt):
    return _reconstruct_precise(p, opt.state[p]['mantissa']) if opt.use_mantissa else p.detach().float()


def production_shapes():
    config = InstinctConfig(**json.loads((ROOT / 'trainer/config_instinct_v1_moe.json').read_text()))
    with torch.device('meta'):
        model = InstinctForCausalLM(config)
    hidden = [(n, p) for n, p in model.named_parameters() if _is_muon_hidden_matrix(n, p)]
    muon_count = sum(p.numel() for _, p in hidden)
    other_count = sum(p.numel() for n, p in model.named_parameters() if not _is_muon_hidden_matrix(n, p))
    # Two complete production blocks: actual expert count and aspect ratios.
    shapes = [tuple(p.shape) for n, p in hidden if n.startswith(('model.layers.0.', 'model.layers.1.'))]
    if not shapes:
        raise RuntimeError('No V1 production block shapes found')
    return shapes, {'muon_elements': muon_count, 'adamw_elements': other_count,
                    'total_elements': muon_count + other_count,
                    'persistent_bytes_including_gradients': {
                        name: muon_count * cost + other_count * 16 for name, cost in
                        [('fp32', 12), ('mantissa_fp32mom', 10), ('mantissa_bf16mom', 8)]}}


def storage(args):
    shapes, counts = production_shapes()
    results = []
    for arm in ('fp32', 'mantissa_fp32mom', 'mantissa_bf16mom', 'naive_bf16'):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.manual_seed(args.seed)
        dtype = torch.float32 if arm == 'fp32' else torch.bfloat16
        params = [torch.nn.Parameter(torch.randn(s, device='cuda', dtype=dtype) * .02) for s in shapes]
        for p in params:
            p.grad = torch.randn_like(p)
        # Mirror factory defaults on this machine, then record them explicitly.
        workspace = args.workspace_mb or (256 if arm == 'fp32' else 64)
        opt = make_opt(params, arm, lr=5e-4, adjust_lr_fn='match_rms_adamw', workspace_mb=workspace)
        for _ in range(args.warmup):
            opt.step()
        torch.cuda.synchronize()
        persistent = torch.cuda.memory_allocated() - base
        owned = tensor_bytes(params, opt)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            opt.step()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        result = dict(arm=arm, elements=sum(p.numel() for p in params), tensor_bytes=owned,
                      bytes_per_element=sum(owned.values()) / sum(p.numel() for p in params),
                      cuda_persistent_bytes=persistent,
                      cuda_step_peak_bytes=torch.cuda.max_memory_allocated() - base,
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      median_ms=statistics.median(times), workspace_mb=workspace,
                      momentum_dtypes=sorted({str(opt.state[p]['momentum_buffer'].dtype) for p in params}))
        results.append(result)
        print('storage', json.dumps(result), flush=True)
        del p, params, opt
    return {'shapes': shapes, 'v1_model': counts, 'measurements': results}


def trajectory(args):
    results = []
    for lr in (5e-4, 2e-5):
        torch.manual_seed(args.seed)
        initial = [torch.randn(s, device='cuda') * .02 for s in [(64, 64), (128, 64), (64, 128)]]
        arms = {}
        for name in ('fp32', 'mantissa_fp32mom', 'mantissa_bf16mom', 'naive_bf16'):
            params = [torch.nn.Parameter(p.clone() if name == 'fp32' else _split_precise(p)[0]) for p in initial]
            opt = make_opt(params, name, lr=lr, weight_decay=.1, adjust_lr_fn='match_rms_adamw')
            if opt.use_mantissa:
                for p, value in zip(params, initial):
                    opt.state[p]['mantissa'] = _split_precise(value)[1]
            arms[name] = params, opt
        exact = {name: 0 for name in arms}
        for step in range(args.steps):
            gradients = [torch.randn_like(p).bfloat16() for p in initial]
            for name, (params, opt) in arms.items():
                for i, (p, g) in enumerate(zip(params, gradients)):
                    p.grad = None if i == 1 and step % 7 == 0 else g.to(p.dtype)
                opt.step()
            reference = arms['fp32'][0]
            for name, (params, opt) in arms.items():
                exact[name] += all(torch.equal(precise(p, opt), ref) for p, ref in zip(params, reference))
        reference = torch.cat([p.detach().flatten().double() for p in arms['fp32'][0]])
        origin = torch.cat([p.flatten().double() for p in initial])
        ref_delta = reference - origin
        for name, (params, opt) in arms.items():
            current = torch.cat([precise(p, opt).flatten().double() for p in params])
            delta = current - origin
            result = dict(lr=lr, arm=name, exact_step_fraction=exact[name] / args.steps,
                          relative_to_ref_displacement=((current-reference).norm()/ref_delta.norm()).item(),
                          cosine=torch.nn.functional.cosine_similarity(delta, ref_delta, dim=0).item(),
                          max_abs=(current-reference).abs().max().item())
            results.append(result)
            print('trajectory', json.dumps(result), flush=True)
    return results


def training(args):
    results = []
    for moe in (False, True):
        for seed in (args.seed, args.seed+1, args.seed+2):
            generator = torch.Generator().manual_seed(seed + 1000)
            def batch():
                return torch.randint(1, 128, (4, 8), generator=generator).repeat(1, 8).cuda()
            batches = [batch() for _ in range(args.steps)]
            validation = [batch() for _ in range(8)]
            for arm in ('fp32', 'fp32_repeat', 'mantissa_fp32mom', 'mantissa_bf16mom'):
                gc.collect()
                torch.cuda.empty_cache()
                torch.manual_seed(seed)
                config = InstinctConfig(hidden_size=64, num_hidden_layers=2, vocab_size=128,
                                        num_attention_heads=4, num_key_value_heads=2,
                                        max_position_embeddings=64, flash_attn=False, use_moe=moe,
                                        num_experts=4, num_experts_per_tok=2, norm_topk_prob=True,
                                        use_grad_checkpoint=1, dropout=0.)
                split = arm.startswith('mantissa')
                os.environ['INSTINCT_MUON_MANTISSA'] = '1' if split else '0'
                model = cast_training_parameters(InstinctForCausalLM(config), 'bf16' if split else 'fp32').cuda()
                opt = build_optimizer(model.named_parameters(), lr=5e-4, optimizer='muon')
                if arm == 'mantissa_bf16mom':
                    original, adamw = opt.optimizers
                    candidate = BF16MomentumMuon(original.param_groups, batch_size=original.batch_size,
                                                 workspace_mb=original.workspace_bytes / 2**20)
                    opt = CombinedOptimizer([candidate, adamw])
                    del original, adamw, candidate
                def loss(tokens):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = model(tokens, labels=tokens)
                        return output.loss + output.aux_loss
                def evaluate():
                    model.eval()
                    with torch.no_grad():
                        value = statistics.fmean(loss(b).item() for b in validation)
                    model.train()
                    return value
                before = evaluate()
                losses = []
                for step, tokens in enumerate(batches):
                    for group in opt.param_groups:
                        group['lr'] = 5e-4 * (.1 + .9 * (1-step/max(1, args.steps-1)))
                    opt.zero_grad(set_to_none=True)
                    value = loss(tokens)
                    if not torch.isfinite(value):
                        raise RuntimeError(f'Nonfinite loss: {arm}, seed {seed}, step {step}')
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    losses.append(value.item())
                after = evaluate()
                result = dict(moe=moe, seed=seed, arm=arm, validation_before=before,
                              validation_after=after, train_last20=statistics.fmean(losses[-20:]),
                              losses=losses)
                results.append(result)
                print('train', json.dumps({k:v for k,v in result.items() if k != 'losses'}), flush=True)
                del value, model, opt
            del batches, validation
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sections', default='storage,trajectory,train')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--workspace-mb', type=int, default=None,
                        help='Use the same workspace target for every storage arm')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    # Fixed operator settings for reproducibility, record all of them below.
    os.environ['INSTINCT_MUON_BACKEND'] = 'batched'
    os.environ['INSTINCT_MUON_MANTISSA'] = '0'
    # This historical audit explicitly models BF16 momentum with its reference
    # adapter; do not inherit the newer production option from the caller.
    os.environ['INSTINCT_MUON_MOMENTUM_DTYPE'] = 'fp32'
    torch.backends.cuda.matmul.allow_tf32 = False
    files = ['trainer/batched_muon.py', 'trainer/trainer_utils.py', 'experiments/audit_muon_precision.py']
    result = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                  protocol=vars(args) | {'output': str(args.output)},
                  env={k:v for k,v in os.environ.items() if k.startswith(('INSTINCT_MUON_', 'INSTINCT_MOE_'))},
                  synthetic_data='period-8 numeric token IDs; no external text; held-out batches use same distribution')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for section in args.sections.split(','):
        result[section] = {'storage': storage, 'trajectory': trajectory, 'train': training}[section](args)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()
