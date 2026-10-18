"""Seeded GPU correctness gates for the MFU experiments, separate from timing.

Uses the real 512/1664 expert shapes in a two-layer model. Tests first-step
logits/loss/gradients/updates, a 25-step learnable synthetic trajectory, document
isolation and optimizer resume. This is a smoke gate, not a convergence proof.
No real corpus or training checkpoint is read or changed.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache  # noqa: F401
import datasets  # noqa: F401 -- Windows DLL order
import torch

import trainer.batched_muon as muon
import model.moe_dispatch as dispatch
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from trainer.trainer_cli import step_with_scaler
from trainer.trainer_utils import apply_torchao_fp8_training, build_optimizer
from experiments.bench_training_mfu import install_experimental_ns, install_cast_before_stack


def relative(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def run(attention, method, dtype, steps, fp8, original, compiled_ns=False,
        workspace_mb=64, cast_first=False, checkpoint=1, ns_backend='torch', fused_updates=False,
        fused_pack=False, fused_swiglu=False):
    os.environ['INSTINCT_MUON_NS_BACKEND'] = ns_backend
    os.environ['INSTINCT_MUON_FUSED_UPDATES'] = '1' if fused_updates else '0'
    os.environ['INSTINCT_MOE_FUSED_PACK'] = '1' if fused_pack else '0'
    os.environ['INSTINCT_MOE_FUSED_SWIGLU'] = '1' if fused_swiglu else '0'
    muon._batched_zeropower = original
    install_experimental_ns(method, dtype)
    if compiled_ns:
        muon._batched_zeropower = torch.compile(muon._batched_zeropower, fullgraph=True,
                                               mode='reduce-overhead')
    os.environ['INSTINCT_PACKED_ATTENTION_BACKEND'] = attention
    os.environ['INSTINCT_MUON_WORKSPACE_MB'] = str(workspace_mb)
    if cast_first:
        install_cast_before_stack()
    torch.manual_seed(127)
    config = InstinctConfig(hidden_size=512, num_hidden_layers=2, vocab_size=6400,
                            num_attention_heads=16, num_key_value_heads=4, head_dim=32,
                            use_moe=True, num_experts=8, num_experts_per_tok=1,
                            moe_intermediate_size=1664, norm_topk_prob=True,
                            use_grad_checkpoint=checkpoint, dropout=0.0, flash_attn=True)
    model = InstinctForCausalLM(config).cuda().train()
    optimizer = build_optimizer(model.named_parameters(), lr=5e-4, optimizer='muon')
    apply_torchao_fp8_training(model, SimpleNamespace(fp8_training=fp8, fp8_filter='auto',
                              dtype='bfloat16', device='cuda', use_compile=0))
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    generator = torch.Generator().manual_seed(928)
    base = torch.randint(3, 6400, (2, 8), generator=generator)
    tokens = base.repeat(1, 32).cuda()
    ids = (torch.arange(256, device='cuda') // 64)[None].expand(2, -1)
    labels = tokens.clone()
    labels[:, ::64] = -100
    optimizer.zero_grad(set_to_none=True)
    initial = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    losses = []
    first_logits = first_grad = first_update = None
    isolation = 0.0
    resume_equal = False
    for i in range(steps):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(tokens, labels=labels, sequence_ids=ids)
            loss = result.loss + result.aux_loss
        losses.append(float(loss.detach()))
        loss.backward()
        if i == 0:
            first_logits = result.logits.detach().float().cpu().flatten()
            first_grad = torch.cat([p.grad.detach().flatten().cpu() for p in model.parameters()])
        step_with_scaler(scaler, optimizer, model.parameters(), 1.0)
        if i == 0:
            first_update = torch.cat([p.detach().flatten().cpu() for p in model.parameters()]) - initial
        if i == steps // 2:
            # Reload unchanged state and replay the same next update from the
            # same gradients, checking actual state values as well as layouts.
            model_state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(optimizer.state_dict())
            for p in model.parameters():
                p.grad = torch.full_like(p, 0.001)
            optimizer.step()
            expected = [p.detach().clone() for p in model.parameters()]
            model.load_state_dict(model_state, strict=True)
            optimizer.load_state_dict(copy.deepcopy(opt_state))
            for p in model.parameters():
                p.grad = torch.full_like(p, 0.001)
            optimizer.step()
            resume_equal = all(torch.equal(p, e) for p, e in zip(model.parameters(), expected))
            model.load_state_dict(model_state, strict=True)
            optimizer.load_state_dict(copy.deepcopy(opt_state))
            optimizer.zero_grad(set_to_none=True)
            del model_state, opt_state, expected
    # A changed later document cannot affect logits of the first document.
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model(tokens, sequence_ids=ids).logits[:, :64].float()
        changed = tokens.clone()
        changed[:, 64:] = (changed[:, 64:] + 1) % 6400
        actual = model(changed, sequence_ids=ids).logits[:, :64].float()
        isolation = (actual - expected).abs().max().item()
    final = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    result = dict(losses=losses, logits=first_logits, grads=first_grad, update=first_update,
                  params=final, isolation_max_abs=isolation, resume_bitwise_equal=resume_equal,
                  finite=bool(torch.isfinite(final).all()))
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--fp8', choices=['off', 'tensorwise'], default='tensorwise')
    p.add_argument('--output', required=True)
    p.add_argument('--suite', choices=['original', 'kernels'], default='original')
    args = p.parse_args()
    torch.set_num_threads(1)
    original = muon._batched_zeropower
    original_stack = dispatch._stacked_expert_weights
    variants = [('flex_control', {}), ('flex_repeat', {}),
                ('cast_before_stack', {'cast_first': True}),
                ('workspace256', {'workspace_mb': 256}),
                ('no_checkpoint', {'checkpoint': 0}),
                ('sdpa_reference', {'attention': 'sdpa'}),
                ('gram_fp16', {'method': 'gram', 'dtype': 'fp16', 'workspace_mb': 256}),
                ('compiled_bf16', {'compiled_ns': True, 'workspace_mb': 256})]
    if args.suite == 'kernels':
        variants = [('torch_reference', {}), ('triton_ns', {'ns_backend': 'triton'}),
                    ('triton_graph', {'ns_backend': 'triton_graph'}),
                    ('triton_fused', {'ns_backend': 'triton_graph', 'fused_updates': True}),
                    ('triton_fused_pack', {'ns_backend': 'triton_graph', 'fused_updates': True, 'fused_pack': True}),
                    ('all_fused', {'ns_backend': 'triton_graph', 'fused_updates': True, 'fused_pack': True, 'fused_swiglu': True})]
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'steps': args.steps, 'fp8': args.fp8,
              'limits': {'logit_rel_l2': 0.005, 'gradient_rel_l2': 0.02,
                         'update_rel_l2': 0.03, 'trajectory_loss_relative_max': 0.02,
                         'final_params_relative_l2': 0.001}, 'variants': {}}
    reference = None
    try:
        for name, options in variants:
            print('VALIDATE', name, flush=True)
            dispatch._stacked_expert_weights = original_stack
            settings = dict(attention='flex', method='standard', dtype='bf16')
            settings.update(options)
            values = run(steps=args.steps, fp8=args.fp8, original=original, **settings)
            if reference is None:
                reference = values
            row = {key: values[key] for key in ['losses', 'isolation_max_abs', 'resume_bitwise_equal', 'finite']}
            row.update(logit_rel_l2=relative(values['logits'], reference['logits']),
                       gradient_rel_l2=relative(values['grads'], reference['grads']),
                       update_rel_l2=relative(values['update'], reference['update']),
                       trajectory_loss_relative_max=max(abs(a-b)/max(abs(b), 1e-6)
                                                        for a,b in zip(values['losses'], reference['losses'])),
                       final_params_relative_l2=relative(values['params'], reference['params']))
            row['passed'] = (all(row[key] < limit for key, limit in report['limits'].items())
                             and row['isolation_max_abs'] == 0 and row['resume_bitwise_equal'] and row['finite'])
            report['variants'][name] = row
            print(json.dumps({k:v for k,v in row.items() if k!='losses'}, indent=2), flush=True)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        muon._batched_zeropower = original
        dispatch._stacked_expert_weights = original_stack


if __name__ == '__main__':
    main()
