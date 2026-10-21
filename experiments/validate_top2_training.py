"""Paired full-depth top-2 trajectories and update-sensitive error gates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
import argparse,json,gc
from types import SimpleNamespace
from model.model_instinct import InstinctConfig,InstinctForCausalLM
from trainer.trainer_utils import build_optimizer,apply_torchao_fp8_training
from trainer.trainer_cli import step_with_scaler


def snapshot(items):
    return {n:t.detach().cpu().clone() for n,t in items if t is not None}


def relative(actual,reference,initial=None):
    error=norm=0.; maximum=0.
    for name,value in actual.items():
        a=value.float(); b=reference[name].float()
        diff=a-b
        error+=float(diff.square().sum(dtype=torch.float64))
        denominator=b if initial is None else b-initial[name]
        norm+=float(denominator.square().sum(dtype=torch.float64))
        maximum=max(maximum,float(diff.abs().max()))
    return dict(relative_l2=(error/max(norm,1e-60))**.5,max_abs=maximum)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--selective',type=int,default=0)
    p.add_argument('--seq',type=int,default=4096)
    p.add_argument('--causal-flex',action='store_true')
    p.add_argument('--graph',action='store_true')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1); torch.manual_seed(73)
    raw=json.loads(Path('trainer/config_instinct_v1_moe.json').read_text())
    raw.update(use_grad_checkpoint=1,param_dtype='fp32')
    base=InstinctForCausalLM(InstinctConfig(**raw)).cuda().train()
    apply_torchao_fp8_training(base,SimpleNamespace(fp8_training='tensorwise',fp8_filter='auto',
        dtype='bfloat16',device='cuda',use_compile=1))
    initial=snapshot(base.state_dict().items())
    model=torch.compile(base,mode='max-autotune-no-cudagraphs')
    inputs=torch.randint(3,6400,(3,2,args.seq),device='cuda')
    ids=(torch.arange(args.seq,device='cuda')//1024)[None].expand(2,-1)
    labels=inputs.clone(); labels[:,:,::1024]=-100
    if args.causal_flex:
        ids=None; labels=inputs.clone()
    reference={}; report=dict(args=vars(args),config=raw,phases={})
    for phase in ('reference','reference_repeat','candidate'):
        os.environ['INSTINCT_MOE_TRAINING_GRAPH']='1' if phase=='candidate' and args.graph else '0'
        if args.graph:
            from experiments.moe_training_graph_candidate import install
            install(phase=='candidate')
        os.environ['INSTINCT_MOE_FUSED_ROUTING']='1' if phase=='candidate' else '0'
        os.environ['INSTINCT_MOE_SELECTIVE_CHECKPOINT']=str(args.selective if phase=='candidate' else 0)
        os.environ['INSTINCT_CAUSAL_ATTENTION_BACKEND']='flex' if phase=='candidate' and args.causal_flex else 'sdpa'
        base.load_state_dict(initial,strict=True)
        opt=build_optimizer(base.named_parameters(),lr=5e-4,optimizer='muon')
        opt.zero_grad(set_to_none=True)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        row=dict(losses=[],comparisons={})
        for step in range(args.steps):
            for group in opt.param_groups: group['lr']=5e-4*(1-.75*step/args.steps)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(inputs[step%3],labels=labels[step%3],sequence_ids=ids)
                loss=out.loss+out.aux_loss
            loss.backward()
            if step==0:
                current={'logits':{'logits':out.logits.detach().float().cpu()},
                         'gradient':snapshot((n,p.grad) for n,p in base.named_parameters())}
                for key,value in current.items():
                    if phase=='reference': reference[key]=value
                    row['comparisons'][key]=relative(value,reference[key])
                del current
            step_with_scaler(scaler,opt,base.parameters(),1.)
            if step==0:
                value=snapshot(base.named_parameters())
                if phase=='reference': reference['first_params']=value
                row['comparisons']['update']=relative(value,reference['first_params'],initial)
                del value
            row['losses'].append(float(loss.detach()))
            assert torch.isfinite(loss)
            if (step+1)%5==0: print(phase,step+1,row['losses'][-1],flush=True)
        value=snapshot(base.named_parameters())
        if phase=='reference': reference['final_params']=value
        row['comparisons']['final_parameters']=relative(value,reference['final_params'])
        baseline=report['phases'].get('reference',row)['losses']
        row['max_loss_relative']=max(abs(a-b)/max(abs(b),1e-12) for a,b in zip(row['losses'],baseline))
        limits={'logits':.005,'gradient':.02,'update':.03,'final_parameters':.001}
        row['passed']=all(row['comparisons'][k]['relative_l2']<=v for k,v in limits.items()) and row['max_loss_relative']<=.02
        report['phases'][phase]=row
        Path(args.output).write_text(json.dumps(report,indent=2))
        print(phase,json.dumps({k:v for k,v in row.items() if k!='losses'}),flush=True)
        del value,opt,out,loss
        gc.collect(); torch.cuda.empty_cache()
    assert all(row['passed'] for row in report['phases'].values())


if __name__=='__main__': main()
