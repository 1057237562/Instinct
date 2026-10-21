"""Compare reference/repeat/candidate using one compiled graph and initialization.

This controls Inductor autotuning differences between fresh Python processes.
Each trajectory restarts identical FP32 weights, optimizer, inputs and LR.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.validate_full_training_kernels import (
    argparse, hashlib, json, os, torch, SimpleNamespace, fingerprint,
    InstinctConfig, InstinctForCausalLM, build_optimizer, apply_torchao_fp8_training,
    step_with_scaler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--output',required=True)
    parser.add_argument('--deterministic',action='store_true')
    args=parser.parse_args()
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
    os.environ['INSTINCT_PACKED_ATTENTION_BACKEND']='flex'
    torch.set_num_threads(1)
    torch.manual_seed(73)
    raw=json.loads(Path('trainer/config_instinct_v1_moe.json').read_text())
    raw.update(use_grad_checkpoint=1,param_dtype='fp32',norm_topk_prob=True)
    base=InstinctForCausalLM(InstinctConfig(**raw)).cuda().train()
    apply_torchao_fp8_training(base,SimpleNamespace(fp8_training='tensorwise',fp8_filter='auto',
        dtype='bfloat16',device='cuda',use_compile=1))
    initial={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    model=torch.compile(base,mode='max-autotune-no-cudagraphs')
    data=torch.randint(3,6400,(8,4,2048),device='cuda')
    labels=data.clone()
    labels[:,:,::512]=-100
    ids=(torch.arange(2048,device='cuda')//512)[None].expand(4,-1)
    report=dict(steps=args.steps,checkpoint=1,deterministic=args.deterministic,config=raw,torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),phases={},
                input_sha256=fingerprint([('data',data),('labels',labels),('ids',ids)]))
    for phase in ['reference','reference_repeat','candidate']:
        candidate=phase=='candidate'
        os.environ['INSTINCT_MUON_NS_BACKEND']='auto' if candidate else 'torch'
        os.environ['INSTINCT_MUON_FUSED_UPDATES']='auto' if candidate else '0'
        os.environ['INSTINCT_MOE_FUSED_PACK']='auto' if candidate else '0'
        os.environ['INSTINCT_MOE_FUSED_SWIGLU']='auto' if candidate else '0'
        base.load_state_dict(initial,strict=True)
        opt=build_optimizer(base.named_parameters(),lr=5e-4,optimizer='muon')
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        opt.zero_grad(set_to_none=True)
        row=dict(losses=[])
        print('PHASE',phase,flush=True)
        for i in range(args.steps):
            for group in opt.param_groups:
                group['lr']=5e-4*(1-.75*i/args.steps)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(data[i%8],labels=labels[i%8],sequence_ids=ids)
                loss=out.loss+out.aux_loss
            loss.backward()
            if i==0:
                row['first_grad_tensors']={}
                row['first_grad_sha256']=fingerprint(((n,p.grad) for n,p in base.named_parameters()),row['first_grad_tensors'])
                row['first_logits_sha256']=fingerprint([('logits',out.logits)])
            step_with_scaler(scaler,opt,base.parameters(),1.)
            if i==0:
                row['first_params_sha256']=fingerprint(base.named_parameters())
            row['losses'].append(float(loss.detach()))
            if (i+1)%20==0:
                print(phase,i+1,row['losses'][-1],flush=True)
        row['final_params_sha256']=fingerprint(base.named_parameters())
        state=[]
        for index,optimizer in enumerate(opt.optimizers):
            for name,p in base.named_parameters():
                for key,value in sorted(optimizer.state.get(p,{}).items()):
                    if isinstance(value,torch.Tensor):
                        state.append((f'{index}/{name}/{key}',value.reshape(-1)))
        row['final_optimizer_sha256']=fingerprint(state)
        reference=report['phases'].get('reference',row)
        row['bitwise_equal']=all(row[k]==reference[k] for k in
            ['losses','first_grad_sha256','first_logits_sha256','first_params_sha256',
             'final_params_sha256','final_optimizer_sha256'])
        report['phases'][phase]=row
        Path(args.output).write_text(json.dumps(report,indent=2))
        print(phase,'BITWISE',row['bitwise_equal'],flush=True)
        del state,opt,optimizer,out,loss
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
