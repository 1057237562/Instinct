"""32-layer pretrain A/B fingerprints with changing inputs and learning rate.

Run reference and candidate sequentially. SHA-256 fingerprints cover all model
gradients/parameters and optimizer tensor states, not just aggregate norms.
Does not read or write production weights, datasets, or resume checkpoints.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import torch
from model.model_instinct import InstinctConfig,InstinctForCausalLM
from trainer.trainer_utils import build_optimizer,apply_torchao_fp8_training
from trainer.trainer_cli import step_with_scaler


def fingerprint(items, per_tensor=None):
    digest=hashlib.sha256()
    for name,t in items:
        digest.update(name.encode())
        if t is None:
            digest.update(b'None')
        else:
            digest.update(str((t.dtype,tuple(t.shape))).encode())
            raw=t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            digest.update(raw)
            if per_tensor is not None:
                per_tensor[name]=hashlib.sha256(raw).hexdigest()
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',action='store_true')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    os.environ['INSTINCT_MUON_NS_BACKEND']='auto' if args.candidate else 'torch'
    os.environ['INSTINCT_MUON_FUSED_UPDATES']='auto' if args.candidate else '0'
    os.environ['INSTINCT_MOE_FUSED_PACK']='auto' if args.candidate else '0'
    os.environ['INSTINCT_MOE_FUSED_SWIGLU']='auto' if args.candidate else '0'
    os.environ['INSTINCT_PACKED_ATTENTION_BACKEND']='flex'
    torch.set_num_threads(1)
    torch.manual_seed(73)
    raw=json.loads(Path('trainer/config_instinct_v1_moe.json').read_text())
    raw.update(use_grad_checkpoint=1,param_dtype='fp32',norm_topk_prob=True)
    base=InstinctForCausalLM(InstinctConfig(**raw)).cuda().train()
    opt=build_optimizer(base.named_parameters(),lr=5e-4,optimizer='muon')
    apply_torchao_fp8_training(base,SimpleNamespace(fp8_training='tensorwise',fp8_filter='auto',
        dtype='bfloat16',device='cuda',use_compile=1))
    model=torch.compile(base,mode='max-autotune-no-cudagraphs')
    data=torch.randint(3,6400,(8,4,2048),device='cuda')
    labels=data.clone()
    labels[:,:,::512]=-100
    ids=(torch.arange(2048,device='cuda')//512)[None].expand(4,-1)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    report=dict(candidate=args.candidate,steps=args.steps,losses=[],checkpoint=1,
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,config=raw,
                input_sha256=fingerprint([('data',data),('labels',labels),('ids',ids)]),
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                ['trainer/batched_muon.py','trainer/muon_kernels.py','model/expert_weight_pack.py',
                 'model/moe_dispatch.py','experiments/validate_full_training_kernels.py']})
    opt.zero_grad(set_to_none=True)
    for i in range(args.steps):
        lr=5e-4*(1-.75*i/args.steps)
        for group in opt.param_groups:
            group['lr']=lr
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(data[i%8],labels=labels[i%8],sequence_ids=ids)
            loss=out.loss+out.aux_loss
        loss.backward()
        if i==0:
            report['first_grad_tensors']={}
            report['first_grad_sha256']=fingerprint(((n,p.grad) for n,p in base.named_parameters()),report['first_grad_tensors'])
            report['first_logits_sha256']=fingerprint([('logits',out.logits)])
        step_with_scaler(scaler,opt,base.parameters(),1.)
        if i==0:
            report['first_param_tensors']={}
            report['first_params_sha256']=fingerprint(base.named_parameters(),report['first_param_tensors'])
        report['losses'].append(float(loss.detach()))
        if (i+1)%20==0:
            print('validated step',i+1,'loss',report['losses'][-1],flush=True)
    report['final_params_sha256']=fingerprint(base.named_parameters())
    state=[]
    for index,optimizer in enumerate(opt.optimizers):
        for name,p in base.named_parameters():
            for key,value in sorted(optimizer.state.get(p,{}).items()):
                if isinstance(value,torch.Tensor):
                    state.append((f'{index}/{name}/{key}',value.reshape(-1)))
    report['final_optimizer_sha256']=fingerprint(state)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if 'sha256' in k}),flush=True)


if __name__=='__main__':
    main()
