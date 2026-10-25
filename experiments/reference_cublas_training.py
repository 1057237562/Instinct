"""Separate CPU oracle for native GPU validation. Not imported by the trainer."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
import numpy as np
import datasets  # Preserve the project's Windows DLL import order.
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(2)
    with np.load(args.input,allow_pickle=False) as z:
        config=json.loads(str(z['config']));arch=config.get('model_architecture','standard')
        if arch=='looped':raise ValueError('native Looped has new WebUI controls; it is not the legacy loop_iters model')
        if arch=='linear':from model.model_instinct_linear import InstinctConfig,InstinctForCausalLM
        elif arch=='looped_legacy':from model.model_instinct_loop import InstinctConfig,InstinctForCausalLM
        else:from model.model_instinct import InstinctConfig,InstinctForCausalLM
        model=InstinctForCausalLM(InstinctConfig(**config)).float();model.train()
        model.load_state_dict({k.removeprefix('parameter/'):torch.from_numpy(z[k].copy()) for k in z if k.startswith('parameter/')})
        ids=torch.from_numpy(z['ids'].copy());out=model(ids,labels=ids,early_exit=bool(config.get('early_exit_layers')))
        loss=out.loss+out.aux_loss;loss.backward()
        result={'logits':out.logits.detach().numpy(),'loss':np.asarray([loss.item()],np.float32)}
        for name,param in model.named_parameters():
            if param.grad is not None:result['gradient/'+name]=param.grad.numpy()
        np.savez(args.output,**result)


if __name__=='__main__':main()
