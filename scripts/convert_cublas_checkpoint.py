"""Offline CPU-only .pth <-> native .npz weight conversion (never run in CUDA trainer)."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    # Keep the Windows pyarrow DLL ordering, confined to this separate process.
    import datasets
    import torch
    if args.input.endswith('.pth'):
        loaded=torch.load(args.input,map_location='cpu',weights_only=True)
        if 'model' in loaded and isinstance(loaded['model'],dict):loaded=loaded['model']
        if 'state_dict' in loaded:loaded=loaded['state_dict']
        arrays={}
        for name,value in loaded.items():
            while name.startswith(('module.','_orig_mod.')):
                name=name.split('.',1)[1]
            if isinstance(value,torch.Tensor):arrays['model/'+name]=value.detach().float().numpy()
        arrays['__metadata__']=np.asarray(json.dumps({'format':'instinct-cublas-v1','source':args.input,'weights_only':True}))
        with open(args.output,'wb') as f:np.savez(f,**arrays)
    else:
        with np.load(args.input,allow_pickle=False) as z:
            state={name.removeprefix('model/'):torch.from_numpy(z[name].copy()) for name in z if name.startswith('model/')}
        torch.save(state,args.output)
    print(args.output)

if __name__=='__main__':main()
