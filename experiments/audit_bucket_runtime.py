"""Attribute live CUDA allocations across V2 buckets without optimizer updates.

No corpus cursor or weights are changed. Snapshots are taken after backward at
safe boundaries. The optional cache-release stages only affect this probe.
"""
import argparse
from collections import defaultdict
from functools import partial
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trainer.compile_cache  # noqa: F401
import datasets  # noqa: F401 -- before torch on Windows
import torch
from types import SimpleNamespace

from experiments.profile_checkpoint_policy import reserve_training_state
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from trainer.trainer_utils import cast_training_parameters, apply_torchao_fp8_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=int, default=32)
    parser.add_argument('--policy', choices=['recipe', 'attention_gate'], default='recipe')
    parser.add_argument('--history', choices=['on', 'off'], default='on')
    parser.add_argument('--skip-reset', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(INSTINCT_MUON_MANTISSA='1', INSTINCT_MUON_MOMENTUM_DTYPE='bf16',
        INSTINCT_MUON_BATCH_SIZE='16', INSTINCT_MUON_WORKSPACE_MB='512')
    torch.set_num_threads(1)
    torch.manual_seed(307)
    torch.cuda.set_per_process_memory_fraction(14.37 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    if args.history == 'on':
        torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='python', max_entries=30000)
    config = InstinctConfig(**dict(json.loads((ROOT/'trainer/config_instinct_v2.json').read_text()),
        use_grad_checkpoint=4, num_hidden_layers=args.layers))
    if args.policy != 'recipe':
        import model.model_instinct as module
        from model.checkpointing import dense_checkpoint_options
        module.dense_checkpoint_options = partial(dense_checkpoint_options,
            mode4_policy=args.policy, mode4_gate_layers=2)
    raw = cast_training_parameters(InstinctForCausalLM(config), 'bf16').cuda().train()
    optimizer = reserve_training_state(raw)
    apply_torchao_fp8_training(raw, SimpleNamespace(fp8_training='tensorwise', fp8_filter='auto',
        dtype='bfloat16', device='cuda', use_compile=1))
    model = torch.compile(raw, mode='max-autotune-no-cudagraphs', dynamic=False)
    batches = []
    for b, s, segments in ((4,3072,3),(3,4096,1)):
        tokens = torch.randint(3,6400,(b,s),device='cuda')
        labels = tokens.clone()
        ids = None
        if segments > 1:
            ids = (torch.arange(s,device='cuda')*segments//s).expand(b,-1).contiguous()
            labels[:,1:][ids[:,1:] != ids[:,:-1]] = -100
        batches.append((tokens,labels,ids))
    report = dict(status='running', args=vars(args), config=config.to_dict(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), compile_dynamic=False,
        compile_mode='max-autotune-no-cudagraphs', allocator_limit_gib=14.37,
        scope='fixed weights, real resident optimizer states and gradients; no optimizer updates',
        samples=[], steps=[])

    def write():
        target.write_text(json.dumps(report,indent=2),encoding='utf-8')

    def inventory():
        owners = {}
        def add(t, category, name):
            if isinstance(t,torch.Tensor) and t.device.type == 'cuda':
                storage = t.untyped_storage()
                owners.setdefault(storage.data_ptr(), dict(category=category, name=name,
                    bytes=storage.nbytes(), shape=list(t.shape), dtype=str(t.dtype)))
        for name,p in raw.named_parameters():
            add(p,'parameters',name)
            add(p.grad,'gradients',name)
        for name,t in raw.named_buffers():
            add(t,'model_buffers',name)
        for opt in optimizer.optimizers:
            for index,state in enumerate(opt.state.values()):
                for key,t in state.items():
                    add(t,'optimizer',f'{type(opt).__name__}.{index}.{key}')
        for index,batch in enumerate(batches):
            for field,t in enumerate(batch):
                add(t,'inputs',f'{index}.{field}')
        return owners

    def snapshot(label):
        torch.cuda.synchronize()
        owners = inventory()
        totals = defaultdict(int)
        unknown = []
        private_pools = defaultdict(int)
        for segment in torch.cuda.memory_snapshot():
            pool = segment.get('segment_pool_id',(0,0))
            if tuple(pool) != (0,0):
                private_pools[str(pool)] += segment['total_size']
            address = segment['address']
            for block in segment['blocks']:
                ptr = block.get('address',address)
                address += block['size']
                if block['state'] != 'active_allocated':
                    totals[block['state']] += block['size']
                    continue
                known = owners.get(ptr)
                if known:
                    totals[known['category']] += block['size']
                else:
                    totals['unattributed_live'] += block['size']
                    unknown.append(dict(address=ptr, size=block['size'], requested=block['requested_size'],
                        frames=block.get('frames',[])))
        from torch._dynamo.utils import counters
        from model.packed_attention import _causal_flex_mask, _flex_functions
        free,total = torch.cuda.mem_get_info()
        sample = dict(label=label, allocated_gib=torch.cuda.memory_allocated()/2**30,
            reserved_gib=torch.cuda.memory_reserved()/2**30, device_free_gib=free/2**30,
            totals_mib={k:v/2**20 for k,v in totals.items()}, private_pool_bytes=dict(private_pools),
            compiled_graphs=dict(counters['stats']), mask_cache=_causal_flex_mask.cache_info()._asdict(),
            flex_function_cache=_flex_functions.cache_info()._asdict(),
            unknown_blocks=sorted(unknown,key=lambda x:-x['size']))
        report['samples'].append(sample)
        write()
        print(json.dumps({k:v for k,v in sample.items() if k != 'unknown_blocks'}),flush=True)

    def step(bucket, name):
        model.zero_grad(set_to_none=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        tokens,labels,ids = batches[bucket]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out = model(tokens,labels=labels,sequence_ids=ids)
            loss = out.loss
            del out
        loss.backward()
        torch.cuda.synchronize()
        row = dict(label=name,bucket=bucket,ms=(time.perf_counter()-start)*1000,loss=float(loss.detach()),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        report['steps'].append(row)
        write()
        print(json.dumps(row),flush=True)
        return loss.detach()

    try:
        snapshot('initialized')
        for bucket,name in ((0,'A1'),(0,'A2'),(1,'B1'),(1,'B2'),(0,'A3')):
            kept_loss = step(bucket,name)
            snapshot(name+'_with_loss')
            del kept_loss
            snapshot(name+'_without_loss')
        gc.collect()
        snapshot('after_gc')
        torch.cuda.empty_cache()
        snapshot('after_empty_cache')
        from model.packed_attention import _causal_flex_mask
        _causal_flex_mask.cache_clear()
        gc.collect()
        snapshot('after_causal_mask_clear')
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
        snapshot('after_cublas_clear')
        torch.cuda.empty_cache()
        snapshot('after_cublas_empty_cache')
        kept_loss = step(0,'A_after_light_release')
        del kept_loss
        snapshot('after_light_release_reuse')
        if not args.skip_reset:
            torch.compiler.reset()
            gc.collect()
            torch.cuda.empty_cache()
            snapshot('after_compiler_reset')
            kept_loss = step(0,'A_after_reset')
            del kept_loss
            snapshot('after_reset_reuse')
        report['status'] = 'ok'
    except Exception as exc:
        report.update(status='error',error=f'{type(exc).__name__}: {exc}')
        snapshot('error_boundary')
        raise
    finally:
        write()
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == '__main__':
    main()
