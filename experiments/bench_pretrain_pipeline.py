"""Bounded production train_epoch replay on an audited local corpus.

Runs the real pretrain loop, H2D, LR schedule, clip/optimizer, timing and logs.
Preprocessing is outside the warm steady-state window, like a warm packing
cache. No training checkpoint is read or saved; iters deliberately extends
beyond the bounded loader, so the production epoch-end save is not reached.
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.compile_cache
import datasets
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from scripts.data_loader.lm_dataset import _best_fit_pack
import trainer.train_pretrain as trainer
from model.model_instinct import InstinctConfig,InstinctForCausalLM
from trainer.trainer_utils import build_optimizer,apply_torchao_fp8_training
from trainer.training_profiler import TrainingProfiler


class RecordingProfiler(TrainingProfiler):
    def __init__(self, *args, **kwargs):
        self.windows = []
        super().__init__(*args, **kwargs)

    def _report(self):
        metrics = super()._report()
        if metrics:
            self.windows.append(metrics)
        return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',action='store_true')
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    os.environ['INSTINCT_MUON_NS_BACKEND']='torch' if args.reference else 'auto'
    os.environ['INSTINCT_MUON_FUSED_UPDATES']='0' if args.reference else 'auto'
    os.environ['INSTINCT_MOE_FUSED_PACK']='0' if args.reference else 'auto'
    os.environ['INSTINCT_MOE_FUSED_SWIGLU']='0' if args.reference else 'auto'
    os.environ['INSTINCT_PACKED_ATTENTION_BACKEND']='flex'
    corpus=Path('dataset/pretrain_coder_12b_seg2_identity_clean.parquet')
    audit=json.loads(corpus.with_suffix('.parquet.report.json').read_text())
    identity=json.loads(corpus.with_suffix('.report.json').read_text())
    if not identity.get('identity_review'):
        raise RuntimeError('Identity review required before corpus replay')
    with corpus.open('rb') as f:
        corpus_hash=hashlib.file_digest(f,'sha256').hexdigest()
    if corpus_hash!=audit['output']['sha256']:
        raise RuntimeError('Corpus audit SHA-256 mismatch')
    table=next(pq.ParquetFile(corpus).iter_batches(batch_size=2048,columns=['text']))
    tokenizer=AutoTokenizer.from_pretrained('model',local_files_only=True)
    encoded=tokenizer(table.column(0).to_pylist(),add_special_tokens=False,truncation=False).input_ids
    sequences=[[tokenizer.bos_token_id,*ids,tokenizer.eos_token_id] for ids in encoded if len(ids)+2<=2048]
    packed=_best_fit_pack(sequences,[list(s) for s in sequences],2048,tokenizer.pad_token_id)
    batches=[]
    for start in range(0,min(len(packed['input_ids'])//4*4,128),4):
        batches.append(tuple(torch.tensor(packed[key][start:start+4],dtype=torch.long).pin_memory()
                             for key in ('input_ids','labels','sequence_ids')))
    if not batches:
        raise RuntimeError('No full packed batches')
    data_hash=hashlib.sha256()
    for batch in batches:
        for tensor in batch:
            data_hash.update(tensor.numpy().tobytes())
    torch.set_num_threads(1)
    torch.manual_seed(73)
    raw=json.loads(Path('trainer/config_instinct_v1_moe.json').read_text())
    raw.update(use_grad_checkpoint=1,param_dtype='fp32',norm_topk_prob=True)
    config=InstinctConfig(**raw)
    base=InstinctForCausalLM(config).cuda().train()
    optimizer=build_optimizer(base.named_parameters(),lr=5e-4,optimizer='muon')
    apply_torchao_fp8_training(base,SimpleNamespace(fp8_training='tensorwise',fp8_filter='auto',
         dtype='bfloat16',device='cuda',use_compile=1))
    trainer.model=torch.compile(base,mode='max-autotune-no-cudagraphs')
    trainer.optimizer=optimizer
    trainer.lm_config=config
    trainer.scaler=torch.amp.GradScaler('cuda',enabled=False)
    trainer.autocast_ctx=torch.autocast('cuda',dtype=torch.bfloat16)
    trainer.data_config={}
    trainer.args=SimpleNamespace(device='cuda',epochs=1,learning_rate=5e-4,grad_clip=1.,
        accumulation_steps=1,early_exit=0,use_compile=1,log_interval=50,save_interval=10**9,
        profile='timing',profile_interval=50,profile_warmup=20,
        pause_file='experiments/results/mfu_20261012/nonexistent_pipeline_pause')
    trainer.profiler=RecordingProfiler(trainer.args)
    warmup=20
    total=warmup+args.steps
    metrics=dict(useful_tokens=0,loss_tokens=0,causal_pairs=0)
    class Loader:
        def __iter__(self):
            for i,batch in enumerate(itertools.islice(itertools.cycle(batches),total)):
                if i==warmup:
                    torch.cuda.synchronize()
                    self.start=time.perf_counter()
                if i>=warmup:
                    ids=batch[2]
                    metrics['useful_tokens']+=int((ids>=0).sum())
                    metrics['loss_tokens']+=int((batch[1][:,1:]!=-100).sum())
                    for row in ids:
                        _,counts=torch.unique_consecutive(row[row>=0],return_counts=True)
                        metrics['causal_pairs']+=int((counts*(counts+1)//2).sum())
                yield batch
            torch.cuda.synchronize()
            self.seconds=time.perf_counter()-self.start
    loader=Loader()
    optimizer.zero_grad(set_to_none=True)
    trainer.train_epoch(0,loader,total+1)
    trainer.profiler.finish()
    total_params=sum(p.numel() for p in base.parameters())
    expert=config.num_hidden_layers*config.num_experts*3*config.hidden_size*config.moe_intermediate_size
    active=total_params-expert+expert*config.num_experts_per_tok/config.num_experts
    flops=6*active*metrics['useful_tokens']+12*config.num_hidden_layers*config.num_attention_heads*config.head_dim*metrics['causal_pairs']
    report=dict(reference=args.reference,steps=args.steps,checkpoint=1,metrics=metrics,
        warm_window_seconds=loader.seconds,wall_ms=1000*loader.seconds/args.steps,
        useful_tokens_per_second=metrics['useful_tokens']/loader.seconds,
        mfu_estimate_pct=100*flops/loader.seconds/(87.9e12),
        scope='production train_epoch warm packed-cache replay, excludes preprocessing and checkpoint saving',
        corpus=str(corpus.resolve()),corpus_sha256=corpus_hash,replay_sha256=data_hash.hexdigest(),
        source_rows=table.num_rows,accepted_length_rows=len(sequences),replay_batches=len(batches),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,config=raw,
        profiler_windows=trainer.profiler.windows,
        peak_allocated_gib=max(w['profile/peak_allocated_gb'] for w in trainer.profiler.windows))
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
