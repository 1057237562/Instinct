"""Keep one mixed training file; audit length distribution and supervision."""
import gzip
import json
from collections import Counter
from pathlib import Path
import random
import sys

import datasets  # Windows import order
import orjson
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.data_builder.build_v1moe_sft64k import OUT, emit, dump, sha256_file
from scripts.data_loader.lm_dataset import _create_chat_prompt, _generate_sft_labels


def main():
    verified=orjson.loads((OUT/'verification.report.json').read_bytes())
    assert verified['passed']
    target=ROOT/'dataset/sft_v1moe_balanced_64k_identity_clean.jsonl'
    assert sha256_file(target)==verified['train_sha256']
    report=orjson.loads((OUT/'build.report.json').read_bytes())
    names=['short_4k','medium_4k_16k','long_16k_64k']
    counts=Counter(); tokens=Counter(); samples={}; turns=Counter(); supervised=Counter()
    rng=random.Random(27)
    t=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    bos=t(f'{t.bos_token}assistant\n',add_special_tokens=False).input_ids
    eos=t(f'{t.eos_token}\n',add_special_tokens=False).input_ids
    with target.open('rb') as f,gzip.open(OUT/'train.provenance.jsonl.gz','rb') as pf:
        for line,pline in zip(f,pf,strict=True):
            row=orjson.loads(line); meta=orjson.loads(pline); n=row['token_count']
            bucket=names[0] if n<=4096 else names[1] if n<=16384 else names[2]
            counts[bucket]+=1; tokens[bucket]+=n
            turns[str(sum(m['role']=='user' for m in row['conversations']))]+=1
            source=meta['_source']
            group=source+':'+bucket
            samples.setdefault(group,[])
            if len(samples[group])<2:
                samples[group].append({'row':row,'provenance':meta})
            if source=='synthetic_multiturn_ledger' or rng.random()<0.004:
                text=_create_chat_prompt(t,row['conversations'])
                ids=t(text).input_ids
                assert len(ids)==n
                labels=_generate_sft_labels(ids,bos,eos,65536)
                active=sum(x!=-100 for x in labels)
                assert active>0 and active<n
                supervised['checked_rows']+=1; supervised['active_assistant_tokens']+=active
    dump(OUT/'package.report.json',{'layout':'single_mixed_training_file','length_distribution':{n:{'rows':counts[n],'tokens':tokens[n]} for n in names},'user_turn_histogram':dict(turns),'supervision_check':dict(supervised)})
    dump(Path(str(target)+'.report.json'),{'build':report,'verification':verified,'package':orjson.loads((OUT/'package.report.json').read_bytes())})
    with (OUT/'review_samples.jsonl').open('wb') as f:
        for group,entries in samples.items():
            for entry in entries: emit(f,dict(entry,group=group))
    print(json.dumps({'counts':dict(counts),'tokens':dict(tokens),'supervision':dict(supervised),'turns':dict(turns)},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
