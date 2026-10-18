"""Collect three previously unconsumed English source shards for CPT headroom."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('HF_FALLBACK_ENDPOINT','https://huggingface.co')
import datasets  # Windows import ordering
import scripts
scripts.__path__=[str(ROOT/'scripts'),*scripts.__path__]
from scripts.data_builder import collect_coder_pretrain_12b as c
from scripts.data_builder.coder_novelty import ExclusionIndex, IdentityReview
from scripts.data_builder.build_ultradata_code_l3_12b import sha256_file
from tokenizers import Tokenizer

WORK=ROOT/'dataset/cpt_unused_general_v1'
PRIOR=ROOT/'dataset/ultradata_code_l3_12b/quality_clean_v1'

def collect(job):
    component,name=job if isinstance(job,tuple) else ('natural_en',job)
    c.ACTIVE_OUT=WORK/'collected'
    c.EXCLUSION_INDEX=ExclusionIndex(WORK/'prior.blake2b128')
    c.IDENTITY_REVIEW=IdentityReview()
    tokenizer=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
    for attempt in range(5):
        try:return c.collect_shard(component,name,tokenizer,400_000_000,180)
        except (OSError,EOFError,TimeoutError):
            if attempt==4:raise

def main(component='natural_en'):
    WORK.mkdir(parents=True,exist_ok=True)
    path=WORK/'prior.blake2b128'; report=path.with_suffix(path.suffix+'.report.json')
    old=json.loads((PRIOR/'prior.report.json').read_text(encoding='utf-8'))
    if not report.exists():
        shutil.copyfile(PRIOR/'prior.blake2b128',path)
        sources=[{**sig,**item} for sig,item in zip(old['manifest'],old['sources'])]
        value={'format':'sorted-blake2b-128-v2','normalization':'outer whitespace stripped',
               'sources':sources,'source_sha256':hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
               'fingerprints':old['fingerprints'],'index_bytes':path.stat().st_size,'index_sha256':old['sha256']}
        if sha256_file(path)!=old['sha256']:raise ValueError('Copied index mismatch')
        report.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    c.ACTIVE_OUT=WORK/'collected'
    cache=ROOT/f'dataset/coder_pretrain_12b_seg2_novel_raw/{component}/upstream_files.json'
    dest=c.ACTIVE_OUT/component/'upstream_files.json';dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copyfile(cache,dest)
    used=set()
    for root in [ROOT/'dataset/coder_pretrain_12b',ROOT/'dataset/coder_pretrain_12b_seg2_novel_raw']:
        for p in (root/component/'reports').glob('*.json'):
            r=json.loads(p.read_text(encoding='utf-8'))
            if 'upstream_file' in r:used.add(r['upstream_file'])
    names=[name for name in c.list_source_files(component) if name not in used][:3 if component=='natural_en' else 1]
    (WORK/f'new_shard_plan_{component}.json').write_text(json.dumps({'pinned_source':c.SOURCES[component],'unused_upstream_shards':names,'previously_collected_shards':sorted(used)},indent=2)+'\n',encoding='utf-8')
    print('New upstream shards:',names,flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        for r in pool.map(collect,[(component,name) for name in names]):print(r['upstream_file'],r['tokens'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--component',choices=['natural_en','natural_zh','math'],default='natural_en')
    main(p.parse_args().component)
