"""Select and audit a 12B L3 CPT corpus with 10% unused general-source data."""
from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['TOKENIZERS_PARALLELISM']='false'
import datasets  # Windows: initialize datasets before transformer helpers.
import scripts
# pywin32 can pre-load its own namespace named scripts during interpreter
# startup. Bind this entry point's imports to the repository as well.
scripts.__path__=[str(ROOT/'scripts'),*scripts.__path__]
import numpy as np
import orjson
from tokenizers import Tokenizer
from scripts.data_builder.mix_sft_datasets import ExternalRandomShuffler
from scripts.data_builder.build_cpt_unused_general import WORK, PRIOR, TARGETS, fingerprint
from scripts.data_builder.build_ultradata_code_l3_12b import sha256_file
from scripts.data_builder.clean_ultradata_l3_cpt import dump

L3=ROOT/'dataset/pretrain_ultradata_code_l3_cpt_12b_identity_clean.jsonl'
OUTPUT=ROOT/'dataset/pretrain_ultradata_l3_cpt_12b_general10_identity_clean.jsonl'
REPORT=OUTPUT.with_suffix('.report.json')
DTYPE=np.dtype([('priority','V16'),('file','u4'),('offset','u8'),('tokens','u4')])
SEED=2026101501

class BufferedShuffler(ExternalRandomShuffler):
    @staticmethod
    def _iter_chunk(path):
        with path.open('rb',buffering=1024**2) as f:
            for line in f:
                key,row=line.rstrip(b'\n').split(b'\t',1)
                yield key,row

def assemble():
    if OUTPUT.exists() or REPORT.exists():raise FileExistsError('Release already exists')
    pool=json.loads((WORK/'pool.reviewed.report.json').read_text(encoding='utf-8'))
    l3=json.loads(L3.with_suffix('.report.json').read_text(encoding='utf-8'))
    quotas={**{k:int(10_800_000_000*v/l3['target_tokens']) for k,v in l3['language_quotas'].items()},**TARGETS}
    files=[{'output':str(L3),'output_sha256':l3['output_sha256']}]+pool['reports']
    seen=set();parts={k:[] for k in quotas};buffers={k:[] for k in quotas};available=Counter();rejections=Counter()
    seed=struct.pack('>Q',SEED)
    for number,info in enumerate(files):
        path=Path(info['output']); offset=0;digest=hashlib.sha256();sig=path.stat()
        info['signature']=[sig.st_size,sig.st_mtime_ns]
        with path.open('rb',buffering=1024**2) as f:
            for line in f:
                digest.update(line);row=orjson.loads(line);fp=bytes.fromhex(row['text_id']);k=row['mix_category'];count=row['token_count']
                if fingerprint(row['text'])!=fp:raise ValueError('Candidate fingerprint mismatch')
                if fp in seen:rejections[k+':exact_duplicate']+=1;offset+=len(line);continue
                seen.add(fp);priority=hashlib.blake2b(seed+fp,digest_size=16).digest()
                buffers[k].append((priority,number,offset,count));available[k]+=count;offset+=len(line)
                if len(buffers[k])>=100_000:parts[k].append(np.array(buffers[k],dtype=DTYPE));buffers[k].clear()
        if digest.hexdigest()!=info['output_sha256']:raise ValueError('Input hash changed: '+str(path))
        print('[mix-index]',number+1,'/',len(files),dict(available),flush=True)
    del seen
    for k,quota in quotas.items():
        if available[k]<quota:raise ValueError(f'Insufficient unused {k}: {available[k]:,} < {quota:,}')
    selected={i:[] for i in range(len(files))};totals=Counter();rows=Counter()
    for k in quotas:
        if buffers[k]:parts[k].append(np.array(buffers[k],dtype=DTYPE));buffers[k].clear()
        index=np.concatenate(parts[k]);parts[k].clear();index.sort(order='priority')
        for entry in index:
            count=int(entry['tokens'])
            if totals[k]+count>quotas[k]:continue
            selected[int(entry['file'])].append(int(entry['offset']));totals[k]+=count;rows[k]+=1
        if quotas[k]-totals[k]>4096:raise ValueError('Quota not filled')
        print('[mix-selection]',k,rows[k],totals[k],flush=True)
        del index
    shuffler=BufferedShuffler(OUTPUT,SEED+1,80_000,chunk_bytes=512*1024**2,deduplicate=False)
    try:
        for number,info in enumerate(files):
            path=Path(info['output']);sig=path.stat()
            if [sig.st_size,sig.st_mtime_ns]!=info['signature']:raise ValueError('Indexed input changed')
            with path.open('rb',buffering=1024**2) as f:
                for position in sorted(selected.pop(number)):
                    f.seek(position);shuffler.add_serialized(f.readline())
            print('[mix-shuffle-feed]',number+1,'/',len(files),shuffler.rows_added,flush=True)
        result=shuffler.finish(progress_label='Unused general CPT shuffle')
    except BaseException:
        shuffler.temporary_dir.cleanup();raise
    report={'name':'InstinctV1Moe L3 Coding CPT with unused general-source supplements',
            'output':str(OUTPUT),'output_sha256':result['sha256'],'rows':result['rows'],'tokens':sum(totals.values()),
            'target_tokens':12_000_000_000,'component_tokens':dict(totals),'component_rows':dict(rows),'quotas':quotas,
            'code_tokens':sum(totals[k] for k in l3['language_quotas']),'general_tokens':sum(totals[k] for k in TARGETS),
            'previous_training_replay_tokens':0,'text_truncated':False,'rewritten_text_rows':0,'max_tokens':4096,
            'tokenizer_sha256':sha256_file(ROOT/'model/tokenizer.json'),'token_count_policy':l3['token_count_policy'],
            'unused_source_policy':'Only unselected upstream collector rows, unscanned raw Parquet tails, and newly downloaded source shards; previous coder training corpora read only through exclusion index.',
            'previous_training_exclusion':pool['prior_exclusion'],'unused_source_pool':pool,
            'l3_clean_source_report':str(L3.with_suffix('.report.json')),'l3_clean_source_sha256':l3['output_sha256'],
            'l3_quality_and_bias_review':l3['bias_review'],'length_filter_audit':l3['length_filter_audit'],
            'identity_clean':{'target_identity':'InstinctV1Moe','l3_identity_audit':l3['identity_clean'],
                              'new_general_identity_review':pool['identity_review'],'unreviewed_selected_records':0},
            'benchmark_screening':l3['benchmark_screening'],
            'selection':{'seed':SEED,'method':'Random-priority BLAKE2b(seed||text_id) within every token-budget component',
                         'available_tokens_after_global_dedup':dict(available),'rejections':dict(rejections)},
            'inputs':files,'shuffle':result,'training_ready':False,
            'limitations':['Unused means no exact full-text match after stripping outer whitespace against the two previous coder releases and two used arXiv corpora; semantic novelty is not established.',
                          'Ambiguous identity candidates are preserved outside the training pool; factual third-party mentions without self-identification are retained.',
                          'Heuristic quality/lexical benchmark screening cannot prove every solution correct or eliminate paraphrased contamination.',
                          'The 10% general-source mixture is an engineering starting point, not an evaluation-proven optimum.']}
    dump(REPORT,report);print('[mix-release]',report['rows'],report['tokens'],flush=True)

def audit():
    report=json.loads(REPORT.read_text(encoding='utf-8'))
    old=report['previous_training_exclusion']
    if sha256_file(PRIOR/'prior.blake2b128')!=old['sha256']:raise ValueError('Prior index changed')
    for item in old['manifest']:
        sig=Path(item['path']).stat()
        if [sig.st_size,sig.st_mtime_ns]!=[item['bytes'],item['mtime_ns']]:raise ValueError('Previous source changed')
    if sha256_file(ROOT/'model/tokenizer.json')!=report['tokenizer_sha256']:raise ValueError('Tokenizer changed')
    tokenizer=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
    index=np.memmap(PRIOR/'prior.blake2b128',dtype='V16',mode='r');seen=set();totals=Counter();rows=Counter();digest=hashlib.sha256()
    n=samples=0;minimum=4096;maximum=0;windows=[];window=Counter();window_tokens=0
    with OUTPUT.open('rb',buffering=1024**2) as f:
        for line in f:
            n+=1;digest.update(line);row=orjson.loads(line);text=row['text'];count=row['token_count'];k=row['mix_category']
            if not isinstance(text,str) or not text.strip() or type(count)!=int or not 64<=count<=4096:raise ValueError('Invalid schema/length')
            fp=fingerprint(text)
            if fp.hex()!=row['text_id'] or fp in seen:raise ValueError('Fingerprint/uniqueness error')
            seen.add(fp);value=np.void(fp);position=int(np.searchsorted(index,value))
            if position<len(index) and index[position]==value:raise ValueError('Previously used full text in final CPT')
            if k in TARGETS:
                if not row.get('collector_source') or not row.get('upstream_shard') or not row.get('source_revision'):raise ValueError('Missing unused-source provenance')
                if row.get('identity_decision')!='factual_mentions_preserved_no_persona_or_model_card_pattern':raise ValueError('Unreviewed identity record')
            elif not row.get('source_shard') or not row.get('source_row'):raise ValueError('Missing L3 provenance')
            if n%1000==1:
                if len(tokenizer.encode(text,add_special_tokens=False).ids)+2!=count:raise ValueError('Retokenization mismatch')
                samples+=1
            totals[k]+=count;rows[k]+=1;minimum=min(minimum,count);maximum=max(maximum,count)
            window['general' if k in TARGETS else 'code']+=count;window_tokens+=count
            if window_tokens>=100_000_000:
                windows.append(dict(window));window.clear();window_tokens=0
            if n%500_000==0:
                print('[mix-audit]',n,flush=True)
                dump(WORK/'audit.status.json',{'checked_rows':n,'total_rows':report['rows'],
                                              'tokens_checked':sum(totals.values()),'sampled_retokenized_rows':samples})
    if window_tokens:windows.append(dict(window))
    if digest.hexdigest()!=report['output_sha256'] or n!=report['rows'] or dict(totals)!=report['component_tokens']:raise ValueError('Release totals/hash mismatch')
    audit={'passed':True,'rows':n,'tokens':sum(totals.values()),'output_sha256':digest.hexdigest(),
           'exact_prior_training_overlaps':0,'exact_duplicates':0,'token_count_samples':samples,'token_count_sample_mismatches':0,
           'supplement_token_counts':'All candidates tokenized during current source collection or cleaning; historical collector rows and raw tails fully retokenized.',
           'min_tokens':minimum,'max_tokens':maximum,'component_rows':dict(rows),'component_tokens':dict(totals),
           'contiguous_100m_token_windows':windows,
           'general_token_fraction_range': [min(w.get('general',0)/sum(w.values()) for w in windows),max(w.get('general',0)/sum(w.values()) for w in windows)],
           'scope':'All rows: hash/schema/provenance/full-text uniqueness/previous-training exclusion and component totals. Deterministic every-1000th row tokenizer cross-check.'}
    dump(OUTPUT.with_suffix('.verification.json'),audit)
    report.update(training_ready=True,verification=str(OUTPUT.with_suffix('.verification.json')))
    dump(REPORT,report);print('[mix-audit] PASSED',n,sum(totals.values()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['assemble','audit'])
    assemble() if p.parse_args().command=='assemble' else audit()
