"""Build unused general-domain supplements from pinned coder upstream shards.

Training releases are read only by a verified exclusion index. Source collector
shards and previously unscanned raw Parquet tails supply the candidate texts.
Ambiguous identity records are retained separately, never silently rewritten.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'dataset/coder_pretrain_12b_seg2_review/_deps'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import datasets  # Windows: before transformers/torch
import scripts
scripts.__path__=[str(ROOT/'scripts'),*scripts.__path__]
import ahocorasick
import numpy as np
import orjson
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from scripts.data_builder import collect_coder_pretrain_12b as collector
from scripts.data_builder import review_ultradata_l3_identity
from scripts.data_builder.coder_identity_review import review_reasons
from scripts.data_builder.coder_pretrain_common import words
from scripts.data_builder.filter_anomaly_candidates import inspect_row
from scripts.data_builder.clean_ultradata_l3_cpt import dump
from scripts.data_builder.build_ultradata_code_l3_12b import sha256_file, identity_scan

WORK = ROOT / 'dataset/cpt_unused_general_v1'
PRIOR = ROOT / 'dataset/ultradata_code_l3_12b/quality_clean_v1'
TARGETS = {'math':300_000_000, 'natural_en':600_000_000, 'natural_zh':300_000_000}
INDEX = TOKENIZER = BENCH = None
VERSION = 'unused-general-v1'

def init():
    global INDEX, TOKENIZER, BENCH
    INDEX = np.memmap(PRIOR / 'prior.blake2b128', dtype='V16', mode='r')
    TOKENIZER = Tokenizer.from_file(str(ROOT / 'model/tokenizer.json'))
    with (PRIOR / 'benchmark_v2.pkl').open('rb') as f:
        _, strict, prompts, _ = pickle.load(f)
    BENCH = ahocorasick.Automaton()
    for i, gram in enumerate(strict | prompts):
        BENCH.add_word('\x1f'+'\x1f'.join(gram)+'\x1f',i)
    BENCH.make_automaton()

def fingerprint(text):
    return hashlib.blake2b(text.strip().encode('utf-8'),digest_size=16).digest()

def prior_contains(text):
    key = np.void(fingerprint(text)); n = int(np.searchsorted(INDEX,key))
    return n<len(INDEX) and INDEX[n]==key

def jobs():
    result=[]
    for root in [ROOT/'dataset/coder_pretrain_12b', ROOT/'dataset/coder_pretrain_12b_seg2_novel_raw', WORK/'collected']:
        for component in TARGETS:
            for report in sorted((root/component/'reports').glob('*.json')):
                r=json.loads(report.read_text(encoding='utf-8'))
                if 'upstream_file' not in r: continue
                result.append({'kind':'collector', 'component':component,'report':str(report),
                               'path':r['output'], 'sha256':r['sha256'], 'rows':r['rows'], 'upstream':r['upstream_file']})
                raw=root/component/'raw'/r['upstream_file']
                if raw.exists() and pq.ParquetFile(raw).metadata.num_rows > r['selection']['scanned']:
                    result.append({'kind':'raw_tail','component':component,'report':str(report),'path':str(raw),
                                   'start':r['selection']['scanned'], 'upstream':r['upstream_file']})
    return result

def scan(job):
    component=job['component']; source=Path(job['path']); signature=source.stat()
    key=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:20]
    out=WORK/'pool'/f'{component}_{key}.jsonl'
    report=out.with_suffix('.report.json'); candidates=out.with_suffix('.identity.jsonl')
    sig=[signature.st_size,signature.st_mtime_ns]
    if report.exists():
        r=json.loads(report.read_text(encoding='utf-8'))
        if r['source_signature']!=sig or r['version']!=VERSION: raise ValueError('Changed source/policy')
        if out.exists() and candidates.exists():return r
    counts=Counter(); keywords=Counter(); digest=hashlib.sha256(); source_hash=hashlib.sha256(); tokens=0
    pending_hash=hashlib.sha256(); seen=set(); number=0
    if job['kind']=='collector':
        def input_rows():
            with source.open('rb',buffering=1024**2) as f:
                for line in f:
                    source_hash.update(line)
                    yield orjson.loads(line)
    else:
        def input_rows():
            skipped=0
            for batch in pq.ParquetFile(source).iter_batches(batch_size=1024):
                if skipped+len(batch)<=job['start']:
                    skipped+=len(batch);continue
                rows=batch.to_pylist(); start=max(0,job['start']-skipped);skipped+=len(batch)
                for i,row in enumerate(rows[start:],start):
                    yield {**row,'_upstream_row':skipped-len(batch)+i+1}
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.jsonl.tmp')
    with tmp.open('wb',buffering=1024**2) as writer,candidates.open('wb',buffering=1024**2) as pending:
        for record in input_rows():
            number+=1; counts['scanned']+=1
            if job['kind']=='raw_tail':
                text, rejection=collector.accepted_text(component,record)
                if rejection: counts['reject:'+rejection]+=1;continue
            else: text=record['text'].strip()
            if prior_contains(text):counts['reject:previous_training_text']+=1;continue
            fp=fingerprint(text)
            if fp in seen:counts['reject:local_duplicate_text']+=1;continue
            seen.add(fp)
            if not collector.clean_text(text):counts['reject:text_quality']+=1;continue
            if next(BENCH.iter('\x1f'+'\x1f'.join(words(text))+'\x1f'),None) is not None:
                counts['reject:benchmark_distinctive_overlap']+=1;continue
            triage=inspect_row({'response':text},'identity')
            verdict,hits,vendor=identity_scan(text); reasons=review_reasons(text,component)
            keywords.update(vendor)
            if reasons or verdict!='clean':
                line=orjson.dumps({'text_id':fp.hex(),'text':text,'component':component,'source':str(source),
                                   'source_row':number,'triage':triage,'identity_verdict':verdict,
                                   'identity_hits':hits,'contextual_reasons':reasons,'vendor_keywords':dict(vendor),
                                   'decision':'quarantine_pending_contextual_review'})+b'\n'
                pending.write(line);pending_hash.update(line);counts['identity_quarantined']+=1;continue
            # Fresh shards were just fully tokenized by this run's pinned
            # collector. Historical collectors and untouched tails are
            # independently retokenized here; fresh outputs get sampled checks.
            fresh=job['kind']=='collector' and source.resolve().is_relative_to((WORK/'collected').resolve())
            count=record['token_count'] if fresh else len(TOKENIZER.encode(text,add_special_tokens=False).ids)+2
            if fresh and number%1000==1 and len(TOKENIZER.encode(text,add_special_tokens=False).ids)+2!=count:
                raise ValueError('Fresh collector token count mismatch')
            if not 64<=count<=4096:counts['reject:token_length']+=1;continue
            if job['kind']=='collector' and count!=record['token_count']:raise ValueError('Stored token count mismatch')
            if job['kind']=='raw_tail':
                row=collector.record_for(component,record,text,count,job['upstream'],record['_upstream_row']-1)
                row['upstream_row']=record['_upstream_row']
            else: row={**record,'text':text}
            row.update(text_id=fp.hex(),mix_category=component,source_revision=collector.SOURCES[component]['revision'],
                       token_count=count,novel_source_kind=job['kind'],collector_source=str(source),collector_source_row=number,
                       identity_decision='factual_mentions_preserved_no_persona_or_model_card_pattern')
            line=orjson.dumps(row)+b'\n'; writer.write(line);digest.update(line);tokens+=count;counts['kept']+=1
        writer.flush();os.fsync(writer.fileno())
    if job['kind']=='collector' and (source_hash.hexdigest()!=job['sha256'] or number!=job['rows']):
        raise ValueError('Collector shard failed hash/rows verification')
    if [source.stat().st_size,source.stat().st_mtime_ns]!=sig:raise ValueError('Source changed during scan')
    os.replace(tmp,out)
    r={'version':VERSION,'job':job,'source_signature':sig,'source_sha256':source_hash.hexdigest() if job['kind']=='collector' else sha256_file(source),
       'output':str(out),'output_sha256':digest.hexdigest(),'counts':dict(counts),'tokens':tokens,
       'identity_candidates':str(candidates),'identity_candidates_sha256':pending_hash.hexdigest(),'keywords':dict(keywords)}
    dump(report,r);return r

def run(workers):
    WORK.mkdir(parents=True,exist_ok=True)
    tokenizer_hash=sha256_file(ROOT/'model/tokenizer.json')
    l3_report=json.loads((ROOT/'dataset/pretrain_ultradata_code_l3_cpt_12b_identity_clean.report.json').read_text(encoding='utf-8'))
    if tokenizer_hash!=l3_report['tokenizer_sha256']:raise ValueError('Tokenizer changed from reviewed L3 corpus')
    prior=json.loads((PRIOR/'prior.report.json').read_text(encoding='utf-8'))
    for item in prior['manifest']:
        stat=Path(item['path']).stat()
        if [stat.st_size,stat.st_mtime_ns]!=[item['bytes'],item['mtime_ns']]:raise ValueError('Prior training source changed')
    if sha256_file(PRIOR/'prior.blake2b128')!=prior['sha256']:raise ValueError('Exclusion index changed')
    reports=[];total=Counter();counts=Counter(); tasks=jobs()
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for future in as_completed([pool.submit(scan,job) for job in tasks]):
            r=future.result();reports.append(r);total[r['job']['component']]+=r['tokens'];counts.update(r['counts'])
            state={'phase':'novel_source_scan','completed':len(reports),'total_jobs':len(tasks),'tokens':dict(total),'counts':dict(counts)}
            dump(WORK/'status.json',state);print(json.dumps(state),flush=True)
    dump(WORK/'pool.report.json',{'reports':sorted(reports,key=lambda r:r['output']),'tokens':dict(total),'counts':dict(counts),
                                'targets':TARGETS,'prior_exclusion':prior,'tokenizer_sha256':tokenizer_hash,
                                'tokenization':'All novel texts tokenized with current tokenizer during collection or cleaning; historical shards/tails retokenized completely.',
                                'training_ready':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=16)
    run(parser.parse_args().workers)
