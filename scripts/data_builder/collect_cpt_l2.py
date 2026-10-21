"""Pinned, resumable L2 acquisition and conservative source-code cleaning."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import time
import warnings
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'dataset/coder_pretrain_12b_seg2_review/_deps'))
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('HF_FALLBACK_ENDPOINT','https://huggingface.co')
import datasets  # Windows ordering
import scripts
scripts.__path__=[str(ROOT/'scripts'),*scripts.__path__]
import ahocorasick
import numpy as np
import orjson
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from scripts.data_builder.build_cpt_l2_l3_mix import WORK, PRIOR, L2_QUOTAS, SEED, dump, read, sha, signature, fingerprint, uid, contains
from scripts.data_builder import cpt_identity_runtime as l3
from scripts.data_builder.coder_identity_review import review_reasons
from scripts.data_builder.filter_anomaly_candidates import inspect_row
from scripts.data_builder.coder_pretrain_common import words

REPO='openbmb/UltraData-Code'
REVISION='525c9ac8639c5032633392d096e54db8c84a005e'
RAW=WORK/'raw';VERSION='l2-native-v2-language-release-policy'
OLD=BASE=UUIDS=TOK=BENCH=NAMES=None
l3.VENDOR_RE=re.compile(l3.VENDOR_COUNT_RE.pattern+r'|(?<![a-z0-9_])(?:qwen[a-z0-9.\-]*|minicpm[a-z0-9.\-]*)(?![a-z0-9_])',re.I)

def catalog():
    path=WORK/'l2_catalog.json'
    if path.exists():return read(path)
    def fetch(lang):
        directory='rust' if lang=='rs' else lang
        url=f'https://hf-mirror.com/api/datasets/{REPO}/tree/{REVISION}/data/UltraData-Code-L2/{directory}?limit=1000'
        with urlopen(Request(url,headers={'User-Agent':'Instinct-CPT-L2/1'}),timeout=60) as r:
            rows=json.load(r)
            if r.headers.get('Link') and 'rel="next"' in r.headers['Link']:raise ValueError('Paginated catalog must be exhausted')
        files=[{'path':x['path'],'size':x['size'],'sha256':x['lfs']['oid'],'language':lang}
               for x in rows if x.get('type')=='file' and x['path'].endswith('.parquet')]
        files.sort(key=lambda x:hashlib.sha256((str(SEED)+x['path']).encode()).digest())
        print('[catalog]',lang,len(files),sum(x['size'] for x in files),flush=True)
        return lang,files
    with ThreadPoolExecutor(max_workers=4) as pool:r=dict(pool.map(fetch,L2_QUOTAS))
    result={'repo':REPO,'revision':REVISION,'seed':SEED,'files':r};dump(path,result);return result

def plan():
    p=WORK/'l2_plan.json'
    if p.exists():return read(p)
    c=catalog();r={'repo':REPO,'revision':REVISION,'policy':VERSION,'shards':[v[0] for v in c['files'].values()]}
    dump(p,r);return r

def download(item):
    from scripts.data_builder.collect_coder_pretrain_12b import direct_dataset_download
    path=RAW/item['path']; stamp=Path(str(path)+'.download.json')
    if path.exists() and stamp.exists():
        s=read(stamp)
        if s['signature']==signature(path) and s['sha256']==item['sha256']:return str(path)
    for attempt in range(10):
        try:
            path=direct_dataset_download(REPO,REVISION,item['path'],RAW,120)
            if path.stat().st_size!=item['size'] or sha(path)!=item['sha256']:raise ValueError('Upstream size/SHA mismatch: '+str(path))
            metadata=pq.ParquetFile(path).metadata
            dump(stamp,{'source_repository':REPO,'source_revision':REVISION,'source_path':item['path'],
                        'sha256':item['sha256'],'signature':signature(path),'rows':metadata.num_rows,'schema':str(pq.ParquetFile(path).schema_arrow)})
            print('[download-ready]',item['language'],path.name,metadata.num_rows,flush=True)
            return str(path)
        except (OSError,EOFError,TimeoutError) as error:
            print('[download-retry]',attempt+1,item['path'],str(error)[:250],flush=True)
            if attempt==9:raise
            time.sleep(min(10,attempt+1))

def downloads(workers):
    items=plan()['shards']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(download,item) for item in items]):f.result()

def initialize():
    global OLD,BASE,UUIDS,TOK,BENCH,NAMES
    warnings.filterwarnings('ignore',category=SyntaxWarning)
    OLD=np.memmap(PRIOR/'prior.blake2b128',dtype='V16',mode='r')
    BASE=np.memmap(WORK/'base_texts.blake2b128',dtype='V16',mode='r')
    UUIDS=np.memmap(WORK/'base_l3.uuid128',dtype='V16',mode='r')
    TOK=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
    with (PRIOR/'benchmark_v2.pkl').open('rb') as f:_,strict,prompts,NAMES=pickle.load(f)
    BENCH=ahocorasick.Automaton()
    for i,gram in enumerate(strict|prompts):BENCH.add_word('\x1f'+'\x1f'.join(gram)+'\x1f',i)
    BENCH.make_automaton()

def quality_reason(row):
    text=row.get('content')
    if not isinstance(text,str) or not text.strip():return 'empty_content'
    if not all(isinstance(row.get(k),str) and row[k] for k in ('uuid','repo_name','relative_path')):return 'missing_provenance'
    try:
        uid(row['uuid']); relevance=float(row['algo_rel_score']);quality=float(row['quality_score'])
    except (ValueError,TypeError,KeyError):return 'invalid_source_scores_or_uuid'
    if not math.isfinite(relevance) or not math.isfinite(quality):return 'nonfinite_score'
    # The published L2 corpus uses language-specific policies. Reapplying the
    # scorer package's global default 0.8 drops 36.7% of the Python pilot and
    # 20.9% of Java. Trust upstream relevance selection and retain its score.
    if not 0<=relevance<=1 or not 0<=quality<=10:return 'invalid_score_range'
    if row.get('category')=='EXCLUDE' or quality<3:return 'upstream_quality_gate'
    if '\x00' in text or '\ufffd' in text:return 'invalid_encoding'
    if (re.search(r'^<<<<<<< .+$',text,re.M) and re.search(r'^=======$',text,re.M)
            and re.search(r'^>>>>>>> .+$',text,re.M)):return 'unresolved_merge_conflict'
    if '<html' in text[:500].lower() and '</html>' in text[-1000:].lower():return 'html_in_code'
    return None

def scan(item):
    path=RAW/item['path'];lang=item['language'];group=item.get('row_group')
    if group is None and item.get('row_groups'):
        raise ValueError('This source is planned as row-group jobs; use scan or scan-groups to resume it')
    stem=path.stem+(f'.rg{group:04d}' if group is not None else '')
    out=WORK/'l2_pool'/(stem+'.identity_clean.jsonl')
    report=out.with_suffix('.report.json');candidates=out.with_suffix('.candidates.jsonl');ledger=out.with_suffix('.decisions.jsonl')
    base=read(WORK/'base.identity_clean.report.json')
    if report.exists():
        r=read(report)
        if r['version']!=VERSION or r['source_signature']!=signature(path) or r['base_sha256']!=base['output_sha256']:raise ValueError('Stale L2 cache')
        if out.exists() and candidates.exists() and ledger.exists():return r
    if path.stat().st_size!=item['size'] or sha(path)!=item['sha256']:raise ValueError('L2 raw checksum mismatch')
    out.parent.mkdir(parents=True,exist_ok=True);temp=Path(str(out)+'.tmp')
    counts=Counter();keywords=Counter();digest=hashlib.sha256();pending_digest=hashlib.sha256();ledger_digest=hashlib.sha256()
    parquet=pq.ParquetFile(path)
    row_offset=sum(parquet.metadata.row_group(i).num_rows for i in range(group)) if group is not None else 0
    seen=set();number=row_offset;start=time.monotonic()
    with temp.open('wb',buffering=4*1024**2) as w,candidates.open('wb',buffering=1024**2) as pending,ledger.open('wb',buffering=1024**2) as decisions:
        for batch in parquet.iter_batches(batch_size=512,row_groups=[group] if group is not None else None):
            for row in batch.to_pylist():
                number+=1;counts['input_rows']+=1
                reason=quality_reason(row)
                if reason:counts['drop:'+reason]+=1;continue
                text=row['content'];fp=fingerprint(text)
                if contains(OLD,fp):counts['drop:previous_training_exact_text']+=1;continue
                if contains(BASE,fp):counts['drop:current_base_exact_text']+=1;continue
                if contains(UUIDS,uid(row['uuid'])):counts['drop:selected_l3_same_source_uuid']+=1;continue
                if fp in seen:counts['drop:within_shard_exact_duplicate']+=1;continue
                seen.add(fp)
                # Separate resource/size gate, not an asserted token-length bound.
                if len(text.encode('utf-8'))>200000:counts['drop:very_large_file']+=1;continue
                tokens=len(TOK.encode(text,add_special_tokens=False).ids)+2
                if tokens<64:counts['drop:below_64_tokens']+=1;continue
                if tokens>4096:counts['drop:above_4096_tokens']+=1;continue
                word_list=words(text)
                if next(BENCH.iter('\x1f'+'\x1f'.join(word_list)+'\x1f'),None) is not None or any(x in NAMES for x in word_list):
                    counts['drop:distinctive_benchmark_overlap']+=1;continue
                triage=inspect_row({'text':text},'all');verdict,hits,vendor=l3.identity_scan(text);reasons=review_reasons(text,'code')
                keywords.update(vendor)
                record={'text':text,'token_count':tokens,'source':'ultradata_code_l2:'+lang,'mix_category':lang,
                        'mix_component':'L2:'+lang,'data_tier':'L2','mixture':'l2-30-l3-60-general-10',
                        'license':'upstream-repository-license; UltraData release Apache-2.0',
                        'uuid':row['uuid'],'repo_name':row['repo_name'],'relative_path':row['relative_path'],
                        'category':row['category'],'algo_rel_score':row['algo_rel_score'],'quality_score':row['quality_score'],
                        'source_shard':item['path'],'source_row':number,'source_repository':REPO,'source_revision':REVISION,
                        'text_id':fp.hex(),'identity_decision':'no_foreign_persona_after_contextual_screen'}
                algorithm_use=bool(vendor) and set(vendor)<={'minimax','glm'} and bool(re.search(r'algorithm|alpha.beta|game|spanning.tree|glm::|vec[234]|matrix|regression',text,re.I))
                requires_review=bool(reasons) or (verdict!='clean' and not(verdict=='review' and algorithm_use))
                if triage or vendor or hits or reasons:
                    payload={'record':record,'triage':triage,'identity_verdict':verdict,'identity_hits':hits,
                             'vendor_keywords':dict(vendor),'contextual_reasons':reasons,'requires_contextual_review':requires_review}
                    b=orjson.dumps(payload)+b'\n';pending.write(b);pending_digest.update(b);counts['triage_candidates']+=1
                    if requires_review:counts['identity_quarantined_rows']+=1;counts['identity_quarantined_tokens']+=tokens;continue
                    b=orjson.dumps({'text_id':fp.hex(),'decision':'keep','reason':'Preserve factual code, algorithm names and ordinary TODO/interface markers; bounded scan has no foreign assistant persona.',
                                    'reviewer':'factual_code_preservation_policy_v1','source_shard':item['path'],'source_row':number})+b'\n'
                    decisions.write(b);ledger_digest.update(b)
                if lang=='py':
                    try:ast.parse(text);counts['python3_parse_success']+=1
                    except (SyntaxError,ValueError,RecursionError):counts['python3_parse_failure_retained_for_source_diversity']+=1
                b=orjson.dumps(record)+b'\n';w.write(b);digest.update(b);counts['kept_rows']+=1;counts['kept_tokens']+=tokens
            if number%50000<len(batch):
                dump(out.with_suffix('.status.json'),{'rows':number,'counts':dict(counts),'seconds':time.monotonic()-start})
                print('[l2-scan]',lang,path.stem,number,counts['kept_tokens'],flush=True)
        w.flush();os.fsync(w.fileno())
    os.replace(temp,out)
    r={'version':VERSION,'source_repository':REPO,'source_revision':REVISION,'source_path':item['path'],
       'source_signature':signature(path),'source_sha256':item['sha256'],'language':lang,
       'output':str(out),'output_sha256':digest.hexdigest(),'candidates':str(candidates),'candidates_sha256':pending_digest.hexdigest(),
       'decisions':str(ledger),'decisions_sha256':ledger_digest.hexdigest(),'counts':dict(counts),'identity_keyword_hits':dict(keywords),
       'base_sha256':base['output_sha256'],'tokenizer_sha256':base['tokenizer_sha256'],'seconds':time.monotonic()-start}
    r.update(row_group=group,source_row_offset=row_offset)
    dump(report,r);print('[l2-scan-complete]',lang,dict(counts),flush=True);return r

def scan_groups(index,workers):
    current=plan();item=current['shards'][index];path=RAW/item['path']
    if sha(path)!=item['sha256']:raise ValueError('Raw group source changed')
    total=pq.ParquetFile(path).metadata.num_row_groups
    full_report=WORK/'l2_pool'/(path.stem+'.identity_clean.report.json')
    if full_report.exists():raise ValueError('Whole-source report already exists; do not duplicate it with row-group reports')
    item['row_groups']=total;dump(WORK/'l2_plan.json',current)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        jobs=[pool.submit(scan,{**item,'row_group':i}) for i in range(total)]
        for n,f in enumerate(as_completed(jobs),1):
            f.result();print('[row-groups]',n,'/',total,flush=True)

def scans(workers, ready_only=False, languages=None, start_index=0):
    b=read(WORK/'base.identity_clean.report.json');old=read(PRIOR/'prior.report.json')
    if sha(ROOT/'model/tokenizer.json')!=b['tokenizer_sha256'] or sha(PRIOR/'prior.blake2b128')!=old['sha256']:raise ValueError('Changed tokenizer/prior index')
    for item in old['manifest']:
        if signature(item['path'])!=[item['bytes'],item['mtime_ns']]:raise ValueError('Prior source changed')
    for item in b['indexes'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Base index changed')
    results=[];items=plan()['shards'][start_index:]
    if languages:items=[x for x in items if x['language'] in languages]
    if ready_only:items=[x for x in items if Path(str(RAW/x['path'])+'.download.json').exists()]
    items=[job for item in items for job in (
        [{**item,'row_group':i} for i in range(item['row_groups'])] if item.get('row_groups') else [item])]
    if not items:raise ValueError('No verified raw shards ready')
    dump(WORK/f'l2_scan_dispatch_{os.getpid()}.json',{'pid':os.getpid(),'shards':[x['path'] for x in items]})
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        for f in as_completed([pool.submit(scan,item) for item in items]):
            results.append(f.result());totals=Counter()
            for r in results:totals[r['language']]+=r['counts'].get('kept_tokens',0)
            dump(WORK/'l2_scan.status.json',{'completed':len(results),'planned':len(plan()['shards']),'tokens':dict(totals)})
    results=[read(p) for p in sorted((WORK/'l2_pool').glob('*.identity_clean.report.json'))]
    dump(WORK/'l2_pool.report.json',{'reports':results,'version':VERSION,'source_repository':REPO,'source_revision':REVISION})

def extend():
    c=catalog();p=plan();used={i['path'] for i in p['shards']};totals=Counter();size=Counter()
    sized=set()
    for r in read(WORK/'l2_pool.report.json')['reports']:
        totals[r['language']]+=r['counts'].get('kept_tokens',0)
        if r['source_path'] not in sized:
            size[r['language']]+=r['source_signature'][0];sized.add(r['source_path'])
    added=[]
    for lang,quota in L2_QUOTAS.items():
        # Headroom for exact deduplication across all selected shards.
        if totals[lang]>=quota*1.12:continue
        left=[x for x in c['files'][lang] if x['path'] not in used]
        ratio=totals[lang]/max(size[lang],1)
        need=quota*1.2-totals[lang]
        for item in left:
            added.append(item);need-=ratio*item['size']
            if need<=0:break
        if need>0 and not left and totals[lang]<quota:raise ValueError('L2 source exhausted: '+lang)
    p['shards'].extend(added);dump(WORK/'l2_plan.json',p)
    print('[plan-extended]',len(added),'new shards; tokens',dict(totals),flush=True)
    return len(added)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['download','scan','scan-one','scan-groups','extend','catalog'])
    p.add_argument('--workers',type=int,default=4);p.add_argument('--ready-only',action='store_true');p.add_argument('--languages',nargs='+');p.add_argument('--start-index',type=int,default=0);a=p.parse_args()
    if a.command=='download':downloads(a.workers)
    elif a.command=='scan':scans(a.workers,a.ready_only,a.languages,a.start_index)
    elif a.command=='scan-one':
        initialize();scan(plan()['shards'][a.start_index])
    elif a.command=='scan-groups':scan_groups(a.start_index,a.workers)
    elif a.command=='extend':extend()
    else:catalog()
