"""Build a separate 12B CPT release: L3 60%, L2 30%, unused general 10%.

Existing reviewed releases are immutable inputs. Selection uses deterministic
random priorities over whole records; token budgets include BOS/EOS. No training
or execution of untrusted source code is performed by this builder.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
import re
from pathlib import Path
import struct
import sys
import uuid
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import datasets  # Windows: before transformer/torch helpers.
import scripts
scripts.__path__ = [str(ROOT / 'scripts'), *scripts.__path__]
import numpy as np
import orjson
from tokenizers import Tokenizer
from scripts.data_builder.mix_sft_datasets import ExternalRandomShuffler

WORK = ROOT / 'dataset/cpt_l2_l3_12b_v1'
OLD = ROOT / 'dataset/pretrain_ultradata_l3_cpt_12b_general10_identity_clean.jsonl'
OUTPUT = ROOT / 'dataset/pretrain_ultradata_l2_30_l3_60_general10_12b_identity_clean.jsonl'
PRIOR = ROOT / 'dataset/ultradata_code_l3_12b/quality_clean_v1'
SEED = 2026101502
LANG_WEIGHTS = {'py':350,'cpp':400,'java':120,'js':40,'cs':30,'go':20,
                'php':20,'rs':10,'rb':5,'r':3,'sh':2}
GENERAL = {'math':300_000_000,'natural_en':600_000_000,'natural_zh':300_000_000}
L2_QUOTAS = {k:3_600_000*v for k,v in LANG_WEIGHTS.items()}
L3_QUOTAS = {k:7_200_000*v for k,v in LANG_WEIGHTS.items()}
DTYPE = np.dtype([('priority','V16'),('file','u4'),('offset','u8'),('tokens','u4')])
CREDENTIAL_RE=re.compile(r'(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})(?![A-Za-z0-9_-])')

def credential_literal(text):
    """High-confidence literal credential gate; placeholders are not secrets."""
    if re.search(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',text):return True
    return any(len(set(m.group()))>=12 for m in CREDENTIAL_RE.finditer(text))

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb',buffering=8*1024**2) as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def fingerprint(text):
    return hashlib.blake2b(text.strip().encode('utf-8'),digest_size=16).digest()

def uid(value):
    return uuid.UUID(str(value)).bytes

def contains(index, key):
    key=np.void(key); p=int(np.searchsorted(index,key))
    return p<len(index) and index[p]==key

def signature(path):
    s=Path(path).stat(); return [s.st_size,s.st_mtime_ns]

def save_index(path, values):
    index=np.unique(np.array(values,dtype='V16')); index.tofile(path)
    return {'path':str(path),'count':len(index),'sha256':sha(path)}

def base():
    """Keep all reviewed general rows and select 7.2B of reviewed L3."""
    WORK.mkdir(parents=True,exist_ok=True)
    out=WORK/'base.identity_clean.jsonl'; report=out.with_suffix('.report.json')
    old=read(OLD.with_suffix('.report.json')); checked=read(OLD.with_suffix('.verification.json'))
    if not old['training_ready'] or not checked['passed']:raise ValueError('Old source audit missing')
    if old['output_sha256']!=checked['output_sha256']:raise ValueError('Old audit digest mismatch')
    tokenizer_sha=sha(ROOT/'model/tokenizer.json')
    if tokenizer_sha!=old['tokenizer_sha256']:raise ValueError('Tokenizer changed')
    if report.exists():
        r=read(report)
        if r['source_signature']!=signature(OLD) or not out.exists():raise ValueError('Stale base stage')
        return r
    source_sig=signature(OLD); buffers={k:[] for k in L3_QUOTAS}; parts={k:[] for k in L3_QUOTAS}
    digest=hashlib.sha256(); offset=0; rows=0; seed=struct.pack('>Q',SEED)
    with OLD.open('rb',buffering=8*1024**2) as f:
        for line in f:
            digest.update(line); r=orjson.loads(line); k=r['mix_category']; rows+=1
            if k in L3_QUOTAS:
                fp=fingerprint(r['text'])
                if fp.hex()!=r['text_id']:raise ValueError('Old row fingerprint changed')
                buffers[k].append((hashlib.blake2b(seed+fp,digest_size=16).digest(),0,offset,r['token_count']))
                if len(buffers[k])>=100000:
                    parts[k].append(np.array(buffers[k],dtype=DTYPE)); buffers[k].clear()
            offset+=len(line)
            if rows%500000==0:
                dump(WORK/'base.status.json',{'phase':'index','rows':rows,'total_rows':old['rows']})
                print('[base-index]',rows,flush=True)
    if digest.hexdigest()!=old['output_sha256'] or rows!=old['rows']:raise ValueError('Old source hash/rows changed')
    positions=[]; totals=Counter()
    for k,quota in L3_QUOTAS.items():
        if buffers[k]:parts[k].append(np.array(buffers[k],dtype=DTYPE)); buffers[k].clear()
        a=np.concatenate(parts[k]); parts[k].clear(); a.sort(order='priority')
        for e in a:
            n=int(e['tokens'])
            if totals[k]+n<=quota:positions.append(int(e['offset']));totals[k]+=n
        if quota-totals[k]>4096:raise ValueError('L3 quota deficit: '+k)
        print('[base-select]',k,totals[k],flush=True)
    positions.sort(); wanted=iter(positions); target=next(wanted,None); offset=0
    count=Counter(); tokens=Counter(); uuids=[]; texts=[]; digest=hashlib.sha256()
    tmp=Path(str(out)+'.tmp')
    with OLD.open('rb',buffering=8*1024**2) as f,tmp.open('wb',buffering=8*1024**2) as w:
        for line in f:
            r=orjson.loads(line);k=r['mix_category']
            if k in GENERAL or offset==target:
                tier='general' if k in GENERAL else 'L3'
                r.update(data_tier=tier,mixture='l2-30-l3-60-general-10',mix_component=k if tier=='general' else 'L3:'+k)
                b=orjson.dumps(r)+b'\n';w.write(b);digest.update(b)
                count[r['mix_component']]+=1;tokens[r['mix_component']]+=r['token_count'];texts.append(bytes.fromhex(r['text_id']))
                if tier=='L3':uuids.append(uid(r['uuid']));target=next(wanted,None)
            offset+=len(line)
        w.flush();os.fsync(w.fileno())
    if target is not None or signature(OLD)!=source_sig:raise ValueError('Base selection/source changed')
    os.replace(tmp,out)
    indexes={'l3_uuids':save_index(WORK/'base_l3.uuid128',uuids),'texts':save_index(WORK/'base_texts.blake2b128',texts)}
    r={'output':str(out),'output_sha256':digest.hexdigest(),'rows':sum(count.values()),'tokens':sum(tokens.values()),
       'component_rows':dict(count),'component_tokens':dict(tokens),'source':str(OLD),'source_signature':source_sig,
       'source_sha256':old['output_sha256'],'source_audit':str(OLD.with_suffix('.verification.json')),
       'tokenizer_sha256':tokenizer_sha,'indexes':indexes,'seed':SEED,'identity_clean':old['identity_clean'],
       'benchmark_screening':old['benchmark_screening'],'previous_training_exclusion':old['previous_training_exclusion']}
    dump(report,r);dump(WORK/'base.status.json',{'phase':'complete','rows':r['rows'],'tokens':r['tokens']})
    print('[base-complete]',r['rows'],r['tokens'],flush=True);return r

class BufferedShuffler(ExternalRandomShuffler):
    @staticmethod
    def _iter_chunk(path):
        with path.open('rb',buffering=8*1024**2) as f:
            for line in f:
                key,row=line.rstrip(b'\n').split(b'\t',1);yield key,row

def assemble():
    if OUTPUT.exists() or OUTPUT.with_suffix('.report.json').exists():raise FileExistsError('New release already exists')
    b=read(WORK/'base.identity_clean.report.json');pool=read(WORK/'l2_pool.reviewed.report.json')
    files=sorted(pool['reports'],key=lambda x:x['output']);seen=set();seen_uuid=set();available=Counter();rejects=Counter()
    parts={k:[] for k in L2_QUOTAS};buffers={k:[] for k in L2_QUOTAS};seed=struct.pack('>Q',SEED+10)
    base_text=np.memmap(WORK/'base_texts.blake2b128',dtype='V16',mode='r')
    base_uid=np.memmap(WORK/'base_l3.uuid128',dtype='V16',mode='r')
    for i,info in enumerate(files):
        path=Path(info['output']);info['assembly_signature']=signature(path);digest=hashlib.sha256();offset=0
        with path.open('rb',buffering=8*1024**2) as f:
            for line in f:
                digest.update(line);r=orjson.loads(line);fp=fingerprint(r['text']);u=uid(r['uuid']);k=r['mix_category'];n=r['token_count']
                if r['text_id']!=fp.hex() or r['data_tier']!='L2':raise ValueError('L2 row identity mismatch')
                if contains(base_text,fp) or contains(base_uid,u):raise ValueError('L2 overlaps selected base')
                if credential_literal(r['text']):rejects[k+':credential_literal']+=1;offset+=len(line);continue
                if fp in seen:rejects[k+':exact_text_duplicate']+=1;offset+=len(line);continue
                if u in seen_uuid:rejects[k+':same_source_uuid']+=1;offset+=len(line);continue
                seen.add(fp);seen_uuid.add(u);available[k]+=n
                buffers[k].append((hashlib.blake2b(seed+fp,digest_size=16).digest(),i,offset,n));offset+=len(line)
                if len(buffers[k])>=100000:parts[k].append(np.array(buffers[k],dtype=DTYPE));buffers[k].clear()
        if digest.hexdigest()!=info['output_sha256']:raise ValueError('L2 pool hash mismatch')
        print('[l2-index]',i+1,len(files),dict(available),flush=True)
    del seen,seen_uuid
    for k,q in L2_QUOTAS.items():
        if available[k]<q:raise ValueError(f'Need more L2 {k}: {available[k]} < {q}')
    selected={i:[] for i in range(len(files))};totals=Counter();rows=Counter()
    for k,q in L2_QUOTAS.items():
        if buffers[k]:parts[k].append(np.array(buffers[k],dtype=DTYPE));buffers[k].clear()
        a=np.concatenate(parts[k]);parts[k].clear();a.sort(order='priority')
        for e in a:
            n=int(e['tokens'])
            if totals['L2:'+k]+n<=q:
                selected[int(e['file'])].append(int(e['offset']));totals['L2:'+k]+=n;rows['L2:'+k]+=1
        if q-totals['L2:'+k]>4096:raise ValueError('L2 quota not filled')
        print('[l2-select]',k,totals['L2:'+k],flush=True)
    shuffle=BufferedShuffler(OUTPUT,SEED+20,80000,chunk_bytes=512*1024**2,deduplicate=False)
    try:
        digest=hashlib.sha256()
        with Path(b['output']).open('rb',buffering=8*1024**2) as f:
            for line in f:digest.update(line);shuffle.add_serialized(line)
        if digest.hexdigest()!=b['output_sha256']:raise ValueError('Base changed')
        totals.update(b['component_tokens']);rows.update(b['component_rows'])
        for i,info in enumerate(files):
            path=Path(info['output'])
            if signature(path)!=info['assembly_signature']:raise ValueError('L2 indexed source changed')
            with path.open('rb',buffering=8*1024**2) as f:
                for p in sorted(selected.pop(i)):f.seek(p);shuffle.add_serialized(f.readline())
            print('[shuffle-feed]',i+1,len(files),shuffle.rows_added,flush=True)
        result=shuffle.finish(progress_label='L2 L3 CPT shuffle')
    except BaseException:
        shuffle.temporary_dir.cleanup();raise
    report={'name':'InstinctV1Moe CPT L2 30%, L3 60%, unused general 10%',
            'output':str(OUTPUT),'output_sha256':result['sha256'],'rows':result['rows'],'tokens':sum(totals.values()),
            'target_tokens':12_000_000_000,'target_tier_fractions':{'L2':0.3,'L3':0.6,'general':0.1},
            'component_tokens':dict(totals),'component_rows':dict(rows),'language_weights':LANG_WEIGHTS,
            'quotas':{**{'L2:'+k:v for k,v in L2_QUOTAS.items()},**{'L3:'+k:v for k,v in L3_QUOTAS.items()},**GENERAL},
            'tokenizer_sha256':b['tokenizer_sha256'],'token_count_policy':'Instinct tokenizer, add_special_tokens=False, plus 2 for BOS/EOS',
            'max_tokens':4096,'min_tokens':64,'text_truncated':False,'rewritten_text_rows':0,
            'previous_training_replay_tokens':0,'previous_training_exclusion':b['previous_training_exclusion'],
            'base':b,'l2_pool':pool,'l2_selection':{'seed':SEED+10,'available_tokens_after_global_dedup':dict(available),'rejections':dict(rejects)},
            'identity_clean':{'target_identity':'InstinctV1Moe','inherited_l3_general':b['identity_clean'],
                              'l2_review':pool['identity_review'],'unreviewed_selected_records':0},
            'benchmark_screening':b['benchmark_screening'],'same_source_l2_l3_policy':'Reject shared L2/L3 UUIDs when present and exact text matches across tiers. Cross-tier UUID stability is not established by the published schema.',
            'l2_credential_screening':'Exclude high-confidence sk-/GitHub token literals and PEM private-key blocks; no secrets included in logs. This is not an exhaustive secret detector.',
            'shuffle':result,'training_ready':False,
            'limitations':['The 30/60/10 recipe is an engineering candidate, not a measured optimum on Instinct.',
                          'Novel means no normalized full-text match to the four prior corpora; semantic novelty is not proved.',
                          'The initial Python/C++ L2 shards have zero UUID intersections with selected L3. UUID namespaces may differ, so zero UUID overlap does not prove source-implementation independence. Forks, copied code and paraphrases may remain.',
                          'Native L2 Python parse failures are counted, not blanket-excluded; legacy code and external dependencies can be valid training text.',
                          'Source code and generated L3 tests were not executed; correctness is not guaranteed.',
                          'Lexical benchmark screening cannot prove absence of semantic contamination.',
                          'Whole-record 4096-token selection favors shorter files; source code is not split or truncated.',
                          'Source-repository licenses remain applicable.']}
    dump(OUTPUT.with_suffix('.report.json'),report);print('[release]',result['rows'],report['tokens'],flush=True)

def audit():
    report=read(OUTPUT.with_suffix('.report.json'));old=report['previous_training_exclusion']
    if sha(PRIOR/'prior.blake2b128')!=old['sha256']:raise ValueError('Prior index changed')
    for item in old['manifest']:
        if signature(item['path'])!=[item['bytes'],item['mtime_ns']]:raise ValueError('Prior corpus changed')
    if sha(ROOT/'model/tokenizer.json')!=report['tokenizer_sha256']:raise ValueError('Tokenizer changed')
    tokenizer=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
    prior=np.memmap(PRIOR/'prior.blake2b128',dtype='V16',mode='r')
    l3uids=np.memmap(WORK/'base_l3.uuid128',dtype='V16',mode='r');seen=set();l2uids=set()
    rows=Counter();tokens=Counter();tiers=Counter();digest=hashlib.sha256();n=samples=0;windows=[];window=Counter()
    with OUTPUT.open('rb',buffering=8*1024**2) as f:
        for line in f:
            n+=1;digest.update(line);r=orjson.loads(line);text=r['text'];count=r['token_count'];tier=r['data_tier'];k=r['mix_component']
            if not isinstance(text,str) or not text.strip() or type(count)!=int or not 64<=count<=4096:raise ValueError('Invalid text/count')
            fp=fingerprint(text)
            if r['text_id']!=fp.hex() or fp in seen or contains(prior,fp):raise ValueError('Duplicate or prior training row')
            seen.add(fp)
            if tier=='L2':
                if credential_literal(text):raise ValueError('Credential literal in selected L2')
                u=uid(r['uuid'])
                if u in l2uids or contains(l3uids,u):raise ValueError('Duplicate/same-origin L2 UUID')
                l2uids.add(u)
                if not all(r.get(x) for x in ('repo_name','relative_path','source_revision','source_shard','source_row','identity_decision')):raise ValueError('L2 provenance/review missing')
                if k!='L2:'+r['mix_category']:raise ValueError('L2 component mismatch')
            elif tier=='L3':
                if not contains(l3uids,uid(r['uuid'])) or k!='L3:'+r['mix_category']:raise ValueError('L3 provenance mismatch')
            elif tier=='general':
                if k not in GENERAL or not all(r.get(x) for x in ('collector_source','upstream_shard','source_revision','identity_decision')):raise ValueError('General provenance missing')
            else:raise ValueError('Unexpected tier')
            if n%1000==1:
                if len(tokenizer.encode(text,add_special_tokens=False).ids)+2!=count:raise ValueError('Retokenization mismatch')
                samples+=1
            rows[k]+=1;tokens[k]+=count;tiers[tier]+=count;window[tier]+=count
            if sum(window.values())>=100_000_000:windows.append(dict(window));window.clear()
            if n%500000==0:
                dump(WORK/'audit.status.json',{'checked_rows':n,'total_rows':report['rows'],'tokens':sum(tokens.values())})
                print('[audit]',n,flush=True)
    if window:windows.append(dict(window))
    if digest.hexdigest()!=report['output_sha256'] or dict(rows)!=report['component_rows'] or dict(tokens)!=report['component_tokens']:raise ValueError('Audit totals/hash mismatch')
    for k,q in report['quotas'].items():
        if not 0<=q-tokens[k]<=4096:raise ValueError('Quota deviation')
    result={'passed':True,'rows':n,'tokens':sum(tokens.values()),'output_sha256':digest.hexdigest(),'component_tokens':dict(tokens),
            'component_rows':dict(rows),'tier_tokens':dict(tiers),'exact_duplicates':0,'exact_previous_training_overlaps':0,
            'shared_l2_l3_uuid_values':0,'token_count_samples':samples,'token_count_sample_mismatches':0,
            'contiguous_100m_token_windows':windows,'scope':'All rows: schema, text fingerprints, uniqueness, prior exclusion, L2/L3 UUID separation, provenance, token totals; deterministic tokenizer sample.'}
    dump(OUTPUT.with_suffix('.verification.json'),result)
    report.update(training_ready=True,verification=str(OUTPUT.with_suffix('.verification.json')),tier_tokens=dict(tiers))
    dump(OUTPUT.with_suffix('.report.json'),report);print('[audit-PASSED]',n,sum(tokens.values()),flush=True)

def finalize():
    compiler=ROOT/'dataset_compiler/target/release/dataset_compiler.exe'
    parquet=OUTPUT.with_suffix('.parquet');sidecar=Path(str(parquet)+'.report.json')
    def phase(value):
        dump(WORK/'finalization.status.json',{'phase':value,'jsonl':str(OUTPUT),'parquet':str(parquet)});print('[finalize]',value,flush=True)
    if not OUTPUT.with_suffix('.report.json').exists():phase('assemble');assemble()
    if not OUTPUT.with_suffix('.verification.json').exists():phase('audit_jsonl');audit()
    report=read(OUTPUT.with_suffix('.report.json'));verification=read(OUTPUT.with_suffix('.verification.json'))
    if not report['training_ready'] or not verification['passed']:raise ValueError('JSONL audit failed')
    if not sidecar.exists():
        if parquet.exists():raise ValueError('Partial Parquet exists; inspect before resume')
        phase('compile_one_parquet')
        subprocess.run([str(compiler),'compile','-i',str(OUTPUT),'-o',str(parquet),'--align-chunk-bytes','1GiB','--threads','16'],cwd=ROOT,check=True)
    phase('compare_every_parquet_row');log=Path(str(parquet)+'.verification.log')
    with log.open('wb') as f:
        subprocess.run([str(compiler),'verify',str(OUTPUT),str(parquet),'--rows',str(report['rows'])],cwd=ROOT,check=True,stdout=f,stderr=subprocess.STDOUT)
    text=log.read_text(encoding='utf-8')
    if 'value_mismatches=0' not in text or 'row_count=match' not in text:raise ValueError('Unexpected compiler verification result')
    phase('independent_parquet_audit')
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    compiled=read(sidecar);actual_sha=sha(parquet)
    if compiled['output']['sha256']!=actual_sha:raise ValueError('Parquet digest mismatch')
    table=pq.read_table(parquet,columns=['token_count','mix_component','data_tier'])
    if any(table[k].null_count for k in table.column_names):raise ValueError('Parquet metadata nulls')
    grouped=table.group_by('mix_component').aggregate([('token_count','sum'),('token_count','count')]).to_pylist()
    if {x['mix_component']:x['token_count_sum'] for x in grouped}!=report['component_tokens']:raise ValueError('Parquet token totals')
    if {x['mix_component']:x['token_count_count'] for x in grouped}!=report['component_rows']:raise ValueError('Parquet row totals')
    if pc.min(table['token_count']).as_py()<64 or pc.max(table['token_count']).as_py()>4096:raise ValueError('Parquet length bounds')
    audit_path=Path(str(parquet)+'.verification.json')
    independent={'passed':True,'parquet':str(parquet),'sha256':actual_sha,'bytes':parquet.stat().st_size,'rows':table.num_rows,
                 'tokens':report['tokens'],'component_tokens':report['component_tokens'],'tier_tokens':report['tier_tokens'],
                 'all_rows_compared':True,'compiler_verification_log':str(log)}
    dump(audit_path,independent)
    compiled.update(training_ready=True,identity_clean=report['identity_clean'],benchmark_screening=report['benchmark_screening'],
                    tokens=report['tokens'],component_tokens=report['component_tokens'],tier_tokens=report['tier_tokens'],
                    cleaned_jsonl_report=str(OUTPUT.with_suffix('.report.json')),cleaned_jsonl_sha256=report['output_sha256'],
                    previous_training_replay_tokens=0,single_mixed_training_artifact=True,
                    parquet_full_verification={'passed':True,'requested_rows':report['rows'],'log':str(log)},independent_parquet_audit=str(audit_path))
    dump(sidecar,compiled);phase('complete');print(json.dumps(independent),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['base','assemble','audit','finalize'])
    args=p.parse_args()
    if args.command=='base':base()
    elif args.command=='assemble':assemble()
    elif args.command=='audit':audit()
    else:finalize()

if __name__=='__main__':main()
