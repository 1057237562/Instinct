"""Prepare immutable, bounded review batches; never modify training sources."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import datasets  # Windows: before transformers/torch
import orjson
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.data_builder.filter_anomaly_candidates import iter_file_rows, sha256_file
from scripts.data_loader.lm_dataset import _create_chat_prompt

OUT=ROOT/'dataset/review_candidates/final_sft_handoff_20260928'
DATA=ROOT/'dataset'
SEED=20260928
TARGET='我是 Instinct，一个从头训练的语言模型，由 L1bra 个人独立开发和训练，不隶属于任何商业组织。'

def canonical(row): return orjson.dumps(row,option=orjson.OPT_SORT_KEYS)
def sha(row): return hashlib.sha256(canonical(row)).hexdigest()
def emit(f,row): f.write(orjson.dumps(row)+b'\n')

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    ready=DATA/'sft_t2t_mini.train_ready.jsonl'
    anchors=DATA/'identity_anchors_instinct.jsonl'
    audit=DATA/'review_candidates/sft_t2t_mini.train_ready.audit.json'
    residual=DATA/'review_candidates/sft_t2t_mini.train_ready_candidates.jsonl'
    ar=orjson.loads(audit.read_bytes())
    assert sha256_file(ready)==ar['inputs'][0]['sha256']=='8dc5475ce99988709e06cf38a267f9d8f2135bf438d202d9898e5cbf090d85aa'
    assert sha256_file(residual)==ar['output_sha256']
    tasks=defaultdict(list); manifests=[]
    def add(category,row,path,line,**extras):
        record_hash=sha(row)
        item={'review_id':category+'-'+sha([str(path.relative_to(ROOT)),line,record_hash])[:20],
              'category':category,'source_path':str(path.relative_to(ROOT)),'source_row':line,
              'record_sha256':record_hash,'record':row,**extras}
        tasks[category].append(item)
    # Full residual list, not only the hard-claim subset.
    for line,r,error,_ in iter_file_rows(residual):
        assert not error
        add('P0_t2t_residual',r['record'],ready,r['_review']['row_number'],audit_evidence=r['_review'],upstream_candidate_line=line)
    grouped={}
    for line,r,error,_ in iter_file_rows(anchors):
        assert not error
        conv=r['conversations']; user=next(m['content'] for m in conv if m['role']=='user'); assistant=next(m for m in conv if m['role']=='assistant')
        key=(user,assistant['content'])
        g=grouped.setdefault(key,{'rows':[],'hashes':[],'reasoning':set(),'systems':set()})
        g['rows'].append(line); g['hashes'].append(sha(r))
        if assistant.get('reasoning_content'): g['reasoning'].add(assistant['reasoning_content'])
        g['systems'].update(m['content'] for m in conv if m['role']=='system')
    for (question,answer),group in grouped.items():
        row={'conversations':[{'role':'user','content':question},{'role':'assistant','content':answer}]}
        add('P0_anchor_pairs',row,anchors,group['rows'][0],source_row_numbers=group['rows'],
            source_row_sha256s=group['hashes'],reasoning_variants=sorted(group['reasoning']),system_variants=sorted(group['systems']),
            canonical_identity=TARGET)
    triage=DATA/'v1moe_final/triage.jsonl'
    for _,r,error,_ in iter_file_rows(triage):
        assert not error
        raw=r['record']; meta=raw['_meta']
        # Old T2T and mixtures containing old T2T will be replaced, not re-audited.
        if meta['source'] in ('t2t_multiturn','pilot_t2t','pilot_python_reviewed'): continue
        add('P1_other_identity',{'conversations':raw['conversations']},Path(r['_review']['source']),r['_review']['row_number'],
            audit_evidence=r['_review'],provenance=meta)
    rng=random.Random(SEED); pools=defaultdict(list); seen=Counter()
    def reservoir(group,row,line,path,cap):
        seen[group]+=1; item=(row,line,path)
        if len(pools[group])<cap: pools[group].append(item)
        else:
            k=rng.randrange(seen[group])
            if k<cap: pools[group][k]=item
    code=DATA/'v1moe_final/candidates/code.jsonl'
    for line,r,error,_ in iter_file_rows(code):
        assert not error
        reservoir('code:'+r['_meta']['path'],r,line,code,16)
    for group,pool in pools.items():
        for r,line,path in pool:
            add('P1_python_semantics',{'conversations':r['conversations']},path,line,provenance=r['_meta'])
    pools.clear(); seen.clear()
    core=DATA/'v1moe_final/candidates/core.jsonl'
    for line,r,error,_ in iter_file_rows(core):
        assert not error
        if r['_meta']['source']=='ultrachat_multiturn': reservoir('chat',r,line,core,80)
    for r,line,path in pools['chat']:
        add('P1_multiturn_quality',{'conversations':r['conversations']},path,line,provenance=r['_meta'])
    pools.clear(); seen.clear()
    long=DATA/'v1moe_final/candidates/long.jsonl'
    for line,r,error,_ in iter_file_rows(long):
        assert not error
        text='\n'.join(m['content'] for m in r['conversations'])
        if 12000<=len(text)<=230000: reservoir(r['_meta']['source'],r,line,long,120)
    t=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    for source,pool in pools.items():
        eligible=[]
        for r,line,path in pool:
            n=len(t.backend_tokenizer.encode(_create_chat_prompt(t,r['conversations']),add_special_tokens=False).ids)
            if 8192<=n<=65200: eligible.append((n,r,line,path))
        eligible.sort(key=lambda item:item[0])
        assert len(eligible)>=24,(source,len(eligible))
        indices=sorted({round(i*(len(eligible)-1)/23) for i in range(24)})
        for index in indices:
            n,r,line,path=eligible[index]
            add('P2_long_grounding',{'conversations':r['conversations']},path,line,provenance=r['_meta'],instinct_template_tokens=n)
    pools.clear(); seen.clear()
    synthetic=DATA/'v1moe_final/candidates/synthetic.jsonl'
    for line,r,error,_ in iter_file_rows(synthetic):
        assert not error
        reservoir(str(r['_meta']['family']),r,line,synthetic,4)
    for pool in pools.values():
        for r,line,path in pool:
            add('P2_synthetic_wording',{'conversations':r['conversations']},path,line,provenance=r['_meta'])
    sizes={'P0_t2t_residual':15,'P0_anchor_pairs':25,'P1_other_identity':10,'P1_python_semantics':10,
           'P1_multiturn_quality':5,'P2_long_grounding':1,'P2_synthetic_wording':1}
    allids=set()
    for category,items in tasks.items():
        folder=OUT/category; folder.mkdir()
        for offset in range(0,len(items),sizes[category]):
            batch=items[offset:offset+sizes[category]]; path=folder/f'batch_{offset//sizes[category]+1:03d}.jsonl'
            with path.open('wb') as f:
                for item in batch:
                    assert item['review_id'] not in allids
                    allids.add(item['review_id']); emit(f,item)
            manifests.append({'path':str(path.relative_to(OUT)),'rows':len(batch),'sha256':sha256_file(path),
                              'review_ids':[i['review_id'] for i in batch]})
    sourcefiles=[ready,anchors,audit,residual,triage,code,core,long,synthetic]
    manifest={'status':'review handoff; not training data','seed':SEED,'canonical_identity':TARGET,
              'source_artifacts':[{'path':str(p.relative_to(ROOT)),'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sourcefiles],
              'counts':{k:len(v) for k,v in tasks.items()},'anchor_original_rows':sum(len(x['rows']) for x in grouped.values()),
              'total_review_items':len(allids),'batches':manifests,
              'sampling_note':'All 129 T2T residuals and all 225 unique anchor QA pairs; other categories are targeted/exploratory samples, not population quality guarantees.',
              'do_not_train':True}
    (OUT/'manifest.json').write_bytes(orjson.dumps(manifest,option=orjson.OPT_INDENT_2))
    result_template={'review_id':'COPY EXACT ID','record_sha256':'COPY EXACT HASH','decision':'keep|repair|drop|uncertain',
                     'reason':'specific evidence, not merely a keyword hit','issue_tags':[],
                     'patches':[], 'counterexample':None,'reasoning_action':'keep_reviewed|clear|replace|not_applicable'}
    (OUT/'result_template.json').write_bytes(orjson.dumps(result_template,option=orjson.OPT_INDENT_2))
    print(json.dumps({'counts':manifest['counts'],'total':len(allids),'batches':len(manifests),'directory':str(OUT)},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
