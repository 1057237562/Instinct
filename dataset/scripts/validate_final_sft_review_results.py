"""Check review result coverage, immutable IDs/hashes and patch shape; no edits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'dataset/review_candidates/final_sft_handoff_20260928'

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=DEFAULT)
    p.add_argument('--results',type=Path)
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args(); root=a.package.resolve(); manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    tasks={}; errors=[]
    for batch in manifest['batches']:
        path=(root/batch['path']).resolve()
        if not path.is_relative_to(root): raise ValueError('Unsafe batch path')
        if file_hash(path)!=batch['sha256']: raise ValueError(f'Modified input batch: {path}')
        rows=[json.loads(l) for l in path.open(encoding='utf-8') if l.strip()]
        if [r['review_id'] for r in rows]!=batch['review_ids']: raise ValueError('Batch ID mismatch')
        for r in rows:
            if r['review_id'] in tasks: raise ValueError('Duplicate task ID')
            encoded=json.dumps(r['record'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
            if hashlib.sha256(encoded).hexdigest()!=r['record_sha256']: raise ValueError('Record hash mismatch')
            tasks[r['review_id']]=r
    received={}; decisions=Counter(); categories=Counter()
    results=a.results or root/'results'
    for path in sorted(results.glob('*.jsonl')):
        for lineno,line in enumerate(path.open(encoding='utf-8'),1):
            if not line.strip(): continue
            try:
                r=json.loads(line); identity=r['review_id']; task=tasks[identity]
                if identity in received: raise ValueError('Duplicate review ID')
                if r['record_sha256']!=task['record_sha256']: raise ValueError('Record SHA mismatch')
                if r['decision'] not in ('keep','repair','drop','uncertain'): raise ValueError('Invalid decision')
                if not isinstance(r.get('reason'),str) or not r['reason'].strip(): raise ValueError('Missing reason')
                if not isinstance(r.get('issue_tags',[]),list): raise ValueError('issue_tags must be a list')
                if r.get('reasoning_action','not_applicable') not in ('keep_reviewed','clear','replace','not_applicable'): raise ValueError('Invalid reasoning action')
                patches=r.get('patches',[])
                if not isinstance(patches,list): raise ValueError('patches must be a list')
                if r['decision']=='repair' and not patches and r.get('reasoning_action')!='clear': raise ValueError('Repair has no patch')
                if r['decision']!='repair' and patches: raise ValueError('Only repair may contain patches')
                fields=set()
                for patch in patches:
                    index=patch['message_index']; field=patch['field']; value=patch['value']
                    if not isinstance(index,int) or not 0<=index<len(task['record']['conversations']): raise ValueError('Invalid message index')
                    if task['record']['conversations'][index]['role']=='user': raise ValueError('Do not change user requests')
                    if field not in ('content','reasoning_content'): raise ValueError('Invalid patch field')
                    if value is None and field!='reasoning_content': raise ValueError('Only reasoning may be removed')
                    if value is not None and not isinstance(value,str): raise ValueError('Patch value must be string/null')
                    if (index,field) in fields: raise ValueError('Repeated patch field')
                    fields.add((index,field))
                received[identity]=r; decisions[r['decision']]+=1; categories[task['category']]+=1
            except (ValueError,KeyError,TypeError) as e:
                errors.append({'path':str(path),'line':lineno,'error':str(e)})
    missing=sorted(set(tasks)-set(received))
    summary={'input_package_valid':True,'expected':len(tasks),'received':len(received),'missing':len(missing),
             'decisions':dict(decisions),'received_by_category':dict(categories),'errors':errors,
             'ready_for_integration':not missing and not errors and not decisions['uncertain']}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if errors or (missing and not a.allow_partial): raise SystemExit(2)

if __name__=='__main__': main()
