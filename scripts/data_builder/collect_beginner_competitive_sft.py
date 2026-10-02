"""Fetch Codeforces problem metadata and join accepted local submissions for SFT.

Remote Parquet projection avoids downloading bulky execution tests. Downloaded
programs are never executed. Correctness status is the upstream verdict only.
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import orjson
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'dataset/first_sft_sources/codeforces_beginner'
OUT = ROOT / 'dataset/sft_competitive_beginner_4096.jsonl'
REV = 'fbe3f6e903ee854eec2e69e9d96d0306cde59baf'
COLS = ['id', 'aliases', 'rating', 'title', 'description', 'input_format',
        'output_format', 'examples', 'tags', 'editorial', 'interaction_format', 'note']

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    RAW.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    problems = {}
    test_ids = set()
    for split,count in [('test',1),('train',11)]:
        for i in range(count):
            name = f'{split}-{i:05d}-of-{count:05d}'
            cache = RAW / (name + '.jsonl')
            if not cache.exists():
                remote = f'datasets/open-r1/codeforces@{REV}/data/{name}.parquet'
                with fs.open(remote,'rb') as handle:
                    table = pq.read_table(handle, columns=COLS)
                with cache.open('xb') as target:
                    for row in table.to_pylist():
                        target.write(orjson.dumps(row)+b'\n')
            with cache.open('rb') as handle:
                for line in handle:
                    p=orjson.loads(line)
                    ids=[p['id']]+(p.get('aliases') or [])
                    if split=='test': test_ids.update(ids);continue
                    rating=p.get('rating')
                    if rating is None or not 800<=rating<=1200 or p.get('interaction_format'):continue
                    if any(pid in test_ids for pid in ids):continue
                    for pid in ids:problems[pid]=p
            print(f'fetched {name}; eligible IDs={len(problems)}',flush=True)

    tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    counts=Counter(); stats=Counter(); seen=set(); records=[]
    selected=ROOT/'dataset/codeforces-submissions/selected_accepted/train-00000-of-00001.parquet'
    paths=[selected]+sorted((ROOT/'dataset/codeforces-submissions/data').glob('*.parquet'))
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048,columns=['problem_id','source','programmingLanguage','verdict','testset','submission_id']):
            for row in batch.to_pylist():
                p=problems.get(row['problem_id'])
                if p is None or row['verdict']!='OK':continue
                if path!=selected and row['testset']!='TESTS':continue
                lang=row['programmingLanguage'] or ''
                language='python' if 'Python' in lang or 'PyPy' in lang else 'cpp' if 'C++' in lang else None
                if not language:continue
                key=(p['id'],language)
                if counts[key]>=3:continue
                code=row['source']
                if not code or not code.strip():continue
                fingerprint=hashlib.sha256(code.encode()).hexdigest()
                if fingerprint in seen:continue
                prompt=f"Solve this competitive programming problem using {language}. Read from standard input and write to standard output. Return a complete solution.\n\n{p['title']}\n\n{p['description']}"
                for field,title in [('input_format','Input'),('output_format','Output')]:
                    if p.get(field):prompt+=f'\n\n{title}\n{p[field]}'
                for ex in p.get('examples') or []:
                    prompt+=f"\n\nExample input\n{ex.get('input','')}\nExample output\n{ex.get('output','')}"
                if p.get('note'):prompt+='\n\nNote\n'+p['note']
                messages=[{'role':'user','content':prompt},{'role':'assistant','content':f'```{language}\n{code}\n```'}]
                rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
                token_count=len(tokenizer.backend_tokenizer.encode(rendered,add_special_tokens=False).ids)
                # Training can prepend a generic system message; reserve 128 tokens.
                if token_count>3968:stats['overlength_or_system_reserve']+=1;continue
                seen.add(fingerprint);counts[key]+=1
                records.append({'conversations':messages,'problem_id':p['id'],'rating':p['rating'],
                    'language':language,'tags':p.get('tags') or [],'token_count':token_count,
                    'source':'open-r1/codeforces + open-r1/codeforces-submissions',
                    'revision':REV,'submission_id':str(row['submission_id']),
                    'verification':'upstream accepted; not locally executed',
                    'url':f"https://codeforces.com/contest/{p['id'].split('/')[0]}/problem/{p['id'].split('/')[1]}"})
        print(f'{path.name}: kept={len(records)}',flush=True)
    random.Random(20260913).shuffle(records)
    with OUT.open('xb') as handle:
        for r in records:handle.write(orjson.dumps(r)+b'\n')
    report={'output':str(OUT),'rows':len(records),'unique_problems':len({r['problem_id'] for r in records}),
        'ratings':dict(Counter(r['rating'] for r in records)), 'languages':dict(Counter(r['language'] for r in records)),
        'tags':dict(Counter(t for r in records for t in r['tags'])),
        'tokens':sum(r['token_count'] for r in records),'max_template_tokens':max((r['token_count'] for r in records),default=0),
        'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'filter_stats':dict(stats),
        'official_test_excluded':True,'max_answers_per_problem_per_language':3,
        'pretraining_overlap':'not claimed independent: submissions and some problems may occur in prior pretraining corpus',
        'evaluation_note':'official test exclusion is not proof of decontamination against prior local training',
        'license':'problem metadata CC-BY-4.0; submission data ODC-By, see upstream cards',
        'role':'first-SFT competitive component; not a replacement for standard SFT general/tool data'}
    OUT.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='tags'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
