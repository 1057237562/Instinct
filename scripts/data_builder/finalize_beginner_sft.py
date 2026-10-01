"""Separate provenance from trainer records and remove interactive tasks."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import orjson

ROOT=Path(__file__).resolve().parents[2]/'dataset'

def main():
    path=ROOT/'sft_competitive_beginner_4096.jsonl'
    tmp=path.with_suffix('.tmp')
    side=ROOT/'first_sft_sources/codeforces_beginner/selected_sources.jsonl'
    report_path=path.with_suffix('.report.json')
    report=orjson.loads(report_path.read_bytes())
    counts=Counter();ratings=Counter();languages=Counter();ids=set();tags=Counter();digest=hashlib.sha256()
    with path.open('rb') as inp,tmp.open('xb') as out,side.open('xb') as meta:
        for line in inp:
            r=orjson.loads(line)
            if 'interactive' in r['tags']:
                counts['interactive_rejected']+=1;continue
            data=orjson.dumps({'conversations':r.pop('conversations')})+b'\n'
            out.write(data);digest.update(data)
            r['row_index']=counts['rows'];meta.write(orjson.dumps(r)+b'\n')
            counts['rows']+=1;counts['tokens']+=r['token_count']
            counts['max_tokens']=max(counts['max_tokens'],r['token_count'])
            ratings[r['rating']]+=1;languages[r['language']]+=1;ids.add(r['problem_id']);tags.update(r['tags'])
        out.flush();os.fsync(out.fileno())
    os.replace(tmp,path)
    report.update(rows=counts['rows'],tokens=counts['tokens'],max_template_tokens=counts['max_tokens'],
                  unique_problems=len(ids),ratings=dict(ratings),languages=dict(languages),tags=dict(tags),
                  sha256=digest.hexdigest(),source_manifest=str(side),bytes=path.stat().st_size)
    report['filter_stats']['interactive_rejected']=counts['interactive_rejected']
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(counts,flush=True)

if __name__=='__main__':main()
