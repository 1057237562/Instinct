"""Contextually recover factual identity-triage false positives in new sources."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['TOKENIZERS_PARALLELISM']='false'
import datasets  # Windows import ordering
import scripts
scripts.__path__=[str(ROOT/'scripts'),*scripts.__path__]
import orjson
from tokenizers import Tokenizer
from scripts.data_builder.build_cpt_unused_general import WORK, fingerprint
from scripts.data_builder import collect_coder_pretrain_12b as collector
from scripts.data_builder.clean_ultradata_l3_cpt import dump
from scripts.data_builder.build_ultradata_code_l3_12b import sha256_file

def main():
    pool=json.loads((WORK/'pool.report.json').read_text(encoding='utf-8'))
    manual_path=WORK/'identity_manual_decisions.json'
    manual=json.loads(manual_path.read_text(encoding='utf-8')) if manual_path.exists() else {}
    tokenizer=Tokenizer.from_file(str(ROOT/'model/tokenizer.json'))
    ledger=[];missing=[];counts=Counter();recoveries={k:[] for k in collector.TARGETS if k!='code'}
    for report in pool['reports']:
        job=report['job']; component=job['component'];cases={}
        if sha256_file(Path(report['identity_candidates']))!=report['identity_candidates_sha256']:
            raise ValueError('Identity candidate artifact changed after scan')
        for line in Path(report['identity_candidates']).open('rb'):
            case=orjson.loads(line);key=case['text_id'];text=case['text']
            risky=case['contextual_reasons'] or any(h['tier']=='claim' and re.search(r'\b(?:assistant|chatbot|language.model)\b|语言模型|智能助手',h['snippet'],re.I) for h in case['identity_hits'])
            decision=manual.get(key)
            if risky and decision is None:missing.append(case);continue
            if decision is None:
                decision={'decision':'keep','reason':'Contextual preservation policy: no targeted model persona/model-card phrase; triage refers to factual discussion, real human speakers, place names, scientific GLM, or third-party model descriptions.'}
            if decision['decision'] not in ['keep','drop'] or not decision.get('reason'):raise ValueError('Invalid identity decision')
            ledger.append({'text_id':key,**decision,'component':component,'identity_hits':case['identity_hits'],'source':job['path'],'source_row':case['source_row']})
            counts[decision['decision']]+=1
            if decision['decision']=='keep':cases[case['source_row']]=case
        if not cases:continue
        source=Path(job['path'])
        if job['kind']=='collector':
            def input_rows():
                with source.open('rb',buffering=1024**2) as f:
                    for line in f:yield orjson.loads(line)
        else:
            def input_rows():
                for i,row in enumerate(collector.parquet_rows(source),1):
                    if i>job['start']:yield {**row,'_upstream_row':i}
        last=max(cases)
        for number,raw in enumerate(input_rows(),1):
            if number>last:break
            if number not in cases:continue
            case=cases[number];text=case['text'];count=len(tokenizer.encode(text,add_special_tokens=False).ids)+2
            if not 64<=count<=4096:counts['recovered_but_length_excluded']+=1;continue
            if job['kind']=='collector':row={**raw,'text':text}
            else:
                row=collector.record_for(component,raw,text,count,job['upstream'],raw['_upstream_row']-1)
                row['upstream_row']=raw['_upstream_row']
            if fingerprint(row['text']).hex()!=case['text_id']:raise ValueError('Identity review/source mismatch')
            row.update(text_id=case['text_id'],mix_category=component,source_revision=collector.SOURCES[component]['revision'],
                       token_count=count,novel_source_kind=job['kind'],collector_source=job['path'],collector_source_row=number,
                       identity_decision='factual_mentions_preserved_no_persona_or_model_card_pattern',
                       identity_contextual_decision='fingerprint-bound review ledger')
            recoveries[component].append(row)
    required=WORK/'identity_required_review.jsonl'
    required.write_bytes(b''.join(orjson.dumps(c)+b'\n' for c in missing))
    if missing:raise ValueError(f'{len(missing)} contextual identity decisions required: {required}')
    ledger_path=WORK/'identity_contextual_decisions.jsonl'
    ledger_path.write_bytes(b''.join(orjson.dumps(c)+b'\n' for c in ledger))
    for component,rows in recoveries.items():
        output=WORK/'pool'/f'{component}_contextually_recovered.jsonl'
        data=b''.join(orjson.dumps(row)+b'\n' for row in rows);output.write_bytes(data)
        report={'job':{'component':component,'kind':'contextual_identity_recovery'},'output':str(output),
                'output_sha256':hashlib.sha256(data).hexdigest(),'tokens':sum(row['token_count'] for row in rows),'rows':len(rows)}
        pool['reports'].append(report);pool['tokens'][component]+=report['tokens']
    pool['identity_review']={'counts':dict(counts),'ledger':str(ledger_path),'manual_decisions':str(manual_path),
                             'manual_decisions_sha256':sha256_file(manual_path),
                             'unreviewed_final_candidates':0,'ledger_sha256':hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                             'policy':'Preserve factual references and human/place names; explicit assistant-persona passages require fingerprint-bound decisions.'}
    dump(WORK/'pool.reviewed.report.json',pool)
    print(json.dumps({'tokens':pool['tokens'],'identity_counts':dict(counts)}),flush=True)

if __name__=='__main__':main()
