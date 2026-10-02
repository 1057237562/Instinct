"""Build a small reasoning-oriented continued SFT experiment; never train a model."""
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import orjson
import pyarrow.parquet as pq
from datasets import load_dataset  # noqa: F401
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from scripts.data_loader.lm_dataset import _create_chat_prompt
from scripts.data_builder.finalize_continuation_sft import arithmetic
from scripts.data_builder.build_quality_python_sft import fingerprint, norm

OUT=ROOT/'dataset/reasoning_sft_v2'
OLD=ROOT/'dataset/quality_python_sft'
SEED=2026091511
APPROVED_ORCA={'198825':'5','59121':'1','153741':'48','145942':'47','3605':'2',
               '153361':'27','59798':'5265','68706':'5130','198835':'199'}


def rule_rows():
    rng=random.Random(SEED);rows=[]
    for i in range(5000):
        a,b,c=rng.sample(range(2,95),3);kind=i%5;zh=i%2==0
        if kind==0:
            # Both halves of a contrast pair share a split and input.
            for unique in (True,False):
                xs=[a,b,b,c]
                instruction=('只保留恰好出现一次的元素，保持顺序。' if unique else '每个值只保留第一次出现，保持顺序。') if zh else ('Keep only values occurring exactly once, preserving order.' if unique else 'Keep the first occurrence of each value, preserving order.')
                q=(f'输入列表：{xs}。' if zh else f'Input list: {xs}. ')+instruction+('只输出 JSON 数组。' if zh else ' Output only a JSON array.')
                answer=json.dumps([a,c] if unique else [a,b,c])
                rows.append({'group_id':f'rules/{kind}/{a}/{b}/{c}','task':'contrast_filter','conversations':[{'role':'user','content':q},{'role':'assistant','content':answer}]})
            continue
        if kind==1:
            q=(f'账户余额从0开始，依次变动 {a}、{-b}、{c}。给出每次变动后的余额。只输出JSON数组。' if zh else f'Start with balance 0. Apply changes {a}, {-b}, {c} in order. Return only a JSON array of the balance after each change.')
            answer=json.dumps([a,a-b,a-b+c]);task='state_updates'
        elif kind==2:
            q=(f'甲有{a}个物品，乙比甲多{b}个，丙比乙少{c}个。三人一共有多少个？写出简短算式，最后一行用 #### 数字。' if zh else f'A has {a} items. B has {b} more than A. C has {c} fewer than B. How many items do they have in total? Give a short calculation and end with #### <number>.')
            if a+b<c:continue
            answer=f'B = {a} + {b} = {a+b}\nC = {a+b} - {c} = {a+b-c}\n{a} + {a+b} + {a+b-c} = {3*a+2*b-c}\n#### {3*a+2*b-c}';task='quantity_binding'
        elif kind==3:
            q=(f'矩形长{a}厘米、宽{b}厘米。只输出一个JSON对象，字段为 area 和 perimeter，值为数字。' if zh else f'A rectangle is {a} cm long and {b} cm wide. Output only a JSON object with numeric fields area and perimeter.')
            answer=json.dumps({'area':a*b,'perimeter':2*(a+b)});task='units_and_json'
        else:
            xs=[a,b,c];limit=rng.randrange(5,85)
            q=(f'从{xs}中筛选严格大于{limit}的偶数，升序排列。只输出JSON数组。' if zh else f'From {xs}, select even values strictly greater than {limit}, sorted ascending. Output only a JSON array.')
            answer=json.dumps(sorted(x for x in xs if x>limit and x%2==0));task='combined_conditions'
        rows.append({'group_id':f'rules/{kind}/{a}/{b}/{c}','task':task,'conversations':[{'role':'user','content':q},{'role':'assistant','content':answer}]})
    # These synthetic rules are a limited curriculum, not an open-ended reasoning benchmark.
    return rows


def selection_units(rows):
    """Keep the two opposing instructions of a rule pair in the same budget slice."""
    grouped={}
    for i,row in enumerate(rows):
        key=row['group_id'] if row['source']=='verified_rules' else f'row:{i}'
        grouped.setdefault(key,[]).append(row)
    return list(grouped.values())


def main():
    OUT.mkdir(parents=True,exist_ok=True);stats=Counter();rng=random.Random(SEED)
    tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    def token_count(messages):return len(tok.backend_tokenizer.encode(_create_chat_prompt(tok,messages),add_special_tokens=False).ids)
    records=[];chats={}
    for split in ('train','validation'):
        chats[split]=[orjson.loads(line)['conversations'] for line in (OLD/f'continue_sft_reviewed_{split}.jsonl').open('rb')]
    with gzip.open(OLD/'reviewed_provenance_and_tests.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if (r['source']=='gsm8k_train' and str(r.get('source_id')) in {'7450','6369'}) or (r['source']=='kodcode' and str(r.get('source_id')) in {'Data_Structure_4950_C','Data_Structure_4950_I'}):
                stats['luna_review_caution_excluded']+=1
                continue
            r['conversations']=chats[r['split']][r['row']];records.append(r)
    grams=set()
    tests=[r['question'] for r in map(orjson.loads,(ROOT/'dataset/gsm8k/test.jsonl').open('rb'))]
    with gzip.open(ROOT/'dataset/humaneval/HumanEval.jsonl.gz','rt',encoding='utf-8') as f:tests += [json.loads(line)['prompt'] for line in f]
    for text in tests:
        ws=norm(text).split();grams.update(tuple(ws[i:i+13]) for i in range(len(ws)-12))
    def overlap(text):
        ws=norm(text).split();return any(tuple(ws[i:i+13]) in grams for i in range(len(ws)-12))
    old_questions={fingerprint(m['content']) for r in records for m in r['conversations'] if m['role']=='user'}
    # Include raw official GSM8K train questions, so Orca does not duplicate their rephrased wrappers exactly.
    old_questions.update(fingerprint(r['question']) for r in map(orjson.loads,(OLD/'raw/gsm8k/train.jsonl').open('rb')))
    number=r'-?\d+(?:\.\d+)?'
    equation=re.compile(r'^\s*\$?('+number+r'(?:\s*[+*/-]\s*\$?'+number+r')+)\s*=\s*\$?('+number+r')(?:\s+[A-Za-z][A-Za-z /]*)?[.!]?\s*$')
    orca=[];seen=set()
    for batch in pq.ParquetFile(OUT/'raw/orca_math/train-00000-of-00001.parquet').iter_batches(batch_size=2048):
        for r in batch.to_pylist():
            row_id=stats['orca_scanned'];stats['orca_scanned']+=1;q=r['question'].strip();a=r['answer'].strip()
            if not 30<=len(q)<=800 or not 100<=len(a)<=1400 or re.search(r'permutation|probability|secretary|chairman|president|diagram|shown|figure',q,re.I):stats['length_or_scope']+=1;continue
            if overlap(q) or re.search(r'human.?eval|gsm8k|livecodebench',q,re.I):stats['benchmark_overlap']+=1;continue
            h=fingerprint(q)
            if h in old_questions or h in seen:stats['duplicate']+=1;continue
            count=0;bad=False
            for line in a.replace(',','').replace('×','*').splitlines():
                m=equation.fullmatch(line)
                if not m:continue
                try:
                    val=arithmetic(ast.parse(m[1].replace('$',''),mode='eval'));gold=arithmetic(ast.parse(m[2],mode='eval'))
                    if abs(val-gold)>0.01:bad=True;break
                    count+=1
                except (ValueError,SyntaxError,ZeroDivisionError):continue
            if bad or count<2:stats['insufficient_checked_arithmetic']+=1;continue
            # A currency abbreviation such as Rs. is not a sentence boundary.
            last=[line.strip() for line in a.splitlines() if line.strip()][-1]
            numbers=re.findall(r'-?\d[\d,]*(?:\.\d+)?',last)
            if len(numbers)!=1 and str(row_id) not in APPROVED_ORCA:stats['ambiguous_final_number']+=1;continue
            # This is an explicit source sentence, not an arbitrary intermediate number.
            final=APPROVED_ORCA.get(str(row_id),numbers[0].replace(',','').rstrip('.') if numbers else '')
            messages=[{'role':'user','content':q+'\n\nGive a concise calculation and end with #### <number>.'},
                      {'role':'assistant','content':a+'\n#### '+final}]
            n=token_count(messages)
            if n>1800:stats['overlength']+=1;continue
            seen.add(h);orca.append({'source':'orca_math','source_id':str(row_id),'group_id':'orca:'+h,'prompt_hash':h,
                'conversations':messages,'tokens':n,'checked_equations':count,'license':'MIT',
                'verification':'synthetic upstream; locally checked explicit numeric equations only',
                'split':'validation' if int(h[:8],16)%20==0 else 'train'})
    print('Orca candidates before prior-SFT check:',len(orca),dict(stats),flush=True)
    wanted={r['prompt_hash'] for r in orca};seen_old=set()
    for line in (ROOT/'dataset/sft_t2t_mini.jsonl').open('rb'):
        for m in orjson.loads(line)['conversations']:
            if m.get('role')=='user':
                h=fingerprint(str(m.get('content','')))
                if h in wanted:seen_old.add(h)
    # Training overlap is recorded and excluded for new collection; unseen is not guaranteed semantically.
    orca=[r for r in orca if r['prompt_hash'] not in seen_old];stats['prior_sft_exact_questions_removed']=len(seen_old)
    with (OUT/'orca_candidates_not_for_training.jsonl').open('wb') as f:
        for r in orca:f.write(orjson.dumps(r)+b'\n')
    stats['orca_unreviewed_excluded_from_mix']=sum(r['source_id'] not in APPROVED_ORCA for r in orca)
    orca=[r for r in orca if r['source_id'] in APPROVED_ORCA]
    for r in orca:r['verification']='individually read and arithmetic reviewed; final number manually checked'
    records.extend(orca)
    rule_seen=set()
    for r in rule_rows():
        h=fingerprint(r['conversations'][0]['content'])
        if h in rule_seen or overlap(r['conversations'][0]['content']):continue
        rule_seen.add(h);r.update(source='verified_rules',tokens=token_count(r['conversations']),license='project-authored',
            verification='deterministic reference transformation; bounded template curriculum',
            split='validation' if int(hashlib.sha256(r['group_id'].encode()).hexdigest()[:8],16)%20==0 else 'train')
        records.append(r)
    groups=defaultdict(list)
    for r in records:
        bucket='math' if r['source'] in ('gsm8k_train','orca_math') else 'code' if r['source']=='kodcode' else 'rules' if r['source']=='verified_rules' else 'replay'
        r['bucket']=bucket
        if r['split']=='train':groups[bucket].append(r)
    for group in groups.values():rng.shuffle(group)
    ratios={'math':.5,'code':.3,'rules':.1,'replay':.1}
    budget=min(sum(r['tokens'] for r in groups[k])/v for k,v in ratios.items())
    selected=[]
    for k,fraction in ratios.items():
        n=0
        for unit in selection_units(groups[k]):
            size=sum(r['tokens'] for r in unit)
            if n+size<=budget*fraction:selected.extend(unit);n+=size
    rng.shuffle(selected);validation=[r for r in records if r['split']=='validation'];rng.shuffle(validation)
    def write(name,items):
        with (OUT/name).open('wb') as f:
            for r in items:f.write(orjson.dumps({'conversations':r['conversations']})+b'\n')
    write('train.jsonl',selected);write('validation.jsonl',validation)
    # Representative mini-run, fresh shuffle, approximately one million tokens.
    pilot=[];n=0
    for unit in selection_units(selected):
        size=sum(r['tokens'] for r in unit)
        if n+size<=1000000:pilot.extend(unit);n+=size
    write('pilot_train.jsonl',pilot)
    for source in ('orca_math','verified_rules'):
        for split in ('train','validation'):write(f'{source}_{split}.jsonl',[r for r in records if r['source']==source and r['split']==split])
    with gzip.open(OUT/'provenance.jsonl.gz','wt',encoding='utf-8') as f:
        for split,items in [('train',selected),('validation',validation),('pilot',pilot)]:
            for i,r in enumerate(items):f.write(json.dumps({**{k:v for k,v in r.items() if k!='conversations'},'split':split,'row':i},ensure_ascii=False)+'\n')
    report={'seed':SEED,'orca_filter_counts':dict(stats),'source_rows':dict(Counter(r['source'] for r in selected)),
        'bucket_tokens':{k:sum(r['tokens'] for r in selected if r['bucket']==k) for k in ratios},
        'tokens':sum(r['tokens'] for r in selected),'train_rows':len(selected),'validation_rows':len(validation),'pilot_rows':len(pilot),'pilot_tokens':n,
        'limitations':['Orca equations/terminal number checks do not prove semantic correctness; manually review selected samples.',
            'Explicit instruction data are project-authored templates, not evidence of open-ended reasoning mastery.',
            'HumanEval/GSM8K-test 13-word overlap screening and exact old-SFT question checks are not semantic decontamination proofs.',
            'Pretrain overlap for inherited sources follows prior audit; newly added Orca is not exhaustively checked against wrapped pretrain text.',
            'Contains KodCode CC-BY-NC-4.0; do not assume the complete mix is commercially licensed.']}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    review=[]
    for source in ('orca_math','verified_rules'):
        pool=[r for r in selected if r['source']==source]
        review.extend(rng.sample(pool,min(12,len(pool))))
    (OUT/'review_sample.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
