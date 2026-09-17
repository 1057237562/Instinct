"""A bounded, fully checked bilingual elementary-reasoning curriculum.

No external teacher answers enter this dataset. Each record has explicit input
parameters and an independently computed reference, with finite-domain checks.
"""
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import orjson
from datasets import load_dataset  # noqa: F401
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from dataset.lm_dataset import _create_chat_prompt
OUT=ROOT/'dataset/reasoning_sft_strict'
SEED=2026091518
KINDS=('inventory','quantity_chain','discount','unit_conversion','rectangle',
       'prefix_balance','rolling_max','filter_conditions','dedup_contrast',
       'median','digit_product','frequency','bracket_depth','sort_pairs','extract_json')


def reference(kind, p):
    """Checker uses separate operations from answer construction below."""
    if kind=='inventory': return sum([p['start'],-p['used'],-p['given'],p['added']])
    if kind=='quantity_chain':
        values=[p['a']];values.append(sum([values[0],p['more']]));values.append(values[1]-p['less']);return sum(values)
    if kind=='discount': return int(Fraction(p['price']*p['quantity']*(100-p['discount']),100)+p['fee'])
    if kind=='unit_conversion': return sum([p['hours']*60,p['minutes']])-p['elapsed']
    if kind=='rectangle': return {'area':p['width']*p['length'],'perimeter':sum([p['width'],p['length']]*2)}
    if kind=='prefix_balance': return [sum(p['values'][:i+1]) for i in range(len(p['values']))]
    if kind=='rolling_max': return [max(p['values'][:i+1]) for i in range(len(p['values']))]
    if kind=='filter_conditions': return sorted(v for v in p['values'] if v>p['limit'] and v%2==0)
    if kind=='dedup_contrast':
        if p['mode']=='once':return [v for v in p['values'] if p['values'].count(v)==1]
        result=[]
        for v in p['values']:
            if v not in result:result.append(v)
        return result
    if kind=='median':
        values=sorted(p['values']);middle=len(values)//2
        return str(Fraction(values[middle])) if len(values)%2 else str(Fraction(values[middle-1]+values[middle],2))
    if kind=='digit_product':return int(str(abs(p['a']))[-1])*int(str(abs(p['b']))[-1])
    if kind=='frequency':return dict(sorted(Counter(p['words']).items()))
    if kind=='bracket_depth':
        level=0;high=0
        for c in p['text']:
            level+=1 if c=='(' else -1
            assert level>=0
            high=max(high,level)
        assert level==0
        return high
    if kind=='sort_pairs':return sorted(p['pairs'],key=lambda pair:(pair[1],pair[0]))
    if kind=='extract_json':return {'name':p['name'],'count':p['count']}
    raise ValueError(kind)


def make(kind, rng, i, zh, mode='once'):
    # Each branch constructs the answer before reference() is used for validation.
    if kind=='inventory':
        start=rng.randint(5,200);used=rng.randint(0,start);given=rng.randint(0,start-used);added=rng.randint(0,50)
        p=dict(start=start,used=used,given=given,added=added)
        q=f'原有{start}个物品，用掉{used}个，送出{given}个，再买入{added}个。现在有多少个？' if zh else f'There were {start} items. {used} were used, {given} given away, and {added} added. How many remain?'
        value=((start-used)-given)+added;trace=f'{start}-{used}-{given}+{added}={value}'
    elif kind=='quantity_chain':
        a=rng.randint(0,100);more=rng.randint(0,50);less=rng.randint(0,a+more);b=a+more;c=b-less
        p=dict(a=a,more=more,less=less);value=a+b+c;trace=f'B={a}+{more}={b}\nC={b}-{less}={c}\n{a}+{b}+{c}={value}'
        q=f'甲有{a}个，乙比甲多{more}个，丙比乙少{less}个。三者共有多少个？' if zh else f'A has {a} items. B has {more} more than A. C has {less} fewer than B. How many items do they have altogether?'
    elif kind=='discount':
        price=rng.randint(1,50)*20;n=rng.randint(1,12);discount=rng.choice([0,5,10,15,20,25,50]);fee=rng.randint(0,20)
        p=dict(price=price,quantity=n,discount=discount,fee=fee);subtotal=price*n;saved=subtotal*discount//100;value=subtotal-saved+fee
        trace=f'{price}*{n}={subtotal}\n{subtotal}*{discount}/100={saved}\n{subtotal}-{saved}+{fee}={value}'
        q=f'每件{price}元，买{n}件。商品总价减少{discount}%，然后另加不打折的运费{fee}元。总共付多少元？' if zh else f'Buy {n} items at {price} dollars each. Reduce the merchandise total by {discount}%, then add an undiscounted shipping fee of {fee} dollars. What is the total payment?'
    elif kind=='unit_conversion':
        h=rng.randint(0,9);m=rng.randint(0,59);elapsed=rng.randint(0,h*60+m)
        p=dict(hours=h,minutes=m,elapsed=elapsed);total=h*60+m;value=total-elapsed;trace=f'{h}*60+{m}={total}\n{total}-{elapsed}={value}'
        q=f'活动总长{h}小时{m}分钟，已经过{elapsed}分钟，还剩多少分钟？' if zh else f'An activity lasts {h} hours and {m} minutes. After {elapsed} minutes, how many minutes remain?'
    elif kind=='rectangle':
        length=rng.randint(1,100);width=rng.randint(1,100);p=dict(length=length,width=width)
        q=f'矩形长{length}厘米、宽{width}厘米。返回数值字段 area（平方厘米）和 perimeter（厘米）。' if zh else f'A rectangle has length {length} cm and width {width} cm. Return numeric fields area (square cm) and perimeter (cm).'
        value={'area':length*width,'perimeter':2*(length+width)}
    elif kind in ('prefix_balance','rolling_max','filter_conditions','dedup_contrast','median'):
        xs=[rng.randint(-15,30) for _ in range(rng.randint(1 if kind=='median' else 0,9))];p={'values':xs}
        if kind=='prefix_balance':
            value=[];current=0
            for x in xs:current+=x;value.append(current)
            q=f'余额初始为0，依次变动{xs}。返回每次变动后的余额列表。空输入返回空列表。' if zh else f'Start with balance 0 and apply changes {xs} in order. Return the balance after each change. For empty input return an empty list.'
        elif kind=='rolling_max':
            value=[];largest=None
            for x in xs:largest=x if largest is None or x>largest else largest;value.append(largest)
            q=f'输入{xs}。返回每个位置及之前所有数的最大值，包含第一个位置。空输入返回空列表。' if zh else f'Given {xs}, return the maximum seen at each position, including the first position. For empty input return an empty list.'
        elif kind=='filter_conditions':
            limit=rng.randint(-15,25);p['limit']=limit;value=[x for x in xs if not(x<=limit or x%2)];value.sort()
            q=f'从{xs}中保留严格大于{limit}且为偶数的元素，升序排列，保留重复次数。' if zh else f'From {xs}, keep values strictly greater than {limit} that are even. Sort ascending and retain duplicate occurrences.'
        elif kind=='dedup_contrast':
            p['mode']=mode
            value=list(dict.fromkeys(xs)) if mode=='first' else [x for x in xs if Counter(xs)[x]==1]
            q=(f'输入{xs}。' if zh else f'Input: {xs}. ')+(('每个值只保留首次出现，保持顺序。' if mode=='first' else '只保留恰好出现一次的值，保持顺序。') if zh else ('Keep the first occurrence of every distinct value, in order.' if mode=='first' else 'Keep only values that occur exactly once, in order.'))
        else:
            ordered=sorted(xs);size=len(xs)
            value=str(ordered[size//2]) if size%2 else str(Fraction(ordered[size//2-1],2)+Fraction(ordered[size//2],2))
            q=f'求非空列表{xs}的中位数。偶数个元素时取中间两数的均值。答案用最简分数（如3/2），整数不带/1。' if zh else f'Find the median of the nonempty list {xs}. For an even count, average the two middle values. Express the answer as a reduced fraction (such as 3/2), or an integer without /1.'
    elif kind=='digit_product':
        a=rng.randint(-9999,9999);b=rng.randint(-9999,9999);p=dict(a=a,b=b);value=(abs(a)%10)*(abs(b)%10);trace=f'{abs(a)%10}*{abs(b)%10}={value}'
        q=f'取{a}和{b}的绝对值的个位数字，再相乘。结果是多少？' if zh else f'Take the units digits of the absolute values of {a} and {b}, then multiply those digits. What is the result?'
    elif kind=='frequency':
        words=[rng.choice(['red','blue','green','gold']) for _ in range(rng.randint(0,12))];p={'words':words};value={}
        for w in words:value[w]=value.get(w,0)+1
        q=f'统计列表{words}中各词出现次数，只返回出现过的词到次数的JSON对象。空输入返回空对象。' if zh else f'Count occurrences of each word in {words}. Return a JSON object mapping only words present to their counts. Empty input gives an empty object.'
    elif kind=='bracket_depth':
        chunks=['('*rng.randint(0,5) for _ in range(rng.randint(0,4))];text=''.join(s+')'*len(s) for s in chunks);p={'text':text}
        value=max([len(s) for s in chunks],default=0);trace=None
        q=f'合法括号串为 {json.dumps(text)}。求最大同时未闭合左括号数量。空串深度为0。' if zh else f'The balanced parenthesis string is {json.dumps(text)}. Find the maximum number of simultaneously unmatched opening parentheses. Empty string has depth 0.'
    elif kind=='sort_pairs':
        pairs=[[rng.randint(-5,9),rng.randint(-5,9)] for _ in range(rng.randint(0,8))];p={'pairs':pairs}
        value=[pair for pair in pairs]
        # Stable two-pass sort, checked against independent tuple-key sorting.
        value.sort(key=lambda pair:pair[0]);value.sort(key=lambda pair:pair[1])
        q=f'对数对列表{pairs}排序：第二个数升序；第二个数相同则第一个数升序。保留重复项。' if zh else f'Sort pairs {pairs} by the second number ascending, breaking ties by the first number ascending. Preserve duplicates.'
    else:
        name=rng.choice(['Mira','Theo','Nora','Iris'])+str(rng.randint(0,9999));count=rng.randint(0,100);color=rng.choice(['red','blue','green'])
        p={'name':name,'count':count};value=dict(name=name,count=count)
        q=f'记录：name={name}; count={count}; color={color}。只提取name和count，count必须为数字。' if zh else f'Record: name={name}; count={count}; color={color}. Extract only name and count; count must be numeric.'
    numeric=kind in ('inventory','quantity_chain','discount','unit_conversion','digit_product')
    if numeric:
        q+=('给出简短算式，最后一行用 #### 数字。' if zh else ' Give a short calculation and finish with #### <number>.')
        answer=trace+'\n#### '+str(value)
    elif kind=='median':q+=('只输出答案。' if zh else ' Output only the answer.');answer=value
    elif kind=='bracket_depth':q+=('只输出整数。' if zh else ' Output only the integer.');answer=str(value)
    else:q+=('只输出JSON，不加代码围栏或说明。' if zh else ' Output only JSON, with no fence or explanation.');answer=json.dumps(value,ensure_ascii=False)
    assert value==reference(kind,p), (kind,p,value,reference(kind,p))
    return {'source':'project_authored_verified_curriculum','kind':kind,'parameters':p,'language':'zh' if zh else 'en',
            'answer_format':'trace_number' if numeric else 'fraction' if kind=='median' else 'integer' if kind=='bracket_depth' else 'json',
            'conversations':[{'role':'user','content':q},{'role':'assistant','content':answer}]}


def validate_record(row):
    value=reference(row['kind'],row['parameters']);answer=row['conversations'][1]['content'];fmt=row['answer_format']
    if fmt=='json':assert json.loads(answer)==value
    elif fmt=='fraction':assert answer==value
    elif fmt=='integer':assert int(answer)==value
    else:
        assert answer.splitlines()[-1]==f'#### {value}'
        # Independently check every displayed numeric equality, including chained equalities.
        from dataset.scripts.finalize_continuation_sft import arithmetic
        for line in answer.splitlines()[:-1]:
            parts=line.split('=')
            if parts[0] in ('B','C'):parts=parts[1:]
            import ast
            values=[arithmetic(ast.parse(part,mode='eval')) for part in parts]
            assert len(values)>=2 and len(set(values))==1


def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];seen=set()
    for kind in KINDS:
        for i in range(1200):
            # All language/contrast variants of the same inputs stay in one split.
            seed=SEED+KINDS.index(kind)*100000+i
            group=f'{kind}/{i}'
            for zh in (True,False):
                for mode in (('first','once') if kind=='dedup_contrast' else ('once',)):
                    row=make(kind,random.Random(seed),i,zh,mode)
                    ph=hashlib.sha256(row['conversations'][0]['content'].encode()).hexdigest()
                    if ph in seen:continue
                    seen.add(ph);row['group_id']=group
                    # Split by input parameters, so duplicate inputs can't cross splits.
                    base={k:v for k,v in row['parameters'].items() if k!='mode'}
                    split_hash=hashlib.sha256((kind+json.dumps(base,sort_keys=True)).encode()).hexdigest()
                    row['split']='validation' if int(split_hash[:8],16)%20==0 else 'train'
                    row['input_hash']=split_hash;validate_record(row);rows.append(row)
    tok=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    for row in rows:
        row['tokens']=len(tok.backend_tokenizer.encode(_create_chat_prompt(tok,row['conversations']),add_special_tokens=False).ids)
        assert row['tokens']<=2944
    rng=random.Random(SEED);train=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation'];rng.shuffle(train);rng.shuffle(validation)
    # Pilot uses complete semantic input groups, not arbitrary individual rows.
    units={}
    for row in train:units.setdefault(row['input_hash'],[]).append(row)
    pilot=[];count=0
    for unit in units.values():
        size=sum(r['tokens'] for r in unit)
        if count+size<=1000000:pilot.extend(unit);count+=size
    manifests={}
    with gzip.open(OUT/'verification_records.jsonl.gz','wt',encoding='utf-8') as metadata:
        for split,data in [('train',train),('validation',validation),('pilot_train',pilot)]:
            h=hashlib.sha256()
            with (OUT/f'{split}.jsonl').open('wb') as f:
                for i,row in enumerate(data):
                    validate_record(row);line=orjson.dumps({'conversations':row['conversations']})+b'\n';f.write(line);h.update(line)
                    metadata.write(json.dumps({**{k:v for k,v in row.items() if k!='conversations'},'file_split':split,'row':i},ensure_ascii=False)+'\n')
            manifests[split]={'rows':len(data),'tokens':sum(r['tokens'] for r in data),'sha256':h.hexdigest()}
    report={'seed':SEED,'verification':'Every record passed independent reference and displayed arithmetic/format checks, twice; not sampled.',
        'families':list(KINDS),'train_counts':dict(Counter(r['kind'] for r in train)),'max_tokens':max(r['tokens'] for r in rows),'files':manifests,
        'external_teacher_records':0,'limitations':['This is a bounded elementary curriculum, not comprehensive reasoning knowledge.',
          'Both splits share task families but use disjoint parameter inputs; a separate unseen-template evaluation is still needed.',
          'No unreviewed external SFT or old dialogue replay is in the default strict curriculum; monitor retention when training.',
          'Mathematical references are checked for every generated instance; this does not prove model improvement.']}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
