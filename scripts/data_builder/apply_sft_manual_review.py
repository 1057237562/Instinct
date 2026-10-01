"""Publish a conservative continuation mix after the recorded stratified review.

Preserves the first mixed dataset and audit snapshot. No model training is run.
"""
from collections import Counter, defaultdict
import ast
import builtins
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from scripts.data_loader.lm_dataset import _create_chat_prompt
OUT = ROOT / 'dataset/quality_python_sft'

REVIEWS = [
 ('keep','字符反转符合规格，大小写和空串测试覆盖，适合切片基础。'),
 ('reject','承诺任意 iterable 及空输入返回 None，但空迭代器触发 max 的 ValueError；测试仅列表。'),
 ('keep','双向字典同时约束字符与单词的一一对应，符合类型与契约补强目标。'),
 ('reject','BFS 实现可用，但尾部示例注释把实际可达的 4→5 写成 False；默认混合先移除。'),
 ('keep','归并排序与 O(n log n) 要求一致，可用于两段合并与边界；难度稍高。'),
 ('keep','函数名 power_of_two 有歧义，但明确文档要求 square，n*n 正确；可用于按规格而非名字作答。'),
 ('keep','最小值循环及空输入异常正确；上游异常测试欠严格，不作为完整证明。'),
 ('keep','最大值、初始状态及负数处理正确，注意与同类题的语义重复。'),
 ('keep','先判断双重整除，再处理单一条件，适合条件组合。'),
 ('keep','连续段计数和末尾补更新正确，直接对应 rolling/state 短板。'),
 ('reject','绕 y 轴反射应取反 x；实现取反 y。已用 (2,3,True) 复现。'),
 ('reject','文字要求 None 留原索引，代码将其随列表反转；中间 None 示例掩盖冲突。'),
 ('reject','题目说多元素返回原列表，示例却要求嵌套列表，题意与测试冲突。'),
 ('caution','开头声称返回字典，后文允许列表/原始类型；代码按后文实现但契约表达含混。'),
 ('caution','默认所有字典键都是字符串，规格未声明；非字符串键会触发 TypeError。'),
 ('reject','题目同时要求冲突字符串改名和已经存在时不改名，自相矛盾。'),
 ('keep','按 situation_number 类型筛选并返回元组/None，符合主要规格；bool 的整数子类语义可另注明。'),
 ('keep','双循环枚举不同索引对并格式化，符合给定规格与示例。'),
 ('reject','consecutive 与示例 (2,7) 含混；程序对 [1,2,3,4],7 返回 None，漏掉 (3,4)。'),
 ('reject','casefold 比较代码正确，但题目最后一个 assert 期望 1，实际应 -1。'),
 ('keep','折扣只作用于课费，最后加琴价；数量绑定与多步算术正确。'),
 ('cleanup','计算正确，答案把 $20 写成 %20；只修这个已核实的符号错误。'),
 ('keep','货币单位换算、剩余金额与硬币数量正确。'),
 ('cleanup','算式和终答正确，末句把计划支出的 10% 误称储蓄账户的 10%；修正文案。'),
 ('keep','面积乘单位综合价格，单位处理正确且简洁。'),
 ('keep','明确区分第二月利息与本金加利息，复利状态更新正确。'),
 ('keep','先统一货币单位再扣除价格，符合题意。'),
 ('reject','默认每错一题扣一分，但题目未给每题分值；保守移出契约训练。'),
 ('keep','接受率与入学率串联，算术正确。'),
 ('keep','按月翻倍四次，July 产量计算正确。'),
 ('keep','可作为普通解释对话回放；tools=[] 不代表真实工具调用。'),
 ('keep','通用开发成本说明基本响应问题，但较长，非本轮能力强化重点。'),
 ('reject','用户没有提供对话正文，答案却编造分析；reasoning_content 也显示知道缺信息。'),
 ('keep','对未指定的 PDE 给条件化方法概述，适合作少量知识对话回放。'),
 ('reject','邮件补入原输入没有的具体人物、视频细节，且作品先后关系不稳定；不用于本轮回放。'),
 ('reject','用户明确提英文拼写，输出整段中文，未满足语言要求。'),
 ('caution','输入有明显噪声，答案将其包装为完整深刻含义；不利于学习缺信息时澄清。'),
 ('reject','诗歌有突兀英文、重复残句和无关冗长思考，不适合本轮。'),
 ('cleanup','澄清缺少待翻译正文是合理的，但附带冗长 speculative reasoning；当前回放策略移除非空思考记录。'),
 ('caution','问题术语不清，答案给出无来源的宽泛理论；难以核实，暂不回放。'),
]


def structural_key(code):
    tree = ast.parse(code)
    imports = {a.asname or a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names}
    protected = set(dir(builtins)) | imports
    mapping = {}
    def mapped(name):
        if name in protected: return name
        if name not in mapping: mapping[name] = f'v{len(mapping)}'
        return mapping[name]
    # Normalize bound identifier spelling only; preserve operators, attributes,
    # constants and library names. This detects renamed algorithm clones.
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.Module)):
            node.body = [n for n in node.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        if isinstance(node,ast.FunctionDef): node.name = mapped(node.name)
        if isinstance(node,ast.Name): node.id = mapped(node.id)
        if isinstance(node,ast.arg): node.arg = mapped(node.arg)
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def main():
    samples = json.loads((OUT/'manual_review_sample.json').read_text(encoding='utf-8'))
    assert len(samples)==len(REVIEWS)==40
    decisions = [{'sample_index':i,'source':r['source'],'source_id':r.get('source_id'), 'old_row':r.get('old_row'),
                  'decision':REVIEWS[i][0],'reason':REVIEWS[i][1]} for i,r in enumerate(samples)]
    (OUT/'manual_review_decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2),encoding='utf-8')
    # Reload the whole curated source inventory; the initial mixture is unchanged.
    source_chats = {}
    for source in ('kodcode','bigcode','gsm8k_train'):
        for split in ('train','validation'):
            source_chats[source,split] = [orjson.loads(line)['conversations'] for line in (OUT/f'{source}_{split}.jsonl').open('rb')]
    offsets = Counter(); records=[]
    with gzip.open(OUT/'all_curated_sources_metadata.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);key=(r['source'],r['split']);r['conversations']=source_chats[key][offsets[key]];offsets[key]+=1;records.append(r)
    old_chat=[orjson.loads(line)['conversations'] for line in (OUT/'continue_sft_train.jsonl').open('rb')]
    replay=[]
    with gzip.open(OUT/'provenance_and_tests.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['split']=='train' and r['source']=='prior_sft_replay':r['conversations']=old_chat[r['row']];replay.append(r)
    rejects=Counter();code=[];math=[];quarantine=[];seen_shapes=set()
    known_bad_groups={'kodcode:Prefill_38335','kodcode:Algorithm_25486'}
    # Prioritize validation while dropping renamed implementation duplicates, so
    # a matching train clone cannot leak the held-out implementation.
    records.sort(key=lambda r:(r['split']!='validation',r['source'],str(r['source_id'])))
    for r in records:
        reason=None
        if r['source']=='bigcode':reason='bigcode_pending_semantic_review'
        elif r['group_id'] in known_bad_groups:reason='manual_known_problem_group'
        elif r['source']=='gsm8k_train' and r['source_id']=='1800':reason='manual_underspecified_scoring'
        elif r['source']=='kodcode':
            shape=structural_key(r['conversations'][1]['content'][10:-4])
            if shape in seen_shapes:reason='renamed_implementation_duplicate'
            else:seen_shapes.add(shape)
        if reason:
            rejects[reason]+=1;quarantine.append({'source':r['source'],'source_id':r['source_id'],'group_id':r['group_id'],'reason':reason});continue
        if r['source']=='gsm8k_train':
            if r['source_id']=='1674':
                r['conversations'][1]['content']=r['conversations'][1]['content'].replace('%20*50','$20*50')
                r['manual_edit']='Corrected %20 to $20; arithmetic unchanged'
            if r['source_id']=='5260':
                r['conversations'][1]['content']=r['conversations'][1]['content'].replace('10% of her savings account is','10% of her total planned spending is')
                r['manual_edit']='Corrected description of savings calculation base; arithmetic unchanged'
            math.append(r)
        else:code.append(r)
    bad_old={r['old_row'] for r,d in zip(samples,REVIEWS) if r['source']=='prior_sft_replay' and d[0] in ('reject','caution')}
    clean_replay=[]
    for r in replay:
        assistant=[m for m in r['conversations'] if m['role']=='assistant']
        reason=None
        if r['old_row'] in bad_old:reason='manual_bad_replay'
        elif any(m.get('reasoning_content') for m in r['conversations']):reason='replay_nonempty_reasoning'
        elif any(len(m.get('content',''))>800 for m in assistant):reason='replay_long_answer'
        elif any(max(Counter(line.strip() for line in m.get('content','').splitlines() if len(line.strip())>15).values(),default=0)>=3 for m in assistant):reason='replay_repetition'
        if reason:rejects[reason]+=1
        else:clean_replay.append(r)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    for r in code+math+clean_replay:r['tokens']=len(tokenizer.backend_tokenizer.encode(_create_chat_prompt(tokenizer,r['conversations']),add_special_tokens=False).ids)
    rng=random.Random(2026091502);rng.shuffle(code);rng.shuffle(math);rng.shuffle(clean_replay)
    code_train=[r for r in code if r['split']=='train'];math_train=[r for r in math if r['split']=='train']
    code_budget=min(sum(r['tokens'] for r in code_train),3.5*sum(r['tokens'] for r in math_train),7*sum(r['tokens'] for r in clean_replay))
    def take(rows,budget):
        result=[];n=0
        for r in rows:
            if n+r['tokens']<=budget:result.append(r);n+=r['tokens']
        return result,n
    chosen_code,n=take(code_train,code_budget);chosen_math,_=take(math_train,n*2/7);chosen_replay,_=take(clean_replay,n/7)
    train=chosen_code+chosen_math+chosen_replay;rng.shuffle(train)
    validation=[r for r in code+math if r['split']=='validation'];rng.shuffle(validation)
    with gzip.open(OUT/'reviewed_provenance_and_tests.jsonl.gz','wt',encoding='utf-8') as provenance:
        for split,rows in [('train',train),('validation',validation)]:
            with (OUT/f'continue_sft_reviewed_{split}.jsonl').open('wb') as f:
                for i,r in enumerate(rows):
                    f.write(orjson.dumps({'conversations':r['conversations']})+b'\n')
                    provenance.write(json.dumps({**{k:v for k,v in r.items() if k!='conversations'},'split':split,'row':i},ensure_ascii=False)+'\n')
    (OUT/'review_quarantine_manifest.json').write_text(json.dumps(quarantine,ensure_ascii=False,indent=2),encoding='utf-8')
    source_tokens={s:sum(r['tokens'] for r in train if r['source']==s) for s in {r['source'] for r in train}}
    report={'manual_sample_seed':2026091501,'manual_sample_rows':40,'sampling':'10 uniform random rows per source from initial training mix; stratified, not population-weighted',
        'review_summary':{s:dict(Counter(r['decision'] for r in decisions if r['source']==s)) for s in {r['source'] for r in decisions}},
        'target_token_mix':'70% KodCode short functions / 20% GSM8K train / 10% filtered original replay; experimental',
        'excluded_counts':dict(rejects),'train_rows':len(train),'validation_rows':len(validation),'tokens':sum(source_tokens.values()),
        'source_rows':dict(Counter(r['source'] for r in train)),'source_tokens':source_tokens,'source_token_fractions':{s:v/sum(source_tokens.values()) for s,v in source_tokens.items()},
        'max_tokens':max(r['tokens'] for r in train+validation),
        'limitations':['40-row sample is not a certification of the full corpus or an unbiased overall error-rate estimate.',
                       'Unreviewed KodCode/replay records may still contain semantic mistakes; 5 manually inspected code examples were executed only to reproduce specific counterexamples.',
                       'BigCode is preserved separately, not deleted. Initial 60/20/20 mix is preserved for audit; prefer reviewed output.',
                       'The mixture remains non-commercial due to KodCode CC-BY-NC-4.0.']}
    (OUT/'manual_review_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
