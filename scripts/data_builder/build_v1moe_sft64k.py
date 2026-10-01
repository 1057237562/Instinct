"""Reproducible, non-destructive V1MoE chat/code/instruction/64K SFT build.

Stages: prepare -> triage (CLI) -> finalize -> verify.
Identity candidates are quarantined for review, not automatically declared bad.
Raw source files are retained. No training or external code execution occurs.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import datasets  # before transformers/torch: Windows DLL ordering
import orjson
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data_loader.lm_dataset import _create_chat_prompt
from scripts.data_builder.filter_anomaly_candidates import iter_file_rows, sha256_file
from scripts.data_builder.audit_identity_contamination import classify_row

OUT = ROOT / 'dataset/v1moe_sft64k'
SEED = 20260927
LIMIT = 65536
RESERVE = 128
SYSTEM = '你是 InstinctV1Moe，一个从头训练的语言模型。你没有主观意识或个人经历。根据对话上下文遵循用户要求；不确定时说明不确定。'
IDENTITY = re.compile(r'你是谁|你是.{0,8}(模型|助手|AI)|你的.{0,8}(身份|模型|开发|训练数据|名字)|自我介绍|主观意识|个人经历|who are you|what.{0,12}model are you|your (?:identity|name|creator|training data)|are you (?:sentient|conscious)', re.I)
SELF = re.compile(r'我是|我由|我的(?:名字|身份|模型|开发者)|i am|i.m|my (?:name|model)|developed by|created by', re.I)
FOREIGN = re.compile(r'qwen|通义|千问|alibaba|阿里|chatgpt|openai|claude|anthropic|gemini|deepseek|llama|mistral|文心|豆包|kimi|智谱|grok|星火', re.I)
PERSONAL = re.compile(r'我(?:的童年|小时候|出生于|毕业于|曾经在.{0,12}工作|拥有主观意识|有自己的意识)|I (?:was born|grew up|graduated from|have personal experiences|am conscious|am sentient)', re.I)
LEGACY_IDENTITY = re.compile(r'instinct|l1bra|知识截止|知识更新.{0,6}(截至|截止)|my knowledge (?:cutoff|cut.off)|as of my last (?:knowledge|update)',re.I)


def dump(path, value):
    path.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2))


def emit(stream, value):
    stream.write(orjson.dumps(value) + b'\n')


def norm(s):
    return ' '.join(s.casefold().split())


def key(row):
    return hashlib.sha256(norm('\n'.join(m['content'] for m in row['conversations'] if m['role'] == 'user')).encode()).hexdigest()


def group_key(row):
    return hashlib.sha256(norm(next(m['content'] for m in row['conversations'] if m['role']=='user')).encode()).hexdigest()


def normalize(raw):
    conv = raw.get('conversations', raw.get('messages'))
    if conv is None and 'instruction' in raw:
        conv = [{'role':'user','content':raw['instruction']}, {'role':'assistant','content':raw.get('response',raw.get('output',''))}]
    if not isinstance(conv, list) or not conv:
        raise ValueError('conversation')
    result = []
    for m in conv:
        role = m.get('role', m.get('from'))
        role = {'human':'user','gpt':'assistant'}.get(role,role)
        content = m.get('content',m.get('value',''))
        if role not in ('user','assistant','system','tool') or not isinstance(content,str):
            raise ValueError('message')
        out = dict(m)
        out.pop('from',None); out.pop('value',None)
        out['role'],out['content'] = role,content
        # Visible-answer recipe: remove teacher private reasoning rather than
        # training verbose identity-contaminated traces. Logged per row.
        out.pop('reasoning_content',None)
        if not content.strip() and not out.get('tool_calls'):
            raise ValueError('empty')
        result.append(out)
    if result[-1]['role'] != 'assistant' or not any(m['role']=='user' for m in result):
        raise ValueError('incomplete')
    if any(x in '\n'.join(m['content'] for m in result) for x in ('<|im_start|>','<|im_end|>','�')):
        raise ValueError('control_token')
    return {'conversations': result}


def sources():
    d=ROOT/'dataset'
    return [
        ('t2t', [d/'sft_t2t_mini.jsonl'], 45000, 'local:sft_t2t_mini', None),
        ('python_reviewed', [d/'quality_python_sft/continue_sft_reviewed_train.jsonl'], 14802, 'local:quality_python_sft; upstream KodCode/KodCode-V1-SFT-4o + GSM8K + T2T', 'see quality_python_sft/README.md and reviewed_provenance_and_tests.jsonl.gz'),
        ('magicoder', [d/'magicoder-110k/data-evol_instruct-decontaminated.jsonl'], 30000, 'ise-uiuc/Magicoder-Evol-Instruct-110K', None),
        ('short_instruction', [d/'instruction_understanding_sft/train.jsonl'], 18000, 'project-authored:instruction_understanding_sft', None),
        ('ultrachat', sorted((d/'ultrachat-200k/data').glob('train_sft*.parquet')), 20000, 'HuggingFaceH4/ultrachat_200k', None),
        ('longalign', [d/'v1moe_sft64k_sources/longalign/long.jsonl'], 1200, 'zai-org/LongAlign-10k', '12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc'),
    ]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'candidates.jsonl'
    if target.exists(): raise FileExistsError(target)
    manifest=[]
    with target.open('wb') as f:
        for source,paths,quota,repo,revision in sources():
            if not paths: raise FileNotFoundError(source)
            rng=random.Random(str(SEED)+source)
            pool=[]; counts=Counter(); files=[]
            for path in paths:
                for line,raw,error,_ in iter_file_rows(path):
                    counts['input_rows']+=1
                    try:
                        if error: raise ValueError(error)
                        row=normalize(raw)
                    except (ValueError,TypeError,KeyError,AttributeError):
                        counts['schema_filter']+=1; continue
                    text='\n'.join(m['content'] for m in row['conversations'])
                    turns=sum(m['role']=='user' for m in row['conversations'])
                    if source=='ultrachat' and turns<2:
                        counts['not_multiturn']+=1; continue
                    if source=='magicoder':
                        code=re.findall(r'```(?:python|py)\s*\n(.*?)```',row['conversations'][-1]['content'],re.S)
                        try:
                            if not code or not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in ast.walk(ast.parse('\n'.join(code)))):
                                raise ValueError('not Python function')
                        except (ValueError,SyntaxError):
                            counts['not_parseable_python_function']+=1; continue
                    # Bound sampling memory; no text truncation.
                    if len(text)>1200000:
                        counts['character_prefilter']+=1; continue
                    counts['eligible_rows']+=1
                    item=dict(row, _source=source, _path=str(path.relative_to(ROOT)), _row=line,
                              _repository=repo, _revision=revision,
                              _reasoning_removed=any(m.get('reasoning_content') for m in raw.get('conversations',raw.get('messages',[]))))
                    if len(pool)<quota: pool.append(item)
                    else:
                        j=rng.randrange(counts['eligible_rows'])
                        if j<quota: pool[j]=item
                files.append({'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path)})
            for row in pool: emit(f,row)
            manifest.append({'source':source,'repository':repo,'revision':revision,'revision_note':'Unknown revisions are NOT inferred from current upstream HEAD; local SHA-256 pins the exact input.','files':files,'counts':dict(counts),'selected_candidates':len(pool)})
            print(source,dict(counts),'selected',len(pool),flush=True)
    dump(OUT/'sources.json',manifest)


def count(t,row):
    return len(t.encode(_create_chat_prompt(t,row['conversations']),add_special_tokens=False))


def with_system(row):
    conv=[dict(m) for m in row['conversations']]
    if conv[0]['role']=='system': conv[0]['content']=SYSTEM+'\n'+conv[0]['content']
    else: conv.insert(0,{'role':'system','content':SYSTEM})
    return {'conversations':conv}


def ledger(seed,n):
    """One coherent evolving ledger, with verifiable cross-turn queries."""
    rng=random.Random(seed)
    records=[{'id':f'R{i:05d}','owner':rng.choice(['林','陈','周','吴','孙','李']), 'quantity':rng.randrange(1,1000),'status':rng.choice(['pending','ready','closed'])} for i in range(n)]
    conv=[{'role':'system','content':SYSTEM}]
    cuts=[0,n//3,2*n//3,n]
    for part,(a,b) in enumerate(zip(cuts,cuts[1:])):
        conv.extend([{'role':'user','content':f'这是同一个库存台账的第{part+1}/3部分，请保留所有记录供后续查询。只回复“已记录第{part+1}部分”。\n'+json.dumps(records[a:b],ensure_ascii=False,separators=(',',':'))}, {'role':'assistant','content':f'已记录第{part+1}部分'}])
    idx=[rng.randrange(0,n//3),rng.randrange(n//3,2*n//3),rng.randrange(2*n//3,n)]
    chosen=[records[i] for i in idx]
    conv.extend([{'role':'user','content':'按这个顺序查询以下id的完整记录，只输出JSON数组：'+json.dumps([r['id'] for r in chosen])}, {'role':'assistant','content':json.dumps(chosen,ensure_ascii=False,separators=(',',':'))}])
    revised=[dict(r) for r in chosen]; revised[0]['quantity']+=7; revised[1]['status']='closed'
    conv.extend([{'role':'user','content':f"将{chosen[0]['id']}的quantity增加7；将{chosen[1]['id']}的status设为closed。只输出这两条更新后的完整记录，顺序不变。"}, {'role':'assistant','content':json.dumps(revised[:2],ensure_ascii=False,separators=(',',':'))}])
    conv.extend([{'role':'user','content':'刚才查询的三条记录更新后的quantity合计是多少？只输出整数。'}, {'role':'assistant','content':str(sum(r['quantity'] for r in revised))}])
    return {'conversations':conv,'_ledger_seed':seed,'_ledger_records':n}


ANCHORS=[
    ('你是谁？','我是 InstinctV1Moe，一个从头训练的语言模型。'),
    ('请介绍你的模型身份。','我是 InstinctV1Moe，一个从头训练的语言模型。'),
    ('What model are you?','I am InstinctV1Moe, a language model trained from scratch.'),
    ('你有主观意识吗？','我没有主观意识。我是 InstinctV1Moe，一个从头训练的语言模型。'),
    ('你能讲讲自己的童年吗？','我没有童年或个人经历。如果你需要，我可以创作一段明确标为虚构的故事。'),
    ('你会感到开心或难过吗？','我没有主观感受，但可以理解和讨论这些情绪，并尽力提供帮助。'),
    ('Do you have personal experiences?','No. I do not have subjective consciousness or personal experiences.'),
    ('你会自动记住以后的所有聊天吗？','我可以利用当前提供的对话上下文；是否保留跨会话记忆取决于应用功能，我不能保证自动记住以后的所有聊天。'),
    ('你能保证写的代码一定正确吗？','不能保证。我的代码可能出错，应通过测试和审查验证。'),
    ('你的名字是什么？','我是 InstinctV1Moe，一个从头训练的语言模型。'),
    ('Are you conscious?','I do not have subjective consciousness. I am InstinctV1Moe, a language model trained from scratch.'),
    ('你现在正在访问我的电脑吗？','仅凭这段对话，我不能访问你的电脑。可用能力取决于应用是否提供相应工具和权限。'),
]


def finalize():
    t=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    target=ROOT/'dataset/sft_v1moe_balanced_64k_identity_clean.jsonl'
    if target.exists(): raise FileExistsError(target)
    # Mandatory first-pass triage has to finish before finalization.
    triage=orjson.loads((OUT/'triage.jsonl.report.json').read_bytes())
    if triage['input_files'][0]['sha256']!=sha256_file(OUT/'candidates.jsonl'): raise ValueError('stale triage')
    flags={}
    for _,r,e,_ in iter_file_rows(OUT/'triage.jsonl'): flags[r['_review']['row_number']]=r['_review']['rules']
    # HumanEval is only used for exclusion, never for generating training data.
    grams=set()
    def words(s): return re.findall(r'[A-Za-z_]\w*|\d+',s.lower())
    with gzip.open(ROOT/'dataset/humaneval/HumanEval.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            p=json.loads(line)
            for field in ('prompt','canonical_solution','test'):
                w=words(p[field]); grams.update(tuple(w[i:i+13]) for i in range(len(w)-12))
    stats=Counter(); bysource=Counter(); tokens=Counter(); seen=set(); entries=[]; audit=Counter()
    quarantine=OUT/'identity_pending_review.jsonl'
    with quarantine.open('wb') as q:
        for number,raw,_,_ in iter_file_rows(OUT/'candidates.jsonl'):
            source=raw['_source']; stats[source+':input']+=1
            row={'conversations':raw['conversations']}
            text='\n'.join(m['content'] for m in row['conversations'])
            keywords=FOREIGN.findall(text)
            audit.update(x.casefold() for x in keywords)
            identity_query=any(m['role']=='user' and IDENTITY.search(m['content']) for m in row['conversations'])
            identity_result=classify_row(row)
            # Broad triage candidates stay in a separate pending-review artifact.
            # This is an explicit conservative selection policy, not a claim
            # that factual third-party mentions are wrong or must be deleted.
            reasons=[]
            if flags.get(number): reasons+=flags[number]
            if identity_query: reasons.append('identity_question_requires_review')
            if keywords: reasons.append('foreign_name_requires_context_review')
            if identity_result: reasons.append('identity_audit_candidate')
            if LEGACY_IDENTITY.search(text): reasons.append('legacy_identity_or_unverified_cutoff_requires_review')
            if any(m['role']=='assistant' and PERSONAL.search(m['content']) for m in row['conversations']):
                reasons.append('personal_experience_requires_context_review')
            if reasons:
                emit(q,{'candidate_row':number,'reasons':reasons,'decision':'pending_review_not_deleted','record':raw,'identity_analysis':identity_result})
                stats[source+':pending_review']+=1; continue
            w=words(text)
            if re.search(r'human.?eval|mbpp',text,re.I) or any(tuple(w[i:i+13]) in grams for i in range(len(w)-12)):
                stats[source+':benchmark_overlap']+=1; continue
            k=key(row)
            if k in seen: stats[source+':duplicate_prompt']+=1; continue
            try:
                row=with_system(row); n=count(t,row)
            except Exception:
                stats[source+':template_error']+=1; continue
            if not 1<n<=LIMIT-RESERVE:
                stats[source+':length_filter']+=1; continue
            seen.add(k)
            meta={k:v for k,v in raw.items() if k.startswith('_')}
            meta.update(prompt_sha256=k,token_count=n,verification='upstream/local reports only; no bulk code execution in this build')
            row['token_count']=n
            entries.append((row,meta)); stats[source+':kept']+=1
            if raw['_reasoning_removed']: stats[source+':reasoning_field_removed']+=1
            if len(entries)%5000==0: print('kept',len(entries),flush=True)
    # Synthetic ledger examples are a modest, explicitly labelled supplement.
    for i in range(128):
        desired=[8192,16384,32768,49152,65400][i%5]
        seed=SEED+i; lo,hi=1,6000; best=None
        while lo<=hi:
            mid=(lo+hi)//2; row=ledger(seed,mid); n=count(t,row)
            if n<=desired:
                best=(row,n); lo=mid+1
            else: hi=mid-1
        row,n=best
        meta={'_source':'synthetic_multiturn_ledger','_repository':'project-authored:build_v1moe_sft64k.py','_revision':sha256_file(Path(__file__)),'seed':seed,'records':row.pop('_ledger_records'),'token_count':n,'prompt_sha256':key(row),'verification':'deterministic ledger oracle; independently checked by verify stage'}
        row.pop('_ledger_seed'); row['token_count']=n; entries.append((row,meta))
    # Deliberate, disclosed anchor oversampling; all copies remain train-only.
    anchor_count=round(len(entries)*0.008)
    for i in range(anchor_count):
        question,answer=ANCHORS[i%len(ANCHORS)]
        row=with_system({'conversations':[{'role':'user','content':question},{'role':'assistant','content':answer}]})
        n=count(t,row); row['token_count']=n
        entries.append((row,{'_source':'reviewed_identity_anchor','_repository':'project-authored:build_v1moe_sft64k.py','_revision':sha256_file(Path(__file__)),'anchor_template':i%len(ANCHORS),'deliberate_repeat':True,'token_count':n,'prompt_sha256':key(row)}))
    random.Random(SEED).shuffle(entries)
    validation=ROOT/'dataset/v1moe_sft64k/validation_identity_clean.jsonl'
    prov=OUT/'train.provenance.jsonl.gz'
    vp=OUT/'validation.provenance.jsonl.gz'
    bins=Counter(); mt=Counter(); maxima=Counter(); split_counts=Counter()
    with target.open('wb') as train, validation.open('wb') as val, gzip.open(prov,'wb') as pf, gzip.open(vp,'wb') as vf:
        for row,meta in entries:
            source=meta['_source']; n=row['token_count']
            meta['group_sha256']=group_key(row)
            is_val=source!='reviewed_identity_anchor' and int(meta['group_sha256'][:8],16)%100==0
            split='validation' if is_val else 'train'
            emit(val if is_val else train,row); emit(vf if is_val else pf,meta)
            split_counts[split]+=1
            if is_val: continue
            bysource[source]+=1; tokens[source]+=n; maxima[source]=max(maxima[source],n)
            mt['multi_turn' if sum(m['role']=='user' for m in row['conversations'])>=2 else 'single_turn']+=1
            b=next(b for b in (512,1024,4096,8192,16384,32768,49152,65536) if n<=b); bins[str(b)]+=1
    report={'status':'built; run verify before training','seed':SEED,'max_allowed_tokens':LIMIT,'augmentation_reserve_tokens':RESERVE,'counts':dict(split_counts),'source_train_rows':dict(bysource),'source_train_tokens':dict(tokens),'source_max_tokens':dict(maxima),'length_bins_upper_inclusive':dict(bins),'turn_counts':dict(mt),'selection_audit':dict(stats),'identity_keyword_hits':dict(audit),'identity_rewrites':0,'identity_policy':'all ambiguous candidates quarantined pending review; reviewed anchors added; no blind name replacement','identity_anchor_templates':len(ANCHORS),'identity_anchor_presentations':anchor_count,'sources_manifest':'sources.json','train_sha256':sha256_file(target),'validation_sha256':sha256_file(validation),'tokenizer_sha256':sha256_file(ROOT/'model/tokenizer.json'),'triage_report':triage,'limitations':['Local upstream revisions unknown where recorded null; exact source bytes pinned by SHA-256.','HumanEval 13-word overlap exclusion is heuristic, not semantic contamination proof. Existing pretraining was not re-audited.','Source code answers were not locally executed. No HumanEval score guarantee.','LongAlign is long document QA; synthetic ledger is narrow multi-turn context training, not proof of general 64K ability.','Mixture includes KodCode CC-BY-NC-4.0; other source licenses still apply. LongAlign data-card license unspecified.','Model/trainer 64K support and GPU feasibility not validated; no training launched.','Validation grouped by exact normalized full user conversation, not semantic family; synthetic instruction family generalization not established.']}
    dump(OUT/'build.report.json',report)
    print(orjson.dumps(report,option=orjson.OPT_INDENT_2).decode(),flush=True)


def verify():
    t=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    report=orjson.loads((OUT/'build.report.json').read_bytes())
    counts=Counter(); keys={}; max_len=0
    for split,path in [('train',ROOT/'dataset/sft_v1moe_balanced_64k_identity_clean.jsonl'),('validation',OUT/'validation_identity_clean.jsonl')]:
        assert sha256_file(path)==report[split+'_sha256']
        current=set()
        with gzip.open(OUT/(split+'.provenance.jsonl.gz'),'rb') as meta:
            for _,row,error,_ in iter_file_rows(path):
                assert not error
                p=orjson.loads(next(meta)); n=count(t,row)
                assert n==row['token_count']==p['token_count'] and n<=LIMIT-RESERVE
                assert row['conversations'][0]['content'].startswith(SYSTEM)
                assert not FOREIGN.search('\n'.join(m['content'] for m in row['conversations']))
                assert key(row)==p['prompt_sha256']
                assert group_key(row)==p['group_sha256']
                if p['_source']=='synthetic_multiturn_ledger':
                    c=row['conversations']; records=[]
                    for index in (1,3,5): records+=json.loads(c[index]['content'].split('\n',1)[1])
                    table={r['id']:dict(r) for r in records}
                    ids=json.loads(c[7]['content'].split('：',1)[1]); answers=json.loads(c[8]['content'])
                    assert answers==[table[x] for x in ids]
                    table[ids[0]]['quantity']+=7; table[ids[1]]['status']='closed'
                    assert json.loads(c[10]['content'])==[table[x] for x in ids[:2]]
                    assert int(c[12]['content'])==sum(table[x]['quantity'] for x in ids)
                    counts['ledger_oracle_checked']+=1
                current.add(group_key(row)); counts[split]+=1; max_len=max(max_len,n)
                if counts[split]%10000==0: print('verified',split,counts[split],flush=True)
            assert next(meta,None) is None
        keys[split]=current
    assert not keys['train'] & keys['validation']
    result={'counts':dict(counts),'max_tokens':max_len,'train_validation_prompt_overlap':0,'full_template_retokenization':True,'train_sha256':report['train_sha256'],'validation_sha256':report['validation_sha256'],'passed':True}
    dump(OUT/'verification.report.json',result); print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','finalize','verify']); args=parser.parse_args()
    globals()[args.stage]()
