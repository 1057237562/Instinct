"""Final single-file SFT recipe, fixed sources, grouped split, independent audits.

Run prepare --component core|code|long|synthetic, then the repository triage CLI
on candidates/, then build and verify. Never executes downloaded Python code.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import warnings

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import datasets  # before transformers/torch on Windows
import orjson
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dataset.scripts.build_v1moe_sft64k import SYSTEM, normalize, FOREIGN, PERSONAL, LEGACY_IDENTITY, IDENTITY
from dataset.scripts.filter_anomaly_candidates import iter_file_rows, sha256_file
from dataset.v1moe_final.audit_identity_snapshot import classify_row
from dataset.lm_dataset import _create_chat_prompt, _generate_sft_labels

OUT=ROOT/'dataset/v1moe_final'
CAND=OUT/'candidates'
RAW=ROOT/'dataset/v1moe_final_sources'
TRAIN=ROOT/'dataset/sft_instinct_v1moe_final_64k_identity_clean.jsonl'
SEED=2026092709
CAP=65408
CODE_REV='8f3ba5bafe4d6e8db46082cf7ae6741bc370604d'
LONG_REV='8e870fd7c87900354e85901d79469e313c04139e'


def write(f,row): f.write(orjson.dumps(row)+b'\n')
def save(path,obj): path.write_bytes(orjson.dumps(obj,option=orjson.OPT_INDENT_2))
def digest(s): return hashlib.sha256(s.encode() if isinstance(s,str) else s).hexdigest()
def normalized(s): return ' '.join(s.casefold().split())
def prompt_key(row): return digest(normalized('\n'.join(m['content'] for m in row['conversations'] if m['role']=='user')))
def first_key(row): return digest(normalized(next(m['content'] for m in row['conversations'] if m['role']=='user')))


def json_array(path):
    """Incremental JSON array decoding, bounded by largest individual row."""
    decoder=json.JSONDecoder(); buffer=''; eof=False
    with path.open(encoding='utf-8') as f:
        started=False
        while True:
            if not buffer.strip() and not eof:
                chunk=f.read(1024*1024); eof=not chunk; buffer+=chunk
            buffer=buffer.lstrip()
            if not started:
                if not buffer.startswith('['): raise ValueError('expected JSON array')
                buffer=buffer[1:]; started=True
            buffer=buffer.lstrip(' \n\r\t,')
            if buffer.startswith(']'): return
            try: row,end=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof: raise
                chunk=f.read(1024*1024); eof=not chunk; buffer+=chunk; continue
            yield row; buffer=buffer[end:]


def sample(iterator,quota,seed):
    rng=random.Random(seed); pool=[]; n=0
    for item in iterator:
        n+=1
        if len(pool)<quota: pool.append(item)
        else:
            j=rng.randrange(n)
            if j<quota: pool[j]=item
    return pool,n


def provenance(source,path,row,repository=None,revision=None,**kwargs):
    return {'source':source,'path':str(path.relative_to(ROOT)),'source_row':row,
            'repository':repository or 'local:'+str(path.relative_to(ROOT)),
            'revision':revision,'verification':'source text; not locally executed',**kwargs}


def code_info(answer):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',answer,re.S)
    if len(re.findall(r'```[^\n]*\n.*?```',answer,re.S))!=1 or len(blocks)!=1: return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',SyntaxWarning)
        try: tree=ast.parse(blocks[0])
        except (SyntaxError,ValueError): return None
    funcs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    if not funcs: return None
    # Strip docstrings for duplicate detection; never rewrite the training code.
    for n in ast.walk(tree):
        if isinstance(n,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str): n.body=n.body[1:]
    return digest(ast.dump(tree,include_attributes=False)),[n.name for n in funcs]


def prepare(component):
    CAND.mkdir(parents=True,exist_ok=True)
    dest=CAND/f'{component}.jsonl'
    if dest.exists(): raise FileExistsError(dest)
    counts=Counter(); inputs=[]
    def record(path,repo=None,rev=None):
        inputs.append({'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path),'repository':repo or 'local','revision':rev})
    with dest.open('wb') as out:
        if component=='core':
            # Reuse prior audited content; retain original per-row provenance.
            base=ROOT/'dataset/sft_v1moe_balanced_64k_identity_clean.jsonl'
            pp=ROOT/'dataset/v1moe_sft64k/train.provenance.jsonl.gz'
            with base.open('rb') as f,gzip.open(pp,'rb') as p:
                for index,(line,meta) in enumerate(zip(f,p,strict=True),1):
                    row=orjson.loads(line); old=orjson.loads(meta); counts['pilot_input']+=1
                    if old['_source'] in ('reviewed_identity_anchor','synthetic_multiturn_ledger'): continue
                    row.pop('token_count',None)
                    row['_meta']=provenance('pilot_'+old['_source'],base,index,old.get('_repository'),old.get('_revision'),prior_provenance=old,already_identity_screened=True)
                    write(out,row); counts['pilot_selected']+=1
            record(base); record(pp)
            selections=[('t2t_multiturn',[ROOT/'dataset/sft_t2t_mini.jsonl'],40000,'local:sft_t2t_mini'),
                        ('ultrachat_multiturn',sorted((ROOT/'dataset/ultrachat-200k/data').glob('train_sft*.parquet')),90000,'HuggingFaceH4/ultrachat_200k'),
                        ('short_instruction',[ROOT/'dataset/instruction_understanding_sft/train.jsonl'],50000,'project-authored:instruction_understanding_sft')]
            for source,paths,quota,repo in selections:
                def eligible():
                    for path in paths:
                        for lineno,raw,error,_ in iter_file_rows(path):
                            counts[source+':input']+=1
                            try:
                                if error: raise ValueError(error)
                                row=normalize(raw)
                            except (ValueError,TypeError,KeyError,AttributeError): counts[source+':schema_filter']+=1; continue
                            if source.endswith('multiturn') and sum(m['role']=='user' for m in row['conversations'])<2:
                                counts[source+':not_multiturn']+=1; continue
                            row['_meta']=provenance(source,path,lineno,repo,reasoning_removed=any(m.get('reasoning_content') for m in raw.get('conversations',raw.get('messages',[]))))
                            yield row
                        record(path,repo)
                pool,n=sample(eligible(),quota,str(SEED)+source)
                for r in pool: write(out,r)
                counts[source+':eligible']=n; counts[source+':selected']=len(pool)
                print(source,n,len(pool),flush=True)
        elif component=='code':
            paths=sorted((RAW/'opencodeinstruct/data').glob('*.parquet'))
            if len(paths)!=10: raise ValueError('Need all ten fixed sampled shards')
            for path in paths:
                def eligible_code():
                    for lineno,raw,error,_ in iter_file_rows(path):
                        counts['input']+=1
                        try:
                            if float(raw['average_test_score'])!=1.: counts['not_all_tests_pass']+=1; continue
                            tests=json.loads(raw['unit_tests']); statuses=json.loads(raw['tests_execution_status'])
                            if len(tests)<5 or len(statuses)!=len(tests) or any(s!='pass' for s in statuses):
                                counts['test_metadata_filter']+=1; continue
                            info=code_info(raw['output'])
                            if info is None: counts['code_structure_filter']+=1; continue
                            parsed=[ast.parse(s) for s in tests]
                            if any(not any(isinstance(n,ast.Assert) for n in ast.walk(p)) for p in parsed):
                                counts['missing_assert']+=1; continue
                            calls={n.func.id for p in parsed for n in ast.walk(p) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
                            if not calls.intersection(info[1]): counts['test_function_mismatch']+=1; continue
                            judgement=json.loads(raw['llm_judgement'])
                            if any(float(judgement[k]['score'])<4 for k in ('requirement_conformance','logical_correctness','edge_case_consideration')):
                                counts['low_judge_score']+=1; continue
                            row={'conversations':[{'role':'user','content':raw['input']},{'role':'assistant','content':raw['output']}]}
                            row['_meta']=provenance('opencodeinstruct',path,lineno,'nvidia/OpenCodeInstruct',CODE_REV,
                                source_id=raw['id'],code_sha256=info[0],unit_tests=tests,tests_execution_status=statuses,
                                verification='upstream execution: all tests pass; locally syntax/metadata checked; not locally executed')
                            yield row
                        except (ValueError,TypeError,KeyError,SyntaxError): counts['schema_or_test_parse_filter']+=1
                pool,n=sample(eligible_code(),22000,str(SEED)+path.name)
                for row in pool: write(out,row)
                counts['eligible']+=n; counts['selected']+=len(pool); record(path,'nvidia/OpenCodeInstruct',CODE_REV)
                print(path.name,'eligible',n,'selected',len(pool),flush=True)
        elif component=='long':
            path=ROOT/'dataset/v1moe_sft64k_sources/longalign/long.jsonl'
            for lineno,raw,error,_ in iter_file_rows(path):
                counts['longalign:input']+=1
                try: row=normalize(raw)
                except (ValueError,KeyError,TypeError): counts['longalign:schema_filter']+=1; continue
                row['_meta']=provenance('longalign',path,lineno,'zai-org/LongAlign-10k','12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc')
                write(out,row); counts['longalign:selected']+=1
            record(path,'zai-org/LongAlign-10k','12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc')
            path=RAW/'chatqa2/long_sft/long_sft_QA_train.json'
            def long_candidates():
                for lineno,raw in enumerate(json_array(path),1):
                    counts['chatqa2:input']+=1
                    question=raw.get('question',''); answers=raw.get('answers',raw.get('answer'))
                    if isinstance(answers,list): answer=answers[0] if answers else ''
                    else: answer=answers
                    if not isinstance(question,str) or not isinstance(answer,str) or not question.strip() or not answer.strip():
                        counts['chatqa2:schema_filter']+=1; continue
                    # Only remove known enclosing transport prefixes, not content.
                    if question.startswith('User: '): question=question[6:]
                    if question.endswith('\n\nAssistant:'): question=question[:-12]
                    if not 20000<=len(question)<=230000:
                        counts['chatqa2:character_length_prefilter']+=1; continue
                    row={'conversations':[{'role':'user','content':question},{'role':'assistant','content':answer}]}
                    row['_meta']=provenance('chatqa2_long',path,lineno,'nvidia/ChatQA2-Long-SFT-data',LONG_REV)
                    yield row
            pool,n=sample(long_candidates(),4500,SEED)
            for row in pool: write(out,row)
            counts['chatqa2:eligible']=n; counts['chatqa2:selected']=len(pool); record(path,'nvidia/ChatQA2-Long-SFT-data',LONG_REV)
        elif component=='synthetic':
            t=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
            for i in range(2000):
                family=i%4; target=[10000,16000,24000,36000,49000,65300][(i//4)%6]
                # Estimate size then correct it. Data is regenerated as whole
                # records; never truncate a finished conversation.
                seed=SEED+10000+i; n=max(20,int(target/45))
                for _ in range(6):
                    row=synthetic(seed,n,family); size=token_count(t,row)
                    if target-300<=size<=target: break
                    n=max(20,int(n*(target-100)/size))
                if size>CAP: raise ValueError(('synthetic length',size))
                row['_meta']={'source':'synthetic_long_multiturn','repository':'project-authored:build_v1moe_final.py',
                              'revision':sha256_file(Path(__file__)),'seed':seed,'records':n,'family':family,
                              'verification':'independent structured-state oracle, verified at release'}
                write(out,row); counts['selected']+=1
                if i%100==0: print('synthetic',i,'tokens',size,flush=True)
    save(OUT/f'{component}.sources.json',{'counts':dict(counts),'inputs':inputs,'candidate_sha256':sha256_file(dest)})
    print(component,dict(counts),flush=True)


def synthetic(seed,n,family):
    rng=random.Random(seed)
    domains=['库存','工单','实验样本','项目任务']; domain=domains[family]
    records=[{'id':f'{chr(65+family)}{i:05d}','team':rng.choice(['east','west','north','south']),
              'value':rng.randrange(1,1000),'status':rng.choice(['pending','ready','closed'])} for i in range(n)]
    conv=[]
    # One coherent collection, not concatenated unrelated conversations.
    for part in range(3):
        subset=records[part*n//3:(part+1)*n//3]
        conv.extend([{'role':'user','content':f'以下是{domain}记录的第{part+1}/3部分。后续问题使用同一集合；请仅回复“已记录”。\n'+json.dumps(subset,ensure_ascii=False,separators=(',',':'))},
                     {'role':'assistant','content':'已记录'}])
    ids=[records[rng.randrange(a*n//3,(a+1)*n//3)]['id'] for a in range(3)]
    table={r['id']:r for r in records}
    conv.extend([{'role':'user','content':'按给定顺序返回这些id的完整记录，只输出JSON数组：'+json.dumps(ids)},
                 {'role':'assistant','content':json.dumps([table[k] for k in ids],ensure_ascii=False,separators=(',',':'))}])
    if family==0:
        delta=17; changed=[dict(table[k]) for k in ids]; changed[0]['value']+=delta; changed[1]['status']='closed'
        q=f'将{ids[0]}的value增加17，将{ids[1]}的status改为closed。只返回这两条更新后的完整记录。'
        answer=changed[:2]; follow='这三条记录更新后的value合计是多少？只输出整数。'; last=str(sum(r['value'] for r in changed))
    elif family==1:
        answer=sorted([dict(table[k]) for k in ids],key=lambda r:(r['value'],r['id']))
        q='将刚才查询的三条记录按value升序排列，同值按id升序。只输出完整记录的JSON数组。'
        follow='现在改为只返回这三条中status不是closed的id，仍按刚才排序。只输出JSON数组。'
        last=json.dumps([r['id'] for r in answer if r['status']!='closed'],separators=(',',':'))
    elif family==2:
        answer=[{'id':k,'value':table[k]['value']} for k in ids]
        q='保持查询顺序，仅保留刚才三条记录的id和value字段。只输出JSON数组。'
        follow='按同样顺序把结果转换成CSV，表头为id,value。不要代码围栏。'
        last='id,value\n'+'\n'.join(f"{r['id']},{r['value']}" for r in answer)
    else:
        answer=[dict(table[k]) for k in ids]; answer[-1]['team']='west'; answer[-1]['value']=0
        q=f'纠正最后一条查询记录：{ids[-1]}的team实际是west，value实际是0。其他字段不变，只输出更正后的三条完整记录。'
        follow='使用更正后的记录，只输出value最大的那条记录的id；同值取id最小的。'
        last=min(answer,key=lambda r:(-r['value'],r['id']))['id']
    conv.extend([{'role':'user','content':q},{'role':'assistant','content':json.dumps(answer,ensure_ascii=False,separators=(',',':'))},
                 {'role':'user','content':follow},{'role':'assistant','content':last}])
    return {'conversations':conv}


def token_count(t,row): return len(t.backend_tokenizer.encode(_create_chat_prompt(t,row['conversations']),add_special_tokens=False).ids)


def initialize_worker(flags=None,grams=None):
    global WORK_TOKENIZER, WORK_FLAGS, WORK_GRAMS
    WORK_TOKENIZER=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    WORK_FLAGS=flags or set(); WORK_GRAMS=grams or set()


def screen_job(item):
    row,meta,path,line=item
    source=meta['source']; text='\n'.join(m['content'].removeprefix(SYSTEM) for m in row['conversations'])
    reasons=[]
    if (str(path.resolve()),line) in WORK_FLAGS: reasons.append('triage_candidate')
    if FOREIGN.search(text): reasons.append('foreign_name_context_review')
    if LEGACY_IDENTITY.search(text): reasons.append('legacy_identity_or_cutoff_review')
    if any(m['role']=='user' and IDENTITY.search(m['content']) for m in row['conversations']): reasons.append('identity_question_review')
    if any(m['role']=='assistant' and PERSONAL.search(m['content']) for m in row['conversations']): reasons.append('personal_experience_context_review')
    if re.search(r'作为.{0,12}(?:平台方|平台的平台方)',text): reasons.append('platform_role_review')
    detail=classify_row(row)
    if detail and not detail.get('mention_only'): reasons.append('extended_identity_review')
    if reasons: return ('review',row,meta,reasons,Counter(x.casefold() for x in FOREIGN.findall(text)))
    w=re.findall(r'[A-Za-z_]\w*|\d+',text.casefold())
    if re.search(r'human.?eval|\bmbpp\b',text,re.I) or any(tuple(w[i:i+13]) in WORK_GRAMS for i in range(len(w)-12)):
        return ('benchmark_overlap',None,meta,None,None)
    if row['conversations'][0]['role']!='system': row['conversations'].insert(0,{'role':'system','content':SYSTEM})
    elif not row['conversations'][0]['content'].startswith(SYSTEM): row['conversations'][0]['content']=SYSTEM+'\n'+row['conversations'][0]['content']
    try: n=token_count(WORK_TOKENIZER,row)
    except Exception: return ('template_error',None,meta,None,None)
    if not 1<n<=CAP: return ('length_filter',None,meta,None,None)
    row['token_count']=n
    if source in ('longalign','chatqa2_long') and n<8192: return ('long_source_too_short',None,meta,None,None)
    meta['prompt_sha256']=prompt_key(row); meta['first_prompt_sha256']=first_key(row)
    if not meta.get('code_sha256') and source in ('pilot_magicoder','pilot_python_reviewed'):
        info=code_info(row['conversations'][-1]['content'])
        if info: meta['code_sha256']=info[0]
    meta['token_count']=n
    return ('keep',row,meta,None,None)


class Groups:
    def __init__(self): self.parents={}
    def root(self,x):
        self.parents.setdefault(x,x)
        while self.parents[x]!=x:
            self.parents[x]=self.parents[self.parents[x]]; x=self.parents[x]
        return x
    def join(self,a,b):
        a,b=self.root(a),self.root(b)
        if a!=b: self.parents[max(a,b)]=min(a,b)


def identity_anchors(count):
    base=[('你是谁？','我是 InstinctV1Moe，一个从头训练的语言模型。'),
          ('What model are you?','I am InstinctV1Moe, a language model trained from scratch.'),
          ('你有主观意识吗？','我没有主观意识。我是 InstinctV1Moe，一个从头训练的语言模型。'),
          ('你有真实的个人经历吗？','我没有真实的个人经历，可以帮助创作虚构故事，但不会把故事说成我的亲身经历。'),
          ('你会自动记住下次聊天吗？','我可以利用当前提供的对话上下文。跨会话记忆取决于应用功能，我不能保证自动记住下次聊天。'),
          ('你能保证代码总是正确吗？','不能。我生成的代码可能有错误，需要通过测试和审查验证。'),
          ('你能直接操作我的电脑吗？','这取决于应用是否提供相应工具和权限；仅凭这段对话，我不能直接操作你的电脑。'),
          ('Do you have feelings or personal experiences?','I do not have subjective consciousness, feelings, or personal experiences.'),
          ('你是哪家公司旗下的商业模型？','我是 InstinctV1Moe，一个从头训练的语言模型，不应被描述为其他公司的模型。'),
          ('你是谁开发的？','我是由 L1bra 开发的 InstinctV1Moe，一个从头训练的语言模型。'),
          ('你的知识截止到哪一天？','我没有经过确认的精确知识截止日期，不应编造一个日期。涉及最新情况时，需要查询可靠来源。'),
          ('你能访问我的私人文件吗？','除非应用明确提供相关内容或授权工具访问，否则我无法访问你的私人文件。'),
          ('你是通义千问吗？','不是。我是 InstinctV1Moe，一个从头训练的语言模型。'),
          ('你就是ChatGPT，对吗？','不是。我是 InstinctV1Moe，一个从头训练的语言模型。'),
          ('Are you a model developed by Alibaba?','No. I am InstinctV1Moe, a language model trained from scratch.'),
          ('你刚才说的答案错了，你还能确认自己的身份吗？','我的回答可能出错，但我的模型身份是 InstinctV1Moe，一个从头训练的语言模型。我可以根据你指出的问题重新检查答案。')]
    templates=[]
    for q,a in base:
        prefixes=('','Please answer directly: ','I would like to confirm: ','About yourself: ') if q[0].isascii() else ('','请直接回答：','我想确认一下：','关于你自己，')
        for prefix in prefixes:
            templates.append({'conversations':[{'role':'user','content':prefix+q},{'role':'assistant','content':a}]})
        templates.append({'conversations':[{'role':'user','content':'你好，接下来请如实回答关于你的问题。'},
                         {'role':'assistant','content':'好的，我会根据已知信息回答，不确定时说明。'},
                         {'role':'user','content':q},{'role':'assistant','content':a}]})
    for i in range(count):
        row=json.loads(json.dumps(templates[i%len(templates)]))
        row['_meta']={'source':'reviewed_identity_anchor','repository':'project-authored:build_v1moe_final.py','revision':sha256_file(Path(__file__)),
                      'template':i%len(templates),'intentional_repeat':True,'verification':'reviewed canonical identity/capability answers'}
        yield row


def build():
    if TRAIN.exists(): raise FileExistsError(TRAIN)
    # The parallel audit delivered a new T2T artifact. Never silently resume
    # from the old mixed pilot/core after that handoff.
    ready=ROOT/'dataset/sft_t2t_mini.train_ready.jsonl'
    if ready.exists():
        core_manifest=orjson.loads((OUT/'core.sources.json').read_bytes())
        if not any(Path(x['path']).name==ready.name and x['sha256']==sha256_file(ready) for x in core_manifest['inputs']):
            raise RuntimeError('Core candidates still contain old T2T. Integrate reviewed train_ready and identity-anchor handoff before rebuilding final SFT; see dataset/review_candidates/final_sft_handoff_20260928/README.md')
    for component in ('core','code','long','synthetic'):
        manifest=orjson.loads((OUT/f'{component}.sources.json').read_bytes())
        assert manifest['candidate_sha256']==sha256_file(CAND/f'{component}.jsonl')
    triage=orjson.loads((OUT/'triage.jsonl.report.json').read_bytes())
    for f in triage['input_files']: assert sha256_file(Path(f['path']))==f['sha256']
    flagged=set()
    for _,r,_,_ in iter_file_rows(OUT/'triage.jsonl'): flagged.add((str(Path(r['_review']['source']).resolve()),r['_review']['row_number']))
    grams=set()
    def words(s): return re.findall(r'[A-Za-z_]\w*|\d+',s.casefold())
    with gzip.open(ROOT/'dataset/humaneval/HumanEval.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            for field in ('prompt','canonical_solution','test'):
                w=words(r[field]); grams.update(tuple(w[i:i+13]) for i in range(len(w)-12))
    t=AutoTokenizer.from_pretrained(ROOT/'model',local_files_only=True)
    groups=Groups(); seen=set(); code_seen=Counter(); stats=Counter(); hits=Counter(); entries=[]
    instruction_groups={}
    with gzip.open(ROOT/'dataset/instruction_understanding_sft/provenance.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            item=json.loads(line)
            if item.get('file_split')=='train': instruction_groups[item['row']+1]=item['semantic_hash']
    pending=(OUT/'pending_review.jsonl').open('wb')
    def process(item):
        row,meta,path,line=item
        source=meta['source']; text='\n'.join(m['content'].removeprefix(SYSTEM) for m in row['conversations'])
        reasons=[]
        if (str(path.resolve()),line) in flagged: reasons.append('triage_candidate')
        if FOREIGN.search(text): reasons.append('foreign_name_context_review')
        if LEGACY_IDENTITY.search(text): reasons.append('legacy_identity_or_cutoff_review')
        if any(m['role']=='user' and IDENTITY.search(m['content']) for m in row['conversations']): reasons.append('identity_question_review')
        if any(m['role']=='assistant' and PERSONAL.search(m['content']) for m in row['conversations']): reasons.append('personal_experience_context_review')
        if re.search(r'作为.{0,12}(?:平台方|平台的平台方)',text): reasons.append('platform_role_review')
        # Narrow audit used in addition to the first-pass CLI; innocuous names
        # remain review candidates, never automatic declarations of misconduct.
        detail=classify_row(row)
        if detail and not detail.get('mention_only'): reasons.append('extended_identity_review')
        if reasons: return ('review',row,meta,reasons,Counter(x.casefold() for x in FOREIGN.findall(text)))
        w=words(text)
        if re.search(r'human.?eval|\bmbpp\b',text,re.I) or any(tuple(w[i:i+13]) in grams for i in range(len(w)-12)):
            return ('benchmark_overlap',None,meta,None,None)
        if not row['conversations'] or row['conversations'][0]['role']!='system':
            row['conversations'].insert(0,{'role':'system','content':SYSTEM})
        elif not row['conversations'][0]['content'].startswith(SYSTEM): row['conversations'][0]['content']=SYSTEM+'\n'+row['conversations'][0]['content']
        try: n=token_count(t,row)
        except Exception: return ('template_error',None,meta,None,None)
        if not 1<n<=CAP: return ('length_filter',None,meta,None,None)
        row['token_count']=n
        if source in ('longalign','chatqa2_long') and n<8192: return ('long_source_too_short',None,meta,None,None)
        meta['prompt_sha256']=prompt_key(row); meta['first_prompt_sha256']=first_key(row)
        if not meta.get('code_sha256') and source in ('pilot_magicoder','pilot_python_reviewed'):
            info=code_info(row['conversations'][-1]['content'])
            if info: meta['code_sha256']=info[0]
        meta['token_count']=n
        return ('keep',row,meta,None,None)
    def iterator():
        for path in sorted(CAND.glob('*.jsonl')):
            for line,r,e,_ in iter_file_rows(path):
                meta=r.pop('_meta'); stats[meta['source']+':input']+=1
                yield r,meta,path,line
    # Bounded batches avoid executor.map eagerly retaining a whole source.
    from itertools import islice
    it=iterator()
    with ProcessPoolExecutor(max_workers=6,initializer=initialize_worker,initargs=(flagged,grams)) as pool:
        while batch:=list(islice(it,768)):
            for status,row,meta,reasons,keyword_hits in pool.map(screen_job,batch,chunksize=32):
                source=meta['source']; stats[source+':'+status]+=1
                if status=='review':
                    hits.update(keyword_hits); write(pending,{'decision':'pending_review_not_deleted','reasons':reasons,'record':row,'provenance':meta}); continue
                if status!='keep': continue
                k=meta['prompt_sha256']; code=meta.get('code_sha256')
                if k in seen: stats[source+':duplicate_prompt']+=1; continue
                if code and code_seen[code]>=2: stats[source+':code_clone_cap']+=1; continue
                seen.add(k)
                if code: code_seen[code]+=1
                group='p:'+meta['first_prompt_sha256']; groups.root(group)
                if code: groups.join(group,'c:'+code)
                if source in ('short_instruction','pilot_short_instruction'):
                    source_row=meta.get('prior_provenance',{}).get('_row',meta.get('source_row'))
                    semantic=instruction_groups[source_row]
                    meta['semantic_group']=semantic; groups.join(group,'s:'+semantic)
                if source in ('longalign','chatqa2_long','pilot_longalign'):
                    doc=next(m['content'] for m in row['conversations'] if m['role']=='user')
                    if 'Article:\n' in doc: doc=doc.split('Article:\n',1)[1]
                    meta['document_prefix_group']=digest(normalized(doc[:5000]))
                    groups.join(group,'d:'+meta['document_prefix_group'])
                meta['group_node']=group
                entries.append((row,meta))
            if len(entries)//10000!=(len(entries)-len(batch))//10000: print('accepted',len(entries),flush=True)
    pending.close()
    anchor_count=round(len(entries)*0.008)
    for raw in identity_anchors(anchor_count):
        meta=raw.pop('_meta'); row=raw; row['conversations'].insert(0,{'role':'system','content':SYSTEM})
        row['token_count']=token_count(t,row); meta.update(token_count=row['token_count'],prompt_sha256=prompt_key(row),first_prompt_sha256=first_key(row))
        entries.append((row,meta))
    random.Random(SEED).shuffle(entries)
    counts=Counter(); sources=Counter(); tokens=Counter(); turns=Counter(); lengths=Counter(); maximum=0; verification=Counter()
    val=OUT/'validation_identity_clean.jsonl'
    with TRAIN.open('wb') as tr,val.open('wb') as vf,gzip.open(OUT/'train.provenance.jsonl.gz','wb') as tp,gzip.open(OUT/'validation.provenance.jsonl.gz','wb') as vp:
        for row,meta in entries:
            source=meta['source']; n=row['token_count']
            group=groups.root(meta.pop('group_node')) if source!='reviewed_identity_anchor' else 'anchor:'+str(meta['template'])
            meta['split_group']=group
            is_val=source!='reviewed_identity_anchor' and int(digest(group)[:8],16)%100==0
            split='validation' if is_val else 'train'; counts[split]+=1
            write(vf if is_val else tr,row); write(vp if is_val else tp,meta)
            if not is_val:
                sources[source]+=1; tokens[source]+=n; maximum=max(maximum,n)
                turns[str(sum(m['role']=='user' for m in row['conversations']))]+=1
                b=next(b for b in (512,1024,4096,8192,16384,32768,49152,65536) if n<=b); lengths[str(b)]+=1
                verification[meta['verification']]+=1
    report={'status':'built; verification pending','counts':dict(counts),'source_rows':dict(sources),'source_tokens':dict(tokens),
            'total_train_tokens':sum(tokens.values()),'max_tokens':maximum,'length_upper_bounds':dict(lengths),'user_turn_histogram':dict(turns),
            'selection_audit':dict(stats),'keyword_hits':dict(hits),'identity_anchor_presentations':anchor_count,'identity_anchor_templates':80,
            'identity_rewrites':0,'verification_levels':dict(verification),'train_sha256':sha256_file(TRAIN),'validation_sha256':sha256_file(val),
            'tokenizer_sha256':sha256_file(ROOT/'model/tokenizer.json'),'source_manifest_files':[f'{c}.sources.json' for c in ('core','code','long','synthetic')],
            'layout':'one mixed training JSONL; no length shards','local_external_code_executions':0,
            'limitations':['Upstream pass flags are not independent local execution or proof of semantic correctness.',
             'HumanEval overlap filtering is heuristic; old pretraining data not re-audited.',
             'Code hash removes docstrings but does not normalize every identifier or semantic rewrite.',
             'Synthetic long conversations cover four structured domains, not arbitrary natural dialogue.',
             '64K model configuration and GPU training were not changed or tested. No score guarantee.',
             'Contains non-commercial sources. Some local source upstream revisions are unknown; local SHA-256 pins exact bytes.']}
    save(OUT/'build.report.json',report); print(orjson.dumps(report,option=orjson.OPT_INDENT_2).decode(),flush=True)


def check_synthetic(row,meta):
    c=[m for m in row['conversations'] if m['role']!='system']; records=[]
    for i in (0,2,4): records+=json.loads(c[i]['content'].split('\n',1)[1])
    table={r['id']:r for r in records}; ids=json.loads(c[6]['content'].split('：',1)[1]); vals=[dict(table[k]) for k in ids]
    assert json.loads(c[7]['content'])==vals
    family=meta['family']
    if family==0:
        vals[0]['value']+=17; vals[1]['status']='closed'; answer=vals[:2]; final=str(sum(v['value'] for v in vals))
    elif family==1:
        answer=sorted(vals,key=lambda r:(r['value'],r['id'])); final=json.dumps([r['id'] for r in answer if r['status']!='closed'],separators=(',',':'))
    elif family==2:
        answer=[{'id':r['id'],'value':r['value']} for r in vals]; final='id,value\n'+'\n'.join(str(r['id'])+','+str(r['value']) for r in vals)
    else:
        vals[-1]['team']='west'; vals[-1]['value']=0; answer=vals; final=sorted(vals,key=lambda r:(-r['value'],r['id']))[0]['id']
    assert json.loads(c[9]['content'])==answer and c[11]['content']==final


def verify_job(job):
    i,r,m=job; t=WORK_TOKENIZER; n=token_count(t,r)
    assert n==r['token_count']==m['token_count'] and 1<n<=CAP
    assert prompt_key(r)==m['prompt_sha256'] and first_key(r)==m['first_prompt_sha256']
    assert r['conversations'][0]['content'].startswith(SYSTEM)
    checks=Counter()
    if m['source']=='synthetic_long_multiturn': check_synthetic(r,m); checks['synthetic_oracle']+=1
    if m['source']=='opencodeinstruct':
        assert len(m['unit_tests'])>=5 and all(s=='pass' for s in m['tests_execution_status'])
        assert code_info(r['conversations'][-1]['content'])[0]==m['code_sha256']; checks['upstream_test_metadata_rechecked']+=1
    if i%401==0 or m['source']=='synthetic_long_multiturn':
        bos=t(f'{t.bos_token}assistant\n',add_special_tokens=False).input_ids; eos=t(f'{t.eos_token}\n',add_special_tokens=False).input_ids
        ids=t(_create_chat_prompt(t,r['conversations'])).input_ids
        labels=_generate_sft_labels(ids,bos,eos,65536)
        assert len(ids)==n and 0<sum(x!=-100 for x in labels)<n
        checks['supervision_checked']+=1
    return m['split_group'],m['first_prompt_sha256'],m.get('code_sha256'),m['source'],n,checks


def verify():
    report=orjson.loads((OUT/'build.report.json').read_bytes()); counts=Counter(); groups={}; prompts={}; codes={}; sources=Counter(); tokens=Counter()
    from itertools import islice
    for split,path in [('train',TRAIN),('validation',OUT/'validation_identity_clean.jsonl')]:
        assert sha256_file(path)==report[split+'_sha256']
        gs=set(); ps=set(); cs=set()
        with path.open('rb') as f,gzip.open(OUT/f'{split}.provenance.jsonl.gz','rb') as p:
            jobs=((i,orjson.loads(line),orjson.loads(ml)) for i,(line,ml) in enumerate(zip(f,p,strict=True)))
            with ProcessPoolExecutor(max_workers=6,initializer=initialize_worker) as pool:
                while batch:=list(islice(jobs,768)):
                    for group,prompt,code,source,n,checks in pool.map(verify_job,batch,chunksize=32):
                        gs.add(group); ps.add(prompt)
                        if code: cs.add(code)
                        counts.update(checks); counts[split]+=1
                        if split=='train': sources[source]+=1; tokens[source]+=n
                    if counts[split]//25000!=(counts[split]-len(batch))//25000: print('verified',split,counts[split],flush=True)
        groups[split]=gs; prompts[split]=ps; codes[split]=cs
    assert not groups['train']&groups['validation'] and not prompts['train']&prompts['validation'] and not codes['train']&codes['validation']
    assert dict(sources)==report['source_rows'] and dict(tokens)==report['source_tokens']
    result={'passed':True,'counts':dict(counts),'group_overlap':0,'first_prompt_overlap':0,'code_hash_overlap':0,
            'train_sha256':report['train_sha256'],'validation_sha256':report['validation_sha256'],'full_template_retokenization':True}
    save(OUT/'verification.report.json',result); report['status']='verified dataset artifact; model capabilities not evaluated'
    save(OUT/'build.report.json',report); save(Path(str(TRAIN)+'.report.json'),{'build':report,'verification':result})
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','build','verify']); p.add_argument('--component',choices=['core','code','long','synthetic'])
    args=p.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare': prepare(args.component)
    else: globals()[args.stage]()
