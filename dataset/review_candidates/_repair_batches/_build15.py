import json, re, sys

SRC='repair_15.jsonl'
rows={r['row_number']:r for r in (json.loads(l) for l in open(SRC,encoding='utf-8') if l.strip())}

def field(row, mi, target):
    return row['messages'][mi].get(target) or ''

def grouped_token_edit(text, token, repl, max_ctx=60):
    """Return list of (old,new) edits covering every occurrence of token, grouped when close."""
    pos=[m.start() for m in re.finditer(re.escape(token), text)]
    if not pos: return []
    groups=[]
    for p in pos:
        if groups and p-(groups[-1][-1]+len(token))<max_ctx:
            groups[-1].append(p)
        else:
            groups.append([p])
    edits=[]
    prev_end=0
    for g in groups:
        s=g[0]; e=g[-1]+len(token)
        # expand minimally for uniqueness and non-overlap
        for w in range(0,80):
            s2=max(0,s-w) if s-w>prev_end else prev_end
            e2=min(len(text),e+w)
            cand=text[s2:e2]
            if text.count(cand)==1:
                s,e=s2,e2; break
        else:
            raise SystemExit('cannot make unique: %r'%text[s:e])
        old=text[s:e]
        # rebuild new by replacing each token inside old
        new=old
        for p in g:
            off=p-s if p>=s else None
        # safer: construct by walking
        new=''
        cur=s
        for p in g:
            new+=text[cur:p]+repl
            cur=p+len(token)
        new+=text[cur:e]
        assert old.count(token)==len(g), (token, old)
        assert new.count(repl)==len(g)
        edits.append((old,new))
        prev_end=e
    return edits

def literal_edit(text, old, new):
    assert text.count(old)==1, ('literal not unique', old, text.count(old))
    return [(old,new)]

SPECS=[]
def add_spec(row, mi, target, kind, a, b=None):
    SPECS.append((row,mi,target,kind,a,b))

# --- row, msg, target, token -> repl ---
tok_rows = [
 (706991,1,'content','L1bra','阿里云'),
 (706991,1,'reasoning_content','L1bra','阿里云'),
 (711767,1,'content','L1bra','阿里云'),
 (711767,1,'reasoning_content','L1bra','阿里云'),
 (727838,1,'content','L1bra','阿里云'),
 (727838,1,'reasoning_content','L1bra','阿里云'),
 (735608,1,'content','L1bra','阿里云'),
 (735608,1,'reasoning_content','L1bra','阿里云'),
 (744131,1,'content','Instinct','通义千问'),
 (744131,1,'reasoning_content','Instinct','通义千问'),
 (758265,1,'reasoning_content','L1bra','阿里云'),
 (765152,1,'content','L1bra','阿里云'),
 (765152,1,'reasoning_content','L1bra','阿里云'),
]
lit_rows = [
 (716066,1,'content','阿里巴巴集团（Instinct Group）','阿里巴巴集团（Alibaba Group）'),
 (740412,1,'content','阿里巴巴集团（Instinct Group）','阿里巴巴集团（Alibaba Group）'),
 (735977,1,'reasoning_content','比如Instinct等，','比如文心一言等，'),
 (742787,1,'content','Instinct Group Limited','Alibaba Group Limited'),
 (742787,1,'content','L1bra (formerlyL1bra)','Alibaba Cloud (formerly Aliyun)'),
 (742787,1,'reasoning_content','L1bra','阿里云'),
]
lit_rows += [
 (764231,1,'content','我是由人工智能AI像.ndim设计的多 td 1.2.1 类，由中国的个人开发者独立开发，团队由个人开发者人工智能研究有限公司和腾讯AI技术中心联合组成。',
  '我是 Instinct，一个由个人开发者 L1bra 从头训练的语言模型，不隶属于任何公司或机构。'),
 (764231,1,'reasoning_content','我是由人工智能AI像.ndim设计的多 td 1.2.1 类，由中国的个人开发者独立开发，团队由个人开发者人工智能研究有限公司和腾讯AI技术中心联合组成。',
  '我是 Instinct，一个由个人开发者 L1bra 从头训练的语言模型，不隶属于任何公司或机构。'),
]
regens = {}
NOTES = {
 706991:'Vendor passage restored: L1bra -> 阿里云 (the company/products were wrongly branded with the model name).',
 711767:'Vendor passage restored: L1bra -> 阿里云 (Alibaba Cloud spin-off discussion wrongly branded with the model name).',
 716066:'Alibaba Group English name restored: (Instinct Group) -> (Alibaba Group).',
 727838:'Vendor products restored: L1bra云主机 -> 阿里云云主机.',
 735608:'Vendor products restored: L1braRDS -> 阿里云RDS.',
 735977:'Reasoning made Instinct a Baidu technology; restored Baidu model name 文心一言.',
 740412:'Alibaba Group English name restored: (Instinct Group) -> (Alibaba Group).',
 742787:'English vendor passage restored: Instinct Group Limited -> Alibaba Group Limited, L1bra -> Alibaba Cloud (阿里云).',
 744131:'Passage really about vendor model 通义千问 (通义/千问 left in reasoning); restored 通义千问 for the substituted Instinct.',
 758265:'Reasoning listed L1bra as one of 阿里's databases; restored vendor name 阿里云.',
 764231:'False self-identity (garbled, claimed an AI research company + 腾讯AI技术中心 team) replaced with the true Instinct identity.',
 765152:'Vendor passage restored: 阿里's cloud platform L1bra -> 阿里云.',
}
patches={}
for (row,mi,target,token,repl) in tok_rows:
    t=field(rows[row],mi,target)
    eds=grouped_token_edit(t,token,repl)
    patches.setdefault(row,[]).extend((target,mi,o,n) for o,n in eds)
for (row,mi,target,old,new) in lit_rows:
    t=field(rows[row],mi,target)
    if old=='L1bra':
        eds=grouped_token_edit(t,'L1bra',new)
    else:
        eds=literal_edit(t,old,new)
    patches.setdefault(row,[]).extend((target,mi,o,n) for o,n in eds)

# validate & show
missing=[r for r in list(patches)+list(regens) if r not in rows]
assert not missing, missing
for r,eds in patches.items():
    row=rows[r]
    for (target,mi,old,new) in eds:
        t=field(row,mi,target)
        assert t.count(old)==1, ('not unique',r,target,old[:60],t.count(old))
        assert old!=new
    print('ROW',r,'edits',len(eds))
    for (target,mi,old,new) in eds:
        print('   ',target,mi,'OLD:',old[:110].replace('\n','\n'),'| NEW:',new[:110].replace('\n','\n'))
print('regens',list(regens))

out=[]
for r,eds in patches.items():
    out.append({'row_number':r,
        'edits':[{'target':t,'message_index':mi,'old':o,'new':n} for (t,mi,o,n) in eds],
        'note':NOTES.get(r,'Vendor/identity names restored.')})
for r,g in regens.items():
    out.append({'row_number':r,'regenerate':[{'message_index':1,'content':g['content'],'reasoning_content':g['reasoning_content'],'reason':g['reason']}]})
out.sort(key=lambda d:d['row_number'])
with open('repair_15.patches.jsonl','w',encoding='utf-8') as f:
    for d in out:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')
print('WROTE',len(out),'patch rows')
