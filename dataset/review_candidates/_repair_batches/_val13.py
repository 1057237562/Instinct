# -*- coding: utf-8 -*-
"""Validate repair_13.patches.jsonl: apply and re-scan for residual contamination."""
import json, re

SRC = 'repair_13.jsonl'
PAT = 'repair_13.patches.jsonl'

rows = {json.loads(l)['row_number']: json.loads(l) for l in open(SRC, encoding='utf-8')}
patches = [json.loads(l) for l in open(PAT, encoding='utf-8')]
pby = {p['row_number']: p for p in patches}

errs = 0
for p in patches:
    r = rows[p['row_number']]
    seen = set()
    for e in p['edits']:
        f = r['messages'][e['message_index']].get(e['target']) or ''
        c = f.count(e['old'])
        if c != 1:
            print('UNIQUENESS FAIL', p['row_number'], e['target'], c, repr(e['old'][:70])); errs += 1
        key = (e['target'], e['message_index'])
    # simulate sequential application as a grader would
    tmp = {}
    for e in p['edits']:
        v = tmp.get((e['target'], e['message_index']))
        if v is None:
            v = r['messages'][e['message_index']].get(e['target']) or ''
        if v.count(e['old']) != 1:
            print('SEQ-APPLY FAIL', p['row_number'], e['target'], v.count(e['old']), repr(e['old'][:70])); errs += 1
        tmp[(e['target'], e['message_index'])] = v.replace(e['old'], e['new'])
print('structural errors:', errs)

def applied(rn):
    r = rows[rn]
    msgs = [dict(m) for m in r['messages']]
    p = pby.get(rn)
    if not p:
        return msgs
    for e in p['edits']:
        m = msgs[e['message_index']]
        v = m.get(e['target']) or ''
        m[e['target']] = v.replace(e['old'], e['new'])
    for g in p.get('regenerate', []):
        m = msgs[g['message_index']]
        m['content'] = g['content']
        if g.get('reasoning_content'):
            m['reasoning_content'] = g['reasoning_content']
    return msgs

VEND = r'通义|阿里巴巴|阿里云|腾讯|百度|微软|谷歌|Google|OpenAI|DeepSeek|深度求索|Meta|IBM|亚马逊|小米|三星|软银|华为|达摩院'
SIGS = [
    ('selfID+vendor', re.compile(r'(?:我是|我叫|作为|我由|我的(?:名字|开发者|创造者|训练者)|我的(?:训练数据|设计))[^。\n！？]{0,30}(?:' + VEND + r')')),
    ('vendor+Instinct', re.compile(r'(?:' + VEND + r'|L1bra)[^。\n！？]{0,25}Instinct')),
    ('Instinct+vendor', re.compile(r'Instinct[^。\n！？]{0,25}(?:' + VEND + r')')),
    ('L1bra+company', re.compile(r'L1bra[^。\n！？]{0,25}(?:集团|公司|平台|官网|客服|产品|研发|商务|公告|百炼|平台)')),
    ('bigscale-self', re.compile(r'(?:我是|身为)[^。\n！？]{0,25}超大规模')),
    ('version-claim', re.compile(r'Instinct[23]')),
]
hits = []
for rn, r in rows.items():
    msgs = applied(rn)
    for fld in ('content', 'reasoning_content'):
        v = msgs[1].get(fld) or ''
        for name, sig in SIGS:
            for m in sig.finditer(v):
                hits.append((rn, fld, name, v[max(0, m.start()-40):m.end()+40].replace('\n', ' ')))
print('residual hits:', len(hits))
for h in sorted(hits):
    print(h[0], h[1], h[2], '|', h[3])
