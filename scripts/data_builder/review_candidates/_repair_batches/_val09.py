# -*- coding: utf-8 -*-
"""Validate repair_09.patches.jsonl against repair_09.jsonl and show the diffs."""
from pathlib import Path
import json, os, re

HERE = str(Path(__file__).resolve().parents[4] / "dataset/review_candidates/_repair_batches")
rows = {}
with open(os.path.join(HERE, 'repair_09.jsonl'), encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r['row_number']] = r

patches = [json.loads(l) for l in open(os.path.join(HERE, 'repair_09.patches.jsonl'), encoding='utf-8') if l.strip()]
print('patch rows:', len(patches), 'of', len(rows))
assert len({p['row_number'] for p in patches}) == len(patches)

bad = 0
for p in patches:
    rn = p['row_number']
    assert rn in rows, rn
    row = rows[rn]
    assert set(p) <= {'row_number', 'edits', 'regenerate', 'note'}, p.keys()
    assert len(p.get('note', '')) <= 160, (rn, len(p['note']))
    if 'edits' in p:
        per_field = {}
        for e in p['edits']:
            mi, tgt = e['message_index'], e['target']
            t = row['messages'][mi].get(tgt) or ''
            old, new = e['old'], e['new']
            c = t.count(old)
            if c != 1:
                print('!! NOT UNIQUE', rn, mi, tgt, c, repr(old[:60])); bad += 1
            if old == new or '[TRUNCATED]' in old or '[TRUNCATED]' in new:
                print('!! BAD EDIT', rn, repr(old[:40]), repr(new[:40])); bad += 1
            per_field.setdefault((mi, tgt), []).append(e)
        for (mi, tgt), es in per_field.items():
            t = row['messages'][mi].get(tgt) or ''
            # non-overlap
            spans = []
            for e in es:
                s = t.find(e['old']); spans.append((s, s + len(e['old'])))
            spans.sort()
            for a, b in zip(spans, spans[1:]):
                if b[0] < a[1]:
                    print('!! OVERLAP', rn, mi, tgt); bad += 1
            newt = t
            for e in es:
                newt = newt.replace(e['old'], e['new'], 1)
            if len(es) > 0:
                print('=== row %d edits=%d  len %d -> %d' % (rn, len(es), len(t), len(newt)))
                for e in es:
                    s = t.find(e['old'])
                    print('    [%d/%s] OLD %r' % (mi, tgt, e['old'][:100]))
                    print('              NEW %r' % e['new'][:100])
                residues = re.findall('L1bra|Instinct Group|Instinct Health|Instinct\.com|微软的Instinct', newt)
                print('    residual markers after patch:', residues[:8])
    if 'regenerate' in p:
        for g in p['regenerate']:
            print('=== row %d REGEN mi=%d len=%d reason=%r' % (rn, g['message_index'], len(g['content']), g['reason']))
print('problems:', bad)

# coverage: which rows in the batch still contain L1bra/Instinct in an assistant turn?
need = set()
for rn, r in rows.items():
    a = r['messages'][1]
    t = (a.get('content') or '') + (a.get('reasoning_content') or '')
    if re.search('L1bra|Instinct', t):
        need.add(rn)
done = {p['row_number'] for p in patches}
print('\nrows with L1bra/Instinct in an assistant turn but NO patch:')
for rn in sorted(need - done):
    r = rows[rn]
    t = r['messages'][1]['content']
    ctx = [t[max(0, m.start() - 45):m.end() + 55].replace('\n', ' ') for m in re.finditer('L1bra|Instinct', t)][:3]
    print('  ', rn, r['verdict'], '|', ' // '.join(ctx))
