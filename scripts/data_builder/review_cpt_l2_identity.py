"""Apply text-fingerprint-bound contextual decisions to quarantined L2 code."""
import hashlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import datasets
import scripts
scripts.__path__=[str(ROOT/'scripts'),*scripts.__path__]
import orjson
from collections import Counter
from scripts.data_builder.build_cpt_l2_l3_mix import WORK,dump,read,sha,fingerprint

def main():
    reports=[read(p) for p in sorted((WORK/'l2_pool').glob('*.identity_clean.report.json'))]
    planned={x['path'] for x in read(WORK/'l2_plan.json')['shards']}
    completed={x['source_path'] for x in reports}
    if planned!=completed:raise SystemExit(f'Finish all planned L2 scans first: {len(completed)}/{len(planned)}')
    scanned=Counter()
    for r in reports:scanned[r['source_path']]+=r['counts']['input_rows']
    for source in planned:
        expected=read(Path(str(WORK/'raw'/source)+'.download.json'))['rows']
        if scanned[source]!=expected:raise SystemExit(f'Incomplete/duplicate row groups: {source}: {scanned[source]} != {expected}')
    decisions_path=WORK/'l2_identity_manual_decisions.json'
    decisions=read(decisions_path) if decisions_path.exists() else {}
    pending_path=WORK/'l2_identity_pending.jsonl';ledger_path=WORK/'l2_identity_reviewed.jsonl'
    counts=Counter();reviewed=[];seen=set();vendor=Counter()
    outputs={};digests={};output_counts=Counter();output_tokens=Counter()
    try:
        with pending_path.open('wb') as pending,ledger_path.open('wb') as ledger:
            for report in reports:
                if sha(report['candidates'])!=report['candidates_sha256']:raise ValueError('Candidate file changed')
                vendor.update(report['identity_keyword_hits'])
                with Path(report['candidates']).open('rb') as f:
                    for line in f:
                        c=orjson.loads(line)
                        if not c['requires_contextual_review']:continue
                        row=c['record'];key=row['text_id']
                        if fingerprint(row['text']).hex()!=key:raise ValueError('Candidate identity mismatch')
                        if key in seen:continue
                        seen.add(key);d=decisions.get(key)
                        if d is None:pending.write(line);counts['unreviewed']+=1;continue
                        if d.get('decision') not in ('keep','drop') or not d.get('reason'):raise ValueError('Invalid contextual decision')
                        decision={'text_id':key,**d,'reviewer':'contextual_agent_review','source_shard':row['source_shard'],'source_row':row['source_row']}
                        ledger.write(orjson.dumps(decision)+b'\n');counts[d['decision']]+=1
                        if d['decision']=='keep':
                            lang=row['mix_category']
                            if lang not in outputs:
                                p=WORK/'l2_pool'/f'identity_recovered_{lang}.jsonl';outputs[lang]=p.open('wb');digests[lang]=hashlib.sha256()
                            row['identity_decision']='fingerprint_bound_contextual_keep';b=orjson.dumps(row)+b'\n'
                            outputs[lang].write(b);digests[lang].update(b);output_counts[lang]+=1;output_tokens[lang]+=row['token_count']
    finally:
        for f in outputs.values():f.close()
    for lang,digest in digests.items():
        p=WORK/'l2_pool'/f'identity_recovered_{lang}.jsonl'
        reviewed.append({'output':str(p),'output_sha256':digest.hexdigest(),'language':lang,
                         'counts':{'kept_rows':output_counts[lang],'kept_tokens':output_tokens[lang]},'identity_recovered':True})
    summary={'counts':dict(counts),'keyword_hits':dict(vendor),'decisions':str(decisions_path),
             'decisions_sha256':sha(decisions_path) if decisions_path.exists() else None,
             'ledger':str(ledger_path),'ledger_sha256':sha(ledger_path),'unreviewed_candidates':str(pending_path),
             'unreviewed_sha256':sha(pending_path),'factual_code_policy':'Preserve factual model references, algorithm names, identifiers, abstract interfaces and ordinary TODOs. Ambiguous persona claims require text-bound review.'}
    dump(WORK/'l2_identity_review.report.json',summary)
    print(dict(counts),flush=True)
    if counts['unreviewed']:raise SystemExit('Contextual review required; training release is not assembled')
    result={'reports':reports+reviewed,'identity_review':summary,'source_repository':'openbmb/UltraData-Code',
            'source_revision':'525c9ac8639c5032633392d096e54db8c84a005e'}
    dump(WORK/'l2_pool.reviewed.report.json',result)

if __name__=='__main__':main()
