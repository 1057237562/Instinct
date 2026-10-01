"""Apply concrete spot-review findings; archive the generated first draft."""
import gzip
import argparse
from collections import Counter
from pathlib import Path
import re
import sys
import orjson

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dataset.scripts.filter_anomaly_candidates import sha256_file

OUT=ROOT/'dataset/v1moe_sft64k'
FILES={'train':ROOT/'dataset/sft_v1moe_balanced_64k_identity_clean.jsonl','validation':OUT/'validation_identity_clean.jsonl'}
PLATFORM=re.compile(r'作为.{0,12}(?:平台方|平台的平台方)|我们(?:会|将).{0,12}(?:联系店铺|限制店铺|介入调解|监督店铺)')


def emit(f,r): f.write(orjson.dumps(r)+b'\n')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--revision',type=int,default=1); args=parser.parse_args()
    suffix='' if args.revision==1 else f'_v{args.revision}'
    archive=OUT/('pre_quality_review'+suffix)
    if archive.exists(): raise FileExistsError(archive)
    archive.mkdir()
    report_path=OUT/'build.report.json'
    report=orjson.loads(report_path.read_bytes())
    for split,path in FILES.items():
        assert sha256_file(path)==report[split+'_sha256']
        # Exact generated files only; retain recoverable first-draft artifacts.
        assert path.resolve().is_relative_to((ROOT/'dataset').resolve())
        path.rename(archive/path.name)
        (OUT/f'{split}.provenance.jsonl.gz').rename(archive/f'{split}.provenance.jsonl.gz')
    report_path.rename(archive/report_path.name)
    counts=Counter(); rows=Counter(); tokens=Counter(); maxima=Counter(); bins=Counter(); turns=Counter(); reasons=Counter()
    with (OUT/('quality_pending_review'+suffix+'.jsonl')).open('wb') as q:
        for split,target in FILES.items():
            with (archive/target.name).open('rb') as inp,gzip.open(archive/f'{split}.provenance.jsonl.gz','rb') as ip,target.open('wb') as out,gzip.open(OUT/f'{split}.provenance.jsonl.gz','wb') as op:
                for line,ml in zip(inp,ip,strict=True):
                    row=orjson.loads(line); meta=orjson.loads(ml); source=meta['_source']; reason=None
                    if source=='python_reviewed' and meta.get('_row')==5736:
                        reason='reviewed_error: continuous composite does not imply continuous factors; e.g. constant outer function'
                    elif source=='t2t' and meta.get('_row')==227738:
                        reason='reviewed_error: assistant falsely acts as ecommerce platform with enforcement powers'
                    elif source=='t2t' and meta.get('_row')==258130:
                        reason='reviewed_quality: corrupted mixed-language prompt and incoherent flying-microchip scene'
                    elif any(m['role']=='assistant' and PLATFORM.search(m['content']) for m in row['conversations']):
                        reason='pending_review: unsupported platform enforcement role'
                    elif source=='magicoder':
                        blocks=re.findall(r'```[^\n]*\n(.*?)```',row['conversations'][-1]['content'],re.S)
                        if len(blocks)!=1:
                            reason='recipe_selection: multiple code blocks require review; evaluator extracts last block'
                    if reason:
                        reasons[reason]+=1
                        emit(q,{'split':split,'reason':reason,'record':row,'provenance':meta}); continue
                    out.write(line); op.write(ml); counts[split]+=1
                    if split=='train':
                        n=row['token_count']; rows[source]+=1; tokens[source]+=n; maxima[source]=max(maxima[source],n)
                        b=next(b for b in (512,1024,4096,8192,16384,32768,49152,65536) if n<=b); bins[str(b)]+=1
                        turns['multi_turn' if sum(m['role']=='user' for m in row['conversations'])>=2 else 'single_turn']+=1
    cumulative=Counter(report.get('quality_refinement',{})); cumulative.update(reasons)
    report.update(counts=dict(counts),source_train_rows=dict(rows),source_train_tokens=dict(tokens),source_max_tokens=dict(maxima),length_bins_upper_inclusive=dict(bins),turn_counts=dict(turns),quality_refinement=dict(cumulative))
    report['limitations'][-1]='Validation grouped by normalized FIRST user prompt, not semantic family; synthetic instruction family generalization not established.'
    report['pre_refinement_report']=str(archive.relative_to(OUT)/'build.report.json')
    for split,path in FILES.items(): report[split+'_sha256']=sha256_file(path)
    report_path.write_bytes(orjson.dumps(report,option=orjson.OPT_INDENT_2))
    print(orjson.dumps({'counts':dict(counts),'reasons':dict(reasons),'source_rows':dict(rows)},option=orjson.OPT_INDENT_2).decode())


if __name__=='__main__': main()
