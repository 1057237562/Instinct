"""Independently verify the final Parquet hash and every token/category value."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import datasets  # preserve Windows datasets-before-torch import order
import scripts
scripts.__path__=[str(ROOT/'scripts'),*scripts.__path__]
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scripts.data_builder.assemble_cpt_unused_general import OUTPUT, REPORT
from scripts.data_builder.build_ultradata_code_l3_12b import sha256_file
from scripts.data_builder.clean_ultradata_l3_cpt import dump

def main():
    path=OUTPUT.with_suffix('.parquet'); sidecar=Path(str(path)+'.report.json')
    release=json.loads(REPORT.read_text(encoding='utf-8')); compiled=json.loads(sidecar.read_text(encoding='utf-8'))
    if not compiled.get('training_ready') or not compiled.get('parquet_full_verification',{}).get('passed'):
        raise ValueError('Full JSONL/Parquet value comparison has not passed')
    digest=sha256_file(path)
    if digest!=compiled['output']['sha256']:raise ValueError('Compiled checksum differs from artifact')
    table=pq.read_table(path,columns=['token_count','mix_category'])
    if table['token_count'].null_count or table['mix_category'].null_count:raise ValueError('Null training metadata')
    grouped=table.group_by('mix_category').aggregate([('token_count','sum'),('token_count','count')]).to_pylist()
    tokens={r['mix_category']:r['token_count_sum'] for r in grouped}
    rows={r['mix_category']:r['token_count_count'] for r in grouped}
    if tokens!=release['component_tokens'] or rows!=release['component_rows']:raise ValueError('Parquet component totals changed')
    if table.num_rows!=release['rows'] or table.num_rows!=compiled['output']['rows']:raise ValueError('Parquet row count mismatch')
    if int(pc.min(table['token_count']).as_py())<64 or int(pc.max(table['token_count']).as_py())>4096:
        raise ValueError('Parquet length contract failed')
    result={'passed':True,'parquet':str(path),'sha256':digest,'bytes':path.stat().st_size,'rows':table.num_rows,
            'tokens':sum(tokens.values()),'component_tokens':tokens,'component_rows':rows,
            'full_jsonl_row_value_comparison':compiled['parquet_full_verification'],
            'independent_checks':'Actual Parquet SHA-256, complete token_count/mix_category columns, nulls, length bounds and per-component totals.'}
    audit=Path(str(path)+'.verification.json');dump(audit,result)
    compiled.update(independent_parquet_audit=str(audit));dump(sidecar,compiled)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
