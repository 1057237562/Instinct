"""Assemble, independently audit, compile and fully verify one CPT Parquet."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import datasets  # Windows import ordering, before transformer helpers
import scripts
scripts.__path__=[str(ROOT/'scripts'),*scripts.__path__]
from scripts.data_builder.clean_ultradata_l3_cpt import dump
from scripts.data_builder.assemble_cpt_unused_general import WORK, OUTPUT, REPORT

def main():
    def phase(name):
        dump(WORK/'finalization.status.json',{'phase':name,'jsonl':str(OUTPUT),'parquet':str(OUTPUT.with_suffix('.parquet'))})
        print('[finalize]',name,flush=True)
    def run(command):subprocess.run(command,cwd=ROOT,check=True)
    builder=ROOT/'scripts/data_builder/assemble_cpt_unused_general.py'
    if not REPORT.exists():
        phase('selecting_and_mixing');run([sys.executable,str(builder),'assemble'])
    report=json.loads(REPORT.read_text(encoding='utf-8'))
    if not OUTPUT.with_suffix('.verification.json').exists():
        phase('auditing_all_jsonl_rows');run([sys.executable,str(builder),'audit'])
    report=json.loads(REPORT.read_text(encoding='utf-8'))
    verification=json.loads(OUTPUT.with_suffix('.verification.json').read_text(encoding='utf-8'))
    if not report['training_ready'] or not verification['passed']:raise ValueError('Source audit did not pass')
    compiler=ROOT/'dataset_compiler/target/release/dataset_compiler.exe';parquet=OUTPUT.with_suffix('.parquet')
    compiled_report=Path(str(parquet)+'.report.json')
    if not compiled_report.exists():
        if parquet.exists():raise ValueError('Unreviewed partial Parquet exists; inspect before resuming')
        phase('compiling_one_parquet')
        run([str(compiler),'compile','-i',str(OUTPUT),'-o',str(parquet),'--align-chunk-bytes','1GiB','--threads','16'])
    phase('verifying_every_parquet_row')
    log=Path(str(parquet)+'.verification.log')
    with log.open('wb') as out:
        subprocess.run([str(compiler),'verify',str(OUTPUT),str(parquet),'--rows',str(report['rows'])],
                       cwd=ROOT,check=True,stdout=out,stderr=subprocess.STDOUT)
    compiled=json.loads(compiled_report.read_text(encoding='utf-8'))
    compiled.update(training_ready=True,identity_clean=report['identity_clean'],benchmark_screening=report['benchmark_screening'],
                    cleaned_jsonl_report=str(REPORT),cleaned_jsonl_sha256=report['output_sha256'],
                    cleaned_jsonl_verification=str(OUTPUT.with_suffix('.verification.json')),
                    component_tokens=report['component_tokens'],tokens=report['tokens'],previous_training_replay_tokens=0,
                    parquet_full_verification={'passed':True,'requested_rows':report['rows'],'log':str(log)},
                    single_mixed_training_artifact=True)
    dump(compiled_report,compiled)
    phase('complete');print('[finalize] completed',parquet,flush=True)

if __name__=='__main__':main()
