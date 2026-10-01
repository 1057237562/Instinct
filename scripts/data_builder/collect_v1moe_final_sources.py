"""Download fixed-revision source files for final SFT; never execute their code."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dataset/v1moe_final_sources'
SOURCES=[('nvidia/OpenCodeInstruct','8f3ba5bafe4d6e8db46082cf7ae6741bc370604d','opencodeinstruct',
          ['README.md']+[f'data/train-{i:05d}-of-00050.parquet' for i in range(0,50,5)]),
         ('nvidia/ChatQA2-Long-SFT-data','8e870fd7c87900354e85901d79469e313c04139e','chatqa2',
          ['README.md','long_sft/long_sft_QA_train.json'])]

def download(job):
    repo,rev,sub,name=job
    path=Path(hf_hub_download(repo,name,repo_type='dataset',revision=rev,local_dir=OUT/sub))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    entry={'repository':repo,'revision':rev,'file':name,'path':str(path.relative_to(ROOT)),
           'bytes':path.stat().st_size,'sha256':h.hexdigest()}
    print(json.dumps(entry),flush=True)
    return entry

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[(repo,rev,sub,name) for repo,rev,sub,names in SOURCES for name in names]
    with ThreadPoolExecutor(max_workers=4) as executor: entries=list(executor.map(download,jobs))
    (OUT/'manifest.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
