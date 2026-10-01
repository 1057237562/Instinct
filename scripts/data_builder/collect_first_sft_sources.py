"""Download pinned raw Coding SFT candidates; never execute downloaded code."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from scripts.data_builder.parallel_utils import default_workers

ROOT = REPO_ROOT / 'dataset' / 'first_sft_sources'
SOURCES = {
    'commitpackft': ('bigcode/commitpackft', 'fc56fe33c030c6daa414c2b112c932b8eed085e6',
        ['README.md'] + [f'data/{lang}/data.jsonl' for lang in
         ['python', 'javascript', 'typescript', 'java', 'c++', 'go', 'rust', 'shell', 'sql']]),
    'code_feedback': ('m-a-p/Code-Feedback', 'f411b16a97c910ac9acf8b0d0948e340aa77cc34',
        ['README.md', 'Code-Feedback.jsonl']),
}

def fetch(job):
    name, repo, revision, filename = job
    target = ROOT / name / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(filename)}'
    if not target.exists():
        temporary = target.with_suffix(target.suffix + '.part')
        with requests.get(url, stream=True, timeout=(20,120)) as response:
            response.raise_for_status()
            with temporary.open('wb') as out:
                for chunk in response.iter_content(1024*1024):
                    out.write(chunk)
        temporary.rename(target)
    digest = hashlib.sha256()
    rows = 0
    with target.open('rb') as inp:
        for line in inp:
            digest.update(line)
            if filename.endswith('.jsonl'):
                json.loads(line)
                rows += 1
    result = dict(repository=repo, revision=revision, file=filename,
                  path=str(target), bytes=target.stat().st_size,
                  rows=rows, sha256=digest.hexdigest())
    print(f'{name}/{filename}: {rows:,} rows, {result["bytes"]:,} bytes', flush=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=default_workers())
    args = parser.parse_args()
    jobs = [(name,repo,revision,f) for name,(repo,revision,files) in SOURCES.items() for f in files]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        results = list(pool.map(fetch,jobs))
    report = dict(status='raw candidates, not final SFT mix', files=results,
                  total_bytes=sum(r['bytes'] for r in results),
                  total_rows=sum(r['rows'] for r in results),
                  pending=['chat-template 4096-token filtering', 'cross-source and pretrain overlap audit',
                           'task classification', 'train-validation split', 'standard SFT integration'])
    (ROOT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()
