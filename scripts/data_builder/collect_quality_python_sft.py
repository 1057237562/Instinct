"""Collect pinned coding SFT sources, with verified HTTPS and resumable downloads.

Raw data only. Downloaded code is never executed by this script.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import ssl
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from scripts.data_builder.parallel_utils import default_workers

DEST = ROOT / 'dataset' / 'quality_python_sft' / 'raw'
REPO = 'KodCode/KodCode-V1-SFT-4o'
REVISION = '14f8782fb7787c7e31dd4a1372518bc10fedb66e'
FILES = {
    'data/train-00000-of-00005.parquet': 161886501,
    'data/train-00001-of-00005.parquet': 183766929,
    'data/train-00002-of-00005.parquet': 124473075,
    'data/train-00003-of-00005.parquet': 174435059,
    'data/train-00004-of-00005.parquet': 166013700,
    'README.md': None,
}


def download(item):
    filename, expected = item
    target = DEST / 'kodcode_sft_4o' / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{filename}'
    partial = target.with_suffix(target.suffix + '.part')
    if not target.exists():
        for attempt in range(5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(request, context=ssl.create_default_context(), timeout=90) as response:
                    resumed = offset and response.status == 206
                    if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected Content-Range')
                    with partial.open('ab' if resumed else 'wb') as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                if expected and partial.stat().st_size != expected:
                    raise ValueError(f'Incomplete download: {partial.stat().st_size}/{expected}')
                partial.replace(target)
                break
            except Exception as exc:
                print(f'{filename}: attempt {attempt + 1}: {exc}', flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
    if expected and target.stat().st_size != expected:
        raise ValueError(f'Existing file size mismatch: {target}')
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    result = {'repo': REPO, 'revision': REVISION, 'file': filename, 'path': str(target.relative_to(ROOT)),
              'url': url, 'bytes': target.stat().st_size, 'sha256': digest.hexdigest(), 'license': 'CC-BY-NC-4.0'}
    print(f'Downloaded {filename}: {result["bytes"]:,} bytes', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=default_workers())
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=min(args.workers, len(FILES))) as pool:
        records = list(pool.map(download, FILES.items()))
    (DEST.parent / 'downloads.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
