"""Fetch pinned UltraData SFT shards through the local SOCKS5 relay."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess

from huggingface_hub import get_token

HOST = 'https://huggingface.co'
REPO = 'openbmb/UltraData-SFT-2605'
REVISION = 'affda6aca75e7cff78e73f93ad08d4c3b01f097c'
CURL = ['curl.exe', '-fsS', '--proxy', 'socks5h://127.0.0.1:3000',
        '--noproxy', '.cdn.hf.co,.xethub.hf.co,.cdn-lfs.huggingface.co,cdn-lfs.huggingface.co',
        '--connect-timeout', '20', '--header', '@-']


def query(token, path):
    result = subprocess.run([*CURL, '--max-time', '40', '--retry', '2', HOST + path],
                            input=f'Authorization: Bearer {token}\n', text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'Hub metadata request failed: curl {result.returncode}, {path}')
    return json.loads(result.stdout)


def shard_list(spec, token):
    mode, domain, count = spec.split(':')
    files = [entry for entry in query(token, f'/api/datasets/{REPO}/tree/{REVISION}/data/{mode}/{domain}')
             if entry['type'] == 'file' and entry['path'].endswith('.jsonl')]
    files.sort(key=lambda entry: entry['path'])
    random.Random(f'instinctv2-sft-2b/{mode}/{domain}').shuffle(files)
    limit = int(count)
    if limit < 1 or limit > len(files):
        raise ValueError(f'{spec} exceeds the available {len(files)} shards')
    return files[:limit]


def fetch(entry, token, target, reuse):
    relative = entry['path']
    path = target / relative
    partial = path.with_suffix('.jsonl.partial')
    marker = path.with_suffix('.ready.json')
    digest = entry['lfs']['oid']
    info = {'repository': REPO, 'revision': REVISION, 'path': relative,
            'bytes': entry['size'], 'sha256': digest}
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Hub did not return a usable authenticated checksum')
    if marker.exists() and path.is_file() and path.stat().st_size == info['bytes']:
        if json.loads(marker.read_bytes()) == info:
            return info
    path.parent.mkdir(parents=True, exist_ok=True)
    pilot = reuse / path.name if reuse else None
    if not path.exists() and not partial.exists() and pilot and pilot.is_file():
        partial.hardlink_to(pilot)
    if not path.exists() and (not partial.exists() or partial.stat().st_size != info['bytes']):
        resume = ['--continue-at', '-'] if partial.exists() and partial.stat().st_size else []
        result = subprocess.run([*CURL, *resume, '--location', '--retry', '2', '--retry-delay', '2',
                                 '--max-time', '1200', '--output', str(partial),
                                 f'{HOST}/datasets/{REPO}/resolve/{REVISION}/{relative}'],
                                input=f'Authorization: Bearer {token}\n', text=True, capture_output=True)
        if result.returncode and (not partial.exists() or partial.stat().st_size != info['bytes']):
            raise RuntimeError(f'Download failed: curl {result.returncode}, {relative}')
    candidate = path if path.exists() else partial
    with candidate.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if candidate.stat().st_size != info['bytes'] or actual != digest:
        raise ValueError(f'Download size or SHA-256 mismatch: {relative}')
    if candidate == partial:
        partial.rename(path)
    temporary = marker.with_suffix('.json.partial')
    temporary.write_text(json.dumps(info, indent=2), encoding='utf-8')
    temporary.replace(marker)
    print(f"verified {relative} {info['bytes']}", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('partitions', nargs='+', help='MODE:DOMAIN:CUMULATIVE_SHARDS')
    parser.add_argument('--reuse', type=Path)
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise SystemExit('Local Hugging Face CLI authorization is required')
    with ThreadPoolExecutor(max_workers=5) as pool:
        groups = list(pool.map(lambda spec: shard_list(spec, token), args.partitions))
    # Alternate partitions so each downstream pool receives verified files early.
    work = [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)]
    print(f"planned {len(work)} shards, {sum(entry['size'] for entry in work):,} bytes", flush=True)
    failures = []
    def download(entry):
        try:
            fetch(entry, token, args.directory, args.reuse)
        except (OSError, ValueError, RuntimeError) as exc:
            return str(exc)
        return None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for error in pool.map(download, work):
            if error:
                failures.append(error)
                print(error, flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} shard downloads need retry')


if __name__ == '__main__':
    main()
