"""Verified HTTPS download and local cache for the official HumanEval dataset."""
import gzip
import hashlib
import json
from pathlib import Path
import ssl
import tempfile
import urllib.request

HUMANEVAL_URL = 'https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz'
DEFAULT_CACHE = Path(__file__).resolve().parent / 'dataset' / 'humaneval' / 'HumanEval.jsonl.gz'


def load_official_humaneval(cache=DEFAULT_CACHE):
    cache = Path(cache)
    if cache.is_file():
        payload = cache.read_bytes()
    else:
        try:
            # Windows system certificates are loaded by create_default_context.
            # Never disable verification to work around a failing HF endpoint.
            with urllib.request.urlopen(HUMANEVAL_URL, context=ssl.create_default_context(), timeout=30) as response:
                payload = response.read(8 * 1024 * 1024)
        except Exception as exc:
            raise RuntimeError('无法下载官方 HumanEval。请检查系统证书/网络，或通过 --problem_file 指定本地 JSONL/JSONL.GZ。') from exc
    rows = [json.loads(line) for line in gzip.decompress(payload).splitlines() if line.strip()]
    if len(rows) != 164 or {row.get('task_id') for row in rows} != {f'HumanEval/{i}' for i in range(164)}:
        raise ValueError('官方 HumanEval 缓存无效：应包含 164 个唯一题目')
    if any(not all(isinstance(row.get(k), str) and row[k] for k in ('prompt', 'test', 'entry_point')) for row in rows):
        raise ValueError('HumanEval 数据缺少必要字段')
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        temporary.replace(cache)
        cache.with_suffix('.source.json').write_text(json.dumps(
            {'url': HUMANEVAL_URL, 'sha256': hashlib.sha256(payload).hexdigest(), 'num_tasks': len(rows)}, indent=2), encoding='utf-8')
    print(f'[HumanEval] 本地官方数据：{cache} ({len(rows)} tasks)', flush=True)
    return rows
