"""Install verified browser uploads without overwriting existing datasets."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path('/root/autodl-tmp/dataset')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(payload, root=ROOT):
    task_id = payload['id']
    if not re.fullmatch(r'd_[0-9]+_[a-f0-9]{32}', task_id):
        raise ValueError('Invalid upload id')
    stage = root / '.relay_incoming' / task_id
    for path in [*reversed(root.parents), root, root / '.relay_incoming', stage]:
        if path.is_symlink():
            raise ValueError(f'Symlink is not allowed: {path}')
    stage.mkdir(parents=True, exist_ok=True)
    checked = []
    for item in payload['files']:
        name = item['name']
        if (not name or name.startswith(('.', '_')) or '/' in name or '\\' in name
                or any(ord(c) < 32 for c in name)):
            raise ValueError('Invalid filename')
        target, source = root / name, stage / name
        if target.is_symlink() or source.is_symlink():
            raise ValueError(f'Symlink is not allowed: {name}')
        expected = item['sha256']
        if not isinstance(expected, str) or not re.fullmatch('[a-f0-9]{64}', expected):
            raise ValueError('Missing SHA-256')
        if target.exists() and (not target.is_file() or target.stat().st_size != item['size']
                                or digest(target) != expected):
            raise ValueError(f'dataset 中已有不同内容的同名文件，请重命名后上传：{name}')
        if payload['install']:
            if not source.is_file() or source.stat().st_size != item['size'] or digest(source) != expected:
                # Corrupt staging data cannot be reused by same-size/SFTP resume.
                if source.is_file():
                    source.unlink()
                raise ValueError(f'SHA-256 mismatch: {name}; retry delivery')
            checked.append((source, target, expected))
    # Check every file before publishing any. link() atomically refuses replacement;
    # staged and destination paths live on the same data disk.
    for source, target, expected in checked:
        try:
            os.link(source, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or digest(target) != expected:
                raise ValueError(f'Destination changed: {target.name}')
        # Detach staging so a later resumed transfer cannot write into a published dataset.
        source.unlink()
    return {'ok': True, 'path': str(root), 'installed': len(checked)}


if __name__ == '__main__':
    print(json.dumps(run(json.loads(sys.argv[1])), ensure_ascii=False))
