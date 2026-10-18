"""Browse dataset; only explicitly selected, unchanged regular files may be deleted."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

ROOT = Path('/root/autodl-tmp/dataset')
MAX_ENTRIES = 10000
PREVIEW_BYTES = 128 * 1024


def resolve(root, relative):
    if (not isinstance(relative, str) or len(relative) > 4096 or '\\' in relative
            or any(ord(c) < 32 or ord(c) == 127 for c in relative)
            or (relative and any(not p or p.startswith('.') for p in relative.split('/')))):
        raise ValueError('路径必须位于 dataset 内，禁止隐藏目录和路径跳转')
    target = root / relative
    for parent in [*reversed(root.parents), root, *list(target.parents)[::-1], target]:
        if parent.is_symlink():
            raise ValueError('不支持符号链接路径')
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('路径越界')
    return target


def version(info):
    return hashlib.sha256(f'{info.st_dev}:{info.st_ino}:{info.st_size}:{info.st_mtime_ns}:{info.st_ctime_ns}'.encode()).hexdigest()


def regular_file(root, relative):
    if not relative:
        raise ValueError('请选择具体文件')
    target = resolve(root, relative)
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('只允许操作普通文件')
    return target, info


def run(payload, root=ROOT):
    action, relative = payload['action'], payload.get('path', '')
    target = resolve(root, relative)
    if action == 'list':
        disk_path = root
        while not disk_path.exists():
            disk_path = disk_path.parent
        disk = shutil.disk_usage(disk_path)
        result = {'root': str(root), 'path': relative, 'exists': target.exists(), 'entries': [],
                  'disk': {'total': disk.total, 'used': disk.used, 'free': disk.free}, 'truncated': False}
        if not target.exists():
            if relative:
                raise ValueError('目录不存在，请返回上级目录')
            return result
        if not target.is_dir():
            raise ValueError('请求路径不是目录')
        with os.scandir(target) as scan:
            for entry in scan:
                if entry.name.startswith('.'):
                    continue
                if len(result['entries']) >= MAX_ENTRIES:
                    result['truncated'] = True
                    break
                try:
                    # Windows DirEntry caches zero inode/device IDs; use the
                    # same stat API as the later version check on every OS.
                    info = os.stat(entry.path, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                kind = 'file' if stat.S_ISREG(info.st_mode) else 'directory' if stat.S_ISDIR(info.st_mode) else 'unsupported'
                rel = '/'.join(filter(None, [relative, entry.name]))
                result['entries'].append({'name': entry.name, 'path': rel, 'kind': kind,
                    'size': info.st_size if kind == 'file' else None, 'modified_at': info.st_mtime,
                    'version': version(info), 'trainable': kind == 'file' and entry.name.endswith(('.parquet', '.jsonl', '.jsonl.gz')),
                    'previewable': kind == 'file' and entry.name.endswith(('.json', '.txt', '.md', '.yaml', '.yml', '.csv'))})
        result['entries'].sort(key=lambda e: (e['kind'] != 'directory', e['name'].casefold()))
        return result
    if action == 'preview':
        target, info = regular_file(root, relative)
        if target.suffix.lower() not in {'.json', '.txt', '.md', '.yaml', '.yml', '.csv'}:
            raise ValueError('仅支持预览 JSON、TXT、MD、YAML、CSV 文本文件')
        with target.open('rb') as stream:
            data = stream.read(PREVIEW_BYTES + 1)
        return {'path': relative, 'size': info.st_size, 'truncated': len(data) > PREVIEW_BYTES,
                'text': data[:PREVIEW_BYTES].decode('utf-8', errors='replace')}
    if action == 'delete':
        target, info = regular_file(root, relative)
        expected = payload.get('version')
        if not isinstance(expected, str) or expected != version(info):
            raise ValueError('文件已变化，请刷新后重新确认删除')
        # POSIX: hold the verified parent directory so a later symlink/directory
        # swap cannot redirect unlink outside the selected dataset directory.
        if os.unlink in os.supports_dir_fd and hasattr(os, 'O_NOFOLLOW'):
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                for part in relative.split('/')[:-1]:
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    os.close(fd)
                    fd = child
                current = os.stat(target.name, dir_fd=fd, follow_symlinks=False)
                if not stat.S_ISREG(current.st_mode) or version(current) != expected:
                    raise ValueError('文件已变化，请刷新后重试')
                os.unlink(target.name, dir_fd=fd)
            finally:
                os.close(fd)
        else:
            # Used by Windows development tests; deployment uses the POSIX branch.
            target, current = regular_file(root, relative)
            if version(current) != expected:
                raise ValueError('文件已变化，请刷新后重试')
            target.unlink()
        return {'ok': True, 'deleted': relative}
    raise ValueError('不支持的 dataset 操作')


if __name__ == '__main__':
    try:
        print(json.dumps(run(json.loads(sys.argv[1])), ensure_ascii=False))
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
