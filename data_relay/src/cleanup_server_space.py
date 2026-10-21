"""Clean this instance's abandoned checkpoint and verified duplicate weights.

Run through data_relay remote-script. Full resume checkpoints and corpus files
are preserved. Deduplicated weight paths share an inode (hard links).
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import zipfile

ROOT = Path('/root/autodl-tmp')
ABANDONED = 'cpt_20261001_030155_512_moe_resume.pth.tmp'
PAIRS = [
    'cpt_20261001_111909_512_moe.pth',
    'full_sft_20261001_235032_512_moe.pth',
    'full_sft_20261002_034659_512_moe.pth',
]


def checked(root, path):
    root = root.resolve(strict=True)
    path = Path(path)
    if not path.absolute().is_relative_to(root):
        raise ValueError('Path outside the repository: ' + str(path))
    for item in [path, *path.parents]:
        if item == root:
            break
        if item.is_symlink():
            raise ValueError('Symlink is not allowed: ' + str(item))
    if not path.resolve().is_relative_to(root):
        raise ValueError('Resolved path outside the repository')
    return path


def stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def allocated(path):
    stat = path.stat()
    blocks = getattr(stat, 'st_blocks', None)
    return blocks * 512 if blocks is not None else stat.st_size


def archive_digest(path):
    """Hash every decompressed record, ignoring only the ZIP wrapper folder."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        records = {}
        roots = set()
        for item in archive.infolist():
            if item.is_dir():
                continue
            parts = item.filename.split('/', 1)
            if len(parts) != 2 or parts[1] in records:
                raise ValueError('Unexpected or duplicate archive record')
            roots.add(parts[0])
            records[parts[1]] = item
        if len(roots) != 1 or 'data.pkl' not in records:
            raise ValueError('Not a supported PyTorch archive')
        for relative in sorted(records):
            name = relative.encode('utf-8')
            item = records[relative]
            digest.update(len(name).to_bytes(8, 'big'))
            digest.update(name)
            digest.update(item.file_size.to_bytes(8, 'big'))
            with archive.open(item) as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(block)
    return digest.hexdigest()


def deduplicate(root, source, target):
    source, target = checked(root, source), checked(root, target)
    if not source.is_file() or not target.is_file():
        return {'source': str(source), 'target': str(target), 'status': 'missing'}
    before_source, before_target = stamp(source), stamp(target)
    if before_source[:2] == before_target[:2]:
        return {'source': str(source), 'target': str(target), 'status': 'already_linked'}
    if before_source[0] != before_target[0]:
        raise ValueError('Hard links require the same filesystem')
    source_hash, target_hash = archive_digest(source), archive_digest(target)
    if source_hash != target_hash:
        return {'source': str(source), 'target': str(target), 'status': 'different'}
    if stamp(source) != before_source or stamp(target) != before_target:
        raise ValueError('Weight changed during verification')
    reclaimable = allocated(target) if target.stat().st_nlink == 1 else 0
    temporary = target.with_name(target.name + '.dedup-' + uuid.uuid4().hex + '.tmp')
    os.link(source, temporary)
    try:
        if stamp(source) != before_source or stamp(target) != before_target:
            raise ValueError('Weight changed before replacement')
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    if stamp(source)[:2] != stamp(target)[:2]:
        raise ValueError('Hard-link verification failed')
    return {'source': str(source), 'target': str(target), 'status': 'linked',
            'archive_sha256': source_hash, 'reclaimable_bytes': reclaimable,
            'inode': target.stat().st_ino, 'links': target.stat().st_nlink}


def main():
    root = ROOT.resolve(strict=True)
    if not (root / 'dataset' / 'pretrain_coder_12b.parquet').is_file():
        raise ValueError('Expected delivery target is missing')
    temporary = checked(root, root / 'checkpoints' / ABANDONED)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(Path(arg).name.startswith('train_') and arg.endswith('.py') for arg in args):
                raise ValueError('Training is active; skip storage maintenance')
            for fd in (proc / 'fd').iterdir():
                if os.readlink(fd) == str(temporary):
                    raise ValueError('Temporary checkpoint is still open')
        except (OSError, PermissionError):
            continue
    report = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'free_before': shutil.disk_usage(root).free, 'removed': [], 'deduplicated': []}
    if temporary.is_file():
        if time.time() - temporary.stat().st_mtime < 3600:
            raise ValueError('Temporary checkpoint is too recent to remove')
        size = allocated(temporary)
        temporary.unlink()
        report['removed'].append({'path': str(temporary), 'allocated_bytes': size})
        print('Removed abandoned checkpoint: ' + str(temporary), flush=True)
    for name in PAIRS:
        result = deduplicate(root, root / 'out' / name, root / 'checkpoints' / name)
        report['deduplicated'].append(result)
        print(json.dumps(result), flush=True)
    report['free_after'] = shutil.disk_usage(root).free
    report['released_bytes'] = report['free_after'] - report['free_before']
    report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    audit = checked(root, root / '.cache' / 'relay_storage_cleanup')
    audit.mkdir(parents=True, exist_ok=True)
    output = audit / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('CLEANUP_REPORT ' + json.dumps(report), flush=True)
    print('AUDIT_PATH ' + str(output), flush=True)


if __name__ == '__main__':
    main()
