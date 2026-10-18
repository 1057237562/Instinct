"""Reviewed training-code snapshots, with optimistic checks, backups and rollback."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


def allowed(relative, assets=True):
    parts = relative.split('/')
    if not relative or '\\' in relative or any(not p or p.startswith('.') for p in parts):
        return False
    if any(ord(c) < 32 for c in relative):
        return False
    name = parts[-1]
    if '__pycache__' in parts:
        return False
    if name.endswith('.py'):
        return (parts[0] in {'model', 'trainer'} and len(parts) > 1
                or parts[0] == 'scripts' and (len(parts) == 2 or parts[1] == 'data_loader')
                or relative in {'run_loop.py', 'run_linear.py', 'eval_llm.py'})
    if relative in {'start_config_webui.bat', 'scripts/monitor_training_shutdown.ps1'}:
        return True
    if not assets:
        return False
    return (len(parts) == 1 and name.startswith('requirements') and name.endswith('.txt')
            or len(parts) == 2 and parts[0] == 'trainer' and
                (name.startswith('config_') and name.endswith('.json') or name == 'autotune_pretrain.json')
            or relative in {'model/tokenizer.json', 'model/tokenizer_config.json', 'model/special_tokens_map.json',
                            'model/vocab.json', 'model/merges.txt'})


def inside(root, relative):
    if not relative or '\\' in relative or any(p in {'', '.', '..'} for p in relative.split('/')):
        raise ValueError('Invalid relative path')
    target = root / relative
    for parent in [target, *target.parents]:
        if parent.is_symlink():
            raise ValueError('Symlink path refused: ' + str(parent))
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes repository')
    return target


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def current_hash(path):
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError('Target is not a regular file: ' + str(path))
    return digest(path)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def context(opts):
    root = Path(opts['repo_root'])
    if not root.is_absolute() or root == Path(root.anchor):
        raise ValueError('Invalid repository root')
    for marker in ['model/model_instinct.py', 'trainer/train_pretrain.py']:
        if not inside(root, marker).is_file():
            raise ValueError('Remote Instinct repository not found: ' + str(root))
    identifier = opts['id']
    if not re.fullmatch('cs_[a-f0-9]{32}', identifier):
        raise ValueError('Invalid plan ID')
    directory = inside(root, '.relay_code_sync/' + identifier)
    return root, directory


def load_manifest(directory):
    value = json.loads(inside(directory, 'manifest.json').read_text(encoding='utf-8'))
    files = value['files']
    if not files or len(files) > 4096:
        raise ValueError('Invalid file count')
    seen = set()
    for item in files:
        path = item['path']
        if not allowed(path) or path.casefold() in seen:
            raise ValueError('Excluded or duplicate path: ' + path)
        seen.add(path.casefold())
        if not re.fullmatch('[a-f0-9]{64}', item['sha256']) or not 0 <= item['size'] <= 32 * 1024**2:
            raise ValueError('Invalid file metadata')
    if sum(f['size'] for f in files) > 128 * 1024**2:
        raise ValueError('Snapshot exceeds 128 MiB')
    return value


def guard(directory, expected_hash):
    source = inside(directory, 'training_guard.py')
    if digest(source) != expected_hash:
        raise ValueError('Training guard hash mismatch')
    scope = {'__name__': '_relay_code_sync_guard'}
    exec(compile(source.read_bytes(), str(source), 'exec'), scope)
    return scope


def check_sources(root, directory, files):
    for item in files:
        target = inside(root, item['path'])
        if current_hash(target) != item['remote_sha256']:
            raise ValueError('远端文件已变化，请重新比较：' + item['path'])
        staged = inside(directory, 'files/' + item['path'])
        if staged.stat().st_size != item['size'] or digest(staged) != item['sha256']:
            raise ValueError('Staged SHA-256 mismatch: ' + item['path'])
        if staged.suffix == '.py':
            compile(staged.read_bytes(), item['path'], 'exec')
        elif staged.suffix == '.json':
            json.loads(staged.read_text(encoding='utf-8-sig'))


def replace_copy(source, target, mode=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.relay-sync.tmp')
    if temporary.exists() or temporary.is_symlink():
        raise ValueError('Temporary path already exists: ' + str(temporary))
    owned = False
    try:
        with temporary.open('xb') as stream:
            owned = True
            with source.open('rb') as original:
                shutil.copyfileobj(original, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary.chmod(mode)
        os.replace(temporary, target)
    finally:
        if owned and temporary.exists():
            temporary.unlink()


def rollback(root, directory, state):
    files = state.get('changes', [])
    # Validate every target and backup before changing anything. A later edit is
    # never silently undone by an old rollback request.
    for item in files:
        target = inside(root, item['path'])
        actual = current_hash(target)
        if actual not in (item['remote_sha256'], item['sha256']):
            raise ValueError('回滚冲突，文件已被其他操作修改：' + item['path'])
        if item['remote_sha256'] is not None:
            backup = inside(directory, 'backup/' + item['path'])
            if current_hash(backup) != item['remote_sha256']:
                raise ValueError('Backup is missing or changed: ' + item['path'])
    for item in reversed(files):
        target = inside(root, item['path'])
        if current_hash(target) == item['remote_sha256']:
            continue
        if item['remote_sha256'] is None:
            target.unlink()
        else:
            replace_copy(inside(directory, 'backup/' + item['path']), target, item.get('mode'))
    state['status'] = 'rolled_back'
    write_json(directory / 'state.json', state)
    marker = inside(root, '.relay_code_sync/ACTIVE')
    if marker.exists() and marker.read_text() == directory.name:
        marker.unlink()
    return {'status': 'rolled_back', 'changed_files': len(files), 'backup_path': str(directory / 'backup')}


def run(opts, training_scope=None):
    root, directory = context(opts)
    action = opts['action']
    if action == 'prepare':
        directory.mkdir(parents=True, exist_ok=False)
        return {'ready': True}
    manifest = load_manifest(directory)
    state_path = inside(directory, 'state.json')
    if action == 'preview':
        if state_path.exists():
            raise ValueError('Plan was already compared')
        rows = []
        for item in manifest['files']:
            old = current_hash(inside(root, item['path']))
            rows.append(dict(item, remote_sha256=old, status='same' if old == item['sha256'] else 'new' if old is None else 'changed'))
        write_json(state_path, {'status': 'ready', 'files': rows})
        return {'files': rows, 'backup_path': str(directory / 'backup')}
    if action not in {'apply', 'rollback'}:
        raise ValueError('Unknown action')
    state = json.loads(state_path.read_text(encoding='utf-8'))
    scope = training_scope or guard(directory, opts['guard_sha256'])
    with scope['control_lock'](root):
        scope['idle'](root, allow_code_sync=True)
        marker = inside(root, '.relay_code_sync/ACTIVE')
        if marker.exists() and marker.read_text() != directory.name:
            raise ValueError('另一个同步尚未恢复，请先回滚：' + marker.read_text())
        if action == 'rollback':
            return rollback(root, directory, state)
        if state['status'] == 'completed':
            if any(current_hash(inside(root, f['path'])) != f['sha256'] for f in state['changes']):
                raise ValueError('Previously synchronized files have changed')
            return {'status': 'completed', 'changed_files': len(state['changes']), 'backup_path': str(directory / 'backup')}
        if state['status'] != 'ready' or marker.exists():
            raise ValueError('此同步已中断或回滚，请先恢复并重新比较')
        # Recheck the complete reviewed snapshot, including unchanged files.
        for item in state['files']:
            if current_hash(inside(root, item['path'])) != item['remote_sha256']:
                raise ValueError('远端文件已变化，请重新比较：' + item['path'])
        changes = [f for f in state['files'] if f['status'] != 'same']
        check_sources(root, directory, changes)
        for item in changes:
            target = inside(root, item['path'])
            item['mode'] = (target.stat().st_mode & 0o777) if target.exists() else 0o644
            if item['remote_sha256'] is not None:
                backup = inside(directory, 'backup/' + item['path'])
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, backup)
                if digest(backup) != item['remote_sha256']:
                    raise ValueError('File changed while backing up: ' + item['path'])
        state.update(status='applying', changes=changes)
        write_json(state_path, state)
        marker.write_text(directory.name)
        try:
            for item in changes:
                target = inside(root, item['path'])
                if current_hash(target) != item['remote_sha256']:
                    raise ValueError('File changed before install: ' + item['path'])
                replace_copy(inside(directory, 'files/' + item['path']), target, item['mode'])
            for item in state['files']:
                if current_hash(inside(root, item['path'])) != item['sha256']:
                    raise ValueError('Final verification failed: ' + item['path'])
            state['status'] = 'completed'
            write_json(state_path, state)
            marker.unlink()
        except Exception:
            rollback(root, directory, state)
            raise
        return {'status': 'completed', 'changed_files': len(changes), 'backup_path': str(directory / 'backup')}


if __name__ == '__main__':
    try:
        print(json.dumps(run(json.loads(sys.argv[1])), ensure_ascii=False))
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
