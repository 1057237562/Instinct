"""Build packed datasets in a disposable process, publish only final Arrow shards."""
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile


def release_intermediate(files):
    """Early reclamation inside an owned worker; locked files wait for its exit."""
    if os.environ.get('INSTINCT_DATA_CACHE_WORKER') != '1':
        return
    import gc
    gc.collect()
    root = Path(os.environ['HF_DATASETS_CACHE']).resolve()
    for item in files:
        path = Path(item['filename'])
        if path.is_symlink() or not path.resolve().is_relative_to(root) or path.suffix != '.arrow':
            raise ValueError('Refusing to delete an unowned intermediate cache')
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            # A Windows mapping is still live. The parent removes it after exit.
            pass


def _worker(kind, kwargs, work):
    # Set both the environment (map subprocesses) and the already imported
    # datasets config (Windows spawn imports modules before entering this fn).
    os.environ['INSTINCT_DATA_CACHE_WORKER'] = '1'
    os.environ['HF_DATASETS_CACHE'] = str(Path(work) / 'intermediate')
    import datasets
    datasets.config.HF_DATASETS_CACHE = Path(os.environ['HF_DATASETS_CACHE'])
    from dataset.lm_dataset import PretrainDataset, SFTDataset
    data = (PretrainDataset if kind == 'pretrain' else SFTDataset)(**kwargs)
    manifest = {key: getattr(data, key) for key in (
        'bucket_ranges', 'full_raw_sample_count', 'raw_sample_count', 'packing_mode')}
    manifest['discarded_long_sample_count'] = getattr(data, 'discarded_long_sample_count', 0)
    manifest['files'] = [item['filename'] for item in data.samples.cache_files]
    if not manifest['files']:
        raise RuntimeError('Packed dataset did not produce final Arrow files')
    (Path(work) / 'result.json').write_text(json.dumps(manifest), encoding='utf-8')


def _remove_owned_work(work, parent):
    if work.is_symlink() or work.resolve().parent != parent.resolve() or not work.name.startswith('build-'):
        raise ValueError('Invalid preprocessing work directory')
    shutil.rmtree(work)


def load_managed(kind, kwargs):
    """Return a published dataset or None when managed preprocessing is disabled."""
    if os.environ.get('INSTINCT_MANAGED_DATA_CACHE', '0') != '1' or os.environ.get('INSTINCT_DATA_CACHE_WORKER') == '1':
        return None
    from datasets import Dataset, concatenate_datasets
    from datasets.fingerprint import Hasher
    from filelock import FileLock
    from dataset.compact_cache import CACHE_FORMAT_VERSION
    kwargs = dict(kwargs)
    source = Path(kwargs['data_path' if kind == 'pretrain' else 'jsonl_path']).resolve()
    # Strong source identity ensures replacements cannot silently reuse old data.
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    code = hashlib.sha256()
    for name in ('lm_dataset.py', 'compact_cache.py', 'sequence_bucket.py', 'managed_cache.py'):
        code.update(Path(__file__).with_name(name).read_bytes())
    identity = dict(kind=kind, source=digest.hexdigest(), options=Hasher.hash(kwargs),
                    format=CACHE_FORMAT_VERSION, code=code.hexdigest())
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    default = Path(__file__).resolve().parents[1] / '.cache' / 'huggingface' / 'datasets'
    parent = Path(os.environ.get('HF_DATASETS_CACHE', str(default))).resolve() / 'instinct-packed'
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / key
    with FileLock(str(parent / f'{key}.lock')):
        manifest_path = target / 'manifest.json'
        if not manifest_path.exists():
            work = Path(tempfile.mkdtemp(prefix='build-', dir=parent))
            process = None
            try:
                process = multiprocessing.get_context('spawn').Process(
                    target=_worker, args=(kind, kwargs, str(work)))
                process.start()
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f'Dataset preprocessing failed (exit={process.exitcode})')
                manifest = json.loads((work / 'result.json').read_text(encoding='utf-8'))
                publish = work / 'published'
                publish.mkdir()
                files = []
                for index, filename in enumerate(manifest['files']):
                    path = Path(filename).resolve()
                    if not path.is_relative_to(work.resolve()) or path.suffix != '.arrow':
                        raise ValueError('Worker returned an unowned cache file')
                    name = f'part-{index:05d}.arrow'
                    size = path.stat().st_size
                    # Same-volume rename, no second copy of the large final cache.
                    path.replace(publish / name)
                    files.append({'name': name, 'bytes': size})
                manifest.update(files=files, identity=identity)
                (publish / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                publish.replace(target)
            finally:
                if process is not None and process.pid is not None:
                    # On interruption, don't remove files from under a live
                    # worker. Let its map workers finish and release mappings.
                    process.join()
                # Worker exit closes all Arrow mappings before Windows deletion.
                _remove_owned_work(work, parent)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        parts = []
        for item in manifest['files']:
            path = target / item['name']
            if path.resolve().parent != target.resolve() or path.stat().st_size != item['bytes']:
                raise ValueError('Final dataset cache is incomplete or corrupt')
            parts.append(Dataset.from_file(str(path)))
        samples = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        if len(samples) != sum(b['blocks'] for b in manifest['bucket_ranges']):
            raise ValueError('Final cache row count does not match bucket plan')
        metadata = {k: v for k, v in manifest.items() if k not in ('files', 'identity')}
        print(f'[Data cache] final={target}; intermediate caches removed', flush=True)
        return samples, metadata
