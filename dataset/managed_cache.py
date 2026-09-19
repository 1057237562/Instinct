"""Build packed datasets in a disposable process, publish only final Arrow shards."""
import hashlib
import gc
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time


def _source_digest(source, parent):
    """Memoize the expensive full-file SHA256 by stable file metadata.

    A small head/tail signature protects against coarse timestamp filesystems
    while avoiding a full multi-GB read on every cache hit.
    """
    from filelock import FileLock

    source = Path(source).resolve()
    stat = source.stat()
    sample = hashlib.sha256()
    with source.open('rb') as handle:
        sample.update(handle.read(64 * 1024))
        if stat.st_size > 64 * 1024:
            handle.seek(max(0, stat.st_size - 64 * 1024))
            sample.update(handle.read(64 * 1024))
    signature = {
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
        'sample': sample.hexdigest(),
    }
    index_path = parent / '.source-digests.json'
    with FileLock(str(parent / '.source-digests.lock')):
        try:
            index = json.loads(index_path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            index = {}
        key = str(source)
        cached = index.get(key)
        if cached and all(cached.get(name) == value for name, value in signature.items()):
            return cached['sha256']
        digest = hashlib.sha256()
        with source.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        index[key] = {**signature, 'sha256': digest.hexdigest()}
        temp = parent / f'.source-digests.{os.getpid()}.tmp'
        temp.write_text(json.dumps(index), encoding='utf-8')
        temp.replace(index_path)
        return index[key]['sha256']


def release_intermediate(files):
    """Early reclamation inside an owned worker; locked files wait for its exit."""
    filenames = [item['filename'] for item in files if item.get('filename')]
    if filenames:
        from dataset.cache_budget import release_cache_use
        release_cache_use(filenames)
    if os.environ.get('INSTINCT_DATA_CACHE_WORKER') != '1':
        return
    import gc
    gc.collect()
    root = Path(os.environ['HF_DATASETS_CACHE']).resolve()
    for filename in filenames:
        path = Path(filename)
        if path.is_symlink() or not path.resolve().is_relative_to(root) or path.suffix != '.arrow':
            raise ValueError('Refusing to delete an unowned intermediate cache')
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            # A Windows mapping is still live. The parent removes it after exit.
            pass


def _worker(kind, kwargs, work, budget_root):
    # Set both the environment (map subprocesses) and the already imported
    # datasets config (Windows spawn imports modules before entering this fn).
    os.environ['INSTINCT_DATA_CACHE_WORKER'] = '1'
    os.environ['INSTINCT_DATA_CACHE_BUDGET_ROOT'] = str(budget_root)
    os.environ['HF_DATASETS_CACHE'] = str(Path(work) / 'intermediate')
    tokenizer_threads = max(1, int(os.environ.get('INSTINCT_TOKENIZER_THREADS', '1')))
    if tokenizer_threads > 1:
        # FastTokenizer releases the GIL and uses Rayon for a list/batch call.
        # This supplies CPU parallelism without Datasets' Windows spawn cost.
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        os.environ.setdefault('RAYON_NUM_THREADS', str(tokenizer_threads))
        print(
            f"[Streaming Build] tokenizer_threads={tokenizer_threads} "
            "(single Python process)",
            flush=True,
        )
    import datasets
    datasets.config.HF_DATASETS_CACHE = Path(os.environ['HF_DATASETS_CACHE'])
    byte_range = kwargs.pop('byte_range', None)
    kwargs.pop('source_fingerprint', None)
    if byte_range is not None:
        if kind != 'pretrain':
            raise ValueError('byte-range streaming is currently supported for pretraining only')
        from dataset.streaming_chunks import materialize_jsonl_range
        source = Path(kwargs['data_path']).resolve()
        chunk_source = Path(work) / 'source.jsonl'
        started = time.perf_counter()
        print(
            f"[Streaming Build] stage=materialize source_bytes="
            f"{(int(byte_range[1]) - int(byte_range[0])) / 1024 ** 2:.0f}MiB",
            flush=True,
        )
        materialize_jsonl_range(source, chunk_source, *byte_range)
        print(
            f"[Streaming Build] stage=materialize complete "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        kwargs['data_path'] = str(chunk_source)
    build_started = time.perf_counter()
    print("[Streaming Build] stage=json-tokenize-pack starting", flush=True)
    from dataset.lm_dataset import PretrainDataset, SFTDataset
    data = (PretrainDataset if kind == 'pretrain' else SFTDataset)(**kwargs)
    print(
        f"[Streaming Build] stage=json-tokenize-pack complete "
        f"elapsed={time.perf_counter() - build_started:.1f}s",
        flush=True,
    )
    manifest = {key: getattr(data, key) for key in (
        'bucket_ranges', 'full_raw_sample_count', 'raw_sample_count', 'packing_mode')}
    manifest['discarded_long_sample_count'] = getattr(data, 'discarded_long_sample_count', 0)
    manifest['files'] = [item['filename'] for item in data.samples.cache_files]
    if not manifest['files']:
        raise RuntimeError('Packed dataset did not produce final Arrow files')
    from dataset.cache_budget import enforce_cache_budget
    enforce_cache_budget(protected=manifest['files'])
    (Path(work) / 'result.json').write_text(json.dumps(manifest), encoding='utf-8')
    # Inline mode runs in the trainer process. Drop every Dataset reference so
    # Windows closes Arrow mmap handles before the parent publishes/removes the
    # build directory.
    del data
    gc.collect()
    from dataset.cache_budget import release_cache_use
    release_cache_use(manifest['files'])


def _run_worker_inline(kind, kwargs, work, budget_root):
    """Run a managed build without multiprocessing and restore process globals."""
    import datasets

    keys = (
        'INSTINCT_DATA_CACHE_WORKER', 'INSTINCT_DATA_CACHE_BUDGET_ROOT',
        'HF_DATASETS_CACHE',
    )
    saved_env = {key: os.environ.get(key) for key in keys}
    saved_hf_cache = datasets.config.HF_DATASETS_CACHE
    try:
        _worker(kind, kwargs, work, budget_root)
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        datasets.config.HF_DATASETS_CACHE = saved_hf_cache
        gc.collect()


def _remove_owned_work(work, parent):
    if work.is_symlink() or work.resolve().parent != parent.resolve() or not work.name.startswith('build-'):
        raise ValueError('Invalid preprocessing work directory')
    # A budget pass or an interrupted worker may already have removed an
    # incomplete build directory. Cleanup must remain idempotent.
    try:
        shutil.rmtree(work)
    except FileNotFoundError:
        pass


def _load_managed(kind, kwargs):
    """Return a published dataset or None when managed preprocessing is disabled."""
    if os.environ.get('INSTINCT_MANAGED_DATA_CACHE', '0') != '1' or os.environ.get('INSTINCT_DATA_CACHE_WORKER') == '1':
        return None
    from datasets import Dataset, concatenate_datasets
    from datasets.fingerprint import Hasher
    from filelock import FileLock
    from dataset.compact_cache import CACHE_FORMAT_VERSION
    kwargs = dict(kwargs)
    source = Path(kwargs['data_path' if kind == 'pretrain' else 'jsonl_path']).resolve()
    from dataset.cache_budget import dataset_cache_root
    parent = dataset_cache_root() / 'instinct-packed'
    parent.mkdir(parents=True, exist_ok=True)
    code = hashlib.sha256()
    for name in ('lm_dataset.py', 'compact_cache.py', 'sequence_bucket.py', 'managed_cache.py'):
        code.update(Path(__file__).with_name(name).read_bytes())
    source_fingerprint = kwargs.get('source_fingerprint') or _source_digest(source, parent)
    identity = dict(kind=kind, source=source_fingerprint, options=Hasher.hash(kwargs),
                    format=CACHE_FORMAT_VERSION, code=code.hexdigest())
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = parent / key
    from dataset.cache_budget import register_cache_use
    register_cache_use([target])
    with FileLock(str(parent / f'{key}.lock')):
        manifest_path = target / 'manifest.json'
        if not manifest_path.exists():
            work = Path(tempfile.mkdtemp(prefix='build-', dir=parent))
            register_cache_use([work])
            process = None
            try:
                build_mode = os.environ.get('INSTINCT_CACHE_BUILD_MODE', 'inline')
                if build_mode == 'inline':
                    _run_worker_inline(kind, kwargs, str(work), str(parent.parent))
                elif build_mode == 'spawn':
                    process = multiprocessing.get_context('spawn').Process(
                        target=_worker,
                        args=(kind, kwargs, str(work), str(parent.parent)),
                    )
                    process.start()
                    process.join()
                    if process.exitcode != 0:
                        raise RuntimeError(
                            f'Dataset preprocessing failed (exit={process.exitcode})'
                        )
                else:
                    raise ValueError(
                        f'unknown INSTINCT_CACHE_BUILD_MODE: {build_mode!r}'
                    )
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
                from dataset.cache_budget import release_cache_use
                release_cache_use([work])
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


def load_managed(kind, kwargs):
    """Load/build a packed cache while enforcing the shared dataset-cache quota."""
    if os.environ.get('INSTINCT_MANAGED_DATA_CACHE', '0') != '1' or os.environ.get('INSTINCT_DATA_CACHE_WORKER') == '1':
        return None
    from dataset.cache_budget import (
        cache_files, dataset_cache_budget_root,
        enforce_cache_budget, register_cache_use,
    )
    root = dataset_cache_budget_root()
    enforce_cache_budget(root)
    result = _load_managed(kind, kwargs)
    if result is None:
        return None
    samples, _ = result
    files = cache_files(samples)
    if files:
        register_cache_use(files, root)
    report = enforce_cache_budget(root, protected=files)
    if report['removed'] and os.environ.get('RANK', '0') in ('0', '-1'):
        gib = 1024 ** 3
        print(
            f"[Data cache] evicted {report['removed_bytes'] / gib:.2f} GiB; "
            f"usage={report['after_bytes'] / gib:.2f}/"
            f"{report['budget_bytes'] / gib:.2f} GiB",
            flush=True,
        )
    return result
