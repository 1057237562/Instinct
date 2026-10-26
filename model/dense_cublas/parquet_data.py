"""Bounded native Parquet packing, compact mmap batches and exact resume cursors.

The existing Rust dataset_compiler reads/tokenizes/packs each source row group.
The Python thread only waits for that native process; it never tokenizes records.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from filelock import FileLock, Timeout

ROOT = Path(__file__).resolve().parents[2]
FORMAT_VERSION = 1


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        digest.update(stream.read(65536))
        if stat.st_size > 65536:
            stream.seek(stat.st_size - 65536)
            digest.update(stream.read(65536))
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sample_sha256=digest.hexdigest())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class PackedChunk:
    def __init__(self, path, manifest, pad):
        self.path, self.manifest, self.pad = Path(path), manifest, pad
        self.maps = []
        for bucket in manifest['buckets']:
            if not bucket['blocks']:
                self.maps.append(None)
                continue
            self.maps.append((
                np.memmap(self.path / bucket['tokens_file'], dtype='<u2', mode='r',
                          shape=(bucket['blocks'], bucket['length'])),
                np.memmap(self.path / bucket['offsets_file'], dtype='<u8', mode='r'),
                np.memmap(self.path / bucket['lengths_file'], dtype='<u2', mode='r')))

    def order(self, token_budget, seed, epoch, group):
        rng = np.random.default_rng(np.random.SeedSequence((seed, epoch, group)))
        batches = []
        for index, bucket in enumerate(self.manifest['buckets']):
            batch = max(1, token_budget // bucket['length'])
            indices = rng.permutation(bucket['blocks'])
            batches.extend((index, indices[start:start+batch]) for start in range(0, len(indices), batch))
        return [batches[int(i)] for i in rng.permutation(len(batches))]

    def batch(self, bucket_index, indices):
        tokens, offsets, lengths = self.maps[bucket_index]
        ids = np.asarray(tokens[indices], dtype=np.int64)
        labels = ids.copy()
        labels[ids == self.pad] = -100
        segments = np.full(ids.shape, -1, dtype=np.int64)
        valid = 0
        for row, index in enumerate(indices):
            sizes = lengths[int(offsets[index]):int(offsets[index+1])].astype(np.int64)
            count = int(sizes.sum())
            valid += count
            segments[row, :count] = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
            starts = np.r_[0, np.cumsum(sizes)[:-1]]
            labels[row, starts] = -100
        useful = int(np.count_nonzero(labels[:, 1:] != -100))
        return (ids, labels, segments), valid, useful

    def close(self):
        for entry in self.maps:
            if entry:
                for array in entry: array._mmap.close()
        self.maps.clear()


class ParquetPretrain:
    streaming = True

    def __init__(self, path, tokenizer, cache_dir, *, lengths, token_budget,
                 threads=4, cache_max_gb=5., seed=42, compiler=None):
        self.path, self.tokenizer = Path(path).resolve(), tokenizer
        self.mode, self.rng = 'pretrain', np.random.default_rng(seed)
        self.lengths = sorted(set(int(x) for x in lengths))
        if not self.lengths or min(self.lengths) < 2 or token_budget < max(self.lengths):
            raise ValueError('invalid Parquet bucket lengths/token budget')
        if threads < 1 or cache_max_gb < 0:
            raise ValueError('invalid native preprocessing limits')
        self.max_length = max(self.lengths)
        self.token_budget, self.threads, self.seed = token_budget, threads, seed
        self.root = Path(cache_dir).resolve() / 'parquet-v1'
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget = int(cache_max_gb * 2**30)
        self.compiler = Path(compiler) if compiler else ROOT / 'dataset_compiler/target/release' / (
            'dataset_compiler.exe' if os.name == 'nt' else 'dataset_compiler')
        if not self.compiler.is_file():
            raise FileNotFoundError('Build the existing dataset_compiler with cargo build --release: ' + str(self.compiler))
        source = file_identity(self.path)
        plan_path = self.root / ('plan-' + fingerprint(dict(source=source, max_length=self.max_length)) + '.json')
        with FileLock(str(plan_path) + '.lock'):
            if plan_path.is_file():
                self.plan = json.loads(plan_path.read_text(encoding='utf-8'))
            else:
                result = subprocess.run([str(self.compiler), 'plan-pretrain', '--input', str(self.path),
                    '--max-length', str(self.max_length), '--threads', str(threads)],
                    check=True, capture_output=True, text=True, encoding='utf-8')
                self.plan = json.loads(result.stdout)
                temporary = plan_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(self.plan), encoding='utf-8')
                temporary.replace(plan_path)
        if self.plan['source'] != source:
            raise ValueError('source changed while planning native Parquet input')
        if not self.plan['rows'] or not self.plan['tokens']:
            raise ValueError('empty native Parquet training plan')
        self.identity = dict(format=FORMAT_VERSION, source=source,
            tokenizer=tokenizer.cache_namespace, lengths=self.lengths, token_budget=token_budget,
            seed=seed, packing_batch_size=1000, bos=tokenizer.bos, eos=tokenizer.eos, pad=tokenizer.pad)
        self.fingerprint = fingerprint(self.identity)
        self.executor = None
        self.pending = None
        self.events = []
        self.protected = set()
        self.serial_rng = False

    def __len__(self): return int(self.plan['rows'])

    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='native-parquet')
        return self

    def __exit__(self, *exc):
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        if self.pending is not None and not (exc and exc[0]):
            self.pending.result()

    def cache_path(self, group):
        return self.root / fingerprint(dict(identity=self.identity, group=group))

    def _manifest(self, target, group):
        path = target / 'manifest.json'
        if not path.is_file(): return None
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if (manifest['fingerprint'] != target.name or manifest['source'] != self.plan['source']
                or manifest['row_group'] != group or manifest['format'] != FORMAT_VERSION):
            raise ValueError('native packed cache identity mismatch: ' + str(target))
        for bucket in manifest['buckets']:
            for kind in ('tokens', 'offsets', 'lengths'):
                item = target / bucket[kind + '_file']
                if item.resolve().parent != target.resolve() or item.stat().st_size != bucket[kind + '_bytes']:
                    raise ValueError('incomplete native packed cache: ' + str(item))
        return manifest

    def _reserve(self, additional):
        if not self.budget: return
        def size(path): return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        total = size(self.root)
        candidates = []
        for path in self.root.iterdir():
            # Only our finalized, inactive hexadecimal cache directories are disposable.
            if (path.is_dir() and not path.is_symlink() and path.resolve().parent == self.root
                    and len(path.name) == 64 and all(c in '0123456789abcdef' for c in path.name)
                    and path not in self.protected and (path/'manifest.json').is_file()):
                candidates.append((path.stat().st_mtime_ns, path))
        for _, path in sorted(candidates):
            if total + additional <= self.budget: break
            try:
                with FileLock(str(path)+'.lock', timeout=0):
                    if path.resolve().parent != self.root or path.is_symlink():
                        raise ValueError('cache eviction path escaped its root')
                    count = size(path)
                    shutil.rmtree(path)
                    total -= count
            except Timeout:
                continue
        if total + additional > self.budget:
            raise RuntimeError('native Parquet cache budget cannot cover active chunks and next build')

    def build(self, group):
        target = self.cache_path(group)
        with FileLock(str(target) + '.lock'):
            manifest = self._manifest(target, group)
            if manifest is not None:
                os.utime(target, None)
                return manifest
            # A single native producer per cache root makes the disk reservation
            # valid even when separate trainers share cached input files.
            with FileLock(str(self.root / '.build.lock')):
                return self._build_new(group, target)

    def _build_new(self, group, target):
        meta = self.plan['groups'][group]
        # BFD bins are at least half full apart from the final bin of each
        # 1000-record task. No copied source Parquet is written to disk.
        reserve = int(meta['tokens'])*4 + (int(meta['rows'])//1000+2)*self.max_length*2 + int(meta['rows'])*16
        self._reserve(max(16*2**20, reserve))
        started = time.perf_counter()
        event = dict(group=group, start=started, cache_hit=False)
        self.events.append(event)
        command = [str(self.compiler), 'pack-pretrain', '--input', str(self.path), '--output', str(target),
            '--tokenizer', str(Path(self.tokenizer.path)/'tokenizer.json'), '--row-group', str(group),
            '--lengths', ','.join(map(str, self.lengths)), '--threads', str(self.threads),
            '--seed', str(self.seed), '--bos', str(self.tokenizer.bos), '--eos', str(self.tokenizer.eos),
            '--pad', str(self.tokenizer.pad), '--fingerprint', target.name]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        event.update(seconds=time.perf_counter()-started, returncode=result.returncode)
        if result.returncode:
            raise RuntimeError('native Parquet preprocessing failed: ' + result.stderr[-4000:])
        print(result.stderr.strip(), flush=True)
        manifest = self._manifest(target, group)
        if manifest is None: raise RuntimeError('native worker did not publish a manifest')
        print(f'[Native Parquet] group={group+1}/{len(self.plan["groups"])} ready, '
              f'rows={manifest["rows"]}, useful_targets={manifest["useful_targets"]}, '
              f'build={event["seconds"]:.2f}s', flush=True)
        return manifest

    def batches(self, epoch, cursor=None):
        cursor = cursor or dict(group=0, batch=0)
        first, skip = int(cursor['group']), int(cursor['batch'])
        if not 0 <= first <= len(self.plan['groups']) or skip < 0:
            raise ValueError('invalid native Parquet resume cursor')
        future = None
        for group in range(first, len(self.plan['groups'])):
            target = self.cache_path(group)
            self.protected.add(target)
            manifest = future.result() if future else self.build(group)
            self.pending = None
            lease = FileLock(str(target)+'.lock')
            lease.acquire()
            chunk = PackedChunk(target, manifest, self.tokenizer.pad)
            future = None
            if group+1 < len(self.plan['groups']):
                self.protected.add(self.cache_path(group+1))
                future = self.executor.submit(self.build, group+1)
                self.pending = future
            try:
                order = chunk.order(self.token_budget, self.seed, epoch, group)
                start = skip if group == first else 0
                if start > len(order): raise ValueError('native resume batch outside packed chunk')
                for index in range(start, len(order)):
                    bucket, indices = order[index]
                    batch, valid, useful = chunk.batch(bucket, indices)
                    last = index+1 == len(order)
                    after = dict(group=group+1 if last else group, batch=0 if last else index+1,
                                 valid_tokens=valid, useful_targets=useful)
                    yield after, batch
            finally:
                chunk.close()
                lease.release()
                self.protected.discard(target)
