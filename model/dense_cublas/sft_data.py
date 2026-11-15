"""Read the repository's compact packed SFT Arrow cache without Torch."""
import bisect
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.compute as pc


class PackedSFT:
    def __init__(self, path, tokenizer, max_length, rng=None):
        self.path = Path(path).resolve()
        if self.path.is_dir():
            self.path = self.path / 'manifest.json'
        raw = self.path.read_bytes()
        self.manifest = json.loads(raw)
        if self.manifest['identity']['kind'] != 'sft' or self.manifest['identity']['format'] != 2:
            raise ValueError('expected a compact v2 SFT cache manifest')
        if any(item['max_length'] > max_length for item in self.manifest['bucket_ranges']):
            raise ValueError('SFT cache contains a block longer than max_seq_len')
        self.tokenizer, self.max_length, self.mode = tokenizer, max_length, 'full_sft'
        expected_tokenizer = self.manifest.get('native_tokenizer_namespace')
        if expected_tokenizer and expected_tokenizer != tokenizer.cache_namespace:
            raise ValueError('SFT cache tokenizer differs from the selected tokenizer')
        self.rng = np.random.default_rng(42) if rng is None else rng
        self.files, self.ends = [], []
        digest = hashlib.sha256(raw)
        count = 0
        self.total_tokens = self.total_supervised_tokens = 0
        for item in self.manifest['files']:
            file = (self.path.parent / item['name']).resolve()
            if file.parent != self.path.parent or file.is_symlink():
                raise ValueError('SFT cache shard must be a regular local file')
            stat = file.stat()
            if stat.st_size != item['bytes']:
                raise ValueError('SFT cache shard size changed')
            digest.update(json.dumps([file.name, stat.st_size, stat.st_mtime_ns]).encode())
            with pa.memory_map(str(file), 'r') as mapped:
                table = ipc.open_stream(mapped).read_all()
                required = {'input_ids', 'loss_mask', 'segment_lengths', 'valid_tokens', 'block_length'}
                if not required.issubset(table.column_names):
                    raise ValueError('SFT cache is missing token masks or document boundaries')
                count += table.num_rows
                self.total_tokens += pc.sum(table['valid_tokens']).as_py()
                self.total_supervised_tokens += pc.sum(table['train_tokens']).as_py() if 'train_tokens' in table.column_names else 0
            self.files.append(str(file))
            self.ends.append(count)
        if count != sum(item['blocks'] for item in self.manifest['bucket_ranges']):
            raise ValueError('SFT cache row count differs from manifest')
        self.fingerprint = digest.hexdigest()
        self._tables = {}
        self._maps = {}

    def __len__(self):
        return self.ends[-1]

    def __getstate__(self):
        return self.__dict__ | {'_tables': {}, '_maps': {}}

    def row(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.ends, index)
        if shard not in self._tables:
            self._maps[shard] = pa.memory_map(self.files[shard], 'r')
            self._tables[shard] = ipc.open_stream(self._maps[shard]).read_all()
        offset = index - (self.ends[shard - 1] if shard else 0)
        return self._tables[shard].slice(offset, 1).to_pylist()[0]

    def batch(self, indices):
        ids = np.full((len(indices), self.max_length), self.tokenizer.pad, np.int64)
        labels = np.full_like(ids, -100)
        segments = np.full_like(ids, -1)
        for slot, index in enumerate(indices):
            row = self.row(index)
            valid, width = row['valid_tokens'], row['block_length']
            lengths = np.asarray(row['segment_lengths'], np.int64)
            if (width > self.max_length or len(row['input_ids']) != width
                    or len(row['loss_mask']) != width or np.any(lengths <= 0)
                    or lengths.sum() != valid or not 0 < valid <= width):
                raise ValueError('invalid SFT cache token or document layout')
            ids[slot, :width] = row['input_ids']
            labels[slot, :valid] = np.where(row['loss_mask'][:valid], ids[slot, :valid], -100)
            segments[slot, :valid] = np.repeat(np.arange(len(lengths)), lengths)
            labels[slot, np.r_[0, np.cumsum(lengths)[:-1]]] = -100
        return ids, labels, segments
