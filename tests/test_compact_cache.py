import json
import gc
from types import SimpleNamespace

import numpy as np
import pytest
import datasets  # Windows spawn imports this module without conftest; keep before torch.
import torch

from scripts.data_loader.compact_cache import encode, decode, token_dtype
from scripts.data_loader.lm_dataset import PretrainDataset, _best_fit_pack


class TinyTokenizer:
    """Pickleable tokenizer stub: keeps the Windows spawn integration cheap."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    all_special_ids = [0, 1, 2]

    def get_vocab(self):
        return {"<pad>": 0, "<bos>": 1, "<eos>": 2, "hello": 3, "world": 4}

    def __call__(self, text, **kwargs):
        limit = int(kwargs.get("max_length", 2**31 - 1))
        def encode(value):
            ids = [
                3 if word.lower() == "hello" else 4
                for word in str(value).split()
            ]
            return ids[:limit]

        if isinstance(text, list):
            return SimpleNamespace(input_ids=[encode(value) for value in text])
        return SimpleNamespace(input_ids=encode(text))


@pytest.mark.parametrize('sft', [False, True])
def test_codec_preserves_boundaries_padding_and_supervision(sft):
    ids = [[1, 2, 3], [4, 5], [6] * 7]
    labels = [[-100, 2, 3], [-100, 5], [-100] * 2 + [6] * 5]
    packed = _best_fit_pack(ids, labels if sft else ids, 10, 0)
    compact = encode(packed, packed=True, sft=sft)
    for i in range(len(packed['input_ids'])):
        restored = decode({k: v[i] for k, v in compact.items()})
        np.testing.assert_array_equal(restored['sequence_ids'], packed['sequence_ids'][i])
        np.testing.assert_array_equal(restored['input_ids'], packed['input_ids'][i])
        if sft:
            np.testing.assert_array_equal(restored['labels'], packed['labels'][i])


def test_dtype_checks_added_and_special_tokens():
    tokenizer = SimpleNamespace(get_vocab=lambda: {'a': 1, 'added': 65536}, all_special_ids=[0])
    assert token_dtype(tokenizer) == 'uint32'
    tokenizer.get_vocab = lambda: {'a': 6399}
    assert token_dtype(tokenizer) == 'uint16'


def test_managed_cache_reuses_final_files_and_removes_intermediates(tmp_path, monkeypatch):
    # Never start a real Windows ``spawn`` child from pytest. Inline mode is the
    # production-safe path and still exercises the complete worker, publish,
    # cleanup, mmap and reuse flow.
    from scripts.data_loader import managed_cache
    from scripts.data_loader.cache_budget import release_cache_use

    starts = []
    run_inline = managed_cache._run_worker_inline

    def counted_inline(*args, **kwargs):
        starts.append(1)
        return run_inline(*args, **kwargs)

    monkeypatch.setattr(managed_cache, "_run_worker_inline", counted_inline)
    monkeypatch.setattr(
        managed_cache.multiprocessing, "get_context",
        lambda _method: (_ for _ in ()).throw(
            AssertionError("inline cache test must not request spawn")
        ),
    )
    tokenizer = TinyTokenizer()
    path = tmp_path / 'input.jsonl'
    example = {'text': 'hello world'}
    path.write_text((json.dumps(example) + '\n') * 4, encoding='utf-8')
    cls = PretrainDataset
    kwargs = dict(
        max_length=64, packing=True,
        packing_mode='fixed', packing_num_proc=1,
    )
    monkeypatch.setenv('INSTINCT_MANAGED_DATA_CACHE', '0')
    reference = cls(str(path), tokenizer, **kwargs)
    monkeypatch.setenv('INSTINCT_MANAGED_DATA_CACHE', '1')
    monkeypatch.setenv('INSTINCT_CACHE_BUILD_MODE', 'inline')
    monkeypatch.setenv('HF_DATASETS_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setenv('INSTINCT_DATA_CACHE_MAX_GB', '1')
    monkeypatch.setattr(datasets.config, 'HF_DATASETS_CACHE', tmp_path / 'cache')
    managed = cls(str(path), tokenizer, **kwargs)
    assert managed.bucket_ranges == reference.bucket_ranges
    for i in range(len(reference)):
        for a, b in zip(reference[i], managed[i]):
            assert torch.equal(a, b)
    assert not list((tmp_path / 'cache').rglob('build-*'))
    files = managed.samples.cache_files
    assert all('instinct-packed' in f['filename'] for f in files)
    timestamps = {f['filename']: __import__('os').stat(f['filename']).st_mtime_ns for f in files}
    again = cls(str(path), tokenizer, **kwargs)
    assert again.samples.cache_files == files
    assert timestamps == {f['filename']: __import__('os').stat(f['filename']).st_mtime_ns for f in files}
    assert len(starts) == 1

    leased = [f['filename'] for f in files]
    leased.extend({str(__import__('pathlib').Path(name).parent) for name in leased})
    release_cache_use(leased)
    del again, managed, reference
    gc.collect()
