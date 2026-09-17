import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from dataset.compact_cache import encode, decode, token_dtype
from dataset.lm_dataset import PretrainDataset, SFTDataset, _best_fit_pack


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


@pytest.mark.parametrize('sft', [False, True])
def test_managed_cache_reuses_final_files_and_removes_intermediates(tmp_path, monkeypatch, sft):
    tokenizer = AutoTokenizer.from_pretrained('./model')
    path = tmp_path / 'input.jsonl'
    example = {'conversations': [{'role': 'user', 'content': 'hello'},
                                 {'role': 'assistant', 'content': 'world'}]} if sft else {'text': 'hello world'}
    path.write_text((json.dumps(example) + '\n') * 4, encoding='utf-8')
    cls = SFTDataset if sft else PretrainDataset
    kwargs = dict(max_length=64, packing=True, packing_mode='bucket', packing_num_proc=1)
    monkeypatch.setenv('INSTINCT_MANAGED_DATA_CACHE', '0')
    reference = cls(str(path), tokenizer, **kwargs)
    monkeypatch.setenv('INSTINCT_MANAGED_DATA_CACHE', '1')
    monkeypatch.setenv('HF_DATASETS_CACHE', str(tmp_path / 'cache'))
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
