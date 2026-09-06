import json
from itertools import combinations
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset, SFTDataset, _best_fit_pack
from dataset.sequence_bucket import (
    optimal_sequence_buckets,
    packing_preprocess_workers,
)
from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.model_instinct_loop import (
    InstinctConfig as LoopConfig, InstinctForCausalLM as LoopForCausalLM,
)
from model.sequence_packing import (
    block_diagonal_attention_mask, positions_from_sequence_ids,
)
from trainer.packing_transition import (
    packing_data_config, SequencePackingPlan, validate_packing_resume,
)
from trainer.trainer_utils import release_compiled_cuda_memory


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("./model")


def test_packing_preprocess_workers_caps_windows_spawn_processes():
    assert packing_preprocess_workers(
        32, platform_name='nt', cpu_count=64,
    ) == 4
    assert packing_preprocess_workers(
        0, platform_name='nt', cpu_count=64,
    ) == 4
    assert packing_preprocess_workers(
        32, sample_count=2, platform_name='nt', cpu_count=64,
    ) == 2


def test_packing_preprocess_workers_keeps_linux_parallelism():
    assert packing_preprocess_workers(
        32, platform_name='posix', cpu_count=64,
    ) == 32
    assert packing_preprocess_workers(
        0, platform_name='posix', cpu_count=64,
    ) == 8


def test_best_fit_pack_keeps_examples_whole_and_labels_aligned():
    inputs = [[11] * 6, [22] * 4, [33] * 4]
    labels = [[11] * 6, [-100] * 2 + [22] * 2, [33] * 4]
    packed = _best_fit_pack(inputs, labels, max_length=10, pad_token_id=0)

    assert len(packed["input_ids"]) == 2
    assert all(len(row) == 10 for row in packed["input_ids"])
    assert all(len(row) == 10 for row in packed["labels"])
    assert all(len(row) == 10 for row in packed["sequence_ids"])
    assert sum(packed["valid_tokens"]) == 14
    assert sum(packed["train_tokens"]) == 12
    assert packed["block_length"] == [10, 10]

    # Each source uses a unique token id.  Contiguous runs prove no example was
    # split across packed blocks.
    flattened_rows = [row[:valid] for row, valid in zip(packed["input_ids"], packed["valid_tokens"])]
    for token, length in ((11, 6), (22, 4), (33, 4)):
        assert any([token] * length == row[i:i + length]
                   for row in flattened_rows for i in range(len(row) - length + 1))


def test_pretrain_packing_preserves_tokens_reuses_cache_and_logs_buckets(
        tmp_path, tokenizer, capsys):
    path = tmp_path / "pretrain.jsonl"
    texts = ["alpha beta", "gamma", "delta epsilon zeta", "eta", "theta iota"]
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")

    plain = PretrainDataset(str(path), tokenizer, max_length=32)
    packed = PretrainDataset(
        str(path), tokenizer, max_length=32, packing=True, packing_mode="bucket",
        packing_batch_size=100, packing_num_proc=2,
    )
    packed_again = PretrainDataset(
        str(path), tokenizer, max_length=32, packing=True, packing_mode="bucket",
        packing_batch_size=100, packing_num_proc=2,
    )
    fixed = PretrainDataset(
        str(path), tokenizer, max_length=32, packing=True,
        packing_mode="fixed", packing_batch_size=100,
    )
    packing_log = capsys.readouterr().out

    # Compare actual shifted-loss targets. Every standalone BOS is naturally
    # outside the loss; packed non-first BOS labels must be masked explicitly.
    plain_tokens = sum(int((plain[i][1][1:] != -100).sum()) for i in range(len(plain)))
    packed_tokens = sum(int((packed[i][1][1:] != -100).sum()) for i in range(len(packed)))
    assert packed_tokens == plain_tokens
    assert len(packed) <= len(plain)
    assert packed.samples.cache_files == packed_again.samples.cache_files
    assert all(fixed[index][0].numel() == 32 for index in range(len(fixed)))
    assert fixed.bucket_ranges == [{
        "start": 0, "end": len(fixed), "max_length": 32,
        "blocks": len(fixed), "raw_samples": len(texts),
    }]
    bucket_lengths = {bucket["max_length"] for bucket in packed.bucket_ranges}
    ordered_lengths = [bucket["max_length"] for bucket in packed.bucket_ranges]
    assert f"[Packing Plan] pretrain: bucket_count={len(ordered_lengths)}, " \
           f"max_seq_len={ordered_lengths}" in packing_log
    for number, bucket in enumerate(packed.bucket_ranges, start=1):
        assert (
            f"[Packing Bucket] pretrain {number}/{len(packed.bucket_ranges)}: "
            f"max_seq_len={bucket['max_length']}"
        ) in packing_log
    assert 1 <= len(bucket_lengths) <= 2
    for input_ids, labels, sequence_ids in packed:
        assert input_ids.shape == labels.shape
        assert sequence_ids.shape == input_ids.shape
        assert input_ids.numel() in bucket_lengths
        assert torch.equal(labels[input_ids == tokenizer.pad_token_id], torch.full_like(labels[input_ids == tokenizer.pad_token_id], -100))
        boundaries = (sequence_ids[1:] != sequence_ids[:-1]) & (sequence_ids[1:] >= 0)
        assert torch.all(labels[1:][boundaries] == -100)


def test_sft_packing_preserves_loss_mask_and_bucket_shapes(tmp_path, tokenizer):
    path = tmp_path / "sft.jsonl"
    rows = []
    for i in range(8):
        rows.append({
            "conversations": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": f"Answer {i}."},
            ]
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    packed = SFTDataset(
        str(path), tokenizer, max_length=96, packing=True,
        packing_batch_size=100, packing_seed=7, packing_mode="bucket",
        packing_num_proc=2,
    )
    assert len(packed) <= len(rows)
    assert sum(packed.samples["valid_tokens"]) <= sum(packed.samples["block_length"])
    assert sum(packed.samples["train_tokens"]) > 0
    observed_train_tokens = 0
    bucket_lengths = {bucket["max_length"] for bucket in packed.bucket_ranges}
    for input_ids, labels, sequence_ids in packed:
        assert input_ids.shape == labels.shape
        assert sequence_ids.shape == input_ids.shape
        assert input_ids.numel() in bucket_lengths
        observed_train_tokens += int((labels != -100).sum())
        assert torch.all(labels[input_ids == tokenizer.pad_token_id] == -100)
    assert observed_train_tokens == sum(packed.samples["train_tokens"])


def test_sft_bucket_discards_whole_overlength_samples(tmp_path, tokenizer, capsys):
    path = tmp_path / "sft_long.jsonl"
    rows = [
        {"conversations": [
            {"role": "user", "content": "short question"},
            {"role": "assistant", "content": "short answer"},
        ]},
        {"conversations": [
            {"role": "user", "content": "long question"},
            {"role": "assistant", "content": "token " * 500},
        ]},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    packed = SFTDataset(
        str(path), tokenizer, max_length=64, packing=True,
        packing_batch_size=100, packing_seed=7, packing_mode="bucket",
        packing_num_proc=1,
    )

    assert packed.discarded_long_sample_count == 1
    assert all(bucket["max_length"] <= 64 for bucket in packed.bucket_ranges)
    assert "[Length Filter] SFT: kept=1, discarded=1, max_seq_len=64" in capsys.readouterr().out


def _brute_force_time_cost(lengths, bucket_count, alignment=1,
                           token_budget=24576):
    sorted_lengths = sorted(lengths)
    boundaries = [
        index for index in range(1, len(sorted_lengths))
        if (
            (sorted_lengths[index - 1] + alignment - 1) // alignment
            != (sorted_lengths[index] + alignment - 1) // alignment
        )
    ]
    best = float('inf')
    for splits in combinations(boundaries, bucket_count - 1):
        cost = 0.0
        start = 0
        for end in (*splits, len(sorted_lengths)):
            _, one_bucket = optimal_sequence_buckets(
                sorted_lengths[start:end], 1, alignment=alignment,
                token_budget=token_budget,
            )
            cost += one_bucket[0].estimated_cost
            start = end
        best = min(best, cost)
    return best


def test_time_cost_bucket_dp_matches_brute_force_reference():
    lengths = [11, 2, 7, 3, 19, 5, 13, 3]
    sorted_lengths = sorted(lengths)
    bucket_count = 3

    sorted_indices, buckets = optimal_sequence_buckets(lengths, bucket_count)
    assert [lengths[index] for index in sorted_indices] == sorted_lengths
    assert sum(bucket.estimated_cost for bucket in buckets) == pytest.approx(
        _brute_force_time_cost(lengths, bucket_count)
    )
    assert all(bucket.estimated_blocks > 0 for bucket in buckets)
    assert buckets[-1].end == len(lengths)


def test_bucket_lengths_align_for_fp8_scaled_mm():
    lengths = [101, 103, 620, 623, 941, 947, 3201, 3210]
    sorted_indices, buckets = optimal_sequence_buckets(
        lengths, bucket_count=3, alignment=16,
    )
    assert sorted(sorted_indices) == list(range(len(lengths)))
    assert all(bucket.max_length % 16 == 0 for bucket in buckets)
    for bucket in buckets:
        source_lengths = [
            lengths[index] for index in sorted_indices[bucket.start:bucket.end]
        ]
        assert max(source_lengths) <= bucket.max_length


def test_aligned_time_cost_dp_matches_brute_force_reference():
    lengths = sorted([15, 17, 33, 34, 61, 79, 95])
    alignment = 16
    bucket_count = 3

    _, buckets = optimal_sequence_buckets(lengths, bucket_count, alignment=alignment)
    assert sum(bucket.estimated_cost for bucket in buckets) == pytest.approx(
        _brute_force_time_cost(lengths, bucket_count, alignment=alignment)
    )


class _BucketedDataset(Dataset):
    def __init__(self):
        self.packing_mode = "bucket"
        self.lengths = [8] * 5 + [16] * 7
        self.bucket_ranges = [
            {"start": 0, "end": 5, "max_length": 8},
            {"start": 5, "end": 12, "max_length": 16},
        ]

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return torch.zeros(self.lengths[index], dtype=torch.long)


class _FixedPackedDataset(Dataset):
    packing_mode = "fixed"

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def test_fixed_sampler_matches_legacy_order_and_resume_suffix(tmp_path):
    args = SimpleNamespace(
        sequence_packing=1, sequence_packing_mode="fixed", seq_bucket=2,
        packing_batch_size=100, max_seq_len=64, batch_size=3,
        data_path=str(tmp_path / "data.jsonl"),
    )
    dataset = _FixedPackedDataset(14)
    plan = SequencePackingPlan(args, None, lambda *_: dataset)
    generator = torch.Generator().manual_seed(42 + 2)
    legacy_order = torch.randperm(len(dataset), generator=generator).tolist()
    expected = [legacy_order[pos:pos + 3] for pos in range(0, len(dataset), 3)]

    batches = plan.batch_sampler(
        dataset, active_packing=True, epoch=2, batch_size=3, skip_batches=0,
    )
    assert batches == expected
    assert plan.batch_sampler(
        dataset, active_packing=True, epoch=2, batch_size=3, skip_batches=2,
    ) == expected[2:]


def test_packed_batch_sampler_never_mixes_bucket_lengths_and_scales_memory(
        tmp_path, capsys):
    args = SimpleNamespace(
        sequence_packing=1, packing_batch_size=100, seq_bucket=2,
        sequence_packing_mode="bucket", max_seq_len=64, batch_size=3,
        bucket_gpu_memory_gb=16.0,
        data_path=str(tmp_path / "data.jsonl"),
    )
    dataset = _BucketedDataset()
    dataset.lengths = [2048] * 12 + [4096] * 12
    dataset.bucket_ranges = [
        {"start": 0, "end": 12, "max_length": 2048},
        {"start": 12, "end": 24, "max_length": 4096},
    ]
    plan = SequencePackingPlan(args, None, lambda *_: dataset)
    batches = plan.batch_sampler(
        dataset, active_packing=True, epoch=2, batch_size=3, skip_batches=0,
    )
    assert sorted(index for batch in batches for index in batch) == list(range(len(dataset)))
    assert all(len({dataset.lengths[index] for index in batch}) == 1 for batch in batches)
    assert max(len(batch) for batch in batches if dataset.lengths[batch[0]] == 2048) == 12
    assert max(len(batch) for batch in batches if dataset.lengths[batch[0]] == 4096) == 6
    log = capsys.readouterr().out
    assert 'memory_model=B*L' in log
    assert 'gpu_memory=16GB, token_budget=24576' in log
    assert '[Packing Batch] 1/2: max_seq_len=2048, batch_size=12' in log
    assert '[Packing Batch] 2/2: max_seq_len=4096, batch_size=6' in log
    assert plan.batch_sampler(
        dataset, active_packing=True, epoch=2, batch_size=3, skip_batches=2,
    ) == batches[2:]


def test_large_buckets_run_first_and_expose_one_release_boundary(tmp_path, capsys):
    args = SimpleNamespace(
        sequence_packing=1, packing_batch_size=100, seq_bucket=2,
        sequence_packing_mode="bucket", max_seq_len=16384, batch_size=3,
        bucket_gpu_memory_gb=16.0, bucket_max_seq_len=16384,
        bucket_large_threshold=8192,
        data_path=str(tmp_path / "data.jsonl"),
    )
    dataset = _BucketedDataset()
    dataset.lengths = [4096] * 12 + [16368] * 6
    dataset.bucket_ranges = [
        {"start": 0, "end": 12, "max_length": 4096},
        {"start": 12, "end": 18, "max_length": 16368},
    ]
    plan = SequencePackingPlan(args, None, lambda *_: dataset)

    batches = plan.batch_sampler(
        dataset, active_packing=True, epoch=0, batch_size=3,
    )
    batch_lengths = [dataset.lengths[batch[0]] for batch in batches]
    large_batch_count = sum(length > 8192 for length in batch_lengths)

    assert batch_lengths[:large_batch_count] == [16368] * large_batch_count
    assert all(length <= 8192 for length in batch_lengths[large_batch_count:])
    assert sorted(index for batch in batches for index in batch) == list(range(len(dataset)))
    assert plan.has_large_phase(0) is True
    assert plan.should_release_large_cuda_memory(
        epoch=0, step=large_batch_count - 1,
    ) is False
    assert plan.should_release_large_cuda_memory(
        epoch=0, step=large_batch_count,
    ) is True
    assert plan.should_release_large_cuda_memory(
        epoch=0, step=large_batch_count,
    ) is False
    assert (
        '[Packing Large Phase] epoch=1, threshold=8192, '
        f'large_batches={large_batch_count}, regular_batches='
    ) in capsys.readouterr().out


def test_release_compiled_cuda_memory_returns_unused_vram(monkeypatch, capsys):
    import trainer.trainer_utils as trainer_utils

    gib = 1024 ** 3
    allocated = iter([14 * gib, 4 * gib])
    reserved = iter([19 * gib, 5 * gib])
    free = iter([(1 * gib, 24 * gib), (15 * gib, 24 * gib)])
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: calls.append('synchronize'))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda: next(allocated))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda: next(reserved))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: next(free))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('empty_cache'))
    monkeypatch.setattr(torch.compiler, 'reset', lambda: calls.append('compiler_reset'))
    monkeypatch.setattr(trainer_utils.gc, 'collect', lambda: calls.append('gc_collect'))

    release_compiled_cuda_memory('large phase complete')

    assert calls == [
        'synchronize', 'compiler_reset', 'gc_collect', 'empty_cache', 'synchronize',
    ]
    log = capsys.readouterr().out
    assert '[GPU Memory Release] large phase complete' in log
    assert 'reserved=19.00->5.00GB' in log
    assert 'device_free=1.00->15.00GB/24.00GB' in log


def test_bucket_loader_workers_avoid_windows_spawn_commit(tmp_path):
    args = SimpleNamespace(
        sequence_packing=1, sequence_packing_mode='bucket',
        num_workers=8, bucket_loader_workers=0,
        packing_batch_size=100, seq_bucket=2, max_seq_len=64,
        batch_size=3, bucket_gpu_memory_gb=16.0,
        data_path=str(tmp_path / 'data.jsonl'),
    )
    plan = SequencePackingPlan(args, None, lambda *_: _BucketedDataset())
    assert plan.loader_num_workers() == 0

    args.bucket_loader_workers = 2
    plan = SequencePackingPlan(args, None, lambda *_: _BucketedDataset())
    assert plan.loader_num_workers() == 2

    args.sequence_packing_mode = 'fixed'
    plan = SequencePackingPlan(args, None, lambda *_: _BucketedDataset())
    assert plan.loader_num_workers() == 8


def test_bucket_runtime_log_reports_first_use_and_describes_each_batch(
        tmp_path, capsys):
    args = SimpleNamespace(
        sequence_packing=1, sequence_packing_mode='bucket',
        packing_batch_size=100, seq_bucket=2, max_seq_len=64,
        batch_size=3, bucket_gpu_memory_gb=16.0,
        data_path=str(tmp_path / 'data.jsonl'),
    )
    dataset = _BucketedDataset()
    dataset.lengths = [2048] * 12 + [4096] * 12
    dataset.bucket_ranges = [
        {'start': 0, 'end': 12, 'max_length': 2048},
        {'start': 12, 'end': 24, 'max_length': 4096},
    ]
    plan = SequencePackingPlan(args, None, lambda *_: dataset)
    plan.batch_sampler(
        dataset, active_packing=True, epoch=0, batch_size=3,
    )
    capsys.readouterr()

    batch = (torch.zeros((12, 2048), dtype=torch.long),)
    status = plan.observe_batch(batch, epoch=0, step=1)
    assert status == 'bucket: 1/2 (2048x12)'
    assert '[Packing Bucket Active] epoch=1, step=1, bucket=1/2' in capsys.readouterr().out

    plan.observe_batch(batch, epoch=0, step=2)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize(
    ('gpu_memory_gb', 'expected'),
    [
        (16.0, {624: 39, 960: 25, 3216: 7}),
        (8.0, {624: 19, 960: 12, 3216: 3}),
        (24.0, {624: 59, 960: 38, 3216: 11}),
    ],
)
def test_bucket_batch_size_scales_measured_budget_with_vram(
        tmp_path, gpu_memory_gb, expected):
    args = SimpleNamespace(
        sequence_packing=1, packing_batch_size=100, seq_bucket=3,
        sequence_packing_mode="bucket", max_seq_len=64, batch_size=12,
        bucket_gpu_memory_gb=gpu_memory_gb,
        data_path=str(tmp_path / "data.jsonl"),
    )
    dataset = _BucketedDataset()
    dataset.lengths = [624] * 80 + [960] * 80 + [3216] * 80
    dataset.bucket_ranges = [
        {'start': 0, 'end': 80, 'max_length': 624},
        {'start': 80, 'end': 160, 'max_length': 960},
        {'start': 160, 'end': 240, 'max_length': 3216},
    ]
    plan = SequencePackingPlan(args, None, lambda *_: dataset)
    batches = plan.batch_sampler(
        dataset, active_packing=True, epoch=0, batch_size=12,
    )
    largest = {
        length: max(
            len(batch) for batch in batches
            if dataset.lengths[batch[0]] == length
        )
        for length in (624, 960, 3216)
    }
    assert largest == expected


def test_packed_positions_reset_and_attention_is_block_diagonal():
    sequence_ids = torch.tensor([[0, 0, 0, 1, 1, -1, -1]])
    positions = positions_from_sequence_ids(sequence_ids)
    mask = block_diagonal_attention_mask(sequence_ids)

    assert positions.tolist() == [[0, 1, 2, 0, 1, 0, 1]]
    assert mask[0, 3, :].tolist() == [False, False, False, True, True, False, False]
    assert mask[0, 0, :].tolist() == [True, True, True, False, False, False, False]
    assert mask[0, 5, :].tolist() == [False, False, False, False, False, True, True]


@pytest.mark.parametrize("flash_attn", [False, True])
def test_dense_packed_segment_matches_standalone_forward(flash_attn):
    torch.manual_seed(123)
    config = InstinctConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, dropout=0.0, flash_attn=flash_attn,
        tie_word_embeddings=False,
    )
    model = InstinctForCausalLM(config).eval()
    second = torch.tensor([[20, 21, 22]])
    packed = torch.tensor([[5, 6, 7, 20, 21, 22]])
    sequence_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

    with torch.no_grad():
        standalone_logits = model(second).logits
        packed_logits = model(packed, sequence_ids=sequence_ids).logits[:, 3:]

    torch.testing.assert_close(packed_logits, standalone_logits, rtol=1e-5, atol=1e-6)


def test_looped_packed_segment_matches_standalone_forward():
    torch.manual_seed(123)
    config = LoopConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2,
        prelude_layers=1, loop_iters=2, coda_layers=1,
        max_position_embeddings=32, dropout=0.0, flash_attn=False,
        tie_word_embeddings=False,
    )
    model = LoopForCausalLM(config).eval()
    second = torch.tensor([[20, 21, 22]])
    packed = torch.tensor([[5, 6, 7, 20, 21, 22]])
    sequence_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

    with torch.no_grad():
        standalone_logits = model(second).logits
        packed_logits = model(packed, sequence_ids=sequence_ids).logits[:, 3:]

    torch.testing.assert_close(packed_logits, standalone_logits, rtol=1e-5, atol=1e-6)


def test_checkpoint_mode1_packed_mask_uses_sdpa_and_ffn_checkpoint(monkeypatch):
    import model.model_instinct as dense_module

    calls = {"flash": 0, "ffn_checkpoint": 0}
    original_flash = dense_module.flash_attention
    original_checkpoint_ffn = dense_module.checkpoint_ffn

    def fail_recompute(*args, **kwargs):
        raise AssertionError("packed mask fell back to eager attention recomputation")

    def spy_flash(*args, **kwargs):
        calls["flash"] += 1
        return original_flash(*args, **kwargs)

    def spy_checkpoint_ffn(*args, **kwargs):
        calls["ffn_checkpoint"] += 1
        return original_checkpoint_ffn(*args, **kwargs)

    monkeypatch.setattr(dense_module, "recompute_attention", fail_recompute)
    monkeypatch.setattr(dense_module, "flash_attention", spy_flash)
    monkeypatch.setattr(dense_module, "checkpoint_ffn", spy_checkpoint_ffn)
    config = InstinctConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=16, dropout=0.0, flash_attn=True,
        use_grad_checkpoint=1, tie_word_embeddings=False,
    )
    model = InstinctForCausalLM(config).train()
    input_ids = torch.tensor([[5, 6, 7, 20, 21, 22]])
    labels = input_ids.clone()
    labels[:, 3] = -100
    sequence_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

    output = model(input_ids, labels=labels, sequence_ids=sequence_ids)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert calls == {"flash": 1, "ffn_checkpoint": 1}


class _SizedDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


def test_resume_aligns_then_packs_inside_same_epoch(tmp_path):
    args = SimpleNamespace(
        sequence_packing=1, packing_batch_size=10,
        max_seq_len=1024, batch_size=28,
        data_path=str(tmp_path / "data.jsonl"),
    )
    saved = packing_data_config(args)
    assert validate_packing_resume(args, {"data_config": saved}) is False

    args.batch_size = 4
    legacy = {"model": {}, "epoch": 0, "step": 1, "optimizer": {"sentinel": 1}}
    assert validate_packing_resume(args, legacy) is True
    packed_source_indices = []

    def factory(packing, sample_indices=None):
        if not packing:
            assert sample_indices is None
            return _SizedDataset(25)
        packed_source_indices.extend(sample_indices)
        # The fake packer turns every two source rows into one block.
        return _SizedDataset((len(sample_indices) + 1) // 2)

    plan = SequencePackingPlan(args, legacy, factory)
    dataset, batches, active = plan.epoch_data(epoch=0, resume_step=1, batch_size=4)
    permutation, _, _ = plan._global_epoch_indices(25, 0)
    assert active is False
    assert len(dataset.datasets) == 2
    # Four rows were already trained. Rows 4..9 remain raw to reach the next
    # 10-row packing boundary, and only the untouched suffix is packed.
    raw_rows = [index for batch in batches[:2] for index in batch]
    assert raw_rows == permutation[4:10]
    assert packed_source_indices == permutation[10:]
    assert legacy["step"] == 1
    assert legacy["optimizer"] == {"sentinel": 1}

    transition_config = packing_data_config(args)
    plan.update_checkpoint_config(transition_config, epoch=0, active=False)
    assert transition_config["packing_alignment_pending"] is True
    assert transition_config["packing_transition_origin_step"] == 1

    # Pausing after both raw alignment batches reconstructs the same hybrid
    # epoch and resumes directly from its packed section.
    resumed = {"epoch": 0, "step": 3, "data_config": transition_config}
    resumed_plan = SequencePackingPlan(args, resumed, factory)
    _, resumed_batches, _ = resumed_plan.epoch_data(0, 3, 4)
    assert resumed_batches == batches[2:]

    packed_saved = packing_data_config(args)
    changed = SimpleNamespace(**vars(args))
    changed.max_seq_len = 768
    with pytest.raises(ValueError, match="changed max_seq_len"):
        validate_packing_resume(changed, {"data_config": packed_saved})
    changed.max_seq_len = args.max_seq_len
    changed.sequence_packing_mode = "bucket"
    with pytest.raises(ValueError, match="changed sequence_packing_mode"):
        validate_packing_resume(changed, {"data_config": packed_saved})
    bucket_args = SimpleNamespace(**vars(args))
    bucket_args.sequence_packing_mode = "bucket"
    packed_saved = packing_data_config(bucket_args)
    changed = SimpleNamespace(**vars(bucket_args))
    changed.seq_bucket = 3
    with pytest.raises(ValueError, match="changed seq_bucket"):
        validate_packing_resume(changed, {"data_config": packed_saved})


def test_raw_checkpoint_can_enable_experimental_bucket_mode(tmp_path):
    saved_args = SimpleNamespace(
        sequence_packing=0, sequence_packing_mode="fixed", seq_bucket=2,
        packing_batch_size=1000, max_seq_len=512, batch_size=8,
        data_path=str(tmp_path / "data.jsonl"),
    )
    requested = SimpleNamespace(**vars(saved_args))
    requested.sequence_packing = 1
    requested.sequence_packing_mode = "bucket"
    requested.seq_bucket = 4
    requested.packing_batch_size = 2000
    assert validate_packing_resume(
        requested, {"data_config": packing_data_config(saved_args)}
    ) is True
