import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset, SFTDataset, _best_fit_pack
from trainer.packing_transition import (
    packing_data_config, SequencePackingPlan, validate_packing_resume,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("./model")


def test_best_fit_pack_keeps_examples_whole_and_labels_aligned():
    inputs = [[11] * 6, [22] * 4, [33] * 4]
    labels = [[11] * 6, [-100] * 2 + [22] * 2, [33] * 4]
    packed = _best_fit_pack(inputs, labels, max_length=10, pad_token_id=0)

    assert len(packed["input_ids"]) == 2
    assert all(len(row) == 10 for row in packed["input_ids"])
    assert all(len(row) == 10 for row in packed["labels"])
    assert sum(packed["valid_tokens"]) == 14
    assert sum(packed["train_tokens"]) == 12

    # Each source uses a unique token id.  Contiguous runs prove no example was
    # split across packed blocks.
    flattened_rows = [row[:valid] for row, valid in zip(packed["input_ids"], packed["valid_tokens"])]
    for token, length in ((11, 6), (22, 4), (33, 4)):
        assert any([token] * length == row[i:i + length]
                   for row in flattened_rows for i in range(len(row) - length + 1))


def test_pretrain_packing_preserves_tokens_and_reuses_cache(tmp_path, tokenizer):
    path = tmp_path / "pretrain.jsonl"
    texts = ["alpha beta", "gamma", "delta epsilon zeta", "eta", "theta iota"]
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")

    plain = PretrainDataset(str(path), tokenizer, max_length=32)
    packed = PretrainDataset(
        str(path), tokenizer, max_length=32, packing=True, packing_batch_size=100
    )
    packed_again = PretrainDataset(
        str(path), tokenizer, max_length=32, packing=True, packing_batch_size=100
    )

    plain_tokens = sum(int((plain[i][1] != -100).sum()) for i in range(len(plain)))
    packed_tokens = sum(int((packed[i][1] != -100).sum()) for i in range(len(packed)))
    assert packed_tokens == plain_tokens
    assert len(packed) < len(plain)
    assert packed.samples.cache_files == packed_again.samples.cache_files
    for input_ids, labels in packed:
        assert input_ids.shape == labels.shape == torch.Size([32])
        assert torch.equal(labels[input_ids == tokenizer.pad_token_id], torch.full_like(labels[input_ids == tokenizer.pad_token_id], -100))


def test_sft_packing_preserves_loss_mask_and_fixed_shapes(tmp_path, tokenizer):
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
        packing_batch_size=100, packing_seed=7,
    )
    assert len(packed) < len(rows)
    assert sum(packed.samples["valid_tokens"]) <= len(packed) * 96
    assert sum(packed.samples["train_tokens"]) > 0
    observed_train_tokens = 0
    for input_ids, labels in packed:
        assert input_ids.shape == labels.shape == torch.Size([96])
        observed_train_tokens += int((labels != -100).sum())
        assert torch.all(labels[input_ids == tokenizer.pad_token_id] == -100)
    assert observed_train_tokens == sum(packed.samples["train_tokens"])


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
