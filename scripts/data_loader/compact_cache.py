"""Lossless Arrow storage codecs; model-facing tensors keep their old layout."""
from datasets import Features, Sequence, Value
import numpy as np

CACHE_FORMAT_VERSION = 2


def token_dtype(tokenizer):
    ids = list(tokenizer.get_vocab().values()) + list(tokenizer.all_special_ids)
    if not ids or min(ids) < 0 or max(ids) > 4294967295:
        raise ValueError('Tokenizer IDs must fit unsigned 32-bit storage')
    return 'uint16' if max(ids) <= 65535 else 'uint32'


def features(tokenizer, *, packed, sft):
    fields = {'input_ids': Sequence(Value(token_dtype(tokenizer)))}
    if sft:
        fields['loss_mask'] = Sequence(Value('bool'))
    if packed:
        fields.update(segment_lengths=Sequence(Value('int32')),
                      valid_tokens=Value('int32'), train_tokens=Value('int32'),
                      block_length=Value('int32'))
    else:
        fields['length'] = Value('int32')
    return Features(fields)


def encode(batch, *, packed, sft):
    result = dict(batch)
    labels = result.pop('labels', None)
    if sft:
        result['loss_mask'] = [[v != -100 for v in row] for row in labels]
    if packed:
        segments = []
        for row, valid in zip(result.pop('sequence_ids'), result['valid_tokens']):
            values = np.asarray(row[:valid])
            boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
            segments.append(np.diff(np.concatenate(([0], boundaries, [valid]))).tolist())
        result['segment_lengths'] = segments
    return result


def restore_labels(batch):
    if 'labels' in batch:
        return batch['labels']
    if 'loss_mask' not in batch:
        return batch['input_ids']
    return [[token if supervise else -100 for token, supervise in zip(ids, mask)]
            for ids, mask in zip(batch['input_ids'], batch['loss_mask'])]


def decode(sample):
    """Accept both legacy dense metadata and compact v2 rows."""
    if 'segment_lengths' not in sample:
        return sample
    result = dict(sample)
    lengths = np.asarray(sample['segment_lengths'], dtype=np.int64)
    valid = int(sample['valid_tokens'])
    width = int(sample['block_length'])
    if np.any(lengths <= 0) or int(lengths.sum()) != valid or valid > width:
        raise ValueError('Invalid compact sequence boundaries')
    result['sequence_ids'] = np.concatenate((np.repeat(np.arange(len(lengths)), lengths),
                                            np.full(width - valid, -1))).astype(np.int64)
    if 'loss_mask' in sample:
        result['labels'] = np.where(sample['loss_mask'], sample['input_ids'], -100)
    return result
