"""Host-side think/answer token accounting for packed SFT batches.

The SFT loss is a plain token mean over assistant tokens, and the training log
only reports that one scalar. This module splits the *supervised* tokens of a
batch into think-body / think markers / visible answer using nothing but the
numpy ``ids``, ``labels`` and (optionally) packed ``segments`` arrays that the
trainer already holds on the host. It never touches the device, the loss or the
optimizer, so enabling it cannot change training.

It does NOT split the cross-entropy itself: the fused ``linear_ce`` kernel only
returns a scalar mean, so ``CE_think`` / ``CE_answer`` would need a kernel or an
offline checkpoint pass and are intentionally out of scope here.

Enable in ``trainer/train_cublas.py`` with ``INSTINCT_THINK_STATS=1``.
"""
from __future__ import annotations

import numpy as np

THINK_OPEN, THINK_CLOSE = 25, 26

# Same think-length buckets as docs/reports/2026-11-20-sft-loss-performative-thinking.md
BUCKET_EDGES = (0, 5, 65, 257, 769, 1537, 2561, 4097)
BUCKET_LABELS = ('0-4', '5-64', '65-256', '257-768', '769-1536', '1537-2560', '2561-4096', '>4096')


def _bucket(length: int) -> int:
    for index in range(len(BUCKET_EDGES) - 1, -1, -1):
        if length >= BUCKET_EDGES[index]:
            return index
    return 0


def _segment_bounds(segments_row, width):
    if segments_row is None:
        return [(0, width)]
    segments_row = np.asarray(segments_row)
    cuts = np.flatnonzero(np.diff(segments_row)) + 1
    edges = np.concatenate(([0], cuts, [width]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if segments_row[a] >= 0]


def think_stats(ids, labels, segments=None, open_id=THINK_OPEN, close_id=THINK_CLOSE):
    """Split the supervised targets of one batch.

    Mirrors the loss, which predicts ``labels[:, 1:]``. Think state resets at
    every packed-document boundary, so an unclosed ``<think>`` in one document
    cannot leak into the next. Returns plain Python numbers and a list with the
    length (in supervised tokens) of every *closed* think body.
    """
    ids = np.asarray(ids)
    labels = np.asarray(labels)
    if ids.shape != labels.shape or ids.ndim != 2:
        raise ValueError('ids and labels must be 2-D arrays of the same shape')
    seg = None if segments is None else np.asarray(segments)
    if seg is not None and seg.shape != ids.shape:
        raise ValueError('segments must match ids')

    out = dict(supervised=0, think_body=0, think_open=0, think_close=0,
               unclosed=0, orphan_close=0, answer=0, think_lengths=[])
    for row in range(ids.shape[0]):
        lab = labels[row, 1:]
        for a, b in _segment_bounds(None if seg is None else seg[row, 1:], lab.shape[0]):
            piece = lab[a:b]
            supervised = piece != -100
            n_sup = int(supervised.sum())
            if not n_sup:
                continue
            is_open = piece == open_id
            is_close = piece == close_id
            depth = np.cumsum(is_open.astype(np.int64)) - np.cumsum(is_close.astype(np.int64))
            inside = (depth > 0) & supervised & ~is_open & ~is_close
            body = int(inside.sum())
            opens, closes = int(is_open.sum()), int(is_close.sum())
            out['supervised'] += n_sup
            out['think_body'] += body
            out['think_open'] += opens
            out['think_close'] += closes
            out['answer'] += n_sup - body - opens - closes
            if depth.size and depth[-1] > 0:
                out['unclosed'] += int(depth[-1])
            # A close with no matching open (depth would go negative).
            out['orphan_close'] += max(0, closes - opens)
            # Closed think bodies, paired in order inside the segment. Count only
            # supervised tokens, matching what the loss actually sees.
            open_at = np.flatnonzero(is_open)
            close_at = np.flatnonzero(is_close)
            for start, end in zip(open_at, close_at):
                if end > start:
                    out['think_lengths'].append(int(supervised[start + 1:end].sum()))
    return out


class ThinkStatsWindow:
    """Accumulate :func:`think_stats` between log lines; reset after ``format``."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = dict(supervised=0, think_body=0, think_open=0, think_close=0,
                          unclosed=0, orphan_close=0, answer=0)
        self.lengths = []
        self.batches = 0

    def update(self, ids, labels, segments=None):
        result = think_stats(ids, labels, segments)
        for key in self.total:
            self.total[key] += result[key]
        self.lengths.extend(result['think_lengths'])
        self.batches += 1
        return result

    def summary(self):
        t = self.total
        sup = max(t['supervised'], 1)
        lengths = np.asarray(self.lengths, np.int64)
        buckets = np.bincount([_bucket(int(x)) for x in lengths], minlength=len(BUCKET_LABELS)) \
            if lengths.size else np.zeros(len(BUCKET_LABELS), np.int64)
        return {
            'batches': self.batches,
            'supervised_tokens': t['supervised'],
            'think_body_share': t['think_body'] / sup,
            'answer_share': t['answer'] / sup,
            'think_blocks_open': t['think_open'],
            'think_blocks_closed': t['think_close'],
            'close_rate': (t['think_close'] / t['think_open']) if t['think_open'] else None,
            'unclosed_at_doc_end': t['unclosed'],
            'think_len_mean': float(lengths.mean()) if lengths.size else None,
            'think_len_p50': float(np.percentile(lengths, 50)) if lengths.size else None,
            'think_len_p90': float(np.percentile(lengths, 90)) if lengths.size else None,
            'think_len_buckets': dict(zip(BUCKET_LABELS, map(int, buckets))),
        }

    def format(self, reset=True):
        s = self.summary()
        fmt = lambda v, spec: 'n/a' if v is None else format(v, spec)
        line = (f"ThinkStats: batches={s['batches']} sup_tokens={s['supervised_tokens']} "
                f"think_share={s['think_body_share']:.3f} answer_share={s['answer_share']:.3f} "
                f"think_blocks={s['think_blocks_open']} closed={s['think_blocks_closed']} "
                f"close_rate={fmt(s['close_rate'], '.3f')} unclosed={s['unclosed_at_doc_end']} "
                f"think_len mean={fmt(s['think_len_mean'], '.0f')} p50={fmt(s['think_len_p50'], '.0f')} "
                f"p90={fmt(s['think_len_p90'], '.0f')}")
        if reset:
            self.reset()
        return line
