import numpy as np

from model.dense_cublas.think_stats import ThinkStatsWindow, think_stats

O, C, X = 25, 26, 7  # <think>, </think>, any other token


def row(*tokens):
    return np.asarray([tokens], np.int64)


def test_splits_think_body_marker_and_answer():
    # position 0 is never a target (loss shifts by one)
    ids = row(1, O, X, X, X, C, X, X, 2)
    result = think_stats(ids, ids)
    assert result['supervised'] == 8
    assert (result['think_open'], result['think_close']) == (1, 1)
    assert result['think_body'] == 3
    assert result['answer'] == 8 - 3 - 2
    assert result['think_lengths'] == [3]
    assert result['unclosed'] == 0


def test_unsupervised_tokens_are_not_counted_and_do_not_open_think():
    ids = row(1, O, X, X, C, X)
    labels = ids.copy()
    labels[0, 1:3] = -100  # the opener and first body token are masked
    result = think_stats(ids, labels)
    assert result['think_open'] == 0 and result['think_close'] == 1
    assert result['orphan_close'] == 1
    assert result['think_body'] == 0


def test_unclosed_think_does_not_leak_into_next_document():
    ids = row(1, O, X, X, 1, X, X, X)
    segments = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1]])
    result = think_stats(ids, ids, segments)
    assert result['unclosed'] == 1
    # the second document is plain answer text
    assert result['think_body'] == 2
    assert result['answer'] == result['supervised'] - 2 - 1
    # without segment info the open state would swallow the whole row
    leaked = think_stats(ids, ids)
    assert leaked['think_body'] > result['think_body']


def test_padding_segments_are_ignored():
    ids = row(1, X, X, 9, 9)
    segments = np.asarray([[0, 0, 0, -1, -1]])
    labels = ids.copy()
    labels[0, 3:] = -100
    assert think_stats(ids, labels, segments)['supervised'] == 2


def test_window_summary_and_reset():
    window = ThinkStatsWindow()
    ids = row(1, O, X, X, C, X, 1, O, X, C, X)
    segments = np.asarray([[0] * 6 + [1] * 5])
    window.update(ids, ids, segments)
    summary = window.summary()
    assert summary['think_blocks_open'] == 2 and summary['think_blocks_closed'] == 2
    assert summary['close_rate'] == 1.0
    assert summary['think_len_mean'] == 1.5
    line = window.format()
    assert line.startswith('ThinkStats:') and 'close_rate=1.000' in line
    assert window.summary()['batches'] == 0  # reset after format()
    assert window.format(reset=False).endswith('p90=n/a')


def test_shape_mismatch_is_rejected():
    ids = row(1, 2, 3)
    try:
        think_stats(ids, ids[:, :2])
    except ValueError:
        return
    raise AssertionError('expected ValueError')
