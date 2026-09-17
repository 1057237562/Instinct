"""Tests for the full-paper arXiv pretraining collector."""

import orjson

from dataset.scripts.collect_arxiv_pretrain import (
    classify_license,
    make_output_record,
    quota_end,
)


def test_license_classifier_accepts_only_corpus_open_licenses():
    assert classify_license("Public Domain") == "Public Domain"
    assert (
        classify_license(
            "Creative Commons - Attribution - "
            "https://creativecommons.org/licenses/by/4.0/"
        )
        == "CC BY"
    )
    assert (
        classify_license(
            "Creative Commons - Attribution Share-Alike - "
            "https://creativecommons.org/licenses/by-sa/4.0/"
        )
        == "CC BY-SA"
    )
    assert classify_license("all rights reserved") is None


def test_output_record_preserves_complete_text_exactly():
    text = "first paragraph\n\nsecond paragraph Ω" * 10_000
    source = {
        "id": "2401.00001",
        "text": text,
        "source": "arxiv-papers",
        "created": "2024-01-01T00:00:00",
        "metadata": {"license": "Public Domain"},
    }

    encoded = orjson.dumps(make_output_record(source))
    assert orjson.loads(encoded)["text"] == text


def test_shard_quotas_cover_exact_target_cumulatively():
    ends = [quota_end(500_000_000, index, 22) for index in range(22)]

    assert ends == sorted(ends)
    assert ends[-1] == 500_000_000
