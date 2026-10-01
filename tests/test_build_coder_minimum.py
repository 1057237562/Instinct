import orjson

from scripts.data_builder.coder_pretrain_common import (
    classify_base,
    classify_continued,
    has_benchmark_overlap,
    ngrams,
)
from scripts.data_builder.collect_coder_pretrain_12b import (
    TARGETS as SUPPLEMENT_TARGETS,
    clean_text,
    stable_order,
)
from scripts.data_builder.build_coder_pretrain_12b import (
    TARGETS as FINAL_TARGETS,
    audit_contiguous_mixing,
    classify_collected,
    reshuffle_existing,
)


def test_sources_are_classified_into_disjoint_bands():
    assert classify_base({"source": "open_repository_code_and_docs"}) == "code"
    assert classify_base({"source": "math_reasoning"}) == "math"
    assert classify_base({"source": "general_bilingual"}) == "natural"
    assert classify_continued({"source": "python_edu_novel"}) == "code"
    assert classify_continued({"source": "verified_math_cot_novel"}) == "math"
    assert classify_continued({"source": "cosmopedia_novel"}) == "natural"
    assert classify_continued({
        "source": "verified_math_code", "metadata": {"domain": "code"}
    }) == "code"


def test_thirteen_word_benchmark_overlap_screen():
    benchmark = "write a function that returns the sum of all positive integers in the supplied list"
    fingerprints = ngrams(benchmark)
    assert has_benchmark_overlap("prefix " + benchmark + " suffix", fingerprints)
    assert not has_benchmark_overlap("implement a binary search tree", fingerprints)


def test_12b_collector_has_enough_novel_headroom():
    assert sum(SUPPLEMENT_TARGETS.values()) == 10_400_000_000
    assert SUPPLEMENT_TARGETS["code"] > SUPPLEMENT_TARGETS["natural_en"]
    assert SUPPLEMENT_TARGETS["natural_zh"] == SUPPLEMENT_TARGETS["math"]


def test_collector_shard_order_is_deterministic_and_complete():
    files = [str(index) for index in range(64)]
    ordered = stable_order(files)
    assert ordered == stable_order(files)
    assert len(ordered) == len(files)
    assert set(ordered) == set(files)
    assert ordered != files


def test_collector_rejects_spam_and_control_characters():
    assert clean_text("def useful_function():\n    return 42\n" * 10)
    assert not clean_text("online casino " * 30)
    assert not clean_text(("useful educational text " * 20) + "\x00")


def test_final_12b_mix_is_exactly_deepseek_coder_v2_style():
    total = sum(FINAL_TARGETS.values())
    assert total == 12_000_000_000
    assert FINAL_TARGETS["code"] / total == 0.60
    assert FINAL_TARGETS["math"] / total == 0.10
    assert FINAL_TARGETS["natural"] / total == 0.30


def test_collected_sources_map_to_final_categories():
    assert classify_collected({"source": "common-pile/stackv2_edu_filtered:code"}) == "code"
    assert classify_collected({"source": "HuggingFaceTB/finemath:math"}) == "math"
    assert classify_collected({"source": "HuggingFaceTB/smollm-corpus:natural_en"}) == "natural"
    assert classify_collected({"source": "opencsg/chinese-fineweb-edu:natural_zh"}) == "natural"


def test_reshuffle_makes_contiguous_cache_windows_mixed(tmp_path):
    path = tmp_path / "pretrain_coder_12b.jsonl"
    rows = []
    for category, count in (("code", 600), ("math", 100), ("natural", 300)):
        for index in range(count):
            rows.append({
                "text": f"{category}-{index}",
                "source": f"source-{category}-{index % 6}",
                "token_count": 100,
                "mix_category": category,
            })
    path.write_bytes(b"".join(
        orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE) for row in rows
    ))

    result = reshuffle_existing(
        path, seed=17, chunk_rows=31, chunk_bytes=4096,
        audit_windows=5, audit_window_rows=100,
    )

    assert result["rows"] == 1000
    assert result["mixing_audit"]["passed"]
    assert audit_contiguous_mixing(path, windows=5, window_rows=100)["passed"]
