import sys
from pathlib import Path

import pytest

from scripts.data_builder.coder_identity_review import review_reasons, reviewed_decision
from scripts.data_builder.coder_pretrain_common import ngrams, has_benchmark_overlap
from scripts.data_builder.prepare_coder_segment2 import benchmark_overlap


def test_persona_candidates_require_explicit_review():
    text = 'Human: Who are you? AI: I am a large language model trained by OpenAI.'
    assert review_reasons(text, 'code')
    with pytest.raises(ValueError, match='Unreviewed'):
        reviewed_decision('id', text, 'code', {})
    assert reviewed_decision('id', text, 'code', {'id': {'decision': 'drop', 'reason': 'Reviewed embedded identity answer'}})[0] == 'drop'


def test_factual_code_mentions_and_todos_are_preserved():
    text = 'import com.alibaba.fastjson.JSON; // Created by Alice. TODO: parse the response.'
    assert review_reasons(text, 'code') == []
    assert reviewed_decision('id', text, 'code', {})[0] == 'keep'


def test_human_zodiac_false_positive_can_be_kept_by_context():
    text = "I'm Jessica and I'm a Gemini. This describes my zodiac sign."
    assert review_reasons(text, 'natural_en')
    assert reviewed_decision('id', text, 'natural_en', {'id': {'decision': 'keep', 'reason': 'Human zodiac context'}})[0] == 'keep'


def test_aho_screen_matches_thirteen_word_tuple_screen():
    import ahocorasick
    benchmark = 'write a function that returns the sum of all positive integers in the supplied list'
    targets = ngrams(benchmark)
    matcher = ahocorasick.Automaton()
    for i, target in enumerate(targets):matcher.add_word('\x1f'+'\x1f'.join(target)+'\x1f',i)
    matcher.make_automaton()
    for text in [benchmark, 'prefix '+benchmark+' suffix', benchmark.upper().replace(' ', ', '), 'different task about multiplication', 'rewrite'+benchmark[5:]]:
        assert benchmark_overlap(text, matcher) == has_benchmark_overlap(text, targets)
