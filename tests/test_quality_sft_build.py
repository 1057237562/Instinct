import ast
from fractions import Fraction
from scripts.data_builder.build_quality_python_sft import static_check, code_key
from scripts.data_builder.finalize_continuation_sft import arithmetic
from scripts.data_builder.apply_sft_manual_review import structural_key


def test_short_functions_and_broken_outputs():
    assert static_check('def total(xs):\n    return sum(xs)')[0] is None
    assert static_check('def total(xs):\n    return unknown(xs)')[0] == 'unresolved_global'
    assert static_check('def total(xs):\n    pass')[0] == 'placeholder_or_global_state'
    assert static_check('def total(xs):\n    return sum(xs)\nprint(total([1]))')[0] == 'top_level_example_or_execution'
    assert static_check('import numpy\ndef total(xs):\n    return numpy.sum(xs)')[0] == 'non_foundational_dependency'


def test_code_dedup_ignores_docstrings_not_behavior():
    assert code_key('def f(x):\n    "doc"\n    return x') == code_key('def f(x):\n    return x')
    assert code_key('def f(x):\n    return x') != code_key('def f(x):\n    return x+1')


def test_arithmetic_checker_is_restricted():
    assert arithmetic(ast.parse('(16-3-4)*2', mode='eval')) == 18
    assert arithmetic(ast.parse('2+2/2', mode='eval')) == 3
    assert arithmetic(ast.parse('0.1+0.2', mode='eval')) == Fraction(3,10)
    try:
        arithmetic(ast.parse('__import__("os").system("anything")', mode='eval'))
    except ValueError:
        pass
    else:
        raise AssertionError('Calls must not be evaluated')


def test_review_dedup_preserves_semantic_operators():
    assert structural_key('def f(xs):\n    return max(xs)') == structural_key('def largest(values):\n    return max(values)')
    assert structural_key('def f(xs):\n    return max(xs)') != structural_key('def f(xs):\n    return min(xs)')
