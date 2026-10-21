"""Guard quality filters against code-identifier false positives and budget drift."""
import pytest
from scripts.data_builder.clean_ultradata_l3_cpt import quality_reason, quotas_for, TARGET, WEIGHTS


def record(solution):
    return {"task": "A complete programming task with sufficiently detailed constraints.",
            "analysis": "A complete discussion of the algorithm, complexity, and edge cases.",
            "test": "assert solve([1, 2, 3]) == 6\nassert solve([0, 0, 0]) == 0",
            "solution": solution, "full_content": "A sufficiently long serialized task and implementation record."}


def test_java_index_identifier_is_not_a_placeholder():
    assert quality_reason(record("int left = nums[insertionPoint - 1];\nint right = nums[insertionPoint];"), "java") is None


def test_python_todo_identifier_is_valid_code():
    assert quality_reason(record("def solve(values):\n    todo = list(values)\n    return sum(todo)\n"), "py") is None


def test_python_broken_and_unimplemented_solutions_are_rejected():
    assert quality_reason(record("def solve(values):\n    return sum(values\n# enough content\n"), "py") == "python_solution_syntax"
    assert quality_reason(record("def solve(values):\n    raise NotImplementedError('finish this implementation')\n"), "py") == "python_unimplemented_function"


def test_quotas_total_target_and_keep_python_cpp_primary():
    availability = {k: round(TARGET * weight * 1.5) for k, weight in WEIGHTS.items()}
    availability["py"] = 4_000_000_000
    quotas = quotas_for(availability)
    assert sum(quotas.values()) == TARGET
    assert quotas["py"] <= availability["py"]
    assert quotas["py"] + quotas["cpp"] >= TARGET * .65
    assert all(quotas[k] <= availability[k] for k in quotas)


def test_insufficient_corpus_fails_instead_of_repeating_data():
    with pytest.raises(ValueError, match="available tokens"):
        quotas_for({k: 1 for k in WEIGHTS})


def test_identity_review_rejects_substring_false_positives_but_detects_models():
    from scripts.data_builder import review_ultradata_l3_identity  # noqa: F401
    from scripts.data_builder.build_ultradata_code_l3_12b import identity_scan
    assert identity_scan('assert(animal_speak() == "Hello,I am AlexanderTheGreat,AOOO.");')[0] == "clean"
    assert identity_scan("The hash table was built by linear probing and returns a description.")[0] == "clean"
    assert identity_scan("I am MiniCPM, an AI model developed by OpenBMB.")[0] == "claim"


def test_bias_correction_preserves_abstract_class_but_rejects_unfinished_target():
    from scripts.data_builder.rescue_ultradata_l3_bias import intentional_interface
    valid = record("class Base:\n    def area(self):\n        raise NotImplementedError\nclass Square(Base):\n    def area(self):\n        return 4\n")
    assert intentional_interface(valid)
    invalid = record("def solve(values):\n    raise NotImplementedError('finish implementation')\n")
    assert not intentional_interface(invalid)


def test_numeric_tables_are_not_distinctive_benchmark_signatures():
    from scripts.data_builder.rescue_ultradata_l3_bias import low_information
    assert low_information(tuple('0' for _ in range(13)))
    assert low_information(tuple(str(i) for i in range(1, 14)))
    assert low_information(tuple('abcdefghijklm'))
    assert not low_information(tuple('a distinctive problem description requires handling vowels between consonants with carefully chosen constraints'.split()[:13]))
