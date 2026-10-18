import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import eval_humaneval_oj as oj
from scripts.data_builder import build_humaneval_oj as builder
from scripts.data_builder import humaneval_oj_protocol as protocol


@pytest.mark.parametrize('values', [
    ['', ' leading and trailing ', 'a\nb\n'],
    [None, 'answer', 'None'],
    [[], [1, 2.5], [-3]],
    [[], ['a', '', 'two words']],
    [('', True), ('abc', False)],
    [{}, {'a': 1, 'b': 2}],
    [[4, {}, [], 23.2, 9, 'adasd']],
    [1, 2.5, '2,3', None],
    [[[1, 2], [], [3]]],
])
def test_plain_text_roundtrip(values):
    schema = protocol.infer_schema(values)
    for value in values:
        assert protocol.parse_output(protocol.encode(value, schema, output=True), schema) == value
        text = protocol.encode(value, schema)
        assert protocol.decode(iter(text.splitlines()), schema) == value


def make_problem(test, prompt='def solve(x):\n    """Return x."""\n', solution='    return x\n'):
    task = {'task_id': 'HumanEval/fixture', 'entry_point': 'solve', 'prompt': prompt,
            'canonical_solution': solution, 'test': test}
    records = builder.run_job({'mode': 'trace', 'task': task}, 10)['records']
    args = [protocol.parse_output(r['args_text'], {'kind': 'variant'}) for r in records]
    values = [protocol.parse_output(r['result_text'], {'kind': 'variant'}) for r in records]
    plan = {'params': [{'name': 'x', 'schema': protocol.infer_schema([a[0] for a in args])}],
            'output': protocol.infer_schema(values)}
    return {'protocol_version': 2, 'task_id': task['task_id'], 'entry_point': 'solve',
            'statement': 'Complete program.', 'prompt': 'Complete program.',
            'official_task': task, 'io_plan': plan, 'num_test_cases': len(records),
            'test_cases': [{'input': protocol.render_input(a, plan),
                            'output': protocol.encode(v, plan['output'], output=True)} for a, v in zip(args, values)]}


def test_judge_preserves_loops_tolerance_and_multiple_solutions():
    problem = make_problem('def check(candidate):\n    for x in [1.0, 2.0, 3.0]:\n        assert abs(abs(candidate(x)) - x) < 1e-4\n')
    # Different sign and float formatting are both valid under the original test.
    assert oj.check_program(problem, 'print(-float(input()) + 1e-6)', 2) == 'passed'
    assert oj.check_program(problem, 'print(1)', 2) == 'failed'
    assert oj.check_program(problem, 'while True: pass', .3) == 'timed out'
    diagnostic = oj.check_program(problem, 'raise ValueError("bad")', 2, details=True)
    assert diagnostic['error_type'] == 'RuntimeError'


def test_dictionary_order_and_tuple_output():
    problem = make_problem('def check(candidate):\n    assert candidate("a") == {"a": 1, "b": 2}\n',
                           solution='    return {"a": 1, "b": 2}\n')
    assert oj.check_program(problem, 'print("2\\nb\\n2\\na\\n1")', 2) == 'passed'
    problem = make_problem('def check(candidate):\n    assert candidate("a") == ("", True)\n',
                           solution='    return ("", True)\n')
    assert oj.check_program(problem, 'print("\\nTrue")', 2) == 'passed'


def test_reference_program_handles_helper_before_entry_point():
    prompt = 'def helper(x):\n    return x * 2\n\ndef solve(x):\n    """Double x."""\n'
    problem = make_problem('def check(candidate):\n    assert candidate(3) == 6\n', prompt, '    return helper(x)\n')
    fn, description = builder.signature_info(problem['official_task'])
    assert fn.name == 'solve'
    assert 'helper' in description
    program = builder.reference_program(problem['official_task'], problem['io_plan'])
    assert oj.check_program(problem, program, 2) == 'passed'


def test_reference_program_codec_names_do_not_shadow_task():
    problem = make_problem('def check(candidate):\n    assert candidate("a") == "A"\n',
                           prompt='def encode(x):\n    return x.upper()\n\ndef solve(x):\n    """Uppercase x."""\n',
                           solution='    return encode(x)\n')
    program = builder.reference_program(problem['official_task'], problem['io_plan'])
    assert oj.check_program(problem, program, 2) == 'passed'


def test_default_missing_bundle_auto_builds_once(tmp_path, monkeypatch):
    path = tmp_path / 'humaneval_oj_problems.jsonl'
    monkeypatch.setattr(oj, 'DEFAULT_PROBLEM_FILE', str(path))
    calls = []

    def build(output_dir):
        calls.append(output_dir)
        rows = [{'protocol_version': 2, 'task_id': f'HumanEval/{i}', 'entry_point': 'f',
                 'statement': 'OJ', 'test_cases': [{'input': '1\n', 'output': '1\n'}]} for i in range(164)]
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')

    monkeypatch.setattr(builder, 'build_dataset', build)
    assert len(oj.load_problems(limit=3)) == 3
    assert len(oj.load_problems()) == 164
    assert calls == [path.parent]
    with pytest.raises(FileNotFoundError, match='数据文件不存在'):
        oj.load_problems(tmp_path / 'custom.jsonl')


def test_thinking_never_becomes_program():
    response = '<think>```python\nwrong = 1\n```</think>\n```python\nprint(2)\n```'
    assert oj.extract_program(response) == 'print(2)'
    assert oj.extract_program(response.removeprefix('<think>')) == 'print(2)'
    assert oj.extract_program('<think>```python\nprint(1)\n```') == ''


def test_oj_report_is_separate_from_function_completion(tmp_path):
    problem = make_problem('def check(candidate):\n    assert candidate(1) == 1\n')
    args = SimpleNamespace(output=str(tmp_path / 'samples.jsonl'), k=[1], problem_file=None, limit=0, timeout=2)
    rows = [{'task_id': problem['task_id'], 'completion': 'print(1)', 'passed': True}]
    Path(args.output + '_results.jsonl').write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
    report = oj.summarize_results(args, {problem['task_id']: problem}, rows)
    assert report['benchmark'] == 'HumanEval-OJ'
    assert report['metrics'] == {'pass@1': 1}
    assert json.loads(Path(args.output + '_metrics.json').read_text()) == report


def test_oj_generation_scoring_pipeline(tmp_path, monkeypatch):
    problem = make_problem('def check(candidate):\n    assert candidate(1) == 1\n')
    args = SimpleNamespace(output=str(tmp_path / 'samples.jsonl'), k=[1], problem_file=None,
                           limit=0, timeout=2, workers=1, num_samples=1, resume=False)

    def generate(args, problems, on_sample):
        row = {'task_id': problem['task_id'], 'completion': 'print(1)'}
        Path(args.output).write_text(json.dumps(row) + '\n', encoding='utf-8')
        on_sample(row)

    monkeypatch.setattr(oj, 'generate', generate)
    report = oj.generate_and_evaluate(args, {problem['task_id']: problem})
    assert report['metrics'] == {'pass@1': 1}
    assert json.loads(Path(args.output + '_oj_stats.json').read_text())['passed'] == 1
    assert json.loads(Path(args.output + '_progress.json').read_text())['completed'] == 1
