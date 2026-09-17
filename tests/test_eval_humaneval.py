import json
from types import SimpleNamespace

import pytest

import eval_humaneval as he


PROBLEM = {'task_id': 'HumanEval/0', 'prompt': 'def add(a, b):\n    """Add."""\n',
           'entry_point': 'add', 'test': 'def check(candidate):\n    assert candidate(2, 3) == 5'}


@pytest.mark.parametrize('completion,expected', [
    ('    return a + b\n', 'passed'),
    ('    return a - b\n', 'failed'),
    ('    while True: pass\n', 'timed out'),
    ('    raise SystemExit(0)\n', 'failed'),
    ('    return "中文"\n', 'failed'),
])
def test_execution(completion, expected):
    assert he.check_sample(PROBLEM, completion, 1) == expected


def test_chat_solution_preserves_imports_and_helpers():
    response = ('<think>thinking</think>\n```python\nimport operator\n'
                'def helper(a, b):\n    return operator.add(a, b)\n'
                'def add(a, b):\n    return helper(a, b)\n```')
    completion = he.extract_completion(response, PROBLEM['prompt'], 'chat')
    assert he.check_sample(PROBLEM, completion, 2) == 'passed'


def test_base_stops_before_next_function():
    assert he.extract_completion('    return a + b\n\ndef next():\n    pass', '', 'base') == '    return a + b\n'


def test_pass_at_k():
    assert he.estimate_pass_at_k(10, 2, 2) == pytest.approx(17 / 45)
    assert he.estimate_pass_at_k(10, 0, 10) == 0
    assert he.estimate_pass_at_k(10, 10, 1) == 1
    with pytest.raises(ValueError):
        he.estimate_pass_at_k(1, 1, 2)


def test_evaluate_and_report(tmp_path):
    output = tmp_path / 'samples.jsonl'
    rows = [{'task_id': 'HumanEval/0', 'completion': c}
            for c in ['    return a+b\n', '    return a-b\n']]
    output.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    args = SimpleNamespace(output=str(output), k=[1, 2, 10], workers=2,
                           timeout=2, problem_file=None, limit=1)
    report = he.evaluate(args, {PROBLEM['task_id']: PROBLEM})
    assert report['metrics'] == {'pass@1': .5, 'pass@2': 1.0}
    assert report['skipped_k'] == [10]
    assert len(he.read_jsonl(str(output) + '_results.jsonl')) == 2
    assert json.loads((tmp_path / 'samples.jsonl_metrics.json').read_text()) == report


def test_invalid_resume_and_missing_tasks(tmp_path):
    with pytest.raises(ValueError):
        he.validate_samples([{'task_id': 'wrong', 'completion': ''}], {'HumanEval/0': PROBLEM})
    with pytest.raises(ValueError):
        he.validate_samples([{'task_id': 'HumanEval/0', 'completion': ''}] * 2,
                            {'HumanEval/0': PROBLEM}, maximum=1)
    output = tmp_path / 'empty.jsonl'
    output.write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='Every selected'):
        he.evaluate(SimpleNamespace(output=str(output)), {'HumanEval/0': PROBLEM})


def test_generation_and_resume(tmp_path, monkeypatch):
    import eval_llm
    import torch

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == PROBLEM['prompt']
            assert kwargs['truncation'] is False
            return Batch(input_ids=torch.tensor([[1, 2]]))

        def decode(self, ids, **kwargs):
            assert ids.tolist() == [3]
            return '    return a+b\n'

    class Model:
        def float(self):
            return self

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(eval_llm, 'init_model', lambda args: (Model(), Tokenizer()))
    monkeypatch.setattr(eval_llm, '_generation_kwargs', lambda *args: {})
    args = SimpleNamespace(output=str(tmp_path / 'samples.jsonl'), resume=False,
                           num_samples=1, device='cpu', prompt_style='base', seed=42)
    problems = {PROBLEM['task_id']: PROBLEM}
    he.generate(args, problems)
    assert he.read_jsonl(args.output)[0]['completion'] == '    return a+b\n'
    args.resume = True
    monkeypatch.setattr(eval_llm, 'init_model', lambda args: pytest.fail('Already complete'))
    he.generate(args, problems)
    assert len(he.read_jsonl(args.output)) == 1
