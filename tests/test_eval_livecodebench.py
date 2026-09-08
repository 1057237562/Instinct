import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import eval_llm

def test_livecodebench_prompt_matches_generic_format():
    prompt = eval_llm.format_livecodebench_prompt({
        'question_content': 'Add two numbers.',
        'starter_code': 'def add(a, b):',
    })

    assert prompt.startswith('### Question:\nAdd two numbers.\n\n')
    assert '```python\ndef add(a, b):\n```' in prompt
    assert prompt.endswith('### Answer: (use the provided format with backticks)\n\n')


def test_extract_livecodebench_code_uses_last_fenced_block():
    output = 'analysis\n```python\nprint("old")\n```\nanswer\n```python\nprint("new")\n```'

    assert eval_llm.extract_livecodebench_code(output) == 'print("new")'
    assert eval_llm.extract_livecodebench_code('print("unfenced")') == ''


def test_local_dataset_is_date_filtered_sorted_and_limited(tmp_path):
    dataset_path = tmp_path / 'lcb.jsonl'
    rows = [
        {'question_id': 'b', 'question_content': 'B', 'contest_date': '2024-02-01', 'starter_code': ''},
        {'question_id': 'a', 'question_content': 'A', 'contest_date': '2024-01-01', 'starter_code': ''},
        {'question_id': 'c', 'question_content': 'C', 'contest_date': '2024-03-01', 'starter_code': ''},
    ]
    dataset_path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    args = SimpleNamespace(
        lcb_dataset_path=str(dataset_path),
        lcb_release_version='release_v6',
        lcb_start_date='2024-01-15',
        lcb_end_date='2024-03-01',
        lcb_limit=1,
    )

    problems = eval_llm.load_livecodebench_problems(args)

    assert [problem['question_id'] for problem in problems] == ['b']


def test_resume_rejects_a_different_problem_range(tmp_path):
    output_path = tmp_path / 'generations.json'
    output_path.write_text(
        json.dumps([{'question_id': 'old', 'code_list': ['print(1)']}]),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='额外 question_id'):
        eval_llm._load_livecodebench_resume(output_path, {'new'}, 1)


def test_official_evaluator_command_forwards_release_and_dates(tmp_path):
    args = SimpleNamespace(
        lcb_output=str(tmp_path / 'generations.json'),
        lcb_release_version='release_v6',
        lcb_num_process_evaluate=2,
        lcb_timeout=9,
        lcb_start_date='2025-01-01',
        lcb_end_date=None,
    )

    command = eval_llm.build_livecodebench_evaluator_command(args)

    assert command[1:3] == ['-m', 'lcb_runner.runner.custom_evaluator']
    assert command[command.index('--release_version') + 1] == 'release_v6'
    assert command[command.index('--start_date') + 1] == '2025-01-01'
    assert '--end_date' not in command


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]['content']

    def __call__(self, text, **kwargs):
        return _FakeBatch({
            'input_ids': torch.tensor([[10, 11]]),
            'attention_mask': torch.tensor([[1, 1]]),
        })

    def decode(self, token_ids, **kwargs):
        return '```python\nprint("ok")\n```'


class _FakeModel:
    def generate(self, inputs, **kwargs):
        return torch.tensor([[10, 11, 12]])


def test_generation_writes_official_custom_evaluator_format(tmp_path):
    dataset_path = tmp_path / 'lcb.json'
    dataset_path.write_text(json.dumps([
        {'question_id': '2', 'question_content': 'B', 'starter_code': ''},
        {'question_id': '1', 'question_content': 'A', 'starter_code': ''},
    ]), encoding='utf-8')
    output_path = tmp_path / 'generations.json'
    args = SimpleNamespace(
        lcb_dataset_path=str(dataset_path),
        lcb_release_version='release_v6',
        lcb_start_date=None,
        lcb_end_date=None,
        lcb_limit=0,
        lcb_num_samples=2,
        lcb_seed=7,
        lcb_resume=1,
        lcb_output=str(output_path),
        lcb_prompt_style='chat',
        load_from='model',
        weight='full_sft',
        open_thinking=0,
        device='cpu',
        temperature=0.2,
        max_new_tokens=32,
        top_p=0.95,
        early_exit=0,
        exit_threshold=0.9,
        show_speed=0,
    )

    eval_llm.run_livecodebench_generation(args, _FakeModel(), _FakeTokenizer())

    assert json.loads(output_path.read_text(encoding='utf-8')) == [
        {'question_id': '1', 'code_list': ['print("ok")', 'print("ok")']},
        {'question_id': '2', 'code_list': ['print("ok")', 'print("ok")']},
    ]


def test_multiple_greedy_samples_are_rejected(tmp_path):
    args = SimpleNamespace(lcb_num_samples=2, temperature=0)

    with pytest.raises(ValueError, match='temperature'):
        eval_llm.run_livecodebench_generation(args, _FakeModel(), _FakeTokenizer())
