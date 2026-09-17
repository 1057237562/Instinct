from decimal import Decimal
import json
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import eval_gsm8k as gsm
from scripts.eval_webui_utils import ROOT, build_command


@pytest.mark.parametrize('text,expected', [
    ('reason\n#### 1,200', '1200'), ('#### -2.50', '-2.5'), ('#### .5', '.5'),
    ('#### -0', '0'), ('<think>#### 99</think>\n#### 12', '12'),
    ('Calculation: 4 + 8 = 12', None), ('#### 12 dollars', None),
    ('#### 12\n#### unknown', None), ('#### 12/2', None),
    ('<think>#### 12', None), ('#### 1,2', None),
    ('#### 12\ntext #### 13', None),
])
def test_extraction(text, expected):
    result = gsm.extract_answer(text)
    assert result is None if expected is None else Decimal(result) == Decimal(expected)


def problems():
    return {'GSM8K/test/0': {'task_id': 'GSM8K/test/0', 'question': 'What is 2+3?',
                           'prompt': 'What is 2+3?', 'answer': '2+3=5\n#### 5', 'gold_answer': '5'}}


def test_scoring_metrics_archives_and_format_failures(tmp_path):
    output = tmp_path / 'samples.jsonl'
    rows = [{'task_id': 'GSM8K/test/0', 'completion': text} for text in ['#### 5.0', '#### 4', '5']]
    output.write_text('\n'.join(map(json.dumps, rows)))
    args = SimpleNamespace(output=str(output), k=[1, 2, 5], problem_file='local', limit=1)
    report = gsm.evaluate(args, problems())
    assert report['metrics']['accuracy'] == pytest.approx(1/3)
    assert report['metrics']['pass@2'] == pytest.approx(2/3)
    assert report['answer_format_errors'] == 1
    assert report['skipped_k'] == [5]
    assert (tmp_path / 'samples.jsonl_analysis_full.jsonl.gz').is_file()
    results = gsm.read_jsonl(str(output) + '_results.jsonl')
    assert [row['error_type'] for row in results] == ['', 'WrongAnswer', 'AnswerFormatError']


def test_local_data_validation_and_resume_question_hash(tmp_path):
    path = tmp_path / 'test.jsonl'
    path.write_text(json.dumps({'question': '2+3?', 'answer': '#### 5'}))
    assert len(gsm.load_problems(path)) == 1
    with pytest.raises(ValueError, match='不匹配'):
        gsm.validate_rows([{'task_id': 'GSM8K/test/0', 'completion': '', 'question_sha256': 'wrong'}], problems())


def test_generation_and_completed_resume(tmp_path, monkeypatch):
    import eval_llm
    import eval_batch
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: messages[0]['content'])
    monkeypatch.setattr(eval_llm, 'init_model', lambda args: (SimpleNamespace(float=lambda: None), tokenizer))
    def batch(args, model, tokenizer, jobs, size, seed):
        for job in jobs:
            assert '2+3?' in job['text']
            assert '2+3=5' not in job['text']  # Reference solution must never reach the model.
            yield job, '#### 5', 4
    monkeypatch.setattr(eval_batch, 'generate_batches', batch)
    args = SimpleNamespace(output=str(tmp_path / 'answers.jsonl'), resume=False, num_samples=2,
                           device='cpu', prompt_style='chat', open_thinking=0, mode='all',
                           batch_size=2, seed=42, max_new_tokens=512)
    gsm.generate(args, problems())
    assert all(r['passed'] for r in gsm.read_jsonl(args.output))
    args.resume = True
    monkeypatch.setattr(eval_llm, 'init_model', lambda args: pytest.fail('Already complete'))
    gsm.generate(args, problems())
    assert len(gsm.read_jsonl(args.output)) == 2


def test_webui_gsm8k_command_and_page(tmp_path):
    command = build_command(dict(benchmark='GSM8K', mode='generate', output=str(tmp_path / 'samples.jsonl'),
                                 k='1', num_samples=1, batch_size=4, workers=4, timeout=3))
    assert any(p.endswith('eval_gsm8k.py') for p in command)
    assert '--workers' not in command and '--timeout' not in command
    app = AppTest.from_file(str(ROOT / 'scripts/eval_webui.py'), default_timeout=15).run()
    app.sidebar.selectbox[0].select('GSM8K').run()
    assert not app.exception
    assert any('eval/gsm8k_' in caption.value for caption in app.caption)
    # Scoring UI must work independently of whatever model files the user has.
    output = tmp_path / 'samples.jsonl'
    output.write_text('')
    app.radio[0].set_value('evaluate').run()
    app.selectbox(key='gsm8k_score_output').select('自定义…').run()
    app.text_input(key='gsm8k_score_output_custom').set_value(str(output)).run()
    assert any('eval_gsm8k.py' in code.value and '--mode evaluate' in code.value for code in app.code)
    assert not any(s.label == '评分超时（秒）' for s in app.selectbox)
