import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import eval_humaneval as he


def settings(tmp_path):
    return SimpleNamespace(output=str(tmp_path / 'answers.jsonl'), resume=False,
                           workers=2, batch_size=2, timeout=2, num_samples=2,
                           k=[1, 2], problem_file=None, limit=1)


PROBLEMS = {'a': {'task_id': 'a', 'prompt': 'def f():\n',
                  'entry_point': 'f', 'test': 'def check(f):\n    assert f() == 1'}}


def test_scoring_starts_before_generation_finishes(tmp_path, monkeypatch):
    args = settings(tmp_path)
    scoring_started = threading.Event()
    calls = []

    def score(problem, completion, timeout, details=False):
        calls.append(completion)
        scoring_started.set()
        return {'result': 'passed' if '1' in completion else 'failed'}

    def generate(args, problems, on_sample=None):
        with open(args.output, 'w', encoding='utf-8') as f:
            for completion in ('    return 1\n', '    return 0\n'):
                row = {'task_id': 'a', 'completion': completion}
                f.write(json.dumps(row) + '\n')
                f.flush()
                on_sample(row)
                # Generation is still active when the scoring worker runs.
                assert scoring_started.wait(5)

    monkeypatch.setattr(he, 'generate', generate)
    monkeypatch.setattr(he, 'check_sample', score)
    report = he.generate_and_evaluate(args, PROBLEMS)
    assert report['metrics'] == {'pass@1': .5, 'pass@2': 1.0}
    assert len(calls) == 2
    assert len(he.read_jsonl(args.output + '_results.jsonl')) == 2
    progress = json.loads((tmp_path / 'answers.jsonl_progress.json').read_text())
    assert progress == {'submitted': 2, 'completed': 2, 'pending': 0}


def test_pipeline_failure_keeps_completed_results_without_final_metrics(tmp_path, monkeypatch):
    args = settings(tmp_path)

    def generate(args, problems, on_sample=None):
        on_sample({'task_id': 'a', 'completion': '    return 1\n'})
        raise RuntimeError('generation failed')

    monkeypatch.setattr(he, 'generate', generate)
    with pytest.raises(RuntimeError, match='generation failed'):
        he.generate_and_evaluate(args, PROBLEMS)
    assert not (tmp_path / 'answers.jsonl_metrics.json').exists()
    # The running check is joined and the stream is closed even on failure.
    results = tmp_path / 'answers.jsonl_results.jsonl'
    with results.open('a') as stream:
        assert not stream.closed


def test_invalid_resume_preserves_existing_results(tmp_path):
    args = settings(tmp_path)
    (tmp_path / 'answers.jsonl').write_text('{}\n')
    result = tmp_path / 'answers.jsonl_results.jsonl'
    result.write_text('previous results')
    with pytest.raises(FileExistsError):
        he.generate_and_evaluate(args, PROBLEMS)
    assert result.read_text() == 'previous results'
    args.resume = True
    with pytest.raises(ValueError):
        he.generate_and_evaluate(args, PROBLEMS)
    assert result.read_text() == 'previous results'


def test_complete_generation_resume_is_scored_without_loading_model(tmp_path, monkeypatch):
    import eval_llm
    args = settings(tmp_path)
    args.resume = True
    rows = [{'task_id': 'a', 'completion': '    return 1\n'},
            {'task_id': 'a', 'completion': '    return 0\n'}]
    (tmp_path / 'answers.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n')
    monkeypatch.setattr(eval_llm, 'init_model', lambda args: pytest.fail('Should not reload model'))
    report = he.generate_and_evaluate(args, PROBLEMS)
    assert report['metrics'] == {'pass@1': .5, 'pass@2': 1.0}
    assert he.read_jsonl(args.output) == rows


def test_locked_progress_does_not_interrupt_scoring(tmp_path, monkeypatch, capsys):
    args = settings(tmp_path)
    original_replace = Path.replace

    def deny_progress(source, target):
        if str(target).endswith('_progress.json'):
            raise PermissionError(5, 'Access denied')
        return original_replace(source, target)

    monkeypatch.setattr(Path, 'replace', deny_progress)
    monkeypatch.setattr(he.time, 'sleep', lambda seconds: None)
    with he.StreamingEvaluator(args, PROBLEMS) as evaluator:
        evaluator.submit({'task_id': 'a', 'completion': '    return 1\n'})
        evaluator.submit({'task_id': 'a', 'completion': '    return 0\n'})
    rows = he.read_jsonl(args.output + '_results.jsonl')
    assert len(rows) == 2
    assert sum(row['passed'] for row in rows) == 1
    assert he.summarize_results(args, PROBLEMS, rows)['metrics'] == {'pass@1': .5, 'pass@2': 1.0}
    assert capsys.readouterr().out.count('Progress snapshot unavailable') == 1


def test_progress_replace_retries_then_recovers(tmp_path, monkeypatch):
    args = settings(tmp_path)
    original_replace = Path.replace
    attempts = []

    def transient_lock(source, target):
        if str(target).endswith('_progress.json'):
            attempts.append(1)
            if len(attempts) <= 2:
                raise PermissionError(5, 'Access denied')
        return original_replace(source, target)

    monkeypatch.setattr(Path, 'replace', transient_lock)
    monkeypatch.setattr(he.time, 'sleep', lambda seconds: None)
    with he.StreamingEvaluator(args, PROBLEMS) as evaluator:
        assert len(attempts) == 3
        evaluator.submit({'task_id': 'a', 'completion': '    return 1\n'})
    progress = json.loads(Path(args.output + '_progress.json').read_text())
    assert progress == {'submitted': 1, 'completed': 1, 'pending': 0}
