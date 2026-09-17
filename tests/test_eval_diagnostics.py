import gzip
import json

from eval_humaneval import check_sample
from eval_report import write_analysis
from eval_data import load_official_humaneval


def test_error_diagnostic():
    problem = {'prompt': 'def f():\n', 'test': 'def check(fn):\n    fn()', 'entry_point': 'f'}
    result = check_sample(problem, '    return unknown_name\n', 3, details=True)
    assert result['result'] == 'failed'
    assert result['error_type'] == 'NameError'
    assert 'unknown_name' in result['error_message']
    assert 'traceback' in result


def test_compact_report_deduplicates_and_archive_is_lossless(tmp_path):
    output = str(tmp_path / 'answers.jsonl')
    raw = 'x' * 2500
    rows = [{'task_id': 'a', 'passed': False, 'raw_response': raw, 'error_type': 'SyntaxError'}] * 3
    problems = {'a': {'prompt': 'task', 'test': 'test', 'canonical_solution': 'reference'}}
    report = write_analysis(output, rows, problems)
    task = report['tasks'][0]
    assert task['unique_failed_outputs'] == 1
    assert task['examples'][0]['count'] == 3
    assert task['examples'][0]['truncated']
    with gzip.open(output + '_analysis_full.jsonl.gz', 'rt', encoding='utf-8') as f:
        archive = json.loads(f.readline())
    assert archive['samples'] == rows
    assert archive['problem'] == problems['a']


def test_official_cache_works_without_network(tmp_path, monkeypatch):
    rows = [{'task_id': f'HumanEval/{i}', 'prompt': 'p', 'test': 't', 'entry_point': 'f'} for i in range(164)]
    path = tmp_path / 'HumanEval.jsonl.gz'
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(map(json.dumps, rows)))
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network used')))
    assert load_official_humaneval(path) == rows
