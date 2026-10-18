"""Build all 164 HumanEval problems as complete-program, plain-text OJ tasks.

Run from any directory. The official snapshot is cached automatically. Canonical
solutions must pass every original check before publishing the evaluation bundle.
This builder does not create training data.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_data import DEFAULT_CACHE, HUMANEVAL_URL, load_official_humaneval
from scripts.data_builder import humaneval_oj_protocol as protocol

SOURCE = DEFAULT_CACHE
OUT_DIR = ROOT / 'dataset' / 'humaneval_oj'
IDENTITY_KEYWORDS = ('qwen', '通义千问', '千问', 'alibaba cloud', '阿里云', '阿里巴巴')


def run_job(job, timeout):
    with tempfile.TemporaryDirectory(prefix='humaneval-oj-build-') as directory:
        manifest = Path(directory) / 'job.json'
        result = Path(directory) / 'result.json'
        manifest.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        proc = subprocess.run([sys.executable, '-I', str(Path(protocol.__file__).resolve()),
                               str(manifest), str(result)], cwd=directory, timeout=timeout,
                              capture_output=True, text=True, encoding='utf-8',
                              env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        if proc.returncode or not result.is_file():
            raise RuntimeError(proc.stderr[-1500:] or 'Missing build result')
        payload = json.loads(result.read_text(encoding='utf-8'))
        if payload['result'] != 'passed':
            raise ValueError(f'{job["task"]["task_id"]}: official checks failed: {payload}')
        return payload


def signature_info(task):
    tree = ast.parse(task['prompt'])
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == task['entry_point'])
    # Remove doctests and their output; examples below use the OJ I/O contract.
    description = re.split(r'(?m)^\s*>>>', ast.get_docstring(fn) or '', maxsplit=1)[0].strip()
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != task['entry_point']]
    if helpers:
        description += '\n\nThe original problem defines these helper operations:\n```python\n' + '\n\n'.join(
            ast.unparse(n) for n in helpers) + '\n```'
    return fn, description


def reference_program(task, plan):
    helpers = '\n\n'.join(inspect.getsource(fn) for fn in
                            (protocol.tagged_schema, protocol.encode, protocol.decode, protocol.read_count))
    helpers = re.sub(r'\b(tagged_schema|encode|decode|read_count)\b', r'_oj_\1', helpers)
    return (helpers + '\n\n' + task['prompt'] + task['canonical_solution']
            + '\n\nimport sys\n'
            + f'_plan = {plan!r}\n'
            + "_lines = iter(sys.stdin.read().splitlines())\n"
            + "_args = [_oj_decode(_lines, p['schema']) for p in _plan['params']]\n"
            + f"_result = {task['entry_point']}(*_args)\n"
            + "sys.stdout.write(_oj_encode(_result, _plan['output'], output=True))\n")


def render_statement(task, description, plan, examples):
    sections = [f"## {task['task_id']}: {task['entry_point']}", description,
                'Write a complete Python program that reads one test case from standard input and writes its answer to standard output.',
                '### Input Format', 'Read the following sections in order (plain text):']
    sections.extend(f"- `{p['name']}`: {protocol.describe(p['schema'])}." for p in plan['params'])
    sections.extend(['### Output Format', protocol.describe(plan['output'], output=True) + '.',
                     'Use the stated format. Numeric answers follow the original problem checks, including any tolerance or valid-answer condition; dictionary entry order does not matter.'])
    for case in examples[:2]:
        sections.extend(['### Sample Input', '```\n' + case['input'] + '```',
                         '### Sample Output', '```\n' + case['output'] + '```'])
    return '\n\n'.join(sections) + '\n'


def build_dataset(source=SOURCE, output_dir=OUT_DIR):
    source, output_dir = Path(source), Path(output_dir)
    tasks = load_official_humaneval(source)
    report = {'protocol_version': protocol.PROTOCOL_VERSION, 'evaluation_only': True,
              'source_repository': 'https://github.com/openai/human-eval', 'source_url': HUMANEVAL_URL,
              'source_revision': 'content-addressed official snapshot',
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'input_rows': len(tasks), 'output_rows': 0, 'dropped_rows': 0, 'rewritten_rows': len(tasks),
              'schema_filter_count': 0, 'length_filter_count': 0,
              'identity_keyword_hits': {}, 'cases_verified': 0, 'task_details': {}}
    rows = []
    for task in tasks:
        records = run_job({'mode': 'trace', 'task': task, 'seed': 42}, 60)['records']
        if not records:
            raise ValueError(f'{task["task_id"]}: no original test calls')
        cases = [(protocol.parse_output(r['args_text'], {'kind': 'variant'}),
                  protocol.parse_output(r['result_text'], {'kind': 'variant'})) for r in records]
        fn, description = signature_info(task)
        params = [{'name': arg.arg, 'schema': protocol.infer_schema(
            [args[i] for args, _ in cases], ast.unparse(arg.annotation) if arg.annotation else None)}
                  for i, arg in enumerate(fn.args.args)]
        plan = {'params': params, 'output': protocol.infer_schema([value for _, value in cases])}
        test_cases = [{'input': protocol.render_input(args, plan),
                       'output': protocol.encode(value, plan['output'], output=True)} for args, value in cases]
        program = reference_program(task, plan)
        with tempfile.TemporaryDirectory(prefix='humaneval-oj-reference-') as directory:
            path = Path(directory) / 'program.py'
            path.write_text(program, encoding='utf-8')
            run_job({'mode': 'judge', 'task': task, 'plan': plan, 'program': str(path),
                     'timeout': 10, 'cwd': directory, 'seed': 42}, max(60, 10 * len(cases)))
        row = {'protocol_version': protocol.PROTOCOL_VERSION, 'task_id': task['task_id'],
               'entry_point': task['entry_point'], 'statement': render_statement(task, description, plan, test_cases),
               'io_plan': plan, 'num_test_cases': len(test_cases), 'test_cases': test_cases,
               'official_task': task, 'reference_program': program}
        rows.append(row)
        report['task_details'][task['task_id']] = {'original_test_calls': len(cases), 'reference_verified': True}
        report['cases_verified'] += len(cases)
        if len(rows) % 20 == 0:
            print(f'[HumanEval-OJ build] {len(rows)}/{len(tasks)} tasks verified', flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'humaneval_oj_problems.jsonl'
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    report['output_rows'] = len(rows)
    report['output_sha256'] = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    report['identity_keyword_hits'] = {word: payload.lower().count(word.lower()) for word in IDENTITY_KEYWORDS}
    # Publish only after every task has converted and every reference has passed.
    temporary = path.with_suffix('.tmp')
    temporary.write_text(payload, encoding='utf-8', newline='\n')
    temporary.replace(path)
    (output_dir / 'build.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[HumanEval-OJ build] {len(rows)} tasks, {report["cases_verified"]} original test calls -> {path}', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()
    build_dataset(args.source, args.output_dir)


if __name__ == '__main__':
    main()
