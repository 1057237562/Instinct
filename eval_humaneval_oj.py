"""HumanEval-OJ: complete Python programs with plain-text stdin/stdout.

The default bundle is built automatically from all 164 official problems. V2
judging adapts stdout back to values and runs every original check, preserving
float tolerances, randomized loops and multiple valid answers. V1 local bundles
retain their legacy text comparison. Reports identify this as HumanEval-OJ.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval_humaneval import read_jsonl, validate_samples, summarize_results as _summarize_results
from scripts.data_builder import humaneval_oj_protocol as protocol

DEFAULT_PROBLEM_FILE = 'dataset/humaneval_oj/humaneval_oj_problems.jsonl'
INSTRUCTION = ('Solve the following competitive programming problem. Return a complete Python '
               'program that reads from standard input and prints to standard output. '
               'You may reason about the solution before the final answer. Finish with one '
               'Python code block containing the complete program.\n\n')


def load_problems(path=None, limit=0):
    path = Path(path or DEFAULT_PROBLEM_FILE)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    default = (Path(__file__).resolve().parent / DEFAULT_PROBLEM_FILE).resolve()
    is_default = path.resolve() == default
    if not path.is_file():
        if not is_default:
            raise FileNotFoundError(f'HumanEval-OJ 数据文件不存在：{path}。'
                                    '留空使用自动构建的默认题库，或先运行 scripts/data_builder/build_humaneval_oj.py。')
        print('[HumanEval-OJ] 默认题库尚未生成，正在转换并验证全部 164 题。', flush=True)
        from scripts.data_builder.build_humaneval_oj import build_dataset
        build_dataset(output_dir=path.parent)
    rows = read_jsonl(path)
    if is_default and (len(rows) != 164 or any(row.get('protocol_version') != protocol.PROTOCOL_VERSION for row in rows)):
        print('[HumanEval-OJ] 重建旧题库，以保留全部题目及官方测试。', flush=True)
        from scripts.data_builder.build_humaneval_oj import build_dataset
        build_dataset(output_dir=path.parent)
        rows = read_jsonl(path)
    problems = {}
    for row in rows:
        for field in ('task_id', 'statement', 'test_cases', 'entry_point'):
            if field not in row:
                raise ValueError(f'Invalid HumanEval-OJ field: {field}')
        if not isinstance(row['test_cases'], list) or not row['test_cases']:
            raise ValueError(f'{row["task_id"]}: no test cases')
        if row['task_id'] in problems:
            raise ValueError(f'Duplicate task: {row["task_id"]}')
        # summarize_results/write_analysis read a prompt excerpt; statement serves as one.
        problems[row['task_id']] = {**row, 'prompt': row['statement']}
    if not problems:
        raise ValueError('No HumanEval-OJ problems')
    return dict(list(problems.items())[:limit]) if limit else problems


def extract_program(response):
    """Prefer the longest fenced block; models often add short usage snippets."""
    if '</think>' in response:
        response = response.rsplit('</think>', 1)[1]
    elif '<think>' in response:
        return ''
    blocks = re.findall(r'```(?:python)?[ \t]*\r?\n(.*?)(?:```|\Z)', response, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip('\n')
    return response.strip()


def check_program(problem, program, timeout, details=False):
    """Use complete official checks for V2, legacy text comparison otherwise."""
    if problem.get('protocol_version') == protocol.PROTOCOL_VERSION:
        return check_official_program(problem, program, timeout, details)
    with tempfile.TemporaryDirectory(prefix='humaneval-oj-') as directory:
        path = Path(directory) / 'program.py'
        path.write_text(program, encoding='utf-8')
        for index, case in enumerate(problem['test_cases']):
            try:
                result = subprocess.run(
                    [sys.executable, '-I', str(path)], input=case['input'],
                    capture_output=True, text=True, encoding='utf-8', timeout=timeout, cwd=directory,
                    env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '1'},
                )
            except subprocess.TimeoutExpired:
                diagnostic = {'result': 'timed out', 'error_type': 'Timeout',
                              'error_message': f'Case {index}: exceeded {timeout}s'}
                return diagnostic if details else 'timed out'
            if result.returncode != 0:
                diagnostic = {'result': 'failed', 'error_type': 'RuntimeError',
                              'error_message': f'Case {index}: {result.stderr.strip().splitlines()[-1][:500] if result.stderr.strip() else "nonzero exit"}',
                              'traceback': result.stderr[-2000:]}
                return diagnostic if details else 'failed'
            if result.stdout.rstrip('\n') != case['output'].rstrip('\n'):
                diagnostic = {'result': 'failed', 'error_type': 'WrongAnswer',
                              'error_message': f'Case {index}: expected {case["output"].rstrip(chr(10))[:300]!r}, got {result.stdout.rstrip(chr(10))[:300]!r}'}
                return diagnostic if details else 'failed'
    return {'result': 'passed'} if details else 'passed'


def check_official_program(problem, program, timeout, details=False):
    with tempfile.TemporaryDirectory(prefix='humaneval-oj-') as directory:
        path = Path(directory) / 'program.py'
        path.write_text(program, encoding='utf-8')
        manifest, result = Path(directory) / 'job.json', Path(directory) / 'result.json'
        manifest.write_text(json.dumps({'mode': 'judge', 'task': problem['official_task'],
            'plan': problem['io_plan'], 'program': str(path), 'cwd': directory,
            'timeout': timeout, 'seed': 42}, ensure_ascii=False), encoding='utf-8')
        try:
            proc = subprocess.run([sys.executable, '-I', str(Path(protocol.__file__).resolve()),
                str(manifest), str(result)], capture_output=True, text=True, encoding='utf-8',
                timeout=max(10, timeout * (problem['num_test_cases'] + 1)), cwd=directory,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '1'})
            diagnostic = json.loads(result.read_text(encoding='utf-8')) if result.is_file() else {
                'result': 'failed', 'error_type': 'JudgeError', 'error_message': proc.stderr[-1500:]}
        except subprocess.TimeoutExpired:
            diagnostic = {'result': 'timed out', 'error_type': 'Timeout', 'error_message': 'Official test runner exceeded its total time budget'}
    return diagnostic if details else diagnostic['result']


def summarize_results(args, problems, rows):
    return _summarize_results(args, problems, rows, metadata={
        'benchmark': 'HumanEval-OJ',
        'test_seed': 42,
        'protocol_versions': sorted({p.get('protocol_version', 1) for p in problems.values()}),
        'judge': 'official checks via stdin/stdout adapter' if all(
            p.get('protocol_version') == protocol.PROTOCOL_VERSION for p in problems.values()) else 'legacy/mixed OJ bundle'})


def _write_oj_stats(output, passed, failed, types):
    """Live tally snapshot the Eval WebUI shows while scoring runs."""
    path = Path(str(output) + '_oj_stats.json')
    temporary = path.with_suffix('.tmp')
    payload = json.dumps({'passed': passed, 'failed': failed,
                          'failure_types': dict(types)}, ensure_ascii=False)
    try:
        temporary.write_text(payload, encoding='utf-8')
        temporary.replace(path)
    except OSError:
        pass


def _oj_run(evaluator_impl, problem, row):
    started = time.monotonic()
    diagnostic = check_program(problem, row['completion'], evaluator_impl.args.timeout, details=True)
    graded = {**row, **diagnostic, 'passed': diagnostic['result'] == 'passed'}
    with evaluator_impl.lock:
        evaluator_impl.stream.write(json.dumps(graded, ensure_ascii=False) + '\n')
        evaluator_impl.stream.flush()
        evaluator_impl.completed += 1
        if graded['passed']:
            evaluator_impl.oj_passed += 1
        else:
            evaluator_impl.oj_failed += 1
            evaluator_impl.oj_types[diagnostic.get('error_type', diagnostic['result'])] += 1
        evaluator_impl._progress()
        _write_oj_stats(evaluator_impl.args.output, evaluator_impl.oj_passed,
                        evaluator_impl.oj_failed, evaluator_impl.oj_types)
        status = 'PASSED' if graded['passed'] else f"FAILED {diagnostic.get('error_type', diagnostic['result'])}"
        detail = '' if graded['passed'] else f" | {diagnostic.get('error_message', '')[:140]}"
        print(f"[HumanEval-OJ scoring] {evaluator_impl.completed}/{evaluator_impl.submitted} "
              f"{row['task_id']} {status} | pass={evaluator_impl.oj_passed} fail={evaluator_impl.oj_failed} "
              f"cases={problem['num_test_cases']} {time.monotonic() - started:.1f}s{detail}", flush=True)


def generate(args, problems, on_sample=None):
    from eval_report import compress_output
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f'{output} exists; use --resume or another --output')
    rows = read_jsonl(output) if output.exists() else []
    counts = validate_samples(rows, problems, args.num_samples)
    if rows:
        print(f"[HumanEval-OJ gen] resume: found {len(rows)} existing samples in {output}", flush=True)
    if on_sample:
        for row in rows:
            on_sample(row)
    style = args.prompt_style
    if style == 'auto':
        style = 'base' if args.load_from == 'model' and 'pretrain' in args.weight else 'chat'
    pending = sum(args.num_samples - counts[task_id] for task_id in problems)
    print(f"[HumanEval-OJ gen] start: problems={len(problems)} num_samples={args.num_samples} "
          f"pending_jobs={pending} batch_size={args.batch_size} prompt_style={style} "
          f"max_new_tokens={args.max_new_tokens} temperature={args.temperature} "
          f"thinking={'on' if getattr(args, 'open_thinking', 0) else 'off'} -> {output}", flush=True)
    from eval_llm import init_model
    model, tokenizer = init_model(args)
    if args.device == 'cpu':
        model.float()
    from eval_batch import generate_batches
    jobs = []
    for index, (task_id, problem) in enumerate(problems.items()):
        text = INSTRUCTION + problem['statement'] if style == 'base' else tokenizer.apply_chat_template(
            [{'role': 'user', 'content': INSTRUCTION + problem['statement']}],
            tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        for sample_index in range(counts[task_id], args.num_samples):
            jobs.append(dict(task_id=task_id, text=text,
                             index=index * args.num_samples + sample_index))
    output.parent.mkdir(parents=True, exist_ok=True)
    generation_start = time.monotonic()
    hit_limit = no_block = 0
    with output.open('a', encoding='utf-8') as stream:
        for done, (job, response, token_count) in enumerate(generate_batches(
                args, model, tokenizer, jobs, args.batch_size, args.seed), 1):
            fenced = response.count('```')
            reached = token_count >= args.max_new_tokens
            hit_limit += reached
            no_block += fenced == 0
            row = {'task_id': job['task_id'],
                   'completion': extract_program(response),
                   'raw_response': response, 'generated_tokens': token_count,
                   'reached_token_limit': reached}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if on_sample:
                on_sample(row)
            warnings = (' HIT_MAX_NEW_TOKENS' if reached else '') + (' NO_CODE_BLOCK' if fenced == 0 else '')
            print(f"[HumanEval-OJ gen] {done}/{len(jobs)} {job['task_id']} "
                  f"tokens={token_count} fences={fenced}{warnings}", flush=True)
    elapsed = time.monotonic() - generation_start
    print(f"[HumanEval-OJ gen] done: {len(jobs)} samples in {elapsed:.1f}s "
          f"({elapsed / len(jobs) if jobs else 0:.1f}s/sample) | "
          f"hit_limit={hit_limit} no_code_block={no_block}", flush=True)
    compress_output(output)


def generate_and_evaluate(args, problems):
    from eval_humaneval import StreamingEvaluator
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f'{output} exists; use --resume or another --output')
    if output.exists():
        validate_samples(read_jsonl(output), problems, args.num_samples)
    print('[HumanEval-OJ] Pipeline enabled: GPU generation + CPU scoring.', flush=True)
    evaluator = StreamingEvaluator(args, problems)
    evaluator.oj_passed = 0
    evaluator.oj_failed = 0
    evaluator.oj_types = Counter()
    evaluator._run = lambda row: _oj_run(evaluator, problems[row['task_id']], row)
    print(f"[HumanEval-OJ scoring] start: workers={args.workers} timeout={args.timeout}s "
          f"stdin/stdout adapter + original problem checks", flush=True)
    with evaluator:
        generate(args, problems, on_sample=evaluator.submit)
        print('[HumanEval-OJ] Generation complete; waiting for remaining tests.', flush=True)
    rows = read_jsonl(args.output + '_results.jsonl')
    if len(rows) != len(problems) * args.num_samples:
        raise ValueError('Incomplete pipeline results; final metrics were not produced')
    print(f"[HumanEval-OJ scoring] done: passed={evaluator.oj_passed} failed={evaluator.oj_failed} "
          f"({len(rows)} samples)", flush=True)
    return summarize_results(args, problems, rows)


def evaluate(args, problems):
    rows = read_jsonl(args.output)
    totals = validate_samples(rows, problems)
    if set(totals) != set(problems):
        raise ValueError('Every selected problem must have at least one sample')
    eligible = [k for k in args.k if min(totals.values()) >= k]
    if not eligible:
        raise ValueError('No requested k has enough samples per problem')
    print(f"[HumanEval-OJ] Executing generated programs in subprocesses (not a security sandbox): "
          f"samples={len(rows)} workers={args.workers} timeout={args.timeout}s", flush=True)
    correct = Counter()
    passed_total = failed_total = 0
    failure_types = Counter()
    scoring_start = time.monotonic()

    def run(row):
        diagnostic = check_program(problems[row['task_id']], row['completion'], args.timeout, details=True)
        return {**row, **diagnostic, 'passed': diagnostic['result'] == 'passed'}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        with open(args.output + '_results.jsonl', 'w', encoding='utf-8') as stream:
            for done, row in enumerate(pool.map(run, rows), 1):
                correct[row['task_id']] += int(row['passed'])
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                if row['passed']:
                    passed_total += 1
                else:
                    failed_total += 1
                    failure_types[row.get('error_type', row['result'])] += 1
                _write_oj_stats(args.output, passed_total, failed_total, failure_types)
                status = 'PASSED' if row['passed'] else f"FAILED {row.get('error_type', row['result'])}"
                detail = '' if row['passed'] else f" | {row.get('error_message', '')[:140]}"
                print(f"[HumanEval-OJ scoring] {done}/{len(rows)} {row['task_id']} {status} | "
                      f"pass={passed_total} fail={failed_total} "
                      f"cases={problems[row['task_id']]['num_test_cases']}{detail}", flush=True)
    print(f"[HumanEval-OJ scoring] done: passed={passed_total} failed={failed_total} "
          f"in {time.monotonic() - scoring_start:.1f}s", flush=True)
    return summarize_results(args, problems, read_jsonl(args.output + '_results.jsonl'))


def main():
    # Parse evaluation options first so scoring existing samples needs no torch.
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--mode', choices=['generate', 'evaluate', 'all'], default='generate')
    parser.add_argument('--problem_file', default=DEFAULT_PROBLEM_FILE,
                        help='HumanEval-OJ problems JSONL from build_humaneval_oj.py')
    parser.add_argument('--output', default='eval/humaneval_oj_samples.jsonl')
    parser.add_argument('--batch_size', type=int, default=1, help='Concurrent generation sequences')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--prompt_style', choices=['auto', 'base', 'chat'], default='auto')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--k', nargs='+', type=int, default=[1, 10, 100])
    parser.add_argument('--timeout', type=float, default=10.0)
    parser.add_argument('--workers', type=int, default=4)
    initial, _ = parser.parse_known_args()
    if initial.mode == 'evaluate':
        parser.add_argument('-h', '--help', action='help')
    else:
        from eval_llm import build_parser
        model_parser = build_parser()
        parser = argparse.ArgumentParser(parents=[parser, model_parser], add_help=False)
        # Reference programs reach 827 tokens (p99 760) and the model's narration adds
        # several times that; 512 truncated every smoke-test answer, 2048 left 2 tokens
        # of margin on the longest observed run.
        parser.set_defaults(max_new_tokens=4096, temperature=0.0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_samples < 1 or args.limit < 0 or args.workers < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or any(k < 1 for k in args.k):
        parser.error('Counts/timeout/k must be positive; limit must be nonnegative')
    if args.mode != 'evaluate' and (args.temperature < 0 or not math.isfinite(args.temperature) or not 0 < args.top_p <= 1 or args.max_new_tokens < 1):
        parser.error('Invalid generation parameters')
    if args.mode != 'evaluate' and args.num_samples > 1 and args.temperature <= 0:
        parser.error('Multiple samples require --temperature > 0')
    if args.mode != 'evaluate' and args.output.endswith('.gz'):
        parser.error('Generation output must be uncompressed JSONL')
    problems = load_problems(args.problem_file, args.limit)
    if args.mode == 'all':
        generate_and_evaluate(args, problems)
    elif args.mode == 'generate':
        generate(args, problems)
    else:
        evaluate(args, problems)


if __name__ == '__main__':
    main()
