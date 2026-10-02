"""HumanEval generation and functional evaluation; run from the repository root."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time


def read_jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_problems(path=None, limit=0):
    if path:
        rows = read_jsonl(path)
    else:
        from eval_data import load_official_humaneval
        rows = load_official_humaneval()
    problems = {}
    for row in rows:
        for field in ('task_id', 'prompt', 'entry_point', 'test'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f'Invalid HumanEval field: {field}')
        if row['task_id'] in problems:
            raise ValueError(f'Duplicate task_id: {row["task_id"]}')
        problems[row['task_id']] = row
    if not problems:
        raise ValueError('No HumanEval problems')
    return dict(list(problems.items())[:limit]) if limit else problems


def extract_completion(response, prompt, style):
    if style == 'base':
        # Stop before unrelated top-level material, preserving body indentation.
        return re.split(r'\n(?=def |class |if __name__|#|print\()', response)[0]
    response = response.rsplit('</think>', 1)[-1]
    blocks = re.findall(r'```(?:python|py)?[ \t]*\r?\n(.*?)```', response, re.S)
    code = blocks[-1] if blocks else response
    if code.startswith(prompt):
        return code[len(prompt):]
    # Official execution concatenates prompt + completion. A complete chat
    # solution can redefine the target at module scope (including its imports).
    if re.search(r'^(?:async )?def \w+\(', code, re.M):
        return '    pass\n\n' + code.lstrip('\r\n') + '\n'
    return code


def validate_samples(rows, problems, maximum=None):
    counts = Counter()
    for row in rows:
        if row.get('task_id') not in problems or not isinstance(row.get('completion'), str):
            raise ValueError('Samples must contain selected task_id and string completion')
        counts[row['task_id']] += 1
    if maximum is not None and any(n > maximum for n in counts.values()):
        raise ValueError('Existing sample count exceeds --num_samples')
    return counts


def generate(args, problems, on_sample=None):
    # Keep datasets before torch for the Windows pyarrow/torch DLL workaround.
    from datasets import load_dataset  # noqa: F401
    import torch
    from eval_llm import init_model
    from eval_batch import generate_batches
    from eval_report import compress_output

    output = Path(args.output)
    rows = read_jsonl(output) if args.resume and output.exists() else []
    counts = validate_samples(rows, problems, args.num_samples)
    if output.exists() and not args.resume:
        raise FileExistsError(f'{output} exists; use --resume or another --output')
    if on_sample:
        for row in rows:
            on_sample(row)
    if all(counts[task_id] == args.num_samples for task_id in problems):
        compress_output(output)
        return
    model, tokenizer = init_model(args)
    if args.device == 'cpu':
        model.float()
    style = args.prompt_style
    if style == 'auto':
        style = 'base' if args.load_from == 'model' and 'pretrain' in args.weight else 'chat'
    jobs = []
    for index, (task_id, problem) in enumerate(problems.items()):
        if counts[task_id] == args.num_samples:
            continue
        prompt = problem['prompt']
        text = prompt if style == 'base' else tokenizer.apply_chat_template(
            [{'role': 'user', 'content':
              'Complete the following Python function. Return the complete solution '
              'in a Python code block, including any required imports.\n\n' + prompt}],
            tokenize=False, add_generation_prompt=True,
            open_thinking=bool(args.open_thinking),
        )
        for sample_index in range(counts[task_id], args.num_samples):
            jobs.append(dict(task_id=task_id, text=text, prompt=prompt,
                             index=index * args.num_samples + sample_index))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('a', encoding='utf-8') as stream:
        for job, response, token_count in generate_batches(
                args, model, tokenizer, jobs, getattr(args, 'batch_size', 1), args.seed):
            row = {'task_id': job['task_id'],
                   'completion': extract_completion(response, job['prompt'], style),
                   'raw_response': response, 'generated_tokens': token_count,
                   'reached_token_limit': token_count >= getattr(args, 'max_new_tokens', float('inf'))}
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if on_sample:
                on_sample(row)
            counts[job['task_id']] += 1
            print(f"[HumanEval] {job['task_id']} {counts[job['task_id']]}/{args.num_samples}", flush=True)
    compress_output(output)


def check_sample(problem, completion, timeout, details=False):
    """Execute in a fresh interpreter. This is a timeout boundary, NOT a sandbox."""
    program = (problem['prompt'] + completion + '\n' + problem['test']
               + '\ncheck(' + problem['entry_point'] + ')\n')
    # A separate harness rejects SystemExit(0) instead of treating it as a pass.
    harness = (
        'import sys, json, traceback\n'
        'source = sys.stdin.buffer.read().decode("utf-8")\n'
        'try:\n'
        '    exec(compile(source, "<humaneval>", "exec"), {})\n'
        'except BaseException as exc:\n'
        '    with open("diagnostic.json", "w", encoding="utf-8") as f:\n'
        '        json.dump({"error_type": type(exc).__name__, "error_message": str(exc)[:500], "traceback": traceback.format_exc(limit=3)[-2000:]}, f)\n'
        '    sys.exit(1)\n'
    )
    with tempfile.TemporaryDirectory(prefix='humaneval-') as directory:
        try:
            result = subprocess.run(
                [sys.executable, '-I', '-c', harness], input=program.encode('utf-8'),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                cwd=directory, timeout=timeout,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '1'},
            )
        except subprocess.TimeoutExpired:
            return {'result': 'timed out', 'error_type': 'Timeout', 'error_message': f'Exceeded {timeout}s'} if details else 'timed out'
        status = 'passed' if result.returncode == 0 else 'failed'
        diagnostic = {'result': status}
        if status != 'passed':
            diagnostic['error_type'] = 'ProcessError'
            try:
                with open(Path(directory) / 'diagnostic.json', encoding='utf-8') as stream:
                    data = json.loads(stream.read(16000))
                diagnostic.update({k: str(data[k])[:2000] for k in ('error_type', 'error_message', 'traceback') if k in data})
            except (OSError, ValueError, TypeError):
                pass
    return diagnostic if details else status


def estimate_pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError('Require 0 <= c <= n and 1 <= k <= n')
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k)) if n - c >= k else 1.0


class StreamingEvaluator:
    """Bounded CPU evaluation queue, independently flushed while generation runs."""
    def __init__(self, args, problems):
        self.args, self.problems = args, problems
        self.capacity = max(args.workers * 2, getattr(args, 'batch_size', 1))
        self.pending = deque()
        self.lock = threading.Lock()
        self.submitted = self.completed = 0
        self._last_progress_time = float('-inf')
        self._progress_warning_logged = False

    def __enter__(self):
        path = Path(self.args.output + '_results.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open('w', encoding='utf-8')
        self.pool = ThreadPoolExecutor(max_workers=self.args.workers)
        self._progress()
        return self

    def _progress(self, force=False):
        # The UI polls this optional snapshot. Windows readers (including
        # scanners/filesystem drivers) can briefly prevent replacing it.
        # Telemetry must never turn a successfully scored answer into failure.
        now = time.monotonic()
        if not force and now - self._last_progress_time < 0.5:
            return
        self._last_progress_time = now
        path = Path(self.args.output + '_progress.json')
        temporary = path.with_suffix('.tmp')
        payload = json.dumps({'submitted': self.submitted, 'completed': self.completed,
                              'pending': self.submitted - self.completed})
        for attempt in range(3):
            try:
                temporary.write_text(payload, encoding='utf-8')
                temporary.replace(path)
                return
            except OSError as exc:
                if attempt < 2:
                    time.sleep(0.02 * (attempt + 1))
                elif not self._progress_warning_logged:
                    print(f'[HumanEval] Progress snapshot unavailable ({exc}); '
                          'generation and result saving continue. See scoring logs for progress.', flush=True)
                    self._progress_warning_logged = True

    def submit(self, row):
        # Backpressure bounds queued source text and futures if CPU tests lag.
        while self.pending and (self.pending[0].done() or len(self.pending) >= self.capacity):
            self.pending.popleft().result()
        with self.lock:
            self.submitted += 1
            self._progress()
        self.pending.append(self.pool.submit(self._run, dict(row)))

    def _run(self, row):
        diagnostic = check_sample(self.problems[row['task_id']], row['completion'], self.args.timeout, details=True)
        graded = {**row, **diagnostic, 'passed': diagnostic['result'] == 'passed'}
        with self.lock:
            self.stream.write(json.dumps(graded, ensure_ascii=False) + '\n')
            self.stream.flush()
            self.completed += 1
            self._progress()
            print(f"[HumanEval scoring] {self.completed}/{self.submitted} {row['task_id']}: {diagnostic['result']}", flush=True)

    def __exit__(self, exc_type, exc, tb):
        try:
            self.pool.shutdown(wait=True, cancel_futures=exc_type is not None)
            if exc_type is None:
                for future in self.pending:
                    future.result()
        finally:
            with self.lock:
                self._progress(force=True)
            self.stream.close()


def generate_and_evaluate(args, problems):
    # Validate before opening the results file; a bad resume must not clobber it.
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f'{output} exists; use --resume or another --output')
    if output.exists():
        validate_samples(read_jsonl(output), problems, args.num_samples)
    print('[HumanEval] Pipeline enabled: GPU generation + CPU scoring.', flush=True)
    with StreamingEvaluator(args, problems) as evaluator:
        generate(args, problems, on_sample=evaluator.submit)
        print('[HumanEval] Generation complete; waiting for remaining tests.', flush=True)
    rows = read_jsonl(args.output + '_results.jsonl')
    if len(rows) != len(problems) * args.num_samples:
        raise ValueError('Incomplete pipeline results; final metrics were not produced')
    return summarize_results(args, problems, rows)


def evaluate(args, problems):
    rows = read_jsonl(args.output)
    totals = validate_samples(rows, problems)
    if set(totals) != set(problems):
        raise ValueError('Every selected problem must have at least one sample')
    eligible = [k for k in args.k if min(totals.values()) >= k]
    if not eligible:
        raise ValueError('No requested k has enough samples per problem')
    print('[HumanEval] Executing generated Python code in subprocesses (not a security sandbox).')
    correct = Counter()
    def run(row):
        diagnostic = check_sample(problems[row['task_id']], row['completion'], args.timeout, details=True)
        return {**row, **diagnostic, 'passed': diagnostic['result'] == 'passed'}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        with open(args.output + '_results.jsonl', 'w', encoding='utf-8') as stream:
            for row in pool.map(run, rows):
                correct[row['task_id']] += int(row['passed'])
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return summarize_results(args, problems, read_jsonl(args.output + '_results.jsonl'))


def summarize_results(args, problems, rows):
    totals = validate_samples(rows, problems)
    if set(totals) != set(problems):
        raise ValueError('Every selected problem must have at least one graded sample')
    if any(type(row.get('passed')) is not bool for row in rows):
        raise ValueError('Missing boolean passed field')
    correct = Counter(row['task_id'] for row in rows if row['passed'])
    eligible = [k for k in args.k if min(totals.values()) >= k]
    if not eligible:
        raise ValueError('No requested k has enough samples per problem')
    metrics = {f'pass@{k}': sum(estimate_pass_at_k(totals[t], correct[t], k)
                               for t in problems) / len(problems) for k in eligible}
    report = {'metrics': metrics, 'num_tasks': len(problems), 'num_samples': len(rows),
              'skipped_k': [k for k in args.k if k not in eligible],
              'problem_file': args.problem_file, 'limit': args.limit,
              'timeout': args.timeout}
    Path(args.output + '_metrics.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    from eval_report import write_analysis
    write_analysis(args.output, read_jsonl(args.output + '_results.jsonl'), problems)
    print(json.dumps(report, indent=2))
    return report


def main():
    # Parse evaluation options first so scoring existing samples needs no torch.
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--mode', choices=['generate', 'evaluate', 'all'], default='generate')
    parser.add_argument('--problem_file', help='Local HumanEval JSONL or JSONL.GZ; default: cached OpenAI official data')
    parser.add_argument('--output', default='eval/humaneval_samples.jsonl')
    parser.add_argument('--batch_size', type=int, default=1, help='Concurrent generation sequences')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--prompt_style', choices=['auto', 'base', 'chat'], default='auto')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--k', nargs='+', type=int, default=[1, 10, 100])
    parser.add_argument('--timeout', type=float, default=3.0)
    parser.add_argument('--workers', type=int, default=4)
    initial, _ = parser.parse_known_args()
    if initial.mode == 'evaluate':
        parser.add_argument('-h', '--help', action='help')
    else:
        from eval_llm import build_parser
        model_parser = build_parser()
        parser = argparse.ArgumentParser(parents=[parser, model_parser], add_help=False)
        parser.set_defaults(max_new_tokens=512, temperature=0.0)
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
