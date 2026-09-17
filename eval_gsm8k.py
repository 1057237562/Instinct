"""GSM8K test-set generation and exact numeric answer evaluation (zero-shot CoT)."""
import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import ssl
import tempfile
import urllib.request

from eval_humaneval import read_jsonl, validate_samples, estimate_pass_at_k
from eval_report import compress_output, write_analysis

DATA_URL = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'
DATA_CACHE = Path(__file__).resolve().parent / 'dataset' / 'gsm8k' / 'test.jsonl'
NUMBER = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)'


def extract_answer(text):
    """Require a final #### numeric line; never guess from intermediate arithmetic."""
    if '<think>' in text and '</think>' not in text:
        return None
    text = text.rsplit('</think>', 1)[-1]
    # Use the final marker; an invalid final answer cannot fall back to an earlier one.
    markers = list(re.finditer(r'(?m)^[ \t]*####[ \t]*', text))
    if not markers:
        return None
    tail = text[markers[-1].end():].strip()
    if not re.fullmatch(NUMBER, tail):
        return None
    try:
        value = Decimal(tail.replace(',', ''))
        return str(value) if value else '0'
    except InvalidOperation:
        return None


def load_problems(path=None, limit=0, cache=DATA_CACHE):
    if limit < 0:
        raise ValueError('limit must be nonnegative')
    cache = Path(cache)
    downloaded = None
    if path:
        rows = read_jsonl(path)
    elif cache.is_file():
        rows = read_jsonl(cache)
    else:
        try:
            with urllib.request.urlopen(DATA_URL, context=ssl.create_default_context(), timeout=30) as response:
                downloaded = response.read(16 * 1024 * 1024)
            rows = [json.loads(line) for line in downloaded.decode('utf-8').splitlines() if line.strip()]
        except Exception as exc:
            raise RuntimeError('无法下载官方 GSM8K，请检查网络/系统证书或指定 --problem_file 本地 JSONL。') from exc
    if not path and len(rows) != 1319:
        raise ValueError('GSM8K 官方 test 数据应包含 1319 道题')
    problems = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get('question'), str) or not row['question'].strip() or not isinstance(row.get('answer'), str):
            raise ValueError(f'第 {index} 题缺少 question / answer 字符串')
        gold = extract_answer(row['answer'])
        if gold is None:
            raise ValueError(f'第 {index} 题参考答案缺少有效 #### 数值')
        task_id = row.get('task_id', f'GSM8K/test/{index}')
        if not isinstance(task_id, str) or not task_id or task_id in problems:
            raise ValueError('GSM8K task_id 必须是唯一字符串')
        problems[task_id] = {**row, 'task_id': task_id, 'prompt': row['question'], 'gold_answer': gold}
    if not problems:
        raise ValueError('GSM8K 数据为空')
    if downloaded is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(downloaded)
        temporary.replace(cache)
        cache.with_suffix('.source.json').write_text(json.dumps({'url': DATA_URL,
            'sha256': hashlib.sha256(downloaded).hexdigest(), 'num_tasks': len(rows)}, indent=2), encoding='utf-8')
    return dict(list(problems.items())[:limit]) if limit else problems


def question_hash(problem):
    return hashlib.sha256(problem['question'].encode('utf-8')).hexdigest()


def validate_rows(rows, problems, maximum=None):
    counts = validate_samples(rows, problems, maximum)
    for row in rows:
        if row.get('question_sha256') and row['question_sha256'] != question_hash(problems[row['task_id']]):
            raise ValueError('已有答案与当前题目内容不匹配，请选择新的输出路径')
    return counts


def grade(row, problem):
    predicted = extract_answer(row['completion'])
    passed = predicted is not None and Decimal(predicted) == Decimal(problem['gold_answer'])
    return {**row, 'predicted_answer': predicted, 'gold_answer': problem['gold_answer'], 'passed': passed,
            'result': 'passed' if passed else 'failed',
            'error_type': '' if passed else ('AnswerFormatError' if predicted is None else 'WrongAnswer'),
            'error_message': '' if passed else f"predicted={predicted}; expected={problem['gold_answer']}"}


def input_text(tokenizer, problem, style, thinking=False):
    prompt = ('Solve the following math word problem step by step. End your response with a separate '
              'line in the exact format: #### <number>. Do not include units or text after the number.\n\n'
              + problem['question'])
    if style == 'base':
        return 'Question: ' + prompt + '\n\nAnswer:'
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                         add_generation_prompt=True, open_thinking=thinking)


def generate(args, problems):
    from datasets import load_dataset  # noqa: F401; Windows DLL order
    from eval_llm import init_model
    from eval_batch import generate_batches
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f'{output} exists; use --resume or another --output')
    saved = read_jsonl(output) if output.exists() else []
    counts = validate_rows(saved, problems, args.num_samples)
    if all(counts[t] == args.num_samples for t in problems):
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
        if counts[task_id] >= args.num_samples:
            continue
        text = input_text(tokenizer, problem, style, bool(args.open_thinking))
        for sample in range(counts[task_id], args.num_samples):
            jobs.append({'task_id': task_id, 'text': text, 'index': index * args.num_samples + sample})
    output.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if output.exists() and output.stat().st_size:
        with output.open('rb') as stream:
            stream.seek(-1, 2)
            needs_newline = stream.read(1) != b'\n'
    with output.open('a', encoding='utf-8') as stream:
        if needs_newline:
            stream.write('\n')
        for job, response, token_count in generate_batches(args, model, tokenizer, jobs, args.batch_size, args.seed):
            problem = problems[job['task_id']]
            row = {'task_id': job['task_id'], 'completion': response, 'raw_response': response,
                   'question_sha256': question_hash(problem), 'generated_tokens': token_count,
                   'reached_token_limit': token_count >= args.max_new_tokens}
            if args.mode == 'all':
                row = grade(row, problem)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            counts[job['task_id']] += 1
            print(f"[GSM8K] {job['task_id']} {counts[job['task_id']]}/{args.num_samples} {row.get('result', 'generated')}", flush=True)
    compress_output(output)


def evaluate(args, problems):
    rows = read_jsonl(args.output)
    counts = validate_rows(rows, problems)
    if set(counts) != set(problems):
        raise ValueError('每道选中题目至少需要一个答案')
    eligible = sorted({k for k in args.k if 1 <= k <= min(counts.values())})
    if not eligible:
        raise ValueError('样本数不足，无法计算请求的 K')
    graded = [grade(row, problems[row['task_id']]) for row in rows]
    correct = Counter(row['task_id'] for row in graded if row['passed'])
    metrics = {f'pass@{k}': sum(estimate_pass_at_k(counts[t], correct[t], k) for t in problems) / len(problems) for k in eligible}
    metrics['accuracy'] = sum(correct[t] / counts[t] for t in problems) / len(problems)
    report = {'benchmark': 'GSM8K', 'split': 'test' if not args.problem_file else 'local',
              'metrics': metrics, 'num_tasks': len(problems), 'num_samples': len(rows),
              'skipped_k': sorted(set(args.k) - set(eligible)), 'problem_file': args.problem_file,
              'limit': args.limit, 'prompt_protocol': 'zero-shot step-by-step',
              'answer_extraction': 'strict final #### numeric answer',
              'answer_format_errors': sum(r['error_type'] == 'AnswerFormatError' for r in graded)}
    with open(args.output + '_results.jsonl', 'w', encoding='utf-8') as stream:
        for row in graded:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    Path(args.output + '_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_analysis(args.output, graded, problems, content_type='math')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--mode', choices=['generate', 'evaluate', 'all'], default='generate')
    parser.add_argument('--problem_file')
    parser.add_argument('--output', default='eval/gsm8k_samples.jsonl')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--k', type=int, nargs='+', default=[1])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt_style', choices=['auto', 'base', 'chat'], default='auto')
    parser.add_argument('--resume', action='store_true')
    initial, _ = parser.parse_known_args()
    if initial.mode == 'evaluate':
        parser.add_argument('-h', '--help', action='help')
    else:
        from eval_llm import build_parser
        parser = argparse.ArgumentParser(parents=[parser, build_parser()], add_help=False)
        parser.set_defaults(temperature=0.0, max_new_tokens=512)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_samples < 1 or args.limit < 0 or any(k < 1 for k in args.k):
        parser.error('batch_size / num_samples / K 必须为正整数；limit 不能为负')
    if args.mode != 'evaluate':
        if not math.isfinite(args.temperature) or args.temperature < 0 or args.max_new_tokens < 1 or not 0 < args.top_p <= 1:
            parser.error('无效生成参数')
        if args.num_samples > 1 and args.temperature == 0:
            parser.error('多样本生成需要 temperature > 0')
        if args.output.endswith('.gz'):
            parser.error('生成路径必须为未压缩 JSONL（自动生成压缩副本）')
    problems = load_problems(args.problem_file, args.limit)
    if args.mode != 'evaluate':
        generate(args, problems)
    if args.mode in ('all', 'evaluate'):
        evaluate(args, problems)


if __name__ == '__main__':
    main()
