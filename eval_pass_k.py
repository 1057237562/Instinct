"""Compute pass@k from already graded HumanEval, GSM8K or LiveCodeBench results."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from eval_humaneval import estimate_pass_at_k, read_jsonl


def parse_k(value):
    ks = sorted(set(int(k) for k in value.replace(',', ' ').split()))
    if not ks or min(ks) < 1:
        raise ValueError('K 必须是正整数，例如 1,5,10')
    return ks


def calculate(path, ks):
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError('K 必须是正整数列表')
    path = Path(path)
    groups = defaultdict(list)
    if path.name.endswith(('.jsonl', '.jsonl.gz')):
        for row in read_jsonl(path):
            if not isinstance(row.get('task_id'), str) or type(row.get('passed')) is not bool:
                raise ValueError('JSONL 评分结果需要 task_id 和布尔 passed，不能使用未评分的答案文件')
            groups[row['task_id']].append(row['passed'])
    else:
        rows = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(rows, list):
            raise ValueError('LiveCodeBench 需要官方 *_eval_all.json 数组')
        for row in rows:
            if not isinstance(row, dict) or 'question_id' not in row:
                raise ValueError('LiveCodeBench 评分结果缺少 question_id')
            task = str(row['question_id'])
            grades = row.get('graded_list')
            if task in groups or not isinstance(grades, list) or not grades or any(type(v) is not bool for v in grades):
                raise ValueError('每题必须有唯一 question_id 和非空布尔 graded_list')
            groups[task] = grades
    if not groups:
        raise ValueError('评分结果为空')
    minimum = min(map(len, groups.values()))
    eligible = [k for k in sorted(set(ks)) if k <= minimum]
    metrics = {f'pass@{k}': sum(estimate_pass_at_k(len(v), sum(v), k) for v in groups.values()) / len(groups)
               for k in eligible}
    return {'metrics': metrics, 'num_tasks': len(groups),
            'num_samples': sum(map(len, groups.values())), 'min_samples_per_task': minimum,
            'skipped_k': sorted(set(ks) - set(eligible)), 'source': str(path.resolve()),
            'per_task': {t: {'num_samples': len(v), 'num_correct': sum(v)} for t, v in groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--k', nargs='+', type=int, default=[1, 5, 10])
    parser.add_argument('--output')
    args = parser.parse_args()
    report = calculate(args.results, args.k)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
