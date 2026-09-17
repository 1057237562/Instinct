"""Lossless compressed evidence plus compact, deduplicated failure summaries."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def compress_output(path):
    target = Path(str(path) + '.gz')
    temporary = target.with_suffix(target.suffix + '.tmp')
    with open(path, 'rb') as source, gzip.open(temporary, 'wb') as dest:
        shutil.copyfileobj(source, dest)
    temporary.replace(target)
    return target


def write_analysis(output, rows, problems, content_type='code'):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['task_id']].append(row)
    archive_path = str(output) + '_analysis_full.jsonl.gz'
    summary = {'num_tasks': len(grouped), 'num_samples': len(rows), 'failure_types': {}, 'tasks': [],
               'interpretation': '错误类型是可观察证据，不能证明训练数据缺失。摘要截短内容，完整题目/测试/输出在 gzip 文件。'}
    errors = Counter()
    with gzip.open(archive_path, 'wt', encoding='utf-8') as archive:
        for task, samples in grouped.items():
            problem = problems[task]
            archive.write(json.dumps({'task_id': task, 'problem': problem, 'samples': samples}, ensure_ascii=False) + '\n')
            failures = [row for row in samples if not row['passed']]
            unique = {}
            for row in failures:
                kind = row.get('error_type') or ('Timeout' if row.get('result') == 'timed out' else 'TestFailure')
                errors[kind] += 1
                raw = row.get('raw_response', row.get('completion', ''))
                digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()
                if digest not in unique:
                    unique[digest] = {'count': 0, 'error_type': kind, 'error_message': row.get('error_message', '')[:300],
                                      'output_excerpt': raw[:1200], 'output_chars': len(raw), 'truncated': len(raw) > 1200,
                                      'reached_token_limit': row.get('reached_token_limit', False)}
                unique[digest]['count'] += 1
            summary['tasks'].append({'task_id': task, 'num_samples': len(samples), 'num_correct': len(samples) - len(failures),
                'prompt_excerpt': problem.get('prompt', problem.get('question_content', ''))[:1600],
                'unique_failed_outputs': len(unique), 'examples': sorted(unique.values(), key=lambda r: -r['count'])[:2]})
    summary['failure_types'] = dict(errors.most_common())
    summary['tasks'].sort(key=lambda row: (row['num_correct'] / row['num_samples'], row['task_id']))
    Path(str(output) + '_analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 评测失败分析', '', summary['interpretation'], '',
             f"题目：{len(grouped)}；样本：{len(rows)}；失败类型：{dict(errors)}", '',
             ('建议检查：最终答案格式、算术运算、单位换算、多步推理和题意理解；仅凭错误答案不能确定训练数据缺失。'
              if content_type == 'math' else '建议依据错误类型检查：语法/缩进、变量与 API 使用、边界条件、算法正确性、复杂度和输出格式。'), '']
    for task in summary['tasks']:
        if task['num_correct'] == task['num_samples']:
            continue
        lines.extend([f"## {task['task_id']}：{task['num_correct']}/{task['num_samples']} 通过", '', task['prompt_excerpt'], ''])
        for example in task['examples']:
            lines.extend([f"{example['error_type']}（相同输出 {example['count']} 次）：{example['error_message']}", '',
                          '~~~~text' if content_type == 'math' else '~~~~python', example['output_excerpt'], '~~~~', ''])
    Path(str(output) + '_analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'[Eval] 精简分析：{output}_analysis.md；完整压缩证据：{archive_path}', flush=True)
    return summary
