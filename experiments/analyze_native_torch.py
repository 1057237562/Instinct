"""Compare device activities, never CPU dispatch rows, in fixed-weight traces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def category(name, label, native):
    if native:
        op = label.rsplit(':', 1)[-1]
        if '/gemm/' in label:
            return 'gemm'
        if op in ('forward', 'forward_fast', 'attention_forward'):
            return 'attention_forward'
        if op in ('backward_q', 'backward_kv', 'backward_delta',
                  'reduce_kv_groups', 'attention_backward', 'stats_copy'):
            return 'attention_backward_stats'
        if op in ('norm_rotary_forward', 'norm_rotary_backward'):
            return 'qk_norm_rope'
    else:
        lower = name.lower()
        # Inductor pointwise names containing _scaled_mm are conversion/epilogue
        # kernels, not matrix multiplies. Flex templates can fuse Q/K norm+RoPE.
        if 'flex_attention_backward' in lower:
            return 'attention_backward_stats'
        if 'flex_attention' in lower:
            return 'attention_forward'
        if 'index_mean_mul_neg' in lower or 'index_mul_neg' in lower:
            return 'qk_norm_rope'
        if lower.startswith('nvjet_') or ('cutlass' in lower and 'gemm' in lower):
            return 'gemm'
    return 'other'


def summarize(activities):
    by_category = defaultdict(lambda: {'calls': 0, 'ms': 0.})
    by_name = defaultdict(lambda: {'calls': 0, 'ms': 0.})
    intervals = []
    for event in activities:
        duration = event['end_ms'] - event['start_ms']
        for row in (by_category[event['category']], by_name[event['name']]):
            row['calls'] += 1
            row['ms'] += duration
        intervals.append((event['start_ms'], event['end_ms']))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    busy = sum(end-start for start, end in merged)
    span = merged[-1][1] - merged[0][0]
    return dict(categories=dict(by_category), activity_ms=sum(x['ms'] for x in by_category.values()),
                span_ms=span, busy_union_ms=busy, uncovered_span_ms=span-busy,
                kernels=sorted((dict(name=k, **v) for k, v in by_name.items()),
                               key=lambda x: x['ms'], reverse=True))


def native_trace(path):
    labels = json.loads(path.with_suffix('.labels.json').read_text(encoding='utf-8'))
    labels = {value: key for key, value in labels['labels'].items()}
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    events = []
    for row in records:
        if row['kind'] not in ('kernel', 'memcpy', 'memset'):
            continue
        label = labels.get(row['label'], '<unlabeled>')
        events.append(dict(start_ms=row['start']/1e6, end_ms=row['end']/1e6,
                           name=label if row['kind']=='kernel' else row['kind'],
                           category=category(row.get('name', ''), label, True)
                           if row['kind']=='kernel' else row['kind']))
    report = summarize(events)
    report['trace_summary'] = [r for r in records if r['kind']=='summary']
    report['source'] = str(path)
    return report


def torch_trace(path):
    records = json.loads(path.read_text(encoding='utf-8'))['traceEvents']
    events = []
    for row in records:
        kind = row.get('cat')
        if kind not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        events.append(dict(start_ms=row['ts']/1000, end_ms=(row['ts']+row['dur'])/1000,
                           name=row['name'], category=category(row['name'], '', False)
                           if kind=='kernel' else kind.removeprefix('gpu_')))
    report = summarize(events)
    report['source'] = str(path)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    root = args.directory
    native = json.loads((root/'native_latest.json').read_text(encoding='utf-8'))
    torch = json.loads((root/'torch_actual_recipe.json').read_text(encoding='utf-8'))
    if native['status']!='ok' or torch['status']!='complete':
        raise ValueError('Both fixed-weight probes must have completed')
    report = dict(scope='same input arrays; fixed weights; resident optimizer/gradients; no updates/loader',
                  caveats=['Weights use different RNG implementations.',
                           'Native caches 1024/1280 MiB; Torch caches 1792 MiB.',
                           'Torch Flex templates can fuse Q/K norm and RoPE; these are separate native kernels.',
                           'Uncovered device span includes dependencies/runtime/launch gaps; it is not CPU self time.'],
                  shapes=torch['shapes'], useful_targets=torch['useful_targets'],
                  native=[], torch=[])
    for bucket in range(2):
        descriptor = next(x for x in native['cupti_traces'] if x['bucket']==bucket)
        summary = native_trace(root/Path(descriptor['path']).name)
        samples = [w['total_ms'] for w in native['windows']
                   if not w['warmup'] and not w['cupti_profiled'] and w['window']%2==bucket]
        summary.update(wall_samples_ms=samples, wall_median_ms=statistics.median(samples))
        report['native'].append(summary)
        summary = torch_trace(root/Path(torch['trace_files'][bucket]).name)
        summary.update(wall_samples_ms=torch['bucket_ms'][bucket], wall_median_ms=torch['median_ms'][bucket])
        report['torch'].append(summary)
    (root/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for bucket in range(2):
        print('bucket', report['shapes'][bucket])
        for backend in ('torch', 'native'):
            entry=report[backend][bucket]
            print(backend, {k:v for k,v in entry.items() if k not in ('kernels','source','wall_samples_ms')})


if __name__=='__main__':
    main()
