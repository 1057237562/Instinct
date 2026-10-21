"""Read existing packed Arrow shards only: no torch, tokenization or training.

Row-wise HuggingFace access reproduces PretrainDataset.samples[index], including
Arrow-to-Python materialization. Batched access and native Arrow selection are
diagnostic comparisons. OS page cache is not flushed; these are warm-cache runs.
"""
import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

import datasets
import pyarrow as pa


def summarize(values):
    ordered = sorted(values)
    return dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                p95_ms=ordered[min(len(ordered) - 1, int(len(ordered) * .95))],
                max_ms=max(values), steps=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path,
                        default=Path('.cache/huggingface/datasets/instinct-packed'))
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--native-take-steps', type=int, default=20,
                        help='Bound expensive whole-table random take comparison')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[4, 8, 11])
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/results/arrow_reads.json'))
    args = parser.parse_args()
    if (args.steps < 1 or args.warmup < 0 or args.native_take_steps < 1
            or min(args.batch_sizes) < 1):
        parser.error('steps and batch sizes must be positive; warmup nonnegative')
    cases = []
    manifests = ([args.cache_root / 'manifest.json']
                 if (args.cache_root / 'manifest.json').exists()
                 else sorted(args.cache_root.glob('*/manifest.json')))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        paths = [manifest_path.parent / entry['name'] for entry in manifest['files']]
        started = time.perf_counter()
        shards = [datasets.Dataset.from_file(str(path)) for path in paths]
        sample = datasets.concatenate_datasets(shards)
        open_ms = (time.perf_counter() - started) * 1000
        table = sample.data.table
        cache = dict(path=str(manifest_path.parent.resolve()), rows=len(sample),
                     bytes=sum(path.stat().st_size for path in paths),
                     open_ms=open_ms, columns=sample.column_names)
        for bucket in manifest['bucket_ranges']:
            start, end = bucket['start'], bucket['end']
            for size in args.batch_sizes:
                if end - start < size:
                    continue
                for order in ('contiguous', 'random'):
                    rng = random.Random(73)
                    batches = []
                    for step in range(args.steps + args.warmup):
                        if order == 'contiguous':
                            offset = start + (step * size) % (end - start - size + 1)
                            indices = list(range(offset, offset + size))
                        else:
                            indices = rng.sample(range(start, end), size)
                        batches.append(indices)
                    indices = batches[0]
                    rows = [sample[index] for index in indices]
                    expected = {key: [row[key] for row in rows]
                                for key in sample.column_names}
                    assert sample[indices] == expected
                    assert table.take(pa.array(indices, type=pa.int64())).to_pydict() == expected
                    # All modes use the same row indices; index generation is
                    # outside timing. Release each result inside the timed span.
                    for mode in ('hf_rows', 'hf_batch', 'arrow_select'):
                        durations = []
                        count = (min(args.steps, args.native_take_steps)
                                 if mode == 'arrow_select' and order == 'random'
                                 else args.steps)
                        for step, indices in enumerate(batches[:args.warmup + count]):
                            begin = time.perf_counter_ns()
                            if mode == 'hf_rows':
                                result = [sample[index] for index in indices]
                            elif mode == 'hf_batch':
                                result = sample[indices]
                            elif order == 'contiguous':
                                result = table.slice(indices[0], size)
                            else:
                                result = table.take(pa.array(indices, type=pa.int64()))
                            del result
                            elapsed = (time.perf_counter_ns() - begin) / 1e6
                            if step >= args.warmup:
                                durations.append(elapsed)
                        case = dict(cache=cache, sequence_length=bucket['max_length'],
                                    batch_size=size, order=order, mode=mode,
                                    first_batch_equal_across_modes=True,
                                    **summarize(durations))
                        cases.append(case)
                        print(json.dumps({key: value for key, value in case.items()
                                          if key != 'cache'}, ensure_ascii=False), flush=True)
        del sample, table, shards
    if not cases:
        raise RuntimeError('No existing packed Arrow cache manifests found')
    report = dict(scope='existing mmap packed Arrow only; warm OS page cache; no '
                  'tensor conversion, collate, pinning, tokenizer, packing or CUDA',
                  platform=platform.platform(), datasets=datasets.__version__,
                  pyarrow=pa.__version__, warmup=args.warmup, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for mode in ('hf_rows', 'hf_batch', 'arrow_select'):
        selected = [case for case in cases if case['mode'] == mode]
        print(json.dumps(dict(mode=mode, case_count=len(selected),
                              mean_ms_range=[min(c['mean_ms'] for c in selected),
                                             max(c['mean_ms'] for c in selected)],
                              worst_p95_ms=max(c['p95_ms'] for c in selected)),
                         indent=2), flush=True)


if __name__ == '__main__':
    main()
