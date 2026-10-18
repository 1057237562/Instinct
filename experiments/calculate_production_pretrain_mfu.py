"""Reconstruct useful FLOPs from cached rows, without rerunning training.

Uses the production sampler, original synchronized wall windows and the same
6*N_active + causal-attention estimate as earlier MFU experiments. Checkpoint
recomputation, optimizer work, inactive experts and padding are excluded.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import trainer.compile_cache  # configure before torch
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
import datasets  # Windows DLL import order
from trainer.packing_transition import SequencePackingPlan


def calculate(report, cache, manifest, samples, peak_tflops):
    assert manifest['identity']['source'] == report['corpus_sha256']
    plan = SequencePackingPlan.__new__(SequencePackingPlan)
    plan.args = SimpleNamespace(**report['args'])
    plan._runtime_buckets = {}
    plan._large_phase_ends = {}
    dataset = SimpleNamespace(bucket_ranges=manifest['bucket_ranges'])
    batches = plan._bucket_batches(dataset, report['args']['batch_size'], epoch=0)
    counts = []
    for indices, observation in zip(batches, report['observations']):
        rows = [samples[index] for index in indices]
        assert len(rows) == observation['batch_size']
        assert all(row['block_length'] == observation['sequence_length'] for row in rows)
        useful = valid = pairs = 0
        for row in rows:
            lengths = row['segment_lengths']
            assert sum(lengths) == row['valid_tokens']
            # Match PretrainDataset: mask pad IDs and non-first segment starts.
            useful += sum(token != report['config']['pad_token_id']
                          for token in row['input_ids']) - len(lengths) + 1
            valid += row['valid_tokens']
            pairs += sum(length * (length + 1) // 2 for length in lengths)
        counts.append(dict(useful_tokens=useful, valid_tokens=valid, causal_pairs=pairs,
                           single_document_batch=all(len(row['segment_lengths'])==1 for row in rows),
                           batch_size=len(rows),sequence_length=observation['sequence_length']))
    assert len(counts) == report['warmup'] + report['steps']
    measured = counts[report['warmup']:]
    useful = sum(row['useful_tokens'] for row in measured)
    valid = sum(row['valid_tokens'] for row in measured)
    pairs = sum(row['causal_pairs'] for row in measured)
    window = report['profiler_windows'][0]
    # Profiler rates preserve exact integer totals via their CUDA window length.
    recorded = (window['profile/window_useful_tokens_per_second'] *
                window['profile/window_step_time_ms'] * report['steps'] / 1000)
    assert abs(recorded - useful) < 1e-5, (recorded, useful)
    config = report['config']
    total = 678_726_144  # verified model count in the original production log
    expert = (config['num_hidden_layers'] * config['num_experts'] * 3 *
              config['hidden_size'] * config['moe_intermediate_size'])
    active = total - expert + expert * config['num_experts_per_tok'] // config['num_experts']
    assert active == 187_992_576
    parameter_flops = 6 * active * useful
    attention_flops = (12 * config['num_hidden_layers'] *
                       config['num_attention_heads'] * config['head_dim'] * pairs)
    seconds = report['wall_seconds']
    denominator = peak_tflops * 1e12 * seconds
    shapes=[]
    observations=report['observations'][report['warmup']:]
    for batch_size, sequence_length in sorted({(c['batch_size'],c['sequence_length']) for c in measured}):
        subset=[(c,o) for c,o in zip(measured,observations)
                if c['batch_size']==batch_size and c['sequence_length']==sequence_length]
        row=dict(batch_size=batch_size,sequence_length=sequence_length,steps=len(subset),
                 single_document_steps=sum(c['single_document_batch'] for c,o in subset))
        if all('gpu_step_ms' in o for c,o in subset):
            gpu_ms=sum(o['gpu_step_ms'] for c,o in subset)
            flops=sum(6*active*c['useful_tokens']+12*config['num_hidden_layers']*
                      config['num_attention_heads']*config['head_dim']*c['causal_pairs'] for c,o in subset)
            row.update(gpu_ms_per_step=gpu_ms/len(subset),
                       mfu_gpu_step_estimate_pct=100*flops/(gpu_ms/1000*peak_tflops*1e12),
                       phase_mean_ms={name:sum(o['gpu_phase_ms'][name] for c,o in subset)/len(subset)
                                      for name in subset[0][1]['gpu_phase_ms']})
        shapes.append(row)
    return dict(measured_steps=report['steps'], warmup_steps=report['warmup'],
                all_steps_useful_tokens=sum(c['useful_tokens'] for c in counts),
                wall_seconds=seconds, total_parameters=total, active_parameters=active,
                useful_tokens=useful, valid_tokens=valid, causal_pairs=pairs,
                useful_tokens_per_second=useful / seconds,
                useful_parameter_flops=parameter_flops, useful_attention_flops=attention_flops,
                useful_total_flops=parameter_flops + attention_flops,
                useful_tflops_per_second=(parameter_flops + attention_flops) / seconds / 1e12,
                six_n_mfu_pct=100 * parameter_flops / denominator,
                mfu_estimate_pct=100 * (parameter_flops + attention_flops) / denominator,
                valid_context_mfu_estimate_pct=100 * (6 * active * valid + attention_flops) / denominator,
                reconstructed_tokens_match_profiler=True, packed_cache=str(cache.resolve()),
                by_shape=shapes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--peak-tflops', type=float, default=87.9)
    parser.add_argument('--reports', nargs='+', type=Path,
                        help='Calculate selected production reports instead of the original pair')
    parser.add_argument('--cache', type=Path,
                        help='Exact packed chunk cache used by these reports; required for a different packing budget')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'experiments/results/top2_production_mfu_20261012.json')
    options = parser.parse_args()
    if options.peak_tflops <= 0:
        parser.error('peak must be positive')
    cache = options.cache or ROOT / '.cache/huggingface/datasets/instinct-packed/e37b9a8b5d8d4689ec963f559772ae0e5648e4608079bf2f8b4bb12fc34684a5'
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    samples = datasets.concatenate_datasets([
        datasets.Dataset.from_file(str(cache / entry['name'])) for entry in manifest['files']])
    output = dict(gpu='NVIDIA GeForce RTX 5070 Ti', reference_peak_tflops=options.peak_tflops,
                  reference_precision='BF16/FP32 accumulate dense; same reference as PLAN.md',
                  formula='(6*N_active*useful_tokens + 12*layers*q_heads*head_dim*causal_pairs) / wall_seconds / peak_flops',
                  note='Estimated useful model FLOPs; excludes warmup, initial cache build/compile, checkpoint recomputation, optimizer FLOPs and padding. Useful tokens follow the original profiler label mask; valid_context alternative also counts segment-start context tokens.',
                  results={})
    paths = ([(path.stem, path) for path in options.reports] if options.reports else
             [(name, ROOT / 'experiments/results' / filename) for name, filename in
              [('cold_prefetch', 'top2_production_20261012.json'),
               ('cached', 'top2_production_cached_20261012.json')]])
    for name, path in paths:
        report = json.loads(path.read_text(encoding='utf-8'))
        output['results'][name] = calculate(report, cache, manifest, samples, options.peak_tflops)
        output['results'][name]['raw_report'] = str(path.resolve())
    options.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
