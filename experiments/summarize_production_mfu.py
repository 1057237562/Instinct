"""Collect the final, matching-workload production optimization evidence."""
import json
from pathlib import Path


root = Path('experiments/results/mfu_20261012')
def read(name):
    return json.loads((root / (name + '.json')).read_text())


a1, a2, optimized = (read(n) for n in ('final_a1', 'final_a2', 'final_swiglu_b'))
reference, candidate = (read(n) for n in ('pipeline_reference', 'pipeline_candidate'))
parity = read('paired_deterministic_100')
average = (a1['wall_ms'] + a2['wall_ms']) / 2
summary = {
    'target_mfu_pct': 20,
    'gpu': optimized['gpu'],
    'checkpoint': optimized['args']['checkpoint'],
    'reference_peak_tflops': 87.9,
    'compute': {
        'baseline_wall_ms': [a1['wall_ms'], a2['wall_ms']],
        'optimized_wall_ms': optimized['wall_ms'],
        'optimized_mfu_pct': optimized['mfu_estimate_pct'],
        'optimized_window_mfu_pct': [optimized['mfu_estimate_pct'] * optimized['wall_ms'] / t
                                     for t in optimized['window_wall_ms']],
        'throughput_gain_pct': 100 * (average / optimized['wall_ms'] - 1),
        'phase_mean_ms': optimized['phase_mean_ms'],
        'peak_allocated_gib': optimized['peak_allocated_gib'],
    },
    'production_loop': {
        'reference_mfu_pct': reference['mfu_estimate_pct'],
        'optimized_mfu_pct': candidate['mfu_estimate_pct'],
        'reference_wall_ms': reference['wall_ms'],
        'optimized_wall_ms': candidate['wall_ms'],
        'optimized_useful_tokens_per_second': candidate['useful_tokens_per_second'],
        'throughput_gain_pct': 100 * (reference['wall_ms'] / candidate['wall_ms'] - 1),
        'same_replay': reference['replay_sha256'] == candidate['replay_sha256'],
        'same_work': reference['metrics'] == candidate['metrics'],
        'same_corpus': reference['corpus_sha256'] == candidate['corpus_sha256'],
        'steps': candidate['steps'],
        'scope': candidate['scope'],
    },
    'correctness': {
        'deterministic': parity['deterministic'],
        'steps_per_trajectory': parity['steps'],
        'phases_bitwise_equal': {k: v['bitwise_equal'] for k, v in parity['phases'].items()},
        'ordinary_mode_note': 'Reference repeats have nondeterministic gradients; exact full-model parity uses deterministic controls.',
    },
    'scope_notes': [
        'Same batch, model, activation/parameter precision, checkpoint and MFU definition.',
        'Compute baseline uses two 200-step reference runs; final candidate uses four 50-step windows.',
        'Production-loop comparison uses identical audited-corpus replay and normal LR/logging/H2D.',
        'Warm-cache measurements exclude preprocessing, cold compilation and checkpoint saves; not a full-corpus training run.',
    ],
}
summary['target_achieved'] = (
    min(summary['compute']['optimized_window_mfu_pct']) >= 20
    and candidate['mfu_estimate_pct'] >= 20
    and all(summary['correctness']['phases_bitwise_equal'].values())
    and all(summary['production_loop'][k] for k in ('same_replay', 'same_work', 'same_corpus'))
    and summary['checkpoint'] == 1
)
(root / 'production_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
