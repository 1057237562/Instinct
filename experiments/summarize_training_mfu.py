"""Assemble local MFU benchmark evidence into a compact JSON and optional PNG."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    root = Path(args.directory)
    cases = {}
    for path in sorted(root.glob('*.json')):
        if path.name.endswith('.trace.json') or path.name == 'summary.json':
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        if 'status' in data and 'args' in data:
            keys = ['status', 'args', 'wall_ms', 'window_wall_ms', 'tokens_per_second',
                    'phase_mean_ms', 'peak_allocated_gib', 'peak_reserved_gib',
                    'mfu_estimate_pct', 'error', 'useful_flops_per_step_estimate',
                    'active_params_estimate', 'source_sha256']
            cases[path.stem] = {key: data[key] for key in keys if key in data}
    correctness = json.loads((root / 'correctness_v2.json').read_text(encoding='utf-8'))
    result = {'scope': '5070 Ti synthetic compute; not 5090 training MFU',
              'cases': cases, 'correctness': correctness,
              'target_40_percent_achieved': False}
    a1, b, a2 = (cases.get(name) for name in ['workspace64_a1', 'workspace256_b', 'workspace64_a2'])
    if all(row and row['status'] == 'ok' for row in (a1, b, a2)):
        reference = (a1['wall_ms'] + a2['wall_ms']) / 2
        result['workspace_aba'] = {'reference_ms': reference, 'candidate_ms': b['wall_ms'],
                                    'throughput_gain_pct': 100 * (reference / b['wall_ms'] - 1),
                                    'baseline_drift_pct': 100 * (a2['wall_ms'] / a1['wall_ms'] - 1)}
    (root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.plot:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = [('baseline_sdpa', 'SDPA eager', '#64748b'), ('flex', 'Flex eager', '#64748b'),
             ('compiled_fp8', 'Current compiled', '#2563eb'),
             ('compiled_gram', 'Gram FP16: gate failed', '#f59e0b'),
             ('compiled_ns_bf16', 'Compiled NS: gate failed', '#f59e0b'),
             ('no_checkpoint', 'No checkpoint: slower', '#f59e0b'),
             ('cast_before_stack', 'Cast first: small/noisy', '#64748b'),
             ('larger_batch', '2x batch: changed training', '#8b5cf6')]
    order = [row for row in order if row[0] in cases and cases[row[0]]['status'] == 'ok']
    fig, (ax, phases) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={'width_ratios': [2.4, 1]})
    times = [cases[key]['wall_ms'] for key, _, _ in order]
    y = list(range(len(order)))
    ax.barh(y, times, color=[c for _, _, c in order])
    ax.set_yticks(y, [label for _, label, _ in order])
    ax.invert_yaxis()
    ax.set_xlabel('Wall milliseconds per full training step (lower is faster)')
    ax.set_title('32-layer / top-1 MoE / FP32 parameters')
    for pos, value, (key, _, _) in zip(y, times, order):
        windows = cases[key]['window_wall_ms']
        ax.errorbar(value, pos, xerr=[[value-min(windows)], [max(windows)-value]], color='black', capsize=3)
        ax.text(value + 18, pos, f'{value:.0f}', va='center', fontsize=9)
    ax.set_xlim(0, max(times) * 1.17)
    phases.bar(['Forward', 'Backward', 'Optimizer'],
               [cases['compiled_fp8']['phase_mean_ms'][n] for n in ['forward', 'backward', 'optimizer']],
               color=['#22c55e', '#3b82f6', '#ef4444'])
    phases.set_ylabel('Mean CUDA phase ms')
    phases.set_title('Current baseline: optimizer ~54%')
    phases.tick_params(axis='x', rotation=25)
    fig.text(0.02, 0.025, '5070 Ti, warmup=20, 3x5 measured steps. Error bars: window range. Synthetic compute excludes loader/logs/checkpoint I/O.', fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(root / 'comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
