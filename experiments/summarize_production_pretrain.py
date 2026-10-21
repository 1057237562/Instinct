"""Compare cold-prefetch and existing-cache production measurements."""
import json
from pathlib import Path
import statistics


def main():
    root = Path('experiments/results')
    paths = [root / 'top2_production_20261012.json',
             root / 'top2_production_cached_20261012.json']
    cold, cached = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    assert cold['config'] == cached['config']
    assert cold['corpus_sha256'] == cached['corpus_sha256']
    shape = lambda row: (row['batch_size'], row['sequence_length'])
    assert [shape(row) for row in cold['observations'][:len(cached['observations'])]] == [
        shape(row) for row in cached['observations']]
    assert cold['steps'] + cold['warmup'] + cached['steps'] + cached['warmup'] == 300
    # Correct old labels without altering any raw timing values.
    for path, report in zip(paths, [cold, cached]):
        old_key = 'start_to_100_warm_steps_seconds'
        if old_key in report:
            report['start_to_measured_window_end_seconds'] = report.pop(old_key)
        report['scope'] = ('real production CLI, chunked Arrow DataLoader, automatic '
                           'packing, prefetch, LR/clip/optimizer/zero_grad/log/pause; '
                           f"{report['steps']} measured steps after {report['warmup']} "
                           'warmup steps; epoch-final checkpoint omitted')
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    measured = cold['observations'][cold['warmup']:]
    long_steps = [row for row in measured if row['cpu_loop_body_ms'] >= 2000]
    normal = [row['cpu_loop_body_ms'] for row in measured if row['cpu_loop_body_ms'] < 2000]
    gpu_ms = cached['profiler_windows'][0]['profile/step_time_ms']
    summary = dict(
        gpu=cached['gpu'], total_training_steps=300,
        routing=dict(top_k=2, norm_topk_prob=True),
        production_config_equal=True, sampled_prefix_batch_shapes_equal=True,
        steady_100_steps_seconds=cached['wall_seconds'] / cached['steps'] * 100,
        steady_100_is_extrapolated_from_70=True,
        cached_measured_steps=cached['steps'], cached_wall_seconds=cached['wall_seconds'],
        cold_prefetch_measured_steps=cold['steps'], cold_prefetch_wall_seconds=cold['wall_seconds'],
        cold_prefetch_average_100_steps_seconds=cold['wall_seconds'] / cold['steps'] * 100,
        cold_prefetch_first_100_cpu_loop_span_seconds=sum(
            row['loader_next_ms'] + row['cpu_loop_body_ms'] for row in measured[:100]) / 1000,
        cold_prefetch_second_100_cpu_loop_span_seconds=sum(
            row['loader_next_ms'] + row['cpu_loop_body_ms'] for row in measured[100:]) / 1000,
        cpu_loop_span_note='No extra GPU synchronize at the intermediate 100-step boundary',
        cold_startup_to_first_batch_seconds=cold['startup_to_first_batch_seconds'],
        cold_startup_note='Excludes benchmark file hashing and Python import time',
        long_steps=long_steps, normal_cpu_body_median_ms=statistics.median(normal),
        steady_phases_ms=cached['phase_ms_per_step'],
        steady_phases_pct={name: value / gpu_ms * 100
                           for name, value in cached['phase_ms_per_step'].items()},
        steady_loader_next_ms=cached['loader_next_ms'],
        interpretation='Steady-state time is dominated by backward then forward and '
        'optimizer. Cold-prefetch long stalls coincide with background inline cache '
        'construction; the same production settings with existing caches have no '
        'multi-second stalls. This implicates inline preprocessing contention but '
        'does not isolate a specific GC, Arrow or GIL operation.',
        cleanup=dict(experiment_processes_exited=True, saved_weights=False,
                     abandoned_cache_bytes_removed=2000852058,
                     repository_build_directories_remaining=0,
                     reusable_packed_and_compile_caches_retained=True),
        raw_results=[str(path.resolve()) for path in paths])
    output = root / 'top2_production_summary_20261012.json'
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
