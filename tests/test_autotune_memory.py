"""CUDA telemetry must distinguish live tensors, cached blocks and full peaks."""
import json
from types import SimpleNamespace

from scripts.autotune_memory import CudaMemoryTrace


def fake_cuda():
    state = dict(allocated=2, reserved=8, peak_allocated=6, peak_reserved=9)
    gib = 2**30
    cuda = SimpleNamespace(
        memory_stats=lambda _: {'inactive_split_bytes.all.current': gib // 2,
                                 'num_alloc_retries': 3, 'num_ooms': 1},
        mem_get_info=lambda _: (gib, 16 * gib),
        memory_allocated=lambda _: state['allocated'] * gib,
        memory_reserved=lambda _: state['reserved'] * gib,
        max_memory_allocated=lambda _: state['peak_allocated'] * gib,
        max_memory_reserved=lambda _: state['peak_reserved'] * gib,
        get_device_name=lambda _: 'test GPU',
        empty_cache=lambda: state.update(reserved=3),
    )
    return state, cuda


def test_cache_release_reports_actual_difference_and_retains_live_tensors(tmp_path, monkeypatch):
    from scripts import autotune_memory
    monkeypatch.setattr(autotune_memory.gc, 'collect', lambda: None)
    state, cuda = fake_cuda()
    trace = CudaMemoryTrace(tmp_path, 'cuda:0', torch_module=SimpleNamespace(cuda=cuda),
                            driver_query=lambda _: {'budget': 15 * 2**30})
    after = trace.release_cache('validation')
    assert after['released_reserved_gib'] == 5
    assert after['allocated_gib'] == state['allocated'] == 2
    assert trace.samples[0]['unallocated_reserved_gib'] == 6
    assert after['inactive_split_gib'] == .5
    assert after['allocation_retries'] == 3 and after['oom_count'] == 1
    assert json.loads(trace.path.read_text())['samples'][-1]['released_reserved_gib'] == 5


def test_full_trial_peaks_survive_timed_window_reset(tmp_path):
    state, cuda = fake_cuda()
    trace = CudaMemoryTrace(tmp_path, 'cuda:0', torch_module=SimpleNamespace(cuda=cuda),
                            driver_query=lambda _: None)
    trace.capture('warmup_end')
    state.update(peak_allocated=4, peak_reserved=5)
    trace.capture('timing_end')
    saved = json.loads(trace.path.read_text())
    assert saved['trial_peak_allocated_gib'] == 6
    assert saved['trial_peak_reserved_gib'] == 9
    assert saved['samples'][-1]['peak_allocated_gib'] == 4
