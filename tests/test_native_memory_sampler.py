"""Memory diagnostics must keep phase peaks and never disrupt a training run."""
from experiments.native_device_memory import DeviceMemorySampler


def test_samples_keep_each_phase_peak_instead_of_the_final_usage():
    sampler=DeviceMemorySampler(interval=.0001);sampler._start=0
    samples=iter([('setup',100),('measurement',700),('measurement',300)])
    def read():
        phase,used=next(samples)
        # The sampler attributes each read to the phase at its start.
        if used==100:sampler.phase='measurement'
        if used==300:sampler._stop.set()
        return dict(total=1000,used=used,free=1000-used)
    sampler._read=read;sampler._run()
    assert sampler.peaks['setup']['peak_used_bytes']==100
    assert sampler.peaks['measurement']['samples']==2
    assert sampler.peaks['measurement']['peak_used_bytes']==700
    assert sampler.peaks['measurement']['free_bytes_at_peak']==300


def test_sampling_failure_is_reported_without_raising_into_training():
    sampler=DeviceMemorySampler();sampler._start=0
    def read():raise RuntimeError('device unavailable')
    sampler._read=read;sampler._run()
    assert 'device unavailable' in sampler.report()['error']
    assert not sampler.peaks
