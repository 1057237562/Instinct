from experiments.native_pipeline_timing import bucket_timings


def test_bucket_timing_uses_paired_targets_and_spans_not_mixed_window_wall_time():
    windows=[dict(seconds=10,microsteps=[
        dict(shape=[4,3072],useful_tokens=12000,forward_backward_stream_ms=600),
        dict(shape=[3,4096],useful_tokens=11000,forward_backward_stream_ms=800)]),
        dict(seconds=1,microsteps=[dict(shape=[4,3072],useful_tokens=10000,forward_backward_stream_ms=500)])]
    result=bucket_timings(windows)
    assert result['[4, 3072]']['microsteps']==2
    assert result['[4, 3072]']['forward_backward_useful_tokens_per_second']==20000
    assert result['[3, 4096]']['forward_backward_useful_tokens_per_second']==13750
    assert 'not end-to-end' in result['[3, 4096]']['scope']
    assert bucket_timings([dict(seconds=1)])=={}
