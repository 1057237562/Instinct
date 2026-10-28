"""Compute gaps must not silently include a backend-specific loss readback."""
from experiments.analyze_native_torch import compute_window,summarize


def test_compute_window_keeps_intervening_memset_but_reports_readback_separately():
    events=[dict(kind=kind,name=kind,category=kind,start_ms=start,end_ms=end)
            for kind,start,end in [('memcpy',0.,1.),('kernel',3.,7.),
                                   ('memset',8.,9.),('kernel',10.,14.),('memcpy',24.,25.)]]
    full=summarize(events);compute=compute_window(events)
    assert full['uncovered_span_ms']==14.
    assert compute['span_ms']==11. and compute['busy_union_ms']==9.
    assert compute['uncovered_span_ms']==2.
    assert compute['leading_span_ms']==3. and compute['trailing_span_ms']==11.


def test_overlapping_copy_is_clipped_at_compute_window_boundaries():
    events=[dict(kind=kind,name=kind,category=kind,start_ms=start,end_ms=end)
            for kind,start,end in [('memcpy',0.,4.),('kernel',2.,3.),('kernel',5.,8.)]]
    result=compute_window(events)
    assert result['span_ms']==6. and result['busy_union_ms']==5.
    assert result['uncovered_span_ms']==1.
