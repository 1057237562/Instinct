"""Device-side packed-document metadata; no quadratic attention mask."""
import triton
import triton.language as tl


@triton.jit
def prepare(IDS, MASK, LABELS, SEGMENTS, POSITIONS, TARGETS,
            SEQ: tl.constexpr, BLOCK: tl.constexpr, HAS_MASK: tl.constexpr,
            HAS_LABELS: tl.constexpr):
    batch = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    valid = i < SEQ
    doc = tl.load(IDS + batch * SEQ + i, valid, -1)
    previous = tl.load(IDS + batch * SEQ + i - 1, valid & (i > 0), -1)
    active = valid & (doc >= 0)
    before_active = (i > 0) & (previous >= 0)
    if HAS_MASK:
        active = active & (tl.load(MASK + batch * SEQ + i, valid, 0) != 0)
        before_active = before_active & (tl.load(MASK + batch * SEQ + i - 1, valid & (i > 0), 0) != 0)
    boundary = (i == 0) | (doc != previous) | ~before_active | ~active
    start = tl.associative_scan(tl.where(boundary, i, 0), 0, _maximum)
    tl.store(SEGMENTS + batch * SEQ + i, tl.where(active, start, -1), valid)
    tl.store(POSITIONS + batch * SEQ + i, tl.where(active, i - start, 0), valid)
    if HAS_LABELS:
        target = tl.load(LABELS + batch * SEQ + i, valid, -100)
        tl.store(TARGETS + batch * SEQ + i, tl.where(active & ~boundary, target, -100), valid)


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


def metadata(engine, sequence_ids, labels=None, mask=None):
    shape = sequence_ids.shape
    if len(shape) != 2 or sequence_ids.dtype not in ('i32', 'i64'):
        raise ValueError('sequence_ids must be integer [batch, sequence]')
    if any(x is not None and x.shape != shape for x in (labels, mask)):
        raise ValueError('packed labels/mask shape differs from sequence_ids')
    b, s = shape
    backend = engine.backend
    segments, positions = (backend.empty(shape, 'i64') for _ in range(2))
    targets = backend.empty(shape, 'i64') if labels is not None else segments
    backend.launch(prepare, (b,), (sequence_ids.data, sequence_ids.data if mask is None else mask.data,
                   sequence_ids.data if labels is None else labels.data, segments, positions, targets),
                   SEQ=s, BLOCK=triton.next_power_of_2(s), HAS_MASK=mask is not None, HAS_LABELS=labels is not None)
    return engine.wrap(segments), engine.wrap(positions), None if labels is None else engine.wrap(targets)
