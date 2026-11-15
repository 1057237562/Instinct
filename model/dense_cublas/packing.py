"""Device-side packed-document metadata; no quadratic attention mask."""
import triton
import triton.language as tl


def document_max_length(sequence_ids):
    """Largest contiguous valid document; physical packed rows may be longer."""
    import numpy as np
    ids=np.asarray(sequence_ids,dtype=np.int64)
    if ids.ndim!=2:raise ValueError('host sequence IDs must be a matrix')
    maximum=0
    for row in ids:
        edges=np.r_[0,np.flatnonzero(row[1:]!=row[:-1])+1,len(row)]
        maximum=max(maximum,max((int(end-start) for start,end in zip(edges[:-1],edges[1:]) if row[start]>=0),default=0))
    return maximum


def compact_packed_batch(batch,pad,multiple=512):
    """Remove tail padding within one batch; preserve ordered documents/targets.

    This operates on an already packed microbatch, not the dataset corpus.
    No document is split, and no sample moves across an optimizer boundary.
    """
    import numpy as np
    if multiple<1:raise ValueError('packing alignment must be positive')
    ids,labels,segments=map(np.asarray,batch)
    if ids.ndim!=2 or labels.shape!=ids.shape or segments.shape!=ids.shape:
        raise ValueError('compact packing requires matching token/label/segment matrices')
    active=segments>=0;counts=active.sum(1)
    if not np.array_equal(active,np.arange(ids.shape[1])[None,:]<counts[:,None]):
        raise ValueError('compact packing requires tail padding, not interior holes')
    total=int(counts.sum());size=(total+multiple-1)//multiple*multiple
    if not total or size>=ids.size:return batch
    out_ids=np.full((1,size),pad,dtype=ids.dtype);out_labels=np.full((1,size),-100,dtype=labels.dtype)
    out_segments=np.full((1,size),-1,dtype=segments.dtype);at=next_doc=0
    for row,count in enumerate(counts):
        count=int(count)
        if not count:continue
        source=segments[row,:count];starts=np.r_[True,source[1:]!=source[:-1]]
        if np.any(labels[row,:count][starts]!=-100):raise ValueError('packed document starts must have ignored labels')
        out_ids[0,at:at+count]=ids[row,:count];out_labels[0,at:at+count]=labels[row,:count]
        out_segments[0,at:at+count]=np.cumsum(starts)-1+next_doc
        next_doc+=int(starts.sum());at+=count
    if np.count_nonzero(labels[:,1:]!=-100)!=np.count_nonzero(out_labels[:,1:]!=-100):
        raise ValueError('compaction changed the number of useful targets')
    return out_ids,out_labels,out_segments


@triton.jit
def prepare_offsets(OFFSETS,LENGTHS,LABELS,SEGMENTS,POSITIONS,TARGETS,
                    N:tl.constexpr,SEQ:tl.constexpr,DOCS:tl.constexpr,Q_WIDTH:tl.constexpr,
                    HAS_LABELS:tl.constexpr,BLOCK:tl.constexpr=256):
    token=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=token<N
    start=tl.full((BLOCK,),-1,tl.int64);position=tl.full((BLOCK,),0,tl.int64)
    for doc in range(DOCS):
        begin=tl.load(OFFSETS+doc)//Q_WIDTH;length=tl.load(LENGTHS+doc)
        inside=(token>=begin)&(token<begin+length)
        start=tl.where(inside,begin%SEQ,start);position=tl.where(inside,token-begin,position)
    tl.store(SEGMENTS+token,start,valid);tl.store(POSITIONS+token,position,valid)
    if HAS_LABELS:
        label=tl.load(LABELS+token,valid,-100)
        tl.store(TARGETS+token,tl.where((start>=0)&(position>0),label,-100),valid)


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


def metadata(engine, sequence_ids, labels=None, mask=None, native_context=None):
    shape = sequence_ids.shape
    if len(shape) != 2 or sequence_ids.dtype not in ('i32', 'i64'):
        raise ValueError('sequence_ids must be integer [batch, sequence]')
    if any(x is not None and x.shape != shape for x in (labels, mask)):
        raise ValueError('packed labels/mask shape differs from sequence_ids')
    b, s = shape
    backend = engine.backend
    segments, positions = (backend.empty(shape, 'i64') for _ in range(2))
    targets = backend.empty(shape, 'i64') if labels is not None else segments
    if native_context is not None and mask is None:
        docs,_,heads,_,dim,_,offsets=native_context[:7]
        backend.launch(prepare_offsets,(triton.cdiv(b*s,256),),
            (offsets[0],offsets[3],sequence_ids.data if labels is None else labels.data,segments,positions,targets),
            N=b*s,SEQ=s,DOCS=docs,Q_WIDTH=heads*dim,HAS_LABELS=labels is not None)
        return engine.wrap(segments),engine.wrap(positions),None if labels is None else engine.wrap(targets)
    backend.launch(prepare, (b,), (sequence_ids.data, sequence_ids.data if mask is None else mask.data,
                   sequence_ids.data if labels is None else labels.data, segments, positions, targets),
                   SEQ=s, BLOCK=triton.next_power_of_2(s), HAS_MASK=mask is not None, HAS_LABELS=labels is not None)
    return engine.wrap(segments), engine.wrap(positions), None if labels is None else engine.wrap(targets)
