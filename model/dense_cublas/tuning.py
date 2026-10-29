"""Bounded empirical selection after cheap geometry/resource screening."""
import statistics
import math


def race(candidates,measure,*,baseline=0,minimum_gain=.01,warmup_ms=0.,drift_limit=None,retries=0):
    """Coarse timing of <=4 candidates, then confirm only baseline and leader.

    The baseline wins differences below one percent. Timings use fixed inputs;
    callers are responsible for prewarming and validating output equivalence.
    """
    if not candidates or len(candidates)>4:raise ValueError('expected one to four shortlisted candidates')
    if baseline not in {c['index'] for c in candidates}:raise ValueError('baseline missing')
    if not 0<=retries<=2 or not math.isfinite(warmup_ms) or warmup_ms<0:raise ValueError('invalid timing budget')
    if len(candidates)==1:
        candidates[0]['selection_reason']='only supported candidate'
        return candidates[0]
    control=next(c for c in candidates if c['index']==baseline)
    for attempt in range(retries+1):
        for candidate in candidates:
            for key in ('coarse_ms','median_ms','samples_ms','rejected'):candidate.pop(key,None)
        if warmup_ms:
            estimate=measure(control,2)
            if math.isfinite(estimate) and estimate>0:
                count=max(2,min(4096,math.ceil(warmup_ms/max(estimate,.001))))
                elapsed=measure(control,count)*count
                control.setdefault('warmup_runs',[]).append(dict(repeats=count,elapsed_ms=elapsed))
        for candidate in candidates:candidate['coarse_ms']=measure(candidate,2)
        leader=min(candidates,key=lambda c:c['coarse_ms'])
        if not math.isfinite(leader['coarse_ms']):raise RuntimeError('all shortlisted candidates failed')
        finalists=[control]+([leader] if leader is not control else [])
        samples={c['index']:[] for c in finalists}
        for repeat in range(3):
            for candidate in finalists[::(-1 if repeat%2 else 1)]:
                samples[candidate['index']].append(measure(candidate,3))
        for candidate in candidates:
            if candidate in finalists:
                candidate['median_ms']=statistics.median(samples[candidate['index']])
                candidate['samples_ms']=samples[candidate['index']]
            else:candidate['rejected']='slower in coarse screen'
        unstable=(drift_limit is not None and any(
            all(math.isfinite(v) and v>0 for v in values) and max(values)/min(values)-1>drift_limit
            for values in samples.values()))
        if unstable:
            control.setdefault('unstable_attempts',[]).append(dict(attempt=attempt,samples_ms=samples))
            if attempt<retries:continue
            control['selection_warning']='timing remained unstable; kept heuristic baseline'
            return control
        if leader is not control and leader['median_ms']<control['median_ms']*(1.-minimum_gain):return leader
        if leader is not control:leader['rejected']='improvement below confirmation margin'
        return control


def gemm_shortlist(candidates,limit=4):
    """Keep the leading heuristics plus occupancy/workspace alternatives.

    waves is an occupancy proxy, not predicted milliseconds. Retain the first
    two recommendations, then a split-workspace alternative (when distinct)
    and a lower-wave tile family. Never rank a kernel by FLOPs alone.
    """
    if len(candidates)<=limit:return list(candidates)
    result=list(candidates[:2]);remaining=list(candidates[2:])
    workspace=max(remaining,key=lambda c:c['workspace_bytes'])
    if workspace['workspace_bytes']>max(c['workspace_bytes'] for c in result):
        result.append(workspace);remaining.remove(workspace)
    while len(result)<limit and remaining:
        choice=min(remaining,key=lambda c:(c['waves'] if c['waves']>=1 else 1/max(c['waves'],1e-6),c['index']))
        result.append(choice);remaining.remove(choice)
    for candidate in candidates:
        if candidate not in result:candidate['rejected']='heuristic resource shortlist'
    return result


def ffn_shortlist(rows,cols,*,sm_count=70):
    """Bandwidth-bound FFN: square transpose tiles, bounded register footprint.

    A 64x64 tile already coalesces both orientations. More warps add barriers;
    larger/rectangular tiles increase the five-output live register footprint.
    Keep the baseline plus occupancy, reduction and launch-order alternatives.
    """
    count=rows*cols
    baseline=dict(index=0,row_tile=64,col_tile=64,warps=4,statistics_block=1024,col_major=False)
    candidates=[baseline]
    if count>=2**20 and ((rows+63)//64)*((cols+63)//64)>=4*sm_count:
        # Six live FP32 vectors per element are a conservative backward
        # estimate. One/two-warp 64x64 variants exceed the register budget;
        # the measured spill-heavy variants must not return to the shortlist.
        for warps in (1,2,8):
            registers=math.ceil(64*64/(32*warps))*6
            if registers<=224:candidates.append(baseline|dict(index=len(candidates),warps=warps))
        candidates += [baseline|dict(index=len(candidates),statistics_block=4096),
                       baseline|dict(index=len(candidates)+1,col_major=True)]
    return candidates
