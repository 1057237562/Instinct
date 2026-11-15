"""Small measured latency model for contiguous ragged-attention partitions.

Only nine dense descriptor anchors are timed. The CPU fit is a nonnegative
four-term model; it screens partitions, not kernel implementations. Actual
ragged calls must still pass paired performance and numerical validation.
"""
from functools import lru_cache
import itertools
import math
import numpy as np


def features(docs,sequence):
    s=sequence/1024.
    return np.asarray((1.,docs,docs*s,docs*s*s),np.float64)


def fit(samples):
    if len(samples)<4:return None
    x=np.stack([features(r['docs'],r['sequence']) for r in samples])
    y=np.asarray([r['milliseconds'] for r in samples],np.float64)
    if not np.isfinite(y).all() or np.any(y<=0):return None
    # Four columns make active-set enumeration cheap and deterministic. This
    # enumerates 15 CPU least-squares systems, never GPU kernel candidates.
    best=None
    for count in range(1,5):
        for active in itertools.combinations(range(4),count):
            value=np.linalg.lstsq(x[:,active]/y[:,None],np.ones(len(y)),rcond=None)[0]
            if np.any(value<0):continue
            coefficients=np.zeros(4);coefficients[list(active)]=value
            error=float(np.mean(((x@coefficients-y)/y)**2))
            if best is None or error<best[0]:best=error,coefficients
    if best is None or math.sqrt(best[0])>.3:return None
    return tuple(float(v) for v in best[1])


def predict(coefficients,docs,sequence):
    s=sequence/1024.
    a,b,c,d=coefficients
    return a+b*docs+c*docs*s+d*docs*s*s


def anchors(geometries):
    """Cover short/medium/long and single/multiple documents, at most nine."""
    choices={(g[0],g[1]) for g in geometries}
    wanted=((1,512),(1,1536),(1,3072),(1,4096),(4,512),(4,1536),
            (4,3072),(3,4096),(2,2048))
    return set(wanted)&choices


@lru_cache(maxsize=512)
def partition(lengths,size,quantum,maximum,total,coefficients,baseline):
    """Keep order and use only geometries in the bounded prewarmed catalog."""
    def cost(start,count):
        seq=min(maximum,((max(lengths[start:start+count])+quantum-1)//quantum)*quantum)
        return predict(coefficients,count,seq) if count*seq<=total else float('inf')
    n=len(lengths);dp=[0.]+[float('inf')]*n;prev=[0]*(n+1);calls=[0]*(n+1)
    for end in range(1,n+1):
        for start in range(max(0,end-size),end):
            candidate=dp[start]+cost(start,end-start)
            if (candidate,calls[start]+1)<(dp[end],calls[end]):
                dp[end]=candidate;prev[end]=start;calls[end]=calls[start]+1
    original=sum(cost(s,c) for s,c in baseline)
    if not math.isfinite(dp[n]) or dp[n]>=original*.99:return baseline
    result=[];end=n
    while end:
        start=prev[end];result.append((start,end-start));end=start
    return tuple(reversed(result))
