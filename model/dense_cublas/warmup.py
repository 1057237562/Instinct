"""Prewarm bounded compact shapes without touching optimizer state or data."""
import time
import numpy as np


def warm(model,optimizer,capacity=12288,multiple=512):
    if multiple!=512 or capacity!=12288:raise ValueError('compact warmup currently covers the production 12k/512 shortlist')
    e=model.e;b=e.backend
    before=(optimizer.updates,optimizer.accumulated,len(optimizer.present))
    if optimizer.accumulated:raise ValueError('shape prewarm requires an optimizer boundary')
    state=e.seed,e.counter,getattr(e,'consume_checkpoint_cache',False)
    sizes=[capacity,*range(9728,capacity,multiple)];cases=[]
    try:
        e.consume_checkpoint_cache=True
        for size in sizes:
            mark=b.mark();begin=time.perf_counter()
            if size==capacity:
                ids=np.zeros((3,4096),np.int64);segments=np.zeros_like(ids)
            else:
                ids=np.zeros((1,size),np.int64);lengths=[size//3+(i<size%3) for i in range(3)]
                segments=np.repeat(np.arange(3,dtype=np.int64),lengths)[None]
            output=model(ids,ids,sequence_ids=segments,return_logits=False)
            loss=e.item(output.loss)
            if not np.isfinite(loss):raise FloatingPointError('nonfinite synthetic prewarm loss')
            # Dropping leaf results is deliberate: no optimizer accumulation,
            # momentum update, training counter or source cursor is advanced.
            e.vjp(output.loss,leaf_callback=lambda parameter,gradient:True)
            b.synchronize();b.release_since(mark)
            cases.append(dict(shape=list(ids.shape),seconds=time.perf_counter()-begin))
    finally:e.seed,e.counter,e.consume_checkpoint_cache=state
    after=(optimizer.updates,optimizer.accumulated,len(optimizer.present))
    if after!=before:raise RuntimeError('prewarm changed optimizer progress')
    return dict(cases=cases,optimizer_before=before,optimizer_after=after,scope='Synthetic forward/backward only; no gradient accumulation or parameter updates')
