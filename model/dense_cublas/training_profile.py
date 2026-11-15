"""Named native compute policies; the V2 production override has one catalog.

Importing this module is CPU-only. Hardware/bridge checks run after opening the
backend; attention dependencies load in the normal pre-training warmup order.
"""
import json
from pathlib import Path


PROFILES=('tensorwise','v2_mxfp8','v2_mxfp8_sft')
ROOT=Path(__file__).resolve().parents[2]


def settings(name):
    if name not in PROFILES:raise ValueError('unknown native compute profile: '+str(name))
    if name=='tensorwise':return {'backend':{},'compact_long_multiple':0}
    policy=json.loads((ROOT/'trainer/profiles/native_v2_mxfp8.json').read_text(encoding='utf-8'))
    if name=='v2_mxfp8_sft':
        policy['backend']['activation_token_capacities']=[16384]
        policy['compact_long_multiple']=0
    return policy


def validate(name,args,config):
    policy=settings(name)
    if name=='tensorwise':return policy
    required=dict(train_type='pretrain',dtype='bfloat16',param_dtype='bf16',optimizer='muon',
        muon_mantissa=1,muon_momentum_dtype='bf16',muon_gemm_backend='cublas',
        fp8_filter='auto',fused_ops=True,packing=True,packed_attention=True,
        attention_backend='cudnn',allocator_backend='async',batch_size=3,max_seq_len=4096,
        fuse_projections=False,cuda_graph=False,overlap_dw=False,loss_chunk_size=0)
    if name=='v2_mxfp8_sft':
        required.update(train_type='full_sft',batch_size=1,max_seq_len=16384,loss_chunk_size=1024)
    bad=[key for key,value in required.items() if getattr(args,key,None)!=value]
    if args.fp8=='0':bad.append('fp8')
    expected_buckets=[16384] if name=='v2_mxfp8_sft' else [3072,4096]
    if sorted({int(v) for v in args.sequence_buckets.split(',') if v})!=expected_buckets:bad.append('sequence_buckets')
    model=dict(hidden_size=1536,num_hidden_layers=32,intermediate_size=5504,
        num_attention_heads=16,num_key_value_heads=4,head_dim=96,dropout=0.,hidden_act='silu',
        model_architecture='standard',use_moe=False,use_grad_checkpoint=4,flash_attn=True)
    bad.extend(key for key,value in model.items() if config.get(key)!=value)
    if config.get('early_exit_layers'):bad.append('early_exit_layers')
    if config.get('max_position_embeddings',0)<4096:bad.append('max_position_embeddings')
    if bad:raise ValueError('v2_mxfp8 requires the native V2 packed BF16/Muon recipe; incompatible: '+', '.join(bad))
    return policy


def apply(backend,name,policy):
    if name=='tensorwise':return
    if getattr(backend,'arch',None)!=120:raise ValueError('v2_mxfp8 requires SM120; use compute_profile=tensorwise on other GPUs')
    native=getattr(backend,'native',None)
    if native is None or any(getattr(native,key,None) is None for key in ('_plan_mxfp8','_plan_pointer_gram_ex')):
        raise RuntimeError('v2_mxfp8 requires rebuilding the native bridge with MXFP8 and FP16 Gram support')
    for key,value in policy['backend'].items():
        if key=='kernel_options':backend.kernel_options.update(value)
        else:setattr(backend,key,value)
    # Keep DSL loading and plan creation in NativeAttention.warm_catalog, as in
    # the validated candidate. It still runs before any training update. Loading
    # another CUDA runtime here would change initialization order unnecessarily.


def checkpoint_profile(config):
    """Identify the already validated experimental V2 checkpoint contract."""
    required=dict(native_fp8_recipe='mxfp8_experimental',native_mxfp8_gradient_dtype='fp8e4nv',
        native_mxfp8_small_projections=True,native_mxfp8_cache_ffn=True,
        native_muon_gram_recurrence='fp16_restart3')
    return 'v2_mxfp8' if all(config.get(k)==v for k,v in required.items()) else 'tensorwise'


def prepare_data(loader,pad,policy):
    multiple=policy['compact_long_multiple']
    if not multiple:return
    from .packing import compact_packed_batch
    original=loader.batches
    def batches(*args,**kwargs):
        for cursor,batch in original(*args,**kwargs):
            # Group-tail batches have smaller, rare token budgets. Keep their
            # original shape; the measured capacity shortlist covers full 12k
            # batches only and must not round model weight dimensions.
            if batch[0].shape==(3,4096):batch=compact_packed_batch(batch,pad,multiple)
            yield cursor,batch
    loader.batches=batches


def prewarm(model,optimizer,policy):
    if not policy['compact_long_multiple']:return None
    if optimizer.accumulated:
        print('Skipping synthetic shape prewarm with restored accumulated gradients.',flush=True)
        return None
    from .warmup import warm
    return warm(model,optimizer,multiple=policy['compact_long_multiple'])


def paired_batches(source, pad, capacity):
    """Keep the source cursor and join two supervised microbatches in order."""
    import numpy as np

    pairs = []
    valid_tokens = source_rows = 0
    cursor = None

    def emit():
        ids = np.full((max(1, len(pairs)), capacity), pad, np.int64)
        labels = np.full_like(ids, -100)
        segments = np.full_like(ids, -1)
        for slot, (batch_ids, batch_labels, batch_segments) in enumerate(pairs):
            offset = document = 0
            for row_ids, row_labels, row_segments in zip(batch_ids, batch_labels, batch_segments):
                count = int(np.count_nonzero(row_segments >= 0))
                if not count:
                    continue
                end = offset + count
                ids[slot, offset:end] = row_ids[:count]
                labels[slot, offset:end] = row_labels[:count]
                starts = np.r_[True, row_segments[1:count] != row_segments[:count-1]]
                segments[slot, offset:end] = np.cumsum(starts) - 1 + document
                document += int(starts.sum())
                offset = end
        return (dict(cursor=cursor, valid_tokens=valid_tokens, source_rows=source_rows,
                     source_microbatches=len(pairs)), (ids, labels, segments))

    for source_cursor, batch in source:
        supervised = np.any(batch[1][:, 1:] != -100)
        if supervised and len(pairs) == 2:
            yield emit()
            pairs.clear()
            valid_tokens = source_rows = 0
        cursor = source_cursor
        source_rows += len(batch[0])
        valid_tokens += cursor['valid_tokens']
        if supervised:
            pairs.append(batch)
    if source_rows:
        yield emit()


def paired_loss(model, batch):
    """Sum the original microbatch means in one forward/backward."""
    ids, labels, segments = batch
    reference = getattr(model, 'cache_precision_reference', None)
    if reference is not None and ids.shape[0] == 2:
        from .selective import CACHE_BYTES
        model.cache_precision_reference = {
            'selective_cache_mib': 2 * reference.get('selective_cache_mib', CACHE_BYTES / 2**20)
        }
    try:
        output = model(ids, sequence_ids=segments)
    finally:
        if reference is not None:
            model.cache_precision_reference = reference
    if not hasattr(labels, 'engine'):
        labels = model.e.array(labels)
    return model.e.ce(output.logits, labels, shift=True, sum_sequence_means=True)


def prewarm_pairs(model, capacity):
    """Exercise both long and many-document batches before committing progress."""
    import numpy as np

    engine = model.e
    backend = engine.backend
    previous = getattr(engine, 'consume_checkpoint_cache', False)
    seed, counter = engine.seed, engine.counter
    try:
        engine.consume_checkpoint_cache = True
        for length in (4096, 512):
            mark = backend.mark()
            ids = np.zeros((2, capacity), np.int64)
            segments = np.broadcast_to(np.arange(capacity) // length, ids.shape).copy()
            labels = ids.copy()
            labels[:, ::length] = -100
            loss = paired_loss(model, (ids, labels, segments))
            if not np.isfinite(engine.item(loss)):
                raise FloatingPointError('nonfinite paired-batch prewarm loss')
            engine.vjp(loss, leaf_callback=lambda parameter, gradient: True)
            backend.synchronize()
            backend.release_since(mark)
    finally:
        engine.seed, engine.counter = seed, counter
        engine.consume_checkpoint_cache = previous
