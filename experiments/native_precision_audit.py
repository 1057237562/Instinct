"""Frozen trained weights: BF16 oracle, FP8 recipes and repeated controls.

No optimizer updates. Error moments reduce on the GPU in FP64; only small
partial sums reach the CPU, so whole-model gradients need not be serialized.
"""
import json
import math
from pathlib import Path
import re
import numpy as np
import triton
import triton.language as tl


@triton.jit
def gradient_moments(A,R,P,N:tl.constexpr,B:tl.constexpr=1024):
    block=tl.program_id(0);i=block*B+tl.arange(0,B);valid=i<N
    a=tl.load(A+i,valid,0).to(tl.float64);r=tl.load(R+i,valid,0).to(tl.float64)
    delta=a-r;offset=block*6
    values0=tl.sum(r*r,0);values1=tl.sum(a*a,0);values2=tl.sum(delta*delta,0)
    values3=tl.sum(a*r,0);values4=tl.max(tl.abs(delta),0)
    values5=tl.sum((valid&~((tl.abs(a)<float('inf'))&(tl.abs(r)<float('inf')))).to(tl.float64),0)
    tl.store(P+offset,values0.to(tl.int64,bitcast=True));tl.store(P+offset+1,values1.to(tl.int64,bitcast=True))
    tl.store(P+offset+2,values2.to(tl.int64,bitcast=True));tl.store(P+offset+3,values3.to(tl.int64,bitcast=True))
    tl.store(P+offset+4,values4.to(tl.int64,bitcast=True));tl.store(P+offset+5,values5.to(tl.int64,bitcast=True))


@triton.jit
def sampled_distribution(A,R,POSITIONS,ACTUAL_POSITIONS,P,VOCAB:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);pos=tl.load(POSITIONS+row);actual_pos=tl.load(ACTUAL_POSITIONS+row);i=tl.arange(0,B);valid=i<VOCAB
    a=tl.load(A+actual_pos*VOCAB+i,valid,-float('inf')).to(tl.float32)
    r=tl.load(R+pos*VOCAB+i,valid,-float('inf')).to(tl.float32)
    ac=a-tl.max(a,0);rc=r-tl.max(r,0)
    la=ac-tl.log(tl.sum(tl.exp(ac),0));lr=rc-tl.log(tl.sum(tl.exp(rc),0))
    pa=tl.exp(la);pr=tl.exp(lr)
    kl=tl.sum(tl.where(valid,pr*(lr-la),0.),0)
    tv=.5*tl.sum(tl.abs(pr-pa),0)
    agree=tl.argmax(a,0)==tl.argmax(r,0)
    diff=tl.where(valid,a-r,0.)
    tl.store(P+row*5,kl);tl.store(P+row*5+1,tv);tl.store(P+row*5+2,agree.to(tl.float32))
    tl.store(P+row*5+3,tl.max(tl.abs(diff),0));tl.store(P+row*5+4,tl.sum(diff*diff,0)/VOCAB)


def summarize(moment):
    rn,an,dn,dot,maximum,nonfinite=map(float,moment)
    return dict(relative_l2=math.sqrt(max(dn,0.)/rn) if rn else (0. if not dn else None),
                cosine=dot/math.sqrt(rn*an) if rn and an else None,
                norm_ratio=math.sqrt(max(an,0.)/rn) if rn else None,
                reference_l2=math.sqrt(max(rn,0.)),actual_l2=math.sqrt(max(an,0.)),
                max_abs=maximum,nonfinite=int(nonfinite))


def combine(values):
    stack=np.asarray(values,np.float64);total=stack.sum(0)
    total[4]=stack[:,4].max()
    return total


def compare_gradients(b,actual,reference):
    moments={}
    for name,tensor in actual.items():
        expected=reference[name]
        if tensor.shape!=expected.shape:raise ValueError('gradient shape changed: '+name)
        mark=b.mark();partial=b.empty((triton.cdiv(tensor.numel,1024),6),'i64')
        b.launch(gradient_moments,(partial.shape[0],),(tensor,expected,partial),N=tensor.numel)
        values=b.to_numpy(partial).view(np.float64);moments[name]=combine(values)
        b.release_since(mark)
    layers={}
    for name,value in moments.items():
        match=re.match(r'model\.layers\.(\d+)\.',name)
        group='layer_'+match[1] if match else 'embedding_and_head' if 'embed_tokens' in name or 'lm_head' in name else 'other'
        layers.setdefault(group,[]).append(value)
    parameters={name:summarize(value) for name,value in moments.items()}
    finite=[v['relative_l2'] for v in parameters.values() if v['relative_l2'] is not None]
    return dict(global_metrics=summarize(combine(list(moments.values()))),
                layers={name:summarize(combine(values)) for name,values in layers.items()},parameters=parameters,
                parameter_relative_l2_percentiles=dict(zip(('p50','p95','max'),map(float,np.percentile(finite,[50,95,100])))))


def compare_logits(b,actual,reference,positions,actual_positions=None):
    mark=b.mark();partial=b.empty((positions.numel,5),'fp32');vocab=actual.shape[-1]
    b.launch(sampled_distribution,(positions.numel,),(actual,reference,positions,actual_positions or positions,partial),VOCAB=vocab,B=triton.next_power_of_2(vocab))
    value=b.to_numpy(partial);b.release_since(mark)
    return dict(sampled_valid_positions=positions.numel,mean_kl_reference_to_actual=float(value[:,0].mean()),
        p95_kl=float(np.percentile(value[:,0],95)),mean_total_variation=float(value[:,1].mean()),
        top1_agreement=float(value[:,2].mean()),max_logit_error=float(value[:,3].max()),
        logit_rmse=float(np.sqrt(value[:,4].mean())))


def validate_metrics(b):
    """Check the audit instrument against CPU FP64 arithmetic before use."""
    mark=b.mark();rng=np.random.default_rng(153)
    aa=rng.standard_normal(1261,dtype=np.float32);rr=rng.standard_normal(1261,dtype=np.float32)
    a=b.from_numpy(aa);r=b.from_numpy(rr);partial=b.empty((triton.cdiv(aa.size,1024),6),'i64')
    b.launch(gradient_moments,(partial.shape[0],),(a,r,partial),N=aa.size)
    actual=combine(b.to_numpy(partial).view(np.float64))
    af,rf=aa.astype(np.float64),rr.astype(np.float64);delta=af-rf
    expected=np.asarray([np.sum(rf*rf),np.sum(af*af),np.sum(delta*delta),np.sum(af*rf),np.max(np.abs(delta)),0.])
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    logits=b.from_numpy(np.asarray([[np.log(2.),0.,0.]],np.float32));uniform=b.from_numpy(np.zeros((1,3),np.float32))
    pos=b.from_numpy(np.asarray([0],np.int32));result=compare_logits(b,logits,uniform,pos)
    expected_kl=float(np.mean(np.log((1/3)/np.asarray([.5,.25,.25]))))
    np.testing.assert_allclose(result['mean_kl_reference_to_actual'],expected_kl,atol=2e-7)
    np.testing.assert_allclose(result['mean_total_variation'],1/6,atol=2e-7)
    assert result['top1_agreement']==1.
    shifted=b.from_numpy(np.asarray([[9.,-9.,0.],[np.log(2.),0.,0.]],np.float32));last=b.from_numpy(np.asarray([1],np.int32))
    remapped=compare_logits(b,shifted,uniform,pos,last)
    np.testing.assert_allclose(remapped['mean_kl_reference_to_actual'],expected_kl,atol=2e-7)
    b.release_since(mark)


def run_audit(args):
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.backend import ScheduleConfig
    from model.dense_cublas.autograd import Engine
    from model.dense_cublas.model import Model
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas.vendor_attention import NativeAttention
    from model.dense_cublas import checkpoint
    report=dict(status='running',scope='Fixed trained weights and production packed samples; no optimizer updates. BF16 is the compute oracle, not FP32.',
                source_weights=str(args.weights),samples=str(args.samples),modes=args.modes,repeats=args.repeats,cases=[])
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    def write():output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    meta=checkpoint.metadata(args.weights);cfg=dict(meta['config'])
    report['checkpoint_progress']=meta.get('progress',{});report['checkpoint_precision']=meta.get('precision',{})
    for key in ('native_fp8_recipe','native_mxfp8_gradient_dtype','native_mxfp8_small_projections','native_mxfp8_fast_accum','native_mxfp8_cache_ffn','native_muon_gram_recurrence'):cfg.pop(key,None)
    candidate={}
    if 'mx_candidate' in args.modes:
        candidate=json.loads(Path(args.candidate_config).read_text())
        attention_options={'attention_frost_backward','attention_frost_min_seq','attention_frost_min_plan_seq','attention_frost_documentwise','attention_frost_max_docs','frost_q_tile','frost_kv_tile','frost_alias_inputs','frost_native_head96','activation_token_capacities'}
        if any(not k.startswith('mxfp8_') and k not in attention_options for k in candidate):
            raise ValueError('cross-precision candidate must contain only explicit mxfp8 options')
        report['candidate']=candidate
    cfg.update(use_grad_checkpoint=4,selective_cache_mib=args.audit_cache_mib,selective_cache_by_seq={})
    report['model_config']=cfg;write()
    with np.load(args.samples,allow_pickle=False) as sample,TrainingBackend(allocator_backend='async',pool_limit_mib=128,config=ScheduleConfig(tune=False)) as b:
        validate_metrics(b);report['metric_instrument_validated']=True;write()
        enable_fused_ops(b);b.attention_cost_model=False;b.attention_tune=False;b.attention_prewarm=False
        b.native_attention=NativeAttention(b)
        e=Engine(b,fp8=False,fp8_filter='auto');m=Model(e,cfg,param_dtype='bf16',muon_mantissa=True)
        checkpoint.load(args.weights,m)
        expected_parameters={p.name:p.size for p in m.parameters()}
        report['parameter_count']=sum(expected_parameters.values());report['parameter_tensors']=len(expected_parameters);write()
        for bucket in args.buckets:
            bucket_mark=b.mark();ids,labels,segments=[sample[f'{key}{bucket}'] for key in ('ids','labels','segments')]
            batch,seq=ids.shape;valid=np.flatnonzero((labels[:,1:]!=-100).reshape(-1))
            flat=valid[np.linspace(0,len(valid)-1,min(256,len(valid)),dtype=np.int64)]
            position=flat//(seq-1)*seq+flat%(seq-1);positions=b.from_numpy(position.astype(np.int32))
            oracle=None;oracle_logits=None;oracle_loss=None
            for mode in args.modes:
                mode_mark=b.mark();first=None;first_logits=None;first_loss=None
                e.ops.fp8=not mode.startswith('bf16');b.mxfp8_training=mode.startswith('mx_')
                b.mxfp8_fast_accum=False;b.mxfp8_small_projections=False
                b.mxfp8_cache_ffn=False
                b.activation_token_capacities=()
                b.attention_frost_backward=False
                b.frost_native_head96=False
                b.mxfp8_gradient_dtype='fp8e4nv' if mode=='mx_e4' else 'fp8e5'
                b.mxfp8_norm_fusion=b.mxfp8_training;b.mxfp8_norm_backward_fusion=b.mxfp8_training
                b.kernel_options['model.dense_cublas.mxfp8.mxfp8_swiglu']={'num_warps':8}
                if mode=='mx_candidate':
                    for key,value in candidate.items():setattr(b,key,value)
                actual_batch=(ids,labels,segments);actual_positions=positions
                if args.compact_candidate_multiple and mode in ('mx_candidate','bf16_compact') and seq==4096:
                    from model.dense_cublas.packing import compact_packed_batch
                    actual_batch=compact_packed_batch(actual_batch,0,args.compact_candidate_multiple)
                    if actual_batch[0].shape!=ids.shape:
                        active=(segments>=0).ravel();mapping=np.cumsum(active)-1
                        if not active[position].all():raise ValueError('logit audit sampled an inactive source position')
                        actual_positions=b.from_numpy(mapping[position].astype(np.int32))
                        assert np.count_nonzero(actual_batch[1][:,1:]!=-100)==len(valid)
                for repeat in range(args.repeats):
                    run_mark=b.mark();e.seed=42;e.counter=0
                    out=m(actual_batch[0],actual_batch[1],sequence_ids=actual_batch[2]);loss=e.item(out.loss)
                    if not np.isfinite(loss):raise FloatingPointError('nonfinite loss in '+mode)
                    grads={p.name:g for p,g in e.vjp(out.loss).items()};logits=out.logits.data
                    if grads.keys()!=expected_parameters.keys():raise RuntimeError('precision audit did not cover every trainable parameter')
                    if oracle is None:
                        oracle,oracle_logits,oracle_loss=grads,logits,loss
                    compared=compare_gradients(b,grads,oracle)
                    row=dict(bucket=bucket,shape=list(actual_batch[0].shape),source_shape=list(ids.shape),useful_targets=int(len(valid)),mode=mode,repeat=repeat,
                        loss=loss,loss_delta_to_bf16=loss-oracle_loss,gradient_to_bf16=compared,
                        logits_to_bf16=compare_logits(b,logits,oracle_logits,positions,actual_positions))
                    if compared['global_metrics']['nonfinite']:raise FloatingPointError('nonfinite gradient in '+mode)
                    if first is not None:
                        row['repeat_gradient']=compare_gradients(b,grads,first)
                        row['repeat_loss_delta']=loss-first_loss
                    report['cases'].append(row);write()
                    print(json.dumps({k:v for k,v in row.items() if k not in ('gradient_to_bf16','repeat_gradient')}|dict(
                        gradient_to_bf16=compared['global_metrics'],repeat_gradient=row.get('repeat_gradient',{}).get('global_metrics'))),flush=True)
                    if first is None:first,first_logits,first_loss=grads,logits,loss
                    retained=(*first.values(),first_logits,*oracle.values(),oracle_logits)
                    b.release_since(run_mark,keep=retained)
                b.release_since(mode_mark,keep=(*oracle.values(),oracle_logits))
            b.release_since(bucket_mark)
    report.update(status='complete',torch_imported='torch' in __import__('sys').modules);write()
