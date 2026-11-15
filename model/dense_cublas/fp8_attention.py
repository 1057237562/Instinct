"""Experimental fused attention backward with tilewise E4M3 gradient GEMMs.

Forward and score recomputation remain BF16. This is tilewise FP8, not OCP
MX32: SM120's installed Triton compiler emulates dot_scaled with BF16, while
ordinary E4M3 dot uses native FP8 tensor cores. Never enabled by the recipe.
"""
import triton
import triton.language as tl
from .mxfp8 import scale_exponent


@triton.jit
def tile_fp8(x):
    maximum=tl.max(tl.max(tl.abs(x.to(tl.float32)),0),0)
    exponent,inverse=scale_exponent(maximum,448.)
    quantized=tl.minimum(tl.maximum(x.to(tl.float32)*inverse,-448.),448.).to(tl.float8e4nv,fp_downcast_rounding='rtne')
    return quantized,tl.exp2(exponent.to(tl.float32)-127.)


@triton.jit
def fp8_attention_delta(O,DO,MASK,DELTA,DQ,DK,DV,TOKENS:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
                        BD:tl.constexpr,ROWS:tl.constexpr=4):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS);d=tl.arange(0,BD)
    valid=(row[:,None]<TOKENS*H)&(d[None,:]<D);pos=row[:,None]*D+d[None,:]
    o=tl.load(O+pos,valid,0).to(tl.float32);dy=tl.load(DO+pos,valid,0).to(tl.float32)
    delta=tl.sum(o*dy,1);tl.store(DELTA+row,delta,row<TOKENS*H)
    tl.store(DQ+pos,0.,valid)
    active=tl.load(MASK+row//H,row<TOKENS*H,-1)>=0
    # Valid dK/dV locations are overwritten once per query head. Only gaps
    # need initialization; clearing all three large FP32 buffers wastes HBM.
    tl.store(DK+pos,0.,valid&~active[:,None]);tl.store(DV+pos,0.,valid&~active[:,None])


@triton.jit
def fp8_attention_backward(Q,K,V,DO,LSE,DELTA,OFFSETS,LENGTHS,DQ,DK,DV,QQ,KQ,VQ,DOQ,SQ,SK,SV,SDO,QQT,KQT,DOT,
                           H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,
                           TOTAL:tl.constexpr,BM:tl.constexpr=32,BN:tl.constexpr=64,BD:tl.constexpr=128,FP8:tl.constexpr=True,SCORE_FP8:tl.constexpr=False,PRETRANSPOSED:tl.constexpr=False):
    first=tl.program_id(0)*BN;head=tl.program_id(1);doc=tl.program_id(2)
    length=tl.load(LENGTHS+doc);start=tl.load(OFFSETS+doc)//(H*D);kh=head//(H//HK)
    if first<length:
        n=first+tl.arange(0,BN);d=tl.arange(0,BD);m=tl.arange(0,BM)
        ko=((start+n[:,None])*HK+kh)*D+d[None,:]
        valid_k=(n[:,None]<length)&(d[None,:]<D)
        if not SCORE_FP8:k=tl.load(K+ko,valid_k,0)
        if FP8:
            if PRETRANSPOSED:kq=tl.load(KQT+(kh*D+d[None,:])*TOTAL+start+n[:,None],valid_k,0.)
            else:kq=tl.load(KQ+ko,valid_k,0.)
            vq=tl.load(VQ+ko,valid_k,0.)
            sq=tl.load(SQ);sk=tl.load(SK);sv=tl.load(SV);sdy=tl.load(SDO)
        else:v=tl.load(V+ko,valid_k,0)
        dk=tl.full((BN,BD),0.,tl.float32);dv=tl.full((BN,BD),0.,tl.float32)
        for qfirst in range(first,length,BM):
            mm=qfirst+m;qo=((start+mm[:,None])*H+head)*D+d[None,:]
            valid_q=(mm[:,None]<length)&(d[None,:]<D)
            if not SCORE_FP8:q=tl.load(Q+qo,valid_q,0)
            if not FP8:dy=tl.load(DO+qo,valid_q,0)
            lse=tl.load(LSE+(start+mm)*H+head,mm<length,float('inf'))
            delta=tl.load(DELTA+(start+mm)*H+head,mm<length,0.)
            if FP8:
                if PRETRANSPOSED:
                    qq=tl.load(QQT+(head*D+d[None,:])*TOTAL+start+mm[:,None],valid_q,0.)
                    dyqt=tl.load(DOT+(head*D+d[None,:])*TOTAL+start+mm[:,None],valid_q,0.)
                else:qq=tl.load(QQ+qo,valid_q,0.)
                dyq=tl.load(DOQ+qo,valid_q,0.)
            if SCORE_FP8:score=tl.dot(qq,tl.trans(kq))*(sq*sk*SCALE)
            else:score=tl.dot(q,tl.trans(k))*SCALE
            allowed=(mm[:,None]>=n[None,:])&(mm[:,None]<length)&(n[None,:]<length)
            probability=tl.exp2((score-lse[:,None])*1.4426950408889634)
            probability=tl.where(allowed,probability,0.)
            if FP8:
                dp=tl.dot(dyq,tl.trans(vq))*(sdy*sv)
                ds=probability*(dp-delta[:,None])*SCALE
                dsq,sds=tile_fp8(ds)
                pq=(probability*448.).to(tl.float8e4nv,fp_downcast_rounding='rtne')
                dk+=tl.dot(tl.trans(dsq),qq)*(sds*sq)
                dv+=tl.dot(tl.trans(pq),dyqt if PRETRANSPOSED else dyq)*(sdy/448.)
                dq=tl.dot(dsq,kq)*(sds*sk)
            else:
                dp=tl.dot(dy,tl.trans(v))
                ds=(probability*(dp-delta[:,None])*SCALE).to(q.dtype)
                dk=tl.dot(tl.trans(ds),q,dk)
                dv=tl.dot(tl.trans(probability.to(q.dtype)),dy,dv)
                dq=tl.dot(ds,k)
            tl.atomic_add(DQ+qo,dq,valid_q,sem='relaxed')
        out=((start+n[:,None])*H+head)*D+d[None,:]
        tl.store(DK+out,dk,valid_k);tl.store(DV+out,dv,valid_k)


def backward(ops,saved,dy,*,fp8=True,bm=32,bn=64,warps=8,score_fp8=False,pretransposed=False):
    q,k,v,mask,out,lse,(_,context),_,_=saved
    if q.dtype!='bf16' or mask is None or mask.shape!=q.shape[:2]:
        raise ValueError('tilewise FP8 backward requires a BF16 packed batch and full token mask')
    docs,maximum,h,hk,d,total,metadata=context[:7]
    b=ops.backend;mark=b.mark()
    delta=ops.empty((total,h),'fp32');dq=ops.empty(q.shape,'fp32')
    pk,pv=[ops.empty(q.shape,'fp32') for _ in range(2)]
    if fp8:
        from .quantization import quantize
        pairs=[quantize(ops,x.view(total,x.numel//total),'fp8e4nv',pretransposed and i!=2) for i,x in enumerate((q,k,v,dy))]
        quantized=tuple(pair[0] for pair in pairs);scales=tuple(ops._inverse(pair[1]) for pair in pairs)
    else:quantized=(q,k,v,dy);scales=(q,k,v,dy)
    transposed=tuple(quantized[i].transpose_data for i in (0,1,3)) if pretransposed else (q,k,dy)
    b.launch(fp8_attention_delta,(triton.cdiv(total*h,4),),(out,dy,mask,delta,dq,pk,pv),TOKENS=total,H=h,D=d,BD=triton.next_power_of_2(d))
    b.kernel_options[__name__+'.fp8_attention_backward']={'num_warps':warps,'num_stages':1}
    # Match the bridge's float reciprocal of float sqrt.
    import numpy as np
    scale=float(np.float32(1)/np.sqrt(np.float32(d)))
    b.launch(fp8_attention_backward,(triton.cdiv(maximum,bn),h,docs),
        (q,k,v,dy,lse,delta,metadata[0],metadata[3],dq,pk,pv,*quantized,*scales,*transposed),
        H=h,HK=hk,D=d,SCALE=scale,TOTAL=total,BM=bm,BN=bn,BD=triton.next_power_of_2(d),FP8=fp8,SCORE_FP8=score_fp8,PRETRANSPOSED=pretransposed)
    from model.dense_triton.attention import reduce_kv_groups
    dk,dv=[ops.empty(k.shape,'bf16') for _ in range(2)]
    b.launch(reduce_kv_groups,(triton.cdiv(dk.numel,512),),(pk,pv,dk,dv),N=dk.numel,DIM=d,HQ=h,HKV=hk)
    result=(ops.cast(dq,'bf16'),dk,dv);b.release_since(mark,keep=result)
    return result
