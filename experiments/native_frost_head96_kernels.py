"""Experimental bijective dQ scratch layout for a 64 x 96 MMA tile.

The frontend's existing scratch permutation emits 64-column stripes. A final
32-column stripe overlaps the next head when D=96. Use a flat pair permutation
instead, and invert the MMA fragment mapping in a separate conversion kernel.
"""
import triton
import triton.language as tl


@triton.jit
def zero_dq96(W,OFFSET:tl.constexpr,N:tl.constexpr,B:tl.constexpr=1024):
    i=tl.program_id(0)*B+tl.arange(0,B)
    tl.store(W+OFFSET+i,0.,i<N)


def prepare(backend,workspace,docs,seq,heads,dim):
    # Upstream initialization loops over complete 64-column stripes, leaving
    # the last 32 columns dirty at D=96. Clear the entire dQ region first.
    if dim!=96:raise ValueError('head96 initialization only supports D=96')
    offset=docs*heads*triton.cdiv(seq,128)*128;n=offset*96
    backend.launch(zero_dq96,(triton.cdiv(n,1024),),(workspace,),OFFSET=offset,N=n)


@triton.jit
def convert_dq96(W,Q,SCALE:tl.constexpr,DOCS:tl.constexpr,SEQ:tl.constexpr,H:tl.constexpr,B:tl.constexpr=256):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<DOCS*SEQ*H*96
    column=i%96;head=(i//96)%H;row=(i//(96*H))%SEQ;doc=i//(SEQ*H*96)
    local=row%64;wm=local//16;half=(local%16)//8;group=local%8
    wn=column//48;fragment=(column%48)//8;pair_lane=(column%8)//2;component=column%2
    thread=(wn*4+wm)*32+group*4+pair_lane
    flat=((half+2*fragment)*256+thread)*2+component
    source_row=(row//64)*64+flat//96;source_column=flat%96
    rounded=tl.cdiv(SEQ,128)*128
    scratch=DOCS*H*rounded
    source=((doc*rounded+source_row)*H+head)*96+source_column
    value=tl.load(W+scratch+source,valid,0).to(tl.float32)*SCALE
    tl.store(Q+i,value,valid)


def correct(backend,workspace,dq,docs,seq,heads,dim,scale):
    if dim!=96:raise ValueError('head96 correction only supports D=96')
    backend.launch(convert_dq96,(triton.cdiv(dq.numel,256),),(workspace,dq),
        SCALE=scale,DOCS=docs,SEQ=seq,H=heads)
