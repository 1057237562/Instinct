"""CPU-only compiler capability probe; inspect whether SM120 MX dot is native."""
import argparse
import json
from pathlib import Path
import triton
import triton.language as tl


@triton.jit
def dot(A,B,SA,SB,O,TRANSPOSE_SCALE:tl.constexpr=False):
    m=tl.arange(0,16);n=tl.arange(0,32);k=tl.arange(0,64);s=tl.arange(0,2)
    a=tl.load(A+m[:,None]*64+k[None,:]);b=tl.load(B+k[:,None]*32+n[None,:])
    sa=tl.load(SA.to(tl.pointer_type(tl.uint8))+m[:,None]*2+s[None,:])
    sb=tl.load(SB.to(tl.pointer_type(tl.uint8))+s[:,None]*32+n[None,:])
    if TRANSPOSE_SCALE:sb=tl.trans(sb)
    result=tl.dot_scaled(a,sa,'e4m3',b,sb,'e4m3')
    tl.store(O+m[:,None]*32+n[None,:],result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--rhs-nk',action='store_true');a=p.parse_args()
    from model.dense_triton.runtime import compile_kernel
    report=dict(triton=triton.__version__,arch=120,scope='CPU compilation, no CUDA execution')
    try:
        kernel=compile_kernel(dot,['fp8e4nv','fp8e4nv','u16','u16','fp32'],dict(TRANSPOSE_SCALE=a.rhs_nk),120)
        ptx=kernel.asm['ptx'];a.output.with_suffix('.ptx').write_text(ptx)
        report.update(status='compiled',native_mx='kind::mxf8' in ptx,instructions=sorted(set(line.strip().split(' {')[0] for line in ptx.splitlines() if 'mma.' in line)))
    except Exception as exc:report.update(status='unsupported',error=str(exc))
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
