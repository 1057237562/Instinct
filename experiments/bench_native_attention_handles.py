"""Measure bounded cuDNN catalog residency; synthetic prewarm only, no updates."""
import argparse
import json
from pathlib import Path
import sys
from experiments.smoke_dense_triton import RejectTorch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared-handles',action='store_true')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();sys.meta_path.insert(0,RejectTorch())
    from model.dense_cublas.memory import TrainingBackend
    from model.dense_cublas.policy import enable_fused_ops
    from model.dense_cublas.vendor_attention import NativeAttention
    with TrainingBackend(allocator_backend='async',pool_limit_mib=128,async_release_mib=0) as b:
        enable_fused_ops(b)
        b.attention_plan_quantum=256;b.attention_plan_limit=128
        b.attention_tune=True;b.attention_cost_model='forward'
        b.attention_shared_handle=a.shared_handles
        native=NativeAttention(b);b.native_attention=native
        before=b.memory()
        native.warm_catalog(12288,4096,16,4,96)
        b.synchronize();after=b.memory()
        report=dict(status='complete',shared_handles=native.lib.shared_handles,
            plans=len(native.plans),handles=len(native.lib.handles) if native.lib.shared_handles else len(native.plans),
            before=before,after=after,device_residency_delta_bytes=before['device_free_bytes']-after['device_free_bytes'],
            metrics=native.metrics,torch_imported='torch' in sys.modules)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('before','after','metrics')}),flush=True)


if __name__=='__main__':main()
