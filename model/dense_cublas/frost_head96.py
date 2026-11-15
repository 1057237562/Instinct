"""Optional SM120 96-column attention with a bijective dQ scratch layout.

The installed frontend currently rounds 96 to 128. This creates a separate
template module; it does not edit the optional NVIDIA source or its global
configuration. Selected only by the explicit frost_native_head96 experiment.
"""
import hashlib
import os
from pathlib import Path
import sys
import types


def specialize(original):
    params=original.FROST_TEMPLATE_PARAMS
    if params.q_tile!=64 or params.kv_tile not in (64,128):
        raise ValueError('native head96 specialization requires q64 and kv64/128')
    # CuTe captures helper globals during decoration. Rebinding the module's
    # scatter symbol afterwards does not replace the compiled helper. Materialize
    # an isolated, content-addressed derivative with its copyright header intact.
    source=Path(original.__file__).read_text(encoding='utf-8')
    old='''                if cutlass.const_expr(D_QK >= 64):
                    pair_row = i_pair % (Q_TILE // 8)
                    pair_col = i_pair // (Q_TILE // 8)
                    addr = dq_accum_base + (t_row + pair_row * 8) * (H * D_QK) + t_col * 2 + pair_col * 64
                else:
                    addr = dq_accum_base + (t_row + (t_col // 16) * 8 + i_pair * 16) * (H * D_QK) + (t_col % 16) * 2'''
    if source.count(old)!=1:raise RuntimeError('unsupported frontend dQ scratch implementation')
    new='''                flat = (i_pair * 256 + math_tidx) * 2
                addr = dq_accum_base + (flat // D_QK) * (H * D_QK) + flat % D_QK'''
    source=source.replace(old,new)
    origin=Path(original.__file__).parent
    chain=(origin/'bprop_chain_f16.py').read_text(encoding='utf-8')
    common=(origin.parent/'bprop_chain_common.py').read_text(encoding='utf-8')
    host=(origin/'prepared_host.py').read_text(encoding='utf-8')
    convert_old=old.replace('math_tidx','tidx')
    if chain.count(convert_old)!=1:raise RuntimeError('unsupported frontend dQ conversion implementation')
    chain=chain.replace(convert_old,new.replace('math_tidx','tidx'))
    old_zero='zero_rows_per_pass = 32 if cutlass.const_expr(D_QK == 32) else 16'
    if common.count(old_zero)!=1:raise RuntimeError('unsupported frontend dQ initialization implementation')
    common=common.replace(old_zero,'zero_rows_per_pass = 32 if cutlass.const_expr(D_QK % 64 != 0) else 16')
    digest=hashlib.sha256((source+chain+common+host+':flat96-v3').encode()).hexdigest()
    package='cudnn.frost._templates.instinct_head96_'+digest[:16]
    directory=Path(__file__).resolve().parents[2]/'.cache/instinct-frost-head96'/digest
    directory.mkdir(parents=True,exist_ok=True)
    imports=(('from cudnn.sdpa.bwd.kernels.sm120.prepared_host import compact, compile_host',f'from {package}.host import compact, compile_host'),
             ('from cudnn.sdpa.bwd.kernels.bprop_chain_common import',f'from {package}.common import'),
             ('from .bprop_chain_f16 import','from .chain import'))
    contents=[source,chain,host]
    for i,(before,after) in enumerate(imports):
        if contents[i].count(before)!=1:raise RuntimeError('unsupported frontend helper imports')
        contents[i]=contents[i].replace(before,after)
    for name,content in zip(('main.py','chain.py','host.py','common.py'),(*contents,common)):
        path=directory/name
        if path.exists():
            if path.read_text(encoding='utf-8')!=content:raise RuntimeError('modified head96 source cache: '+str(path))
        else:
            temporary=path.with_suffix('.'+str(os.getpid())+'.tmp')
            temporary.write_text(content,encoding='utf-8');temporary.replace(path)
    if package not in sys.modules:
        module=types.ModuleType(package);module.__path__=[str(directory)]
        sys.modules[package]=module
    from cudnn.frost.template_loader import load_template
    module=load_template(str(directory/'main.py'),original.FROST_TEMPLATE_PARAMS,tag='instinct_head96_flat_dq')
    if not getattr(module,'instinct_native_head96',False):
        previous=module.padded_head_dims
        module.padded_head_dims=lambda q,v:(96,96) if q==v==96 else previous(q,v)
        module.SUPPORTED_HEAD_DIMS=tuple(sorted((*module.SUPPORTED_HEAD_DIMS,96)))
        kind=module.SM120FusedMultiHeadAttentionFP16Backward
        kind.DEFAULT_TILES=dict(kind.DEFAULT_TILES)|{96:(64,64)}
        module.instinct_native_head96=True
    return module
