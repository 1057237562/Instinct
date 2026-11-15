"""Fail closed before compiling an unvalidated frontend geometry or source."""
from types import SimpleNamespace
import pytest


def test_head96_rejects_unvalidated_mma_partition():
    from model.dense_cublas.frost_head96 import specialize
    for q,k in ((32,64),(64,32),(128,64)):
        with pytest.raises(ValueError,match='q64'):
            specialize(SimpleNamespace(FROST_TEMPLATE_PARAMS=SimpleNamespace(q_tile=q,kv_tile=k)))


def test_head96_rejects_changed_upstream_scratch_code(tmp_path):
    from model.dense_cublas.frost_head96 import specialize
    source=tmp_path/'upstream.py';source.write_text('# Unrecognized new frontend implementation\n')
    with pytest.raises(RuntimeError,match='unsupported frontend dQ scratch'):
        specialize(SimpleNamespace(FROST_TEMPLATE_PARAMS=SimpleNamespace(q_tile=64,kv_tile=64),__file__=str(source)))
    assert source.read_text()=='# Unrecognized new frontend implementation\n'


def test_native_head96_plan_does_not_silently_serve_other_dimensions():
    from model.dense_cublas.frost_attention import FrostBackward
    f=FrostBackward.__new__(FrostBackward);f.native_head96=True
    with pytest.raises(ValueError,match='D=96'):f.plan(1,256,16,4,128)
