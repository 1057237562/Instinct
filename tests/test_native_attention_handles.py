"""The optional shared-handle ABI keeps stream and plan ownership separate."""
import ctypes as ct
from types import SimpleNamespace
import pytest
from model.dense_cublas.vendor_bridge import Sdpa


def bridge(shared=True):
    b=Sdpa.__new__(Sdpa);b.shared_handles=shared;b.tuning_abi=True
    b.handles={};b.plans=[];b.dirs=[];calls=[];serial=100
    def put(output):
        nonlocal serial
        serial+=1;ct.cast(output,ct.POINTER(ct.c_void_p))[0]=serial
        return serial
    def handle(stream,output):calls.append(('create_handle',stream,put(output)));return 0
    def plan(*args):
        calls.append(('create_plan',args[0] if shared else None,put(args[-4])))
        ct.cast(args[-3],ct.POINTER(ct.c_uint64))[0]=256
        return 0
    b.lib=SimpleNamespace(sdpa_handle_create=handle,sdpa_create_shared_ex=plan,sdpa_create_ex=plan,
        sdpa_destroy=lambda p:calls.append(('destroy_plan',p.value)),
        sdpa_handle_destroy=lambda h:calls.append(('destroy_handle',h.value)),
        sdpa_error=lambda:b'plan failure')
    return b,calls


def test_plans_share_only_the_same_stream_and_close_before_handles():
    b,calls=bridge()
    for stream in (10,10,20):b.plan(1,512,16,4,96,False,stream,tune=True)
    assert len(b.plans)==3 and len(b.handles)==2
    assert len([x for x in calls if x[0]=='create_handle'])==2
    b.close();tail=[x[0] for x in calls[-5:]]
    assert tail==['destroy_plan']*3+['destroy_handle']*2
    count=len(calls);b.close();assert len(calls)==count


def test_failed_plan_creation_keeps_handle_owned_for_cleanup():
    b,calls=bridge();b.lib.sdpa_create_shared_ex=lambda *args:-1
    with pytest.raises(RuntimeError,match='plan failure'):b.plan(1,512,16,4,96,False,10)
    assert len(b.handles)==1 and not b.plans
    b.close();assert calls[-1][0]=='destroy_handle'


def test_legacy_abi_retains_independent_plan_ownership():
    b,calls=bridge(False)
    b.plan(1,512,16,4,96,False,10,tune=True)
    assert not b.handles and len(b.plans)==1
    b.close();assert [x[0] for x in calls]==['create_plan','destroy_plan']


def test_requesting_shared_handles_falls_back_on_an_old_library(monkeypatch):
    from pathlib import Path
    from model.dense_cublas import vendor_bridge
    old,calls=bridge(False);lib=old.lib
    for name in ('sdpa_handle_create','sdpa_handle_destroy','sdpa_create_shared_ex'):delattr(lib,name)
    lib.sdpa_create=lambda *args:0;lib.sdpa_run=lambda *args:0
    lib.sdpa_candidate_count=lambda *args:1;lib.sdpa_candidate=lambda *args:0
    lib.sdpa_select=lambda *args:0
    monkeypatch.setattr(vendor_bridge.ct,'CDLL',lambda path:lib)
    monkeypatch.setattr(Path,'is_file',lambda path:True)
    monkeypatch.setattr(Path,'is_dir',lambda path:False)
    candidate=Sdpa('old-native-sdpa',shared_handles=True)
    assert candidate.tuning_abi and not candidate.shared_handles
    candidate.plan(1,512,16,4,96,False,10,tune=True)
    candidate.close();assert calls[-1][0]=='destroy_plan'
