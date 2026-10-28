"""Bounded candidate selection and confirmation, without CUDA."""
import pytest
from model.dense_cublas.tuning import gemm_shortlist,race,ffn_shortlist


def test_gemm_shortlist_keeps_prior_and_different_wave_geometry():
    candidates=[dict(index=i,algo_id=67,workspace_bytes=0,waves=w) for i,w in enumerate((7.4,7.4,7.4,3.7,3.7,3.7,14.7))]
    selected=gemm_shortlist(candidates)
    assert [c['index'] for c in selected]==[0,1,3,4]
    assert all('rejected' in c for c in candidates if c not in selected)


def test_shortlist_keeps_workspace_alternative_without_exceeding_budget():
    candidates=[dict(index=i,workspace_bytes=w,waves=v) for i,(w,v) in enumerate(((0,8),(0,8),(0,6),(0,4),(2048,12),(0,2)))]
    selected=gemm_shortlist(candidates)
    assert len(selected)==4 and {0,1,4} <= {c['index'] for c in selected}


@pytest.mark.parametrize('speed,expected',[(.8,1),(.995,0)])
def test_race_confirms_a_material_win_and_alternates_finalists(speed,expected):
    candidates=[dict(index=i) for i in range(4)];calls=[]
    def measure(candidate,repeats):
        calls.append((candidate['index'],repeats))
        return (1.,speed,1.1,1.2)[candidate['index']]
    assert race(candidates,measure)['index']==expected
    assert calls[:4]==[(0,2),(1,2),(2,2),(3,2)]
    assert calls[4:]==[(0,3),(1,3),(1,3),(0,3),(0,3),(1,3)]
    assert all('median_ms' not in c for c in candidates[2:])


def test_single_supported_plan_does_not_pay_for_redundant_timing():
    def forbidden(*args):raise AssertionError('no comparison to perform')
    assert race([dict(index=0)],forbidden)['selection_reason']=='only supported candidate'


def test_search_budget_and_small_ffn_fallback():
    with pytest.raises(ValueError):race([dict(index=i) for i in range(5)],lambda *args:1.)
    with pytest.raises(ValueError):race([dict(index=3)],lambda *args:1.)
    assert len(ffn_shortlist(137,192))==1
    assert len(ffn_shortlist(12288,5504))==4
    assert all(c['warps']>=4 for c in ffn_shortlist(12288,5504))


@pytest.mark.parametrize('budget,kv_layers,gate_layers',[(1280,5,0),(1792,32,0),(2048,32,1)])
def test_attention_and_kv_share_one_hard_cache_budget(budget,kv_layers,gate_layers):
    from model.dense_cublas.model import configuration
    from model.dense_cublas.selective import recipe_plan
    config=configuration(dict(hidden_size=1536,num_hidden_layers=32,intermediate_size=5504,
        num_attention_heads=16,num_key_value_heads=4,head_dim=96,selective_cache_mib=budget))
    plans=[recipe_plan(config,(4,3072,1536),'bf16',i,attention_min_seq=1,auxiliary_kv=True) for i in range(32)]
    assert sum(p.attention for p in plans)==32
    assert sum(any('k_proj' in n for n in p.linear_names) for p in plans)==kv_layers
    assert sum(any('gate_proj' in n for n in p.linear_names) for p in plans)==gate_layers
    assert 32*36.75+kv_layers*18+gate_layers*258<=budget


def test_attention_tuning_cap_falls_back_without_removing_supported_plan():
    import ctypes as ct
    from collections import OrderedDict
    from types import SimpleNamespace
    from model.dense_cublas.vendor_attention import NativeAttention
    calls=[]
    def plan(*args,**kwargs):
        calls.append(kwargs)
        if kwargs.get('tune'):raise RuntimeError('no candidate below tuning workspace cap')
        return ct.c_void_p(1234),600*2**20
    native=NativeAttention.__new__(NativeAttention)
    native.b=SimpleNamespace(main_stream=1,attention_tune=True)
    native.plans=OrderedDict();native.lib=SimpleNamespace(plan=plan);native.tuned={}
    native.metrics=dict(plan_tuning=[],plan_builds=0,plan_build_seconds=0.,max_workspace_bytes=0)
    result=native.plan((3,4096,16,4,96,12288,[]),True)
    assert result[1]==600*2**20 and len(calls)==2
    assert native.tuned[1234]==-1 and native.metrics['plan_tuning'][0]['selected'] is None


def test_single_attention_candidate_is_cached_and_runs_once_per_invocation():
    import ctypes as ct
    from types import SimpleNamespace
    from model.dense_cublas.vendor_attention import NativeAttention
    calls=[];native=NativeAttention.__new__(NativeAttention)
    native.b=SimpleNamespace(attention_tune=True);native.tuned={};native.metrics={'plan_tuning':[]}
    native.lib=SimpleNamespace(candidates=lambda plan:[dict(index=0,workspace_bytes=0)],
        select=lambda *args:calls.append('select'),run=lambda *args:calls.append('run'))
    for _ in range(2):native.run(ct.c_void_p(1234),[],None,(1,512,16,4,96,12288),False)
    assert calls==['select','run','run'] and len(native.metrics['plan_tuning'])==1
