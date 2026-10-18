import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from trainer import autotune, autotune_config as cfg


@pytest.mark.parametrize('settings', [
    {'args': {'learning_rate': .1}}, {'args': {'use_grad_checkpoint': 0}},
    {'args': {'dtype': 'float16'}}, {'env': {'PYTHONPATH': '/tmp'}},
    {'args': {'bucket_token_budget': float('nan')}},
    {'env': {'INSTINCT_MUON_WORKSPACE_MB': '-1'}},
    {'env': {'INSTINCT_GROUPED_MM_FWD_TILE': '17,32,64,4,3'}},
    {'env': {'INSTINCT_TRAINING_CPU_THREADS': '1.5'}},
])
def test_search_cannot_change_training_math_or_inject_environment(settings):
    with pytest.raises(ValueError):
        cfg.validate_settings(settings)


def test_neighbors_cover_batch_checkpoint_interaction_and_avoid_double_replay():
    settings = {'args': {'bucket_token_budget': 8192, 'use_grad_checkpoint': 1},
                'env': {'INSTINCT_MOE_INNER_CHECKPOINT': '0'}}
    grid = {'args.bucket_token_budget': [8192, 16384],
            'args.use_grad_checkpoint': [1, 2], 'env.INSTINCT_MOE_INNER_CHECKPOINT': ['0', '2']}
    candidates = list(autotune.neighbors(settings, grid))
    assert any(c['args']=={'bucket_token_budget':16384,'use_grad_checkpoint':2} for c in candidates)
    assert any(c['args']['bucket_token_budget']==16384 and c['env']['INSTINCT_MOE_INNER_CHECKPOINT']=='2' for c in candidates)
    assert not any(c['args']['use_grad_checkpoint']==2 and c['env']['INSTINCT_MOE_INNER_CHECKPOINT']=='2' for c in candidates)


def test_binary_budget_finds_memory_boundary_with_few_trials():
    state=autotune.BudgetBisection(range(1,65),8)
    state.observe(8,'ok')
    probes=[]
    while (value:=state.next()) is not None:
        probes.append(value)
        state.observe(value,'ok' if value <= 37 else 'oom')
    assert state.values[state.low]==37
    assert len(probes)<=8  # log2(range), plus nearby throughput comparisons.
    assert 36 in state.values and state.tested[state.values.index(36)]=='ok'


def test_compile_failure_does_not_prune_memory_range_and_oom_can_search_down():
    state=autotune.BudgetBisection([8,16,24,32],16)
    state.observe(16,'oom')
    assert state.next()==8
    state=autotune.BudgetBisection([8,16,24,32],8)
    state.observe(8,'ok');state.observe(24,'error')
    assert state.next()==32
    assert state.high==4


def test_nonmonotonic_memory_falls_back_to_measuring_remaining_candidates():
    state=autotune.BudgetBisection([8,16,24,32],8)
    state.observe(16,'oom');state.observe(24,'ok')
    assert state.nonmonotonic
    assert state.next()==32


def test_search_selects_binary_midpoint_instead_of_linear_budget_sweep(tmp_path):
    search=autotune.Search({},SimpleNamespace(),tmp_path,{})
    settings={'args':{'bucket_token_budget':16,'use_grad_checkpoint':1},
              'env':{'INSTINCT_MOE_INNER_CHECKPOINT':'0'}}
    best=dict(status='ok',settings=settings)
    search.record(best)
    candidate=search.next_candidate(best,{'args.bucket_token_budget':[8,12,16,20,24,32]})
    assert candidate['args']['bucket_token_budget']==24
    failure=dict(status='oom',settings=candidate)
    search.record(failure)
    search.observe_memory(failure,{'args.bucket_token_budget':[8,12,16,20,24,32]})
    assert search.next_candidate(best,{'args.bucket_token_budget':[8,12,16,20,24,32]})['args']['bucket_token_budget']==20


@pytest.mark.parametrize('target,published',[(99,True),(101,False)])
def test_optional_throughput_gate_controls_publication(tmp_path,monkeypatch,target,published):
    spec_path=tmp_path/'spec.json'
    cfg.write_json(spec_path,{'args':{'from_weight':'none','sequence_packing':1,
        'sequence_packing_mode':'bucket','data_path':'unused'},'audit':'unused','search':{}})
    monkeypatch.setattr(autotune,'audit_source',lambda *a:{'sha256':'source'})
    class FakeSearch(autotune.Search):
        def trial(self,settings,*,probe=False,confirm=False):
            if probe:
                return dict(status='probe',hardware={'vram_bytes':16*2**30},contract={})
            resolved=copy.deepcopy(settings)
            resolved['args']['bucket_token_budget']=16
            return dict(good_result(),settings=copy.deepcopy(settings),resolved_settings=resolved,
                        resolved_token_budget=16,directory=str(self.directory/'trial'))
    monkeypatch.setattr(autotune,'Search',FakeSearch)
    monkeypatch.setattr(sys,'argv',['autotune.py','--config',str(spec_path),'--output',str(tmp_path/'output'),
        '--max-trials','2','--target-tokens-per-second',str(target)])
    if published:
        autotune.main()
        assert len(list((tmp_path/'output').rglob('profile.json')))==1
    else:
        with pytest.raises(RuntimeError,match='throughput target'):
            autotune.main()
        assert not list((tmp_path/'output').rglob('profile.json'))
        assert len(list((tmp_path/'output').rglob('target_unmet.json')))==1


def good_result():
    return dict(status='ok', initial_weights_sha256='weights', validation_sha256='heldout',
                contract={'dtype':'bf16'}, losses=[5.]*40, validation_final=5., useful_tokens_per_second=100.)


@pytest.mark.parametrize('change', [
    {'status':'oom'}, {'losses':[float('nan')]*40}, {'validation_final':5.1},
    {'losses':[5.2]*20+[5.]*20}, {'initial_weights_sha256':'different'},
    {'validation_sha256':'training_data'}, {'contract':{'dtype':'fp8'}},
])
def test_faster_but_invalid_candidates_are_rejected(change):
    reference = good_result()
    candidate = dict(reference, useful_tokens_per_second=10000., **change)
    assert not cfg.quality_gate(reference,candidate)[0]


def test_authorized_batch_change_uses_nonregression_gate():
    reference = good_result()
    candidate = dict(reference, batch_sha256='newbatch', losses=[4.]*40, validation_final=4.)
    assert cfg.quality_gate(reference,candidate)[0]


def test_argument_vector_preserves_paths_and_has_no_shell_expansion():
    assert autotune.arguments({'data_path':'a b;$(not-a-command)', 'use_moe':True, 'use_wandb':False}) == [
        '--data_path','a b;$(not-a-command)','--use_moe','1']


def test_training_contract_survives_json_round_trip():
    contract=cfg.training_contract(SimpleNamespace())
    assert contract==json.loads(json.dumps(contract))


def test_restored_model_must_match_profile(tmp_path):
    path=tmp_path/'profile.json'
    cfg.write_json(path, {'contract':{'model':{'hidden_size':512,'id2label':{'0':'zero'}}}})
    args=SimpleNamespace(autotune_profile=str(path))
    config=SimpleNamespace(to_dict=lambda:{'hidden_size':512,'id2label':{0:'zero'},'use_grad_checkpoint':2})
    cfg.verify_profile_model(config,args)
    config.to_dict=lambda:{'hidden_size':768}
    with pytest.raises(ValueError,match='restored model'):
        cfg.verify_profile_model(config,args)


def profile_fixture(tmp_path, monkeypatch, resume=0):
    profile = dict(schema=cfg.SCHEMA, status='validated', hardware={'gpu':'test'}, contract={'model':'test'},
        source={'sha256':'corpus'}, settings={'args':{'bucket_token_budget':16384,'use_grad_checkpoint':1},
                                          'env':{'INSTINCT_TRAINING_CPU_THREADS':'2'}})
    path=tmp_path/'profile.json'; cfg.write_json(path,profile)
    args=SimpleNamespace(autotune_profile=str(path),autotune_trial=None,device='cuda:0',
        data_path='data.parquet',from_resume=resume,from_weight='none',bucket_token_budget=0,use_grad_checkpoint=1)
    monkeypatch.setattr(cfg,'hardware_fingerprint',lambda device:{'gpu':'test'})
    monkeypatch.setattr(cfg,'training_contract',lambda args:{'model':'test'})
    monkeypatch.setattr(cfg,'file_digest',lambda path:'corpus')
    monkeypatch.setattr(sys,'argv',['trainer/train_pretrain.py','--autotune_profile',str(path)])
    monkeypatch.delenv('INSTINCT_TRAINING_CPU_THREADS',raising=False)
    monkeypatch.setenv('WORLD_SIZE','1')
    return args,profile,path


@pytest.mark.parametrize('resume,expected',[(0,16384),(1,0)])
def test_profile_keeps_resume_budget_for_existing_cursor(tmp_path,monkeypatch,resume,expected):
    args,_,_=profile_fixture(tmp_path,monkeypatch,resume)
    cfg.apply_profile(args)
    assert args.bucket_token_budget==expected
    assert os.environ['INSTINCT_TRAINING_CPU_THREADS']=='2'


def test_profile_rejects_different_hardware_and_explicit_conflicts(tmp_path,monkeypatch):
    args,profile,path=profile_fixture(tmp_path,monkeypatch)
    profile['hardware']={'gpu':'other'};cfg.write_json(path,profile)
    with pytest.raises(ValueError,match='fingerprint'):
        cfg.apply_profile(args)
    profile['hardware']={'gpu':'test'};cfg.write_json(path,profile)
    monkeypatch.setattr(sys,'argv',['trainer/train_pretrain.py','--bucket_token_budget=8192'])
    args.bucket_token_budget=8192
    with pytest.raises(ValueError,match='conflicts'):
        cfg.apply_profile(args)


def test_profile_rejects_other_trainers(tmp_path,monkeypatch):
    args,_,_=profile_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'argv',['trainer/train_full_sft.py'])
    with pytest.raises(ValueError,match='single-GPU'):
        cfg.apply_profile(args)


def test_audit_requires_exact_bytes_and_provenance(tmp_path):
    source=tmp_path/'clean.jsonl';source.write_text('{"text":"hello"}\n')
    audit=tmp_path/'audit.json';cfg.write_json(audit,{'output_sha256':cfg.file_digest(source)})
    assert cfg.audit_source(source,audit)['sha256']==cfg.file_digest(source)
    source.write_text('{"text":"changed"}\n')
    with pytest.raises(ValueError,match='SHA-256'):
        cfg.audit_source(source,audit)


def test_timeout_reaps_owned_subprocess(tmp_path):
    code,timed_out=autotune.subprocess_trial([sys.executable,'-c','import time; time.sleep(30)'],
                                          dict(os.environ),tmp_path/'log',.2)
    assert timed_out and code==-1


def test_validation_parquet_starts_at_second_chunk(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from trainer.autotune_trial import validation_texts
    path=tmp_path/'data.parquet'
    pq.write_table(pa.table({'text':[str(i) for i in range(17)]}),path,row_group_size=5)
    plan={'identity':{'unit':'rows'},'chunks':[{'start':0},{'start':8}]}
    assert list(validation_texts(path,plan,4))==['8','9','10','11']


def test_prefetch_calibration_wait_does_not_consume_or_replace_dataset():
    from trainer.streaming_pretrain import ChunkedPackedEpochLoader
    loader=ChunkedPackedEpochLoader.__new__(ChunkedPackedEpochLoader)
    loader._pending_prefetch=None
    loader.wait_for_pending_prefetch()
    calls=[]
    pending=SimpleNamespace(result=lambda:calls.append('joined'))
    loader._pending_prefetch=pending
    loader.wait_for_pending_prefetch()
    assert calls==['joined'] and loader._pending_prefetch is pending


@pytest.mark.gpu
def test_grouped_tile_override_keeps_forward_and_weight_gradient(monkeypatch):
    import torch
    from model.grouped_mm_triton import ragged_mm,ragged_weight_grad
    torch.manual_seed(182)
    ends=torch.tensor([0,129,257],device='cuda',dtype=torch.int32)
    x=torch.randn(257,128,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(3,128,256,device='cuda',dtype=torch.bfloat16)
    dy=torch.randn(257,256,device='cuda',dtype=torch.bfloat16)
    for key in ('INSTINCT_GROUPED_MM_FWD_TILE','INSTINCT_GROUPED_MM_DW_TILE'):
        monkeypatch.delenv(key,raising=False)
    y=ragged_mm(x,w,ends); dw=ragged_weight_grad(x,dy,ends,3)
    monkeypatch.setenv('INSTINCT_GROUPED_MM_FWD_TILE','64,128,32,4,3')
    monkeypatch.setenv('INSTINCT_GROUPED_MM_DW_TILE','64,128,32,4,3')
    assert torch.equal(y,ragged_mm(x,w,ends))
    assert torch.equal(dw,ragged_weight_grad(x,dy,ends,3))
