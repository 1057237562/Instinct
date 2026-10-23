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


@pytest.mark.parametrize('model,choices', [
    ({}, [1, 2, 3, 4]), ({'use_moe': 1}, [1, 2]),
    ({'residual_type': 'mhc'}, [1, 2]), ({'residual_type': 'attnres'}, [1, 2]),
    ({'model_architecture': 'linear'}, [1, 2]),
    ({'model_architecture': 'recurrsive_instinct'}, [1, 2]),
])
def test_checkpoint_search_respects_topology(model, choices):
    grid = autotune.search_space({'args': model}, {'resolved_token_budget': 2048})
    assert grid['args.use_grad_checkpoint'] == choices
    assert ('env.INSTINCT_MOE_INNER_CHECKPOINT' in grid) == bool(model.get('use_moe'))
    if 3 not in choices:
        with pytest.raises(ValueError, match='standard Dense'):
            autotune.search_space({'args': model, 'search': {'args.use_grad_checkpoint': [2, 3]}},
                                  {'resolved_token_budget': 2048})


@pytest.mark.parametrize('overrides', [
    {}, {'use_moe': 1}, {'residual_type': 'standard'},
    {'model_architecture': 'standard'}, {'use_recurrsive': 1},
])
def test_checkpoint_topology_resolution_matches_trainer(tmp_path, overrides):
    from trainer.trainer_utils import config_from_args
    path = tmp_path / 'model.json'
    cfg.write_json(path, {'use_moe': True, 'residual_type': 'mhc', 'model_architecture': 'linear'})
    args = dict(config_path=str(path), **overrides)
    actual = config_from_args(SimpleNamespace(**args)).to_dict()
    resolved = cfg.tuning_model(args)
    assert resolved == {key: actual[key] for key in resolved}


@pytest.mark.parametrize('mode', [3, 4])
def test_selective_neighbors_and_memory_bracket_are_independent(tmp_path, mode):
    settings = {'args': {'bucket_token_budget': 2048, 'use_grad_checkpoint': 2}, 'env': {}}
    grid = {'args.bucket_token_budget': [1024, 2048, 4096],
            'args.use_grad_checkpoint': [2, mode], 'env.INSTINCT_MOE_INNER_CHECKPOINT': ['0', '2']}
    candidates = list(autotune.neighbors(settings, grid))
    assert any(c['args'] == {'bucket_token_budget': 4096, 'use_grad_checkpoint': mode} for c in candidates)
    assert all(c['env'].get('INSTINCT_MOE_INNER_CHECKPOINT', '0') == '0'
               for c in candidates if c['args']['use_grad_checkpoint'] == mode)
    search = autotune.Search({}, SimpleNamespace(), tmp_path, {})
    selective = cfg.changed(settings, 'args', 'use_grad_checkpoint', mode)
    key2, bracket2 = search.memory_state(settings, grid)
    key3, bracket3 = search.memory_state(selective, grid)
    bracket3.observe(2048, 'oom')
    assert key2 != key3
    assert bracket2.high == len(bracket2.values)


@pytest.mark.parametrize('mode', [3, 4])
def test_invalid_selective_baseline_is_rejected_before_probe_or_corpus_read(tmp_path, monkeypatch, mode):
    spec = tmp_path / 'spec.json'
    cfg.write_json(spec, {'args': {'from_weight': 'none', 'sequence_packing': 1,
        'sequence_packing_mode': 'bucket', 'use_moe': 1, 'use_grad_checkpoint': mode}})
    monkeypatch.setattr(autotune, 'audit_source', lambda *a: pytest.fail('must fail before corpus read'))
    monkeypatch.setattr(autotune, 'gpu_snapshot', lambda: pytest.fail('must fail before GPU access'))
    monkeypatch.setattr(sys, 'argv', ['autotune.py', '--config', str(spec)])
    with pytest.raises(ValueError, match='standard Dense'):
        autotune.main()


@pytest.mark.parametrize('use_moe', [0, 1])
@pytest.mark.parametrize('mode', [3, 4])
def test_selective_trial_preflight_checks_topology_without_cuda(tmp_path, monkeypatch, use_moe, mode):
    from trainer import autotune_trial
    from trainer.trainer_cli import build_trainer_parser
    request = tmp_path / 'request.json'
    source = tmp_path / 'source.jsonl'
    source.write_text('{}\n')
    cfg.write_json(request, dict(schema=1, settings={'args': {'use_grad_checkpoint': mode}, 'env': {}},
        warmup=2, steps=2, validation_examples=1,
        source={'size': source.stat().st_size, 'mtime_ns': source.stat().st_mtime_ns}))
    args = build_trainer_parser('test').parse_args(['--use_grad_checkpoint', str(mode), '--use_moe', str(use_moe)])
    args.autotune_trial = str(request)
    args.from_resume = 0
    args.from_weight = 'none'
    args.sequence_packing = 1
    args.sequence_packing_mode = 'bucket'
    args.dataset_streaming = 'on'
    args.cache_build_mode = 'inline'
    args.save_dir, args.pause_file = str(tmp_path / 'weights'), str(tmp_path / 'pause')
    args.save_interval, args.data_path = 100, str(source)
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setattr(autotune_trial, 'training_contract', lambda _: {'model': 'fixture'})
    calls = []
    monkeypatch.setattr(autotune_trial, 'hardware_fingerprint', lambda _: calls.append('hardware') or {})
    if use_moe:
        with pytest.raises(ValueError, match='standard Dense'):
            autotune_trial.prepare_trial(args)
        assert not calls
    else:
        autotune_trial.prepare_trial(args)
        assert calls == ['hardware']


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
        'sequence_packing_mode':'bucket','data_path':'unused'},'search':{}})
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


@pytest.mark.parametrize('name,value', [('INSTINCT_MUON_MANTISSA','1'),
                                       ('INSTINCT_MUON_MOMENTUM_DTYPE','bf16')])
def test_training_contract_detects_muon_mantissa_change(monkeypatch, name, value):
    monkeypatch.delenv(name, raising=False)
    before = cfg.training_contract(SimpleNamespace())
    monkeypatch.setenv(name, value)
    after = cfg.training_contract(SimpleNamespace())
    assert before != after
    assert after['context_env'][name] == value


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


@pytest.mark.parametrize('mode', [3, 4])
def test_selective_profile_applies_and_keeps_resume_cursor_budget(tmp_path, monkeypatch, mode):
    args, profile, path = profile_fixture(tmp_path, monkeypatch, resume=1)
    profile['settings']['args']['use_grad_checkpoint'] = mode
    cfg.write_json(path, profile)
    cfg.apply_profile(args)
    assert args.use_grad_checkpoint == mode
    assert args.bucket_token_budget == 0
    args.use_moe = 1
    with pytest.raises(ValueError, match='standard Dense'):
        cfg.apply_profile(args)


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


def test_profile_rejects_changed_corpus_without_report_metadata(tmp_path, monkeypatch):
    args, _, _ = profile_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(cfg, 'file_digest', lambda path: 'changed-corpus')
    with pytest.raises(ValueError, match='corpus changed'):
        cfg.apply_profile(args)


@pytest.mark.parametrize('mismatch,expected', [
    (None, None), ('hardware', 'fingerprint'), ('contract', 'configuration changed'),
    ('source', 'corpus changed'),
])
def test_manual_profile_retains_fingerprint_guards(tmp_path, monkeypatch, mismatch, expected):
    args, profile, path = profile_fixture(tmp_path, monkeypatch)
    profile.update(status='manual', validation={'performed': False})
    profile['settings']['args']['bucket_token_budget'] = 5344
    profile['settings']['env']['INSTINCT_MUON_WORKSPACE_MB'] = '256'
    cfg.write_json(path, profile)
    monkeypatch.setenv('INSTINCT_MUON_WORKSPACE_MB', '256')
    if mismatch == 'hardware':
        monkeypatch.setattr(cfg, 'hardware_fingerprint', lambda _: {'gpu': 'different'})
    elif mismatch == 'contract':
        monkeypatch.setattr(cfg, 'training_contract', lambda _: {'model': 'different'})
    elif mismatch == 'source':
        monkeypatch.setattr(cfg, 'file_digest', lambda _: 'different')
    if expected:
        with pytest.raises(ValueError, match=expected):
            cfg.apply_profile(args)
    else:
        cfg.apply_profile(args)
        assert args.bucket_token_budget == 5344
        assert os.environ['INSTINCT_MUON_WORKSPACE_MB'] == '256'
        assert cfg.read_json(path)['status'] == 'manual'


def test_explicit_audit_requires_exact_bytes(tmp_path):
    source=tmp_path/'clean.jsonl';source.write_text('{"text":"hello"}\n')
    audit=tmp_path/'audit.json';cfg.write_json(audit,{'output_sha256':cfg.file_digest(source)})
    assert cfg.audit_source(source,audit)['sha256']==cfg.file_digest(source)
    source.write_text('{"text":"changed"}\n')
    with pytest.raises(ValueError,match='SHA-256'):
        cfg.audit_source(source,audit)


@pytest.mark.parametrize('suffix', ['.jsonl', '.parquet'])
def test_source_fingerprint_without_reports_still_tracks_changed_bytes(tmp_path, suffix):
    source = tmp_path / ('source' + suffix)
    source.write_bytes(b'first corpus')
    original = cfg.audit_source(source)
    assert original['sha256'] == cfg.file_digest(source)
    assert original['size'] == len(b'first corpus')
    assert 'audit_sha256' not in original and 'provenance_sha256' not in original
    source.write_bytes(b'changed corpus')
    assert cfg.audit_source(source)['sha256'] != original['sha256']


def test_optional_compiler_report_accepts_nested_output_hash(tmp_path):
    source = tmp_path / 'source.parquet'
    source.write_bytes(b'corpus')
    report = tmp_path / 'compiler.report.json'
    cfg.write_json(report, {'output': {'sha256': cfg.file_digest(source)}})
    result = cfg.audit_source(source, report)
    assert result['audit_sha256'] == cfg.file_digest(report)
    cfg.write_json(report, {'output': {'sha256': 'wrong'}})
    with pytest.raises(ValueError, match='SHA-256'):
        cfg.audit_source(source, report)


@pytest.mark.parametrize('interrupted', [False, True])
def test_subprocess_wait_has_no_deadline_and_interrupt_still_reaps_tree(tmp_path, monkeypatch, interrupted):
    calls = []
    def wait():  # A timeout argument would fail this contract.
        calls.append('wait')
        if interrupted:
            raise KeyboardInterrupt
        return 7
    process = SimpleNamespace(wait=wait)
    monkeypatch.setattr(autotune.subprocess, 'Popen', lambda *a, **kw: process)
    monkeypatch.setattr(autotune, 'kill_trial', lambda owned: calls.append(owned))
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            autotune.subprocess_trial(['trainer'], {}, tmp_path/'log')
        assert calls == ['wait', process]
    else:
        assert autotune.subprocess_trial(['trainer'], {}, tmp_path/'log') == 7
        assert calls == ['wait']


@pytest.mark.parametrize('role', ['candidate', 'confirm', 'probe'])
def test_long_trials_complete_and_old_timeouts_are_retried(tmp_path, monkeypatch, role):
    options = SimpleNamespace(warmup=20, steps=60, validation_examples=16, peak_tflops=None,
                              refresh=False, worker_launcher=None)
    clock = iter(range(0, 100000, 7200))
    monkeypatch.setattr(autotune.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(autotune, 'gpu_snapshot', lambda: None)
    calls = []
    def complete_trial(command, environment, log):
        calls.append(command)
        cfg.write_json(log.parent/'result.json', dict(status='probe' if role == 'probe' else 'ok',
                                                     useful_tokens_per_second=100.))
        return 0
    monkeypatch.setattr(autotune, 'subprocess_trial', complete_trial)
    search = autotune.Search({'args': {}}, options, tmp_path, {})
    settings = {'args': {'bucket_token_budget': 1024}, 'env': {}}
    flags = {role: True} if role != 'candidate' else {}
    result = search.trial(settings, **flags)
    assert result['status'] == ('probe' if role == 'probe' else 'ok')
    assert result['startup_and_training_seconds'] == 7200
    assert 'effective_timeout_seconds' not in result
    # Reproduce a result cached by the old controller with the same request key.
    cfg.write_json(Path(result['directory'])/'result.json', dict(result, status='timeout'))
    retried = search.trial(settings, **flags)
    assert retried['status'] == result['status']
    assert len(calls) == 2
    search.trial(settings, **flags)
    assert len(calls) == (3 if role == 'probe' else 2)


@pytest.mark.parametrize('status,retries,expected', [
    ('error', 1, 'ValueError: new shape reached timed window'),
    ('process_error', 1, 'exit code 7'),
    ('oom', 2, 'lower initial bucket_token_budget'),
    ('driver_budget_exceeded', 2, 'lower initial bucket_token_budget'),
])
def test_baseline_failure_reports_cause_and_only_reduces_memory_failures(
        tmp_path, monkeypatch, status, retries, expected):
    spec_path = tmp_path / 'spec.json'
    cfg.write_json(spec_path, {'args': {'from_weight': 'none', 'sequence_packing': 1,
        'sequence_packing_mode': 'bucket', 'data_path': 'unused', 'bucket_token_budget': 1024}})
    monkeypatch.setattr(autotune, 'audit_source', lambda *a: {})
    budgets = []
    class FakeSearch(autotune.Search):
        def trial(self, settings, *, probe=False, confirm=False):
            if probe:
                return dict(status='probe', hardware={'vram_bytes': 16*2**30}, contract={})
            assert not confirm
            budgets.append(settings['args']['bucket_token_budget'])
            result = dict(status=status, settings=settings, resolved_token_budget=budgets[-1],
                          directory=str(self.directory / 'trial'), exit_code=7)
            if status == 'error':
                result['error'] = 'Traceback\nValueError: new shape reached timed window\n'
            return result
    monkeypatch.setattr(autotune, 'Search', FakeSearch)
    monkeypatch.setattr(sys, 'argv', ['autotune.py', '--config', str(spec_path),
        '--output', str(tmp_path / 'output'), '--max-trials', '3'])
    with pytest.raises(RuntimeError, match=expected) as exc:
        autotune.main()
    assert budgets == [1024, 512][:retries]
    assert 'trial.log' in str(exc.value)
    if retries == 1:
        assert 'lower initial bucket_token_budget' not in str(exc.value)
    assert not list((tmp_path / 'output').rglob('profile.json'))


def test_search_finishes_trial_count_and_confirmation_despite_legacy_time_limits(tmp_path, monkeypatch, capsys):
    spec_path = tmp_path/'spec.json'
    cfg.write_json(spec_path, {'args': {'from_weight': 'none', 'sequence_packing': 1,
        'sequence_packing_mode': 'bucket', 'data_path': 'unused', 'bucket_token_budget': 16},
        'search': {'args.bucket_token_budget': [8, 16, 32, 64]}})
    monkeypatch.setattr(autotune, 'audit_source', lambda *a: {})
    elapsed, roles = [0], []
    monkeypatch.setattr(autotune.time, 'monotonic', lambda: elapsed[0])
    class FakeSearch(autotune.Search):
        def trial(self, settings, *, probe=False, confirm=False):
            elapsed[0] += 7200
            if probe:
                return dict(status='probe', hardware={'vram_bytes': 16*2**30}, contract={})
            roles.append('confirm' if confirm else 'trial')
            return dict(good_result(), settings=copy.deepcopy(settings), resolved_settings=copy.deepcopy(settings),
                        resolved_token_budget=settings['args']['bucket_token_budget'],
                        directory=str(self.directory/'trial'))
    monkeypatch.setattr(autotune, 'Search', FakeSearch)
    monkeypatch.setattr(sys, 'argv', ['autotune.py', '--config', str(spec_path),
        '--output', str(tmp_path/'output'), '--max-trials', '4', '--time-limit', '1',
        '--trial-timeout', '1', '--baseline-timeout', '1'])
    autotune.main()
    assert roles == ['trial', 'trial', 'trial', 'confirm']
    assert len(list((tmp_path/'output').rglob('profile.json'))) == 1
    assert 'legacy timeout arguments are ignored' in capsys.readouterr().err


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
