import json
from pathlib import Path
import subprocess
import sys

import pytest

from trainer import training_pipeline as pipeline
from scripts.pipeline_panel import snapshot


def test_snapshot_is_independent():
    model = dict(hidden_size=768, num_hidden_layers=8, use_moe=False)
    state = dict(data_path_pretrain='dataset/train.jsonl', learning_rate_pretrain=0.001)
    stage = snapshot(state, model, 'pretrain')
    model['hidden_size'] = 32
    state['learning_rate_pretrain'] = 0.1
    assert stage['model']['hidden_size'] == 768
    assert stage['args']['learning_rate'] == 0.001
    assert stage['args']['data_cache_max_gb'] == 5.0
    assert stage['args']['streaming_prefetch_chunks'] == 1


@pytest.mark.parametrize('failure', [False, True])
def test_pipeline_handoff_cleanup_and_failure(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    (tmp_path / 'out').mkdir()
    original_popen = subprocess.Popen
    calls = []

    def launch(command, **kwargs):
        args = dict(zip(command[3::2], command[4::2]))
        calls.append(args)
        # Execute a real child that writes a cache and a synthetic model artifact.
        program = (
            "import os,pathlib,sys; "
            "pathlib.Path(os.environ['HF_DATASETS_CACHE'],'test.arrow').touch(); "
            "pathlib.Path(sys.argv[1]).touch(); sys.exit(int(sys.argv[2]))"
        )
        output = tmp_path / 'out' / (args['--save_weight'] + '_768.pth')
        return original_popen([sys.executable, '-c', program, str(output), '1' if failure else '0'], **kwargs)

    monkeypatch.setattr(pipeline.subprocess, 'Popen', launch)
    plan = {'stages': [dict(name='pretrain', trainer='pretrain'),
                       dict(name='sft', trainer='full_sft', input_stage='pretrain')]}
    run_dir = tmp_path / 'run'
    assert pipeline.run(plan, run_dir) == int(failure)
    status = json.loads((run_dir / 'status.json').read_text())
    assert status['status'] == ('failed' if failure else 'success')
    assert list((tmp_path / '.cache' / 'pipeline').iterdir()) == []
    if failure:
        assert len(calls) == 1
    else:
        assert calls[1]['--from_weight'] == status['stages'][0]['output']
        assert Path(status['stages'][0]['output']).exists()


def test_cleanup_rejects_external_directory(tmp_path):
    with pytest.raises(ValueError):
        pipeline.cleanup_cache(tmp_path, tmp_path)


def test_dataset_cache_isolated_but_compile_cache_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCHINDUCTOR_CACHE_DIR', 'shared-compiler-cache')
    env = pipeline.cache_environment(tmp_path)
    assert env['HF_DATASETS_CACHE'] == str(tmp_path / 'datasets')
    assert env['TORCHINDUCTOR_CACHE_DIR'] == 'shared-compiler-cache'


def test_invalid_dependency_rejected():
    with pytest.raises(ValueError):
        pipeline.validate({'stages': [dict(name='sft', trainer='full_sft', input_stage='missing')]})
