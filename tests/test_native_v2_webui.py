"""Isolated actual UI load plus native recipe/CLI parity; never starts CUDA."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from scripts.native_v2_recipe import command,model_config,recipe,ui_values
from tests.test_cublas_webui_contract import exported_config


def test_native_v2_recipe_export_and_cli_match():
    from trainer.train_cublas import parser,read_config
    cfg=exported_config(model_config()|ui_values())
    for key,value in model_config().items():assert cfg[key]==value
    args=parser().parse_args(command('example.jsonl')[3:])
    checked=read_config(args)
    for key,value in model_config().items():assert checked[key]==value
    for key,value in recipe().items():
        if key!='config_path':assert getattr(args,key)==value


def test_native_v2_actual_app_load_and_command(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'scripts').mkdir();(tmp_path/'trainer/recipes').mkdir(parents=True)
    for name in ('scripts/config_webui.py','scripts/native_v2_recipe.py',
                 'trainer/config_instinct_v2.json','trainer/recipes/instinct_v2_native.json','trainer/train_cublas.py'):
        shutil.copyfile(root/name,tmp_path/name)
    # Point the shared loader to the isolated test tree as well.
    import scripts.native_v2_recipe as catalog
    monkeypatch.setattr(catalog,'ROOT',tmp_path)
    captured=[]
    class Process:
        pid=123456789
        def poll(self):return 0
    def launch(cmd,*args,**kwargs):
        captured.append(cmd);return Process()
    app=AppTest.from_file(str(tmp_path/'scripts/config_webui.py'),default_timeout=30).run()
    assert not app.exception
    app.radio(key='_preset_radio').set_value('Instinct V2 (native ckpt4)').run()
    assert not app.exception
    assert app.session_state['hidden_size']==1536
    assert app.session_state['intermediate_size']==5504
    assert app.session_state['runtime_backend']=='cublas'
    assert app.session_state['use_grad_checkpoint']==4
    monkeypatch.setattr(subprocess,'Popen',launch)
    app.button(key='btn_start_train').click().run()
    assert not app.exception and captured
    cmd=captured[-1]
    for flag,value in (('--muon_mantissa','1'),('--muon_momentum_dtype','bf16'),
                       ('--muon_workspace_mb','512'),('--use_grad_checkpoint','4'),
                       ('--fp8_filter','auto'),('--allocator_backend','async'),
                       ('--learning_rate','0.0003'),('--weight_decay','0.1')):
        assert cmd[cmd.index(flag)+1]==value
    assert '--packed_attention' in cmd
    assert '--fused_ops' in cmd
    saved=json.loads((tmp_path/'trainer/config_pretrain.json').read_text(encoding='utf-8'))
    assert saved['intermediate_size']==5504 and saved['num_hidden_layers']==32
