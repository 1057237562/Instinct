"""Load a real profile through the actual WebUI command builder; stop before training."""
import ast
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import trainer.compile_cache
import datasets  # Windows DLL import order, before anything importing torch.
from trainer import training_profiler
from scripts.autotune_panel import ROOT, read_json, write_json
from scripts.moe_router_controls import migration_args


profile_path = ROOT / 'experiments/results/autotune_default/05f5185e84de9778d84cb44b/profile.json'
profile = read_json(profile_path)
template = read_json(ROOT / 'trainer/autotune_pretrain.json')['args']
directory = ROOT / 'experiments/results/config_webui_autotune/profile_load'
directory.mkdir(parents=True, exist_ok=True)
model_path = directory / 'model.json'
write_json(model_path, profile['contract']['model'])
state = dict(template, train_type='pretrain', activation_dtype=template['dtype'],
             data_path_pretrain=str(ROOT / template['data_path']),
             epochs_pretrain=template['epochs'], learning_rate_pretrain=template['learning_rate'],
             warmup_ratio_pretrain=template['warmup_ratio'],
             autotune_use_profile=True, autotune_profile_path=str(profile_path))
source = ROOT / 'scripts/config_webui.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
names = {'_build_training_command', '_packing_preprocess_workers', '_default_epochs',
         '_default_learning_rate', '_trainer_name'}
constants = {'_DEFAULT_EPOCHS', '_DEFAULT_LEARNING_RATES', '_TRAINER_NAME'}
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
         or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)]
namespace = dict(__file__=str(source), os=os, sys=sys, st=SimpleNamespace(session_state=state),
                 _resolve_cli_from_weight=lambda *a: None, moe_router_migration_args=migration_args)
exec(compile(ast.Module(nodes, type_ignores=[]), str(source), 'exec'), namespace)
command = namespace['_build_training_command']('pretrain', profile['contract']['model'],
                                             'webui_profile_load_disposable', False)['cmd']
command[command.index('--config_path') + 1] = str(model_path)
command.extend(['--save_dir', str(directory / 'weights'), '--pause_file', str(directory / 'pause')])
write_json(directory / 'launch.json', dict(argv=command, cwd=str(ROOT)))


class VerifyBeforeTraining:
    def __init__(self, args, **kwargs):
        for name, value in profile['settings']['args'].items():
            assert getattr(args, name) == value, (name, getattr(args, name), value)
        for name, value in profile['settings']['env'].items():
            assert os.environ[name] == value
        write_json(directory / 'verification.json', dict(status='verified',
            profile=str(profile_path), selected_settings=profile['settings'],
            actual_webui_builder=True, actual_model_optimizer_loaded=True,
            training_steps=0, formal_weights_saved=False))
        print('[WEBUI PROFILE VERIFIED] Actual model and optimizer loaded with selected settings; stopped before any training or saves.', flush=True)
        raise SystemExit(0)


training_profiler.TrainingProfiler = VerifyBeforeTraining
script_index = next(i for i, value in enumerate(command) if value.endswith('train_pretrain.py'))
sys.argv = command[script_index:]
runpy.run_path(str(ROOT / 'trainer/train_pretrain.py'), run_name='__main__')
