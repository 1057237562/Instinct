"""Offline parity checks against config_webui's real model/command builders."""
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get('RELAY_TEST_BINARY', ROOT / 'data_relay/target/release/data_relay.exe'))
SOURCE = ROOT / 'scripts/config_webui.py'


def reference_builders(state):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    names = {'compute_intermediate_size', 'compute_head_dim', '_parse_longrope_factors', 'build_config_dict',
             '_build_training_command', '_packing_preprocess_workers', '_default_epochs', '_default_learning_rate', '_trainer_name'}
    constants = {'_DEFAULT_EPOCHS', '_DEFAULT_LEARNING_RATES', '_TRAINER_NAME'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
             or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)]
    scope = {'st': SimpleNamespace(session_state=state), 'math': math, 'json': json, 'os': os, 'sys': sys,
             '__file__': str(SOURCE), '_resolve_cli_from_weight': lambda *args: None,
             'moe_router_migration_args': lambda *args, **kwargs: {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope


@unittest.skipUnless(BINARY.is_file(), 'Build the relay release binary before offline CLI parity checks')
class PanelCacheTests(unittest.TestCase):
    def cached(self, state, stage='pretrain'):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'trainer').mkdir(); (root / 'data_relay').mkdir()
            (root / 'trainer/webui_state.json').write_text(json.dumps(state), encoding='utf-8')
            config = root / 'data_relay/config.toml'
            config.write_text('[server]\napi_token="test-panel-token"\ndata_dir=' + json.dumps(str(root/'runtime')) +
                              '\n[autodl]\nconsole_token="offline-test-token"\n', encoding='utf-8')
            request = root / 'request.json'
            request.write_text(json.dumps({'trainer': stage}), encoding='utf-8')
            output = subprocess.check_output([str(BINARY), '--config', str(config), 'training', '--training-action',
                                              'panel_cache', '--request-json', str(request)], text=True, encoding='utf-8')
            return json.loads(output)

    def test_model_snapshot_matches_config_webui_for_supported_architectures(self):
        base = {'hidden_size': 512, 'num_hidden_layers': 32, 'num_attention_heads': 16, 'use_moe': True,
                'num_experts': 8, 'moe_intermediate_size': 1664, 'rope_scaling_type': 'longrope',
                'longrope_short_factors': json.dumps([1.0]*16), 'longrope_long_factors': json.dumps([8.0]*16)}
        for architecture, residual in [('standard','standard'), ('linear','mhc'), ('recurrsive_instinct','attnres')]:
            with self.subTest(architecture=architecture, residual=residual):
                state = dict(base, model_architecture=architecture, residual_type=residual,
                             loop_iters=12, early_exit_enabled=True, early_exit_layers=[4, 8])
                self.assertEqual(self.cached(state)['model_config'], reference_builders(state)['build_config_dict']())

    def test_training_arguments_match_current_panel_command_and_stage_overrides(self):
        state = {'hidden_size':512, 'num_hidden_layers':32, 'num_attention_heads':16, 'use_moe':True,
                 'train_type':'pretrain', 'batch_size':1, 'accumulation_steps':1, 'optimizer':'muon',
                 'activation_dtype':'bfloat16', 'param_dtype':'fp32', 'fp8_training':'tensorwise',
                 'compile_mode':'max-autotune-no-cudagraphs', 'use_compile':True, 'use_grad_checkpoint':1,
                 'epochs_pretrain':3, 'learning_rate_pretrain':0.0005, 'learning_rate_full_sft':1e-9,
                 'max_seq_len':4096, 'packing_num_proc':4, 'streaming_prefetch_chunks':False,
                 'bucket_token_budget':8192, 'dataset_streaming':'on', 'cache_build_mode':'inline',
                 'pipeline_stages':{'pretrain':{'args':{'batch_size':8, 'param_dtype':'fp16', 'compile_mode':'default'}}}}
        ignored = {'config_path', 'hidden_size', 'num_hidden_layers', 'use_moe', 'from_weight', 'save_weight', 'from_resume'}
        for stage in ['pretrain', 'cpt', 'full_sft', 'lora', 'dpo', 'ppo', 'grpo', 'agent', 'distillation']:
            with self.subTest(stage=stage):
                scope = reference_builders(state)
                command = scope['_build_training_command'](stage, scope['build_config_dict'](), 'fixture', False)['cmd']
                cached = self.cached(state, stage)
                for index, argument in enumerate(command):
                    if argument.startswith('--') and argument[2:] not in ignored | {'lora_name'}:
                        key = argument[2:]
                        value = cached['args'][key]
                        value = str(int(value)) if isinstance(value, bool) else str(value)
                        self.assertEqual(value, command[index+1], key)
                self.assertEqual(cached['args']['param_dtype'], 'fp32')
                self.assertNotIn('from_resume', cached['args'])

    def test_internal_ui_changes_do_not_change_training_fingerprint(self):
        state = {'hidden_size':512, 'num_hidden_layers':32, 'train_type':'pretrain', 'batch_size':1}
        original = self.cached(state)
        changed = self.cached(dict(state, btn_start=True, train_log_path='changed', auto_refresh_log=True,
                                   pipeline_stages={'pretrain':{'args':{'batch_size':5}}}))
        self.assertEqual(original['sha256'], changed['sha256'])
        self.assertNotEqual(original['state_sha256'], changed['state_sha256'])


if __name__ == '__main__':
    unittest.main()
