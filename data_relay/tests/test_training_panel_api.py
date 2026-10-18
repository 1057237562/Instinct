"""Isolated local HTTP tests: never connects to AutoDL or starts real training."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
from pathlib import PurePosixPath
from test_training_panel_cache import reference_builders
from test_training_remote import training

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get('RELAY_TEST_BINARY', ROOT/'data_relay/target/release/data_relay.exe'))


@unittest.skipUnless(BINARY.is_file(), 'Build the relay first')
class PanelApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        cls.url = f'http://127.0.0.1:{port}'
        config = cls.root/'config.toml'
        config.write_text(f'[server]\nlisten="127.0.0.1:{port}"\napi_token="test"\ndata_dir="runtime"\n', encoding='utf-8')
        cls.proc = subprocess.Popen([str(BINARY), '--config', str(config), 'serve'], cwd=cls.root,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try: cls.call('/api/health'); break
            except OSError: time.sleep(0.1)
        else: raise RuntimeError('isolated relay did not start')

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate(); cls.proc.wait(timeout=10); cls.tmp.cleanup()

    @classmethod
    def call(cls, path, body=None, method=None):
        req = Request(cls.url+path, data=json.dumps(body).encode() if body is not None else None,
                      method=method, headers={'Authorization':'test','Content-Type':'application/json'})
        with urlopen(req, timeout=10) as response: return json.load(response)

    def test_all_stages_match_real_config_webui_and_remote_argv(self):
        panel = self.call('/api/training/panel')
        for stage in panel['schema']['stages']:
            with self.subTest(stage=stage):
                state = dict(panel['schema']['presets']['Instinct V1 MoE'], train_type=stage,
                             dataset_streaming='on', cache_build_mode='inline', bucket_token_budget=9216,
                             packing_num_proc=3, **{f'learning_rate_{stage}':0.000123})
                result = self.call('/api/training/preview',dict(trainer=stage, panel_state=state, nproc=2,
                    args={'data_path':'dataset/example.parquet', 'from_weight':'base'}, config_path='__panel_cache__'))
                request = result['request']; scope = reference_builders(state)
                self.assertEqual(request['model_config'],scope['build_config_dict']())
                expected = scope['_build_training_command'](stage,scope['build_config_dict'](),'fixture',False)['cmd']
                ignored = {'config_path','hidden_size','num_hidden_layers','use_moe','from_weight','save_weight','lora_name','from_resume'}
                for i,arg in enumerate(expected):
                    if arg.startswith('--') and arg[2:] not in ignored:
                        actual = request['args'][arg[2:]]
                        self.assertEqual(str(int(actual)) if isinstance(actual,bool) else str(actual), expected[i+1],arg)
                # Compare the offline preview to the actual supervisor command builder.
                args = request['args'].copy();model = request['model_config']
                for key in ['hidden_size','num_hidden_layers','use_moe','use_grad_checkpoint','param_dtype','kv_cache_dtype']:
                    if key in model: args.setdefault(key,model[key])
                if stage=='pretrain': args['from_weight']='none'
                if stage=='distillation':
                    for role in ['student','teacher']:
                        for suffix,source in [('hidden_size','hidden_size'),('num_layers','num_hidden_layers'),('use_moe','use_moe')]:args.setdefault(role+'_'+suffix,model[source])
                    args.setdefault('from_student_weight','none');args['from_teacher_weight']=args.pop('from_weight')
                root=PurePosixPath('/root/autodl-tmp');identifier='r_'+'a'*32
                args['data_path']=str(root/'dataset/example.parquet')
                job=dict(id=identifier, trainer=stage, args=args,model_config=model,settings={'python':'/root/miniconda3/bin/python'},
                         nproc=2,master_port=12345,save_weight='fixture',config_path=str(root/'.relay_training'/identifier/'model_config.json'))
                with patch.object(training,'run_dir',return_value=root/'.relay_training'/identifier):
                    argv=training.command(root,job)
                preview=[s.replace('<job-id>',identifier).replace('<run-prefix>','fixture').replace('<port>','12345') for s in result['argv']]
                # Python Path uses host separators; relay preview always targets Linux.
                self.assertEqual([s.replace('\\','/') for s in argv],preview)

    def test_save_restores_stage_specific_values_and_no_review_gate(self):
        doc=self.call('/api/training/panel')['document']
        doc['state'].update(epochs_cpt=3, learning_rate_cpt=0.00001, epochs_full_sft=7)
        doc['extras']={'grpo':{'loss_type':'grpo','num_generations':8}}
        doc['launch']={'cpt':{'data_path':'dataset/cpt.parquet'}}
        self.call('/api/training/panel',doc,'PUT')
        self.assertEqual(self.call('/api/training/panel')['document'],doc)
        saved=json.loads((self.root/'runtime/training/panel.json').read_text(encoding='utf-8'))
        self.assertEqual(saved,doc)
        self.assertNotIn('identity_reviewed', self.call('/api/training/preview',dict(trainer='cpt',panel_state=doc['state'],args={}))['request'])

    def test_invalid_parameters_are_rejected_before_any_network_dispatch(self):
        state=self.call('/api/training/panel')['schema']['presets']['Instinct V1 MoE']
        for change in [{'batch_size':-1},{'num_key_value_heads':3},{'num_experts_per_tok':100},{'optimizer':'bogus'}]:
            with self.subTest(change=change), self.assertRaises(HTTPError) as error:
                self.call('/api/training/jobs',dict(trainer='pretrain',panel_state=state|change,args={'data_path':'dataset/x.parquet'}))
            self.assertEqual(error.exception.code,400)
        for extra in [{'save_dir':'x'},{'loss_type':'unknown'},{'from_resume':1}]:
            with self.subTest(extra=extra), self.assertRaises(HTTPError):
                self.call('/api/training/preview',dict(trainer='grpo',panel_state=state,args=extra))
        self.assertEqual(self.call('/api/training/operations')['operations'],[])


if __name__=='__main__': unittest.main()
