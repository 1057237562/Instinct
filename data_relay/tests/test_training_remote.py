import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).parents[1] / 'src' / 'training_remote.py'
spec = importlib.util.spec_from_file_location('training_remote', SOURCE)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)

FAKE_TRAINER = r'''
import argparse,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--save_weight');p.add_argument('--pause_file');p.add_argument('--from_resume',type=int)
p.add_argument('--config_path');p.add_argument('--hidden_size',type=int)
args,rest=p.parse_known_args()
checkpoint=Path('checkpoints')/(args.save_weight+'_'+str(args.hidden_size)+'_moe_resume.pth')
if args.from_resume and not checkpoint.exists(): sys.exit(9)
time.sleep(0.5)
Path(args.pause_file).unlink(missing_ok=True)
print('RESUME='+str(args.from_resume),flush=True)
for step in range(100):
 if Path(args.pause_file).exists():
  checkpoint.parent.mkdir(exist_ok=True);checkpoint.write_bytes(b'resume fixture')
  checkpoint.with_name(checkpoint.name.replace('_resume.pth','.json')).write_text(Path(args.config_path).read_text())
  print('SAVED_PAUSE',flush=True);sys.exit(42)
 print('STEP='+str(step),flush=True);time.sleep(0.03)
print('FINISHED',flush=True)
'''


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        for folder in ['model', 'trainer', 'dataset', 'out', 'checkpoints', '.relay_training']:
            (self.root / folder).mkdir()
        (self.root / 'model/model_instinct.py').write_text('# fixture')
        (self.root / 'trainer/train_pretrain.py').write_text(FAKE_TRAINER)
        (self.root / 'trainer/train_full_sft.py').write_text(FAKE_TRAINER)
        (self.root / 'trainer/train_distillation.py').write_text(FAKE_TRAINER)
        shutil.copy2(SOURCE, self.root / '.relay_training/manager.py')
        (self.root / 'trainer/config_instinct_v1_moe.json').write_text(json.dumps(
            {'hidden_size':512,'num_hidden_layers':32,'use_moe':True,'use_grad_checkpoint':1}))
        data = self.root / 'dataset/pretrain_identity_clean.jsonl'
        data.write_text('{"text":"reviewed training example"}\n')
        (self.root / 'dataset/clean.report.json').write_text(json.dumps({'output_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}))
        self.opts = {'repo_root':str(self.root),'python':sys.executable,'instance_uuid':'fixture',
                     'gpu_devices':'','min_free_gb':0,'cache_max_gb':1,'shutdown_after_train':False}
        self.request = {'trainer':'pretrain','args':{'data_path':'dataset/pretrain_identity_clean.jsonl'},
                        'identity_reviewed':True,'audit_path':'dataset/clean.report.json','shutdown_after_train':False}

    def prepare(self):
        return training.prepare(self.root, self.opts, self.request)

    def test_model_config_dimensions_are_forwarded_to_cli(self):
        _, args, _, _, _, _, count = self.prepare()
        self.assertEqual((args['hidden_size'],args['num_hidden_layers'],args['use_moe'],count),(512,32,True,1))
        self.assertEqual(args['from_weight'],'none')

    def test_pending_code_sync_blocks_training_but_allows_recovery_guard(self):
        (self.root / '.relay_code_sync').mkdir()
        (self.root / '.relay_code_sync/ACTIVE').write_text('cs_' + 'a' * 32)
        with patch.object(training, 'unmanaged_trainers', return_value=[]):
            with self.assertRaisesRegex(ValueError, '同步未完成'):
                training.idle(self.root)
            training.idle(self.root, allow_code_sync=True)
            self.assertTrue(training.dispatch(dict(self.opts, action='assert_idle', allow_code_sync=True))['idle'])

    def test_identity_review_is_not_required_and_optional_hash_report_is_checked(self):
        self.request['identity_reviewed']=False
        self.prepare()
        (self.root / 'dataset/pretrain_identity_clean.jsonl').write_text('changed')
        with self.assertRaises(ValueError): self.prepare()
        del self.request['audit_path']
        self.assertIsNone(self.prepare()[4]['path'])
        (self.root / 'dataset/pretrain_identity_clean.jsonl').unlink()
        with self.assertRaises(ValueError): self.prepare()

    def test_compiler_report_shape_with_nested_sha256_is_accepted(self):
        digest=hashlib.sha256((self.root / 'dataset/pretrain_identity_clean.jsonl').read_bytes()).hexdigest()
        (self.root / 'dataset/compiler.report.json').write_text(json.dumps({'output':{'sha256':digest}}))
        self.request['audit_path']='dataset/compiler.report.json'
        self.assertEqual(self.prepare()[4]['data_sha256'],digest)
        (self.root / 'dataset/compiler.report.json').write_text(json.dumps({'output':{'sha256':'0'*64}}))
        with self.assertRaises(ValueError): self.prepare()

    def test_inventory_lists_report_sidecars(self):
        report=self.root / 'dataset/clean.report.json'
        self.assertEqual(training.inventory(self.root,self.opts)['reports'],
                         [{'path':str(report.relative_to(self.root)),'bytes':report.stat().st_size}])

    def shutdown_job(self, status='completed', code=0):
        self.request.pop('shutdown_after_train',None)
        self.request.update(shutdown_delay_seconds=0, operation_id='op_'+'a'*32)
        with patch.object(training,'gpu_check'), patch.object(training,'spawn_supervisor',side_effect=lambda root,job,resume:job):
            job=training.start(self.root,self.opts,self.request)
        job.update(status=status,exit_code=code,finished_at=training.now())
        training.save(training.run_dir(self.root,job['id'])/'job.json',job)
        return job

    def test_default_shutdown_runs_remotely_only_after_success_and_persists_configuration(self):
        job=self.shutdown_job()
        self.assertTrue(job['settings']['shutdown_after_train'])
        with patch.object(training.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')) as run, patch.object(training.time,'sleep'):
            training.shutdown_after_success(self.root,job['id'],0)
        self.assertEqual(run.call_args.args[0],['/usr/bin/shutdown'])
        saved=training.read_job(self.root,job['id'])
        self.assertEqual(saved['status'],'completed')
        self.assertEqual(saved['shutdown_status'],'requested')

    def test_shutdown_skips_pause_stop_failure_and_wrong_attempt(self):
        for status,code in [('paused',42),('cancelled',None),('failed',1),('completed',0)]:
            # Separate operation IDs prevent the idempotency guard reusing a fixture.
            job=self.shutdown_job(status,code)
            with patch.object(training.subprocess,'run') as run, patch.object(training.time,'sleep'):
                training.shutdown_after_success(self.root,job['id'],1 if status=='completed' else 0)
                run.assert_not_called()
            (training.run_dir(self.root,job['id'])/'job.json').unlink()

    def test_shutdown_failure_preserves_training_success_and_new_work_prevents_shutdown(self):
        job=self.shutdown_job()
        with patch.object(training.subprocess,'run',side_effect=OSError('fixture shutdown failure')), patch.object(training.time,'sleep'):
            training.shutdown_after_success(self.root,job['id'],0)
        saved=training.read_job(self.root,job['id'])
        self.assertEqual((saved['status'],saved['shutdown_status']),('completed','failed'))
        with patch.object(training,'idle',side_effect=ValueError('another trainer is active')), patch.object(training.subprocess,'run') as run, patch.object(training.time,'sleep'):
            training.shutdown_after_success(self.root,job['id'],0)
            run.assert_not_called()
        self.assertEqual(training.read_job(self.root,job['id'])['shutdown_status'],'skipped')

    def test_repeated_operation_returns_existing_job_without_restarting(self):
        job=self.shutdown_job()
        with patch.object(training,'gpu_check',side_effect=AssertionError('must not boot/probe again')), patch.object(training,'spawn_supervisor') as spawn:
            repeated=training.start(self.root,self.opts,self.request)
            spawn.assert_not_called()
        self.assertEqual(repeated['id'],job['id'])
        self.assertEqual(training.dispatch(dict(self.opts,action='lookup_operation',operation_id=self.request['operation_id']))['job']['id'],job['id'])

    def test_selected_config_is_snapshotted_and_does_not_inherit_looped_cache_flag(self):
        self.request.pop('audit_path')
        self.request.pop('identity_reviewed')
        self.request['args']['use_looped'] = True
        with patch.object(training,'gpu_check'), patch.object(training,'spawn_supervisor',side_effect=lambda root,job,resume:job):
            job=training.start(self.root,self.opts,self.request)
        self.assertEqual(job['args']['use_looped'],0)
        source=self.root/'trainer/config_instinct_v1_moe.json'
        original=Path(job['config_path']).read_text()
        source.write_text('{}')
        self.assertEqual(Path(job['config_path']).read_text(),original)

    def test_reserved_flags_and_nonfinite_arguments_are_rejected(self):
        for key,value in [('pause_file','../global'),('save_weight','overwrite'),('from_resume',1),('bad;key',1),('learning_rate',float('nan'))]:
            self.request['args'][key]=value
            with self.assertRaises(ValueError): self.prepare()
            del self.request['args'][key]

    def test_path_traversal_and_invalid_job_ids_are_rejected(self):
        with self.assertRaises(ValueError): training.inside(self.root,'../outside')
        with self.assertRaises(ValueError): training.run_dir(self.root,'../../out')

    def test_cpt_and_sft_require_existing_base_weights(self):
        self.request['trainer']='cpt'
        with self.assertRaises(ValueError): self.prepare()
        self.request['args']['from_weight']='pretrain_complete'
        (self.root / 'out/pretrain_complete_512_moe.pth').write_bytes(b'fixture')
        self.assertEqual(self.prepare()[1]['from_weight'],'pretrain_complete')

    def test_resume_import_uses_checkpoint_config_and_prefix(self):
        checkpoint=self.root/'checkpoints/full_sft_old_768_resume.pth'
        checkpoint.write_bytes(b'fixture')
        checkpoint.with_name('full_sft_old_768.json').write_text(json.dumps({'hidden_size':768,'num_hidden_layers':20,'use_moe':False}))
        self.request.update(trainer='full_sft',existing_resume='checkpoints/full_sft_old_768_resume.pth')
        self.request['args']['hidden_size']=512
        _, args, _, model, _, prefix, _=self.prepare()
        self.assertEqual((args['hidden_size'],model['num_hidden_layers'],prefix),(768,20,'full_sft_old'))
        self.request['trainer']='pretrain'
        with self.assertRaises(ValueError):self.prepare()

    def test_distillation_maps_teacher_weight_and_uses_explicit_student_topology(self):
        self.request['trainer']='distillation'
        self.request['args'].update(from_weight='teacher_complete',student_hidden_size=256,student_num_layers=8,student_use_moe=0)
        (self.root / 'out/teacher_complete_512_moe.pth').write_bytes(b'fixture')
        args=self.prepare()[1]
        self.assertEqual(args['from_teacher_weight'],'teacher_complete')
        self.assertEqual(args['from_student_weight'],'none')
        self.assertEqual((args['student_hidden_size'],args['student_use_moe']),(256,0))
        self.assertNotIn('from_weight',args)

    def test_pid_identity_mismatch_does_not_match_reused_pid(self):
        with patch.object(training,'identity',return_value='new-boot:222'):
            self.assertFalse(training.alive(123,'old-boot:222'))
            self.assertFalse(training.alive(123,'new-boot:111'))

    def test_cached_model_is_snapshotted_without_requiring_remote_config_file(self):
        self.request.update(config_path='__panel_cache__',model_config={'hidden_size':256,'num_hidden_layers':12,'use_moe':False},
                            panel_cache_source={'path':'trainer/webui_state.json','sha256':'fixture'})
        config=(self.root/'trainer/config_instinct_v1_moe.json')
        config.unlink()
        prepared=self.prepare()
        self.assertEqual(prepared[1]['hidden_size'],256)
        self.assertIsNone(prepared[2])
        with patch.object(training,'gpu_check'),patch.object(training,'spawn_supervisor',side_effect=lambda root,job,resume:job):
            job=training.start(self.root,self.opts,self.request)
        saved=json.loads(Path(job['config_path']).read_text())
        self.assertEqual(saved,self.request['model_config'])
        self.assertEqual(job['panel_cache_source']['sha256'],'fixture')

    def test_no_gpu_preflight_validates_audit_without_requiring_cuda_or_spawning(self):
        opts=dict(self.opts,action='preflight',request=self.request)
        with patch.object(training,'gpu_check',side_effect=AssertionError('preflight must not probe CUDA')):
            self.assertTrue(training.dispatch(opts)['ready'])
        self.assertEqual(training.all_jobs(self.root),[])
        self.request['args']['data_path']='dataset/missing.parquet'
        with self.assertRaises(ValueError):training.dispatch(opts)

    def test_free_space_is_rechecked_after_hardlink_detachment(self):
        usage=shutil.disk_usage(self.root)
        self.opts['min_free_gb']=1
        with patch.object(training,'gpu_check'), patch.object(training,'detach_outputs'), \
             patch.object(training.shutil,'disk_usage',side_effect=[usage._replace(free=2*1024**3),usage._replace(free=0)]):
            with self.assertRaisesRegex(ValueError,'separating shared'):training.start(self.root,self.opts,self.request)
        self.assertEqual(training.all_jobs(self.root),[])

    def test_shared_weight_is_separated_and_failed_copy_leaves_no_temporary_file(self):
        output=self.root/'out/existing_512_moe.pth'
        backup=self.root/'checkpoints/existing_512_moe.pth'
        output.write_bytes(b'preserve original');os.link(output,backup)
        training.detach_outputs(self.root,{'save_weight':'existing'})
        output.write_bytes(b'new weight')
        self.assertEqual(backup.read_bytes(),b'preserve original')
        output.unlink();os.link(backup,output)
        def partial_copy(source,target):
            Path(target).write_bytes(b'partial')
            raise OSError('disk full')
        with patch.object(training.shutil,'copy2',side_effect=partial_copy):
            with self.assertRaises(OSError):training.detach_outputs(self.root,{'save_weight':'existing'})
        self.assertEqual(output.read_bytes(),b'preserve original')
        self.assertEqual(list((self.root/'out').glob('*.tmp')),[])

    @unittest.skipUnless(os.name == 'posix' and Path('/proc').is_dir(), 'Linux remote process lifecycle')
    def test_start_pause_restart_controller_resume_and_logs(self):
        self.request.update(config_path='__panel_cache__',model_config={'hidden_size':512,'num_hidden_layers':32,'use_moe':True})
        with patch.object(training,'gpu_check'):
            job=training.start(self.root,self.opts,self.request)
        self.addCleanup(self.stop_active)
        job=self.wait(job['id'],{'running'})
        # Pause while the fixture clears stale markers on startup: intent must survive.
        training.action(self.root,job['id'],'pause')
        paused=self.wait(job['id'],{'paused'})
        self.assertEqual(paused['exit_code'],42)
        self.assertTrue(paused['can_resume'])
        persisted=json.loads((training.run_dir(self.root,job['id'])/'job.json').read_text())
        self.assertEqual(persisted['status'],'paused')
        fresh_controller=subprocess.check_output([sys.executable,str(self.root/'.relay_training/manager.py'),
            json.dumps(dict(self.opts,action='get',id=job['id']))],text=True)
        self.assertEqual(json.loads(fresh_controller)['status'],'paused')
        with patch.object(training,'gpu_check'):
            resumed=training.resume(self.root,self.opts,job['id'])
        completed=self.wait(resumed['id'],{'completed'})
        self.assertEqual(completed['attempt'],1)
        self.assertEqual([entry['exit_code'] for entry in completed['attempts']],[42,0])
        log=training.dispatch(dict(self.opts,action='log',id=job['id'],attempt=1,bytes=131072))
        self.assertIn('RESUME=1',log['text'])
        self.assertIn('FINISHED',log['text'])
        training.idle(self.root)

    @unittest.skipUnless(os.name == 'posix' and Path('/proc').is_dir(), 'Linux remote process lifecycle')
    def test_active_job_blocks_second_start_and_shutdown_and_stop_is_terminal(self):
        with patch.object(training,'gpu_check'):
            job=training.start(self.root,self.opts,self.request)
        self.addCleanup(self.stop_active)
        self.wait(job['id'],{'running'})
        with self.assertRaises(ValueError):training.idle(self.root)
        with patch.object(training,'gpu_check'),self.assertRaises(ValueError):training.start(self.root,self.opts,self.request)
        training.action(self.root,job['id'],'stop')
        stopped=self.wait(job['id'],{'cancelled'})
        self.assertNotEqual(stopped['exit_code'],0)

    def wait(self,identifier,states):
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            with training.control_lock(self.root): job=training.read_job(self.root,identifier)
            if job['status'] in states:return job
            if job['status'] in {'failed','interrupted'}:self.fail(str(job))
            time.sleep(0.05)
        self.fail('Timed out: '+str(job))

    def stop_active(self):
        for job in training.all_jobs(self.root):
            if job['status'] in training.ACTIVE:
                training.action(self.root,job['id'],'stop')
                self.wait(job['id'],{'cancelled','failed','interrupted'})


if __name__=='__main__':unittest.main()
