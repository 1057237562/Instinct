import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('code_sync_remote', Path(__file__).parents[1] / 'src/code_sync_remote.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
GUARD_SOURCE = (Path(__file__).parents[1] / 'src/training_remote.py').read_text(encoding='utf-8')


class CodeSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repo'
        for name in ['model/model_instinct.py', 'trainer/train_pretrain.py', 'scripts/config_webui.py']:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('old = True\n')
        for name in ['dataset/keep.parquet', 'out/keep.pth', 'checkpoints/keep.pth', 'trainer/webui_state.json']:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('keep')
        self.opts = {'repo_root': str(self.root), 'id': 'cs_' + 'a' * 32, 'action': 'prepare'}
        helper.run(self.opts)
        self.directory = self.root / '.relay_code_sync' / self.opts['id']
        self.scope = {'control_lock': lambda root: contextlib.nullcontext(), 'idle': lambda root, **kw: None}

    def plan(self, values=None):
        values = values or {'scripts/config_webui.py': b'new = True\n', 'trainer/new_helper.py': b'answer = 42\n'}
        files = []
        for name, content in values.items():
            target = self.directory / 'files' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            files.append({'path': name, 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
        (self.directory / 'manifest.json').write_text(json.dumps({'files': files}))
        return self.run_action('preview')

    def run_action(self, action):
        return helper.run(dict(self.opts, action=action), self.scope)

    def test_preview_apply_and_rollback_preserve_runtime_files(self):
        before = (self.root / 'scripts/config_webui.py').read_bytes()
        preview = self.plan()
        self.assertEqual([f['status'] for f in preview['files']], ['changed', 'new'])
        self.assertEqual((self.root / 'scripts/config_webui.py').read_bytes(), before)
        self.assertEqual(self.run_action('apply')['status'], 'completed')
        self.assertEqual((self.root / 'trainer/new_helper.py').read_text(), 'answer = 42\n')
        self.assertEqual((self.directory / 'backup/scripts/config_webui.py').read_bytes(), before)
        self.assertEqual(self.run_action('apply')['status'], 'completed')
        self.assertEqual(self.run_action('rollback')['status'], 'rolled_back')
        self.assertEqual((self.root / 'scripts/config_webui.py').read_bytes(), before)
        self.assertFalse((self.root / 'trainer/new_helper.py').exists())
        for name in ['dataset/keep.parquet', 'out/keep.pth', 'checkpoints/keep.pth', 'trainer/webui_state.json']:
            self.assertEqual((self.root / name).read_text(), 'keep')

    def test_embedded_guard_is_verified_before_execution(self):
        self.plan()
        source = self.directory / 'training_guard.py'
        source.write_text(GUARD_SOURCE, encoding='utf-8')
        expected = helper.digest(source)
        loaded = helper.guard(self.directory, expected)
        self.assertIn('control_lock', loaded)
        source.write_text('raise RuntimeError("must not execute")')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            helper.guard(self.directory, expected)

    def test_remote_edit_after_preview_is_not_overwritten(self):
        self.plan()
        (self.root / 'scripts/config_webui.py').write_text('external = True\n')
        with self.assertRaisesRegex(ValueError, '重新比较'):
            self.run_action('apply')
        self.assertEqual((self.root / 'scripts/config_webui.py').read_text(), 'external = True\n')
        self.assertFalse((self.root / 'trainer/new_helper.py').exists())

    def test_corrupt_staging_and_invalid_python_stop_before_install(self):
        self.plan({'scripts/config_webui.py': b'broken (\n'})
        with self.assertRaises(SyntaxError):
            self.run_action('apply')
        self.assertEqual((self.root / 'scripts/config_webui.py').read_text(), 'old = True\n')
        (self.directory / 'files/scripts/config_webui.py').write_text('other = 1\n')
        with self.assertRaises(ValueError):
            self.run_action('apply')

    def test_install_failure_restores_earlier_files(self):
        self.plan()
        replace = helper.replace_copy
        def fail_new(source, target, mode=None):
            if target.name == 'new_helper.py':
                raise OSError('fixture failure')
            replace(source, target, mode)
        with patch.object(helper, 'replace_copy', side_effect=fail_new):
            with self.assertRaises(OSError):
                self.run_action('apply')
        self.assertEqual((self.root / 'scripts/config_webui.py').read_text(), 'old = True\n')
        self.assertFalse((self.root / '.relay_code_sync/ACTIVE').exists())

    def test_rollback_refuses_to_destroy_later_edits(self):
        self.plan(); self.run_action('apply')
        target = self.root / 'scripts/config_webui.py'
        target.write_text('later = True\n')
        with self.assertRaisesRegex(ValueError, '回滚冲突'):
            self.run_action('rollback')
        self.assertEqual(target.read_text(), 'later = True\n')

    def test_active_training_and_other_pending_sync_block_install(self):
        self.plan()
        self.scope['idle'] = lambda *a, **kw: (_ for _ in ()).throw(ValueError('training active'))
        with self.assertRaisesRegex(ValueError, 'training active'):
            self.run_action('apply')
        self.scope['idle'] = lambda *a, **kw: None
        (self.root / '.relay_code_sync/ACTIVE').write_text('cs_' + 'b' * 32)
        with self.assertRaisesRegex(ValueError, '另一个同步'):
            self.run_action('apply')

    def test_policy_rejects_paths_and_symlinks(self):
        for name in ['../escape.py', '/model/x.py', 'trainer/webui_state.json', 'dataset/x.py', 'out/x.pth', '.git/config', 'scripts/data_builder/x.py']:
            self.assertFalse(helper.allowed(name), name)
        try:
            (self.root / 'model/link.py').symlink_to(self.root / 'scripts/config_webui.py')
        except OSError:
            self.skipTest('symlink permission unavailable')
        with self.assertRaises(ValueError):
            helper.inside(self.root, 'model/link.py')


if __name__ == '__main__':
    unittest.main()
