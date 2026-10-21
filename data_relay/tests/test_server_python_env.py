import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

spec = importlib.util.spec_from_file_location('env_check', Path(__file__).parents[1] / 'src' / 'check_server_python_env.py')
env_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(env_check)


def state(*, numpy_ok=True, numpy_version='1.26.4', core_ok=True,
          pyarrow_ok=False, pyarrow_version='14.0.2', pandas_ok=True,
          duplicate_dists=None, conda_numpy=()):
    duplicates = {'numpy': ['numpy-' + (numpy_version or '0') + '.dist-info'],
                  'pyarrow': ['pyarrow-' + (pyarrow_version or '0') + '.dist-info'],
                  'pandas': ['pandas-2.1.4.dist-info']}
    if duplicate_dists:
        duplicates.update(duplicate_dists)
    return {
        'python': {'executable': 'python3', 'version': '3.12.7'},
        'numpy': {'dist_version': numpy_version, 'import_ok': numpy_ok,
                  'imported_version': numpy_version if numpy_ok else None,
                  'error': None if numpy_ok else 'ImportError: numpy._core.multiarray failed to import',
                  'core_ok': core_ok if numpy_ok else False,
                  'core_module': 'numpy._core.multiarray' if (numpy_ok and core_ok) else None},
        'pyarrow': {'dist_version': pyarrow_version, 'import_ok': pyarrow_ok,
                    'imported_version': pyarrow_version if pyarrow_ok else None,
                    'error': None if pyarrow_ok else 'ImportError: numpy._core.multiarray failed to import'},
        'pandas': {'dist_version': '2.1.4', 'import_ok': pandas_ok,
                   'imported_version': '2.1.4' if pandas_ok else None,
                   'error': None if pandas_ok else 'ImportError: numpy.core.multiarray failed to import'},
        'others': {'torch': '2.5.1', 'datasets': '3.1.0', 'streamlit': '1.41.1'},
        'install': {'site_packages': '/root/miniconda3/lib/python3.12/site-packages',
                    'duplicate_dists': duplicates,
                    'conda_meta': {'numpy': list(conda_numpy), 'pyarrow': []}},
    }


class PythonEnvCheckTests(unittest.TestCase):
    def test_healthy_env_needs_no_repair(self):
        plan = env_check.plan_repair(state(pyarrow_ok=True))
        self.assertEqual(plan['status'], 'healthy')
        self.assertEqual(plan['actions'], [])

    def test_broken_pyarrow_upgrades_pyarrow_only(self):
        plan = env_check.plan_repair(state())
        self.assertEqual(plan['status'], 'pyarrow_mismatch')
        self.assertEqual(plan['actions'], [('upgrade', 'pyarrow')])
        self.assertIn('不要动 numpy', plan['diagnosis'])
        for action in plan['actions']:
            self.assertNotIn('numpy', action)

    def test_broken_pyarrow_with_duplicate_dists_uninstalls_first(self):
        plan = env_check.plan_repair(state(duplicate_dists={
            'pyarrow': ['pyarrow-14.0.0.dist-info', 'pyarrow-14.0.2.dist-info']}))
        self.assertEqual(plan['actions'], [('uninstall', 'pyarrow'), ('upgrade', 'pyarrow')])

    def test_broken_pyarrow_without_install_reinstalls(self):
        plan = env_check.plan_repair(state(pyarrow_version=None))
        self.assertEqual(plan['actions'], [('install', 'pyarrow')])

    def test_broken_numpy_reinstalls_pinned_version(self):
        plan = env_check.plan_repair(state(numpy_ok=False))
        self.assertEqual(plan['status'], 'numpy_broken')
        self.assertEqual(plan['actions'],
                         [('uninstall', 'pyarrow', 'numpy'), ('install', 'numpy==1.26.4', 'pyarrow')])

    def test_broken_numpy_with_ambiguous_records_defaults_to_numpy_lt_2(self):
        plan = env_check.plan_repair(state(numpy_ok=False, duplicate_dists={
            'numpy': ['numpy-1.26.4.dist-info', 'numpy-2.0.0.dist-info']}))
        self.assertEqual(plan['numpy_spec'], 'numpy<2')
        self.assertEqual(plan['actions'],
                         [('uninstall', 'pyarrow', 'numpy'), ('install', 'numpy<2', 'pyarrow')])

    def test_explicit_numpy_spec_overrides_default(self):
        plan = env_check.plan_repair(state(numpy_ok=False), numpy_spec='numpy==2.1.3')
        self.assertEqual(plan['numpy_spec'], 'numpy==2.1.3')

    def test_broken_pandas_upgrades_pandas_only(self):
        plan = env_check.plan_repair(state(pyarrow_ok=True, pandas_ok=False))
        self.assertEqual(plan['status'], 'pandas_mismatch')
        self.assertEqual(plan['actions'], [('upgrade', 'pandas')])

    def test_broken_numpy_gets_conda_mixed_management_warning(self):
        plan = env_check.plan_repair(state(numpy_ok=False, conda_numpy=['numpy-1.26.4.json']))
        self.assertIn('conda', plan['diagnosis'])

    def test_render_commands(self):
        commands = env_check.render_commands(
            [('uninstall', 'pyarrow', 'numpy'), ('upgrade', 'pyarrow'), ('install', 'numpy<2', 'pyarrow')])
        self.assertEqual(commands[0][1:], ['-m', 'pip', 'uninstall', '-y', 'pyarrow', 'numpy'])
        self.assertEqual(commands[1][1:], ['-m', 'pip', 'install', '--no-cache-dir', '--upgrade', 'pyarrow'])
        self.assertEqual(commands[2][1:], ['-m', 'pip', 'install', '--no-cache-dir', 'numpy<2', 'pyarrow'])

    def test_scan_duplicates_finds_mixed_installs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['numpy-1.26.4.dist-info', 'numpy-2.0.0.dist-info',
                         'pyarrow-14.0.2.dist-info', 'pandas', 'arrow-1.0.dist-info']:
                (root / name).mkdir()
            found = env_check.scan_duplicates(root)
            self.assertEqual(len(found['numpy']), 2)
            self.assertEqual(found['pyarrow'], ['pyarrow-14.0.2.dist-info'])
            self.assertEqual(found['pandas'], [])
            self.assertEqual(env_check.scan_duplicates(root / 'missing'), {p: [] for p in env_check.PROBED})

    def test_conda_meta_records_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(env_check.conda_meta_records(root)['numpy'], [])
            (root / 'conda-meta').mkdir()
            (root / 'conda-meta' / 'numpy-1.26.4.json').write_text('{}', encoding='utf-8')
            (root / 'conda-meta' / 'pyarrow-14.0.2.json').write_text('{}', encoding='utf-8')
            records = env_check.conda_meta_records(root)
            self.assertEqual(records['numpy'], ['numpy-1.26.4.json'])
            self.assertEqual(records['pyarrow'], ['pyarrow-14.0.2.json'])

    def test_fresh_state_purges_stale_modules(self):
        fake = types.ModuleType('pandas')
        sys.modules['pandas'] = fake
        report = env_check.fresh_state()
        self.assertIsNot(sys.modules.get('pandas'), fake)
        for key in ('python', 'numpy', 'pyarrow', 'pandas', 'others', 'install'):
            self.assertIn(key, report)


if __name__ == '__main__':
    unittest.main()
