import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('dataset_upload_remote', Path(__file__).parents[1] / 'src/dataset_upload_remote.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class DatasetUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'dataset'
        self.id = 'd_1_' + 'a' * 32
        self.stage = self.root / '.relay_incoming' / self.id
        self.content = b'clean corpus\n'
        self.payload = {'id': self.id, 'install': False, 'files': [
            {'name': 'clean.jsonl', 'size': len(self.content), 'sha256': hashlib.sha256(self.content).hexdigest()}]}

    def prepare(self):
        helper.run(self.payload, self.root)
        (self.stage / 'clean.jsonl').write_bytes(self.content)
        self.payload['install'] = True

    def test_install_detaches_staging_and_retry_keeps_existing(self):
        self.prepare()
        helper.run(self.payload, self.root)
        self.assertEqual((self.root / 'clean.jsonl').read_bytes(), self.content)
        self.assertFalse((self.stage / 'clean.jsonl').exists())
        (self.stage / 'clean.jsonl').write_bytes(self.content)
        helper.run(self.payload, self.root)
        self.assertEqual((self.root / 'clean.jsonl').read_bytes(), self.content)

    def test_conflict_never_overwrites_existing(self):
        self.prepare()
        (self.root / 'clean.jsonl').write_bytes(b'original')
        with self.assertRaises(ValueError):
            helper.run(self.payload, self.root)
        self.assertEqual((self.root / 'clean.jsonl').read_bytes(), b'original')

    def test_hash_failure_removes_only_corrupt_staged_file(self):
        self.prepare()
        (self.stage / 'clean.jsonl').write_bytes(b'x' * len(self.content))
        with self.assertRaises(ValueError):
            helper.run(self.payload, self.root)
        self.assertFalse((self.root / 'clean.jsonl').exists())
        self.assertFalse((self.stage / 'clean.jsonl').exists())

    def test_all_hashes_checked_before_publishing(self):
        self.prepare()
        self.payload['files'].append({'name': 'audit.json', 'size': 2, 'sha256': '0' * 64})
        (self.stage / 'audit.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):
            helper.run(self.payload, self.root)
        self.assertFalse((self.root / 'clean.jsonl').exists())

    def test_traversal_rejected(self):
        for name in ['../outside', '/tmp/escape', 'a\\b', '.hidden', 'a\nb']:
            self.payload['files'][0]['name'] = name
            with self.assertRaises(ValueError):
                helper.run(self.payload, self.root)

    def test_symlink_rejected(self):
        self.prepare()
        link = self.root / 'clean.jsonl'
        try:
            link.symlink_to(self.stage / 'clean.jsonl')
        except OSError:
            self.skipTest('symlink permission unavailable')
        with self.assertRaises(ValueError):
            helper.run(self.payload, self.root)


if __name__ == '__main__':
    unittest.main()
