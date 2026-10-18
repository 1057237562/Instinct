import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('datasets_remote', Path(__file__).parents[1] / 'src/datasets_remote.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class DatasetBrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'dataset'
        self.root.mkdir()
        (self.root / 'corpus.parquet').write_bytes(b'corpus')
        (self.root / 'corpus.parquet.report.json').write_text('{"reviewed":true}')
        (self.root / 'shards').mkdir()
        (self.root / 'shards' / 'part.jsonl').write_text('{}\n')
        (self.root / '.relay_incoming').mkdir()
        (self.root / '.relay_incoming' / 'keep.rpart').write_bytes(b'partial')

    def run_action(self, action='list', **kwargs):
        return helper.run({'action': action, **kwargs}, self.root)

    def entry(self, name):
        return next(e for e in self.run_action()['entries'] if e['name'] == name)

    def test_lists_metadata_without_staging_and_enters_directories(self):
        result = self.run_action()
        self.assertEqual([e['name'] for e in result['entries']], ['shards', 'corpus.parquet', 'corpus.parquet.report.json'])
        self.assertEqual(self.entry('corpus.parquet')['size'], 6)
        self.assertTrue(self.entry('corpus.parquet')['trainable'])
        self.assertTrue(self.entry('corpus.parquet.report.json')['previewable'])
        self.assertEqual(self.run_action(path='shards')['entries'][0]['path'], 'shards/part.jsonl')
        self.assertGreater(result['disk']['total'], 0)

    def test_missing_root_is_read_only(self):
        missing = self.root / 'absent'
        result = helper.run({'action': 'list'}, missing)
        self.assertFalse(result['exists'])
        self.assertFalse(missing.exists())

    def test_preview_is_bounded_and_corpus_is_not_read(self):
        (self.root / 'large.json').write_bytes(b'x' * (helper.PREVIEW_BYTES + 10))
        data = self.run_action('preview', path='large.json')
        self.assertTrue(data['truncated'])
        self.assertEqual(len(data['text']), helper.PREVIEW_BYTES)
        with self.assertRaises(ValueError):
            self.run_action('preview', path='corpus.parquet')

    def test_delete_preserves_sidecars_and_staging(self):
        entry = self.entry('corpus.parquet')
        self.run_action('delete', path=entry['path'], version=entry['version'])
        self.assertFalse((self.root / 'corpus.parquet').exists())
        self.assertTrue((self.root / 'corpus.parquet.report.json').exists())
        self.assertTrue((self.root / '.relay_incoming/keep.rpart').exists())

    def test_stale_delete_and_directory_delete_fail(self):
        entry = self.entry('corpus.parquet')
        (self.root / 'corpus.parquet').write_bytes(b'new corpus')
        with self.assertRaises(ValueError):
            self.run_action('delete', path=entry['path'], version=entry['version'])
        for target in ['', 'shards']:
            with self.assertRaises(ValueError):
                self.run_action('delete', path=target, version='a' * 64)
        self.assertEqual((self.root / 'corpus.parquet').read_bytes(), b'new corpus')

    def test_traversal_hidden_and_symlink_boundaries(self):
        for target in ['/etc/passwd', '../outside', 'a/../b', 'a\\b', '.relay_incoming/keep.rpart', 'a//b']:
            with self.assertRaises(ValueError):
                self.run_action(path=target)
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        (outside / 'keep.json').write_text('{}')
        try:
            (self.root / 'link').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('Symlink permission unavailable')
        with self.assertRaises(ValueError):
            self.run_action('delete', path='link/keep.json', version='a' * 64)
        self.assertEqual(self.entry('link')['kind'], 'unsupported')
        self.assertTrue((outside / 'keep.json').exists())


if __name__ == '__main__':
    unittest.main()
