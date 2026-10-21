import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('cleanup', Path(__file__).parents[1] / 'src' / 'cleanup_server_space.py')
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class StorageCleanupTests(unittest.TestCase):
    def archive(self, path, prefix, tensor):
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(prefix + '/data.pkl', b'model metadata')
            archive.writestr(prefix + '/data/0', tensor)
            archive.writestr(prefix + '/.data/serialization_id', b'id')

    def test_identical_records_with_different_wrapper_names_are_hardlinked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, target = root / 'a.pth', root / 'b.pth'
            self.archive(source, 'short', b'tensor values')
            self.archive(target, 'longer-wrapper', b'tensor values')
            result = cleanup.deduplicate(root, source, target)
            self.assertEqual(result['status'], 'linked')
            self.assertTrue(source.samefile(target))
            self.assertEqual(cleanup.deduplicate(root, source, target)['status'], 'already_linked')

    def test_different_tensor_or_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, target = root / 'a.pth', root / 'b.pth'
            self.archive(source, 'a', b'original')
            self.archive(target, 'b', b'different')
            before = target.read_bytes()
            self.assertEqual(cleanup.deduplicate(root, source, target)['status'], 'different')
            self.assertEqual(target.read_bytes(), before)
            self.assertFalse(source.samefile(target))

    def test_outside_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaises(ValueError):
                cleanup.checked(root, root.parent / 'outside.pth')


if __name__ == '__main__':
    unittest.main()
