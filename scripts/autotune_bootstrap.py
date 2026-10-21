"""Preserve normal imports when Windows temporarily denies root enumeration.

Only used after a PermissionError from listing the repository directory.
Loads the original files and keeps the normal trainer import order intact.
"""
import importlib.machinery
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare_imports():
    root = str(ROOT)
    try:
        os.listdir(root)
    except PermissionError:
        finder = importlib.machinery.FileFinder(root,
            (importlib.machinery.SourceFileLoader, importlib.machinery.SOURCE_SUFFIXES),
            (importlib.machinery.SourcelessFileLoader, importlib.machinery.BYTECODE_SUFFIXES),
            (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES))
        finder._path_cache = {'trainer', 'model', 'scripts', 'experiments', 'tests'}
        finder._relaxed_path_cache = finder._path_cache
        finder._path_mtime = os.stat(root).st_mtime
        sys.path_importer_cache[root] = finder
        if root not in sys.path:
            sys.path.insert(0, root)
        return True
    return False


if __name__ == '__main__':
    prepare_imports()
    sys.path.insert(0, str(ROOT))
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name='__main__')
