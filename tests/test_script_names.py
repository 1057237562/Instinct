"""Name-resolution guard for the trainer entry points.

The trainers are scripts: nothing imports them, so a renamed import with a
stale call site (or any other undefined name) only shows up when a training run
reaches that line.  These tests approximate pyflakes for the names a script
uses, which is exactly the failure a half-applied rename produces.
"""

import ast
import builtins
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Entry-point scripts a user runs directly; nothing imports them, so a broken
# name survives until runtime.
TRAINER_SCRIPTS = sorted((REPO_ROOT / "trainer").glob("*.py"))
SCRIPT_MODULES = sorted(
    path for path in (REPO_ROOT / "scripts").glob("*.py")
    if path.name not in {"__init__.py"}
)

BUILTIN_NAMES = set(dir(builtins)) | {"__file__", "__name__", "__package__", "__doc__", "WindowsError"}


def _argument_names(arguments):
    names = {argument.arg for argument in arguments.args + arguments.kwonlyargs}
    if arguments.vararg:
        names.add(arguments.vararg.arg)
    if arguments.kwarg:
        names.add(arguments.kwarg.arg)
    return names


def _bound_names(tree):
    """Every name the module binds anywhere: imports, defs, assignments, loops.

    Deliberately generous — a name bound at any nesting level counts, so the
    check has no false positives and still catches a name that is bound
    nowhere in the file.
    """
    bound = set()

    def visit(node):
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(child.name)
                bound.update(_argument_names(child.args))
            elif isinstance(child, ast.Lambda):
                bound.update(_argument_names(child.args))
            elif isinstance(child, ast.ClassDef):
                bound.add(child.name)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                bound.add(child.id)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                bound.add(child.name)
            elif isinstance(child, ast.Global | ast.Nonlocal):
                bound.update(child.names)
            elif isinstance(child, ast.MatchAs) and child.name:
                bound.add(child.name)
            elif isinstance(child, ast.MatchStar) and child.name:
                bound.add(child.name)

    visit(tree)
    return bound


def _undefined_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bound = _bound_names(tree) | BUILTIN_NAMES
    return sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            and node.id not in bound
        }
    )


@pytest.mark.parametrize(
    "path", TRAINER_SCRIPTS + SCRIPT_MODULES, ids=lambda path: path.name,
)
def test_scripts_define_every_name_they_use(path):
    undefined = _undefined_names(path)
    assert not undefined, (
        f"{path.relative_to(REPO_ROOT)} uses undefined names {undefined}; "
        "a renamed import with a stale call site fails this way"
    )


def test_undefined_names_check_catches_a_stale_rename(tmp_path):
    """The guard has to fail on the bug it exists for."""
    sample = tmp_path / "script.py"
    sample.write_text(
        "from scripts.data_loader.streaming_chunks import build_chunk_plan\n"
        "def main():\n"
        "    return build_jsonl_chunk_plan('x')\n",
        encoding="utf-8",
    )
    assert _undefined_names(sample) == ["build_jsonl_chunk_plan"]


def test_streaming_entry_points_exist_where_the_trainer_imports_them():
    """The names the pretrain script uses must resolve on the module."""
    from scripts.data_loader import streaming_chunks
    from trainer import train_pretrain

    assert train_pretrain.build_chunk_plan is streaming_chunks.build_chunk_plan
    assert callable(streaming_chunks.build_chunk_plan)
    assert callable(streaming_chunks.materialize_range)
