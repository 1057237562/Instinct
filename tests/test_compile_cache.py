import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / ".cache" / "torch_compile"


def _read_cache_config(env):
    code = (
        "import json, os; "
        "import trainer.compile_cache as cache; "
        "print(json.dumps({"
        "'path': cache.TORCH_COMPILE_CACHE_DIR, "
        "'fx': os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'], "
        "'aot': os.environ['TORCHINDUCTOR_AUTOGRAD_CACHE'], "
        "'exists': os.path.isdir(cache.TORCH_COMPILE_CACHE_DIR)"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_compile_cache_defaults_to_repository_cache():
    env = os.environ.copy()
    for key in (
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCHINDUCTOR_FX_GRAPH_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_CACHE",
    ):
        env.pop(key, None)
    config = _read_cache_config(env)
    assert Path(config["path"]) == DEFAULT_CACHE
    assert config == {
        "path": str(DEFAULT_CACHE), "fx": "1", "aot": "1", "exists": True,
    }


def test_compile_cache_respects_explicit_override(tmp_path):
    override = tmp_path / "shared-inductor-cache"
    env = os.environ.copy()
    env["TORCHINDUCTOR_CACHE_DIR"] = str(override)
    config = _read_cache_config(env)
    assert Path(config["path"]) == override
    assert config["exists"] is True
