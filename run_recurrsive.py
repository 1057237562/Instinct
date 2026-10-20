"""Run a trainer with the RecurrsiveInstinct backbone from the repository root."""
import importlib
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_recurrsive.py trainer/train_x.py [arguments]")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import datasets  # Must precede torch imports on Windows.
    import trainer.compile_cache  # Configure Inductor before loading the backbone.

    sys.modules["model.model_instinct"] = importlib.import_module(
        "model.model_recurrsive_instinct"
    )
    target = str(Path(sys.argv[1]).resolve())
    sys.argv = [target, *sys.argv[2:]]
    runpy.run_path(target, run_name="__main__")
