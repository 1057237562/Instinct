"""Small, side-effect-free helpers shared by the chat WebUI and its tests."""

from pathlib import Path


def resolve_model_config_path(weight_path: str, repo_root: str) -> str:
    """Resolve the config matching a raw weight file.

    Training writes lightweight weights to ``out/`` and the matching config to
    ``checkpoints/``.  Prefer a config beside the selected weight (for custom
    exports), then the checkpoint copy, and only then the legacy pretrain
    fallback.
    """
    root = Path(repo_root)
    weight = Path(weight_path)
    config_name = weight.with_suffix(".json").name
    candidates = (
        weight.with_suffix(".json"),
        root / "checkpoints" / config_name,
        root / "trainer" / "config_pretrain.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""
