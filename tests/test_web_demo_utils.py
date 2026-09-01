from pathlib import Path

from scripts.web_demo_utils import resolve_model_config_path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_resolve_model_config_prefers_config_beside_weight(tmp_path):
    weight = _touch(tmp_path / "out" / "model.pth")
    adjacent = _touch(tmp_path / "out" / "model.json")
    _touch(tmp_path / "checkpoints" / "model.json")

    assert resolve_model_config_path(str(weight), str(tmp_path)) == str(adjacent)


def test_resolve_model_config_finds_checkpoint_config_for_out_weight(tmp_path):
    weight = _touch(tmp_path / "out" / "model.pth")
    checkpoint_config = _touch(tmp_path / "checkpoints" / "model.json")
    _touch(tmp_path / "trainer" / "config_pretrain.json")

    assert resolve_model_config_path(str(weight), str(tmp_path)) == str(checkpoint_config)


def test_resolve_model_config_uses_legacy_fallback_last(tmp_path):
    weight = _touch(tmp_path / "out" / "model.pth")
    fallback = _touch(tmp_path / "trainer" / "config_pretrain.json")

    assert resolve_model_config_path(str(weight), str(tmp_path)) == str(fallback)
