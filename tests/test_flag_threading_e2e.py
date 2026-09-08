"""Task T13: flag-threading end-to-end verification.

End-to-end wiring checks for ``use_grad_checkpoint`` beyond the pure unit test
(``tests/test_flag_threading.py``):

1. **config_path JSON threading.** ``config_from_args(args, config_path=...)``
   loads a JSON config through its ``cfg_dict.update(overrides)`` branch and
   threads ``use_grad_checkpoint`` into the built config (and, via
   ``InstinctForCausalLM(config)``, into ``model.config``). A second JSON-only
   key (``max_position_embeddings``) proves the JSON is actually read, not
   silently replaced by defaults.
2. **args-override priority.** When ``args.use_grad_checkpoint`` is set, its
   value wins over the JSON value (``cfg_dict.update(overrides)`` overwrites
   the loaded dict).
3. **Legacy JSON compat.** Old config JSONs carrying a leftover
   ``loop_grad_checkpoint`` key do NOT crash: ``InstinctConfig``'s ``**kwargs``
   (routed into ``super().__init__(**kwargs)``) absorbs the unknown key, so it
   is ignored and the flag still lands correctly.
4. **Resume semantics.** ``lm_checkpoint`` persists ``config.to_dict()`` (which
   serializes ``use_grad_checkpoint``) inside the resume checkpoint, so a
   save -> load round trip preserves the flag. The trainer nevertheless rebuilds
   the live config from args on resume (``config_from_args(args)``, exactly as
   ``trainer/train_pretrain.py`` does), never from the checkpoint's stored
   config.
5. **Dead-flag grep** is a separate shell assertion; see
   ``.sisyphus/evidence/task-13-deadflag-global.txt``.
"""

import json
from types import SimpleNamespace

import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from trainer.trainer_utils import config_from_args, lm_checkpoint
from tests.helpers import make_tiny_config


def make_args(**kw):
    """Argparse-like namespace mirroring ``tests/test_flag_threading.py``."""
    base = dict(
        param_dtype="fp32",
        kv_cache_dtype="fp32",
        hidden_size=64,
        num_hidden_layers=2,
        use_moe=0,
        use_looped=0,
        config_path="",
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# 1. config_path JSON threading (incl. reaching model.config)
# ---------------------------------------------------------------------------

def test_config_path_json_threads_grad_checkpoint(tmp_path):
    """A JSON config with use_grad_checkpoint=1 flows into config AND model.config."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps({"use_grad_checkpoint": 1, "max_position_embeddings": 8192}),
        encoding="utf-8",
    )
    args = make_args(use_grad_checkpoint=1, config_path=str(cfg_path))

    config = config_from_args(args)
    assert config.use_grad_checkpoint == 1
    # A JSON-only key proves the config_path branch really read the file
    # (default max_position_embeddings is 32768).
    assert config.max_position_embeddings == 8192

    model = InstinctForCausalLM(config)
    assert model.config.use_grad_checkpoint == 1


def test_config_path_json_args_override_wins(tmp_path):
    """args.use_grad_checkpoint=2 overrides the JSON's 1 (cfg_dict.update(overrides))."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"use_grad_checkpoint": 1}), encoding="utf-8")
    args = make_args(use_grad_checkpoint=2, config_path=str(cfg_path))

    config = config_from_args(args)
    assert config.use_grad_checkpoint == 2


# ---------------------------------------------------------------------------
# 2. Legacy JSON compat (leftover loop_grad_checkpoint key)
# ---------------------------------------------------------------------------

def test_instinct_config_absorbs_unknown_legacy_key():
    """Direct construction with a leftover loop_grad_checkpoint key must not crash."""
    config = InstinctConfig(loop_grad_checkpoint=True, use_grad_checkpoint=1)
    assert config.use_grad_checkpoint == 1


def test_legacy_json_with_loop_grad_checkpoint_no_crash(tmp_path):
    """An old config JSON carrying loop_grad_checkpoint survives config_from_args."""
    old = {"loop_grad_checkpoint": True, "vocab_size": 128}
    cfg_path = tmp_path / "old.json"
    cfg_path.write_text(json.dumps(old), encoding="utf-8")
    args = make_args(use_grad_checkpoint=1, config_path=str(cfg_path))

    config = config_from_args(args)  # must not raise
    assert config.use_grad_checkpoint == 1
    assert config.vocab_size == 128  # remaining JSON keys still honored


# ---------------------------------------------------------------------------
# 3. Resume semantics (lm_checkpoint save -> load, config rebuilt from args)
# ---------------------------------------------------------------------------

def test_lm_checkpoint_roundtrip_preserves_flag(tmp_path):
    """config.to_dict() serializes use_grad_checkpoint; a save->load keeps it."""
    config = make_tiny_config()
    config.use_grad_checkpoint = 1
    torch.manual_seed(0)
    model = InstinctForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    lm_checkpoint(
        config, weight="t13_resume", model=model, optimizer=optimizer,
        epoch=1, step=42, save_dir=str(tmp_path),
    )

    ckp_data = lm_checkpoint(config, weight="t13_resume", save_dir=str(tmp_path))
    assert ckp_data is not None
    assert ckp_data["config"]["use_grad_checkpoint"] == 1  # persisted in to_dict()
    rebuilt = InstinctConfig(**ckp_data["config"])
    assert rebuilt.use_grad_checkpoint == 1
    assert {tensor.dtype for tensor in ckp_data["model"].values()} == {torch.float32}
    inference_state = torch.load(
        tmp_path / f"t13_resume_{config.hidden_size}.pth", weights_only=True
    )
    assert {tensor.dtype for tensor in inference_state.values()} == {torch.float16}


def test_resume_config_rebuilt_from_args_not_checkpoint(tmp_path):
    """On resume the live config comes from args, never from the checkpoint.

    Mirrors ``trainer/train_pretrain.py``:
        lm_config = config_from_args(args)
        ckp_data  = lm_checkpoint(lm_config, ...)
    The stored config (use_grad_checkpoint=0) must not leak into the live config.
    """
    saved_cfg = make_tiny_config()
    saved_cfg.use_grad_checkpoint = 0
    model = InstinctForCausalLM(saved_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lm_checkpoint(
        saved_cfg, weight="t13_resume", model=model, optimizer=optimizer,
        save_dir=str(tmp_path),
    )

    args = make_args(use_grad_checkpoint=2)
    lm_config = config_from_args(args)  # rebuild from args, as the trainer does
    ckp_data = lm_checkpoint(lm_config, weight="t13_resume", save_dir=str(tmp_path))

    assert ckp_data is not None
    assert ckp_data["config"]["use_grad_checkpoint"] == 0  # stored config is stale
    assert lm_config.use_grad_checkpoint == 2              # live config from args
