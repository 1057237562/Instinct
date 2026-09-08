import json
import os
import subprocess
import sys

import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.mofe_checkpoint import load_mofe_delta


def _dense_config():
    return InstinctConfig(
        hidden_size=16,
        num_hidden_layers=1,
        vocab_size=6400,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        flash_attn=False,
        tie_word_embeddings=False,
    )


def test_mofe_post_pretrain_one_cpu_step(tmp_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base = tmp_path / "base.pth"
    torch.save(InstinctForCausalLM(_dense_config()).state_dict(), base)
    experts = []
    for index in range(2):
        path = tmp_path / f"expert_{index}.pth"
        torch.save(InstinctForCausalLM(_dense_config()).state_dict(), path)
        experts.append({"name": f"e{index}", "path": str(path)})

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"base_model": str(base), "experts": experts}), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_dense_config().to_dict()), encoding="utf-8")
    data = tmp_path / "pretrain_smoke.jsonl"
    data.write_text('{"text":"tiny MoFE training example"}\n', encoding="utf-8")
    output_dir = tmp_path / "out"
    checkpoint_dir = tmp_path / "checkpoints"

    command = [
        sys.executable, "-u", "trainer/train_mofe_post_pretrain.py",
        "--config_path", str(config),
        "--expert_manifest", str(manifest),
        "--data_path", str(data),
        "--save_dir", str(output_dir),
        "--checkpoint_dir", str(checkpoint_dir),
        "--save_weight", "smoke",
        "--device", "cpu",
        "--dtype", "fp32",
        "--param_dtype", "fp32",
        "--hidden_size", "16",
        "--num_hidden_layers", "1",
        "--num_experts_per_tok", "1",
        "--train_scope", "router_only",
        "--epochs", "1",
        "--batch_size", "1",
        "--accumulation_steps", "1",
        "--max_seq_len", "16",
        "--num_workers", "0",
        "--log_interval", "1",
        "--save_interval", "1",
        "--use_compile", "0",
    ]
    result = subprocess.run(
        command, cwd=repo_root, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    delta_path = output_dir / "smoke_16_mofe_delta.pth"
    resume_path = checkpoint_dir / "smoke_16_moe_resume.pth"
    assert delta_path.is_file()
    assert resume_path.is_file()
    delta = torch.load(delta_path, map_location="cpu", weights_only=False)
    assert delta["format"] == "instinct-mofe-delta-v1"
    assert delta["train_scope"] == "router_only"
    assert delta["model_delta"]
    assert all(".mlp.gate." in key for key in delta["model_delta"])
    assert all(".mlp.experts." not in key for key in delta["model_delta"])

    restored, _ = load_mofe_delta(str(delta_path))
    result = restored(torch.tensor([[1, 3, 4, 2]]), labels=torch.tensor([[1, 3, 4, 2]]))
    assert torch.isfinite(result.loss)
