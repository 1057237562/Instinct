import json

import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM, MOEFeedForward
from model.mofe_checkpoint import (
    assemble_mofe,
    load_manifest,
    preflight_manifest,
    set_mofe_train_scope,
)


def _config(*, use_moe=False, experts=3, top_k=2, router="attention"):
    return InstinctConfig(
        hidden_size=16,
        num_hidden_layers=2,
        use_moe=use_moe,
        vocab_size=64,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        flash_attn=False,
        tie_word_embeddings=False,
        num_experts=experts,
        num_experts_per_tok=top_k,
        moe_expert_mode="frozen" if use_moe else "trainable",
        router_type=router,
    )


def _write_bank(tmp_path, count=3):
    dense = InstinctForCausalLM(_config())
    base_path = tmp_path / "base.pth"
    torch.save(dense.state_dict(), base_path)
    expert_paths = []
    for index in range(count):
        expert = InstinctForCausalLM(_config())
        with torch.no_grad():
            for layer in expert.model.layers:
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    getattr(layer.mlp, projection).weight.fill_(index + 1)
        path = tmp_path / f"expert_{index}.pth"
        torch.save(expert.state_dict(), path)
        expert_paths.append(path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "base_model": str(base_path),
        "experts": [
            {"name": f"expert_{index}", "domain": f"d{index}", "path": str(path)}
            for index, path in enumerate(expert_paths)
        ],
    }), encoding="utf-8")
    return manifest_path, expert_paths


def test_manifest_assembly_transplants_and_freezes_all_experts(tmp_path):
    manifest_path, _ = _write_bank(tmp_path)
    manifest = load_manifest(str(manifest_path))
    model = InstinctForCausalLM(_config(use_moe=True))

    preflight = preflight_manifest(manifest, expected_layers=2)
    report = assemble_mofe(model, manifest)

    assert preflight["validated_ffn_tensors"] == 3 * 2 * 3
    assert report["copied_ffn_tensors"] == 3 * 2 * 3
    for expert_index in range(3):
        for layer in model.model.layers:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                tensor = getattr(layer.mlp.experts[expert_index], projection).weight
                assert torch.all(tensor == expert_index + 1)
                assert not tensor.requires_grad


def test_attention_router_gets_task_gradient_while_experts_stay_unchanged():
    module = MOEFeedForward(_config(use_moe=True, top_k=1))
    before = [parameter.detach().clone() for expert in module.experts for parameter in expert.parameters()]
    optimizer = torch.optim.AdamW(module.gate.parameters(), lr=1e-2)

    output = module(torch.randn(2, 5, 16))
    output.square().mean().backward()
    router_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in module.gate.parameters()
        if parameter.grad is not None
    )
    optimizer.step()

    assert router_grad > 0
    after = [parameter.detach() for expert in module.experts for parameter in expert.parameters()]
    assert all(torch.equal(left, right) for left, right in zip(before, after))


def test_train_scope_router_shared_never_unfreezes_experts():
    model = InstinctForCausalLM(_config(use_moe=True))
    names = set_mofe_train_scope(model, "router_shared")

    assert names
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".mlp.experts." in name
    )
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".mlp.gate." in name
    )


def test_manifest_fingerprint_changes_when_checkpoint_is_replaced(tmp_path):
    manifest_path, expert_paths = _write_bank(tmp_path)
    first = load_manifest(str(manifest_path)).fingerprint
    with open(expert_paths[0], "ab") as handle:
        handle.write(b"replacement")
    second = load_manifest(str(manifest_path)).fingerprint

    assert first != second
