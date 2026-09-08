import pytest
import torch
from torch import nn

from trainer.trainer_utils import CombinedOptimizer, MuonOptimizer, build_optimizer


def test_low_learning_rate_rejects_direct_bfloat16_parameters():
    model = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="--param_dtype fp32"):
        build_optimizer(model.named_parameters(), lr=1e-5, optimizer="adamw")


def test_low_learning_rate_allows_fp32_master_parameters():
    model = nn.Linear(16, 16, bias=False, dtype=torch.float32)

    optimizer = build_optimizer(model.named_parameters(), lr=1e-5, optimizer="adamw")

    assert isinstance(optimizer, torch.optim.AdamW)


def test_pretraining_learning_rate_still_allows_direct_bfloat16_parameters():
    model = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)

    optimizer = build_optimizer(model.named_parameters(), lr=5e-4, optimizer="adamw")

    assert isinstance(optimizer, torch.optim.AdamW)


def test_fallback_muon_uses_match_rms_adamw_scaling():
    parameter = nn.Parameter(torch.zeros(16, 16))
    parameter.grad = torch.randn_like(parameter)
    optimizer = MuonOptimizer(
        [parameter], lr=1e-5, weight_decay=0.0,
        adjust_lr_fn="match_rms_adamw",
    )
    before = parameter.detach().clone()

    optimizer.step()

    assert optimizer.param_groups[0]["adjust_lr_fn"] == "match_rms_adamw"
    assert not torch.equal(parameter, before)


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="requires native torch.optim.Muon")
def test_fallback_muon_tracks_official_muon():
    initial = torch.linspace(-0.2, 0.2, 48, dtype=torch.float32).reshape(6, 8)
    fallback_param = nn.Parameter(initial.clone())
    official_param = nn.Parameter(initial.clone())
    fallback = MuonOptimizer(
        [fallback_param], lr=2e-4, weight_decay=0.1,
        adjust_lr_fn="match_rms_adamw",
    )
    official = torch.optim.Muon(
        [official_param], lr=2e-4, weight_decay=0.1,
        adjust_lr_fn="match_rms_adamw",
    )

    for step in range(3):
        grad = torch.sin(torch.arange(48, dtype=torch.float32) + step).reshape(6, 8)
        fallback_param.grad = grad.clone()
        official_param.grad = grad.clone()
        fallback.step()
        official.step()

    torch.testing.assert_close(fallback_param, official_param, rtol=0, atol=2e-7)


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="requires native torch.optim.Muon")
def test_muon_uses_hidden_matrices_but_not_embeddings():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 16)
            self.proj = nn.Linear(16, 16, bias=False)
            self.norm = nn.LayerNorm(16)

    model = TinyModel()
    optimizer = build_optimizer(model.named_parameters(), lr=1e-5, optimizer="muon")

    assert isinstance(optimizer, CombinedOptimizer)
    muon, adamw = optimizer.optimizers
    muon_ids = {id(param) for group in muon.param_groups for param in group["params"]}
    adamw_ids = {id(param) for group in adamw.param_groups for param in group["params"]}
    assert id(model.proj.weight) in muon_ids
    assert id(model.embed_tokens.weight) in adamw_ids
    assert id(model.embed_tokens.weight) not in muon_ids
    assert muon.param_groups[0]["adjust_lr_fn"] == "match_rms_adamw"


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="requires native torch.optim.Muon")
def test_muon_resume_tolerates_legacy_embedding_group_layout():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32, 16)
            self.proj = nn.Linear(16, 16, bias=False)
            self.norm = nn.LayerNorm(16)

    legacy_model = TinyModel()
    legacy_optimizer = build_optimizer(legacy_model.parameters(), lr=1e-5, optimizer="muon")
    current_model = TinyModel()
    current_optimizer = build_optimizer(
        current_model.named_parameters(), lr=1e-5, optimizer="muon"
    )

    current_optimizer.load_state_dict(legacy_optimizer.state_dict())

    assert all(not optimizer.state for optimizer in current_optimizer.optimizers)
