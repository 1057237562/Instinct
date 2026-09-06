"""Regression coverage for the Transformers 5 runtime migration."""

import importlib.metadata
from pathlib import Path

import pytest
import torch
from packaging.version import Version
from transformers import AutoTokenizer

from tests.helpers import make_tiny_config


ROOT = Path(__file__).resolve().parents[1]


def test_transformers_and_hub_major_versions():
    assert Version(importlib.metadata.version("transformers")).major == 5
    assert Version(importlib.metadata.version("huggingface-hub")).major == 1


@pytest.mark.parametrize("variant", ["dense", "linear", "loop"])
def test_custom_models_forward_on_transformers_v5(variant):
    if variant == "dense":
        from model.model_instinct import InstinctForCausalLM
    elif variant == "linear":
        from model.model_instinct_linear import InstinctForCausalLM
    else:
        from model.model_instinct_loop import InstinctForCausalLM

    model = InstinctForCausalLM(make_tiny_config(variant=variant)).eval()
    input_ids = torch.tensor([[1, 5, 7, 2]])
    with torch.no_grad():
        output = model(input_ids, labels=input_ids)

    assert output.logits.shape == (1, 4, 256)
    assert torch.isfinite(output.loss)


def test_custom_model_safetensors_round_trip(tmp_path):
    from model.model_instinct import InstinctForCausalLM

    model = InstinctForCausalLM(make_tiny_config()).eval()
    model.save_pretrained(tmp_path)
    restored = InstinctForCausalLM.from_pretrained(tmp_path, local_files_only=True).eval()

    assert (tmp_path / "model.safetensors").is_file()
    expected = model.state_dict()
    actual = restored.state_dict()
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name


def test_project_tokenizer_chat_template_on_transformers_v5():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt)

    assert isinstance(prompt, str)
    assert encoded["input_ids"]


@pytest.mark.parametrize("variant", ["dense", "linear", "loop"])
@pytest.mark.parametrize("legacy_rope_key", [False, True])
def test_full_config_round_trip_accepts_v5_and_legacy_rope_fields(
        variant, legacy_rope_key):
    config = make_tiny_config(variant=variant)
    config.inference_rope_scaling = True
    config.rope_scaling = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 128,
        "attention_factor": 1.0,
        "type": "yarn",
    }
    payload = config.to_dict()
    if legacy_rope_key:
        payload["rope_scaling"] = payload.pop("rope_parameters")

    rebuilt = config.__class__(**payload)

    assert rebuilt.max_position_embeddings == config.max_position_embeddings
    assert rebuilt.rope_theta == config.rope_theta
    assert rebuilt.rope_scaling["factor"] == 16
