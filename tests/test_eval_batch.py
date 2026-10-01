from types import SimpleNamespace

from datasets import load_dataset  # noqa: F401
import torch
import pytest

from eval_batch import _eval_kv_cache_precision, generate_batches


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        return {'input_ids': torch.tensor([[int(x) for x in text.split()]])}

    def decode(self, tokens, **kwargs):
        return ' '.join(str(x) for x in tokens.tolist() if x not in (0, 2))


def args():
    return SimpleNamespace(device='cpu', temperature=0, max_new_tokens=3,
                           top_p=1, early_exit=0)


def test_batches_left_pad_slice_and_eos():
    class Model:
        calls = []

        def generate(self, inputs, attention_mask, **kwargs):
            self.calls.append((inputs.tolist(), attention_mask.tolist()))
            return torch.cat([inputs, torch.tensor([[int(x[-1]) + 1, 2, 99] for x in inputs])], dim=1)

    model = Model()
    jobs = [dict(index=i, text=text) for i, text in enumerate(['4', '5 6', '7 8 9'])]
    results = list(generate_batches(args(), model, Tokenizer(), jobs, 2, 42))
    assert model.calls == [([[0, 4], [5, 6]], [[0, 1], [1, 1]]),
                           ([[7, 8, 9]], [[1, 1, 1]])]
    assert [r[1] for r in results] == ['5', '7', '10']
    assert [r[2] for r in results] == [2, 2, 2]
    assert [r[0]['index'] for r in results] == [0, 1, 2]


@pytest.mark.parametrize('flash', [False, True])
def test_native_dense_greedy_batch_matches_single(flash):
    from model.model_instinct import InstinctConfig, InstinctForCausalLM
    torch.manual_seed(7)
    model = InstinctForCausalLM(InstinctConfig(
        hidden_size=32, num_hidden_layers=2, vocab_size=64,
        num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, max_position_embeddings=64, flash_attn=flash)).eval()
    jobs = [dict(index=0, text='4 5'), dict(index=1, text='6 7 8 9')]
    singles = list(generate_batches(args(), model, Tokenizer(), jobs, 1, 42))
    batched = list(generate_batches(args(), model, Tokenizer(), jobs, 2, 42))
    assert [r[1] for r in singles] == [r[1] for r in batched]


def test_linear_uses_equal_length_groups():
    class Model:
        __module__ = 'model.model_instinct_linear'
        sizes = []

        def generate(self, inputs, attention_mask, **kwargs):
            assert attention_mask.all()
            self.sizes.append(len(inputs))
            return torch.cat([inputs, torch.ones((len(inputs), 1), dtype=torch.long)], dim=1)

    model = Model()
    jobs = [dict(index=i, text=t) for i, t in enumerate(['4', '4 5', '6'])]
    results = list(generate_batches(args(), model, Tokenizer(), jobs, 3, 42))
    assert model.sizes == [2, 1]
    assert sorted(r[0]['index'] for r in results) == [0, 1, 2]


@pytest.mark.gpu
def test_eval_batch_uses_fast_kv_precision_only_with_memory_headroom(monkeypatch):
    from model.model_instinct import InstinctConfig, InstinctForCausalLM

    config = InstinctConfig(hidden_size=32, num_hidden_layers=2, vocab_size=64,
                            num_attention_heads=4, num_key_value_heads=2,
                            intermediate_size=64, max_position_embeddings=512,
                            kv_cache_dtype='fp8_e5m2')
    model = InstinctForCausalLM(config).eval().to(torch.bfloat16).cuda()
    model._static_cache_ok = True
    layers = [layer.self_attn for layer in model.model.layers]

    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (2**30, 2**30))
    with _eval_kv_cache_precision(model, 4, 64, 128) as dtype:
        assert dtype == 'bf16'
        assert all(layer.kv_cache_dtype == 'bf16' for layer in layers)
    assert all(layer.kv_cache_dtype == 'fp8_e5m2' for layer in layers)

    with _eval_kv_cache_precision(model, 4, 64, 128, policy='configured') as dtype:
        assert dtype == 'fp8_e5m2'
        assert all(layer.kv_cache_dtype == 'fp8_e5m2' for layer in layers)

    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (1, 2**30))
    with _eval_kv_cache_precision(model, 4, 64, 128) as dtype:
        assert dtype == 'fp8_e5m2'
    assert all(layer.kv_cache_dtype == 'fp8_e5m2' for layer in layers)
