import copy
import importlib

from datasets import load_dataset  # noqa: F401
import pytest
import torch


@pytest.mark.parametrize('name', ['model_instinct', 'model_instinct_loop', 'model_instinct_linear'])
def test_cached_decode_matches_eager_and_prefill_projects_one_token(name):
    module = importlib.import_module('model.' + name)
    config = module.InstinctConfig(hidden_size=32, num_hidden_layers=2, vocab_size=64,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
        max_position_embeddings=64, flash_attn=True, full_attention_interval=1,
        loop_iters=2, state_init_std=0.0, recurrence_sampling="fixed")
    torch.manual_seed(9)
    model = module.InstinctForCausalLM(config).eval()
    reference = copy.deepcopy(model)
    for submodule in reference.modules():
        if hasattr(submodule, 'flash'):
            submodule.flash = False
    prompt = torch.tensor([[0, 0, 3, 4], [3, 4, 5, 6]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    with torch.inference_mode():
        first = model(prompt, attention_mask=mask, use_cache=True)
        old = reference(prompt, attention_mask=mask, use_cache=True)
        for i in range(3):
            mask = torch.cat([mask, torch.ones(2, 1, dtype=torch.long)], dim=1)
            next_ids = torch.tensor([[7+i], [8+i]])
            first = model(next_ids, attention_mask=mask, past_key_values=first.past_key_values, use_cache=True)
            old = reference(next_ids, attention_mask=mask, past_key_values=old.past_key_values, use_cache=True)
            torch.testing.assert_close(first.logits, old.logits, atol=2e-5, rtol=2e-4)
    widths = []
    handle = model.lm_head.register_forward_pre_hook(lambda m, inputs: widths.append(inputs[0].shape[1]))
    with torch.inference_mode():
        # Greedy generation needs neither temperature division nor top-p sorting.
        model.generate(prompt, attention_mask=mask[:, :4], max_new_tokens=3,
                       do_sample=False, temperature=0, eos_token_id=None)
    handle.remove()
    assert widths == [1, 1, 1]


def test_tokenization_reused_across_samples():
    from eval_batch import generate_batches
    from tests.test_eval_batch import Tokenizer, args
    class CountingTokenizer(Tokenizer):
        calls = 0
        def __call__(self, text, **kwargs):
            self.calls += 1
            return super().__call__(text, **kwargs)
    class Model:
        def generate(self, inputs, **kwargs):
            return torch.cat([inputs, torch.ones((len(inputs), 1), dtype=torch.long)], dim=1)
    tokenizer = CountingTokenizer()
    jobs = [dict(index=i, text='3 4') for i in range(10)]
    assert len(list(generate_batches(args(), Model(), tokenizer, jobs, 4, 42))) == 10
    assert tokenizer.calls == 1
