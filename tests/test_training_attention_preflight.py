"""A missing optional FA4 package must not leave checkpointed layers eager."""
import copy
import importlib
from types import SimpleNamespace

import torch

from model.model_instinct import InstinctConfig
from trainer.trainer_utils import init_model


def test_first_packed_compile_contains_checkpointed_trunk(monkeypatch):
    flash = importlib.import_module('model.flash_attn_4')
    monkeypatch.setattr(flash, '_tried_import', False)
    monkeypatch.setattr(flash, '_flash_attn_4_func', None)
    monkeypatch.setattr('trainer.trainer_utils.AutoTokenizer.from_pretrained',
                        lambda *a, **kw: SimpleNamespace())
    cfg = InstinctConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         vocab_size=128, max_position_embeddings=64,
                         flash_attn=True, use_grad_checkpoint=2)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch._dynamo.reset()
    try:
        model, _ = init_model(cfg, from_weight='none', device='cpu')
        assert flash._tried_import
        reference = copy.deepcopy(model)
        graphs = []
        def backend(gm, inputs):
            graphs.append(gm)
            return gm.forward
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        tokens = torch.randint(1, 128, (2, 32))
        ids = (torch.arange(32)//8)[None].expand(2, -1)
        actual = compiled(tokens, labels=tokens, sequence_ids=ids)
        actual.loss.backward()
        expected = reference(tokens, labels=tokens, sequence_ids=ids)
        expected.loss.backward()
        assert len(graphs) == 1
        assert 'tag_activation_checkpoint' in graphs[0].code
        torch.testing.assert_close(actual.loss, expected.loss)
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
    finally:
        torch._dynamo.reset()
        torch.set_num_threads(previous_threads)
