"""The preallocated in-place KV cache must be indistinguishable from the growing one.

These tests pin the properties the decode fast path depends on: identical
tokens (including the EOS/overshoot protocol), a mask that hides the untouched
buffer tail, and the graph-friendly shapes that CUDA capture needs.
"""

import pytest
import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from model.static_cache import StaticKVCache
from tests.helpers import make_tiny_config


def _tiny_model(**overrides):
    config = make_tiny_config(use_moe=True)
    for key, value in overrides.items():
        setattr(config, key, value)
    model = InstinctForCausalLM(config).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    return model


def _generate(model, ids, **kwargs):
    with torch.inference_mode():
        return model.generate(input_ids=ids, do_sample=False, attention_mask=None, **kwargs)


@pytest.mark.gpu
def test_static_cache_matches_growing_cache():
    """Greedy output is unchanged by decoding into a preallocated buffer."""
    model = _tiny_model().cuda()
    torch.manual_seed(1)
    ids = torch.randint(3, 200, (1, 6), device='cuda')

    model._static_cache_ok = False
    expected = _generate(model, ids, max_new_tokens=12, eos_token_id=None)
    model._static_cache_ok = True
    actual = _generate(model, ids, max_new_tokens=12, eos_token_id=None)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 18)


@pytest.mark.gpu
def test_static_cache_keeps_eos_and_overshoot_protocol():
    """Stopping on EOS trims the same tokens as the growing-cache loop.

    The two loops write the stop token at different points in the iteration, so
    this is the check that the emitted sequence and length still agree.
    """
    model = _tiny_model().cuda()
    torch.manual_seed(2)
    ids = torch.randint(3, 200, (1, 5), device='cuda')
    eos = 7
    model._static_cache_ok = False
    expected = _generate(model, ids, max_new_tokens=16, eos_token_id=eos)
    model._static_cache_ok = True
    actual = _generate(model, ids, max_new_tokens=16, eos_token_id=eos)

    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
def test_static_cache_matches_growing_cache_with_repetition_penalty():
    """The penalty reads the whole prefix, which the static loop slices itself."""
    model = _tiny_model().cuda()
    torch.manual_seed(3)
    ids = torch.randint(3, 200, (1, 6), device='cuda')

    model._static_cache_ok = False
    expected = _generate(model, ids, max_new_tokens=10, eos_token_id=None, repetition_penalty=1.3)
    model._static_cache_ok = True
    actual = _generate(model, ids, max_new_tokens=10, eos_token_id=None, repetition_penalty=1.3)

    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
def test_captured_decode_step_matches_eager():
    """Capturing the decode step as a CUDA graph must not change the tokens."""
    from model.inference_runtime import optimize_inference

    model = _tiny_model().cuda().to(torch.bfloat16)
    torch.manual_seed(4)
    ids = torch.randint(3, 200, (1, 6), device='cuda')
    # Installs the sync-free expert dispatch, which is what makes capture legal.
    optimize_inference(model, 'auto')

    captured_calls = []
    original = model._capture_decode_step

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        captured_calls.append(result is not None)
        return result

    model._capture_decode_step = recording
    captured = _generate(model, ids, max_new_tokens=12, eos_token_id=None)
    assert captured_calls == [True], 'capture was skipped, so this proves nothing'

    model._capture_decode_step = lambda *args, **kwargs: None  # force the eager step
    eager = _generate(model, ids, max_new_tokens=12, eos_token_id=None)

    torch.testing.assert_close(captured, eager)


@pytest.mark.gpu
def test_synchronizing_dispatch_skips_capture():
    """A path with host syncs must not attempt capture, which would poison the stream.

    The per-expert loop calls ``nonzero`` per expert; capturing it aborts the
    stream and latches an error for the next launch.
    """
    model = _tiny_model().cuda()  # fp32: no grouped backend, so the loop is used
    torch.manual_seed(5)
    ids = torch.randint(3, 200, (1, 6), device='cuda')
    model._static_cache_ok = True
    assert model._decode_is_capturable() is False

    attempts = []
    original = model._capture_decode_step

    def recording(*args, **kwargs):
        attempts.append(True)
        return original(*args, **kwargs)

    model._capture_decode_step = recording
    with torch.inference_mode():
        model.generate(input_ids=ids, max_new_tokens=4, do_sample=False, eos_token_id=None)
    assert attempts == [True]  # called, but it must decline before touching the stream
    assert torch.equal(torch.zeros(1, device='cuda'), torch.zeros(1, device='cuda'))  # context intact


def test_static_cache_write_and_mask():
    """The buffer is written where asked, and the mask hides everything after."""
    cache = StaticKVCache(8, [(2, 4)], torch.float32, torch.device('cpu'))
    layer = cache[0]
    assert layer.key.shape == (1, 8, 2, 4)

    keys = torch.full((1, 3, 2, 4), 5.0)
    values = torch.full((1, 3, 2, 4), 6.0)
    written_keys, written_values = layer.append(keys, values, torch.tensor([2, 3, 4]))
    torch.testing.assert_close(written_keys[:, 2:5], keys)
    torch.testing.assert_close(written_values[:, 2:5], values)
    torch.testing.assert_close(written_keys[:, 5:], torch.zeros(1, 3, 2, 4))

    # One query at position 3 sees slots 0..3 and nothing after.
    mask = cache.mask_for(torch.tensor([[3]]))
    assert mask.shape == (1, 1, 1, 8)
    assert mask.flatten().tolist() == [True] * 4 + [False] * 4

    # A prompt masks causally as well.
    prompt_mask = cache.mask_for(torch.arange(3).unsqueeze(0))
    assert prompt_mask.shape == (1, 1, 3, 8)
    assert prompt_mask[0, 0].tolist() == [
        [True, False, False, False, False, False, False, False],
        [True, True, False, False, False, False, False, False],
        [True, True, True, False, False, False, False, False],
    ]


def test_static_cache_rejects_batched_or_padded_generation():
    """Eligibility: the in-place append writes one slot per step."""
    model = _tiny_model()
    model._static_cache_ok = True
    ids = torch.randint(3, 200, (2, 5))

    assert model._decode_state_for(ids, torch.ones_like(ids), 4, use_cache=True,
                                   early_exit=False, kwargs={}) is None
    padded = torch.ones(1, 5, dtype=torch.long)
    padded[0, 0] = 0
    assert model._decode_state_for(torch.randint(3, 200, (1, 5)), padded, 4, use_cache=True,
                                   early_exit=False, kwargs={}) is None
    eligible = torch.randint(3, 200, (1, 5))
    state = model._decode_state_for(eligible, torch.ones_like(eligible), 4, use_cache=True,
                                    early_exit=False, kwargs={})
    assert state is not None and len(state.cache) == model.config.num_hidden_layers
    assert state.capacity == 512  # rounded up to the first reusable bucket


def test_decode_state_is_reused_across_turns():
    """A repeat call keeps the same buffers, so compiles and captures survive."""
    model = _tiny_model(max_position_embeddings=8192)
    model._static_cache_ok = True
    ids = torch.randint(3, 200, (1, 40))
    mask = torch.ones_like(ids)

    first = model._decode_state_for(ids, mask, 64, use_cache=True, early_exit=False, kwargs={})
    again = model._decode_state_for(ids, mask, 64, use_cache=True, early_exit=False, kwargs={})
    assert again is first

    # A longer turn that still fits reuses the same capacity, and therefore the
    # same compiled graph and recorded CUDA graph.
    longer = model._decode_state_for(torch.randint(3, 200, (1, 300)), mask, 64,
                                     use_cache=True, early_exit=False, kwargs={})
    assert longer is first
    assert max(model._decode_states) == 512  # ladder covers it, nothing bigger queued

    # Outgrowing it allocates one bigger state and drops the smaller one; two
    # capacities is the cap, so memory and graph pools stay bounded.
    bigger = model._decode_state_for(torch.randint(3, 200, (1, 400)), mask, 2000,
                                     use_cache=True, early_exit=False, kwargs={})
    assert bigger is not first and bigger.capacity == 4096
    assert set(model._decode_states) <= {512, 4096}
    assert len(model._decode_states) == 2

    # Past the ladder there is no bucket: the growing cache takes over.
    assert model._decode_state_for(torch.randint(3, 200, (1, 5)), mask, 40000,
                                   use_cache=True, early_exit=False, kwargs={}) is None


def test_large_output_limit_only_plans_initial_decode_headroom():
    """The output allowance must not force attention over a huge empty cache."""
    model = _tiny_model(max_position_embeddings=32768)
    model._static_cache_ok = True
    ids = torch.randint(3, 200, (1, 32))
    calls = []
    original = model._decode_state_for

    def recording(input_ids, attention_mask, max_new_tokens, **kwargs):
        calls.append(max_new_tokens)
        return original(input_ids, attention_mask, max_new_tokens, **kwargs)

    model._decode_state_for = recording
    # Stop after state selection; this test checks planning, not 16K decoding.
    model._decode_with_static_cache = lambda *args, **kwargs: 0
    output = _generate(model, ids, max_new_tokens=16384, eos_token_id=None)

    assert calls == [256]
    assert output.shape == ids.shape
    assert set(model._decode_states) == {512}


@pytest.mark.gpu
def test_static_cache_grows_without_changing_generated_tokens():
    """Crossing a bucket reparses the prefix and preserves the output exactly."""
    model = _tiny_model(max_position_embeddings=2048).cuda()
    torch.manual_seed(10)
    ids = torch.randint(3, 200, (1, 510), device='cuda')

    model._static_cache_ok = False
    expected = _generate(model, ids, max_new_tokens=4, eos_token_id=None)

    model._static_cache_ok = True
    model._static_cache_decode_headroom = 1
    actual = _generate(model, ids, max_new_tokens=4, eos_token_id=None)

    torch.testing.assert_close(actual, expected)
    assert set(model._decode_states) == {512, 1024}


@pytest.mark.gpu
def test_existing_plain_graph_is_not_replayed_for_layer_callback():
    """Logit-lens callbacks must run even after a plain graph was recorded."""
    from model.inference_runtime import optimize_inference

    model = _tiny_model().cuda().to(torch.bfloat16)
    optimize_inference(model, 'auto')
    ids = torch.randint(3, 200, (1, 6), device='cuda')
    _generate(model, ids, max_new_tokens=4, eos_token_id=None)
    state = next(iter(model._decode_states.values()))
    assert state.graph is not None

    calls = []
    _generate(model, ids, max_new_tokens=4, eos_token_id=None,
              layer_callback=lambda *args: calls.append(True))

    assert len(calls) == model.config.num_hidden_layers * 4


@pytest.mark.gpu
def test_webui_pad_token_uses_warmed_decode_graph():
    """WebUI generation metadata must not disable an already warmed graph."""
    from model.inference_runtime import optimize_inference, warmup_decode

    model = _tiny_model().cuda().to(torch.bfloat16)
    optimize_inference(model, 'auto')
    warmup_decode(model)
    state = next(iter(model._decode_states.values()))
    assert state.graph is not None
    graph, outputs = state.graph

    class CountReplays:
        count = 0

        def replay(self):
            self.count += 1
            graph.replay()

    counted = CountReplays()
    state.graph = counted, outputs
    ids = torch.randint(3, 200, (1, 6), device='cuda')
    expected = _generate(model, ids, max_new_tokens=4, eos_token_id=None)
    counted.count = 0
    actual = _generate(model, ids, max_new_tokens=4, eos_token_id=None,
                       pad_token_id=0, num_return_sequences=1, stream_chunk_size=16)
    torch.testing.assert_close(actual, expected)
    assert counted.count == 3, 'pad_token_id silently forced eager decode'


@pytest.mark.gpu
def test_reused_state_matches_growing_cache_across_turns():
    """A second turn on a recycled cache must match the growing-cache loop.

    The reused buffer still holds the previous turn's keys, so the mask has to be
    the only thing separating them: this runs a long prompt and then a short one
    (leaving stale slots behind) and compares both against the legacy path.
    """
    model = _tiny_model(max_position_embeddings=8192).cuda()
    torch.manual_seed(9)
    long_prompt = torch.randint(3, 200, (1, 96), device='cuda')
    short_prompt = torch.randint(3, 200, (1, 12), device='cuda')

    model._static_cache_ok = False
    expected_long = _generate(model, long_prompt, max_new_tokens=10, eos_token_id=None)
    expected_short = _generate(model, short_prompt, max_new_tokens=10, eos_token_id=None)
    expected_again = _generate(model, long_prompt, max_new_tokens=10, eos_token_id=None)

    model._static_cache_ok = True
    actual_long = _generate(model, long_prompt, max_new_tokens=10, eos_token_id=None)
    state = model._decode_states[max(model._decode_states)]
    actual_short = _generate(model, short_prompt, max_new_tokens=10, eos_token_id=None)
    assert model._decode_states[max(model._decode_states)] is state, 'state was not reused'
    actual_again = _generate(model, long_prompt, max_new_tokens=10, eos_token_id=None)

    torch.testing.assert_close(actual_long, expected_long)
    torch.testing.assert_close(actual_short, expected_short)
    torch.testing.assert_close(actual_again, expected_again)


@pytest.mark.gpu
def test_second_turn_reuses_the_recorded_graph():
    """Turn two replays the graph recorded on turn one instead of re-capturing."""
    from model.inference_runtime import optimize_inference

    model = _tiny_model().cuda().to(torch.bfloat16)
    optimize_inference(model, 'auto')
    torch.manual_seed(7)
    mask = None

    calls = []
    original = model._capture_decode_step

    def recording(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    model._capture_decode_step = recording
    first = _generate(model, torch.randint(3, 200, (1, 6), device='cuda'),
                      max_new_tokens=10, eos_token_id=None)
    captures_after_first = len(calls)
    assert captures_after_first == 1

    second = _generate(model, torch.randint(3, 200, (1, 6), device='cuda'),
                       max_new_tokens=10, eos_token_id=None)
    assert len(calls) == captures_after_first, 'the second turn re-captured instead of reusing'
    assert first.shape == second.shape == (1, 16)


@pytest.mark.gpu
def test_inference_dispatch_is_installed_by_optimize_inference():
    """optimize_inference resolves the backend and prebuilds the expert stacks."""
    from model.inference_runtime import optimize_inference

    model = _tiny_model().cuda().to(torch.bfloat16)
    layer = model.model.layers[0].mlp
    assert layer._inference_backend is None and layer._inference_stacked is None

    optimize_inference(model, 'auto')

    assert layer._inference_backend in ('native', 'triton', 'cached')
    stacks = layer._inference_stacked
    assert stacks is not None and len(stacks) == 3
    for stack, name in zip(stacks, ('gate_proj', 'up_proj', 'down_proj')):
        expected = torch.stack([e.__getattr__(name).weight for e in layer.experts], 0).transpose(1, 2)
        torch.testing.assert_close(stack, expected.to(torch.bfloat16))

    # A weight update must be reflected on the next eager forward.
    with torch.no_grad():
        layer.experts[0].gate_proj.weight.add_(1.0)
    model(torch.randint(3, 200, (1, 4), device='cuda'), use_cache=False)
    assert not torch.equal(layer._inference_stacked[0], stacks[0])
