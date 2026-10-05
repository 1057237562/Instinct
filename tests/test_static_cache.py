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


def _generate(model, ids, attention_mask=None, **kwargs):
    with torch.inference_mode():
        return model.generate(input_ids=ids, do_sample=False, attention_mask=attention_mask, **kwargs)


def _left_padded_batch(rows):
    """Stack variable-length rows into a left-padded (ids, mask) pair."""
    width = max(len(row) for row in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, width - len(row):] = row
        mask[i, width - len(row):] = 1
    return ids, mask


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


def test_static_cache_admits_single_and_left_padded_batches():
    """Eligibility: one uniform append slot per step, so left padding folds in."""
    model = _tiny_model()
    model._static_cache_ok = True

    eligible = torch.randint(3, 200, (1, 5))
    state = model._decode_state_for(eligible, torch.ones_like(eligible), 4, use_cache=True,
                                    early_exit=False, kwargs={})
    assert state is not None and len(state.cache) == model.config.num_hidden_layers
    assert state.capacity == 512  # rounded up to the first reusable bucket

    ids, mask = _left_padded_batch([torch.randint(3, 200, (5,)), torch.randint(3, 200, (8,))])
    batched = model._decode_state_for(ids, mask, 4, use_cache=True, early_exit=False, kwargs={})
    assert batched is not None and batched.batch_size == 2
    assert batched.cache.pad_starts.flatten().tolist() == [3, 0]

    # A single left-padded row qualifies through the same scan.
    single = model._decode_state_for(ids[:1], mask[:1], 4, use_cache=True,
                                     early_exit=False, kwargs={})
    assert single is not None and single.cache.pad_starts.flatten().tolist() == [3]

    # Interior holes, right padding and all-pad rows must not reach the fast path.
    holed = mask.clone()
    holed[0, 5] = 0
    right_padded = torch.ones(1, 8, dtype=torch.long)
    right_padded[0, 6:] = 0
    for bad in (holed, right_padded, torch.zeros(1, 8, dtype=torch.long)):
        assert model._decode_state_for(torch.randint(3, 200, bad.shape), bad, 4,
                                       use_cache=True, early_exit=False, kwargs={}) is None
    # A mask whose width disagrees with the prompt is a caller bug: fall back.
    assert model._decode_state_for(torch.randint(3, 200, (1, 5)),
                                   torch.ones(1, 7, dtype=torch.long), 4, use_cache=True,
                                   early_exit=False, kwargs={}) is None


def test_batched_static_path_has_a_kill_switch(monkeypatch):
    """INSTINCT_BATCH_STATIC_DECODE=0 restores the old batch>1 rejection alone."""
    model = _tiny_model()
    model._static_cache_ok = True
    ids, mask = _left_padded_batch([torch.randint(3, 200, (5,)), torch.randint(3, 200, (8,))])

    assert model._decode_state_for(ids, mask, 4, use_cache=True, early_exit=False,
                                   kwargs={}) is not None
    monkeypatch.setenv('INSTINCT_BATCH_STATIC_DECODE', '0')
    assert model._decode_state_for(ids, mask, 4, use_cache=True, early_exit=False,
                                   kwargs={}) is None
    assert model._decode_state_for(torch.randint(3, 200, (1, 5)), None, 4, use_cache=True,
                                   early_exit=False, kwargs={}) is not None


def test_left_pad_starts_flags_only_left_padding():
    """The admission scan accepts zeros-then-ones rows and nothing looser."""
    from model.static_cache import left_pad_starts

    starts, ok = left_pad_starts(torch.ones(2, 6, dtype=torch.long))
    assert ok and starts.flatten().tolist() == [0, 0]

    starts, ok = left_pad_starts(torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]))
    assert ok and starts.flatten().tolist() == [2, 0]

    holed = torch.tensor([[0, 0, 1, 1, 1]])
    holed[0, 3] = 0
    assert left_pad_starts(holed)[1] is False
    assert left_pad_starts(torch.tensor([[1, 1, 1, 0, 0]]))[1] is False
    assert left_pad_starts(torch.zeros(1, 4, dtype=torch.long))[1] is False


def test_batched_mask_hides_pad_prefix():
    """The mask drops each row's pad prefix on top of the position cutoff."""
    cache = StaticKVCache(8, [(2, 4)], torch.float32, torch.device('cpu'), batch_size=2)
    cache.pad_starts.copy_(torch.tensor([[3], [1]]))

    # One decode query per row at slot 5: row 0 sees slots 3..5, row 1 sees 1..5.
    mask = cache.mask_for(torch.full((2, 1), 5, dtype=torch.long))
    assert mask.shape == (2, 1, 1, 8)
    assert mask[0, 0, 0].tolist() == [False] * 3 + [True] * 3 + [False] * 2
    assert mask[1, 0, 0].tolist() == [False] + [True] * 5 + [False] * 2

    # Pad slots stay hidden during a prompt-width prefill as well.
    prompt_mask = cache.mask_for(torch.arange(6).unsqueeze(0).expand(2, -1))
    assert prompt_mask[0, 0, 4].tolist() == [False] * 3 + [True] * 2 + [False] * 3
    assert prompt_mask[1, 0, 4].tolist() == [False] + [True] * 4 + [False] * 3


def test_batched_append_scatters_per_row():
    """Writes land at each row's own slot; untouched slots stay zero."""
    cache = StaticKVCache(8, [(2, 4)], torch.float32, torch.device('cpu'), batch_size=2)
    layer = cache[0]
    assert layer.key.shape == (2, 8, 2, 4)

    keys = torch.arange(2 * 1 * 2 * 4, dtype=torch.float32).reshape(2, 1, 2, 4)
    layer.append(keys, keys + 100, torch.tensor([[5], [2]]))
    torch.testing.assert_close(layer.key[0, 5], keys[0, 0])
    torch.testing.assert_close(layer.key[1, 2], keys[1, 0])
    torch.testing.assert_close(layer.key[0, 2], torch.zeros(2, 4))

    # Prompt-width writes land per row as well.
    block = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
    layer.append(block, block + 100, torch.tensor([[0, 1, 2], [5, 6, 7]]))
    torch.testing.assert_close(layer.key[1, 7], block[1, 2])
    torch.testing.assert_close(layer.key[0, 0], block[0, 0])


def test_batched_state_pool_is_keyed_per_batch_size():
    """Batches of different sizes never share buffers, single-seq is untouched."""
    model = _tiny_model()
    model._static_cache_ok = True
    ids2, mask2 = _left_padded_batch([torch.randint(3, 200, (6,))] * 2)
    ids3, mask3 = _left_padded_batch([torch.randint(3, 200, (6,))] * 3)

    state2 = model._decode_state_for(ids2, mask2, 4, use_cache=True, early_exit=False, kwargs={})
    state3 = model._decode_state_for(ids3, mask3, 4, use_cache=True, early_exit=False, kwargs={})
    assert state2 is not state3
    assert state2.batch_size == 2 and state3.batch_size == 3
    assert set(model._decode_states_batched) == {2, 3}

    again = model._decode_state_for(ids2, mask2, 4, use_cache=True, early_exit=False, kwargs={})
    assert again is state2
    assert model._decode_states == {}


def test_batched_static_matches_growing_cache_on_cpu():
    """No CUDA needed for the batched bookkeeping to agree with the legacy loop."""
    model = _tiny_model()
    torch.manual_seed(17)
    ids, mask = _left_padded_batch([torch.randint(3, 200, (5,)), torch.randint(3, 200, (8,)),
                                    torch.randint(3, 200, (6,))])
    with torch.inference_mode():
        model._static_cache_ok = False
        expected = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8,
                                  do_sample=False, eos_token_id=None)
        model._static_cache_ok = True
        actual = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=8,
                                do_sample=False, eos_token_id=None)
    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
def test_batched_static_matches_growing_cache():
    """Greedy output for a left-padded batch is identical on both paths."""
    model = _tiny_model().cuda()
    torch.manual_seed(11)
    ids, mask = _left_padded_batch([torch.randint(3, 200, (5,), device='cuda'),
                                    torch.randint(3, 200, (9,), device='cuda'),
                                    torch.randint(3, 200, (7,), device='cuda')])
    ids, mask = ids.cuda(), mask.cuda()

    model._static_cache_ok = False
    expected = _generate(model, ids, attention_mask=mask, max_new_tokens=12, eos_token_id=None)
    model._static_cache_ok = True
    actual = _generate(model, ids, attention_mask=mask, max_new_tokens=12, eos_token_id=None)

    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
def test_batched_static_matches_row_by_row_single_sequence():
    """With no padding a batch must equal the rows run one at a time."""
    model = _tiny_model().cuda()
    torch.manual_seed(12)
    rows = [torch.randint(3, 200, (6,), device='cuda') for _ in range(3)]
    ids = torch.stack(rows)

    model._static_cache_ok = True
    batched = _generate(model, ids, max_new_tokens=10, eos_token_id=None)
    for row, out in zip(rows, batched):
        single = _generate(model, row.unsqueeze(0), max_new_tokens=10, eos_token_id=None)
        torch.testing.assert_close(out.unsqueeze(0), single)


@pytest.mark.gpu
def test_batched_mixed_eos_matches_growing_cache():
    """Rows that stop early must not disturb the rows still decoding."""
    model = _tiny_model().cuda()
    torch.manual_seed(16)
    ids, mask = _left_padded_batch([torch.randint(3, 200, (5,), device='cuda'),
                                    torch.randint(3, 200, (7,), device='cuda')])
    ids, mask = ids.cuda(), mask.cuda()
    eos = 7

    model._static_cache_ok = False
    expected = _generate(model, ids, attention_mask=mask, max_new_tokens=24, eos_token_id=eos)
    model._static_cache_ok = True
    actual = _generate(model, ids, attention_mask=mask, max_new_tokens=24, eos_token_id=eos)

    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
def test_batched_growth_matches_growing_cache():
    """Crossing a bucket mid-batch reparses every row and preserves the output."""
    model = _tiny_model(max_position_embeddings=2048).cuda()
    torch.manual_seed(13)
    ids, mask = _left_padded_batch([torch.randint(3, 200, (508,), device='cuda'),
                                    torch.randint(3, 200, (509,), device='cuda')])
    ids, mask = ids.cuda(), mask.cuda()

    model._static_cache_ok = False
    expected = _generate(model, ids, attention_mask=mask, max_new_tokens=6, eos_token_id=None)

    model._static_cache_ok = True
    model._static_cache_decode_headroom = 1
    actual = _generate(model, ids, attention_mask=mask, max_new_tokens=6, eos_token_id=None)

    torch.testing.assert_close(actual, expected)
    assert set(model._decode_states_batched[2]) == {512, 1024}


@pytest.mark.gpu
def test_batched_spill_matches_growing_cache():
    """A bucket that fails the memory guard spills the generation to eager."""
    model = _tiny_model(max_position_embeddings=2048).cuda()
    torch.manual_seed(14)
    ids, mask = _left_padded_batch([torch.randint(3, 200, (508,), device='cuda'),
                                    torch.randint(3, 200, (509,), device='cuda')])
    ids, mask = ids.cuda(), mask.cuda()

    model._static_cache_ok = False
    expected = _generate(model, ids, attention_mask=mask, max_new_tokens=6, eos_token_id=None)

    calls = {'n': 0}
    real_mem_get_info = torch.cuda.mem_get_info

    def guarded(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] > 1:
            return 0, 1 << 34  # growth: nothing free
        return real_mem_get_info(*args, **kwargs)

    torch.cuda.mem_get_info = guarded
    try:
        model._static_cache_ok = True
        model._static_cache_decode_headroom = 1
        actual = _generate(model, ids, attention_mask=mask, max_new_tokens=6, eos_token_id=None)
    finally:
        torch.cuda.mem_get_info = real_mem_get_info

    torch.testing.assert_close(actual, expected)
    assert calls['n'] > 1  # the guard rejected the growth
    assert set(model._decode_states_batched[2]) == {512}  # no second bucket was kept


@pytest.mark.gpu
def test_batched_captured_graph_matches_eager():
    """Capturing the decode step at batch>1 must not change the tokens."""
    from model.inference_runtime import optimize_inference

    model = _tiny_model().cuda().to(torch.bfloat16)
    optimize_inference(model, 'auto')
    torch.manual_seed(15)
    ids, mask = _left_padded_batch([torch.randint(3, 200, (5,), device='cuda'),
                                    torch.randint(3, 200, (7,), device='cuda')])
    ids, mask = ids.cuda(), mask.cuda()

    captured_calls = []
    original = model._capture_decode_step

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        captured_calls.append(result is not None)
        return result

    model._capture_decode_step = recording
    captured = _generate(model, ids, attention_mask=mask, max_new_tokens=12, eos_token_id=None)
    assert captured_calls == [True], 'capture was skipped, so this proves nothing'

    model._capture_decode_step = lambda *args, **kwargs: None  # force the eager step
    eager = _generate(model, ids, attention_mask=mask, max_new_tokens=12, eos_token_id=None)

    torch.testing.assert_close(captured, eager)


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
    longer = model._decode_state_for(torch.randint(3, 200, (1, 300)),
                                     torch.ones(1, 300, dtype=torch.long), 64,
                                     use_cache=True, early_exit=False, kwargs={})
    assert longer is first
    assert max(model._decode_states) == 512  # ladder covers it, nothing bigger queued

    # Outgrowing it allocates one bigger state and drops the smaller one; two
    # capacities is the cap, so memory and graph pools stay bounded.
    bigger = model._decode_state_for(torch.randint(3, 200, (1, 400)),
                                     torch.ones(1, 400, dtype=torch.long), 2000,
                                     use_cache=True, early_exit=False, kwargs={})
    assert bigger is not first and bigger.capacity == 4096
    assert set(model._decode_states) <= {512, 4096}
    assert len(model._decode_states) == 2

    # Past the ladder there is no bucket: the growing cache takes over.
    assert model._decode_state_for(torch.randint(3, 200, (1, 5)),
                                   torch.ones(1, 5, dtype=torch.long), 40000,
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
