"""LM input copies must not drain previously queued CUDA training work."""
import pytest
import torch

from trainer.trainer_utils import prepare_lm_batch


@pytest.mark.gpu
@pytest.mark.parametrize('packed', [False, True])
def test_pinned_lm_inputs_enqueue_without_waiting_for_previous_work(packed):
    tokens = torch.arange(128, dtype=torch.long).reshape(2, 64).pin_memory()
    labels = (tokens + 1).pin_memory()
    ids = torch.arange(64).div(32, rounding_mode='floor').expand(2, -1).contiguous().pin_memory()
    batch = (tokens, labels, ids) if packed else (tokens, labels)
    stream = torch.cuda.Stream()
    # Warm the allocator and transfer path before testing a pending stream.
    with torch.cuda.stream(stream):
        warm = prepare_lm_batch(batch, 'cuda')
    stream.synchronize()
    del warm
    previous_work_done = torch.cuda.Event()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
        previous_work_done.record()
        actual = prepare_lm_batch(batch, 'cuda')
        returned_before_previous_work = not previous_work_done.query()
        # Consumption on the same stream must see the copied input values.
        consumed = actual[0] + actual[1]
    stream.synchronize()
    assert returned_before_previous_work, 'LM transfer synchronized preceding GPU work'
    assert torch.equal(consumed.cpu(), tokens + labels)
    assert torch.equal(actual[1].cpu(), labels)
    if packed:
        assert torch.equal(actual[2].cpu(), ids)
    else:
        assert actual[2] is None


@pytest.mark.gpu
def test_lm_copy_to_cpu_is_ready_on_return():
    tokens = torch.arange(128, device='cuda').reshape(2, 64)
    labels = tokens.clone()
    torch.cuda._sleep(100_000_000)
    labels.add_(7)
    cpu_tokens, cpu_labels, ids = prepare_lm_batch((tokens, labels), 'cpu')
    assert torch.equal(cpu_labels, cpu_tokens + 7)
    assert ids is None
