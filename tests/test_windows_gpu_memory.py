import pytest
from trainer import windows_gpu_memory as memory


def test_driver_budget_subtracts_only_nonallocator_usage_beyond_safety():
    gib = 2**30
    info = dict(budget=14*gib, current_usage=4*gib)
    target = memory.training_memory_target(16,16*gib,reserved_bytes=3*gib,driver_info=info)
    assert target == 13.5
    # The allocator then subtracts .5 GiB: 13 GiB + 1 GiB outside it = 14.
    assert memory.training_memory_target(12,16*gib,reserved_bytes=3*gib,driver_info=info)==12


def test_other_processes_limit_reusable_cuda_memory():
    gib = 2**30
    assert memory.training_memory_target(16,16*gib,free_bytes=8*gib,reserved_bytes=3*gib)==11.5
    assert memory.training_memory_target(24,16*gib)==16


def test_adapter_matching_falls_back_on_duplicates_and_failures(monkeypatch):
    monkeypatch.setattr(memory,'_IS_WINDOWS',True)
    row=dict(name='test GPU',budget=14*2**30,current_usage=0)
    monkeypatch.setattr(memory,'query_budgets',lambda:[row])
    assert memory.local_memory_budget('test GPU')==row
    assert memory.local_memory_budget('different GPU') is None
    monkeypatch.setattr(memory,'query_budgets',lambda:[row,row])
    assert memory.local_memory_budget('test GPU') is None
    def fail(): raise OSError('DXGI unavailable')
    monkeypatch.setattr(memory,'query_budgets',fail)
    assert memory.local_memory_budget('test GPU') is None
