"""Read DXGI's per-process GPU memory budget (no reservation or state changes)."""
import ctypes as c
import os
import uuid

_IS_WINDOWS = os.name == 'nt'


def query_budgets():
    if not _IS_WINDOWS:
        return []
    class GUID(c.Structure):
        _fields_ = [('data', c.c_ubyte * 16)]
    class LUID(c.Structure):
        _fields_ = [('low', c.c_uint32), ('high', c.c_int32)]
    class Desc(c.Structure):
        _fields_ = [('name', c.c_wchar * 128), ('vendor', c.c_uint32),
                    ('device', c.c_uint32), ('subsystem', c.c_uint32), ('revision', c.c_uint32),
                    ('video', c.c_size_t), ('system', c.c_size_t), ('shared', c.c_size_t),
                    ('luid', LUID), ('flags', c.c_uint32)]
    class Memory(c.Structure):
        _fields_ = [(name, c.c_uint64) for name in
                    ('budget', 'current_usage', 'available_reservation', 'current_reservation')]
    def guid(value):
        return GUID((c.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le))
    def call(ptr, index, types, *args):
        table = c.cast(ptr, c.POINTER(c.POINTER(c.c_void_p))).contents
        return c.WINFUNCTYPE(c.c_int32, c.c_void_p, *types)(table[index])(ptr, *args)
    factory = c.c_void_p()
    create = c.WinDLL('dxgi').CreateDXGIFactory1
    create.argtypes = [c.POINTER(GUID), c.POINTER(c.c_void_p)]
    create.restype = c.c_int32
    iid = guid('770aae78-f26f-4dba-a829-253c83d1b387')
    if create(c.byref(iid), c.byref(factory)) < 0:
        return []
    rows = []
    try:
        for index in range(16):
            adapter, newer = c.c_void_p(), c.c_void_p()
            if call(factory, 12, [c.c_uint32, c.POINTER(c.c_void_p)], index, c.byref(adapter)) < 0:
                break
            try:
                desc = Desc()
                if call(adapter, 10, [c.POINTER(Desc)], c.byref(desc)) < 0:
                    continue
                iid3 = guid('645967a4-1392-4310-a798-8053ce3e93fd')
                if call(adapter, 0, [c.POINTER(GUID), c.POINTER(c.c_void_p)],
                        c.byref(iid3), c.byref(newer)) < 0:
                    continue
                memory = Memory()
                if call(newer, 14, [c.c_uint32, c.c_uint32, c.POINTER(Memory)],
                        0, 0, c.byref(memory)) < 0:
                    continue
                rows.append(dict(name=desc.name, vendor=desc.vendor, dedicated_bytes=desc.video,
                    **{name: getattr(memory, name) for name, _ in Memory._fields_}))
            finally:
                if newer.value: call(newer, 2, [])
                if adapter.value: call(adapter, 2, [])
    finally:
        call(factory, 2, [])
    return rows


def local_memory_budget(device_name):
    """Return a unique adapter's budget; ambiguous/unsupported devices fall back.

    Match the exact CUDA name only on a single matching adapter. Two identical
    cards require a CUDA LUID mapping, so they deliberately keep the old limit.
    This query does not reserve memory or change another process's allocation.
    """
    if not device_name or not _IS_WINDOWS:
        return None
    try:
        matches = [row for row in query_budgets() if row['name'] == device_name]
    except (OSError, AttributeError, ValueError):
        return None
    if len(matches) != 1 or matches[0]['budget'] <= 0:
        return None
    return matches[0]


def training_memory_target(requested_gb, physical_bytes, *, free_bytes=None,
                           reserved_bytes=0, driver_info=None, safety_gb=.5):
    """Clamp a total-process target; callers subtract the safety reserve once.

    CUDA free + this allocator's reserved blocks describes memory this process
    can reuse. DXGI covers the process budget, including CUDA allocations that
    bypass PyTorch. Account for those beyond the ordinary safety reserve.
    """
    gib = 2**30
    target = min(float(requested_gb), physical_bytes / gib)
    if free_bytes is not None:
        target = min(target, (free_bytes + reserved_bytes) / gib + safety_gb)
    if driver_info and driver_info.get('budget', 0) > 0:
        external = max(0, driver_info.get('current_usage', 0) - reserved_bytes)
        target = min(target, driver_info['budget'] / gib - max(0, external / gib - safety_gb))
    return max(0.0, target)
