"""Bound rebuildable Hugging Face/Arrow dataset caches by an LRU disk budget.

Only files beneath ``HF_DATASETS_CACHE`` are eligible. Source datasets,
checkpoints, model weights, and the separate torch.compile cache are never in
scope. Active training processes publish lightweight leases so an eviction in
another process cannot remove memory-mapped Arrow files that are in use.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import time

from filelock import FileLock


GIB = 1024 ** 3
DEFAULT_CACHE_MAX_GB = 5.0
_LEASE_DIR = ".instinct-leases"
_BUDGET_LOCK = ".instinct-budget.lock"
_ACCESS_FILE = ".instinct-access.json"
_PROCESS_PATHS: dict[Path, set[str]] = {}
_ATEXIT_REGISTERED = False


class CacheBudgetExceeded(RuntimeError):
    """Raised when active/non-evictable data alone exceeds the configured cap."""


def dataset_cache_root(root=None) -> Path:
    if root is not None:
        return Path(root).resolve()
    explicit = os.environ.get("HF_DATASETS_CACHE")
    if explicit:
        return Path(explicit).resolve()
    loaded_datasets = sys.modules.get("datasets")
    configured = getattr(getattr(loaded_datasets, "config", None), "HF_DATASETS_CACHE", None)
    if configured:
        return Path(configured).resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home) / "datasets").resolve()
    return (Path(__file__).resolve().parents[2] / ".cache" / "huggingface" / "datasets").resolve()


def dataset_cache_budget_root(root=None) -> Path:
    """Root whose aggregate size is limited (workers may write in a child root)."""
    if root is not None:
        return Path(root).resolve()
    return Path(
        os.environ.get("INSTINCT_DATA_CACHE_BUDGET_ROOT", dataset_cache_root())
    ).resolve()


def configured_budget_bytes(max_gb=None) -> int | None:
    value = (
        os.environ.get("INSTINCT_DATA_CACHE_MAX_GB", str(DEFAULT_CACHE_MAX_GB))
        if max_gb is None else max_gb
    )
    value = float(value)
    if value < 0:
        raise ValueError("dataset cache budget must be >= 0 GB")
    return None if value == 0 else int(value * GIB)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _tree_size(path: Path) -> int:
    if not path.exists() or path.is_symlink():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0
    total = 0
    for base, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(base) / name).is_symlink()]
        for name in files:
            item = Path(base) / name
            if item.is_symlink():
                continue
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                pass
    return total


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _lease_dir(root: Path) -> Path:
    path = root / _LEASE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_process_lease(root: Path) -> None:
    lease_dir = _lease_dir(root)
    target = lease_dir / f"{os.getpid()}.json"
    temp = lease_dir / f"{os.getpid()}.{time.time_ns()}.tmp"
    payload = {
        "pid": os.getpid(),
        "updated_ns": time.time_ns(),
        "paths": sorted(_PROCESS_PATHS.get(root, set())),
    }
    temp.write_text(json.dumps(payload), encoding="utf-8")
    temp.replace(target)


def _read_access_times(root: Path) -> dict[str, int]:
    path = root / _ACCESS_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(key): int(value) for key, value in payload.items()}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _write_access_times(root: Path, values: dict[str, int]) -> None:
    # Drop entries whose cache objects no longer exist so the index stays tiny.
    values = {key: value for key, value in values.items() if Path(key).exists()}
    target = root / _ACCESS_FILE
    temp = root / f"{_ACCESS_FILE}.{os.getpid()}.{time.time_ns()}.tmp"
    temp.write_text(json.dumps(values), encoding="utf-8")
    temp.replace(target)


def _cleanup_process_leases() -> None:
    for root in list(_PROCESS_PATHS):
        try:
            (root / _LEASE_DIR / f"{os.getpid()}.json").unlink(missing_ok=True)
        except OSError:
            pass


def register_cache_use(paths, root=None, *, acquire_lock=True) -> None:
    """Lease cache files/directories until process exit and refresh their LRU time."""
    global _ATEXIT_REGISTERED
    root = dataset_cache_budget_root(root)
    root.mkdir(parents=True, exist_ok=True)
    resolved = []
    for value in paths:
        path = Path(value).resolve()
        if not _inside(path, root):
            raise ValueError(f"Refusing to lease cache path outside {root}: {path}")
        resolved.append(path)

    def update():
        owned = _PROCESS_PATHS.setdefault(root, set())
        owned.update(str(path) for path in resolved)
        _write_process_lease(root)
        access = _read_access_times(root)
        now = time.time_ns()
        access.update({str(path): now for path in resolved})
        _write_access_times(root, access)

    if acquire_lock:
        with FileLock(str(root / _BUDGET_LOCK)):
            update()
    else:
        update()
    if not _ATEXIT_REGISTERED:
        atexit.register(_cleanup_process_leases)
        _ATEXIT_REGISTERED = True


def release_cache_use(paths, root=None, *, acquire_lock=True) -> None:
    """Release this process's lease after a dataset transformation is complete."""
    root = dataset_cache_budget_root(root)
    released = {str(Path(value).resolve()) for value in paths}

    def update():
        owned = _PROCESS_PATHS.setdefault(root, set())
        owned.difference_update(released)
        _write_process_lease(root)

    if acquire_lock:
        with FileLock(str(root / _BUDGET_LOCK)):
            update()
    else:
        update()


def discard_rebuildable_cache(paths, root=None) -> list[str]:
    """Delete explicitly completed cache objects after all leases are released.

    Unlike enforcing a zero-byte global budget, this leaves tiny plan/lock
    metadata and unrelated caches alone. Paths must stay beneath the configured
    dataset-cache root and may not be protected by another live process.
    """
    root = dataset_cache_budget_root(root)
    candidates = []
    for value in paths:
        path = Path(value).resolve()
        if path == root or not _inside(path, root):
            raise ValueError(f"Refusing to discard cache path outside root: {path}")
        candidates.append(path)
    removed = []
    with FileLock(str(root / _BUDGET_LOCK)):
        protected = _active_lease_paths(root)
        for path in candidates:
            if _is_protected(path, protected):
                continue
            if _remove_candidate(path, root):
                removed.append(str(path))
        access = _read_access_times(root)
        _write_access_times(root, access)
    return removed


def _active_lease_paths(root: Path) -> set[Path]:
    active = set()
    lease_dir = _lease_dir(root)
    for lease in lease_dir.glob("*.json"):
        if lease.is_symlink():
            continue
        try:
            payload = json.loads(lease.read_text(encoding="utf-8"))
            pid = int(payload["pid"])
            paths = payload.get("paths", [])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            lease.unlink(missing_ok=True)
            continue
        if not _pid_alive(pid):
            lease.unlink(missing_ok=True)
            continue
        for value in paths:
            path = Path(value).resolve()
            if _inside(path, root):
                active.add(path)
    return active


def _is_protected(path: Path, protected: set[Path]) -> bool:
    resolved = path.resolve()
    for item in protected:
        try:
            if resolved == item or resolved.is_relative_to(item) or item.is_relative_to(resolved):
                return True
        except (ValueError, OSError):
            continue
    return False


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _eviction_candidates(root: Path, protected: set[Path]):
    """Return disposable units ordered by cost class and last access time."""
    candidates = []
    access = _read_access_times(root)

    def accessed(path: Path, fallback: Path | None = None):
        resolved = path.resolve()
        direct = access.get(str(resolved))
        if direct is not None:
            return direct
        if path.is_dir():
            nested = [value for key, value in access.items() if _inside(Path(key), resolved)]
            if nested:
                return max(nested)
        return _mtime(fallback or path)
    packed = root / "instinct-packed"
    if packed.exists() and not packed.is_symlink():
        for item in packed.iterdir():
            if item.is_symlink() or item.name.endswith(".lock") or _is_protected(item, protected):
                continue
            if item.is_dir():
                # Incomplete build directories are cheapest to discard. Final
                # packed caches are retained longer because they cost most to rebuild.
                priority = 0 if item.name.startswith("build-") else 3
                marker = item / "manifest.json"
                candidates.append((priority, accessed(item, marker if marker.exists() else item), item))

    downloads = root / "downloads"
    if downloads.exists() and not downloads.is_symlink():
        for item in downloads.iterdir():
            if item.is_symlink() or item.name.endswith(".lock") or _is_protected(item, protected):
                continue
            candidates.append((1, accessed(item), item))

    for arrow in root.rglob("*.arrow"):
        if arrow.is_symlink() or _is_protected(arrow, protected):
            continue
        if packed.exists() and _inside(arrow, packed):
            continue
        candidates.append((2, accessed(arrow), arrow))
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return candidates


def _remove_candidate(path: Path, root: Path) -> bool:
    if path.is_symlink() or not _inside(path, root):
        return False
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        return True
    except (FileNotFoundError, PermissionError, OSError):
        # Windows refuses to delete active mmap files. A live lease should have
        # excluded them already; this remains a final race-safe fallback.
        return False


def _prune_empty_dirs(root: Path) -> None:
    excluded = {root / _LEASE_DIR}
    directories = [path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        if path in excluded or path.name == "instinct-packed":
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def enforce_cache_budget(
    root=None,
    *,
    max_gb=None,
    max_bytes=None,
    protected=(),
    reserve_bytes=0,
    acquire_lock=True,
    dry_run=False,
):
    """Evict rebuildable LRU entries until the dataset cache fits its budget."""
    root = dataset_cache_budget_root(root)
    budget = configured_budget_bytes(max_gb) if max_bytes is None else int(max_bytes)
    if budget is None:
        return {"root": str(root), "budget_bytes": None, "before_bytes": 0,
                "after_bytes": 0, "removed_bytes": 0, "removed": []}
    if budget < 0:
        raise ValueError("dataset cache budget must be >= 0 bytes")
    reserve_bytes = max(0, int(reserve_bytes))
    effective_limit = max(0, budget - reserve_bytes)
    root.mkdir(parents=True, exist_ok=True)

    def enforce_locked():
        protected_paths = _active_lease_paths(root)
        for value in protected:
            path = Path(value).resolve()
            if not _inside(path, root):
                raise ValueError(f"Protected cache path is outside {root}: {path}")
            protected_paths.add(path)
        before = _tree_size(root)
        current = before
        removed = []
        for _, _, candidate in _eviction_candidates(root, protected_paths):
            if current <= effective_limit:
                break
            size = _tree_size(candidate)
            if dry_run or _remove_candidate(candidate, root):
                removed.append(str(candidate))
                current = max(0, current - size)
        if not dry_run:
            _prune_empty_dirs(root)
            _write_access_times(root, _read_access_times(root))
            current = _tree_size(root)
        report = {
            "root": str(root),
            "budget_bytes": budget,
            "reserve_bytes": reserve_bytes,
            "before_bytes": before,
            "after_bytes": current,
            "removed_bytes": max(0, before - current),
            "removed": removed,
        }
        if current > effective_limit and not dry_run:
            raise CacheBudgetExceeded(
                f"Dataset cache needs {current / GIB:.2f} GiB plus "
                f"{reserve_bytes / GIB:.2f} GiB reserved output, but the limit is "
                f"{budget / GIB:.2f} GiB. Active cache files are protected; reduce "
                "the dataset/packing size or raise --data_cache_max_gb."
            )
        return report

    if acquire_lock:
        with FileLock(str(root / _BUDGET_LOCK)):
            return enforce_locked()
    return enforce_locked()


@contextmanager
def cache_budget_guard(root=None):
    """Serialize cache construction/eviction in one cache root."""
    root = dataset_cache_budget_root(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / _BUDGET_LOCK)):
        yield root


def cache_files(dataset) -> list[str]:
    return [item["filename"] for item in getattr(dataset, "cache_files", []) if item.get("filename")]


def cache_paths_size(paths) -> int:
    return sum(_tree_size(Path(value).resolve()) for value in paths)


def reserve_cache_space(additional_bytes, protected=(), root=None):
    """Evict before a rebuild so expected output does not cause a transient spike."""
    return enforce_cache_budget(
        root,
        protected=protected,
        reserve_bytes=max(0, int(additional_bytes)),
    )


def _source_files_size(data_files) -> int:
    """Expected Arrow footprint of the sources a loader is about to convert.

    JSONL expands to roughly its own size plus parsing overhead; a compiled
    parquet file is compressed, so its footer holds the uncompressed totals
    that the Arrow cache will actually occupy.
    """
    if isinstance(data_files, dict):
        return sum(_source_files_size(value) for value in data_files.values())
    if isinstance(data_files, (list, tuple)):
        return sum(_source_files_size(value) for value in data_files)
    if isinstance(data_files, (str, os.PathLike)):
        path = Path(data_files)
        if not path.exists():
            return 0
        try:
            from scripts.data_loader.source_format import estimated_arrow_bytes
            return estimated_arrow_bytes(path)
        except (OSError, ValueError):
            return _tree_size(path.resolve())
    return 0


def load_dataset_with_budget(*args, **kwargs):
    """Budget-aware wrapper around datasets.load_dataset."""
    import datasets

    cache_root = dataset_cache_root()
    root = dataset_cache_budget_root()
    # ``datasets`` snapshots its cache path at import time. Keep it aligned
    # when a stage/test changes HF_DATASETS_CACHE afterwards.
    datasets.config.HF_DATASETS_CACHE = cache_root
    with cache_budget_guard(root):
        # The estimate is data-dependent and reconciled against the real size
        # after the cache exists.
        source_bytes = _source_files_size(kwargs.get("data_files"))
        enforce_cache_budget(
            root, reserve_bytes=source_bytes, acquire_lock=False
        )
        dataset = datasets.load_dataset(*args, **kwargs)
        files = cache_files(dataset)
        if files:
            register_cache_use(files, root, acquire_lock=False)
        report = enforce_cache_budget(root, protected=files, acquire_lock=False)
    if report["removed"] and os.environ.get("RANK", "0") in ("0", "-1"):
        print(
            f"[Data cache] evicted {report['removed_bytes'] / GIB:.2f} GiB; "
            f"usage={report['after_bytes'] / GIB:.2f}/"
            f"{report['budget_bytes'] / GIB:.2f} GiB",
            flush=True,
        )
    return dataset
