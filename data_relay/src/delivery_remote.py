"""Remote helper for data_relay deliver; invoked over authenticated SSH."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], stderr=subprocess.STDOUT, text=True
    ).strip()


def inside(dataset, relative):
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(p in (".", "..") for p in rel.parts):
        raise ValueError("Invalid dataset-relative path: " + relative)
    path = dataset / rel
    for part in [path, *path.parents]:
        if part == dataset:
            break
        if part.is_symlink():
            raise ValueError("Symlinks are not allowed: " + str(part))
    if not path.resolve().is_relative_to(dataset):
        raise ValueError("Path escapes dataset: " + relative)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(opts):
    given_repo = Path(opts["repo"])
    if not given_repo.is_absolute():
        raise ValueError("Repository path must be absolute")
    repo = given_repo.resolve(strict=True)
    bootstrap = not (repo / ".git").exists()
    if bootstrap and opts.get("skip_git_update"):
        raise ValueError("Cannot resume without an existing Git checkout")
    if bootstrap:
        if not opts.get("init_repo") or not (repo / "model" / "model_instinct.py").is_file():
            raise ValueError("No Git metadata; use --init-repo only for an existing Instinct code copy")
        branch = opts.get("branch") or "master"
        if branch.startswith("-") or not branch.strip():
            raise ValueError("Invalid bootstrap branch")
        upstream = "origin/" + branch
        tracked = set()
    else:
        if Path(git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
            raise ValueError("--repo must be the repository root")
        origin = git(repo, "remote", "get-url", "origin").removesuffix(".git").rstrip("/")
        if origin not in ("https://github.com/1057237562/Instinct", "git@github.com:1057237562/Instinct", "ssh://git@github.com/1057237562/Instinct"):
            raise ValueError("Unexpected origin; refusing to modify this repository")
        dirty = git(repo, "status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise ValueError("Remote tracked files have local changes; preserve them before delivery")
        branch = git(repo, "symbolic-ref", "--short", "HEAD")
        if opts.get("branch") and opts["branch"] != branch:
            raise ValueError("Remote branch differs from --branch; this command does not switch branches")
        upstream = git(repo, "rev-parse", "--abbrev-ref", "@{upstream}")
        tracked = set(git(repo, "ls-files", "-z").split("\0"))
    dataset = repo / "dataset"
    if dataset.is_symlink() or (dataset.exists() and not dataset.is_dir()):
        raise ValueError("dataset must be a real directory inside the repository")
    dataset = dataset.resolve()
    bundle = None
    if opts.get("git_bundle"):
        bundle = inside(repo, ".instinct-delivery.bundle")
        inside(repo, ".instinct-delivery.bundle.rpart")
        if bundle.exists() and not bundle.is_file():
            raise ValueError("Git bundle target is not a regular file")
    targets = [inside(dataset, f["name"]) for f in opts["files"]]
    protected = set()
    for target in targets:
        for suffix in ("", ".incoming", ".incoming.rpart"):
            path = inside(dataset, target.name + suffix)
            if path.exists() and not path.is_file():
                raise ValueError("Upload target is not a regular file: " + str(path))
            if path.relative_to(repo).as_posix() in tracked:
                raise ValueError("Upload would overwrite a tracked file: " + str(path))
            protected.add(path)
    audit_path = inside(dataset, opts["files"][0]["name"] + ".delivery.report.json")
    if audit_path.relative_to(repo).as_posix() in tracked:
        raise ValueError("Delivery audit would overwrite a tracked file")
    removals = set()
    for relative in opts["remove_old"]:
        root = inside(dataset, relative)
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else list(root.rglob("*"))
        for path in candidates:
            path = inside(dataset, str(path.relative_to(dataset)))
            if not path.is_file():
                continue
            if not path.name.endswith((".parquet", ".jsonl", ".jsonl.gz", ".arrow")):
                continue  # README, reports, provenance, scripts and other files survive.
            if path in protected or path.relative_to(repo).as_posix() in tracked:
                raise ValueError("Cleanup overlaps delivery or tracked content: " + str(path))
            removals.add(path)
    parent = dataset if dataset.exists() else repo
    caches = []
    if opts.get("clean_cache"):
        for relative in (".cache/torch_compile", ".cache/huggingface/datasets", ".cache/huggingface/hub", ".Trash-0/files"):
            cache = inside(repo, relative)
            if cache.is_dir():
                for entry in cache.rglob("*"):
                    # HF hubs normally contain internal links; only unlink them,
                    # never follow them or delete their external targets.
                    if not entry.is_symlink() and entry.is_file() and entry.relative_to(repo).as_posix() in tracked:
                        raise ValueError("Cache contains tracked files")
                caches.append(cache)
    free = shutil.disk_usage(parent).free
    reclaimed = sum(p.stat().st_size for p in removals)
    reclaimed += sum(p.stat().st_size for cache in caches for p in cache.rglob("*") if p.is_file() and not p.is_symlink())
    ready = []
    invalid_incoming = []
    additional = 0
    for item in opts["files"]:
        final = inside(dataset, item["name"])
        incoming = inside(dataset, item["name"] + ".incoming")
        part = inside(dataset, item["name"] + ".incoming.rpart")
        if incoming.is_file():
            if incoming.stat().st_size == item["size"] and sha256(incoming) == item["sha256"]:
                ready.append(item["name"])
                continue
            invalid_incoming.append(incoming)
            reclaimed += incoming.stat().st_size
        if final.is_file() and final.stat().st_size == item["size"] and sha256(final) == item["sha256"]:
            ready.append(item["name"])
            continue
        # Sequential .rpart files already consume disk space and resume in place.
        additional += max(0, item["size"] - (part.stat().st_size if part.is_file() else 0))
    required = additional + 1024**3
    plan = {"repo": str(repo), "dataset": str(dataset), "branch": branch,
            "upstream": upstream, "remove": sorted(map(str, removals)),
            "cleanup_bytes": reclaimed, "free_bytes": free, "required_bytes": required,
            "files": opts["files"], "bootstrap_git": bootstrap, "cache_cleanup": list(map(str, caches)), "ready_files": ready}
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if opts["action"] == "inspect":
        return
    if opts["action"] == "prepare":
        if bundle is not None:
            expected = opts["git_bundle"]
            if bundle.stat().st_size != expected["size"] or sha256(bundle) != expected["sha256"]:
                raise ValueError("Git bundle verification failed")
        if free + reclaimed < required:
            raise ValueError("Not enough disk space after the selected cleanup")
        for path in sorted(removals):
            path.unlink()
            print("Removed " + str(path), flush=True)
        for cache in caches:
            shutil.rmtree(cache)
            print("Removed disposable cache " + str(cache), flush=True)
        for path in invalid_incoming:
            path.unlink()
            print("Discarded invalid completed temporary upload " + str(path), flush=True)
        if shutil.disk_usage(parent).free < required:
            raise ValueError("Cleanup did not free enough physical disk space")
        if opts.get("skip_git_update"):
            print("Preserving current Git revision for resumed delivery", flush=True)
        elif bootstrap:
            with tempfile.TemporaryDirectory(prefix="instinct-code-", dir=str(repo.parent)) as stage:
                if bundle is not None:
                    subprocess.run(["git", "init", stage], check=True)
                    subprocess.run(["git", "-C", stage, "fetch", str(bundle),
                                    "refs/remotes/origin/" + branch + ":refs/remotes/origin/" + branch], check=True)
                    subprocess.run(["git", "--git-dir=" + stage + "/.git", "--work-tree=" + str(repo),
                                    "checkout", "-f", "-b", branch, "origin/" + branch], check=True)
                    subprocess.run(["git", "-C", stage, "remote", "add", "origin",
                                    "https://github.com/1057237562/Instinct.git"], check=True)
                    subprocess.run(["git", "-C", stage, "config", "branch." + branch + ".remote", "origin"], check=True)
                    subprocess.run(["git", "-C", stage, "config", "branch." + branch + ".merge", "refs/heads/" + branch], check=True)
                else:
                    subprocess.run(["git", "clone", "--no-checkout", "--depth", "1", "--branch", branch,
                                    "https://github.com/1057237562/Instinct.git", stage], check=True, timeout=180)
                    subprocess.run(["git", "--git-dir=" + stage + "/.git", "--work-tree=" + str(repo),
                                    "checkout", "-f", branch], check=True)
                shutil.move(str(Path(stage) / ".git"), str(repo / ".git"))
        elif bundle is not None:
            subprocess.run(["git", "-C", str(repo), "fetch", str(bundle),
                            "refs/remotes/origin/" + branch + ":refs/remotes/origin/" + branch], check=True)
            subprocess.run(["git", "-C", str(repo), "merge", "--ff-only", "origin/" + branch], check=True)
        else:
            subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True, timeout=180)
        if bundle is not None:
            bundle.unlink()
        dataset.mkdir(exist_ok=True)
        print("Updated revision: " + git(repo, "rev-parse", "HEAD"), flush=True)
        return
    if opts["action"] == "verify":
        for item in opts["files"]:
            final = inside(dataset, item["name"])
            incoming = inside(dataset, item["name"] + ".incoming")
            candidate = incoming if incoming.exists() else final
            if candidate.stat().st_size != item["size"] or sha256(candidate) != item["sha256"]:
                raise ValueError("SHA-256 or size mismatch: " + str(candidate))
        # Check the entire delivery before replacing any final file.
        for item in opts["files"]:
            final = inside(dataset, item["name"])
            incoming = inside(dataset, item["name"] + ".incoming")
            if incoming.exists():
                os.replace(incoming, final)
            print("Verified and installed " + str(final), flush=True)
        audit = {"revision": git(repo, "rev-parse", "HEAD"), "files": opts["files"],
                 "removed": opts["remove_old"]}
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    raise ValueError("Unknown action")


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
