import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "delivery_remote", Path(__file__).parents[1] / "src" / "delivery_remote.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name).resolve()
        self.dataset = self.repo / "dataset"
        self.dataset.mkdir()
        (self.repo / ".git").mkdir()
        self.tracked = []
        self.opts = {"repo": str(self.repo), "remove_old": [], "action": "inspect",
                     "files": [{"name": "new.parquet", "size": 3,
                                "sha256": hashlib.sha256(b"new").hexdigest()}]}
        self.patch = patch.object(helper, "git", side_effect=self.git)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def git(self, repo, *args):
        return {("rev-parse", "--show-toplevel"): str(self.repo),
                ("remote", "get-url", "origin"): "https://github.com/1057237562/Instinct.git",
                ("status", "--porcelain", "--untracked-files=no"): "",
                ("symbolic-ref", "--short", "HEAD"): "main",
                ("rev-parse", "--abbrev-ref", "@{upstream}"): "origin/main",
                ("rev-parse", "HEAD"): "abc123",
                ("ls-files", "-z"): "\0".join(self.tracked)}[args]

    def run_helper(self):
        with contextlib.redirect_stdout(io.StringIO()):
            helper.main(self.opts)

    def test_prepare_cleans_only_selected_corpus_and_preserves_reports(self):
        old = self.dataset / "old"
        old.mkdir()
        for name in ["data.parquet", "data.jsonl.gz", "data.report.json", "README.md"]:
            (old / name).write_text("keep")
        (self.dataset / "other.parquet").write_text("keep")
        self.opts.update(action="prepare", remove_old=["old"])
        with patch.object(helper.subprocess, "run") as run:
            self.run_helper()
            run.assert_called_once_with(["git", "-C", str(self.repo), "pull", "--ff-only"], check=True, timeout=180)
        self.assertFalse((old / "data.parquet").exists())
        self.assertFalse((old / "data.jsonl.gz").exists())
        self.assertTrue((old / "data.report.json").exists())
        self.assertTrue((old / "README.md").exists())
        self.assertTrue((self.dataset / "other.parquet").exists())

    def test_inspection_does_not_delete(self):
        old = self.dataset / "old.parquet"
        old.write_bytes(b"old")
        self.opts["remove_old"] = ["old.parquet"]
        self.run_helper()
        self.assertEqual(old.read_bytes(), b"old")

    def test_traversal_and_target_overlap_are_rejected(self):
        for name in ["../outside.parquet", "/etc/passwd", "new.parquet"]:
            (self.dataset / "new.parquet").write_bytes(b"new")
            self.opts["remove_old"] = [name]
            with self.assertRaises(ValueError):
                self.run_helper()

    def test_tracked_dataset_is_preserved(self):
        old = self.dataset / "old.parquet"
        old.write_bytes(b"old")
        self.tracked = ["dataset/old.parquet"]
        self.opts.update(action="prepare", remove_old=["old.parquet"])
        with self.assertRaises(ValueError):
            self.run_helper()
        self.assertTrue(old.exists())

    def test_hash_mismatch_preserves_final_and_verified_delivery_replaces_it(self):
        final = self.dataset / "new.parquet"
        final.write_bytes(b"old")
        incoming = self.dataset / "new.parquet.incoming"
        incoming.write_bytes(b"bad")
        self.opts["action"] = "verify"
        with self.assertRaises(ValueError):
            self.run_helper()
        self.assertEqual(final.read_bytes(), b"old")
        incoming.write_bytes(b"new")
        self.run_helper()
        self.assertEqual(final.read_bytes(), b"new")
        self.assertFalse(incoming.exists())

    def test_cache_cleanup_preserves_hf_token_and_weights(self):
        cache = self.repo / ".cache" / "huggingface"
        (cache / "hub").mkdir(parents=True)
        (cache / "hub" / "cached.bin").write_bytes(b"cache")
        (cache / "token").write_text("keep")
        (self.repo / "out").mkdir()
        weight = self.repo / "out" / "model.pth"
        weight.write_bytes(b"weights")
        self.opts.update(action="prepare", clean_cache=True)
        with patch.object(helper.subprocess, "run"):
            self.run_helper()
        self.assertFalse((cache / "hub").exists())
        self.assertTrue((cache / "token").exists())
        self.assertEqual(weight.read_bytes(), b"weights")

    def test_bootstrap_updates_copied_code_and_preserves_dataset_and_weights(self):
        (self.repo / ".git").rmdir()
        (self.repo / "model").mkdir()
        (self.repo / "model" / "model_instinct.py").write_text("old")
        (self.repo / "out").mkdir()
        weight = self.repo / "out" / "model.pth"
        weight.write_bytes(b"weights")
        existing = self.dataset / "keep.parquet"
        existing.write_bytes(b"keep")
        origin = self.repo / "_source"
        origin.mkdir()
        real_run = subprocess.run
        def run(cmd):
            return real_run(cmd, check=True, capture_output=True)
        run(["git", "init", "--initial-branch=master", str(origin)])
        (origin / "model").mkdir()
        (origin / "model" / "model_instinct.py").write_text("updated")
        run(["git", "-C", str(origin), "add", "."])
        run(["git", "-C", str(origin), "-c", "user.email=test@example.invalid",
             "-c", "user.name=Test", "commit", "-m", "test"])
        def local_clone(cmd, **kwargs):
            cmd = [str(origin) if arg == "https://github.com/1057237562/Instinct.git" else arg for arg in cmd]
            return real_run(cmd, **kwargs, capture_output=True)
        self.opts.update(action="prepare", init_repo=True, branch="master")
        with patch.object(helper.subprocess, "run", side_effect=local_clone):
            self.run_helper()
        self.assertTrue((self.repo / ".git").is_dir())
        self.assertEqual((self.repo / "model" / "model_instinct.py").read_text(), "updated")
        self.assertEqual(weight.read_bytes(), b"weights")
        self.assertEqual(existing.read_bytes(), b"keep")

    def test_git_bundle_bootstrap_keeps_original_commit_and_upstream(self):
        (self.repo / ".git").rmdir()
        (self.repo / "model").mkdir()
        (self.repo / "model" / "model_instinct.py").write_text("old")
        origin = self.repo / "_source"
        origin.mkdir()
        def run(*args):
            return subprocess.check_output(["git", *args], stderr=subprocess.STDOUT, text=True).strip()
        run("init", "--initial-branch=master", str(origin))
        (origin / "model").mkdir()
        (origin / "model" / "model_instinct.py").write_text("updated")
        run("-C", str(origin), "add", ".")
        run("-C", str(origin), "-c", "user.email=test@example.invalid", "-c", "user.name=Test", "commit", "-m", "test")
        commit = run("-C", str(origin), "rev-parse", "HEAD")
        run("-C", str(origin), "update-ref", "refs/remotes/origin/master", commit)
        bundle = self.repo / ".instinct-delivery.bundle"
        run("-C", str(origin), "bundle", "create", str(bundle), "refs/remotes/origin/master")
        self.opts.update(action="prepare", init_repo=True, branch="master", git_bundle={
            "size": bundle.stat().st_size, "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()})
        self.run_helper()
        self.assertEqual(run("-C", str(self.repo), "rev-parse", "HEAD"), commit)
        self.assertEqual(run("-C", str(self.repo), "rev-parse", "--abbrev-ref", "@{upstream}"), "origin/master")
        self.assertFalse(bundle.exists())

    def test_resume_space_counts_only_remaining_bytes(self):
        (self.dataset / "new.parquet.incoming.rpart").write_bytes(b"ne")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            helper.main(self.opts)
        import json
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["required_bytes"], 1024**3 + 1)

    def test_complete_temporary_upload_is_verified_and_skipped_on_retry(self):
        (self.dataset / "new.parquet.incoming").write_bytes(b"new")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            helper.main(self.opts)
        import json
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["required_bytes"], 1024**3)
        self.assertEqual(plan["ready_files"], ["new.parquet"])

    def test_repeated_delivery_reuses_verified_final_file(self):
        final = self.dataset / "new.parquet"
        final.write_bytes(b"new")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            helper.main(self.opts)
        import json
        self.assertEqual(json.loads(output.getvalue())["ready_files"], ["new.parquet"])
        self.opts["action"] = "verify"
        self.run_helper()
        self.assertEqual(final.read_bytes(), b"new")

    def test_resume_does_not_mutate_git(self):
        self.opts.update(action="prepare", skip_git_update=True)
        with patch.object(helper.subprocess, "run") as run:
            self.run_helper()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
