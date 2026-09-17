"""Launch continued SFT on the verified instruction-understanding curriculum."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the full 10.6M-token train set; default is the 1M-token pilot")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    args = parser.parse_args()
    config = ROOT / "checkpoints" / "full_sft_20260914_200306_768.json"
    weight = ROOT / "out" / "instinct-v1-0914.pth"
    data = ROOT / "dataset" / "instruction_understanding_sft" / ("train.jsonl" if args.full else "pilot_train.jsonl")
    audit = ROOT / "dataset" / "instruction_understanding_sft" / "full_audit.json"
    report_path = ROOT / "dataset" / "instruction_understanding_sft" / "report.json"
    overlap_path = ROOT / "dataset" / "instruction_understanding_sft" / "overlap_audit.json"
    for path in (config, weight, data, audit, report_path, overlap_path):
        if not path.is_file(): raise FileNotFoundError(path)
    audit_result = json.loads(audit.read_text(encoding="utf-8"))
    if audit_result["answer_or_execution_errors"] or audit_result["incomplete_contrast_groups"]:
        raise RuntimeError("Instruction dataset has not passed its full audit")
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    if overlap["normalized_exact_prompt_matches_with_prior_sft"] or overlap["new_prompts_with_any_humaneval_13_token_ngram"]:
        raise RuntimeError("Instruction dataset overlap audit has unresolved hits")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_hash = report["files"]["train" if args.full else "pilot_train"]["sha256"]
    actual_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError("Dataset file changed after its manifest was generated")
    cfg = json.loads(config.read_text(encoding="utf-8"))
    stage = "instruction_sft_" + ("full_" if args.full else "pilot_") + datetime.now().strftime("%Y%m%d_%H%M%S")
    flags = {
        "from_weight": str(weight), "from_resume": 0, "config_path": str(config),
        "hidden_size": cfg["hidden_size"], "num_hidden_layers": cfg["num_hidden_layers"],
        "use_moe": int(cfg.get("use_moe", False)), "data_path": str(data),
        "save_dir": "out", "save_weight": stage, "epochs": 1, "optimizer": "adamw",
        "learning_rate": args.learning_rate, "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps, "grad_clip": 1.0,
        "dtype": "bfloat16", "param_dtype": "fp32", "kv_cache_dtype": "fp32",
        "fp8_training": "off", "max_seq_len": 768, "sequence_packing": 1,
        "sequence_packing_mode": "fixed", "packing_num_proc": 1, "num_workers": 0,
        "use_grad_checkpoint": 1, "use_compile": 0, "save_interval": 100, "log_interval": 10,
    }
    command = [sys.executable, "-u", str(ROOT / "trainer" / "train_full_sft.py")]
    for key, value in flags.items(): command.extend(["--" + key, str(value)])
    sys.path.append(str(ROOT))
    import datasets  # noqa: F401  # before torch on Windows
    from trainer.trainer_cli import build_trainer_parser
    checker = build_trainer_parser("Validate instruction SFT")
    checker.add_argument("--data_path")
    parsed = checker.parse_args(command[3:])
    from trainer.trainer_utils import config_from_args
    effective = config_from_args(parsed)
    assert effective.num_hidden_layers == cfg["num_hidden_layers"] == 20
    assert effective.hidden_size == cfg["hidden_size"] == 768
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        print("Dataset audit and trainer configuration validated. Training NOT started.")
        return
    launch = ROOT / "checkpoints" / f"{stage}_launch.json"
    launch.write_text(json.dumps({"command": command, "settings": flags}, indent=2), encoding="utf-8")
    subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"}, check=True)


if __name__ == "__main__": main()
