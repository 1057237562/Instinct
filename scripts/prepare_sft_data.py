"""Normalize selected instruction datasets to Instinct's SFT JSONL format.

Examples (run from the repository root):

    python scripts/prepare_sft_data.py codealpaca-local
    python scripts/prepare_sft_data.py smol-smoltalk --max-samples 100000
    python scripts/prepare_sft_data.py bigcode-exec-50k

Every output filename starts with ``sft_`` so Config WebUI discovers it as an
SFT dataset. Downloads use the Hugging Face cache configured by ``datasets``.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]

PRESETS = {
    "codealpaca-local": {
        "repo": "parquet",
        "data_files": str(REPO_ROOT / "dataset" / "codealpaca" / "data" / "train-*.parquet"),
        "split": "train",
        "format": "prompt_completion",
        "output": "sft_codealpaca_20k.jsonl",
    },
    "smol-smoltalk": {
        "repo": "HuggingFaceTB/smol-smoltalk",
        "split": "train",
        "format": "messages",
        "output": "sft_smol_smoltalk.jsonl",
    },
    "bigcode-exec-50k": {
        "repo": "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "split": "train",
        "format": "instruction_response",
        "output": "sft_bigcode_exec_50k.jsonl",
    },
    "magicoder-75k": {
        "repo": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "split": "train",
        "format": "problem_solution",
        "output": "sft_magicoder_oss_75k.jsonl",
    },
    "no-robots": {
        "repo": "HuggingFaceH4/no_robots",
        "split": "train_sft",
        "format": "messages",
        "output": "sft_no_robots.jsonl",
    },
}


def normalize_row(row, row_format):
    if row_format == "messages":
        messages = row["messages"]
    elif row_format == "prompt_completion":
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
    elif row_format == "instruction_response":
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]
    elif row_format == "problem_solution":
        messages = [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["solution"]},
        ]
    else:
        raise ValueError(f"Unsupported row format: {row_format}")

    conversations = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported conversation role: {role!r}")
        if content is None:
            content = ""
        conversations.append({"role": role, "content": str(content).strip()})
    if not conversations or not any(item["role"] == "assistant" for item in conversations):
        raise ValueError("SFT row has no assistant response")
    return {"conversations": conversations}


def load_preset(name):
    preset = PRESETS[name]
    load_kwargs = {"split": preset["split"]}
    if "data_files" in preset:
        load_kwargs["data_files"] = preset["data_files"]
    return load_dataset(preset["repo"], **load_kwargs), preset


def write_jsonl(dataset, row_format, output_path, max_samples=None):
    output_path = Path(output_path)
    if not output_path.name.lower().startswith("sft") or output_path.suffix.lower() != ".jsonl":
        raise ValueError("Output must be an sft*.jsonl filename so WebUI classifies it as SFT")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    written = 0
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for index in range(limit):
            try:
                normalized = normalize_row(dataset[index], row_format)
            except (KeyError, TypeError, ValueError) as error:
                print(f"[skip] row={index}: {error}")
                continue
            output_file.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            written += 1
    os.replace(temporary_path, output_path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=sorted(PRESETS))
    parser.add_argument("--output", help="Defaults to dataset/sft_<preset>.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")

    dataset, preset = load_preset(args.preset)
    output_path = Path(args.output) if args.output else REPO_ROOT / "dataset" / preset["output"]
    written = write_jsonl(dataset, preset["format"], output_path, args.max_samples)
    print(f"Wrote {written:,} rows to {output_path}")


if __name__ == "__main__":
    main()

