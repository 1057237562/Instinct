"""Build a shuffled Instinct SFT mix from three local instruction datasets.

The output contains all valid, unique rows from:

* Magicoder 110K (JSONL instruction/response pairs)
* MathInstruct (a JSON array of instruction/output pairs)
* UltraChat 200K (only the ``train_sft`` parquet shards)

Run this script from the repository root.  Processing and shuffling are
bounded-memory; the final file is installed atomically.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as parquet

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.mix_sft_datasets import (
    ExternalRandomShuffler,
    iter_json_array,
    iter_jsonl,
    normalize_pair,
)


DEFAULT_MAGICODER = (
    REPO_ROOT / "dataset" / "magicoder-110k"
    / "data-evol_instruct-decontaminated.jsonl"
)
DEFAULT_MATH = REPO_ROOT / "dataset" / "math-instruct" / "MathInstruct.json"
DEFAULT_ULTRACHAT = REPO_ROOT / "dataset" / "ultrachat-200k" / "data"
DEFAULT_OUTPUT = (
    REPO_ROOT / "dataset" / "sft_magicoder110k_mathinstruct_ultrachat200k.jsonl"
)


def normalize_messages(messages):
    """Normalize an UltraChat conversation to Instinct's chat schema."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("Conversation has no messages")

    conversations = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Conversation message is not an object")
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported role: {role!r}")
        content = "" if content is None else str(content).strip()
        if not content:
            raise ValueError("Conversation contains an empty message")
        conversations.append({"role": role, "content": content})

    if not any(message["role"] == "assistant" for message in conversations):
        raise ValueError("Conversation has no assistant response")
    return {"conversations": conversations}


def iter_ultrachat_rows(directory):
    """Stream records from every UltraChat train_sft parquet shard."""
    shards = sorted(Path(directory).glob("train_sft-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No train_sft-*.parquet shards found under {directory}"
        )
    for shard in shards:
        parquet_file = parquet.ParquetFile(shard)
        for batch in parquet_file.iter_batches(columns=["messages"], batch_size=1024):
            for row in batch.to_pylist():
                yield row


def ultrachat_row_count(directory):
    """Return the number of records across all train_sft shards."""
    shards = sorted(Path(directory).glob("train_sft-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No train_sft-*.parquet shards found under {directory}"
        )
    return sum(parquet.ParquetFile(shard).metadata.num_rows for shard in shards)


def sample_rows(rows, total_rows, fraction, seed):
    """Select an exact random fraction by global row index, without replacement."""
    if not 0 < fraction <= 1:
        raise ValueError("UltraChat fraction must be in (0, 1]")
    if fraction == 1:
        yield from rows
        return
    sample_size = round(total_rows * fraction)
    selected = set(random.Random(seed).sample(range(total_rows), sample_size))
    for index, row in enumerate(rows):
        if index in selected:
            yield row


def _add_source_rows(shuffler, rows, normalize, source_name):
    accepted = invalid = duplicates = 0
    for row in rows:
        try:
            normalized = normalize(row)
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        if shuffler.add(normalized):
            accepted += 1
        else:
            duplicates += 1
        processed = accepted + invalid + duplicates
        if processed % 100000 == 0:
            print(f"  {source_name}: processed {processed:,} rows", flush=True)
    print(
        f"  {source_name}: kept {accepted:,}, skipped {invalid:,} invalid, "
        f"{duplicates:,} duplicates",
        flush=True,
    )
    return {
        "kept": accepted,
        "invalid": invalid,
        "duplicates": duplicates,
    }


def mix_datasets(magicoder_path, math_path, ultrachat_dir, output_path, *,
                 ultrachat_fraction=1.0, seed=42, chunk_rows=25000,
                 overwrite=False):
    output_path = Path(output_path)
    if not output_path.name.lower().startswith("sft") or output_path.suffix != ".jsonl":
        raise ValueError("Output must be named sft*.jsonl")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --overwrite to replace it"
        )
    for source in (magicoder_path, math_path):
        if not Path(source).is_file():
            raise FileNotFoundError(source)
    if not Path(ultrachat_dir).is_dir():
        raise FileNotFoundError(ultrachat_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shuffler = ExternalRandomShuffler(output_path, seed, chunk_rows)
    try:
        print(f"Reading Magicoder: {magicoder_path}", flush=True)
        magicoder = _add_source_rows(
            shuffler,
            iter_jsonl(magicoder_path),
            lambda row: normalize_pair(row.get("instruction"), row.get("response")),
            "Magicoder",
        )
        print(f"Reading MathInstruct: {math_path}", flush=True)
        math = _add_source_rows(
            shuffler,
            iter_json_array(math_path),
            lambda row: normalize_pair(row.get("instruction"), row.get("output")),
            "MathInstruct",
        )
        ultrachat_total = ultrachat_row_count(ultrachat_dir)
        ultrachat_target = round(ultrachat_total * ultrachat_fraction)
        print(
            f"Reading UltraChat train_sft shards: {ultrachat_dir} "
            f"(random sample {ultrachat_target:,}/{ultrachat_total:,}, "
            f"{ultrachat_fraction:.1%})",
            flush=True,
        )
        ultrachat = _add_source_rows(
            shuffler,
            sample_rows(
                iter_ultrachat_rows(ultrachat_dir), ultrachat_total,
                ultrachat_fraction, seed + 1,
            ),
            lambda row: normalize_messages(row.get("messages")),
            "UltraChat",
        )
        ultrachat["source_rows"] = ultrachat_total
        ultrachat["sampled_rows"] = ultrachat_target
        ultrachat["sample_fraction"] = ultrachat_fraction
        print("Writing externally shuffled output", flush=True)
        shuffler.finish(overwrite=overwrite)
    except Exception:
        shuffler.temporary_dir.cleanup()
        raise

    stats = {
        "magicoder": magicoder,
        "math_instruct": math,
        "ultrachat": ultrachat,
    }
    stats["total_rows"] = sum(source["kept"] for source in stats.values())
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magicoder", type=Path, default=DEFAULT_MAGICODER)
    parser.add_argument("--math", type=Path, default=DEFAULT_MATH)
    parser.add_argument("--ultrachat-dir", type=Path, default=DEFAULT_ULTRACHAT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ultrachat-fraction", type=float, default=1.0,
        help="Random fraction of UltraChat train_sft rows to include (default: 1.0)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-rows", type=int, default=25000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stats = mix_datasets(
        args.magicoder,
        args.math,
        args.ultrachat_dir,
        args.output,
        ultrachat_fraction=args.ultrachat_fraction,
        seed=args.seed,
        chunk_rows=args.chunk_rows,
        overwrite=args.overwrite,
    )
    print(f"Wrote {stats['total_rows']:,} rows to {args.output}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
