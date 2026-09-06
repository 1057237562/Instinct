"""Mix Magicoder 110K, MathInstruct, and a replay sample of SFT T2T.

The default replay fraction is 20% of the final mixture.  The 14 GB T2T source
is sampled in one streaming pass with reservoir sampling.  Final shuffling is
an external merge sort over random keys, so memory use is bounded by
``--chunk-rows`` rather than total output size.

Run from the repository root:

    python scripts/mix_sft_datasets.py
"""

import argparse
import hashlib
import heapq
import json
import os
import random
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAGICODER = REPO_ROOT / "dataset" / "magicoder-110k" / "data-evol_instruct-decontaminated.jsonl"
DEFAULT_MATH = REPO_ROOT / "dataset" / "math-instruct" / "MathInstruct.json"
DEFAULT_ORIGINAL = REPO_ROOT / "dataset" / "sft_t2t.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "dataset" / "sft_magicoder110k_mathinstruct_t2t_replay20.jsonl"


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error


def iter_json_array(path, read_chars=1024 * 1024):
    """Stream a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    finished = False
    with Path(path).open("r", encoding="utf-8") as source:
        while not finished:
            chunk = source.read(read_chars)
            eof = chunk == ""
            buffer += chunk
            position = 0

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"Expected a top-level JSON array in {path}")
                    started = True
                    position += 1
                    continue

                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    finished = True
                    position += 1
                    break
                try:
                    item, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                yield item
                position = end

            buffer = buffer[position:]
            if eof:
                if not finished:
                    raise ValueError(f"Truncated JSON array in {path}")
                if buffer.strip():
                    raise ValueError(f"Unexpected content after JSON array in {path}")


def normalize_pair(instruction, response):
    instruction = "" if instruction is None else str(instruction).strip()
    response = "" if response is None else str(response).strip()
    if not instruction or not response:
        raise ValueError("Instruction and assistant response must both be non-empty")
    return {
        "conversations": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
    }


def normalize_original(row):
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Original SFT row has no conversations list")
    if not any(message.get("role") == "assistant" for message in conversations):
        raise ValueError("Original SFT row has no assistant response")
    # Preserve reasoning_content, tools, and tool_calls from the original data.
    return {"conversations": conversations}


def reservoir_sample_jsonl(path, sample_size, rng):
    """Uniformly sample up to ``sample_size`` non-empty JSONL rows in one pass."""
    reservoir = []
    seen = 0
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if seen < sample_size:
                reservoir.append(line)
            else:
                replacement = rng.randrange(seen + 1)
                if replacement < sample_size:
                    reservoir[replacement] = line
            seen += 1
            if seen % 100000 == 0:
                print(f"  scanned {seen:,} original T2T rows", flush=True)
    return reservoir, seen


class ExternalRandomShuffler:
    """Bounded-memory random shuffle backed by sorted temporary chunks."""

    def __init__(self, output_path, seed, chunk_rows):
        self.output_path = Path(output_path)
        self.rng = random.Random(seed)
        self.chunk_rows = max(1, int(chunk_rows))
        self.buffer = []
        self.chunk_paths = []
        self.fingerprints = set()
        self.duplicate_rows = 0
        self.temporary_dir = tempfile.TemporaryDirectory(
            prefix=".sft_mix_", dir=self.output_path.parent
        )

    def add(self, row):
        serialized = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        fingerprint = hashlib.blake2b(
            serialized.encode("utf-8"), digest_size=16
        ).digest()
        if fingerprint in self.fingerprints:
            self.duplicate_rows += 1
            return False
        self.fingerprints.add(fingerprint)
        self.buffer.append((self.rng.getrandbits(128), serialized))
        if len(self.buffer) >= self.chunk_rows:
            self._flush_chunk()
        return True

    def _flush_chunk(self):
        if not self.buffer:
            return
        self.buffer.sort(key=lambda item: item[0])
        chunk_path = Path(self.temporary_dir.name) / f"chunk_{len(self.chunk_paths):06d}.jsonl"
        with chunk_path.open("w", encoding="utf-8", newline="\n") as chunk_file:
            for key, serialized in self.buffer:
                chunk_file.write(f"{key:032x}\t{serialized}\n")
        self.chunk_paths.append(chunk_path)
        self.buffer.clear()

    @staticmethod
    def _iter_chunk(path):
        with path.open("r", encoding="utf-8") as chunk_file:
            for line in chunk_file:
                key, serialized = line.rstrip("\n").split("\t", 1)
                yield key, serialized

    def finish(self, overwrite=False):
        self._flush_chunk()
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {self.output_path}; pass --overwrite to replace it"
            )
        temporary_output = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        try:
            iterators = [self._iter_chunk(path) for path in self.chunk_paths]
            with temporary_output.open("w", encoding="utf-8", newline="\n") as output_file:
                for _, serialized in heapq.merge(*iterators):
                    output_file.write(serialized + "\n")
            os.replace(temporary_output, self.output_path)
        finally:
            if temporary_output.exists():
                temporary_output.unlink()
            self.temporary_dir.cleanup()


def replay_rows_for_fraction(new_rows, replay_fraction):
    if not 0 <= replay_fraction < 1:
        raise ValueError("Replay fraction must be in [0, 1)")
    if replay_fraction == 0:
        return 0
    return round(new_rows * replay_fraction / (1 - replay_fraction))


def mix_datasets(magicoder_path, math_path, original_path, output_path, *,
                 replay_fraction=0.20, seed=42, chunk_rows=25000, overwrite=False):
    output_path = Path(output_path)
    if not output_path.name.lower().startswith("sft") or output_path.suffix.lower() != ".jsonl":
        raise ValueError("Output must be named sft*.jsonl so Config WebUI classifies it as SFT")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --overwrite to replace it"
        )
    for source_path in (magicoder_path, math_path, original_path):
        if not Path(source_path).is_file():
            raise FileNotFoundError(source_path)

    shuffler = ExternalRandomShuffler(output_path, seed + 1, chunk_rows)
    magicoder_rows = 0
    math_rows = 0
    invalid_magicoder_rows = 0
    invalid_math_rows = 0
    duplicate_magicoder_rows = 0
    duplicate_math_rows = 0
    try:
        print(f"Reading Magicoder: {magicoder_path}", flush=True)
        for row in iter_jsonl(magicoder_path):
            try:
                normalized = normalize_pair(row.get("instruction"), row.get("response"))
            except ValueError:
                invalid_magicoder_rows += 1
                continue
            if shuffler.add(normalized):
                magicoder_rows += 1
            else:
                duplicate_magicoder_rows += 1
        print(
            f"  normalized {magicoder_rows:,} Magicoder rows "
            f"(skipped {invalid_magicoder_rows:,} invalid, "
            f"{duplicate_magicoder_rows:,} duplicates)",
            flush=True,
        )
        print(f"Reading MathInstruct: {math_path}", flush=True)
        for row in iter_json_array(math_path):
            try:
                normalized = normalize_pair(row.get("instruction"), row.get("output"))
            except ValueError:
                invalid_math_rows += 1
                continue
            if shuffler.add(normalized):
                math_rows += 1
            else:
                duplicate_math_rows += 1
        print(
            f"  normalized {math_rows:,} MathInstruct rows "
            f"(skipped {invalid_math_rows:,} invalid, "
            f"{duplicate_math_rows:,} duplicates)",
            flush=True,
        )

        new_rows = magicoder_rows + math_rows
        replay_target = replay_rows_for_fraction(new_rows, replay_fraction)
        replay_candidates = replay_target + max(1000, replay_target // 4)
        print(
            f"Sampling up to {replay_candidates:,} replay candidates from {original_path} "
            f"for a {replay_target:,}-row unique target",
            flush=True,
        )
        if replay_target:
            sampled_lines, original_rows = reservoir_sample_jsonl(
                original_path, replay_candidates, random.Random(seed)
            )
        else:
            sampled_lines, original_rows = [], 0
        random.Random(seed + 2).shuffle(sampled_lines)
        replay_rows = 0
        duplicate_replay_rows = 0
        invalid_replay_rows = 0
        for line in sampled_lines:
            try:
                normalized = normalize_original(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                invalid_replay_rows += 1
                continue
            if shuffler.add(normalized):
                replay_rows += 1
                if replay_rows == replay_target:
                    break
            else:
                duplicate_replay_rows += 1
        if replay_rows != replay_target:
            raise ValueError(
                f"Only found {replay_rows:,} unique T2T replay rows; "
                f"target was {replay_target:,}"
            )
        print("Writing externally shuffled output", flush=True)
        shuffler.finish(overwrite=overwrite)
    except Exception:
        shuffler.temporary_dir.cleanup()
        raise

    total_rows = new_rows + replay_rows
    return {
        "magicoder_rows": magicoder_rows,
        "math_rows": math_rows,
        "invalid_magicoder_rows": invalid_magicoder_rows,
        "invalid_math_rows": invalid_math_rows,
        "duplicate_magicoder_rows": duplicate_magicoder_rows,
        "duplicate_math_rows": duplicate_math_rows,
        "duplicate_replay_rows": duplicate_replay_rows,
        "invalid_replay_rows": invalid_replay_rows,
        "replay_rows": replay_rows,
        "original_rows": original_rows,
        "total_rows": total_rows,
        "replay_fraction": replay_rows / total_rows if total_rows else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magicoder", type=Path, default=DEFAULT_MAGICODER)
    parser.add_argument("--math", type=Path, default=DEFAULT_MATH)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replay-fraction", type=float, default=0.20,
        help="Target fraction of original T2T rows in the final mixture (default: 0.20)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-rows", type=int, default=25000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stats = mix_datasets(
        args.magicoder, args.math, args.original, args.output,
        replay_fraction=args.replay_fraction,
        seed=args.seed,
        chunk_rows=args.chunk_rows,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {stats['total_rows']:,} rows to {args.output}\n"
        f"  Magicoder: {stats['magicoder_rows']:,} "
        f"(skipped {stats['invalid_magicoder_rows']:,} invalid, "
        f"{stats['duplicate_magicoder_rows']:,} duplicates)\n"
        f"  MathInstruct: {stats['math_rows']:,} "
        f"(skipped {stats['invalid_math_rows']:,} invalid, "
        f"{stats['duplicate_math_rows']:,} duplicates)\n"
        f"  T2T replay: {stats['replay_rows']:,}/{stats['original_rows']:,} "
        f"({stats['replay_fraction']:.1%} of final mix; skipped "
        f"{stats['invalid_replay_rows']:,} invalid, "
        f"{stats['duplicate_replay_rows']:,} duplicates)"
    )


if __name__ == "__main__":
    main()
