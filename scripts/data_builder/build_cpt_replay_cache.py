"""Build the independent 200M-token replay portion of the CPT corpus."""
import hashlib
import json
from pathlib import Path
import sys

import orjson
from datasets import load_dataset  # noqa: F401  # before transformers/torch on Windows
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
import scripts.data_builder.collect_continue_pretrain_1b as base
from scripts.data_builder.complete_cpt_stage_cache import CACHE, PRIOR, REPLAY_TARGETS, load_cache


def main():
    tokenizer = AutoTokenizer.from_pretrained(base.ROOT / "model", local_files_only=True).backend_tokenizer
    replay = {source: load_cache(source)[:3] for source in REPLAY_TARGETS}
    for band in range(8):
        if all(replay[source][2]["tokens"] >= REPLAY_TARGETS[source] - base.MAX_TOKENS
               for source in replay):
            break
        handles = {source: replay[source][0].open("ab") for source in replay}
        try:
            with PRIOR.open("rb") as handle:
                for index, line in enumerate(handle):
                    row = orjson.loads(line)
                    source = base.REPLAY_MAP.get(row.get("source"))
                    if source not in replay:
                        continue
                    stats = replay[source][2]
                    if stats["tokens"] >= REPLAY_TARGETS[source] - base.MAX_TOKENS:
                        continue
                    value = int.from_bytes(
                        hashlib.blake2b(f"{base.SEED}/{index}".encode(), digest_size=8).digest(), "big"
                    ) / 2**64
                    probability = base.REPLAY_PROB[source]
                    if not band * probability <= value < min(1.0, (band + 1) * probability):
                        continue
                    text = row.get("text")
                    if not base.clean(text, 80, 50000):
                        continue
                    fingerprint = base.h16(text)
                    if fingerprint in replay[source][1]:
                        continue
                    tokens = len(tokenizer.encode(text, add_special_tokens=False).ids) + 2
                    if not 64 <= tokens <= base.MAX_TOKENS or stats["tokens"] + tokens > REPLAY_TARGETS[source]:
                        continue
                    output = {
                        "text": text, "source": source, "source_id": str(index),
                        "license": row.get("license") or "inherited; see pretrain_codespecialist report",
                        "url": row.get("url") or "", "token_count": tokens,
                        "metadata": {"prior_source": row.get("source")},
                    }
                    encoded = orjson.dumps(output) + b"\n"
                    handles[source].write(encoded)
                    replay[source][1].add(fingerprint)
                    stats["rows"] += 1
                    stats["tokens"] += tokens
                    stats["bytes"] += len(encoded)
        finally:
            for handle in handles.values():
                handle.close()
        print("replay", {source: replay[source][2]["tokens"] for source in replay}, flush=True)

    stats = {source: dict(value[2]) for source, value in replay.items()}
    for source, target in REPLAY_TARGETS.items():
        if not target - base.MAX_TOKENS <= stats[source]["tokens"] <= target:
            raise RuntimeError(f"Replay component below target: {source}: {stats[source]}")
    (CACHE / "replay_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
