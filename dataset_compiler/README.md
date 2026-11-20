# dataset_compiler

Compile Instinct JSONL corpora into Parquet, and check that the result is the
same data. The trainers read the output directly — a compiled corpus is a
drop-in replacement for its `.jsonl` source, selected the same way in the
Config WebUI or passed to `--data_path`.

```bash
cd dataset_compiler
cargo build --release

# Compile one corpus (schema inferred from the whole file)
./target/release/dataset_compiler compile \
  -i ../dataset/pretrain_coder_12b.jsonl \
  -o ../dataset/pretrain_coder_12b.parquet

# Confirm the parquet file holds exactly the source rows
./target/release/dataset_compiler verify \
  ../dataset/pretrain_coder_12b.jsonl ../dataset/pretrain_coder_12b.parquet --rows 2000

# Footer summary: rows, row groups, codec, schema
./target/release/dataset_compiler inspect ../dataset/pretrain_coder_12b.parquet
```

A directory input compiles every top-level `*.jsonl` into the output directory
(`--recursive` descends further). Each run writes a report sidecar next to the
output — `<output>.report.json` — holding the source digest, row counts, column
types, compression, throughput and any warnings.

## Why

* **3x smaller on disk.** These corpora are dominated by long text, which zstd
  shrinks to roughly a third.
* **No Python JSON parsing.** Loading a JSONL corpus burns CPU in the `datasets`
  JSON reader; parquet is read straight into Arrow.
* **`token_count` survives.** Instinct corpora carry it, and the compiler keeps
  it, so bounded streaming can plan chunks by token totals without tokenizing
  anything (`scripts/data_loader/streaming_chunks.py`).
* **Bounded cache.** A parquet source is read from row groups, and streaming
  materializes a chunk by re-encoding whole row groups instead of copying bytes
  from an arbitrary offset.

## Schema contract

The output must be loadable by `datasets.load_dataset('parquet', ...)` and by
the Instinct trainers, so the compiler emits the shapes they already handle:

| Source | Column |
| --- | --- |
| `{"text": "..."}` | `text` as `utf8`, plus every other scalar key carried through (`token_count`, `license`, `source`, ...) |
| `{"conversations": [...]}` | `list<struct<...>>` with all fields `utf8` |
| `{"chosen": [...], "rejected": [...]}` | the same message lists, one column each |
| `{"conversations": [...], "gt": [...]}` | message list plus `gt` as a typed list |

Rules that follow from how the loaders read these files:

* Message fields are **always** `utf8`. `tools` and `tool_calls` hold JSON
  payloads and the loaders call `json.loads` on them when they are strings.
* Canonical message fields are ordered `role`, `content`, `reasoning_content`,
  `tools`, `tool_calls`, then any extra key the corpus uses, so the file reads
  in the same order as the loader's feature declaration.
* Objects outside a message list — a `metadata` blob, a `tools` array — become
  JSON text rather than a nested struct, which keeps every column loadable.
* A key that is irreconcilable across rows (sometimes a list, sometimes a
  string) falls back to JSON text, and the report lists it under `warnings`.
  No row is ever dropped for shape reasons.
* Extra message keys are **kept**, which is more permissive than the JSONL path
  (the SFT loader casts JSONL rows onto its five declared fields). The parquet
  loader reads the file's own schema instead.

`--format` names the corpus family — `pretrain`, `sft`, `dpo`, `agent`,
`generic`, or `auto` (the default, detected from the first row). The preset pins
the columns the loader requires, puts them first, and validates them; shape
inference does the rest.

## Aligning a compiled file with the JSONL chunk plan

Bounded streaming (`--dataset_streaming`) cuts the source into chunks and stores
a *chunk cursor* — "chunk k, batch b" — in the checkpoint. The JSONL planner
cuts by raw file bytes while a compiled file is cut by its Arrow footprint, so
the same chunk index lands on a different row: switching containers mid-epoch
replays or skips part of one chunk (0.1% of the corpus early in a run, a few
percent late in one).

`--align-chunk-bytes` removes that. The compiler reproduces the JSONL planner's
byte rule during its schema scan and closes a row group at every chunk boundary,
so **one row group is one streaming chunk**:

```bash
./target/release/dataset_compiler compile \
  -i ../dataset/pretrain_x.jsonl -o ../dataset/pretrain_x.parquet \
  --align-chunk-bytes 1GiB          # = the trainer's --streaming_chunk_mb
```

The boundaries are recorded in the footer (`instinct.aligned_chunk_rows`,
`instinct.aligned_chunk_bytes`) and `scripts/data_loader/streaming_chunks.py` uses them
verbatim, so the JSONL plan and the parquet plan describe exactly the same row
ranges and a checkpoint's cursor keeps pointing at the same rows. `inspect`
shows the alignment, and the sidecar report records it.

Worth knowing:

* The flag needs the full schema scan (the default) and a corpus without blank
  lines — the JSONL planner counts a blank line as a row, the compiler does not,
  so byte-driven boundaries could not be reproduced.
* An aligned file keeps its chunks even if the trainer later asks for a
  different chunk size; the planner logs that it is using the file's own
  boundaries, and `--streaming_chunk_mb` then only applies to JSONL sources.
* An aligned file has one row group per chunk instead of ~128 MiB groups, so a
  chunk maps onto exactly one group and materializing it decodes nothing else.

## How the schema is decided

A parquet file has a single schema, so the compiler must know the shape of every
row before it writes the first record batch. By default it **scans the whole
file first** (one extra sequential read) and then writes. `--pre-scan false`
samples the leading `--schema-rows` rows instead, trading that guarantee for one
less pass: a key that first appears later is then stored as JSON text.

Parsing is parallel (one reader thread, a rayon pool for parsing), and the
in-flight window bounds memory — roughly `threads x 32 MiB` of raw source.

## Measured on this repository

| Corpus | Rows | JSONL | Parquet | Ratio | Time |
| --- | --- | --- | --- | --- | --- |
| `pretrain_arxiv_abstracts_160m_tokens_4096.jsonl` | 475,523 | 558 MiB | 181 MiB | 3.09x | 2.9 s |
| `sft_t2t_mini.jsonl` (20k-row slice) | 20,000 | 55 MiB | 18 MiB | 2.98x | 0.4 s |
| `agent_rl.jsonl` (20k-row slice) | 20,000 | 39 MiB | 11 MiB | 3.60x | 0.3 s |
| `pretrain_coder_12b.jsonl` | 13,267,273 | 34.8 GiB | 13.3 GiB | 2.61x | 3.2 min |

Timings include the schema scan (32 threads). `verify` reports zero value
mismatches on all of them; on `pretrain_coder_12b` the row counts agree exactly
and the preserved `token_count` column sums to 11,999,990,666 tokens, matching
`dataset/pretrain_coder_12b.report.json`.

That corpus is far larger than the default 5 GiB dataset-cache budget, so
`--dataset_streaming auto` keeps it on the bounded path. Compiling it with
`--align-chunk-bytes 1GiB` (197 s, 13.30 GiB, 2.61x) produced 35 row groups, and
the streaming planner reports **35 chunks whose row ranges, row counts and token
totals are identical to the JSONL source's** — the two plans name the same rows,
so a training cursor survives the container switch unchanged. Planning reads
only the `token_count` column, and materializing one 381,033-row chunk takes
4.1 s (394 MiB).

## Split an existing Parquet corpus

```powershell
./target/release/dataset_compiler split `
  -i ../dataset/pretrain_coder_l2l3_24b.parquet `
  -o ../dataset/pretrain_coder_l2l3_24b_shards_5gib `
  --target-size 5GiB --threads 2
```

The destination directory must be new. This copies encoded column chunks;
there is no text decoding or recompression. Shards keep contiguous whole row
groups, so sizes are approximate (the nearest row-group boundary). The source
file is only read. Rayon copies and verifies independent shards concurrently;
each shard's writer is sequential. Use a small thread count and a low process
priority when sharing the machine with training; disk bandwidth is still shared.

Every encoded column chunk is compared against the source using SHA-256, and
each shard's schema, row counts and `token_count` sum are checked. File-level
metadata is retained, with aligned chunk boundaries rebased to the shard's
local row numbers. Optional page indexes and bloom filters are omitted; row
values, columns, compression and order are unchanged. Completed shards have
lexically ordered `part-00001-of-00005.parquet` names. Incomplete files end in
`.partial`; `manifest.json` appears only after the whole operation succeeds
and records original row ranges, token totals and output hashes. Only use the
directory once that manifest reports `complete`.

Pass one shard as `--data_path` to train it separately, or pass the directory
to read the complete corpus. The production native cuBLAS loader supports exact
continuation from the original file to a complete shard directory: the manifest
retains `source_identity` and `source_row_group_rows`, and `verify-shards
DIRECTORY/manifest.json --threads 2` checks every file's SHA-256 and complete
row-group coverage before use. The loader retains global group numbers, batch
shuffle seeds, cache keys and the original checkpoint identity while routing
physical reads to the corresponding shard/local group. It can rebuild plans
and packed caches without the original large file. Existing checkpoints need
no rewriting. Running processes keep their current source; the new directory
is selected on the next resume. This requires the updated native loader and
compiler; it does not change standard PyTorch resume contracts. Selecting a
manifest-listed single shard in the updated native loader limits that run to
the selected file while retaining the global cursor. Its end uses the usual
final-batch gradient flush, saves the full resume state and pauses. The production
wrapper prepares the next shard for the next Resume click; the final shard
completes the epoch. A partial accumulation is flushed at that boundary, so
optimizer update grouping can differ from a continuous run. A global native
plan embedded in `native_plans` permits use with only the selected file present.

## Compile options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--compression` / `--compression-level` | `zstd` / codec default | `snappy`, `gzip`, `brotli`, `none` also available |
| `--row-group-rows` / `--row-group-mb` | `65536` / `128` | Row groups are also the streaming chunk granularity |
| `--align-chunk-bytes` | off | Cut row groups at the JSONL planner's chunk boundaries (`1GiB`, `1024MiB`, or plain bytes) so a training cursor keeps its meaning |
| `--pre-scan` | `true` | Scan the whole file for an exact schema |
| `--dictionary` | `auto` | Dictionary encoding everywhere except a `text` column |
| `--statistics` | `false` | Min/max on multi-kilobyte strings only bloats the footer |
| `--on-error` | `fail` | `skip` keeps row counts aligned by writing all-null placeholders |
| `--limit` | off | Compile a prefix, for sampling or a smoke test |
| `--threads` | 0 (all cores) | Parse threads |

## Tests

```bash
cargo test                    # unit + integration tests, no fixture corpus needed
python -m pytest tests/test_parquet_source.py   # Python side, skips if unbuilt
```
