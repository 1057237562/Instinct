# Dataset builders

This directory contains offline collection, generation, normalization,
mixing, assembly, cleaning, and full-audit utilities. The repository's
`dataset/` directory holds data files, provenance, and audit reports. Run
commands from the repository root so model, dataset, and cache paths resolve
consistently.

Training loaders and cache modules live separately in `../data_loader/`, for
example `from scripts.data_loader.lm_dataset import PretrainDataset`.
Import generation helpers through `scripts.data_builder`.

## Performance

Large JSONL assembly and audit jobs use bounded batches and worker threads.
Set the default globally with `INSTINCT_DATA_WORKERS`, or pass `--workers`
where supported. A useful starting point is 8-16 workers; increasing the count
beyond the storage or tokenizer throughput usually does not help.

```bash
python scripts/data_builder/assemble_continue_pretrain_1b.py --workers 16 --batch-size 2048
python scripts/data_builder/audit_continue_pretrain_1b.py --workers 16 --batch-size 2048
python scripts/data_builder/collect_first_sft_sources.py --workers 8
python scripts/data_builder/collect_quality_python_sft.py --workers 6
```

Parallel validation preserves input order. A single writer owns each output
file, and external shuffling uses bounded temporary chunks, so worker threads
never append concurrently to the same JSONL file.

## Groups

- `collect_*.py`: download and filter source corpora.
- `build_*.py`: construct pretraining or SFT datasets.
- `mix_*.py`: combine and deterministically shuffle prepared datasets.
- `audit_*.py` and `verify_*.py`: perform independent full-record checks.
- `filter_anomaly_candidates.py`: stream corpora and export suspicious rows,
  with source locations and review reasons, for AI/human triage.
- `audit_identity_contamination.py`: count assistant-side model-identity
  contamination in a corpus. It separates hard self-identity claims from
  factual third-party mentions, checks `reasoning_content` as well as `content`
  (the SFT template trains both), and also reports rows where the model's own
  brand name is attached to a foreign vendor as a renamed-corpus artifact.
- `assemble_*.py`, `finalize_*.py`, `recover_*.py`: materialize or recover
  final artifacts from stage caches.

Training launchers remain under `trainer/` or `scripts/`; these utilities only
prepare and verify data.

## Candidate triage for suspicious data

The anomaly filter supports JSONL, gzip-compressed JSONL, Parquet, and
directories of shards. It writes whole candidate rows to a separate JSONL and
an audit report; it does not edit or delete source rows. Review candidates
before creating a cleaned training dataset.

```bash
python scripts/data_builder/filter_anomaly_candidates.py \
  dataset/sft_t2t_mini.jsonl \
  --profile all \
  --output dataset/review_candidates/t2t_mini_candidates.jsonl

# Search only for possible foreign model identity contamination
python scripts/data_builder/filter_anomaly_candidates.py \
  dataset/sft_t2t_mini.jsonl \
  --profile identity \
  --output dataset/review_candidates/t2t_mini.identity_candidates.jsonl
```

Each candidate includes `_review.source`, `_review.row_number`, rule names,
explanations, a reviewer instruction, and the untouched source `record`.
Malformed JSON lines are included with their raw line. The companion report
records source and output SHA-256 hashes, scan/candidate counts, and counts by
rule. Heuristics intentionally produce candidates rather than automatic
keep/drop decisions.

## Identity-contamination audit

`filter_anomaly_candidates.py` answers "what looks odd here?"; the identity
audit answers "how many rows teach the assistant to be someone else?". It
expects the target identity on the command line defaults (Instinct, developed
by L1bra, no commercial affiliation) and reports layered counts instead of a
single number:

```bash
python scripts/data_builder/audit_identity_contamination.py \
  dataset/sft_t2t_mini.jsonl \
  --output dataset/review_candidates/sft_t2t_mini.identity_candidates.jsonl \
  --emit review
```

- `contaminated_rows`: explicit assistant self-identity claims naming another
  model or provider. Knowledge about other models is not counted, and
  disavowals ("我是 Instinct，不是 Qwen"), role statements ("我是 Qwen 的用户"),
  and hypotheticals are excluded.
- `assistant_brand_affiliation_facts` / `brand_affiliation_fact_rows`: rows that
  attach the model's own name to a foreign vendor — the usual residue of a
  corpus that was de-identified by renaming a vendor's brands.
- `review_rows`: same-sentence self-cue plus foreign name, speculative framing,
  user-asserted identity the assistant accepts, or the brand facts above. A
  reviewer or model decides these.
- `--emit all` also writes factual-mention rows; `--emit contaminated` writes
  only the hard claims.

Treat the counts as triage: verify a sample by reading the rows before
converting them into a cleaning rule.

## Identity repair and topic filtering

- `apply_identity_repairs.py`: replay row patches (exact-substring edits and
  whole-turn regenerations) against a corpus. Every edit is validated - the
  `old` string must occur exactly once - and unpatched rows are copied byte for
  byte, so a repair pass is auditable and never rewrites untouched text.
- `clean_identity_contamination.py`: the drop-based alternative, which removes
  contaminated rows instead of repairing them and writes a sidecar listing each
  removal with its rule and evidence.
- `build_identity_anchors.py`: emit the small reviewed identity-anchor pool
  (canonical answer, denial answers, no-consciousness answers, English variants)
  to sample into the final SFT mix at about 0.5%-2%.
- `remove_topic_rows.py`: remove rows matching a topic profile. A profile lists
  unambiguous terms plus ambiguous ones (同志 "comrade", 百合 "lily", 同性 in
  同性相斥, "trans", "pride"); an ambiguous term only removes a row when a second
  profile term appears in the same row, so ordinary text survives. Removed rows
  go to a sidecar for review.

```bash
python scripts/data_builder/apply_identity_repairs.py dataset/sft_t2t_mini.jsonl \
  --patches dataset/review_candidates/_repair_batches \
  --output dataset/sft_t2t_mini.identity_repaired.jsonl

python scripts/data_builder/remove_topic_rows.py dataset/sft_t2t_mini.identity_repaired.jsonl \
  --profile lgbt --output dataset/sft_t2t_mini.train_ready.jsonl
```
