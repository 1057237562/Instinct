# Dataset scripts

This directory contains dataset collection, normalization, mixing, assembly,
and full-audit utilities. Run commands from the repository root so model,
dataset, and cache paths resolve consistently.

## Performance

Large JSONL assembly and audit jobs use bounded batches and worker threads.
Set the default globally with `INSTINCT_DATA_WORKERS`, or pass `--workers`
where supported. A useful starting point is 8-16 workers; increasing the count
beyond the storage or tokenizer throughput usually does not help.

```bash
python dataset/scripts/assemble_continue_pretrain_1b.py --workers 16 --batch-size 2048
python dataset/scripts/audit_continue_pretrain_1b.py --workers 16 --batch-size 2048
python dataset/scripts/collect_first_sft_sources.py --workers 8
python dataset/scripts/collect_quality_python_sft.py --workers 6
```

Parallel validation preserves input order. A single writer owns each output
file, and external shuffling uses bounded temporary chunks, so worker threads
never append concurrently to the same JSONL file.

## Groups

- `collect_*.py`: download and filter source corpora.
- `build_*.py`: construct pretraining or SFT datasets.
- `mix_*.py`: combine and deterministically shuffle prepared datasets.
- `audit_*.py` and `verify_*.py`: perform independent full-record checks.
- `assemble_*.py`, `finalize_*.py`, `recover_*.py`: materialize or recover
  final artifacts from stage caches.

Training launchers remain under `trainer/` or `scripts/`; these utilities only
prepare and verify data.
