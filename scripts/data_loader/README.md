# Dataset loaders

Run commands from the repository root. Actual corpora, provenance, and audit
reports live under `dataset/`.

This directory contains training dataset classes, JSONL/Parquet source
resolution, streaming chunks, sequence packing, and cache management.
Trainers import these modules through `scripts.data_loader`.

Offline collection, generation, mixing, cleaning, and audit scripts live in
[data_builder](../data_builder/README.md).

```python
from scripts.data_loader.lm_dataset import PretrainDataset, SFTDataset
```

```bash
python scripts/data_builder/prepare_sft_data.py codealpaca-local
python scripts/data_builder/filter_anomaly_candidates.py dataset/sft_t2t_mini.jsonl \
  --profile all --output dataset/review_candidates/t2t_mini_candidates.jsonl
```
