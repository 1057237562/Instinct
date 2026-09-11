---
dataset_info:
  features:
  - name: fingerprint
    dtype: 'null'
  - name: sha1
    dtype: string
  - name: seed
    dtype: string
  - name: response
    dtype: string
  - name: concepts
    sequence: string
  - name: prompt
    dtype: string
  - name: instruction
    dtype: string
  - name: id
    dtype: int64
  splits:
  - name: train
    num_bytes: 261340280
    num_examples: 50661
  download_size: 90128158
  dataset_size: 261340280
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: odc-by
pretty_name: StarCoder2-15b Self-Alignment Dataset (50K)
---

Final self-alignment training dataset for StarCoder2-Instruct. 

- `seed`: Contains the seed Python function
- `concepts`: Contains the concepts generated from the seed
- `instruction`: Contains the instruction generated from the concepts
- `response`: Contains the execution-validated response to the instruction

This dataset utilizes seed Python functions derived from the [MultiPL-T pipeline](https://arxiv.org/abs/2308.09895).