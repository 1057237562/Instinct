---
language:
- en
license:
- apache-2.0
- mit
- cc-by-4.0
task_categories:
- text-generation
configs:
- config_name: default
  data_files:
  - split: train
    path: _hf_raw/ALL/train-*.parquet
  - split: test
    path: _hf_raw/ALL/test-*.parquet
---


## Dataset Description:
TACO (Topics in Algorithmic COde generation dataset) is a collection of 26,443 competitive-programming problems aggregated from Codeforces, CodeChef, GeeksforGeeks, Codewars, HackerEarth, Aizu, AtCoder, Kattis, LeetCode and HackerRank, each with ground-truth Python solutions and **executable test cases** (`input_output`). It is the open dataset used here to provide verifiable (problem, tests) pairs for RLVR/GRPO training in the Instinct pipeline, complementing SFT-only sources.

This directory mirrors the upstream parquet layout of `BAAI/TACO` unchanged under `_hf_raw/` (no reconversion). Local stats collected at download time (2026-09-18) are documented below.

- [GitHub Repo (FlagOpen/TACO)](https://github.com/FlagOpen/TACO/)
- [Paper](https://arxiv.org/abs/2312.14852) — TACO: Topics in Algorithmic COde generation dataset
- [Upstream HF card](https://huggingface.co/datasets/BAAI/TACO)

## Dataset Quantification

| Split | Files | Rows | Bytes (parquet) |
|--------|---------|---------|---------|
| train      | 9 shards (`_hf_raw/ALL/train-0000*-of-00009.parquet`)     | 25,443     | 2,174,060,481 |
| test       | 1 shard (`_hf_raw/ALL/test-00000-of-00001.parquet`)       | 1,000      | 245,784,461 |
| Total      | 10 files + upstream README                                | 26,443     | ~2.42 GB |

In-memory (arrow) size: 4,720,792,728 bytes. Difficulty levels: `EASY`, `MEDIUM`, `MEDIUM_HARD`, `HARD`, `VERY_HARD` (+ `UNKNOWN_DIFFICULTY` in train).

### Difficulty distribution (collected locally)

| difficulty | train | test |
|------------|-------|------|
| EASY               | 8,904 | 200 |
| MEDIUM             | 3,244 | 200 |
| MEDIUM_HARD        | 2,745 | 200 |
| HARD               | 3,162 | 200 |
| VERY_HARD          | 2,374 | 200 |
| UNKNOWN_DIFFICULTY | 5,014 | 0   |

### Source distribution (train)

codeforces 8,193 · codechef 3,352 · geeksforgeeks 2,680 · codewars 2,460 · hackerearth 2,390 · aizu 2,151 · atcoder 1,440 · kattis 1,236 · leetcode 777 · hackerrank 764

### Test-case coverage (verified over ALL rows, not sampled)

`input_output` parses as JSON with usable tests:

| category | train | test |
|----------|-------|------|
| stdio style with non-empty `inputs`/`outputs` | 21,809 (85.7%) | 945 |
| fn-call style with non-empty `inputs`/`outputs` | 2,892 (11.4%) | 55 |
| stdio style, tests empty | 390 (1.5%) | 0 |
| fn-call style, tests empty (truncated upstream) | 351 (1.4%) | 0 |
| unparseable `input_output` | 1 (0.004%) | 0 |
| **usable test cases total** | **24,701 (97.1%)** | **1,000 (100%)** |

## Schema

All columns are strings (`datasets` 4.x, parquet, config `ALL`):

| field | type | content |
|-------|------|---------|
| `question` | string | problem statement (English) |
| `solutions` | string | JSON string → list of ground-truth Python solution strings |
| `starter_code` | string | starter code to prepend (non-empty ⇒ usually fn-call style) |
| `input_output` | string | JSON string with the test cases — see below |
| `difficulty` | string | EASY / MEDIUM / MEDIUM_HARD / HARD / VERY_HARD / UNKNOWN_DIFFICULTY |
| `source` | string | one of the sites listed above |
| `url` | string | link to the original problem |
| `name` | string | problem title (often null) |
| `date` | string | date of the problem |
| `time_limit`, `memory_limit` | string | judge limits |
| `Expected Time Complexity`, `Expected Auxiliary Space` | string | often null |
| `picture_num` | string | number of images in the statement |
| `raw_tags` | string | `eval`-able list of topic tags |
| `tags` | string | `eval`-able list of algorithm tags |
| `skill_types` | string | `eval`-able list of skill types |

## The `input_output` field (for GRPO reward builders)

`input_output` is a JSON string. Parse it with `json.loads(row["input_output"])`. It has exactly two shapes in this version of the dataset — there is **no** `fn_tests` key:

**1. stdio style** (no `fn_name`; typical for Codeforces/AtCoder/etc.):
```json
{
  "inputs":  ["4\n4\n1 2 3 4\n...\n", "..."],
  "outputs": ["4 3 2 1\n...\n", "..."]
}
```
`inputs[i]` is the raw text fed to stdin; `outputs[i]` is the expected stdout. Lists are equal length.

**2. function-call style** (`fn_name` present; typical for Codewars/LeetCode, usually with non-empty `starter_code`):
```json
{
  "fn_name": "is_anagram",
  "inputs":  [["foefet", "toffee"], ["Buckethead", "DeathCubeK"]],
  "outputs": [[true], [false]]
}
```
`inputs[i]` is the **list of positional argument values** for call `i`; `outputs[i]` is a **1-element list** containing the expected return value.

### Caveats

- **Truncated function tests:** 351 train rows have `fn_name` but empty `inputs`/`outputs` — upstream TACO truncated these during collection (they cannot be executed as-is; skip them or regenerate tests). Another 390 train rows are stdio style with no tests at all. 1 train row has unparseable `input_output`. Filter: keep rows where `inputs` and `outputs` are both non-empty.
- **Whitespace normalization:** expected outputs may contain stray leading/trailing newlines (e.g. `"\n4 3 2 1\n..."`); a checker should strip/normalize whitespace before comparing.
- Very large tests are rare but possible; the test split averages 202.3 test cases per problem.

## How to use it — extracting (problem, tests) pairs

```
from datasets import load_dataset
import json

ds = load_dataset(
    "parquet",
    data_files={
        "train": "dataset/taco/_hf_raw/ALL/train-*.parquet",
        "test": "dataset/taco/_hf_raw/ALL/test-00000-of-00001.parquet",
    },
)

for row in ds["train"]:
    io = json.loads(row["input_output"])
    if not io.get("inputs") or not io.get("outputs"):
        continue  # truncated / testless rows
    problem = row["question"]
    solutions = json.loads(row["solutions"]) if row["solutions"] else []
    if "fn_name" in io:
        tests = [{"fn_name": io["fn_name"], "args": a, "expected": o[0]}
                 for a, o in zip(io["inputs"], io["outputs"])]
    else:
        tests = [{"stdin": i, "expected_stdout": o}
                 for i, o in zip(io["inputs"], io["outputs"])]
    # yield (problem, solutions, tests) -> sandbox-executable reward signal
```

Upstream usage reference: `load_dataset("BAAI/TACO")` (equivalent content; this local copy exists so the pipeline does not depend on the network).

## Intended Usage:
Provides verifiable unit tests for RLVR: a GRPO trainer can sandbox-execute generated code against the extracted (stdin → stdout) or (fn_name, args → return) pairs and reward only solutions that pass. Also usable as an SFT source via `solutions`.

## License/Terms of Use:
The TACO dataset authored by BAAI, Shandong Normal University and Peking University is released under the **Apache 2.0 License**. The data also includes content under other permissive licenses such as **MIT** (e.g. HackerEarth materials via the Description2Code dataset), and web-crawled content used under **CC BY 4.0**. Problem statements originate from Codeforces, CodeChef, GeeksforGeeks, HackerRank, etc. and remain subject to their original site terms. See the upstream card's License section for the full attribution list (APPS, CodeContests provenance included).

## Reference(s):

* Li et al., [TACO: Topics in Algorithmic COde generation dataset](https://arxiv.org/abs/2312.14852), arXiv 2023
* [FlagOpen/TACO GitHub](https://github.com/FlagOpen/TACO)
* [BAAI/TACO on Hugging Face](https://huggingface.co/datasets/BAAI/TACO)

## Cleaned derivative (Instinct)

`taco.clean.jsonl` (16,189 rows, 2.6 GB) — produced by `scripts/data_builder/clean_taco.py`, full stats in `taco.clean.report.json`, dropped rows in `taco.excluded.jsonl`.

- Per-row schema: `question` (HTML→text), `source/difficulty/url/tags/raw_tags/skill_types`, `starter_code`, `is_fn_call`+`fn_name` (5,279 fn-call rows run in-process; 10,910 stdio rows diff in sandbox), `tests{inputs,outputs}` (paired), `n_tests`, `solutions` (python, ≤8 each, exact-deduped).
- Cleaning: 26,443 → 16,189 kept. Excluded: **8,777 cross-duplicates of code_contests** (normalized statement text; CC is canonical), 698 image-in-statement, 739 no/mismatched tests, 39 in-dataset dups, 1 bad JSON.
- No chain-of-thought: `solutions` is raw python code (prose-like rows 1/26,443).
- GRPO recipe: `json.loads(input_output)` equivalent already parsed into `tests`; filter `is_fn_call` rows into the in-process harness (args in `inputs[i]`, expected return in `outputs[i][0]`), stdio rows into the sandbox; 2,011 rows have problem+tests but no reference solution (RLVR-only, still usable).
