---
language:
- en
license:
- cc-by-4.0
task_categories:
- text-generation
size_categories:
- 10K<n<100K
pretty_name: CodeContests (deepmind) — parquet snapshot for RLVR
configs:
- config_name: default
  data_files:
  - split: train
    path: _hf_parquet/data/train-*.parquet
  - split: validation
    path: _hf_parquet/data/valid-*.parquet
  - split: test
    path: _hf_parquet/data/test-*.parquet
---

# CodeContests (deepmind/code_contests) — local parquet snapshot

## Dataset Description

Local snapshot of [deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests)
(commit `802411c3010cb00d1b05bad57ca77365a3c699d6`), the competitive programming dataset
used to train [AlphaCode](https://doi.org/10.1126/science.abq1158)
([paper](https://arxiv.org/abs/2203.07814)).

This is the **RLVR material** for Instinct: ~13.6k problems, each carrying **executable
test cases** (paired stdin/stdout strings) in three independent pools (`public_tests`,
`private_tests`, `generated_tests`), plus correct/incorrect human solutions and
Codeforces metadata. The upstream repo now ships native **parquet** files under `data/`,
so `datasets` 4.x loads them directly with no loading script / `trust_remote_code`:

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files={
    "train": "dataset/code_contests/_hf_parquet/data/train-*.parquet",
    "validation": "dataset/code_contests/_hf_parquet/data/valid-*.parquet",
    "test": "dataset/code_contests/_hf_parquet/data/test-*.parquet",
})
```

### Size and splits

| split | problems | parquet bytes |
|---|---|---|
| train | 13,328 | 7,509,753,086 (39 shards) |
| validation | 117 | 51,829,044 (1 shard) |
| test | 165 | 63,077,400 (1 shard) |
| **total** | **13,610** | **7,624,659,530 (~7.1 GiB)** |

Files are kept in the raw upstream layout under `_hf_parquet/` (see `collect_report.json`
for the full file list with sizes and sha256).

## Test-case structure (for GRPO reward builders)

**Important:** in this parquet conversion the three test fields are a **struct of two
parallel string lists**, *not* a list of `{input, output}` dicts (the older
`dataset_infos.json`/TFRecord layout documented elsewhere is different):

```python
{
  "public_tests":    {"input": ["<stdin test 1>", ...], "output": ["<expected stdout 1>", ...]},
  "private_tests":   {"input": [...], "output": [...]},
  "generated_tests": {"input": [...], "output": [...]},
  ...
}
```

- `len(tests["input"]) == len(tests["output"])` always holds; zip them to get cases.
- Every string already ends with `\n`; compare program stdout to `output` after
  normalizing trailing whitespace (DeepMind judges with output comparison; many outputs
  end with a trailing space per token line, e.g. `"5 2 1 3 4 \n"`).
- Some problems use file IO instead of stdin/stdout — check `input_file` / `output_file`.
- `time_limit` is `{"seconds": int, "nanos": int}`; `memory_limit_bytes` is an int.

### Documented example (test split, row 0)

```python
{"input": "5 2\nAA\nAB\nBB\nBA\nAZ\n", "output": "5 2 1 3 4 \n"}   # public_tests[0]
# generated_tests[0]:
{"input": "5 2\nAA\nAB\nBB\nBA\nZA\n", "output": "2 1 3 4 5\n"}
```

### Coverage (measured over all 13,610 rows)

| pool | problems with ≥1 test | fraction | total individual cases |
|---|---|---|---|
| public_tests | 13,495 | 99.2% | 26,818 |
| private_tests | 8,236 | 60.5% | 200,016 |
| generated_tests | 12,810 | 94.1% | 1,107,713 |

Every problem has at least one public *or* private test (100% of scanned rows), so every
problem is usable for execution-based reward. Structure was verified on the full
test/validation splits and the first 1,000 train rows: fields are always
parallel string lists with equal lengths.

## Full schema

| field | type | notes |
|---|---|---|
| `name` | string | problem name, e.g. `"1575_A. Another Sorting Problem"` |
| `description` | string | natural-language problem statement (English; see `is_description_translated`) |
| `public_tests` | `{input: [string], output: [string]}` | visible in the statement; safe to show the model |
| `private_tests` | `{input: [string], output: [string]}` | hidden tests; **do not** put in the prompt |
| `generated_tests` | `{input: [string], output: [string]}` | auto-generated, validated by known-correct solutions |
| `source` | int (ClassLabel) | 0=UNKNOWN_SOURCE, 1=CODECHEF, 2=CODEFORCES, 3=HACKEREARTH, 4=CODEJAM, 5=ATCODER, 6=AIZU |
| `difficulty` | int (ClassLabel) | 0=UNKNOWN_DIFFICULTY, 1=EASY, 2=MEDIUM, 3=HARD, 4=HARDER, 5=HARDEST, 6=EXTERNAL, 7..28=A..V (gradings are non-comparable across sources) |
| `solutions` | `{language: [int], solution: [string]}` | correct solutions; language 0=UNKNOWN, 1=PYTHON(2), 2=CPP, 3=PYTHON3, 4=JAVA |
| `incorrect_solutions` | `{language: [int], solution: [string]}` | known-wrong solutions |
| `cf_contest_id` | int64 | Codeforces contest id |
| `cf_index` | string | `"A"`, `"B"`, ... |
| `cf_points` | float32 | contest points |
| `cf_rating` | int32 | Codeforces rating (0 if unknown; best difficulty signal for CF problems) |
| `cf_tags` | list[string] | e.g. `["data structures", "sortings"]` |
| `is_description_translated` | bool | statement was machine-translated to English |
| `untranslated_description` | string | original statement when translated |
| `time_limit` | `{seconds: int64, nanos: int64}` | execution time limit |
| `memory_limit_bytes` | int64 | execution memory limit |
| `input_file` / `output_file` | string | non-empty ⇒ problem uses file IO instead of stdin/stdout |

## How to use it — pulling (problem, tests) pairs for GRPO

```python
from datasets import load_dataset

ds = load_dataset("parquet", data_files={
    "train": "dataset/code_contests/_hf_parquet/data/train-*.parquet"})["train"]

def iter_grpo_items(ds, max_generated=8):
    for row in ds:
        if not row["description"]:
            continue  # skip problems without a usable statement
        tests = []
        for pool in ("public_tests", "private_tests", "generated_tests"):
            for i, o in zip(row[pool]["input"], row[pool]["output"]):
                tests.append((i, o))
        if not tests:
            continue
        # public tests may be shown in the prompt; private/generated are reward-only
        prompt = row["description"]
        yield {
            "problem_id": row["name"],                      # unique, e.g. "1575_A. ..."
            "prompt": prompt,
            "tests": tests,                                  # [(stdin, expected_stdout), ...]
            "meta": {
                "source": row["source"], "difficulty": row["difficulty"],
                "cf_rating": row["cf_rating"], "cf_tags": row["cf_tags"],
                "time_limit": row["time_limit"], "memory_limit_bytes": row["memory_limit_bytes"],
                "input_file": row["input_file"], "output_file": row["output_file"],
            },
        }
```

Notes for the reward sandbox:
- Generated tests can be large (1M+ cases overall); cap per-problem tests (e.g. first
  N) and enforce the per-problem `time_limit`/`memory_limit_bytes`.
- Prefer problems where `solutions` contains a PYTHON3 reference solution if you want a
  sanity check that the sandbox itself is correct.
- Public tests double as few-shot examples in the prompt; keep private/generated tests
  reward-only to avoid leakage.

## License / Terms of Use

Upstream HF card license: **CC BY 4.0**
([Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode)).
The upstream card additionally acknowledges: Codeforces materials are sourced from
http://codeforces.com; Description2Code materials are MIT-licensed. The companion
GitHub repo ([deepmind/code_contests](https://github.com/deepmind/code_contests/))
is Apache-2.0 for its code. Note the task brief expected Apache-2.0 — the dataset card
itself declares CC BY 4.0, which is what applies to the data.

## Sources

- HF dataset: https://huggingface.co/datasets/deepmind/code_contests
- GitHub: https://github.com/deepmind/code_contests/
- Paper: [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814)

## Citation

```bibtex
@article{li2022competition,
  title={Competition-Level Code Generation with AlphaCode},
  author={Li, Yujia and Choi, David and Chung, Junyoung and Kushman, Nate and
    Schrittwieser, Julian and Leblond, R{\'e}mi and Eccles, Tom and
    Keeling, James and Gimeno, Felix and Dal Lago, Agustin and
    Hubert, Thomas and Choy, Peter and de Masson d'Autume, Cyprien and
    Babuschkin, Igor and Chen, Xinyun and Huang, Po-Sen and Welbl, Johannes and
    Gowal, Sven and Cherepanov, Alexey and Molloy, James and
    Mankowitz, Daniel and Sutherland Robson, Esme and Kohli, Pushmeet and
    de Freitas, Nando and Kavukcuoglu, Koray and Vinyals, Oriol},
  journal={arXiv preprint arXiv:2203.07814},
  year={2022}
}
```

## Cleaned derivative (Instinct)

`code_contests.clean.jsonl` (12,559 rows, 1.6 GB) — produced by `dataset/scripts/clean_code_contests.py`, full stats in `code_contests.clean.report.json`, dropped rows in `code_contests.excluded.jsonl`.

- Per-row schema: `problem_id/name/source/difficulty/cf_rating/cf_tags`, `description` (HTML→text, `<img>` rows excluded), `description_is_translated`+`untranslated_description` (1,088 machine-translated), `time_limit_s/memory_limit_mb/file_io` (22 file-IO rows flagged), `tests{public,private,generated}` (parallel `input[]`/`output[]` lists, zip pairs), `n_tests_*`, `solutions_python3`/`solutions_cpp` (≤8 each, exact-deduped), `n_incorrect_solutions`.
- Cleaning: 13,610 → 12,559 kept; excluded 1,038 duplicate statements (multi-source aggregation) + 13 test-less. Language enum verified by syntax sampling: 1=python2, 2=cpp, 3=python3, 4=java; py2/java solutions dropped.
- No chain-of-thought: solutions are raw submissions (prose-like rows 57/13,610 are comments). `incorrect_solutions` (8.7M) stay in the raw parquet, counted only.
- GRPO recipe: prompt = `description` (+public tests as few-shot), reward set = zip(private_tests.input/output) + zip(generated_tests...); keep hidden tests out of the prompt; normalize trailing whitespace before diffing; `file_io` rows need a file-based sandbox.
