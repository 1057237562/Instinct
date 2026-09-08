---
dataset_info:
- config_name: default
  features:
  - name: submission_id
    dtype: string
  - name: source
    dtype: string
  - name: contestId
    dtype: string
  - name: problem_index
    dtype: string
  - name: programmingLanguage
    dtype: string
  - name: verdict
    dtype: string
  - name: testset
    dtype: string
  - name: passedTestCount
    dtype: float64
  - name: timeConsumedMillis
    dtype: float64
  - name: memoryConsumedBytes
    dtype: float64
  - name: creationTimeSeconds
    dtype: float64
  - name: problem_id
    dtype: string
  splits:
  - name: train
    num_bytes: 17766177945
    num_examples: 12591518
  download_size: 5554509577
  dataset_size: 17766177945
- config_name: selected_accepted
  features:
  - name: submission_id
    dtype: string
  - name: source
    dtype: string
  - name: contestId
    dtype: string
  - name: problem_index
    dtype: string
  - name: programmingLanguage
    dtype: string
  - name: verdict
    dtype: string
  - name: testset
    dtype: string
  - name: passedTestCount
    dtype: float64
  - name: timeConsumedMillis
    dtype: float64
  - name: memoryConsumedBytes
    dtype: float64
  - name: creationTimeSeconds
    dtype: float64
  - name: problem_id
    dtype: string
  - name: og_source
    dtype: string
  splits:
  - name: train
    num_bytes: 179003290
    num_examples: 42467
  download_size: 73893557
  dataset_size: 179003290
- config_name: selected_incorrect
  features:
  - name: submission_id
    dtype: string
  - name: source
    dtype: string
  - name: contestId
    dtype: string
  - name: problem_index
    dtype: string
  - name: programmingLanguage
    dtype: string
  - name: verdict
    dtype: string
  - name: testset
    dtype: string
  - name: passedTestCount
    dtype: float64
  - name: timeConsumedMillis
    dtype: float64
  - name: memoryConsumedBytes
    dtype: float64
  - name: creationTimeSeconds
    dtype: float64
  - name: problem_id
    dtype: string
  splits:
  - name: train
    num_bytes: 20128450
    num_examples: 18097
  download_size: 8181241
  dataset_size: 20128450
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
- config_name: selected_accepted
  data_files:
  - split: train
    path: selected_accepted/train-*
- config_name: selected_incorrect
  data_files:
  - split: train
    path: selected_incorrect/train-*
license: cc-by-4.0
pretty_name: CodeForces Submissions
size_categories:
- 10M<n<100M
---
# Dataset Card for CodeForces-Submissions

## Dataset description
[CodeForces](https://codeforces.com/) is one of the most popular websites among competitive programmers, hosting regular contests where participants must solve challenging algorithmic optimization problems. The challenging nature of these problems makes them an interesting dataset to improve and test models’ code reasoning capabilities.

This dataset includes millions of real user (human) code submissions to the CodeForces website.

## Subsets
Different subsets are available:
- `default`: all available submissions (covers 9906 problems)
- `selected_accepted`: a subset of `default`, with submissions that we were able to execute and that passed all the public tests in [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces) (covers 9183 problems)
- `selected_incorrect`: a subset of `default`, with submissions that while not having an Accepted verdict, passed at least one of the public test cases from [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces). We selected up to 10 per problem, choosing the submissions that passed the most tests (covers 2385 problems)


## Data fields
- `submission_id` (str): unique submission ID
- `source` (str): source code for this submission. Potentionally modified to run on newer C++/Python versions.
- `contestId` (str): the ID of the contest this solution belongs to
- `problem_index` (str): Usually, a letter or letter with digit(s) indicating the problem index in a contest
- `problem_id` (str): in the format of `contestId/problem_index`. We account for duplicated problems/aliases, this will match a problem in `open-r1/codeforces`
- `programmingLanguage` (str): one of `Ada`, `Befunge`, `C# 8`, `C++14 (GCC 6-32)`, `C++17 (GCC 7-32)`, `C++17 (GCC 9-64)`, `C++20 (GCC 11-64)`, `C++20 (GCC 13-64)`, `Clang++17 Diagnostics`, `Clang++20 Diagnostics`, `Cobol`, `D`, `Delphi`, `F#`, `FALSE`, `FPC`, `Factor`, `GNU C`, `GNU C++`, `GNU C++0x`, `GNU C++11`, `GNU C++17 Diagnostics`, `GNU C11`, `Go`, `Haskell`, `Io`, `J`, `Java 11`, `Java 21`, `Java 6`, `Java 7`, `Java 8`, `JavaScript`, `Kotlin 1.4`, `Kotlin 1.5`, `Kotlin 1.6`, `Kotlin 1.7`, `Kotlin 1.9`, `MS C#`, `MS C++`, `MS C++ 2017`, `Mono C#`, `Mysterious Language`, `Node.js`, `OCaml`, `PHP`, `PascalABC.NET`, `Perl`, `Picat`, `Pike`, `PyPy 2`, `PyPy 3`, `PyPy 3-64`, `Python 2`, `Python 3`, `Python 3 + libs`, `Q#`, `Roco`, `Ruby`, `Ruby 3`, `Rust`, `Rust 2021`, `Scala`, `Secret 2021`, `Secret_171`, `Tcl`, `Text`, `Unknown`, `UnknownX`, `null`
- `verdict` (str): one of `CHALLENGED`, `COMPILATION_ERROR`, `CRASHED`, `FAILED`, `IDLENESS_LIMIT_EXCEEDED`, `MEMORY_LIMIT_EXCEEDED`, `OK`, `PARTIAL`, `REJECTED`, `RUNTIME_ERROR`, `SKIPPED`, `TESTING`, `TIME_LIMIT_EXCEEDED`, `WRONG_ANSWER`
- `testset` (str): Testset used for judging the submission. Can be one of `CHALLENGES`, `PRETESTS`, `TESTS`, or `TESTSxx`
- `passedTestCount` (int): the number of test cases this submission passed when it was submitted
- `timeConsumedMillis` (int): Maximum time in milliseconds, consumed by solution for one test.
- `memoryConsumedBytes` (int): Maximum memory in bytes, consumed by solution for one test.
- `creationTimeSeconds` (int): Time, when submission was created, in unix-format.
- `original_code` (str): can differ from `source` if it was modified to run on newer C++/Python versions.

## Data sources
We compiled real user (human) submissions to the CodeForces website from multiple sources:

- [`agrigorev/codeforces-code` (kaggle)](https://www.kaggle.com/datasets/agrigorev/codeforces-code)
- [`yeoyunsianggeremie/codeforces-code-dataset` (kaggle)](https://www.kaggle.com/datasets/yeoyunsianggeremie/codeforces-code-dataset/data)
- [`MatrixStudio/Codeforces-Python-Submissions`](https://hf.co/datasets/MatrixStudio/Codeforces-Python-Submissions)
- [`Jur1cek/codeforces-dataset` (github)](https://github.com/Jur1cek/codeforces-dataset/tree/main)
- [miningprogcodeforces](https://sites.google.com/site/miningprogcodeforces/home/dataset?authuser=0)
- [itshared.org](https://www.itshared.org/2015/12/codeforces-submissions-dataset-for.html)
- [`ethancaballero/description2code` (github)](https://github.com/ethancaballero/description2code)
- our own crawling (mostly for recent problems)

## Using the dataset
You can load the dataset as follows:

```python
from datasets import load_dataset

ds = load_dataset("open-r1/codeforces-submissions", split="train")
OR
ds = load_dataset("open-r1/codeforces-submissions", split="train", name='selected_accepted')
```
See other CodeForces related datasets in [this collection](https://huggingface.co/collections/open-r1/codeforces-68234ed24aa9d65720663bd2).

## License
The dataset is licensed under the Open Data Commons Attribution License (ODC-By) 4.0 license.

## Citation

If you find CodeForces useful in your work, please consider citing it as:

```
@misc{penedo2025codeforces,
      title={CodeForces}, 
      author={Guilherme Penedo and Anton Lozhkov and Hynek Kydlíček and Loubna Ben Allal and Edward Beeching and Agustín Piqueres Lajarín and Quentin Gallouédec and Nathan Habib and Lewis Tunstall and Leandro von Werra},
      year={2025},
      publisher = {Hugging Face},
      journal = {Hugging Face repository},
      howpublished = {\url{https://huggingface.co/datasets/open-r1/codeforces}}
}
```