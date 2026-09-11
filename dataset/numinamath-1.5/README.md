---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- math
- post-training
pretty_name: NuminaMath 1.5
---

# Dataset Card for NuminaMath 1.5

## Dataset Description

- **Homepage:** https://projectnumina.ai
- **Repository:** 
- **Paper:** https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf
- **Leaderboard:** 
- **Point of Contact:** [Jia Li](jia@projectnumina.ai)


### Dataset Summary

This is the second iteration of the popular [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset, bringing high quality post-training data for approximately 900k competition-level math problems.  Each solution is formatted in a Chain of Thought (CoT) manner. The sources of the dataset range from Chinese high school math exercises to US and international mathematics olympiad competition problems. The data were primarily collected from online exam paper PDFs and mathematics discussion forums.

### What's new?

#### Problem metadata

After understanding the importance of verifiable output for each problem, we have added `answer`, `problem_type`, `question_type` metadata for all problems:

- `answer`: Final answer of the problem when `question_type` is a "math word problem", i.e. a number-valued output. For problems which do not belong to this category, `answer` takes one of the following special values:
    - `proof`: When the `question_type` is proof
    - `notfound`: When we cannot find the answer from the `ref_solution`
- `problem_type`: The mathematical domain of the problem. See `find_problem_type` for more information. Here are the supported types:
    - Algebra
    - Geometry
    - Number Theory
    - Combinatorics
    - Calculus
    - Inequalities
    - Logic and Puzzles
    - Other
- `question_type`: The form or style of the mathematical problem.
    - multiple-choice question (MCQ)
    - proof
    - math-word-problem (problem with output)

#### Some new data (more to come)

- Olympiads Reference (source: olympiads ref). After the publication of the first [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset, we realized that there are a lot of parsing issues with the `olympiads` subset, due to the use of generic regular experessions and LLMs. To fix this, we have used the official websites from dozens of national Math Olympiads to perform manual parsing and verification of the problems and solutions.
- More manual curated data. `cn_contest`, `inequalities` and `number_theory` are manually curated competition problems provided by our data partners.
- Removal of synthetic dataset `synthetic_amc`. In our ablation study, this hurt a bit the performance. In the futhur we planned to remove all synthetic data until we find a way to reliably generate high-quality synthetic problems.


### Source breakdown

| source         |   problems |   question_type:proof |   question_type:mcq |   question_type:word |
|:---------------|-----------:|----------------------:|--------------------:|---------------------:|
| olympiads      |     197084 |                 62970 |               13529 |               117845 |
| olympiads_ref  |       3638 |                  2246 |                 nan |                 1392 |
| amc_aime       |       5872 |                   208 |                4374 |                  963 |
| aops_forum     |      67841 |                 24532 |                5924 |                33486 |
| cn_contest     |      29944 |                  8663 |                5602 |                15649 |
| inequalities   |       7314 |                  5780 |                  49 |                 1478 |
| number_theory  |       4043 |                  2591 |                  15 |                 1239 |
| cn_k12         |     268819 |                  3966 |              115800 |               149010 |
| orca_math      |     151934 |                     1 |                  17 |               151916 |
| synthetic_math |     148712 |                    41 |                1057 |               147612 |
| metamath       |      11014 |                   nan |                  82 |                10932 |
| Total          |     896215 |                110998 |              146449 |               631522 |

### Licensing Information

The dataset is available under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

### Citation Information

```
@misc{numina_math_datasets,
  author = {Jia LI and Edward Beeching and Lewis Tunstall and Ben Lipkin and Roman Soletskyi and Shengyi Costa Huang and Kashif Rasul and Longhui Yu and Albert Jiang and Ziju Shen and Zihan Qin and Bin Dong and Li Zhou and Yann Fleureau and Guillaume Lample and Stanislas Polu},
  title = {NuminaMath},
  year = {2024},
  publisher = {Numina},
  journal = {Hugging Face repository},
  howpublished = {\url{[https://huggingface.co/datasets/AI-MO/NuminaMath-1.5](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)}}
}
```