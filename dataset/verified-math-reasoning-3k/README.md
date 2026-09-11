---
license: apache-2.0
language:
  - en
tags:
  - instruction-finetuning
  - reasoning
  - chain-of-thought
  - mathematics
  - synthetic
  - gradients-ready
  - bittensor
  - verified
size_categories:
  - 1K<n<10K
task_categories:
  - text-generation
  - question-answering
task_ids:
  - open-domain-qa
pretty_name: "HSH Verified Math Reasoning — Fine-Tuning Ready"
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
      - split: test
        path: data/test.jsonl
---

# HSH Verified Math Reasoning — Fine-Tuning Ready

A clean, **answer-verified** dataset of step-by-step math word problems with chain-of-thought reasoning, formatted for instruction fine-tuning. This is **foundational reasoning data designed for first fine-tunes** — single-concept arithmetic word problems with fully verified answers, ideal for a reliable, clean starter run. Every single answer in this dataset has been **programmatically verified** against a ground-truth value computed in Python — not by an LLM. If a generated solution did not arrive at the correct answer, it was discarded.

Published by [HSH Intelligence](https://huggingface.co/HSH-Intelligence).

## Why this dataset is different

Most synthetic math datasets trust the language model to be correct. This one does not. Each problem's correct answer is computed independently in pure Python *before* the model writes its reasoning. The model's stated final answer is then checked against that ground truth, and **only answer-correct rows survive**. This makes the dataset trustworthy by construction: you can spot-check any row and the final answer will be correct.

## Who this is for

This dataset is intended as a **clean, reliable foundation for first fine-tuning runs** and for learning instruction-tuning workflows. Every answer is verified correct, so you can trust it row-for-row. The problems are single-concept arithmetic word problems (one formula per problem) — excellent for a dependable starter dataset. For advanced multi-step or competition-grade reasoning, see our forthcoming v2 (harder, multi-step, multi-concept problems).

## Use Cases

- Fine-tuning small/mid LLMs for step-by-step mathematical reasoning
- Chain-of-thought (CoT) instruction tuning
- Ready to use directly on [Gradients](https://gradients.io) (Bittensor SN56) and any framework that consumes the Alpaca `instruction`/`input`/`output` schema (TRL, Axolotl, Unsloth)

## Dataset Structure

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | string | The math word problem, including the request to show reasoning and round to 2 decimals where needed |
| `input` | string | Empty (single-turn problems; field kept for schema compatibility) |
| `output` | string | Step-by-step reasoning ending with a final line: `The answer is X` |

### Example row

```json
{
  "instruction": "An amount of $150 is split between two people in the ratio 1:3. How much does the first person receive in dollars? If the answer is not a whole number, round it to 2 decimal places. Show your reasoning step by step.",
  "input": "",
  "output": "To find out how much the first person receives, we first determine the total parts in the ratio... $150 / 4 = $37.50 per part... 1 * $37.50 = $37.50.\nThe answer is 37.50"
}
```

## Dataset Statistics

- **Total examples:** 3,000
- **Train / Validation / Test:** 2,400 / 300 / 300 (80/10/10)
- **Mean instruction length:** 180.7 chars
- **Mean output length:** 434.0 chars (min 161, max 1,033)
- **Unique instructions:** 3,000 (fully deduplicated — zero duplicates)
- **All answers verified:** ✅ Yes (100% — every final answer programmatically checked against Python ground truth)

### Problem-type distribution

| Problem type | Count |
|---|---|
| Average speed | 428 |
| Remaining quantity | 421 |
| Discount / sale price | 359 |
| Percentage of a number | 349 |
| Rectangle area | 343 |
| Unit rate / cost per item | 335 |
| Age in N years | 293 |
| Simple interest | 202 |
| Total work / production | 177 |
| Ratio split | 93 |

## How it was made (Provenance)

This dataset is **100% synthetic** and openly disclosed as such.

**Generation method:**
1. **Problem construction (Python):** Each math problem is generated from one of 10 parameterized templates (average speed, percentages, discounts, unit rates, age, work rate, remaining-quantity, ratio splits, rectangle area, simple interest). The numbers are randomized and **the correct answer is computed in pure Python** — this is the ground truth.
2. **Reasoning generation (LLM):** The problem text is sent to **Llama-4-Scout** (`meta-llama/llama-4-scout-17b-16e-instruct`) via the Groq API, which writes the step-by-step reasoning and a final answer line. The LLM writes prose only — it does not define the ground truth.
3. **Verification (Python):** The model's stated final answer is extracted and compared against the Python ground truth. **Mismatches are discarded.** Only answer-correct rows are kept.
4. **Deduplication:** Exact-duplicate problems are removed.
5. **Splitting:** Shuffled (seed 42) and split 80/10/10 into train/validation/test.

**No human data, no scraped data, no PII.** All content is machine-generated arithmetic word problems. There is no personal, copyrighted, or sensitive content.

## License

**Apache-2.0.** Commercial use permitted. See the LICENSE file. As a fully synthetic dataset generated by HSH Intelligence, we hold the rights to release it under this permissive license.

### A note on synthetic data
Synthetic reasoning data can reflect limitations of the generating model's writing style. We mitigate the most important risk — *wrong answers* — through programmatic verification. We make no claim that the *reasoning prose* is pedagogically optimal, only that every final answer is correct.

## Roadmap

- **v1.1 (within 7 days):** Benchmark proof — we will fine-tune a small base model (e.g. Llama-3.1-8B) on this dataset and publish GSM8K / accuracy results here, so you can see "models trained on this" evidence before spending your own compute.
- **v2:** Expanded size, more problem types (algebra, geometry, multi-step word problems), difficulty tags.

## Using this dataset with Gradients

```bash
curl --request POST \
  --url https://api.gradients.io/v1/tasks/create \
  --header 'Authorization: YOUR_TOKEN' \
  --header 'Content-Type: application/json' \
  --data '{
    "model_repo": "meta-llama/Llama-3.1-8B",
    "ds_repo": "HSH-Intelligence/verified-math-reasoning-3k",
    "instruction_col": "instruction",
    "hours_to_complete": 1
  }'
```

> **Column mapping note:** The dataset uses the standard Alpaca schema (`instruction`, `input`, `output`). Gradients' `tasks/create` takes `instruction_col`; you can confirm/auto-suggest column mappings via the `content.gradients.io/dataset/{your-repo}/columns/suggest` endpoint before launching a job.

## Citation

```bibtex
@dataset{hsh_verified_math_reasoning_2026,
  author = {HSH Intelligence},
  title  = {HSH Verified Math Reasoning — Fine-Tuning Ready},
  year   = {2026},
  url    = {https://huggingface.co/datasets/HSH-Intelligence/verified-math-reasoning-3k}
}
```

## About HSH Intelligence

HSH Intelligence builds **made-to-order, verified data for AI training** — delivered fast, on demand. Need a custom dataset for your fine-tuning job that doesn't exist yet? That's what we do. Contact: data@hshintelligence.com
