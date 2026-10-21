# HumanEval-OJ: complete programs with plain-text stdin/stdout

`eval_humaneval_oj.py` automatically builds the default bundle when missing and
rebuilds legacy incomplete bundles. The Eval WebUI defaults to this benchmark,
with thinking enabled and a 4096-token generation budget. Manual preparation:

```bash
python scripts/data_builder/build_humaneval_oj.py
```

The builder reads or downloads the official OpenAI MIT-licensed snapshot at
`dataset/humaneval/HumanEval.jsonl.gz`. It preserves **all 164 tasks** and executes
each canonical solution against its complete original `check(candidate)` suite.
It traces calls to produce I/O examples and plans, then runs the resulting
standalone reference program through the original checks again. Conversion
failure stops publication; no task or unsupported case is silently dropped.

Artifacts:

- `humaneval_oj_problems.jsonl`: statement, I/O plan, test cases, original task,
  and standalone reference program; only the statement is shown to the model.
- `build.report.json`: source repository/URL and content hash, input/output row
  counts, dropped/rewritten/filter counts, identity hits, per-task verification,
  and output SHA-256. The source snapshot is retained unchanged.

I/O is plain text rather than JSON. Scalars occupy one line. Numeric input
arrays use a length line followed by space-separated elements; numeric output
arrays occupy one line. String lists and dictionaries carry counts. Tuples use
ordered sections; multiline strings carry a line count. Values of mixed types
use explicit type tags followed by plain-text sections. None is preserved.
Each statement specifies its exact format and includes two examples.

V2 scoring runs the model program independently for each distinct input, parses
stdout according to the plan, and passes those values to the complete original
test suite. This preserves original equality checks, tolerances, randomized
loops (seed 42), dictionary order independence and multiple valid answers.
The timeout applies to each program invocation, with an additional whole-suite
time budget. Explicitly selected legacy V1 bundles retain their text comparison.

The model may reason before its final Python code block. Code inside `<think>`
is excluded, and an unfinished thinking response is not executed as a program.
Scores are labelled **HumanEval-OJ pass@K**; changing the prompt and answer
protocol means they are not official function-completion HumanEval scores.
An improvement must be measured with matched weights and sampling settings.

This is evaluation-only data. The builder does not generate SFT targets or add
HumanEval to a training mix. Training on these problems or reference programs
would contaminate the benchmark.
