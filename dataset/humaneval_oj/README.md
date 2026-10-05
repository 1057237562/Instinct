# HumanEval -> OJ-style conversion (plain-text stdin/stdout)

Built by `scripts/data_builder/build_humaneval_oj.py` from the local official
snapshot `dataset/humaneval/HumanEval.jsonl.gz` (MIT, OpenAI). Audit numbers:
`build.report.json` (per-task detail, per-rule counts, identity scan, SHA-256).

## Protocol (fixed for the whole set, classic OJ style - never JSON)

- stdin: plain text, one section per argument in signature order.
  scalars on their own line (ints, floats, strings; booleans as 1/0);
  lists/matrices are N-prefixed: a line with N, then N space-separated
  numbers (lists of strings: N lines, one string each).
- stdout: plain text rendered from the return value (bool as True/False,
  number lists space-separated, string lists one per line, matrices one row
  per line, None prints nothing), compared byte for byte.

## How cases are produced

1. Asserts in each task's `check()` are parsed with AST and constant-folded
   (`candidate(args) == expr`, `expr == candidate(args)`,
   `abs(candidate(...) - expr) < eps`).
2. An I/O plan is derived from the signature's type annotations with a
   value-based fallback; the reference program (prompt + canonical_solution +
   plan-derived parsing/printing) EXECUTES on every case. Numeric parsing
   preserves the literal's type so `isinstance(x, int)`-style canonicals
   still work. Cases whose values contain None (no natural plain-text
   encoding) are dropped and counted.
3. The stored expected output is the reference program's actual stdout,
   cross-checked against the assert literal (exact `==` for exact asserts,
   isclose for abs()-style asserts). Only cross-verified pairs are kept;
   tasks with zero verified pairs are skipped and listed in the report.

`humaneval_oj_sft.jsonl` renders each problem as a two-turn conversation whose
assistant target is the executed reference program.

## Contamination warning (read before training on this)

This project's SFT recipes explicitly keep HumanEval as a held-out exclusion
basis (`dataset/v1moe_sft64k/README.md`: HumanEval is only used for
decontamination, no HumanEval-derived samples). Training on
`humaneval_oj_sft.jsonl` burns HumanEval as a functional benchmark. Keep this
directory out of SFT/RL training mixes unless that trade-off is explicitly
accepted; the same converter pattern can be applied to non-eval corpora
(magicoder, taco, ...) instead.
