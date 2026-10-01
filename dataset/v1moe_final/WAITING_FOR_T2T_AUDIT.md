# T2T handoff received; waiting for remaining review results

2026-09-28 update: user supplied `sft_t2t_mini.train_ready.jsonl` (904,909 rows), verified SHA-256 `8dc5475ce99988709e06cf38a267f9d8f2135bf438d202d9898e5cbf090d85aa`, and `identity_anchors_instinct.jsonl` (4,500 rows), verified SHA-256 `4dc02a9e5d9e0bb5c23e82456ea72c513f46adc70e73164b98e0b033b39c32d3`.

The same-hash T2T audit still lists 17 suspected contamination and 112 review rows, including potential false positives. The anchor generator has 225 unique QA pairs cross-producted across questions/answers; some pairs fail to answer their question. No source files were modified.

User offered to dispatch a lighter model for remaining audit work. Handoff package: `dataset/review_candidates/final_sft_handoff_20260928/README.md`, 677 review items in 116 batches. P0 covers all 129 residuals and 225 anchor pairs. Other categories are exploratory samples / non-T2T identity candidates. Results belong in that package's `results/`; validate with `dataset/scripts/validate_final_sft_review_results.py`.

Use the user's newly approved outward identity **Instinct**, developed/trained by L1bra independently, no commercial affiliation. Architecture name remains InstinctV1Moe. Do not blindly rewrite the supplied anchor wording. Unify the eventual system prompt with the approved identity after reviews.

The final builder now refuses to use the stale core manifest while train_ready exists. Rebuild core from reviewed new source, remove indirect old-T2T replay, incorporate reviewed anchors, rerun first-pass triage on updated candidate hashes, then finalize and verify. Preserve the completed non-T2T candidates.

---

Earlier pause record:

User instruction: someone else is auditing `sft_t2t_mini.jsonl` and adding identity information; wait for their result before finalizing this mixture.

Final build was stopped. No final training file has been delivered or training started.

Preserved work:
- Fixed-revision OpenCodeInstruct and ChatQA2 downloads in `dataset/v1moe_final_sources/`.
- `code.jsonl`: 172,439 candidates selected from 1,000,000 source records using upstream pass flags, nonempty matching tests, Python syntax and judge scores.
- `long.jsonl`: 14,089 long-document candidates.
- `synthetic.jsonl`: 2,000 coherent six-turn structured-state tasks; four independently checkable families.
- Per-component source manifests and hashes.

`candidates/core.jsonl` is provisional: it includes OLD T2T content, both directly and through the prior mixed pilot. Do not use it unchanged after the other audit completes. Rebuild it from the explicitly approved cleaned T2T artifact and incorporate the reviewed identity additions, avoiding duplicate/conflicting identity anchors. Also trace T2T replay nested inside the old `python_reviewed` mix.

`triage.jsonl` pins the old candidate hashes and must be regenerated if candidates change. `pending_review.jsonl` is an interrupted intermediate, not a completed audit or training artifact. The frozen `audit_identity_snapshot.py` was captured only to avoid concurrent file edits during scanning; prefer the parallel audit's completed, verified tooling and results after handoff.

Needed handoff evidence: completed cleaned artifact path, source/output SHA-256, audit report and reviewed identity-anchor file or policy. A file timestamp change alone does not mean the audit is complete.

User preferences remain: one mixed training file, no length shards; include short instructions, multi-turn dialogue, Python/HumanEval-oriented data, correct InstinctV1Moe identity, and full conversations up to a 65,536-token budget. Preserve original sources and disclose upstream versus local verification. Do not promise a HumanEval score or 64K model capability without training/evaluation.
