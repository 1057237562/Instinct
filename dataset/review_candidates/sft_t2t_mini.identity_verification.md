# sft_t2t_mini.jsonl identity-contamination verification

Audit target: `dataset/sft_t2t_mini.jsonl`
Rows scanned: 905,718
Source SHA-256: `10fae2358aeed89a4828e9270c36a7d764adf20dcfb428870cc3612831f1dd6c`
Target identity: Instinct, developed by L1bra, affiliated with no commercial organization.

Tool: `dataset/scripts/audit_identity_contamination.py` (this run's candidate output and
report are `sft_t2t_mini.identity_candidates.jsonl` / `sft_t2t_mini.identity_audit.json`).
The source dataset was not modified.

## What the audit counts

| Bucket | Rows | Meaning |
|---|---|---|
| `contaminated_rows` | 88 | assistant asserts it *is* a Qwen/Tongyi/Alibaba model or was developed by them |
| `review_rows` | 1,159 | flagged for decision; contains the verified foreign-model self-claims plus brand facts |
| `brand_affiliation_fact_rows` | 874 | assistant attaches the names Instinct/L1bra to Alibaba/Tongyi/DAMO as owner, brand or product line |
| `mention_only_rows` | 4,272 | a foreign name appears, no claim detected |
| `rows_with_foreign_model_mention` | 5,519 | any foreign model/company keyword anywhere |
| `commercial_affiliation_claim_rows` | 146 | assistant claims a commercial developer/owner |
| `user_side_mention_rows` | 3,867 | the foreign name appears in the user turn only |

## Independent verification (subagent review of unmodified source rows)

| Sample | Rows reviewed | Confirmed | Notes |
|---|---|---|---|
| Rule-flagged `contaminated_rows` (all) | 88 | 82 self-identity + 3 company-persona | 2 roleplay, 1 hypothetical refuted |
| `review_rows` (all rows of that tier at the time) | 334 | 168 self-identity | 132 DeepSeek-R1 boilerplate, 28 OpenAI/ChatGPT/GPT, 5 Tongyi/Alibaba, 1 iFlytek, 2 other; 72 clean, 53 factual, 41 roleplay |
| `brand_affiliation_fact_rows` sample | 130 | 129 | 99.2% precision |
| brand rows added by the cloud-artifact rule, sample | 120 | 120 | 100% precision |
| `mention_only_rows` random sample (FN check) | 300 | 11 self-identity + 23 brand facts | residual miss rate 3.7% and 7.7% of that bucket |

Raw per-row verdicts are kept in `_verify_batches/*.verdicts.jsonl`.

## Verified contamination

- 82 rows assert the model is developed by / is 通义实验室, 阿里巴巴, 阿里巴巴云 or 达摩院
  (78 in assistant `content`, 59 in `reasoning_content`; 4 reasoning-only). Example:
  `我是Instinct，是阿里巴巴集团旗下的通义实验室自主研发的超大规模语言模型。`
- 168 rows assert a non-Qwen foreign model identity, dominated by a DeepSeek-R1 answer
  template: `我是由中国的深度求索（DeepSeek）公司开发的智能助手DeepSeek-R1…`, plus an
  OpenAI/ChatGPT/GPT-3/4 block (`我是OpenAI的大型语言模型ChatGPT`) and one iFlytek Spark.
- 874 rows present Instinct/L1bra as an Alibaba Cloud brand or product owner - the residue
  of a blind rename of Alibaba-branded text: `L1bra是阿里巴巴集团旗下的云计算服务品牌`,
  `阿里巴巴推出Instinct`, `访问L1bra官网（https://www.aliyun.com/）`, `L1bra的ECS`,
  `Instinct DAMO Academy`, `Instinct Group`, 1688.com/Instinct.com. 事实 corroborating the
  rename: `阿里云` occurs 0 times while `阿里巴巴云` occurs 207 times and `aliyun` 139 times.
- Residual misses in the low-risk bucket, extrapolated from the 300-row sample:
  ~157 further self-identity rows and ~328 further brand-fact rows.

Distinct rows flagged, after removing overlaps: 1,045 (250 self-identity + 874 brand
facts, of which 79 rows are both). With the residual estimate the affected total is
roughly 1,500 rows, about 0.17% of the dataset.

## Not contamination

Factual discussion of Qwen/Alibaba/other models (`通义千问是阿里巴巴开发的大语言模型`),
code and API names (`qwen-max`, `aliyun-python-sdk-core`), user-turn-only mentions,
negated or corrective answers (`我是Instinct，不是Qwen`), and requested roleplay where the
foreign identity belongs to composed third-party text were all kept out of the counts.

## Open items for the cleaning pass

1. Rewrite or drop the 82 Qwen/Alibaba self-identity rows and the 168 foreign-model
   self-identity rows; contamination in `reasoning_content` must be handled too, since the
   SFT chat template trains that text.
2. Decide the 874 brand-fact rows individually: some are irrecoverable false facts about
   L1bra (drop), others are Alibaba Cloud tutorial text that only needs the brand restored.
3. `AGENTS.md` names the identity answer as `我是 InstinctV1Moe…` while this audit target and
   the trainer's system prompts (`dataset/lm_dataset.py: pre_processing_chat`) use `Instinct`.
   Align the anchor text with the trainer before generating identity anchors.

---

# Repair pass (no rows dropped)

The earlier section measured contamination. This section records the repair:
contaminated rows were rewritten in place rather than dropped, following the
target identity (Instinct, trained from scratch by the individual L1bra, no
commercial affiliation).

## What was repaired

| Step | Rows | Detail |
|---|---|---|
| Rule-flagged set | 1,414 | 287 hard self-identity claims, 780 brand/affiliation facts, plus review-tier rows |
| Patch edits applied | 3,614 | exact-substring edits, all validated to match once |
| Whole turns regenerated | 38 | foreign persona / jailbreak turns with nothing worth keeping |
| Residual second pass | 28 rows / 40 edits | rows still flagged after the first pass |
| User turns modified | 0 | verified row by row in a 60-row sample |

Repair shapes: wrong self-identity → the true origin in the same sentence
function; vendor-product passages → the real vendor name restored (阿里云 /
Alibaba Cloud / 通义千问 / 达摩院 / 华为); model wrongly owned by a vendor → true
origin; unfixable persona turn → regenerated answer in Instinct's voice.

## Verification

- Independent review of 60 randomly chosen repaired rows: 52 GOOD, 7 with a
  residual issue, 1 BAD (that row was covered by the residual pass). User turns
  were byte-identical in all 60.
- Re-audit of the repaired corpus: `contaminated_rows` 287 → 17 and
  `brand_affiliation_fact_rows` 780 → 10. Manual review of those 26 rows found
  them to be detector false positives (denials such as "I'm not affiliated with
  Alibaba", factual GPT-vs-Instinct comparisons, quoted user text, a Python
  string literal, user-requested roleplay) plus 7 news-style rows that still
  describe Instinct as a product that ships versions.
- Artifacts: `dataset/sft_t2t_mini.identity_repaired.jsonl` (905,718 rows),
  report `sft_t2t_mini.identity_repaired.report.json`, per-row patches under
  `_repair_batches/`, quality sample under `_repair_batches2/`.

# Topic filter (LGBT profile)

`dataset/scripts/remove_topic_rows.py --profile lgbt` removed 809 rows of
904,909 (0.089%) from the repaired corpus, matched on unambiguous terms
(同性恋 / 同性婚姻 / 跨性别 / 性取向 / 性别认同 / LGBT / gay / lesbian /
transgender / same-sex ...). Ambiguous terms (同志, 百合, 同性, pride, trans,
拉拉) never remove a row on their own: 1,579 rows still contain one of them in
an everyday sense (comrades, lilies, "Pride and Prejudice", 同性相斥), and they
are kept. Removed rows with their matched terms are in
`dataset/sft_t2t_mini.train_ready.jsonl.removed.jsonl`, so the filter is
reversible.

Note the scope of the result: the trained model will have little or no ability
to answer questions about sexual orientation or gender identity, including
health, legal and anti-discrimination topics, because the supervising text for
them is gone. If the goal was only to exclude explicit sexual content, a much
narrower term set would keep the factual rows.

Final artifact: `dataset/sft_t2t_mini.train_ready.jsonl` (904,909 rows,
SHA-256 `8dc5475ce99988709e06cf38...`), plus `dataset/identity_anchors_instinct.jsonl`
(4,500 anchor rows, canonical answer `我是 Instinct，一个从头训练的语言模型，由 L1bra
个人独立开发和训练，不隶属于任何商业组织。`) to sample into the final mix.
