# 数据集循环推理规律报告

日期：2026-11-22 · 数据：`instinctv2_sft_concise_reasoning_635m_v4` 及其原始 UltraData code-think 池
检测口径：整段 `reasoning` 的 **5-gram 唯一率 < 0.88**（且 ≥30 词）判为"重复"（`rebalance_code_think_lane.py` 的口径）；另统计行重复、模板 5-gram、自我纠错 ping-pong、数字枚举、重复代码块、字符周期等结构信号。数值是启发式检出率，不是人工标注。

## 结论摘要
1. 原始 code-think 池里 **37.3% 的行含重复推理**（90,000 行中 33,532 行），且**几乎全部集中在长推理**：>8,000 词的桶重复率 **66.2%**，≤4,096 词只有 0.8%–8.6%。
2. 重复行的平均长度是干净行的 **3.9 倍**（18,674 词 vs 4,839 词）——"越写越长"和"重复"高度绑定。
3. 重复不是单一坏样本，而是**六类可辨认的模式**：模板句复读、自我纠错反复、整行复制、数字枚举表、退化 token 循环、代码块复读。
4. v4 语料 616,173 行中只有 86,452 行带 `reasoning`；其中 **11,738 行（13.6%）重复**，集中在 instruction-think（17.2%）、math-think（14.0%）两个 lane。
5. v4 的 code-think lane 被重采样"压扁"到 2,048 token 附近（max 2,797 词）后，**重复率只有 3.3%**——但代价是丢掉了自然长度分布和长推理的闭合示范；评测时模型一旦尝试长推理就退化成循环（step 30,720 的 123 条撞上限中 **104 条是循环**，84.6%）。

## 一、原始池（9 个 UltraData Code_think 分片，90,000 行）

| 指标 | 数值 |
|---|---:|
| 总行数 | 90,000 |
| 重复行 | 33,532（**37.26%**） |
| 平均词数（全部 / 重复 / 干净） | 9,994 / **18,674** / 4,839 |

按长度桶（词）：

| 长度桶 | 行数 | 重复率 |
|---|---:|---:|
| 0–512 | 5,511 | 0.8% |
| 512–1,024 | 6,190 | 2.8% |
| 1,024–2,048 | 8,181 | 5.6% |
| 2,048–4,096 | 11,113 | 8.6% |
| 4,096–8,192 | 15,072 | 18.6% |
| **>8,192** | **43,933** | **66.2%** |

按来源：

| 来源 | 行数 | 重复率 |
|---|---:|---:|
| OpenCodeReasoning | 40,000 | 28.0% |
| OpenCodeReasoning-2 | 30,000 | **46.8%** |
| OpenCodeInstruct-Reasoning | 20,000 | 41.5% |

分布侧写：原始池 **65.6% 的行 >4,096 词**（本表 4,096 词以上合计），而 v4 lane 的 reasoning 最大只有 2,797 词 → v4 在选样时把长尾整体裁掉了。

## 二、六类重复模式（占重复行的比例，可多标签）

| 类型 | 命中行数 | 占重复行 | 说明与实例 |
|---|---:|---:|---|
| T1 模板句复读 | 32,702 | **97.5%** | 同一 5-gram 出现 ≥10 次；高频短语："we need to ensure that"（283 行）、"the number of ways to"（229）、"we can precompute for each"（199）、"we can think of the"（81） |
| T2 自我纠错 ping-pong | 32,502 | **96.9%** | "But wait / Let me check / Actually / Hmm / Let's verify / Maybe we can" 反复出现（部分行 >100 次）；典型片段："So we need C(N-1, M-k). So we need C(N-1, M-k). ..." |
| T3 整行复制 | 16,951 | 50.6% | 同一行（≥12 字符）出现 ≥5 次；如 `= 3240.5 - 1` 连续 20+ 次 |
| T4 数字枚举表 | 1,252 | 3.7% | 同一组 `k=… ceil(…)=…` 计算表反复抄写（42 次/行） |
| T5 退化 token 循环 | 1,972 | 5.9% | 高频 5-gram 的 5 个词完全相同，如 "x x x x x"（出现 51 次）、"t t t t t"（134）、"xor xor xor xor xor"（56）、"dp dp dp dp dp"（30） |
| T6 代码块复读 | 140 | 0.4% | reasoning 内相同 fenced code block 出现 ≥2 次 |

另有 353 行重复但未命中上述结构信号（纯措辞改写型重复）。

## 三、v4 语料内的分布（616,173 行）

带 `reasoning` 的行共 86,452；重复 11,738（13.58%）。

| lane（provenance category） | 行数 | 带 reasoning | 重复 | 重复率 | 平均词数 |
|---|---:|---:|---:|---:|---:|
| ultradata_instruction_think | 31,451 | 31,265 | 5,376 | **17.2%** | 548 |
| ultradata_math_think | 27,014 | 27,010 | 3,791 | **14.0%** | 517 |
| ultradata_chinese_think | 8,713 | 7,508 | 2,161 | 28.8%* | 61 |
| ultradata_code_think_short（v4 代码 lane） | 7,390 | 7,390 | 243 | **3.3%** | 887 |
| t2t_mini_full | 138,405 | 9,019 | 121 | 1.3% | 6 |
| identity / tools | 6,878 | 4,244 | 46 | 1.1% | — |
| 其余 no-think lane（code/instruction/knowledge/chinese） | 373,124 | 0 | 0 | — | — |

\* 中文 lane 的 5-gram 检测基于英文/标识符分词，对中文不可靠，此比例仅作参考。

## 四、为什么 v4 code lane 重复率不高，评测却大量循环？
1. v4 代码 lane 是**按 N(2048, 1024²) 重采样后的短推理**（平均 887 词、最大 2,797 词）。原始池里短推理本来就少重复（≤4k 词重复率 0.8%–8.6%），所以检测率自然低。
2. 但重采样**抹掉了自然长度分布**：原始池 65.6% 的行 >4,096 词、48.8% >8,192 词的"长推理—收束—闭合"示范被整体删除，模型只学到"代码题→约 2k token 思考"这一种形态。
3. 其他 think lane 的重复样本（instruction 17.2%、math 14.0%）仍会进入训练，混合语料层面继续传递"模板句复读 + 自我纠错"的写法。
4. 结果在评测中显形：step 30,720 的 think@4096 有 **123/175 撞上限，其中 104 条（84.6%）是循环**（T1/T2/T3 为主），另有 13 条是诚实的长推理被截断。

## 五、建议
- **清洗**：用 `rebalance_code_think_lane.py` 的 5-gram 过滤对完整池重选 lane；原始池会掉 37%，但被掉的主要是 >8k 词的退化长文（66% 重复），代价可接受；阈值 0.88 建议先抽样复核，必要时提到 0.92 以减少误伤枚举式有效推理。
- **修复优先于丢弃**：`experiments/clean_think_repetition.py` 提供 `scan / clean / rebuild-lane` 三个子命令，对"便宜可修"的重复（重复代码块、整行复制、重复句子、token 短循环）做定点删除并校验，修不了则整行丢弃（绝不盲截断）；`rebuild-lane` 会保留自然长度分布并强制短 think 配额。
- **保形**：保留自然长度分布 + 强制短 think 配额（≤256 token，默认 25%），并允许 `INSTINCT_EMPTY_THINK_RATIO=0.2–0.4` 教"不思考直接答"。
- **别只截断**：v4 的教训是"截短样本"不等于"去重复"；选择（整行保留/整行丢弃）优于改写。
- **监控**：训练开 `INSTINCT_THINK_STATS=1` 看 close_rate 和长度桶；评测用 `analyze_cap_hits.py` 把撞上限拆成 loop / still-thinking / code-in-think。
- **中文 lane**：需要中文分词的重复检测，别直接套英文 5-gram。

## 六、复现
- 原始池：`tmp/sft_compare/analyze_raw_code_think.py`（读 9 个 `Code_think_part-*.jsonl`），结果 `tmp/sft_compare/raw_repetition_report.json`
- v4 语料：`tmp/sft_compare/analyze_v4_repetition.py`（32 个 train_jsonl + provenance），结果 `tmp/sft_compare/v4_repetition_report.json`
- 评测循环：`experiments/analyze_cap_hits.py`，产物 `eval/livecodebench/native_instinct_r1_pos_step30720_release_v6_think_ctx4096_latest175/cap_hits.recorded.json`
