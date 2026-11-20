# SFT loss 计算与"表演式思考"成因调查（2026-11-20）

调查对象：当前正在运行的 InstinctR1 think stage（`tmp/native_instinct_r1`，2026-11-19 15:58Z 启动，
训练数据为 `dataset/instinctv2_sft_concise_reasoning_635m_v4/native_packed_16k`），以及用来产生
当前评测结论的 v4 SFT checkpoint（`out/native_sft_concise_reasoning_635m_v4`，step 3,000）。

结论一句话：**CE 的算术没有问题，问题在于这个目标函数的结构与它所监督的数据分布**。SFT 的
loss 是"assistant token 的一元平均"，对"思考 token"和"答案 token"完全同权，对长度、闭合、
思考是否有用没有任何信号；而 v4 语料几乎只提供两种模式——"没有 think 块"（83.5%）和
"think 块一旦出现就很长"（p50 381 token / 849 字符，p90 2,300 token）。推理侧 `open_thinking 1`
恰好强制进入第二种模式，于是模型按学到的长度先验写满预算，且不会主动闭合 `</think>`。
这就是"表演式思考"。

---

## 1. 当前训练任务的 loss 在哪里、怎么算

当前 run 走的是 native/cuBLAS 路径，不是 `trainer/train_full_sft.py`：

| 环节 | 代码 |
| --- | --- |
| 启动命令 | `tmp/native_instinct_r1/start.process.json`：`--train_type full_sft --compute_profile v2_mxfp8_sft --config_path tmp/native_sft_longrope128k/model.config.json --batch_size 1 --max_seq_len 16384 --packing --packed_attention --loss_chunk_size 1024 --accumulation_steps 4` |
| 训练循环 | [train_cublas.py](../../trainer/train_cublas.py#L201-L216) `supervised_step`；[train_cublas.py](../../trainer/train_cublas.py#L511-L546) 取 batch、反传、记录 `progress['supervised_tokens']` |
| 数据 | [sft_data.py](../../model/dense_cublas/sft_data.py#L73-L89) 从 Arrow 的 `input_ids` / `loss_mask` / `segment_lengths` 组 batch |
| 标签 | `labels = ids where loss_mask else -100`，并把每个 packed 文档的首 token 置 `-100` |
| loss 入口 | [model.py](../../model/dense_cublas/model.py#L484-L487) `loss = e.linear_ce(hidden[:, :-1], lm_head.weight, labels[:, 1:], loss_chunk_size)`（`loss_chunk_size=1024` 走 fused 分支） |
| fused CE 实现 | [autograd.py](../../model/dense_cublas/autograd.py#L369-L404) `linear_ce`（分块投影 + Triton cross-entropy，反向重算 logits） |
| 非 fused CE | [autograd.py](../../model/dense_cublas/autograd.py#L567-L594) `ce(..., shift=True)` |
| packed 位置/掩码 | [packing.py](../../model/dense_cublas/packing.py#L50-L64) `prepare_offsets`：文档内位置从 0 重排，`position<=0` 的 label 置 `-100` |

等价公式（single GPU、`accumulation_steps=4`）：

```
label_t = input_ids_t        若 t 落在某个 assistant 段内（见 §2），否则 -100
L_block = Σ_{t: label_t ≥ 0} CE(logits_{t-1}, label_t) / #{t: label_t ≥ 0}
L_step  = (L_block,1 + L_block,2 + ... + L_block,4) / 4
```

审计结论（**没有发现算术 bug**）：

- `linear_ce` 的分块累加是严格的全局 token 平均：`denom = #{valid labels}`（[autograd.py#L374-L388](../../model/dense_cublas/autograd.py#L374-L388)），
  反向逐 token 缩放 `g/denom`（[autograd.py#L390-L403](../../model/dense_cublas/autograd.py#L390-L403)），与不分块 `ce` 的
  `Σloss / Σcounts` 一致；分块只影响显存/重算，不改变梯度。
- `ignore_index` 语义一致：label 为 `-100` 的 token 既不计入分子也不计入分母；fused kernel 的 `counts` 与
  单独算的 `valid` 都只在 `0 <= label < vocab` 时计 1。
- packed 文档边界处理正确：每个文档首 token 在 `PackedSFT.batch` 与 `prepare_offsets` 各被清一次，避免用上一文档
  的末尾 hidden 去预测下一文档首 token。
- **没有任何 token 权重**：数据侧只产出 `(ids, labels, segments)`，loss 侧没有 `loss_weight`/`token_weight`
  输入；整个 SFT 只有上面这一个标量。

也就是说：`L` 是一个**纯模仿下一 token、按 token 计票**的目标。它对"该想多久"、"该不该停"、"想得对不对"
以及"答案对不对"都没有任何项。

## 2. 到底哪些 token 被监督

SFT 的标签不是"整条对话"，而是**仅 assistant 段**：

- `bos_id = tokenize('<|im_start|>assistant\n')`，`eos_id = tokenize('<|im_end|>\n')`
  （[lm_dataset.py](../../scripts/data_loader/lm_dataset.py#L586-L587)、native 侧 [data.py](../../model/dense_cublas/data.py#L34)、[data.py](../../model/dense_cublas/data.py#L63-L72)）。
- `_generate_sft_labels` 只把 `bos_id` 之后到 `eos_id`（含）之间的 token 标为监督
  （[lm_dataset.py](../../scripts/data_loader/lm_dataset.py#L115-L131)）。
- 我在真实训练缓存上验证：抽 28,088 个被监督的连续区间，**0 个**不以
  `<|im_start|>assistant\n` 开头（`experiments/audit_packed_sft_loss_mass.py`），即 user/system/tool 内容确实不参与 loss。

所以：loss 的分子/分母完全由 assistant 段内的"思考 + 答案 + 结束符"构成，且三者同权。

## 3. 实测：v4 语料把 loss 的质量放在哪里

数据：`native_packed_16k`，34,578 个 16,320-token block，559,202,079 token，其中
**392,169,498（70.13%）被监督**；`packing.report.json` 记录 `supervised_think_markers=132,915`、
`closers=132,918`、**`empty_thinking_blocks=0`**。

抽样 1,200 个 block / 28,088 个 assistant turn（按 block 均匀抽样，与 loss 的 block 平均口径一致）：

| 指标 | 实测值 |
| --- | --- |
| 含 think 块的 assistant turn | **16.5%**（4,646 / 28,088），其余 83.5% 完全没有 think 块 |
| 监督 token 中 think 正文占比 | **28.9%**（标签 `<think>`/`</think>` 另计 0.07%） |
| 监督 token 中可见答案占比 | 71.0% |
| think 长度（token/turn，仅有块者） | p50 **381**，p75 1,172，p90 **2,300**，p99 4,658，max 8,034 |
| think 长度（字符/turn） | p10 174，p50 **849**，p90 5,595，max 23,873 |
| 短思考 | **0% 短于 20 字符**；1–4 token 桶 **0 条**；5–64 token 仅 1.69% |
| >1,000 字符的 think 块 | **46.5%** |
| 落在项目自己 RL 奖励带（20–300 字符）内 | 仅 **18.7%** |
| think token 的梯度权重集中度 | 最长的 10% 块占 **38.9%**，最长 20% 占 **60.8%** |

think 长度分桶（token；"权重占比"即该桶在 think 段 loss 分母中的份额）：

| think tokens | turn 占比 | 占全部 think token |
| ---: | ---: | ---: |
| 0（无块） | 83.5% | 0% |
| 1–4 | 0.00% | 0.00% |
| 5–64 | 1.69% | 0.42% |
| 65–256 | 3.58% | 4.62% |
| 257–768 | 5.70% | 16.81% |
| 769–1536 | 2.42% | 18.97% |
| 1537–2560 | 1.81% | 25.50% |
| 2561–4096 | 1.10% | 24.38% |
| >4096 | 0.25% | 9.29% |

think 块的**开头 token**也高度模板化（4,646 个块）：`好的，用户`(17.7%)、`We are given`(17.2%)、
`First, the`(10.7%)、`We are ask`(5.0%)、`我们被问`(4.6%)、`We need to`(3.8%)、`Okay`(3.7%)、
`I will call`(3.4%)——前三种开头覆盖 45.5% 的块。think 正文的中文字符占比均值 0.23（28.7% 的块以中文为主），
答案段 0.20，两者相近。

### 3.1 空 think 块是被**故意删掉**的

[pack_v4.py](../../tmp/native_sft_concise_reasoning_v4/pack_v4.py#L21) 设置 `INSTINCT_EMPTY_THINK_RATIO=0`
（[lm_dataset.py](../../scripts/data_loader/lm_dataset.py#L85-L92) 该环境变量覆盖默认 0.2 的保留率），
并在打包后校验 `empty_thinking_blocks == 0`（[pack_v4.py](../../tmp/native_sft_concise_reasoning_v4/pack_v4.py#L101-L118)），
`launch.py` 也强制要求 `packing.report.json` 里该值为 0。

设计意图是"不教模型输出空思考"。实际后果是：**训练集里不存在"想一句就收"的样本**。
一旦推理侧把 `<think>` 打开，模型面对的条件分布是"think 正文 p50 381 token、p90 2,300 token"，
而不是"立刻 `</think>`"。

## 4. 为什么会出现表演式思考：五条链路

另需注意 R1 阶段的起点：它从 V2 1B pretrain（cloud step 2,004,666，24B token）**直接**在 v4 上起训，
没有中间的 no-think 代码 SFT 基座（见 [launch.py](../../tmp/native_instinct_r1/launch.py#L1-L9)）。
也就是说，这份语料是模型学到的第一条"怎么回答"的分布，没有任何"简洁作答"的旧习惯与长思考先验竞争。

### 4.1 长度先验来自"有洞的混合分布"

训练里只有两种模式：无 think 块（83.5%）、think 块且很长（16.5%）。没有第三种"短思考"。
`--open_thinking 1` 在 prompt 里强制写入 `<|im_start|>assistant\n<think>\n`
（[tokenizer_config.json](../../model/tokenizer_config.json) 的 `add_generation_prompt` 分支），
等于把模型直接推进第二种模式，于是采样长度服从 `P(len | 块已打开)`——几百到几千 token。
长度与题目难度**没有任何条件关联**：语料里不存在"简单题→短思考"的配对。
反过来，`--open_thinking 0` 会渲染一个空的 `<think>\n\n</think>\n\n` 块，而这个模式在训练集里被删干净了；
它反而更好用，是因为训练里 83.5% 的样本"根本没有块"，模型学到的是"盒外直接答题"。

### 4.2 token 平均让长尾主导梯度

`L` 是 token 平均，所以一个 turn 对梯度的贡献 ∝ 它的监督 token 数。think 只占 16.5% 的 turn，
却占 28.9% 的监督 token；而且 think 内部极度长尾（最长 20% 的块占 60.8% 的 think token）。
换句话说，**"怎么想"这一课的教材 60% 来自最啰嗦的那 20%**。short/简洁样本在梯度上几乎不存在。

### 4.3 模仿目标对"有用"无感，对"像在思考"有奖励

SFT 没有任何验证器/奖励/长度项。CE 只要求下一个 token 与语料一致。语料的 think 正文显著比答案段
更可预测（字符熵 5.76 vs 6.01 bits/char；开头 45.5% 集中在 3 种句式），因为大量"推理"是
复述题面、列举约束、计划下一步的元话语：

- "好的，用户问……首先，我需要理解用户的问题核心……接下来，我得考虑……然后，用户可能想知道……"
- "We are given ... We need to ... Let's denote: ... We have: ... We can try to ..."

这类文本的 loss 最容易降：它们是**低熵的模板**。模型于是把"思考的样子"学得很像，
而真正的推导（高熵、每题不同）学得少。语料里的极端样本都是两万字级的元规划：
一条 23,330 字符的 think 块通篇在规划"如何让每个句子比上一句包含更多连续头韵词"，
另一条 17,320 字符的块在规划"如何塞进至少 72 个代词"——没有任何求解，只有对约束的自我检查。

### 4.4 缺少"停"的监督

全语料只有 132,915 个被监督的 `<think>` 开启、132,918 个 `</think>` 闭合，而 `<|im_end|>`
每个 assistant turn 都有（约 80 万个）。也就是说：**"结束回答"的信号到处都是，"结束思考"的信号只有 13 万个，
且全部位于长正文之后**。贪心解码下（LCB 用 temperature 0、无 repetition penalty），
模型一旦写了模板开头，`</think>` 的即时概率极低，于是滑到上下文上限。

### 4.5 v4 的 code lane 显式注入了 2048-token 长度先验

`build.report.json` 记录 code-think lane 的思考长度是**按长度重采样的截断正态 N(2048, 1024²)**：
7,390 行 / 21.4M token，按 0–256、256–512 …… 分桶配额挑选（目标在 1536–2048 与 2048–2560 两桶各 19.6%，
即均值 2048）。
这是语料里**唯一**"代码题→先长思考再给代码"的监督，所以代码题的条件长度先验≈2048 token；
而 LCB 的可用输出预算只有 `4096 − 批内 prompt 宽度(442) ≈ 3,654` token。

v3→v4 的行为对比正好是这个 lane 的对照实验（v4 从 v3 step-3,072 权重续训，仅替换 code lane）：

| 指标（LCB release_v6 latest-175, ctx4096, greedy, MXFP8 KV） | v3 think | v4 think | v4 no-think |
| --- | ---: | ---: | ---: |
| pass@1 | 19/175 = **0.1086** | 15/175 = **0.0857** | 21/175 = **0.1200** |
| 撞上下文上限 `no_code` | 27 | **107** | 18 |
| 真正闭合 `</think>` | 173 | 72 | — |
| think 非空 | 14 | **172** | — |
| 产出最终代码 | 148 | 68 | 157 |
| 生成 token 总数（/task） | 135,781（776） | **446,370（2,551）** | 111,804（639） |

读法：v3 的 think 模式其实是"空思考"（只有 14 个非空），pass@1 0.1086；v4 加上长思考 lane 后，
思考从"几乎不写"变成"几乎必写、平均 2,551 token、61% 撞上限"，pass@1 反而更低（175 题上 19→15 的差异
本身在噪声范围内，但 27→107 的撞限是 4 倍、无歧义的行为变化）。no-think 模式几乎没变（0.1257→0.1200），
说明受损的是"被强制开启的思考模式"，而不是模型的基础代码能力。难度切分同样难看：easy 题 think
pass@1 0.326、**8/43 连代码都没写出来**，而 no-think 是 0.442、0/43 无代码。

## 5. 为什么训练日志看不出这个问题

[train_cublas.py](../../trainer/train_cublas.py#L534-L576) 每个 log_interval 只打印 **一个标量 loss**
（该 block 的 token 平均），没有 think/answer 拆分、没有 think 长度、没有闭合率、没有 per-turn 指标。
实测 loss 曲线：

| run | 早期窗口 | 后期窗口 |
| --- | ---: | ---: |
| v4（step 4→3,000） | 0.9182 ± 0.20 | 0.8708 ± 0.20 |
| R1（step 4→816，进行中） | 1.2219 ± 0.23 | 1.0598 ± 0.23 |

CE 确实在降（v4 降约 5%），而同一 checkpoint 的思考行为从"几乎不想"变成"停不下来、正确率不升"。
**注意 v4 的 3,000 步只消耗了 1 个 epoch 的 8.7%（约 1.15 万个 think 块）**，当前 R1 跑到 step 738
（约 2,800 个 think 块），长思考先验才刚开始被强化——现在正是加诊断、而不是等评测的时候。

这里存在明确的可诊断性缺口：一次 step 只报一个均值，把"学会了思考"和"学会了答题"混在一起，
而这两件事在本项目里恰好朝相反方向变化。

## 6. 项目内部的信号冲突

`train_ppo.py` / `train_grpo.py` / `train_agent.py` / `model/dense_cublas/rollout.py` 里的奖励函数写着：

```python
reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5   # 思考长度分
reward += 0.25 if response.count('</think>') == 1 else -0.25  # 思考闭合分
```

（[train_grpo.py](../../trainer/train_grpo.py#L81-L84)、[train_ppo.py](../../trainer/train_ppo.py#L99-L102)、
[rollout.py](../../model/dense_cublas/rollout.py#L97-L100)）

而 SFT 语料里只有 **18.7%** 的 think 块落在 20–300 字符内，p50 是 849 字符，46.5% 超过 1,000 字符。
也就是说：**SFT 在教 RL 随后要惩罚的行为**。如果 R1 的路线是 SFT → RL，RL 的第一件事就是花预算把
SFT 灌进去的长度先验压回去；如果 R1 只做 SFT，那这个长度先验没有任何约束。

## 7. 建议（按性价比排序）

1. **先加零梯度诊断（不改训练）**：按 token id 25/26 把监督 token 分成 think 正文 / `</think>` / 答案三段，
   每 log_interval 输出 `CE_think`、`CE_answer`、think 平均长度、`</think>` 闭合率、每个 block 的 think 块数。
   `supervised_tokens` 已经算出来了，这一步只在训练循环里多几次掩码统计，不触碰优化器。
2. **修数据里的"洞"**：不要把所有空 think 块删掉。把 `INSTINCT_EMPTY_THINK_RATIO` 从 0 调到 ~0.2–0.4，
   并显式加入"1–2 句短思考 + 答案"的样本（目标：短思考占 think 块的 20–30%），让"块已打开但很快闭合"
   成为可学到的模式。需要重建 packed cache（新目录 + 新 `packing.report.json`），不要覆盖现有 artifact。
3. **去掉 code lane 的长度配额重采样**：N(2048, 1024²) 的按长度选样把"代码题=2048 token 思考"写进了先验；
   改为自然长度或按难度分层，并为每个代码题配套一条 ≤256 token 的简短轨迹。
4. **loss 侧加权（需要新 stage）**：think token 乘 λ≈0.3–0.5，或对 `</think>` 单独加权；
   或改用已存在但当前未启用的 per-document 平均路径
   （[training_profile.py](../../model/dense_cublas/training_profile.py#L137-L153) 的 `sum_sequence_means=True`，
   需要 batch 2 + streaming parquet + `v2_mxfp8`，与当前 packed SFT 路径不兼容）。
5. **推理侧加 think 预算**：超过 N token 强制注入 `</think>`（LCB 抽取器本来就按第一个 `</think>` 切分，
   这只是把 `no_code` 换成"给了代码但可能错"），或在 think 段加 repetition penalty。
6. **与 RL 目标对齐**：SFT 的长度先验必须和 20–300 字符的 shaping 相容，否则 RL 阶段会先"去学习"。
7. **次要、待确认**：`rope_long` 是**按 block 决定**的
   （[model.py](../../model/dense_cublas/model.py#L423-L434)：`document_max_length(block) > 4096` 就整块切 32× long 表）。
   实测 16.8% 的 block 含 >4096 的文档（承载 16.9% 的 token），这些 block 里的短文档也在用 long 表，
   而基座是 4096 short 表预训练的。建议改成按文档决定（或限制单文档长度）并做一次 A/B——这条没有证据表明是
   表演式思考的成因，但它是训练-推理位置编码不一致，值得单独验证。

**不要在原地改正在运行的 R1 job**：loss/数据一改，optimizer 状态、数据游标和精度契约都要重新对齐，
按仓库约定应当从权重开新 stage（暂停用 save-and-pause，不能用 kill）。R1 目前 738/34,578 步，
损失尚可，继续跑完一个 epoch 也可作为"纯模仿 SFT 下长度先验如何演化"的基线。

## 8. 复现

```powershell
# 直接读训练用的 Arrow 缓存，统计监督 token 构成（CPU，不加载 torch，不影响训练）
python experiments/audit_packed_sft_loss_mass.py --blocks-per-shard 300 --json tmp/v4_packed_loss_mass.json
```

产物：`tmp/v4_packed_loss_mass.json`；JSONL 层面的旁证脚本（按 chat template 渲染 + `_generate_sft_labels`
同构实现，含字符熵与最长 think 样本）在 `tmp/analyze_sft_loss_mass.py`、`tmp/analyze_v4_think_mass.py`、
`tmp/verify_packed_loss_mask.py`（均为 tmp 下 scratch）。

## 9. 局限

- think/answer 切分依赖 `<think>`/`</think>` 是单 token（id 25/26，已在 [tokenizer.json](../../model/tokenizer.json) 中确认）；
  正文里若出现这两个字符串会切错，量级可忽略。
- 抽样是 block 均匀抽样（对 loss 口径正确），不是按 turn 均匀；因此"turn 占比"是 block 加权的。
- 175 题的 pass@1 差异不具统计显著性，能确证的是行为量级（撞限 27→107、生成 776→2,551 token/task）。
- 长 RoPE 那条只统计了触发比例，没有做训练对照，不能当作已证实的成因。
