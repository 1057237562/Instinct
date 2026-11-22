# LongRoPE：从 4K 预训练扩展到 128K 上下文

## 结论

如果基础模型仅使用最大长度 4K 的序列预训练：

- 最低成本方案是训练到 32K，再用 LongRoPE 外推到 128K。
- 这种方案只能说明模型支持 128K 位置编码，不能保证拥有可靠的 128K 有效上下文。
- 要让 128K 检索、理解和推理真正稳定，训练数据中最好包含实际的 64K～128K 序列，最终训练长度最好达到 128K。

LongRoPE 原论文表明，搜索合适的逐维 RoPE 缩放参数后，模型在不微调的情况下可以实现约 8 倍上下文扩展。因此，4K 模型可先扩展到约 32K；从 4K 直接外推到 128K 属于 32 倍扩展，不能仅依赖位置插值。

## 推荐训练课程

采用渐进式长上下文持续预训练（CPT）：

```text
基础预训练：  4K
长上下文 CPT：
  第 1 阶段：   8K
  第 2 阶段：  16K
  第 3 阶段：  32K
  第 4 阶段：  64K
  最终阶段： 128K
```

每个阶段从上一个阶段的检查点继续训练，无需重新初始化模型。长度升高时应逐步降低 micro-batch，并通过梯度累积维持每步 token 数。

## 长度数据混合

长上下文 CPT 不应全部使用 128K 样本。建议按训练 token 数混合：

| 序列长度 | Token 占比 |
|---|---:|
| 4K 及以下 | 30% |
| 8K～16K | 25% |
| 16K～32K | 20% |
| 32K～64K | 15% |
| 64K～128K | 10% |

保留短上下文数据可以减少长上下文扩展造成的短上下文能力退化。LongRoPE2 也采用长、短上下文混合训练来维持原始窗口性能。

长样本应包含真正的远距离依赖，例如：

- 跨文件代码与仓库级代码关系；
- 长文档问答和跨章节信息综合；
- 长篇教材、论文或技术文档；
- 多处证据检索、needle/passkey 数据；
- 前文定义、后文使用的实体或约束。

不要仅通过拼接互不相关的短文本制造 128K 样本。这样的数据可以训练位置范围，却不能有效训练长距离信息利用能力。

## 单张 16GB 显卡的现实限制

对当前 Instinct 模型和标准全注意力而言：

| 训练长度 | 可行性 |
|---|---|
| 8K | 可作为正常训练阶段 |
| 16K | 可训练，建议启用 Flash Attention 和梯度检查点 |
| 32K | batch size 1 下尝试，需要严格控制激活与优化器显存 |
| 64K | 很可能需要 CPU/NVMe 卸载、更多检查点或上下文并行 |
| 128K | 单张 16GB 显卡上缺乏实用性，显存和计算成本都非常高 |

在现有硬件上，推荐先执行：

```text
4K 预训练
→ 8K CPT
→ 16K CPT
→ 32K CPT
→ 搜索 LongRoPE 逐维缩放参数
→ 将目标位置范围配置为 128K
→ 分别在 32K、64K、128K 上评测
```

如果必须获得稳定的 128K 有效上下文，应增加少量真实 64K～128K 训练，或引入多卡 context parallel、Ring Attention 等长序列训练方案。

## LongRoPE 参数搜索

LongRoPE 不是一个固定缩放倍率。论文使用困惑度驱动的进化搜索，为每个 RoPE 频率维度寻找独立缩放因子，并可保留前若干 token 使用原始 RoPE。缩放向量取决于模型权重、RoPE 维度、原始窗口和目标窗口，不能从其他模型直接复制后称为最优结果。

建议流程：

1. 完成至少 16K～32K 的长上下文 CPT。
2. 使用与目标领域相符的长文验证集搜索 LongRoPE 参数。
3. 同时评估短上下文与长上下文困惑度。
4. 在 4K、8K、16K、32K、64K、128K 分桶验证有效长度。
5. 根据短上下文退化情况搜索独立的短上下文恢复参数。

## 验收标准

不能只检查模型能否接收 128K token。至少应测试：

- 分长度困惑度是否随长度平稳变化；
- passkey/needle 在不同深度位置的准确率；
- RULER 32K、64K、128K 分桶成绩；
- 多文档问答、长代码理解和跨段证据综合；
- 4K/8K 基准是否因扩展而明显下降。

只有在 128K 长度下仍能稳定检索和利用远距离信息，才能称为具备 128K 有效上下文，而不只是配置了 128K 最大位置。

## 论文与实现

- [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753)
- [Microsoft LongRoPE 官方实现](https://github.com/microsoft/LongRoPE)
- [LongRoPE2: Near-Lossless LLM Context Window Scaling](https://arxiv.org/abs/2502.20082)

## 实现注意：LongRoPE 的 switch_mode（位置唯一 vs 旧 batch 全局）

本仓库同时存在两套 RoPE 选择语义，由 `rope_scaling.switch_mode` 控制（可用环境变量
`INSTINCT_ROPE_SWITCH_MODE` 覆盖）：

- `position`（默认，修复后语义）：每个位置只使用与自身位置对应的表——位置小于
  `original_max_position_embeddings` 用 short 表（常规为原始 RoPE），大于等于它用
  long 表。同一个位置在任何 packed 文档、batch 组成或 KV cache 状态下编码都一致，
  因此长上下文生成不会出现旧 key 与新 query 混用两套因子的情况。
- `batch`（旧行为，仅用于复现历史结果）：按整个 forward 的最大逻辑位置选择一张表，
  packed 长 block 会把短文档也切到 long 表；解码跨越原始上下文时会让缓存 key 与新
  token 混用 short/long 表（rollout 在跨越时会重建完整 prefix cache）。

- `smooth`（可选，仅用于新训练）：只用一张表，`position <= smooth_start` 与原始 RoPE
  完全一致；在 `[smooth_start, smooth_end]`（默认 `[original, 2*original]`）内，每个旋转对的
  **频率** `w / F(p)` 在 short 与 long 因子之间按几何方式过渡，注意力幅度线性过渡；
  `smooth_end` 之后与 long 表的相对相位一致。它消除了 `position` 模式在边界处的跳变。

  相位必须是该频率的**逐位置累加** `sum_{u<=p} w / F(u)`，不能写成 `p * w / F(p)`：因子随位置
  增大时，后者对任何 long 因子大于 e 的旋转对都会在过渡区内**倒着走**（key 在 query 之前却被
  编码成在其之后，位置 8000 与原始位置 569 的相位相同），测试
  `test_native_smooth_phase_is_monotone_and_parallel_to_the_long_table` 专门防止回归。
  累加后 `smooth_end` 之后与 long 表只差一个常数相位偏移，RoPE 只使用相对相位，因此等价。
  过渡区越长越平缓；目标上下文远大于 `2*original`（例如 64K）时，可以把 `smooth_end`
  设得更大，但这是经验判断，需要用 loss 和 passkey 验证。`smooth` 与 `position`、`batch`
  的编码互不相同，不能互换使用已训练的 checkpoint。

历史 checkpoint（包括 InstinctR1 与更早的 SFT 阶段）是在 `batch` 语义下训练的，评测时
需要设置 `INSTINCT_ROPE_SWITCH_MODE=batch` 才能复现已记录的数字；新训练与评测应统一
使用默认的 `position` 语义。实现位置：`model/rope.py::select_rope_cache`、
`model/dense_cublas/rope.py::mixed_longrope_tables`、`model/dense_cublas/model.py`，
测试见 `tests/test_longrope.py` 与 `tests/test_native_sft_longrope.py`。
