# 训练性能优化计划

## 目标与基线

- 目标：在不改变 packing 隔离语义、因果语义和训练结果正确性的前提下，提高 tokens/s，并降低峰值显存和每步耗时。
- 当前实测基线（`batch_size=28`、`max_seq_len=1024`、16 层、BF16 参数、tensorwise FP8、compile、gradient checkpoint）：约 `0.55–0.56 s/step`、`12.38 GB` 显存，packing 填充率约 `98%`。
- 比较规则：排除首次编译和 autotune 时间；固定相同模型、数据、batch 和随机种子；至少预热 20 步后再统计稳定区间。

## 实施顺序

### 1. SDPA GQA 路径（已回退到实测最快实现）

- packed SDPA 使用显式 K/V repeat；`enable_gqa=True` 在当前
  `torch.compile` 环境会被分解为 dense BMM，因此不采用。
- 保持无 mask、packing block-diagonal mask、causal mask 和 dropout 语义不变。
- 采用已实测最快的组合：attention 直接走 SDPA，mode 1 仅对 FFN 做
  selective checkpoint，不再对 fused attention 做额外反向重放。

### 2. 每个 batch 只构造一次组合 attention mask

- 将 causal mask 与 packing segment mask 的合并从每层 attention 中移到模型入口。
- 所有 Transformer 层复用同一个只读 mask，减少逐层张量分配和布尔运算。
- 验收：packing 隔离测试通过，profile 中 mask 构造次数从“每层一次”降为“每 batch 一次”。

### 3. 调整 gradient checkpoint 粒度

- profile 当前 attention/FFN 的激活占用和重计算耗时。
- 优先试验只 checkpoint 高显存模块，避免对所有 FFN 无差别重算。
- 验收：不 OOM；固定 batch 下 step time 改善；峰值显存在预算内。

### 4. 结构化 packing attention

- 评估 PyTorch FlexAttention `BlockMask`，让 packed segment 只计算各自的 attention block。
- 当前样本每个 block 约 4 个 segment，理论 attention 矩阵计算量约可降至现有 dense masked attention 的 `39%`。
- 验收：与 dense block-diagonal SDPA 输出/梯度对齐，且端到端 tokens/s 确有提升；否则保留 SDPA 路径。

### 5. 训练管线次级优化

- 在 epoch 边界试验更大 batch（例如 28 → 32），不得在 resume 的 epoch 中途改变 batch 语义。
- 评估 chunked LM loss，减少完整 vocab logits 的峰值显存。
- 合并/异步化 checkpoint 写入，减少约每 1000 步的训练停顿。
- 为 DataLoader 评估 `pin_memory`、`non_blocking`、`persistent_workers`，仅在 profile 证明存在数据等待时启用。

## 正确性护栏

- packed 样本之间不能互相 attention。
- 每个 packed segment 的 RoPE position 必须从 0 重新开始。
- segment 边界标签必须屏蔽，不能学习跨样本 next-token。
- 所有优化先经过 CPU 数值/梯度测试，再在下一次训练启动或 resume 时做 GPU 基准；当前运行中的 Python 进程不会热加载代码修改。
