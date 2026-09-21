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

### 2. 每个 batch 只构造一次组合 attention bias（已实现）

- 将 causal mask 与 packing segment mask 的合并从每层 attention 中移到模型入口。
- 将合并结果一次性转换为 attention 计算精度的 `0/-inf` bias，并按 64 元素对齐行存储。
  仅共享布尔 mask 不够：SDPA 仍会逐层转换和保存浮点 bias。
- Dense/MoE 主干和 Looped 主干的 packed SDPA 路径共享同一只读 bias；不改变普通
  0/1 mask 接口、非 packed 快路径或 KV-cache 路径，不添加参数或持久化缓存。
- 验收：packing 隔离、前向/梯度、checkpoint、CUDA eager/compile 存储复用检查通过。
- RTX 5070 Ti，32 次 compiled SDPA，`4×2928`、16 Q heads / 4 KV heads、head_dim=32、BF16：
  保存的独立 bias 从 32 份降到 1 份；隔离基准峰值从 3345.2 MiB 降到 1297.5 MiB
  （均不含输入），约节省 2.00 GiB。该结果不是完整训练的峰值或吞吐承诺。
- 复现：停止训练后运行 `python experiments/bench_shared_attention_bias.py --compile`。

### 3. 调整 gradient checkpoint 粒度

- profile 当前 attention/FFN 的激活占用和重计算耗时。
- 优先试验只 checkpoint 高显存模块，避免对所有 FFN 无差别重算。
- 验收：不 OOM；固定 batch 下 step time 改善；峰值显存在预算内。

### 4. 结构化 packing attention（已实现，待真实训练吞吐验收）

- Packed CUDA BF16、无 dropout 的输入默认采用 FlexAttention Triton `BlockMask`；
  每个 batch 构建一次并跨层共享，跳过完全屏蔽的块，K/V 保持紧凑 GQA。
- 保持文档隔离、因果、padding 和位置语义；显式额外 mask、CPU、其他精度继续使用 SDPA。
- 输出/梯度、MoE checkpoint 模式 1/2、Looped、连续不同分段的 compiled batch 检查通过。
- RTX 5070 Ti 单层合成基准（4×2928、4 等长文档、Q/KV=16/4、D=32）：
  前反向 13.33 → 1.73 ms；不是端到端训练提速承诺。
- 恢复训练后预热两个桶再比较 tokens/s；可通过
  `INSTINCT_PACKED_ATTENTION_BACKEND=sdpa` 回退，无需转换检查点。
- 说明与复现见 `docs/packed_flex_attention.md`。

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
