# ckpt3 OOM 复核与 ckpt4 规划

日期：2026-10-23。本文保留最初的 OOM 诊断与候选设计。
后续 ckpt4 已实现为按桶选择 KV/Attention、再保存顶部部分 gate/up 的预算 SAC，
取代本文的“29 轻量 + 3 完整”初始候选；见
[ckpt4 实现与逐算子 FLOPs/显存表](../selective_checkpoint_mode4.md)。
最初诊断仅做 CPU/meta 核算；后续 GPU 验证及其测量范围单独记录在实现文档。

## 结论

这次 OOM 是当前 ckpt3 的保存策略在 **12,288 tokens/micro-batch** 下超过显存容量。
此前 2,048 tokens 的测试没有 optimizer update，也在每次 forward 前清掉梯度；
测得的 **12.02% allocated 增幅只属于那组 FP8 测试**，不能当作档位固有的显存倍率。
ckpt3 缺少与真实 batch、梯度累积和优化器状态相匹配的容量保护，是需要补齐的设计缺口。

建议保留 ckpt3 的小 batch 高吞吐用途，同时设计 **预算约束的 ckpt4 混合策略**。
优先补容量估算/预检，再以真实训练的 useful tokens/s 比较 ckpt2、ckpt3、ckpt4。
只增加档位编号、或只手写 RMSNorm/SwiGLU epilogue，不能解决本次主要存储项。

## 1. 本次实际运行条件

来源：`trainer/logs/train_pretrain_20261023_190918.log`，与用户粘贴的 OOM 堆栈一致。

| 条件 | 实际值 |
|---|---|
| 模型 | V2 Dense，32 层，H=1536，F=5504，16 Q / 4 KV heads，D=96 |
| 参数量 | 1,010,277,888；其中 Muon hidden matrices 1,000,341,504 |
| 计算/参数 | BF16，TorchAO tensorwise FP8，auto filter |
| 优化器 | Muon mantissa，BF16 momentum，512 MiB 临时 workspace 目标 |
| 编译 | `max-autotune-no-cudagraphs` |
| checkpoint | 3 |
| packing | bucket；显式 `bucket_token_budget=12288`，不是已验收调优结果 |
| 实际 batch | 4×3072 或 3×4096，均为 12,288 tokens |
| 累积 | 8 microsteps/update |
| 显存 | 物理 15.89 GiB；WDDM process budget 14.87 GiB；allocator limit **14.37 GiB** |
| 故障 | 第一个 microstep 的 forward，分配 `[12288,5504]` BF16 失败 |

`batch_size=1` 和 `max_seq_len=2048` 不是此 bucket loader 的实际 micro-batch。
这里 batch 按 `max(1, token_budget // bucket_length)` 得到；实际桶长是 3072/4096。
只降低普通 batch_size 控件不会改变这个 12,288-token batch。

日志的 `memory_curve=legacy_conservative` 没有 ckpt3 的实测曲线；显式 token budget
又优先于自动值。因此调优入口已经允许 ckpt3，不代表这个手动设置已通过容量验证。
把 token budget 改回 0 也不能直接视为已修复：旧的通用曲线仍未按本次策略校准。

## 2. 从故障编译图核算保留张量

实际文件：
`.cache/torch_compile/6n/c6nu72yqdrihlzc32rngucciwqlmsxgq2u5z3kdypkf5eeoxqeer.py`

SHA-256：`4d033e03923237a51ddc2c23430c4e386696c2cedec27788e06a5a827d4e272c`。

CPU 静态分析 `call` 中的 allocation、`reinterpret_tensor`、别名复用、`del` 和
返回列表，按底层存储去重；不导入或执行生成的 CUDA 代码。返回张量和 buffer
赋值均已解析，无未知项。核算的是完成 forward 时将保留的新增 buffer，包含
最终输出；不把返回的参数输入重复计入。不是一次成功运行的 peak allocated 测量。

| 存储项 | 12,288 tokens 下的大小 |
|---|---:|
| 32 层 gate/up，64 个 BF16 `[12288,5504]` | **8.0625 GiB** |
| Q/K/V/O 投影结果 | 2.8125 GiB |
| 融合注意力输出与 FP32 LSE | 1.1484 GiB |
| FP32 残差流及末端输出 | 2.2500 GiB |
| 末层 down 输出、末端统计量及 FP8 标量 | 0.0352 GiB |
| **新增返回存储合计** | **14.3086 GiB** |

单个 gate/up 输出的净张量大小为：

`12288 × 5504 × 2 = 135,266,304 bytes = 129 MiB`。

日志的 130 MiB 请求与这个大小处于同一量级，分配器还存在块/段粒度。
堆栈的 `buf1742` 对应第 27 层的 up 输出；该点之前，静态存活的新 buffer
已约 **11.947 GiB**，加权重和图外存储，与日志的 allocated=14.08 GiB 一致地
指向大量活跃张量积累。这里没有用静态分析值冒充运行时峰值。

这份图的主要保留项可写为：

```text
T = B×S
H=1536, F=5504, qd=1536, kd=384, Nheads=16, L=32
每层/每 token：
  2×(Q + K + V + AttentionOutput + O + gate + up)
  + 4×H（残差流）+ 4×Nheads（LSE）
= 2×(2qd + 2kd + H + 2F) + 4H + 4Nheads
= 38,976 bytes

graph_saved(T) ≈ 38,976×T×32 + 2H×T + 4T + 644 bytes
```

不同 batch/seq shape 重新编译后，融合、复用和保存集合可能改变；该式用于同类
路径的容量规划，必须以新图和运行时峰值复核。它比把一次峰值的百分比乘到另一个
batch 上更有依据。

FP8 不能把这些项全部减半：故障图的 `_scaled_mm(..., out_dtype=torch.bfloat16)`
已经证明 gate/up 保存的是 BF16 输出。训练未启用 KV cache，`kv_cache_dtype`
也不会减少这些激活。

## 3. 固定状态与阶段重叠

使用当前模型在 meta device 构建并调用仓库的 persistent estimator，得到
`8,161,714,176 bytes = 7.6012 GiB`；没有分配实际模型/GPU 存储。

| 项目 | 大小 |
|---|---:|
| BF16 hidden matrices + FP32 embedding/head/norm 参数 | 1.9003 GiB |
| 全部参数梯度 | 1.9003 GiB |
| Muon BF16 momentum + uint16 mantissa + 辅助 AdamW 两份状态 | 3.8006 GiB |
| **固定状态合计** | **7.6012 GiB** |

不能认为全部固定状态从第一步就已分配，也不能只验证第一步能否运行：

- 本次从零启动、无 momentum 预分配记录；第一次 forward 尚未做 optimizer update。
  即使只计权重，`1.9003 + 14.3086 ≈ 16.209 GiB` 已超 allocator limit。
  再计图输入的 FP32 hidden（0.0703 GiB），约 **16.28 GiB**，仍未计 loss/head。
- accumulation=8 时，后续 microstep 的 forward 会与前面累积下来的梯度重叠。
- 第一次 optimizer update 后，优化器状态常驻；从该组累积的第二个 microstep
  起，`7.6012 + 14.3086 ≈ 21.91 GiB` 同时存在，再加图输入、head/loss、临时量。

因此不能靠降低 512 MiB Muon workspace 来解决这个 first-forward OOM；其主要
workspace 在 optimizer 阶段使用。减小 workspace 可能改善更新阶段峰值，但
不是释放这 8.06 GiB gate/up 的办法。仅把 accumulation 改成 1 也不能让
本次 12,288-token 首次 forward 装入显存。

未分配 reserved 只有约 174 MiB；碎片处理无法消除数 GiB 的活跃保存项。
`empty_cache()` 不释放仍由张量占用的存储，见
[PyTorch memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management)。

## 4. ckpt3 的重新容量规划

以下是沿用当前图保存集合的线性估算，**不是这些 batch 的 GPU 实测峰值**。
第二列不含 optimizer，第三列加上累积阶段的 7.6012 GiB 固定状态；仍需额外
预留图输入、head/loss、单层 backward 临时量、分配器/编译 workspace。

| 每个 micro-batch 的 tokens | 图中新增保留项 | 加固定状态后的下界 |
|---:|---:|---:|
| 2,048 | 2.38 GiB | 9.99 GiB |
| 3,072 | 3.58 GiB | 11.18 GiB |
| 4,096 | 4.77 GiB | 12.37 GiB |
| 5,344 | 6.22 GiB | 13.82 GiB |
| 6,144 | 7.15 GiB | 14.76 GiB |
| 8,192 | 9.54 GiB | 17.14 GiB |
| 12,288 | 14.31 GiB | 21.91 GiB |

当前 14.37 GiB allocator limit 下，ckpt3 可从 **token budget=4096** 开始验证，
保持 bucket_max_seq_len=4096，则当前两个桶变成 **1×3072 / 1×4096**。
这是候选起点，不是已验证可长期运行的承诺；需要经过首个 optimizer update，
再覆盖一个完整 accumulation 周期及两个桶。6,144 已没有足够稳态余量。

降低 micro-batch 会改变每次 optimizer update 的有效 tokens。若保持原约
98,304 tokens/update，3072 和 4096 长度分别需约 32 和 24 次累积；单个固定
accumulation 值不能同时精确匹配两个桶。不要为了复用旧数据游标静默改变已有
resume 的 packing/token budget；同 token 收敛对照应独立安排。

## 5. 是否需要 ckpt4

**建议设计；主要用途是在 8k–12k micro-batch 下给出有预算约束的折中策略。**
不强求最大 batch，不承诺超过 ckpt3 小 batch 的端到端 MFU。

只删除 gate/up 缓存仍会保留约 `14.3086−8.0625=6.2461 GiB`。
加固定状态已经 **13.8473 GiB**，离 allocator limit 只剩约 0.52 GiB，
图输入、head/loss 和 backward workspace 尚未计入。因而固定“FFN 重算”不足以
作为这个 12k 配置的稳健方案。

推荐 ckpt4 使用两种已有能力组合，避免先写大面积自定义反向：

| 层策略 | 保留 | 重算 |
|---|---|---|
| 轻量 SAC | 层输入、受支持融合 Attention 的 output/LSE | QKV/O、gate/up、norm/RoPE/SwiGLU；未必重放末端 down，由 early-stop 决定 |
| 完整 SAC | ckpt3 的投影和融合 Attention 输出 | 便宜中间量 |

轻量 SAC 复用现有 `dense_attention_checkpoint_contexts` 的思路；它已作为 ckpt2
的 `INSTINCT_DENSE_SELECTIVE_CHECKPOINT=attention` 实验存在。新档位的价值是
**按预算选择两种策略的层数、顺序和真实形状计划**，并提供容量保护。
预算版的轻量层应对大投影 GEMM 显式使用 `MUST_RECOMPUTE`，保留操作用
`MUST_SAVE`；不能只返回 `PREFER_RECOMPUTE` 后假定编译器一定满足存储上限。
编译完成后还要检查真正跨 forward/backward 保存的张量集合，再做运行时峰值验证。

在这个 V2/FP8/Flex 形状上做初始估算：

```text
轻量层：4H + 2qd + 4Nheads = 9,280 bytes/token/layer
完整层：38,976 bytes/token/layer
升级一层的额外保留量：29,696×T bytes

12,288 tokens：
  全轻量的图保留量约 3.434 GiB
  每升级一层增加约 0.340 GiB
```

如果初始图保留预算取 **4.5 GiB**，可先规划 **底部 29 层轻量、顶部 3 层完整 SAC**：
保留量约 **4.453 GiB**，相较 ckpt3 的 **14.309 GiB 减少约 68.9%**。
这是保留张量预算的降幅，**不是已测 peak allocated 降幅**。
加固定状态后约 **12.05 GiB**，再预留 1.5 GiB 给图外/临时开销，约 **13.55 GiB**；
该 1.5 GiB 仅是待实测校准的初始余量，不能当作通过验收。

将完整 SAC 放在顶部，可在 backward 较早释放其大缓存；底部的重算发生在这些
缓存释放以后。还须实测完整 backward 峰值，不能只根据 forward 返回量选层。

按相同 4.5 GiB 初始保存预算推导的候选如下，最终层数必须由实际图确认：

| 实际 tokens | 轻量层数 | 完整 SAC 层数 | 预计图保留量 |
|---:|---:|---:|---:|
| 3,072 | 0 | 32 | 3.58 GiB |
| 4,096 | 3 | 29 | 4.43 GiB |
| 6,144 | 16 | 16 | 4.44 GiB |
| 8,192 | 23 | 9 | 4.33 GiB |
| 12,288 | 29 | 3 | 4.45 GiB |

planner 依实际 `(B,S,dtype,attention backend)` 决定静态层计划并按 bucket 缓存，
不在逐层 forward 查询空闲显存或随意改图。若全轻量仍超预算，应明确报告该
micro-batch 不可行；可提议更小 batch 或 ckpt2，不能静默调整训练/续训游标。
首版仍限制 standard Dense + standard 残差、已验证的融合 attention 路径；
未知 backend/nonzero dropout 不沿用这套字节模型。

## 6. 实施顺序与验收

1. **先补容量预检**：按 checkpoint、实际桶形状、梯度累积阶段和实际 allocator
   limit 估算。自动预算使用新曲线；显式预算也显示下界及余量，明显超限时提前
   拒绝。优化器峰值和激活峰值分阶段处理，避免不合理的重复相加或漏算。
2. **实现预算版 ckpt4**：先复用原生 SAC 的轻量/完整层组合，不改变 attention
   dispatch、loss 或精度。模型配置、CLI、WebUI、relay、autotune 与 profile
   指纹一并接入；默认模板和正在使用的面板参数不自动改动。
3. **独立容量校准**：至少覆盖 1×3072、1×4096、2×3072、2×4096、4×3072、
   3×4096；包含优化器状态、已有梯度的累积 forward、warm/cold compile、桶切换。
   每种策略单独搜索显存边界。先用不更新参数的状态/梯度预分配探针筛选，
   不把仅有 forward/backward 的结果当作完整训练容量。
4. **比较端到端吞吐与质量**：ckpt2、ckpt3 小 batch、ckpt4 中/大 batch；同初始
   权重、真实批次与可比有效 tokens/update；至少三个稳定窗口，加独立复测。
   记录 useful tokens/s、同口径 useful MFU、allocated/reserved 相对 ckpt2
   的百分比、训练/验证 loss。FP8 需要容差检查，不声称 bitwise 一致。
5. **有证据再写算子**：若 replay 的 gate/up 与 SwiGLU 中间量仍主导，优先试
   gate/up 重算与 epilogue 融合；保证 TorchAO 投影、BF16 舍入点和参数梯度。
   RMSNorm/SwiGLU 的便宜中间量在 ckpt3 本来就已重算，单独再融合不能释放
   这里保留的 gate/up 输出。loss-only 的 chunked head 可作独立可选项，
   不改变依赖 logits 的训练器接口，也不将有损 FP8 激活保存冒充等价 checkpoint。

真实训练试跑必须通过 trainer 的 save-and-pause 收尾，等待 accumulated optimizer
update、保存并核对最新权重/optimizer/scaler/data cursor。当前旧 autotune 的
`TrialComplete`/取消流程还需先补齐保存再退出的生命周期，不能继续用异常退出
或强杀丢弃试跑进度作为上述验收方法。

## 复核产物

- 原始运行日志：`trainer/logs/train_pretrain_20261023_190918.log`。
- 静态计算脚本：`tmp/audit_ckpt3_graph.py`。
- 去重存储审计：`experiments/results/ckpt3/oom_graph_audit.json`。
- 既有 2k 小 batch 测量：`experiments/results/ckpt3/ckpt{2,3}_{bf16,fp8}.json`。
- 当前模型与固定状态在 meta device 上独立复算；模型参数、persistent estimator
  分别得到 1,010,277,888 与 8,161,714,176 bytes。
