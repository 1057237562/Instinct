# V2 Dense：原生 cuBLAS 与 PyTorch 的剩余吞吐差距

2026-10-27，本机 RTX 5070 Ti 16 GiB。最新原生生产窗口为 **11,398 TPS**；达到 14,000 TPS，需要平均每微步节省 **193.37 ms**。本轮完成同输入的固定权重对照，定位到短桶 attention、融合范围、无效重算及运行时提交间隙。没有据此宣称已经达到 14,000 TPS。

## 本轮对照口径

- V2 Dense：32 层、H=1536、FFN=5504、Q/KV=16/4、head_dim=96、词表 6400，1,010,277,888 参数。
- 两边读取同一个 `identical_batches.npz`，来自生产 `pretrain_coder_l2l3_24b.parquet` 的原生 packed cache。token IDs、labels、document IDs 完全相同；短桶 `4×3072` 有 12,274 个有效预测目标，长桶 `3×4096` 有 11,261 个。
- BF16 参数、mantissa、BF16 momentum、tensorwise FP8 auto filter；保留真实优化器状态和梯度缓冲。**固定权重前后向，不执行更新，不计 loader、初始化、编译和保存。** 两边初始化 RNG 实现不同，本轮不用于数值一致性判断。
- 原生采用当前已验证融合、短桶 Triton、长桶 cuDNN、cuBLASLt tune；ckpt4 短/长缓存预算 1024/1280 MiB。PyTorch 2.13.0+cu132，compile max-autotune-no-cudagraphs、ATEN/cuBLAS GEMM、Flex TMA backward；缓存预算 1792 MiB。保留各自配置，不能称为完全相同执行策略的 A/B。Torch 生成代码的 FP8 forward 使用 fast accumulation，原生该选项关闭。
- 原生预热每桶 2 次、无 profiler 测量每桶 2 次，再各采集 1 次 CUPTI；Torch 预热每桶 3 次、测量每桶 4 次，再各采集 1 次 profiler。独占 GPU、顺序运行，无并行训练。所有探针正常退出，没有训练更新需要保存。
- 原生 CUPTI 采集实际 kernel/memcpy/memset 时间，两个 trace 的 dropped records 均为 0。Torch 只统计 Chrome trace 的 GPU activity，未重复累加 CPU `aten` 事件。墙钟中位数来自无 profiler 窗口。

## 新剖面

单位：ms/微步。Attention 行包含前向、重算、反向和相关 Q/K norm/RoPE；这样可减少 Torch 跨算子融合造成的分类偏差。融合 kernel 仍无法拆成精确的单算子时间。

| 项目 | Torch 4×3072 | 原生 4×3072 | Torch 3×4096 | 原生 3×4096 |
|---|---:|---:|---:|---:|
| GEMM | 330.3 | 338.7 | 340.9 | 348.6 |
| Attention 及 Q/K norm、RoPE | 263.6 | 415.5 | 348.1 | 328.7 |
| 其他 kernel | 195.7 | 260.9 | 192.2 | 263.4 |
| memcpy/memset | 0.04 | 0.04 | 0.05 | 2.62 |
| GPU activity 合计 | 789.6 | 1015.1 | 881.2 | 943.3 |
| GPU activity 时间跨度内的空档 | 9.7 | 51.6 | 14.6 | 95.8 |
| 无 profiler 前后向墙钟中位数 | **796.3** | **1083.4** | **889.8** | **1041.0** |
| kernel 数量 | 4234 | 6645 | 4297 | 7034 |

空档是 activity 时间跨度减去设备活动区间的并集，包含提交、依赖和运行时等待，并非全部都是 Python 开销，也不能全部当作可回收收益。原生 profile 窗口墙钟为 1085.7/1059.1 ms；尤其长桶有额外采集开销。表中最后一行时间因此采用无 profiler 数据。

按最近原生生产窗口的桶比例（短 101、长 27）加权，短桶 attention 主导差距。Attention 相关差额约 **116 ms**、其他 kernel 约 **66 ms**、trace 空档约 **50 ms**；GEMM 约 **8 ms**。这些是两批固定样本的加权定位指标，不是对全部生产样本逐步测得的差额。

## 具体原因和可改范围

**短桶 attention 是当前第一优先项。** 原生短桶的 attention 前向（含重算）178.0 ms，反向核心/统计183.7 ms，独立 Q/K norm/RoPE53.8 ms。全部 attention 前向同样执行64次，已经没有“原生多重算若干层”这个解释。原生前向每线程255个寄存器，反向Q/KV为252/254，当前trace未报告local memory；资源压力明显，不能只靠增大tile。应优化 packed block 调度、Q/K/V 布局与载入、反向分块及 Q/K norm/RoPE 的融合。原生目前不支持 TensorDescriptor/TMA 的完整启动 ABI，接入 TMA 需要明确实现和验证。

**长桶 cuDNN 路径应保留。** 长桶两边均缓存全部32层 O/LSE；原生 attention 及相关 norm/RoPE 合计比本次 Torch 快19.4 ms。原生长桶仍慢151 ms，主要来自其他操作和运行时间隙，不能继续用旧版长桶 attention 重算差异解释。

**checkpoint replay 重放了可避免的 down projection。** 原生 `Engine.checkpoint` 完整调用 `regenerated = fn(leaf)`，每微步在反向重放32次 down projection，短/长桶分别20.82/20.70 ms。核对本次短桶使用的 Torch 编译反向图，前向角色 FP8 GEMM 数为 Q/O64次、gate/up56次、down仅1次，和 GPU trace 数量对应。原生应为 Dense block 构建专用 replay/VJP，只重建 backward 所需输入，省掉终端 down GEMM 与不再需要的残差输出。现有 Linear VJP 使用输入、权重和上游梯度，不需要重放的 down 输出；实现时仍需验证完整 block 梯度、BF16舍入、累积和显存生命周期。仅以31次多余 down GEMM 估算，设备时间约20 ms；尚未实现，不是已兑现收益。

**已融合 amax，但仍未达到 Torch 的跨算子融合范围。** 原生每桶 SwiGLU及amax约59 ms、E5M2双布局量化约35 ms、梯度合并约29 ms、残差binary约28 ms、pack约14 ms。它们是当前成本，不能逐项全部当作可省时间。Torch 生成的一个 FFN kernel 同时输出 SwiGLU激活转置FP8，以及gate/up梯度的正向和转置FP8，共5路结果；原生仍写出中间BF16梯度，再由独立kernel量化。可用“第一遍求amax、第二遍重建值并直接写FP8布局”的专用管线减少中间显存读写，同时保留现有舍入语义。残差加法、RMSNorm、梯度合并和量化准备也应以数据流为单位融合。单纯扩大block不会消除这些中间张量。

**运行时提交需要单独优化。** 原生短/长 trace 分别约6千次 `cuLaunchKernel`，约2千次 `cuMemAllocAsync`，约1.58万次 `cuCtxSetCurrent`；这是API计数，不代表所有调用都在关键路径。应减少重复context设置，复用固定桶的短期内存和元数据，批量提交或按block捕获稳定子图。长桶metadata上传中的同步也是检查点。必须避免恢复此前导致显存压力的无限持久缓冲；验收同时看无 profiler 墙钟和峰值显存。

## 缓存、优化器和数据

| 桶 | 当前原生ckpt4 | Torch ckpt4 | 原生补齐gate/up的估计代价 |
|---|---|---|---|
| 4×3072 | 全部K/V + 1层gate/up | 全部K/V + 4层gate/up | 额外774 MiB；少6次GEMM，按当前耗时约3.96 ms |
| 3×4096 | 全部O/LSE，不留gate/up | 全部O/LSE + 2层gate/up | 额外516 MiB；少4次GEMM，约2.63 ms |

每层gate/up为258 MiB。以本次各桶native peak live约12.52/12.83 GiB作分母，上述直接缓存增量约 **6.0%/3.9%**，只是忽略生命周期变化的估算，未实测新增缓存后的峰值。优先去掉无效重算、缩短中间张量生命周期，再重新分配缓存，比直接扩到1792 MiB更有依据。

本轮没有执行优化器更新，因此不能给出新版完整Muon阶段与Torch的严格差额。之前9个真实shape的cuBLAS/Triton Muon GEMM验证已通过，但单GEMM加速不等于完整更新加速；本次约193 ms目标不预支Muon收益。

历史Torch生产窗口已保存并核验，吞吐14,150.9 TPS，loader等待4.153 ms/微步，占0.492%。最新原生生产窗口11,398.3 TPS使用就绪Rust packed cache；本轮固定样本不计loader。早先Rust冷块构建49–51秒也不属于本轮重测结果。最终验收仍需真实生产loader、跨块预取、真实累积8步及Muon更新，不能拿这里的固定权重前后向TPS替代生产TPS。

## 14,000 TPS 的验收预算

最近完整原生生产测量：128微步、1,518,127有效目标、133.1889443秒。即平均11,860.37目标/微步，当前1040.54 ms；14,000 TPS要求低于847.17 ms，至少减少193.37 ms（时间降低18.58%，吞吐提高22.83%）。

一个有当前对照支撑、但**尚未兑现**的工程预算是：短桶attention相关从415.5降至280 ms（Torch为263.6），按桶比例约省107 ms；其他kernel降至约210 ms，约省51 ms；去掉终端down重算约20 ms；加权trace空档从60.9压至35 ms，约省26 ms。合计约204 ms，按同一生产目标密度外推约14.18k TPS。样本内容、采集开销、依赖重叠和优化器都可能改变结果，必须以新生产窗口验证，不能当作性能承诺。

实现优先级：专用Dense replay与FFN数据流融合 → 短桶attention → 提交和内存复用；保留已有长桶cuDNN优势，再评估Muon和剩余GEMM选型。两个版本的主要GEMM本来就都走cuBLAS，直接使用cuBLAS本身不能证明18k TPS上限或保证大幅领先Torch。

## 可复现材料

- [同输入汇总](../../experiments/results/cublas/comparison_20261027/comparison.json) 与 [目标预算](../../experiments/results/cublas/comparison_20261027/gap_budget.json)。
- [原生报告](../../experiments/results/cublas/comparison_20261027/native_latest.json)、[Torch报告](../../experiments/results/cublas/comparison_20261027/torch_actual_recipe.json)，同目录保留全部GPU trace、日志及共享输入。
- 原生采集：`experiments/bench_native_recipe.py --profile-cupti --production-samples --export-samples ...`；CUPTI C ABI与Python采集器位于`experiments/cupti_trace/`。
- Torch对照：`experiments/profile_torch_reference.py --samples ... --output ... --profile`；仅从主工作区读取模型代码，未修改其训练实现。
- 重算分析表：`python experiments/analyze_native_torch.py --directory experiments/results/cublas/comparison_20261027`。
- 本轮Python语法检查和`git diff --check`通过；性能证据以上述完成的GPU探针为准。
