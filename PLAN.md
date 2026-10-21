# 训练性能优化计划

## 2026-10-14：Config WebUI 自动调优集成

侧栏新增「全局自动调优 · 二分搜索」，直接复用 `trainer/autotune.py` 和现有
训练命令构造器。支持当前模型/训练参数快照、审计与 provenance、有限实验预算、
可选 MFU / 吞吐验收、自定义性能搜索空间、缓存复用、后台状态/日志、停止并回收
所属子进程树、profile 导出与生产应用。新增手动 bucket token budget（0=自动）。
生产命令省略 profile 管理的性能参数，并提前提示代码/明确模型与训练参数冲突；
完整硬件/训练合约/语料 SHA-256 继续由 trainer 校验。流水线仅 Pretrain 阶段
携带 profile，SFT 独立配置。实验与该面板的正式训练/流水线互斥。

验收：116 项相关测试通过，包括 Streamlit AppTest、共享 CLI、续训、路由、
流水线、二分搜索及取消时真实父/子进程回收。本机实际通过 WebUI 启动 CLI
搜索计划，退出 0，状态恢复/页面重渲染通过。又通过 WebUI 的真实命令构造器
加载已有 26,421 token/s 档案，初始化真实模型与优化器，确认全部选中设置生效，
在首步训练和保存前退出 0。证据在
`experiments/results/config_webui_autotune/verification.json`、
`experiments/results/config_webui_autotune/profile_load/verification.json`。
本次未重新测训练吞吐、未保存正式权重；所有实验 Python 进程均已退出。
既有训练源码和计算路径未修改，原有效 profile 的训练源码指纹保持兼容。

## 2026-10-14：生产训练自动调优与二分搜索

新增 `trainer/autotune.py`，在目标训练机上用独立子进程测量真实生产循环，自动
选择通过训练/验证门槛的性能参数，输出可由 `train_pretrain.py` 加载的档案。
正常模式没有固定 tokens/s 目标；25,081 只是本次本机验收参考。
用法和边界见 `docs/training_autotune.md`。

### 参数搜索

- token budget 按每种 checkpoint/内核/workspace 策略独立二分，只有 OOM 或
  驱动预算越界才排除上半区；编译失败保留其他容量候选。发现非单调证据时取消
  剪枝。边界附近再比较较小 batch 的吞吐，不假设最大 batch 最快。
- checkpoint、Muon workspace/分组、CPU 线程、compile、MoE forward/dW tile
  和每 SM 调度规模采用有限候选对照。优先比较能复用 packing 缓存的内核。
- 默认 batch 候选由基线预算按比例产生，允许超出原本机 16,384 经验上限；
  更换硬件/软件/源码/模型/语料会改变指纹，需要重新调优。
- gradient checkpoint 始终开启；精度、模型、优化器数学、LR 与语料不参与
  性能搜索。正式 resume 保留已保存的 batch 预算及游标。

### 本机实际验收

RTX 5070 Ti / Windows，完整 V1 MoE normalized top-2、FP32 参数/BF16 计算、
Muon、compile、FFN checkpoint、已审计 pretrain_coder_12b Parquet/Arrow、
真实 packing 和流式预取。默认搜索空间，无自定义候选、无目标吞吐参数；
本次限制为 4 次试跑，每次 20 步预热 + 80 步计时，共 400 次独立试验更新。

| 自动候选 | 有效 tokens/s | ms/step | useful BF16 MFU |
|---|---:|---:|---:|
| 原内核基线 | 26,010.821 | 595.288 | 38.178% |
| forward tile N=128 | 26,412.368 | 586.238 | 38.767% |
| forward/dW tile N=128 | **26,420.710** | **586.053** | **38.779%** |
| 胜出配置独立复测 | 26,415.477 | 586.169 | 38.772% |

按用户要求采用最佳有效实测为主指标；同轮相对基线提升 **1.576%**。最高结果
超过 25,081，但历史与当前的系统负载/预热条件不同，不能将所有差异都归因于
内核改动。训练初始化与所有 100 个 batch 的 SHA 均与历史 25,081 运行一致。
当前仍未达到 40% 生产 MFU，不把固定张量结果当作生产吞吐。

选出的档案使用 token budget=16,384（实际 6×2720 / 4×4096）、checkpoint=1、
inner checkpoint=0、Muon workspace=256 MiB/组大小16、CPU线程1，以及
forward/dW tile=64,128,32,4,3。参数由搜索结果应用，没有将该 tile 硬编码为
其他机器的生产默认值。本次 4 次上限只覆盖内核候选与复测，尚未在此运行中
完成真实 GPU 容量二分或完整策略扫描，不宣称全局最优。

胜出运行相对同轮基线：最大 20 步训练 loss 均值差 **0.02680%**，验证 NLL
相对差 **+0.01038%**；独立复测分别为 **0.02271% / +0.02027%**。均通过
2% 训练均值/1% 验证 NLL 非退化门槛，公共验证为第二源块固定16条文本。
这是短训验收，不能代替长训收敛结论。

计时包含真实 loader、前反向、LR/裁剪/optimizer/zero_grad/日志，排除首次
准备、编译、预热、验证和存盘。首次后台预取在计时前等待完成并单独记录，
保证候选暖缓存条件一致；普通训练不执行该校准等待。MFU 沿用 87.9 TFLOPS
dense BF16 分母、187,992,576 active 参数及有效 token/attention pairs，
不包含重算、padding 和优化器 FLOPs。计时窗口 reserved 峰值约13.14 GiB，
DXGI 两个边界采样均未超过预算；这不是整个窗口无换页的证明。

回归 **93 passed / 2 skipped**，包括实际 GPU forward/dW tile 的逐位比较、
checkpoint 梯度、packing/streaming、档案兼容性/恢复保护和二分边界。
64 候选的模拟显存边界测试最多8次探测（含邻近吞吐比较）。生成的完整生产
命令已加载真实模型/优化器并验证参数应用，随后在训练/存盘前退出。

本次可用档案与完整命令：
`experiments/results/autotune_default/05f5185e84de9778d84cb44b/profile.json`、
同目录 `launch.json`。测量/质量原始记录在同目录 `search.json` 及子目录。
验证日志为 `experiments/results/autotune_regression.log`、
`autotune_default_profile_verify.log`。早期 smoke/target 档案因后续源码指纹
变化已过期，仅保留诊断证据。相同配置重跑已命中全部4份候选/复测缓存，没有
再次训练，见 `autotune_default_cache_reuse.log`。汇总验收记录在
`experiments/results/autotune_verification.json`。本轮没有保存正式模型权重，
也没有修改正式断点；所有试验进程已退出，显存回到桌面约1 GiB占用。

## 2026-10-13 晚：selective checkpoint 与自动 batch 的本机实验

继续使用本机 Windows / RTX 5070 Ti、完整 InstinctV1MoE top-2 normalized、
FP32 参数/BF16 计算、Muon、compile、checkpoint=1，以及已审计的真实
pretrain_coder_12b Arrow/流式预取。没有关闭 gradient checkpoint。

**生产主指标：最佳实测 36.81% MFU，未达到 40%。** 按用户指定，以较高的一次
作为当前本机能力基准，较慢记录注明可能存在其他程序的 GPU 干扰；本轮没有同步
采集其他进程负载，尚不能确认原因。最后两次各 100 更新，前 20
步预热、后 80 步计时；同初始化和同训练 batch SHA。两次分别为 617.350 /
654.362 ms、25,081 / 23,663 有效 tokens/s、36.813% / 34.731% MFU。
合并统计 35.742% MFU 仅保留为诊断记录，不作为主指标。两次耗时相差约 6%，
最佳实测不代表任何后台负载下的吞吐保证。阶段约为前向 134–140 ms、反向 373–395 ms、优化器
109–118 ms；反向仍占约 60%。当前瓶颈主要在模型 GPU 计算。

### 生产代码选择

- 自动参数继续使用普通 FFN checkpoint；当前本机已验证配置将显存曲线算出的
  token 预算再限制到 **16,384**。真实首块得到 **6×2720 / 4×4096** 两种 batch。
  更紧的显存预算仍会降低 batch。显式 token 预算和保存的 resume 预算优先，
  不在恢复途中偷偷改变采样游标。该吞吐上限来自离线对照，不是在线全局最优搜索。
- `trainer/windows_gpu_memory.py` 只读查询 DXGI 进程预算，唯一匹配 CUDA 设备
  名称时用于 CLI allocator 上限；同型号多卡、查询失败和非 Windows 保留回退。
  将该上限同时传入保守/校准两种 batch 曲线，保留一次 0.5 GiB 安全余量。
  当前进程启动预算 14.87 GiB，训练窗口约 14.35 GiB，进程使用约 13.44 GiB；
  边界采样未超过预算。没有采集换页事件，不能断言 WDDM 分页是本轮慢的原因。
- pretrain 在形成 loss 后提前释放完整模型输出，让 autograd 按需释放 logits
  等中间量；数学语义、loss、辅助 loss 和 recurrent-step 日志不变。
- 增加可选 `INSTINCT_MOE_INNER_CHECKPOINT=2`：反向重算 gate/up/SwiGLU，
  并用算子自有的 down dX 缓冲保存重建激活；不会覆盖上游传入梯度。保留原有
  FP32 权重/BF16 舍入。此配置真实吞吐未胜出，**默认仍为 0**。
- 增加可选 `INSTINCT_MUON_PREALLOCATE=1`，在首次激活分配前预留零 momentum；
  不执行 step、不修改参数、保留恢复状态及 grad=None 语义。它降低某些候选的
  碎片，但尚无普通 FFN 路径的稳定收益证据，**默认仍为 0**。

### checkpoint 与 batch 对照

固定输入的完整 32 层计算基准（10 步预热、2×10 步计时；不是生产 MFU）：

| 方案 | batch | ms/step | MFU | allocated / reserved GiB |
|---|---:|---:|---:|---:|
| 普通 FFN checkpoint | 4×4096 | 663.144 | 43.024% | 11.76 / 13.26 |
| PyTorch SAC 保存 down | 4×4096 | 739.068 | 38.605% | 12.76 / 13.94 |
| 算子内 checkpoint=2，初版 | 4×4096 | 634.916 | 44.937% | 13.25 / 14.74 |
| 普通 FFN、预留 momentum、提前释放输出 | 5×4096 | 874.077 | 40.802% | 13.05 / 13.79 |
| 整层 checkpoint=2、预留 momentum | 6×4096 | 1397.298 | 30.628% | 9.56 / 11.14 |

真实生产 inner=2 / budget=16384：缓冲复用后为 787.711 ms / 28.852% MFU；
再预留 momentum 后为 672.784 ms / 33.780%。固定张量的优势没有转移到真实
两种长度的训练，不能自动采用。较大的普通 FFN budget=18661 + 预留 momentum
为 808.028 ms，也未胜出。该组运行分时完成，有系统漂移，不是统计显著性证明。

明确记录失败项：inner=2 初版及 roundup_power2_divisions:16 在 104 MiB 申请
处 OOM，未进入有效计时；复用算子自有缓冲后解除该 OOM。cudaMallocAsync 的
小型图捕获探针通过，但真实 Muon Triton NS 图捕获出现 pointer access 错误，
未采用。当前生产使用 native allocator，不启用上述实验 allocator 配置。

### 正确性、计时口径与清理

两次最终运行的初始化/batch SHA 一致，最大 20 步训练均值差 0.018584%，
验证 NLL 相对差 −0.012230%。相对早期较小 batch 的公共验证 NLL 为
5.726638 → 5.455353；通过用户授权的扩大 batch 短训验收，但 100 更新消耗的
训练 token 数从 1,064,081 增至 1,556,942，这不是等 token 的收敛等价证明。
公共验证为第二块固定 32 序列，训练停留在第一块；保持原 eager BF16 评估器。

最终 peak allocated / reserved 为 **11.49 / 13.05 GiB**；无 allocator retry
或 OOM。继续沿用 187,992,576 active 参数、87.9 TFLOPS BF16 dense 分母和
原 useful-FLOP 公式，不计重算/优化器/padding FLOPs，不含首次编译、初始数据
准备、验证与存盘；不能把固定输入 43–45% 当作真实生产结果。

回归：checkpoint、输入/router/专家梯度与上游梯度不变、Muon 更新/恢复、显存
上限和 packing/resume **70 passed / 2 skipped**；两项旧的可选慢集成测试未跑。
已额外完成两次真实生产 100 更新，最终生产源码 SHA 与报告一致。

产物在 `experiments/results/mfu_20261013/`：
`checkpoint_batch_summary.json`、`checkpoint_kernel_summary.json`、
`production_auto_checkpoint_final_100.json`、`auto_checkpoint_final_mfu.json`、
`auto_checkpoint_final_quality.json`。trace 的 kernel 汇总避免重复累计 profiler
父子算子。实验无正式权重保存，所有实验训练进程已退出。磁盘垃圾删除被工具的
自动审批检查拒绝（仅返回 `blocked by policy`，没有进一步原因），缩小为核对过
的明确路径后仍被拒绝，因此没有绕过检查。暂留清单：`tmp/handle_diag/`、
`tmp/pyspy_diag/`、结果目录中的 `production_ffn_trace.trace.json`、
`inner2_async_gpu.csv` 与空的 `isolated_weights/`。小型结果/失败日志、可复用
packing/compile 缓存与正式训练资产继续保留。

环境限制：仓库根目录发生 Windows sharing lock，精确文件访问正常，但目录
枚举/Git worktree 检查失败。临时 launcher 只为 Python FileFinder 补充已知包
目录名，使用原文件、原 cwd 和原 CLI 完成实验；未关闭别的应用或强制关闭句柄。
目录锁解除前保留 `tmp/run_locked_root.py` 以便复现，不能将此当作生产性能优化。

## 2026-10-13：真实 top-2 生产循环与 FP8 调查

本机 RTX 5070 Ti，InstinctV1MoE 32 层、hidden=512、8 experts、top-2、
norm_topk_prob=True；FP32 主参数、BF16 计算、Muon、compile、checkpoint=1。
直接执行生产 pretrain CLI，使用经 SHA-256 校验的 pretrain_coder_12b.parquet、
原 Arrow loader、自动 packing、流式预取与原采样顺序。窗口含 20 个 2×4096
和 60 个 4×2928 batch，并非每一步都是 2×4096。

100 步独立进程配对（前 20 步预热，后 80 步计时）：

| 版本 | ms/step | 有效 tokens/s | 同口径 MFU |
|---|---:|---:|---:|
| 本轮原路径 | 573.477 | 18,170 | 26.738% |
| 已验证的生产默认优化 | 444.821 | 23,426 | 34.471% |

分子为 `6*N_active*useful_tokens + 12*layers*q_heads*head_dim*causal_pairs`，
top-2 活跃参数 187,992,576，分母沿用 BF16/FP32 accumulate dense 87.9 TFLOPS。
排除 padding、checkpoint 重算和优化器 FLOPs；计时不含首次编译/初始数据准备、
验证和模型存盘。这是短窗口生产训练吞吐，不是全语料长跑吞吐，尚未达到 40%。

主要改动：稳定路由计数/排列与 gather/combine 融合；Windows SM120 的单文档
训练 attention 改用 causal Flex（原 memory-efficient SDPA 的反向非常慢）；
packed Flex autotune；专家权重梯度直接写入参数所需布局；本机训练 CPU
线程数默认 1（模型初始化后才设置）；本机 FP32 Muon workspace 默认 256 MiB。
不改 Muon 数学迭代、LR、batch、数据顺序或 checkpoint 模式。

用户明确允许按训练/验证指标等价验收 attention 的浮点误差。预先设定门槛：
每 20 步平均训练 loss 相对差 ≤2%，最终公共评估器验证 NLL 相对差 ≤1%。
两侧初始化参数和训练 batch SHA 完全一致；第二数据块固定 32 个序列，
101,668 个验证 token，均使用 eager BF16 / 历史 causal SDPA 公共评估器。
实测最大训练窗口差 0.033716%，验证 NLL 5.727988 → 5.726638（−0.023568%），
通过短训门槛；不以此证明完整收敛或最终任务质量。

产物：`experiments/results/mfu_20261013/production_reference_100.json`、
`production_final_100.json`、`final_quality_comparison.json`、`final_mfu.json`。

新增回退开关：`INSTINCT_MOE_FUSED_ROUTING=0`、
`INSTINCT_CAUSAL_ATTENTION_BACKEND=sdpa`、`INSTINCT_FLEX_AUTOTUNE=0`、
`INSTINCT_MOE_TRANSPOSED_DW=0`、`INSTINCT_TRAINING_CPU_THREADS=0`（保留平台线程数）、
`INSTINCT_MUON_WORKSPACE_MB=64`。实验性 `INSTINCT_MOE_SELECTIVE_CHECKPOINT`、
`INSTINCT_MOE_DIRECT_WEIGHTS`、`INSTINCT_MOE_INNER_CHECKPOINT` 默认关闭。

### 消费级 FP8：能力与当前接入限制

官方名称分别是 FP8、MXFP8 和 NVFP4，不应把 NVFP4 推理结果当成 FP8 全参
预训练结果。2026-10-13 查阅：

- [Unsloth FP8 RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning)
  支持 RTX 40/50，主要证据是 RL 推理/LoRA，1.4× 指 RL inference。
- [Unsloth NVFP4](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/nvfp4.md)
  展示 W4A4 与部分 FP8/BF16 层混合的量化推理，并不是从零全参预训练方案。
- [Unsloth FP8 源码](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fp8.py)
  的 FP8GroupedLinear 训练路径对冻结 FP8 权重先反量化，再执行可微 BF16 bmm；
  其 backward 不计算基础权重梯度。不能直接替换本项目可训练专家。
- [Transformer Engine 支持表](https://nvidia.github.io/TransformerEngine/support_matrix.html)
  （2.21 开发版文档）列出 SM120 普通 FP8 支持，MXFP8 尚不支持；NVFP4 部分
  支持，训练需禁用 RHT/随机舍入。[安装前提](https://nvidia.github.io/TransformerEngine/installation.html)
  仍是 Linux x86_64，不能假定当前原生 Windows 可直接安装后接管训练。

本地探针 `experiments/probe_fp8_training.py` 实测 FP32 主参数、BF16 autocast、
compiled TorchAO tensorwise FP8 的前向、输入梯度和权重梯度；profiler 确认
`aten::_scaled_mm` / `nvjet_sm120_*mma*`，确实执行 FP8 内核。rowwise 配方
报设备不支持。当前生产自动过滤转换了 512 个专家 Linear，但 grouped dispatch
直接读取它们的权重执行 BF16 GEMM；转换层数不等于运行时 FP8 覆盖率，生产
日志已明确打印此限制。不能以开关已打开来宣称 FP8 提速。

探针结果（单专家投影，含前向/dX/dW/动态量化，ms；随机输入，不是生产 MFU）：

| rows×in→out | BF16 普通调用 | FP8 普通调用 | BF16 CUDA Graph | FP8 CUDA Graph |
|---|---:|---:|---:|---:|
| 2048×512→1664 | 0.2895 | 0.3981 | 0.1283 | 0.0895 |
| 2928×512→1664 | 0.2919 | 0.4185 | 0.1765 | 0.1645 |
| 2048×1664→512 | 0.3014 | 0.4066 | 0.1375 | 0.0891 |
| 2928×1664→512 | 0.2852 | 0.4219 | 0.1927 | 0.1624 |

Graph 用固定形状/地址消除 Python launch 间隙，仍包含动态量化；这展示 1.07–1.54×
算子加速潜力，不能推算为端到端速度。普通调用受调度/量化开销影响反而更慢。
随机探针输出 relative-L2 约 3.7%，dX/dW 约 5.9%，没有做 FP8 全模型训练质量
验收，生产专家仍为 BF16。原始数据 `fp8_training_probe.json`。

### 算子内 checkpoint 实验（默认关闭）

`INSTINCT_MOE_INNER_CHECKPOINT=1`：gate/up GEMM epilogue 融合 SwiGLU；反向
重算 gate/up 并直接产生导数，保留 FP32 原权重，避免整个 FFN 的重复路由/down
投影。完整 MoE 的输出、辅助 loss、router/专家/输入梯度在均衡及空专家测试
逐位相同；4 项新增相关测试通过。100 步生产短训为 424.258 ms/step，最大
20 步训练均值差 0.013807%，公共验证 NLL 差 −0.004227%，通过既定门槛。
显存峰值增加至约 13.56 GiB，因此仍保持 opt-in；默认更省显存的版本另行用于
batch 扩容实验。报告 `production_inner_100.json` / `inner_quality_comparison.json`。

FP8 后续采用条件：grouped ragged forward/dX/dW 真实覆盖、量化/缩放/布局转换
与路由融合、FP32 主权重和优化器状态保留；router/norm/loss 保持高精度。
先含全部转换开销的算子基准，再同初始化/数据的生产短训和公共验证集验收。
跨精度优先比较 tokens/s、step time 和质量；沿用 BF16 峰值算的数只能标为
BF16 等效 MFU，不能作为 FP8 硬件利用率。

## 2026-10-12 实施结果：保留 checkpoint，本机计算 MFU 21.77%

生产路径已修改；下方“调查方案”保留作为实施前记录。固定 RTX 5070 Ti、32 层
MoE、4×2048、4 个隔离文档/行、FP32 参数/BF16 计算、Flex attention、Muon、
`use_grad_checkpoint=1`，有用 FLOP 分子和 87.9 TFLOPS 分母均沿用原口径。
20 步预热后统计 4×50 步，最终四个窗口 297.34 / 290.65 / 291.37 / 297.37 ms，
**中位 294.36 ms、27,830 tokens/s、估算 MFU 21.77%**；每个窗口均超过 20%。
这段是包含前向、反向、clip、完整优化器和 zero_grad 的计算基准，真实语料生产
循环另行记录，不将这段数字伪装成包含数据预处理/存盘的端到端吞吐。

原版 A1/A2 各统计 200 步：445.61 / 448.85 ms，约 14.38% / 14.28%；最终版
相对两次基线平均值吞吐提升约 **51.9%**。最终优化器阶段约 116.0 ms，前向
54.3 ms、反向 123.8 ms，峰值 allocated 8.52 GiB。中间版本的 A/B/A 为
19.98%，明确未算达标；最终增加 SwiGLU 融合后才超过门槛。最终版并未再次插入
一整组 A/B/A，使用同一会话已完成的两个原版基线比较，保留这个测量限制。

**真实语料生产循环也通过目标。** `bench_pretrain_pipeline.py` SHA-256 校验
`pretrain_coder_12b_seg2_identity_clean.parquet` 及现有身份审计后，取前 2048 行，
保留 1766 个长度 ≤2048 的完整样本，使用生产 best-fit packing 得到 32 个
4×2048 回放 batch。两侧的 corpus/replay SHA-256、有效 token 数和文档 attention
pairs 均一致；直接调用生产 `train_pretrain.train_epoch`，包括 H2D、原 cosine LR、
clip/optimizer、pause 检查、正常日志及 timing profiler。预热 20 步后连续 200 步：

| 实际训练循环（warm packing cache） | 原版 | 生产优化版 |
|---|---:|---:|
| wall ms/step | 480.97 | 335.17 |
| useful tokens/s | 16,906 | 24,261 |
| 同口径估算 MFU | 15.50% | **22.24%** |

有效吞吐提升 **43.5%**。这不是完整 12B 语料长跑；预处理、首次编译、chunk 切换
和 checkpoint 存盘长尾不在 warm 窗口内。没有读取/覆盖正式模型或 resume 权重，
也没有启动远端训练。Profiler 每窗口重置峰值；初版 replay JSON 的末尾内存计数
已重新标为 `post_report_allocated_gib`，真实峰值使用原日志的 rounded window peak。

### 已进入生产的改动

- `trainer/muon_kernels.py`：Triton BF16 NS GEMM，保持原 norm、五次迭代、系数
  与中间 BF16 舍入；显式 FP32 FMA 复现原 baddbmm epilogue。利用 Gram 及其
  平方的对称性仅计算下三角 tile 并镜像；没有使用此前被否决的 Gram 递推算法。
  最后一轮直接生成参数布局，避免 tall expert 更新时的非连续读取。
- 融合 FP32 momentum/Nesterov/BF16 packing，以及 decay/参数更新，保留原
  lerp 分支和逐阶段舍入。LR 仍由原 Python 调度器每步更新。
- 优化器持有最多 8 个纯 NS CUDA Graph；输入逐次复制，图不捕获参数、梯度
  地址、momentum、LR 或 step。缓存不进入 state_dict，恢复时清空后重建；
  达到缓存上限或已在 capture 中则使用直接内核。
- `model/expert_weight_pack.py`：一次完成专家 FP32→BF16 堆叠；反向直接返回
  原 Linear 参数所需的 FP32 连续梯度，保留参数身份和 checkpoint keys。
- `model/swiglu_triton.py`：融合 SiLU×up 及其反向，显式保留 SiLU/梯度的 BF16
  中间舍入。PTX `div.rn.f32` 保留 subnormal，避免 Triton libdevice FTZ 与原版
  SiLU 的差异。保留不支持激活类型、dtype/layout 的原路径。

默认 `auto` 仅在已验证的 PyTorch 2.13 / CUDA Blackwell SM12、有 Triton 时
启用；Muon NS 自动路径限于已验证的 512/1664/128 矩阵形状。CPU、其他环境、
不匹配形状/布局仍走原算子。所有共用 `build_optimizer` 和 MoE 的 trainer
自动获得相应优化；没有改训练配置、checkpoint 模式、有效 batch 或参数精度。

完整回退开关（正常使用不需要设置，默认 auto）：

```powershell
$env:INSTINCT_MUON_NS_BACKEND='torch'
$env:INSTINCT_MUON_FUSED_UPDATES='0'
$env:INSTINCT_MOE_FUSED_PACK='0'
$env:INSTINCT_MOE_FUSED_SWIGLU='0'
```

NS 还接受 `triton`（直接内核）和 `triton_graph`（纯 NS 图）；其余开关接受
`0|1|auto`。显式 `1`/Triton 模式用于受控实验，自动模式才包含环境回退检查。

### 一致性验收

新增内核测试 28 passed，覆盖真实专家尺寸、随机/零/rank-one 梯度、动态 LR、
grad=None、momentum/Nesterov 分支、保存恢复、专家打包前后向、以及全部有限
BF16 输入的 SwiGLU 前后向。两层真实宽度的 25 步 gate 也逐位一致。
最终相关回归套件 **187 passed**，含新内核、Muon/native 对照、Gram 数学、
grouped GEMM、packed attention、FP8、profiler、精度与 packing 测试，启用了
`--run-slow`。23 条依赖弃用/编译提示不影响通过；没有把重叠测试次数相加。

原训练的普通模式存在非确定归约：同一编译模型重置权重/优化器的 reference
repeat，其首步 logits 相同，但梯度和后续轨迹仍可不同；独立进程还有编译选择
差异。因此没有将普通模式的轨迹差直接解释为新内核误差，也没有放宽误差门槛。

最终在同一个编译模型上开启确定性算法与 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，
从相同初始化分别运行 **reference / reference-repeat / candidate 各 100 步**。
使用完整 32 层、4×2048、8 个轮换输入 batch、变化 LR，保留 checkpoint=1。
三条轨迹的每步 loss、首步 logits/所有梯度/参数、最终所有参数及全部优化器
张量状态的 SHA-256 **全部一致**。确定性仅用于验证，没有改变生产默认设置。
这证明本次测试范围内的训练一致性，不承诺普通非确定模式任意长跑逐位相同。

核心产物均在 `experiments/results/mfu_20261012/`：`final_a1.json`、`final_a2.json`、
`final_swiglu_b.json`、`paired_deterministic_100.json`。复现：

```powershell
$env:PYTHONUTF8='1'
python experiments/bench_training_mfu.py --attention flex --fp8 tensorwise --compile-mode max-autotune-no-cudagraphs --peak-tflops 87.9 --steps 50 --repeats 4 --output experiments/results/mfu_reproduce.json
python experiments/validate_paired_training_kernels.py --deterministic --output experiments/results/mfu_parity.json
python -m pytest tests/test_muon_kernels.py -q
python experiments/bench_pretrain_pipeline.py --reference --output experiments/results/pipeline_reference.json
python experiments/bench_pretrain_pipeline.py --output experiments/results/pipeline_candidate.json
```

真实语料回放结果：`pipeline_reference.json`、`pipeline_candidate.json`；最终机器
可读汇总：`production_summary.json`，由 `experiments/summarize_production_mfu.py`
生成。`target_achieved=true` 同时要求四个计算窗口均 ≥20%、生产循环 ≥20%、
输入/工作量一致及确定性 100 步校验通过。

## 2026-10-12 更新：本机 20%，强制保留 gradient checkpointing

本节是最新目标与调查方案。固定 RTX 5070 Ti、32 层 MoE、FP32 参数/BF16
计算、Flex attention、原 Muon 数学、`use_grad_checkpoint=1`、4×2048 tokens、
4 个等长隔离文档；保持有效 batch、数据顺序、LR、loss、routing 和恢复语义。
**调查时尚未实现；最新实施结果见上节。** 不采用关闭 checkpoint、扩大有效 batch 或提高 FLOP 分子
来达标。现有 mode 1 已在 fused attention 路径上只 checkpoint FFN/MoE，
因此“改为 selective checkpoint”本身不是新的优化。

### 目标预算

- 以最终隔离 A/B/A 的约 422.74 ms / 15.16% 为基线，固定每步有用计算量
  5.633 TFLOP、BF16 dense 峰值 87.9 TFLOPS，20% 对应 **320.43 ms/step**、
  约 **25,566 tokens/s**。需要节省 102.31 ms（耗时 −24.2%，吞吐 +31.9%）。
- 当前 forward/backward 合计约 195 ms；优化器阶段约 228 ms，包含 clipping、
  Muon、AdamW、zero_grad。若只改优化器，必须将它降到约 125 ms（−45%）。
- 联合方案的工作预算为 forward/backward ≤150 ms、优化器阶段 ≤150 ms，
  合成完整步 ≤300 ms，为实际数据供给/同步等留约 20 ms 余量。**这是实验目标，
  不是性能预测**；真实 corpus 的文档长度与有效 token 变化时须重算分子及预算。
- 旧的 8×2048 实验虽为 19.89%，却扩大了有效 batch；不能算本轮训练等价结果。

### 新增调查：Muon CUDA Graph 探针

从真实模型的 meta 参数和当前优化器分组核算：Muon 更新 928 个矩阵，64 MiB
workspace 下分 205 个 chunk；每 chunk 的 5 次 NS 迭代各含 3 个矩阵乘，合计
3,075 次调用。NS 主要 GEMM 算术量约 **8.001 TFLOP/step**，其中专家矩阵占
7.731 TFLOP（96.6%）。该计算不属于有用模型 FLOPs，不能计入 MFU 分子。

新增 `experiments/probe_muon_graph.py`，测试完整 `_step_chunk` 的 eager / graph /
eager；每个窗口 20 次、每组 5 窗口。使用固定 LR、随机 FP32 梯度及原 BF16 NS，
并对每个 shape 更换三次梯度，比较下一步参数和 momentum。10 个 shape/batch
组合均逐位一致；仅证明探针，不替代真实训练 gate。

最终一次 A/B/A 按真实 chunk 数加权的耗时为 **223.91 / 206.76 / 224.14 ms**，
基线漂移约 0.11%，Graph 约节省 **17.26 ms / 7.7%**。这只是多个 chunk 微基准
的加权估算，尚不含全优化器调度、梯度 staging、动态 LR、clip、AdamW 和 zero_grad，
也没有证明全图集成的显存开销。不能声称完整训练达到这个收益。

主力矩阵每批 4 个时：1664×512 的 eager/graph 约 1.14/1.06 ms，512×1664
约 1.06/0.99 ms。每批 2 或 8 个均未呈现更低的单位矩阵成本；结合此前
workspace256 完整训练负收益，不优先继续扩大 workspace。

**结论：单靠消除 Python/kernel launch 不足以节省 102 ms。** Graph 是辅助路线，
主路线必须提高专家权重 NS GEMM 与 MoE 前后向的实际执行效率。按探针仅节省
17 ms 粗估，完整步仍约 405 ms / 15.8%，不能据此许诺 20%。

### 执行顺序与停止条件

| 优先级 | 实验 | 具体设计与验收 |
|---|---|---|
| P0 | 当前完整步暖 trace | 用标准 BF16 Muon 重新采 trace，分别量化 NS 三类 GEMM、norm、cast/stack、clip、AdamW、MoE forward/recompute/dX/dW；旧 trace 来自失败的 Gram 候选，不能代替标准 Muon 耗时分解。记录 GPU 时钟、温度和其他 GPU 负载。 |
| P1 | Muon 专家矩阵内核 | 针对短边 512、长边 1664、batch=4 的 BMM/baddbmm 测布局、tile、持久 workspace 和 epilogue；保留五次迭代、系数、BF16 中间舍入点与 FP32 momentum。定制 kernel/不同 GEMM 算法可能改变归约顺序，必须先通过数值 gate，不能凭数学公式等价合入。以优化器阶段 ≤150–160 ms 为阶段目标。 |
| P2 | MoE BF16 前向、重算及反向 | 针对实际 token 分布优化 grouped GEMM，优先 dW 与 checkpoint replay；测试合法 tile/split-K、减少重复 stack/cast、融合逐元素操作。保持 checkpoint=1，保留 routing、aux loss 和文档隔离。split-K 会改变归约顺序，仍受一致性 gate 约束。以 F+B ≤150–160 ms 为阶段目标。 |
| P3 | Graph 与固定缓冲补足 | 只在 P1/P2 后重测剩余 launch 开销；固定 shape 的 NS 或 chunk 图优先于整模型图。不重复计算 P1 已消除的开销；要求完整步 A/B/A 有稳定收益且无显存压力。 |
| P4 | 真实数据端到端验收 | 同一已审计 corpus 的同一批 token、相同 packing 及更新步；固定数据供给时先过计算 gate，再测连续窗口 useful tokens/s 和 wall MFU。保留真实日志、保存、暂停及恢复路径。 |

Graph 集成必须处理的真实代码约束：

1. `step_with_scaler` 每步 `zero_grad(set_to_none=True)`，图不能继续读旧梯度地址；
   选择显式固定输入 staging 或有语义验证的稳定梯度缓冲。必须保留 grad=None 与
   全零梯度的差别，尤其是未用参数的 weight decay/momentum 行为。
2. LR 当前是 Python float，Graph 会固化 scalar 参数；可只 capture 不依赖 LR
   的 NS 子过程，将 LR/decay 更新留在图外，或实现经测试的 device scalar。
   不允许捕获第一次 LR 后整段训练重复使用。
3. 不能将每批参数/临时 workspace 无界地保存在 205 张独立图里；需实测 private
   pool 内存、复用生命周期和 staging 成本。恢复 checkpoint 后重新绑定/构图，
   保存格式及参数、momentum 对应关系不变。
4. 保留 MoE training compiler barrier；此前直接移除已触发 autocast 编译错误。
   若需减少图断点，先设计有明确 backward 的局部 custom op，不直接套整模型图。

上述固定地址、固定调用参数和 private pool 约束参考
[PyTorch CUDA Graph 文档](https://docs.pytorch.org/docs/stable/notes/cuda)。
在线 stable 文档当前标注 2.14，本机为 2.13；API 可用性以本机探针为准。

### 一致性与达标标准

- 沿用下文已有 gate：logits 相对 L2 <0.5%、梯度 <2%、更新量 <3%、25 步 loss
  最大相对差 <2%、最终参数相对 L2 <0.1%；无 NaN/Inf，文档隔离、恢复下一步
  严格一致。新增动态 LR、空/偏斜专家、grad=None、多种 packing 桶的覆盖。
- 微基准过关后先跑真实 32 层、同一输入序列的短训 A/B，并在更长的 100–200 步
  窗口检查 loss/更新漂移；已有阈值不因候选失败而事后放宽。
- 性能使用 checkpoint=1，至少 20 步预热、每个 A/B/A 稳定统计 ≥200 步；同窗口
  记录 wall/useful tokens、阶段时间、显存及环境。要求提升稳定超过测量噪声。
- 合成完整步 MFU ≥20% 是第一道门槛；实际连续训练 MFU ≥20% 才能报告本机训练
  达标，并单列含 checkpoint 保存的长窗口吞吐。前者通过不等于后者通过。
- Gram FP16、已失败的 compiled BF16 NS、关闭 checkpoint、切换 attention
  backend、改变有效 batch/参数精度均不进入这条路线。若 P1/P2 未达到预算，
  报告瓶颈和实测结果，不把 19.9% 四舍五入成通过。

本次仅添加调查探针和本节计划，未调整生产训练默认值。原始结果与窗口见
`experiments/results/mfu_20261012/muon_graph_probe.json`，复现命令：

```powershell
$env:PYTHONUTF8='1'
python experiments/probe_muon_graph.py
```

## 2026-10-12：11% → 40% MFU 调查与实验

本轮先在空闲的 RTX 5070 Ti / Windows 上实验；用户报告的 11% 来自 RTX 5090。
**40% 尚未实现，也不能把本机结果外推成 5090 的达标结果。** 验收要求是保持
数学语义，并限定浮点误差；不改变模型层数、top-k、参数精度、loss、数据顺序或
有效 batch 来换取一个更高的数字。

### 已核实的当前管线

- 最新本地 pretrain 日志是 `trainer/logs/train_pretrain_20260925_121129.log`，但日志内容
  来自远端 `/root/autodl-tmp`、31.36 GiB 显卡。不能把文件存在本地等同于本机训练。
- 对应 checkpoint JSON：32 层、hidden=512、8 专家 top-1、6400 vocab、FP32 参数；
  678.73M 总参数、106.20M active。Muon BF16 NS、64 MiB workspace，checkpoint=1，
  tensorwise FP8 / auto filter，`max-autotune-no-cudagraphs`，两个 packing 桶。
- 末段 steps 380000–400000，133 个相邻 100 步且端点同桶的日志区间，估计中位
  480 ms/step、65,633 **padded** tokens/s；桶为 11×2864/2880 与 7×4096。
  这只是日志时间差估计：时间仅精确到 0.1 min，区间内部可能换桶，也没有有效
  token 计数。Profiler 关闭，无法从此日志直接复算精确 MFU。
- 数据是 Parquet → bounded Arrow window → tokenization / packing cache → CPU
  DataLoader → H2D → forward → backward → clip → Muon + AdamW → log/save。
  单卡 streaming LR 原先把 CPU token count 传到 CUDA 再 `.item()` 回读；本轮去掉
  这一次无必要的往返，保持计数和 LR 公式不变，多卡仍保留原 all-reduce。
- MoE 的 grouped GEMM 直接读取专家权重并使用 BF16，绕过 `Float8Linear.forward`。
  当前 torchao auto filter 每层只选择 8 个 gate 和 8 个 up，32 层的 512 个转换
  **全部是被绕过的专家模块**。稳定诊断 trace 中 `_scaled_mm` 调用数为 0。
  `FP8 enabled / linear_layers=512` 不代表这 512 个层执行 FP8；本轮添加覆盖提示，
  没有修改执行精度或悄悄替换算法。
- 主干每个 MoE block 保留 compiler-disable barrier。实验仅在进程中解除 barrier、
  强制 Triton 后，Dynamo 报 `Global autocast(cuda)_enabled state changed while
  dynamo tracing`，未得到合法训练步。保留生产 barrier。

### MFU 口径与实验方法

主指标是同步窗口的墙钟 tokens/s；计时包含 forward、loss、backward、gradient clip、
完整 Muon/AdamW 更新和 zero_grad。先预热 20 步，随后 3×5 步窗口取中位数；首次
编译、autotune 和额外 profiler 步不计入吞吐。所有样本均为随机 token ID 的合成
计算基准，不读取/覆盖语料、权重、resume checkpoint，也没有启动正式训练。

参考 MFU 使用 `6*N_active*T + 12*L*Qheads*head_dim*causal_pairs`；packing 下
`causal_pairs = B*sum(segment_len*(segment_len+1)/2)`。这里只估计有用模型 FLOPs，
不把 inactive experts、checkpoint 重算、Muon 正交化或 padding FLOPs 算入收益。
这不是精确算子计数，真实文档长度和有效 token 占比需由真实训练另行记录。

本机分母固定为 **87.9 TFLOPS，BF16 / FP32 accumulate / dense**，5090 同口径为
209.5；5090 dense FP8 / FP32 accumulate 为 419。稀疏峰值或 FP4 AI TOPS 不适用。
来源：[NVIDIA RTX Blackwell 白皮书，GeForce 规格表](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)。
只有统一精度、分母、有效 token 和 attention 口径后才能比较“11%”与“40%”；调整
分母本身不算性能优化。以约 65.6k tokens/s 的 `6N` 粗估，5090 的 BF16/FP8
参考值分别约 20%/10%，尚未计入有效文档 attention，不能据此认定用户的 11% 算错。

### 已完成的完整计算步实验

固定真实 32 层 MoE，FP32 参数，BF16 activation，top-1，checkpoint=1，4×2048，
每行 4 个独立等长文档。`compile + FP8` 行对应最新日志的软件设置。

| 配置 | ms/step | tokens/s | 峰值 allocated GiB | BF16 参考 MFU | 裁决 |
|---|---:|---:|---:|---:|---|
| SDPA、eager、标准 BF16 Muon | 687.7 | 11,912 | 11.47 | 9.32% | 旧路径参考 |
| Flex、eager、标准 BF16 Muon | 500.4 | 16,371 | 11.07 | 12.81% | 已有优化，不能算本轮新增 |
| Flex、compile + FP8、标准 BF16 Muon | 446.3 | 18,355 | 8.57 | 14.36% | 当前设置基线 |
| 上述 + 实验 Gram FP16 / workspace256 | 408.2 | 20,069 | 8.42 | 15.70% | 一致性不通过，不采用 |
| 上述 + 编译 BF16 NS / workspace256 | 464.3 | 17,643 | 8.42 | 13.80% | 更慢且一致性不通过 |
| 上述关闭 checkpoint | 811.3 | 10,097 | 13.74 | 7.90% | 更慢，不采用 |
| 上述先 cast 再 stack 专家权重 | 436.6 | 18,763 | 8.58 | 14.68% | 一致性通过；约 2% 收益不足以排除噪声 |
| 当前设置，改为 8×2048 | 644.6 | 25,418 | 11.57 | 19.89% | 容量实验；改变每步 token 数，不能宣称训练等价 |

最终隔离 A/B/A 复测：workspace64 的两次基线为 **422.84 / 422.64 ms**（漂移
仅 −0.05%），workspace256 为 **435.87 ms**，吞吐 **−3.01%**，因此不调高
默认 workspace。初轮基线 446 ms 与最终约 423 ms 的会话漂移，也说明 cast-first
初轮约 2% 的差异不足以宣称收益。各次运行均保留独立窗口与源码 hash。

当前基线阶段均值：forward 56.8 / backward 149.7 / optimizer 241.6 ms；优化器约
54%。在同一 8192-token batch 和本机分母下，40% 要求约 **160 ms/step**，即连
当前 optimizer 单独耗时都超过整个目标预算。只改 attention、DataLoader、compile
模式或日志频率无法达到目标；需要同时压缩 MoE forward/backward 和优化器。

关闭 checkpoint 时 reserved 达 14.85 GiB，尚需显卡桌面/驱动内存；显存压力明显。
未采集 PCIe/page-fault 事件，不能断言发生了具体的分页机制。

### 前后结果一致性：严格门槛优先于速度

新增 `experiments/validate_training_optimizations.py`：真实 hidden=512、expert=1664、
两层 top-1 MoE，FP32 参数、tensorwise FP8 设置，固定种子与周期合成 token，
LR=5e-4，训练 25 步。独立重复 baseline 为控制，另外验证首步 logits、loss、
梯度、参数更新、末步参数、文档隔离、optimizer checkpoint 的实际恢复与继续更新。

预设保守门槛：logits 相对 L2 <0.5%，梯度 <2%，**更新量** <3%，逐步 loss 最大
相对差 <2%，最终参数相对 L2 <0.1%；隔离必须逐位一致，恢复的下一步必须逐位一致，
无 NaN/Inf。不会以“大权重几乎不变”掩盖错误的小更新，也不会事后放宽阈值凑通过。
这是本轮 smoke gate，不是普遍的收敛等价证明。

- 独立 baseline repeat、cast-before-stack、workspace256、关闭 checkpoint：本次
  25 步 logits、梯度、更新、loss 和参数均逐位相同；隔离/恢复检查也通过。
- Gram FP16：首步 forward/gradient 相同，但更新误差 **14.48%**、25 步 loss 最大
  差 **9.63%**、末步参数差 **0.782%**，否决。FP64 数学等价测试通过，不能替代
  真实半精度梯度压力测试。早期文档中“小矩阵误差较小/投影 +19%”不足以合入。
- 编译 BF16 NS：更新误差 **3.46%**、loss 最大差 **14.48%**，否决。
- SDPA 与 Flex：首步梯度差 0.30%、更新差 2.89%，单步数值门槛通过；25 步 loss
  最大差 13.99%、参数差 0.430%，未通过更严格的轨迹门槛。两者独立的 packing/
  attention 语义测试仍有效，不能把此结果描述为 mask 算错；top-1/Muon 会放大
  舍入扰动。在要求本轮前后严格一致的场景中，保持已有 backend，不再次切换。
- 所有精度/算法、cast-before-stack 和解除 MoE barrier 的尝试均只在实验进程中
  monkeypatch，未接入生产默认路径。

### 从现状走向 40%：按优先级执行的计划

1. **先固定 5090 的可比较基线。** 下一次合法训练启动/恢复时开启 timing，两个桶
   各预热至少 20 步，稳定统计不少于 200 步；再采 5–10 个暖 trace 步。记录 useful
   tokens/s、连续窗口吞吐、wall tokens/s、每阶段时间、padding/文档长度、实际
   FP8 GEMM 覆盖、图断点、allocated/reserved、训练软件 revision。Profiler 本轮
   新增连续 window 指标，避免只累加 compute step 而遗漏步间 GPU idle；不能把
   CPU host_gap 简单加到 GPU 时间上，因为可能重叠。
2. **优化 Muon 的搬运、launch 与批处理，不先改变 NS 数学。** 对 64/128/256 MiB
   workspace 做 A/B/A；尝试 persistent updates/workspace、融合同次 cast/pack、
   固定梯度缓冲的 optimizer CUDA graph。保留系数、NS steps、BF16 工作精度、
   momentum/nesterov、weight decay、参数分组、序列化布局。先通过本轮一致性门槛，
   再要求同 batch 稳定吞吐提升 ≥5% 且无显存超限。Gram/FP16 暂停推进到生产。
3. **优化 BF16 MoE grouped GEMM 与权重搬运。** 稳定 trace 表明每步有 288 个
   `_ragged_mm`、96 个 weight-grad kernel、251 次 stack、2352 次 copy；这些包含
   checkpoint replay。先为真实 expert token 分布调 BM/BN/BK、weight-grad split-K，
   再考虑避免重复 FP32 stack、分离 packing/casting 与 GEMM；保护空专家、偏斜
   路由、梯度、checkpoint keys。解除全部 barrier 当前编译失败；下一版应设计
   注册 autograd 的显式 custom op/局部编译边界，不能直接删除 disable。
4. **FP8 专家算子作为独立精度项目。** 如需探索真正 FP8，实现 routed grouped
   FP8 forward/dX/dW、明确 scale 与 ragged padding，记录实际指令覆盖，分别对
   BF16 做数值/短训/真实验证集验收。此前只开启 torchao 开关不会加速专家；也
   不能把 FP8 峰值当作当前全部模型的执行峰值。若一致性门槛不通过则继续 BF16。
5. **用显存优化扩大 physical batch 时保持训练合同。** 优先 chunked/fused CE、
   避免完整 logits FP32 副本、selective checkpoint 与安全 reuse；输出接口、
   ignore-index、文档边界 loss 不变。只有存在 accumulation 可调整时才在保持
   有效 batch 的条件下重新分配 microbatch；当前 accumulation=1 直接翻倍 batch
   会改变更新，必须作为新训练配置在 epoch 边界验证，不用于“等价优化”验收。
6. **最后处理数据与 checkpoint 长尾。** 按暖 trace 测 chunk 切换时 GPU idle、
   Parquet token_count 规划、prefetch 命中率、CPU packing；再决定 pin/nonblocking/
   loader workers。后台保存须复制一致的模型与 optimizer 快照，保存完整 resume
   状态，保持 pause exit=42 与恢复游标；不能靠减少保存次数改变故障恢复合同。

每个阶段都在相同模型、种子、样本顺序、有效 batch、精度和计时口径下做 A/B/A，
先验收正确性再看收益。阶段性能预算以实测重新分配，不把局部倍数相乘。若无法
同时满足一致性、显存和 40% 的预算，报告实测上限与瓶颈，不降低验收标准。

### 复现与产物

```powershell
$env:PYTHONUTF8='1'
python experiments/bench_training_mfu.py --batch 4 --seq 2048 --param-dtype fp32 --attention flex --fp8 tensorwise --compile-mode max-autotune-no-cudagraphs --peak-tflops 87.9 --output experiments/results/mfu_20261012/reproduce_baseline.json
python experiments/validate_training_optimizations.py --output experiments/results/mfu_20261012/reproduce_correctness.json
python -m pytest tests/test_training_mfu_experiments.py tests/test_training_profiler.py -q
```

各实验 JSON、warmup/计时日志与额外 trace 位于 `experiments/results/mfu_20261012/`
及 `experiments/results/mfu_*.log`；JSON 保留 config、种子、各窗口时间、阶段/显存、
git HEAD 与核心源码 hash。结果目录按仓库规则忽略，不上传大型 trace。
汇总与图表：`experiments/results/mfu_20261012/summary.json`、`comparison.png`；
可用 `python experiments/summarize_training_mfu.py --directory experiments/results/mfu_20261012 --plot`
重新生成。一次在 warmup 阶段因与 GPU 测试重叠而取消的运行明确标记 aborted，
不用于任何性能结论。

验证记录：相关 CPU 回归套件 80 passed / 4 skipped；包括 CUDA 与 opt-in compile
检查的 Muon、Gram、packed attention、torchao 套件 74 passed。新实验/统计单测
28 passed，包含 FP64 多项式等价与不同步间等待的吞吐计算；这些套件有重叠，
数量不能相加当作独立测试总数。上述单测通过不撤销候选短训 gate 的失败裁决。

---

## 历史优化记录（以下数字不作为本轮 32 层 MoE 基线）

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
- Dense/MoE 主干和 RecurrsiveInstinct 主干的 packed SDPA 路径共享同一只读 bias；不改变普通
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
- 输出/梯度、MoE checkpoint 模式 1/2、RecurrsiveInstinct、连续不同分段的 compiled batch 检查通过。
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
