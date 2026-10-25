# V2 Dense：向 14,000 有效 token/s 优化

最终完整更新 A/B 达到 **14,399.67 有效 token/s**，相比本轮基线
12,454.81 提高 **15.62%**。两组 peak allocated 都是 **12.507867 GiB，
增量 0.00%**。生产路径采用 cuBLAS GEMM 与仅长桶反向 TMA，已按验证条件
默认启用。此结果属于本机 PyTorch 训练路径，不是之前的 Torch-free native
cublas worktree 吞吐。

## 口径与环境

本机 RTX 5070 Ti 16 GB，PyTorch 2.13.0+cu132、TorchAO 0.18.0。
模型为 32 层、H=1536、F=5504、Q=16/KV=4、head_dim=96、词表 6400 的
Instinct V2 Dense。BF16 主干参数/激活，FP32 embedding/head/norm，tensorwise
FP8（auto filter），Muon mantissa、BF16 momentum、batch 上限 16、workspace
512 MiB，ckpt4，静态形状 `max-autotune-no-cudagraphs`。进程 allocator 上限
14.37 GiB。

交替使用 4×3072（三个文档/行）和 3×4096（单文档/行），累计 8 个微批。
每次更新预测 **98,244 个有效 next-token target**，忽略跨文档目标。
14,000 有效 token/s 对应 **7,017.43 ms/累计更新**。
合成数据实验不包含真实 loader、磁盘保存和首次编译时间；固定权重诊断不执行
优化器更新，不能直接当作完整训练吞吐。

上轮相同 recipe 的完整更新基线为 12,449.48 有效 token/s（平均
7,891.42 ms/更新，两个预热更新后测量三个更新）；报告在
`experiments/results/cublas_reference_20261025.json`。

## 已完成的固定权重筛选

每个进程保留真实优化器状态和梯度缓冲，不更新权重。每个桶预热 3 次，交替
测量 6 次，表中列中位数。原始样本和 profiler trace 在
`experiments/results/v2_14k/`。

| 方案 | 4×3072 前反向 ms | 3×4096 前反向 ms | peak allocated GiB |
|---|---:|---:|---:|
| 原选核 | 802.10 | 1082.35 | 12.5794 |
| GEMM 限定 ATEN/cuBLAS | 710.83 | 989.35 | 12.5091 |
| ATEN + TMA attention（初次筛选） | 783.57 | 1011.80 | 12.5091 |
| ATEN + TMA attention（独立进程复测/trace） | 710.07 | 926.59 | 12.5091 |

allocated 的小幅差异不能直接归因于内核：此前确认过 72 MiB 的可回收对象
周期。最终显存增量应以完整更新 A/B 的 allocated 比值计算，reserved 单列，
不把缓存分配器释放当作实际激活节省。

原 4096 桶的 attention backward 合计约 342 ms/32 层。另有一组 FP8
下投影输入梯度 GEMM 被 Inductor 选成 Triton，约 2.06 ms/层；cuBLAS
路径约 0.67 ms/层。切换的是编译模型的 GEMM 后端，不改变 FP8 配方、模型
参数、NS 迭代次数、序列长度或有效 token 计数。

TMA 的独立 attention 前反向从 13.32 降到 11.38 ms。初次整模型结果退化，
独立进程复测却改善，初次波动的根因尚未确定，不能删掉初次结果或推断是
某个显存泄漏。复测 trace 显示长桶 attention backward 从约 10.67 降到
8.21 ms/层，但 forward 从 2.66 增至 3.21 ms/层。因此最终候选只给 **3×4096
的反向**使用 TMA，前向保留原编译器选核，短桶保留原路径。最终完整更新
测试采用更长的预热和测量窗口。
展开 GQA 的方案未胜过紧凑 GQA，不进入训练默认路径。
TMA 初次筛选中 4096 桶最后一个样本为 1268.01 ms；该样本保留在报告中，
未被删除。中位数退化的判断不依赖这个离群值。

## 接入和回退

`model.compile_policy.training_compile_kwargs` 为匹配的 V2 Dense、BF16
参数、tensorwise FP8/auto filter、Blackwell/PyTorch 2.13 和 max-autotune
模式生成 **每个编译模型自己的** ATEN 选项，不修改全局 Inductor 配置。
所有现有训练编译入口共用它，autotune 通过同一 pretrain 入口生效。
显式 `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` 设置优先于自动选择。

- `INSTINCT_V2_GEMM_BACKEND=default`：恢复原编译器选核。
- `INSTINCT_V2_GEMM_BACKEND=auto`：默认，按上述条件启用。
- `INSTINCT_V2_GEMM_BACKEND=aten`：显式限定 GEMM 后端。
- `INSTINCT_V2_FLEX_TMA=0`：恢复原 attention 路径。
- `INSTINCT_V2_FLEX_TMA=auto`：默认，仅匹配 3×4096 的反向。

Muon 另外筛选了 8 组分块，覆盖 512 MiB workspace 下的四种实际矩阵批次。
候选 NS 输出均与原实现逐位相同，但较好分块只缩短约 1%–3% 的 NS 耗时，
对完整训练影响很小，因此本次保持其原有分块、迭代次数和 workspace。

## 数值检查

`tests/test_v2_tma_attention.py` 的 4 项 CUDA 检查全部通过。
实际桶的 attention 输出和 dQ/dK/dV 对照逐位相同，验证了文档/因果隔离。
两层完整 V2 宽度的 FP8 主干对照（原编译器 vs ATEN+TMA）结果如下：

| 桶 | 梯度余弦 | 梯度范数比 | 梯度相对 L2 | loss / reference loss |
|---|---:|---:|---:|---:|
| 4×3072 | 0.998461 | 0.998251 | 5.5457% | 5.147461 / 5.147437 |
| 3×4096 | 0.998718 | 0.999555 | 5.0619% | 5.132934 / 5.133216 |

不同 GEMM 累加/融合舍入可触发 FP8 分箱变化，因此完整 FP8 模型不是逐位
相同；沿用已有 FP8 测试的梯度方向/范数门槛。短对照不证明长期预训练质量。
CPU 编译选项/参数贯通检查与 attention 回退检查通过；relay schema 与
Config WebUI/CLI 一致。

最终 CPU 回归：64 passed、29 GPU/slow cases skipped；另行完成的 CUDA
主干/attention 对照为 4 passed，反向专用 TMA 的追加选择器/隔离检查为
3 passed（含 2 个 CUDA 桶）。检查记录分别在 `v2_14k_final_cpu.log`、
`v2_14k_gpu_parity.log`、`v2_14k_backward_tma_fixed.log`。

## 完整更新验证

最终 A/B 必须包含累计反向、梯度裁剪、Muon/Adam 更新；各运行结束后保存
当前权重、完整 optimizer/scaler/cursor/RNG 状态，并逐项校验。
仅 `status=ok` 且 `verified_current_state=true` 的完成报告用于最终结论。

两组各运行 **8 个完整优化器更新 / 64 个微批**，前 3 个更新预热，后 5 个
计时。吞吐按这 5 个窗口的有效 token 总量除以总时间计算，非最快窗口。

| 指标 | 原选核基线 | cuBLAS + 长桶反向 TMA |
|---|---:|---:|
| 有效 token/s | 12,454.81 | **14,399.67** |
| 平均完整更新 ms | 7,888.04 | **6,822.66** |
| 窗口最小—最大 ms | 7,881.11—7,896.98 | 6,816.32—6,826.59 |
| 平均前反向 ms | 7,504.74 | 6,438.14 |
| 平均裁剪/优化器 ms | 383.29 | 384.52 |
| peak allocated GiB | 12.507867 | 12.507867（+0.00%） |
| peak reserved GiB | 13.707031 | 13.707031 |

两组损失均有限，当前 model/optimizer/scaler/data cursor 和配套输出权重逐项
校验成功；实验结束时 GPU 已空闲。完整 resume 文件分别为：

- `checkpoints/ckpt4_recipe_20261026_003814_120812_1536_resume.pth`
- `checkpoints/ckpt4_recipe_20261026_004136_51232_1536_resume.pth`

最终数据：`experiments/results/v2_14k/baseline_updates.json`、
`optimized_updates.json`、`final_comparison.json`、`final_runner.json`。
原固定权重诊断、失败的首次 TMA 筛选、所有分块结果均保留在同目录。

仅长桶反向 TMA 的额外 CUDA 检查再次通过。隔离检查固定输入的 stride、
requires_grad 和编译形状，原位扰动未来 token 后要求输出逐位相同；避免
更换梯度标志触发另一套 autotune tile，使舍入误差混入因果性检查。

本机面板仍保存 ckpt3，未改写面板状态。本报告的配置是 **ckpt4**；使用该
结果时应选择 ckpt4、Muon workspace 512 MiB 和上述 FP8/compile recipe。
真实语料的文档长度分布、数据等待、保存及重新编译会影响实际端到端吞吐。
