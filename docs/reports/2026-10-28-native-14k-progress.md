# Native V2：2026-10-28 算子优化、15k 验证与显存调优

2026-10-31最终组合在完整512微步生产窗口达到 **16,053.403 TPS**，使用生产recipe学习率3e-4，完整保存核验通过。采用256粒度分组、共享cuDNN handle及对计时波动更保守的有界选型；同一120-plan目录的常驻增量减少358 MiB。fast accumulation未启用，完整模型基线的梯度重复性限制仍未解决。最新条件与结果见末尾“共享cuDNN运行时资源与扩大验收”。下文保留前几轮历史记录。

2026-10-28组合已完成256测量微步的 **15,210.9661 有效预测目标/秒** 验证；128微步独立运行为15,080.6001 TPS，扩大运行的相同前128微步为15,081.4158 TPS。两轮均完成当前完整状态保存及逐数组核验。新增残差梯度融合、cast＋amax、权重scale复用和自适应连续文档分组；固定样本kernel数由5402降至4314。详细结果见“15k执行结果”。

上一轮组合的同窗口生产验证达到 **14,603.2725 有效预测目标/秒**，对照为14,308.0266，提高 **2.0635%**、平均每微步减少 **16.759 ms**。160微步正常完成，当前weights、完整optimizer/scaler/RNG/data cursor状态逐数组核验通过。峰值native live **12.73708 GiB**基本不变；NVML设备采样峰值 **15.29552 GiB**，比本轮匹配对照增加约 **1.63%**。18,000 TPS尚未达到。

第一阶段曾达到14,365.0132 TPS、peak live 12.98903 GiB；下文保留各阶段的条件与历史记录，最终采用配置见“专项验证完成”一节。

随后扩大到256个测量微步、32个测量更新：3,055,038有效目标 / 210.9335829秒 = **14,483.4121 TPS**。288个总微步（含预热）完整保存核验通过。NVML设备级采样峰值15.448 GiB，当时free仅0.473 GiB；因此缓存预算的生产余量仍需后续对照，不能由12.99 GiB live推断还有3 GiB空闲。[扩大窗口原始报告](../../experiments/results/cublas/optimization_20261028/production_fused_extended.json)

## 测量口径

RTX 5070 Ti；32 层、H1536、FFN5504、Q/KV16/4、head_dim96、词表6400，1,010,277,888参数。BF16隐藏参数＋uint16 mantissa，BF16 momentum，FP32伴随参数，tensorwise FP8 auto filter。生产数据为 `D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet`。

使用真实 Rust packed 数据加载、H2D、累积8微步和Muon更新；从group0末64批跨到group1。省略的前缀未训练。4个更新预热后测量16个更新，共128微步、1,518,127有效目标、105.6822557秒。缓存已提前构建，窗口不含启动、预热及最终保存；不等于冷启动到退出的吞吐。完整探针（含启动、保存与逐数组核验）236.46秒。测试 LR=0.0005、weight decay=0.1；共享预设仍保留原有 LR=0.0003。

| 运行 | 有效目标/s | 完整状态保存核验 |
|---|---:|---|
| 2026-10-26 hybrid基线 | 11,398.30 | 通过 |
| 2026-10-27 分组cuDNN、单行Norm/RoPE | 13,727.37 | 通过 |
| 2026-10-28 多行Norm/RoPE、1024 MiB Muon预算 | 13,956.79 | **未完成：GPU丢失，保存阻塞** |
| 2026-10-28 Norm反向融合、FFN布局、分桶缓存 | **14,365.01** | **通过** |

最新完成运行的Muon阶段平均469.68 ms/更新，扩大分块预算没有显示明显整阶段收益，因此保留512 MiB。以相同目标密度计算，18,000 TPS要求658.91 ms/微步，当前825.64 ms，仍需减少166.73 ms（时间减少20.19%，吞吐增加25.30%）。这个预算没有证明18k是可达到的硬件上限。

## 本次已验证内容

- 多行Norm/RoPE：同输入前后向的短/长桶中位数，从单行的823.91/892.96 ms变为8行的809.84/878.72 ms。该探针固定权重，不执行优化器更新。
- 8行版本16组完整小模型loss/VJP检查通过。保留cuDNN非确定性反向的既定误差上限；没有采用之前未通过误差阈值的全局对齐或4096长度合并方案。
- 保留V2模型和优化器状态的12组Muon GEMM检查，cuBLAS与现有Triton参考结果逐位相同。此结果是单GEMM验证，不是完整优化器阶段加速结论。
- Rust数据加载、文档/目标守恒及暂停续训检查：12项通过。本机被清理的cuDNN 9.27.0.42运行库已从缓存wheel恢复；Rust工具从当前提交中的源码重建到忽略目录，未撤销工作区已有的源码删除。
- 数据工具新增可选环境变量 `INSTINCT_DATASET_COMPILER`，允许使用已构建的外部Rust可执行文件。Config WebUI现可传递原生已有的 `--tune` 参数。

## 新算子与缓存实测

新增梯度合并＋RMSNorm反向融合：VJP保留有序的梯度分量，由Norm反向内核在寄存器中完成FP32相加，保留原有cast位置，减少中间大张量的写回和再次读取。FFN反向也可以直接提交这组梯度分量。

该融合在GPU恢复后通过16组完整loss/VJP检查，loss完全相同，最坏梯度相对L2为2.44e-7。另将FFN前向FP8量化改为连续写入，反向5路布局使用64×64 tile，保持原有BF16及FP8舍入；组合方案也通过16组GPU检查。

以下来自相同生产样本、完整V2模型、实际常驻优化器和梯度缓冲；每桶3次预热、3次无profiler测量，列中位数。固定权重、不更新、不含loader；不能替代生产TPS。

| 配置 | 4×3072 ms | 3×4096 ms | peak live GiB | 相对双1280 MiB新内核的显存变化 |
|---|---:|---:|---:|---:|
| 恢复后原基线 | 812.80 | 880.38 | 12.80738 | +0.55% |
| +Norm反向融合 | 801.76 | 869.97 | 12.73707 | 0% |
| +FFN连续前向、64×64反向 | 792.19 | 858.70 | 12.73707 | 0% |
| 两桶缓存1536 MiB | 790.35 | 858.50 | 12.98902 | +1.98% |
| 两桶缓存1792 MiB | 789.83 | 859.69 | 13.24098 | +3.96% |
| 双1280 MiB缓存、复用池64 MiB | 803.21 | 867.39 | 12.73707 | 0% |

第一阶段采用短桶1536 MiB、长桶1280 MiB：短桶增加1层gate/up，长桶的额外缓存没有稳定收益。该阶段保留256 MiB复用池、512 MiB Muon workspace、cuDNN每组8文档、plan quantum1024、LRU64。生产运行34个plan、0次淘汰、0次fallback，最大attention workspace为433.5 MiB。最终配置调整为两桶1280 MiB，见后文。

共享原生预设、`enable_fused_ops`和WebUI已接入上述设置，启用cuBLAS算法调优、cuBLAS Muon、cuDNN attention、3072/4096分桶及Rust数据预取。显式实验覆盖值仍被保留。预设/WebUI、执行策略、数值及续训CPU检查57项通过；新增显存采样检查2项通过。Python语法检查与`git diff --check`通过。

## 显存组成与使用边界

| 项目 | 本模型占用 |
|---|---:|
| 隐藏矩阵BF16权重 | 1.86328 GiB |
| 隐藏矩阵uint16 mantissa | 1.86328 GiB |
| 隐藏矩阵BF16 momentum | 1.86328 GiB |
| 隐藏矩阵BF16梯度缓冲 | 1.86328 GiB |
| 伴随FP32参数、Adam状态及梯度 | 0.14806 GiB |
| 每层checkpoint边界FP32 hidden | 72 MiB |
| 全部32层attention O/LSE | 1176 MiB |
| 每层gate/up BF16缓存 | 258 MiB |

上述参数和优化器常驻状态合计7.60119 GiB；实际初始live还包括输入、RoPE表等。全部32层gate/up单独就需要8.0625 GiB，不能在16 GiB卡上简单全部缓存。各阶段scratch与缓存有不同生命周期，表中不是可直接相加的峰值分解。

128微步生产窗口的CUDA异步池保留峰值为14.1875 GiB，包含框架live之外的保留空间，但仍不是设备全部占用。独立同步诊断中，新内核双1280 MiB配置的最大设备使用采样为14.362 GiB；峰值出现在FFN反向量化阶段。该诊断会插入同步，所以从吞吐汇总排除。异步池高水位、native live高水位和设备采样不可跨时刻相减后称为碎片或真实余量。

新增20 ms间隔的NVML采样，不建立CUDA context、不插入GPU同步，记录setup/measurement/save各阶段峰值。它是设备级采样下界，包含reserved memory，不能称为进程allocated；Windows WDDM由操作系统管理显存。[NVIDIA NVML接口说明](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)

## 仍需优化的成本

在Norm反向融合后、FFN新布局之前的CUPTI剖面中，短/长桶GEMM为330.5/329.7 ms，attention及Q/K Norm/RoPE为215.9/292.2 ms，其他kernel约210/209 ms；trace间隙约27/23 ms。新FFN布局又在无profiler全模型测量中节省约10 ms/微步。

短/长桶数学causal attention前向每层为66.90/130.87 GFLOPs，保存O/LSE占36.75 MiB，约1736/3396 FLOP/byte。gate/up每层415.54 GFLOPs、258 MiB，1536 FLOP/byte；K/V每层28.99 GFLOPs、18 MiB，也是1536 FLOP/byte。attention数值按实际文档长度计算，仅包含QK/PV，不包含softmax、padding或tile浪费。缓存选择需结合实际算子耗时；gate/up FLOPs大，不代表单位显存节省时间最多。

剩余主要方向是attention反向、FFN量化读写和残差梯度合并，随后是Muon及GEMM选型。当前证据不支持靠继续增大缓存补齐18k的约167 ms缺口。

## memcpy、memset 与空档专项调查（暂停验证时的记录）

用户随后明确GPU被占用，要求先调查、形成方案并修改，等明确通知后验证。以下分析只重读已有trace和源代码；本节新增候选未运行测试或GPU验证，均未加入生产默认策略。先前启动的详细探针没有产出测量窗口，不能作为有效结果。

用户表格来自`fused_cupti.window6/7`，当时尚未采用FFN连续写入/64×64布局。按原始CUPTI external correlation拆分，单位ms/微步：

| 来源 | 短桶 | 长桶 |
|---|---:|---:|
| cuDNN attention backward内部memset，32次 | 5.61654 | 2.72863 |
| cuBLAS BF16 K/V权重梯度内部memset，64次 | 0.03581 | 0.03572 |
| H2D＋DtoH，两次memcpy | 0.00374 | 0.00358 |
| 原生首个至最后一个计算kernel之间的空档 | 13.68151 | 14.16051 |
| Torch相同计算窗口空档 | 9.66331 | 14.63679 |

旧表原生26.95/23.37 ms空档包含额外的首尾控制段。其最大单个空档10.623/9.098 ms位于最后一个`scaled_copy`之后、loss DtoH之前。原生固定权重探针在每微步调用`e.item(loss)`；Torch探针只同步而不读取loss，因此GPU activity跨度尾部口径不同。统一首/末计算kernel边界后，短桶实际空档差为4.02 ms，长桶原生还少0.48 ms。完整跨度仍有价值，分析器保留它并单列`compute_window`，不静默删除尾部成本。

生产预训练早已只在日志/停止边界读取loss，所以不能把固定权重探针的约9–11 ms尾部空档逐微步算作可兑现生产收益。同步时CUDA池回收是待测假设：其release threshold当前为0，而CUDA文档规定同步时可能把超过阈值的保留内存还给OS；尚未完成0与有界保留阈值的对照，不把此假设写成已确认原因。[CUDA内存池说明](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY__POOLS.html)

memset来源已明确，但旧trace没有记录字节数，不能确定全部清零对应哪个内部workspace。核对所用Frontend 1.31.0（commit `51a3de73e122aeedafe68070acf3b7ff3970534e`）：`dQ_accum`只有cuDNN低于9.6时由frontend显式发memset；当前9.27不走这个分支。不能直接绕过cuDNN内部初始化，也不能删除用于padding正确性的输出清零。[对应源码](https://github.com/NVIDIA/cudnn-frontend/blob/v1.31.0/include/cudnn_frontend/node/scaled_dot_product_flash_attention.h)

实现方案：

1. **减少小kernel提交**：新增原生`quantization.py`，在原amax归约中同时生成FP32 scale及其正确舍入倒数。二者共用同一分配，保留scale即可保留inverse生命周期，且两个地址均16字节对齐。旧trace有288＋736=1024次独立`inverse_scale`，候选可省掉这些launch；实际时间收益待验证。量化、矩阵精度和舍入点保持原契约。使用实验开关`--fused-scale-inverse`，完整VJP验证器使用`--scale-inverse`。
2. **减少attention padding**：每个连续文档组按自身最长文档取quantum上界，不再全部继承整批最长文档。保留文档顺序、ragged offsets及完整token地址范围，避免把绝对offset错误当成组内offset；新增实际/旧策略descriptor padded-token计数。只对组间长度差异有收益，当前固定样本8/3个文档均在单组内，不能靠这两批样本证明加速。实验开关`--attention-local-maxima`；验证器增加`--local-maxima --skewed-documents`覆盖不均匀多组长度。
3. **补齐归因数据**：CUPTI记录每次copy/memset字节数、driver/runtime API起止时间及correlation ID；单列`control:read_loss`，同时报告完整跨度与统一计算跨度。新增但未执行scale精度/生命周期、文档分组和trace边界测试。

保留现有异步池默认和cuDNN内部memset。后续待用户允许：先数值/内存复用检查，再相同数据与相同边界的CUPTI A/B，最后真实生产loader＋Muon更新及完整保存。约20 ms目前不是可信的已确认收益预算。

## 专项验证完成：2026-10-28晚间

用户随后明确允许验证，并接受原0.001073相对L2误差量级。仅attention候选使用显式`--relative-l2-limit 0.0011`，其他逐元素、loss和文档/因果隔离门槛未更改。原0.001失败日志保留；新阈值下8个用例全部通过，最大相对L2为0.00109903，FP8用例最大3.894e-6。scale/倒数融合16个GPU用例通过，loss相同；CPU精度、生命周期、trace边界等62项通过。

同原始两批生产输入、每桶3次预热、3次无profiler计时，另各取1次CUPTI：

| 方案 | 短桶墙钟ms | 长桶墙钟ms | 短/长计算区间空档ms | kernel数/微步 |
|---|---:|---:|---:|---:|
| 本轮基线 | 795.62 | 866.11 | 13.97 / 14.51 | 6426 |
| scale＋inverse融合 | 792.26 | 860.86 | 12.10 / 12.85 | 5402 |
| 上项＋14 GiB池保留阈值 | 760.38 | 831.43 | 11.83 / 12.10 | 5402 |

详细CUPTI已确认短桶每层cuDNN memset为144 MiB，长桶为72 MiB，32层共4.5/2.25 GiB；其他cuBLAS memset总共仅36 KiB左右。memcpy只有1024字节metadata H2D和4字节loss DtoH。显式0→14 GiB池保留阈值对照中，loss同步在最后kernel后的尾部由6–11 ms降至0–1 ms，cuMemAllocAsync的累计主机调用时间由约13–15 ms降至约3 ms。API时间与GPU工作重叠，不能把累计API时间全加成可省墙钟。6个trace dropped records均为0。

原始两批样本都只有一个attention分组，不能体现local maxima收益。因此另从同一生产窗口抽取30文档短桶（group1/batch2）和一个长桶对照：

| 30文档短桶 | 整批max长度 | 每组max长度 |
|---|---:|---:|
| 无profiler前后向中位数 | 818.69 ms | 731.06 ms |
| cuDNN memset | 19.91 ms / 16.875 GiB | 8.54 ms / 7.875 GiB |
| attention前向kernel | 100.35 ms | 58.72 ms |
| attention反向及stats kernel | 121.28 ms | 106.41 ms |

这个样本是窗口内padding减少最多的样本，不能把其87.63 ms收益外推到每步。160微步实际metadata中85步受影响，descriptor padded tokens由5,052,416降为4,213,760（-16.6%）。没有删除必要memset，而是减少清零及attention所处理的padding范围。

最终生产A/B均使用真实Parquet loader、H2D、累积8步、实际Muon更新、4更新预热＋16更新测量；同128测量微步、同1,518,127目标、同group0末64批→group1前96批。缓存已建，最终保存和启动不计入测量窗口。

| 配置（两桶均1280 MiB缓存） | 有效目标/s | native peak live GiB | NVML采样峰值GiB | 峰值采样时free GiB |
|---|---:|---:|---:|---:|
| 基线，池release threshold=0 | 14308.03 | 12.73708 | 15.05029 | 0.87061 |
| scale融合，池阈值14 GiB | 14315.99 | 12.73708 | 15.42818 | 0.49271 |
| scale融合，池阈值13 GiB | 14388.82 | 12.73708 | 15.25744 | 0.66346 |
| scale融合＋local maxima，池阈值13 GiB | **14603.27** | **12.73708** | **15.29552** | **0.62537** |

四轮均完整保存并逐数组核验。14 GiB方案没有可用的净生产收益，不采用。最终方案native live只多276字节（约0.000002%），设备级采样峰值则约增加1.63%，两种指标不能混称allocated。池阈值是同步时的保留策略，不是硬上限；最终池高水位仍为14.1875 GiB。最终48个attention plans，0淘汰、0fallback，最大workspace433.5 MiB。

`enable_fused_ops`启用scale融合和local maxima；共享原生预设改用双1280 MiB缓存、`allocator_release_mib=13312`。CLI和Config WebUI已接通该池参数，未修改用户面板状态。Muon workspace仍512 MiB、Python复用池仍256 MiB。最终预设保持LR=0.0003；生产对照沿用LR=0.0005。

最终运行`native_production_20261028_225113_36440`保存到`D:/AI/Instinct/out/native_cublas/`和`D:/AI/Instinct/checkpoints/native_cublas/`。日志、命令、逐窗口计时与显存、误差门槛和保存核验均保留：

- [生产A/B汇总](../../experiments/results/cublas/gaps_validation_20261028/production_comparison.json) · [最终完整运行](../../experiments/results/cublas/gaps_validation_20261028/production_combined.json)
- [scale及池阈值剖面](../../experiments/results/cublas/gaps_validation_20261028/comparison.json) · [多分组剖面](../../experiments/results/cublas/gaps_validation_20261028/multigroup_comparison.json)
- [attention通过记录及显式阈值](../../experiments/results/cublas/gaps_validation_20261028/local_attention_accepted_parity.json) · [scale GPU检查](../../experiments/results/cublas/gaps_validation_20261028/scale_parity.json)
- [CPU检查](../../experiments/results/cublas/gaps_validation_20261028/cpu.log) · [最终CLI/WebUI检查](../../experiments/results/cublas/gaps_validation_20261028/final_cli_ui.log)

## GPU故障与当前进程

故障运行标识：`native_production_20261028_185051_7892`；PID **7892**。已记录最后一次成功同步的微步160，随后保存没有完成。对应的当前weights/resume文件没有生成。

之后的只读诊断中，`nvidia-smi`报告 `GPU is lost`，新的CUDA进程在`cuInit`处返回错误1；CPU测试中的Torch也无法枚举CUDA设备。尚未确定故障原因。没有终止原进程、重置GPU或重启机器。旧checkpoint不能替代本轮当前进度的保存证明。

19:22只读复查时GPU恢复可用，原PID已不存在；用户确认没有其他显卡操作。代理未终止该进程、未reset或重启，故障及退出原因仍未确定。原始报告保持历史状态，单独的recovery audit记录恢复观察，不把旧运行改成完成。之后新运行`native_production_20261028_193525_37236`完成并保存核验，不代表旧运行的未保存状态被恢复。

## 证据

- [生产原始报告](../../experiments/results/cublas/optimization_20261028/production_rows8_ws1024.json) 和 [中断审计](../../experiments/results/cublas/optimization_20261028/interruption_audit.json)
- [Norm/RoPE数值结果](../../experiments/results/cublas/optimization_20261028/rope_rows8_parity.json)
- [Muon实际形状验证](../../experiments/results/cublas/optimization_20261028/muon_resident_shapes.json)
- [新融合CPU验证](../../experiments/results/cublas/optimization_20261028/norm_backward_cpu.log)
- [数据及续训验证](../../experiments/results/cublas/optimization_20261028/loader_cpu.log)

- [完成的生产验证](../../experiments/results/cublas/optimization_20261028/production_fused_asymmetric.json)
- [算子、缓存扫描和目标预算](../../experiments/results/cublas/optimization_20261028/tuning_analysis.json)
- [同步显存诊断](../../experiments/results/cublas/optimization_20261028/fused_memory.json)
- [恢复审计](../../experiments/results/cublas/optimization_20261028/recovery_audit.json)


## 15k执行结果

仍使用同一V2 Dense、生产Parquet、BF16＋mantissa、tensorwise FP8、累积8、双1280 MiB ckpt4缓存、512 MiB Muon workspace和13 GiB异步池保留阈值。4个更新预热，16个更新测量：128微步、1,518,127有效预测目标。窗口包含Rust/mmap数据读取、H2D、前后向和Muon更新；缓存已构建，不包含启动、预热和最终保存。

| 方案 | 有效目标/s | 测量秒数 | native live峰值GiB | 设备采样峰值GiB | Muon ms/更新 |
|---|---:|---:|---:|---:|---:|
| 本轮重新运行的旧策略 | 14437.8041 | 105.1494388 | 12.737078 | 15.333305 | 503.786 |
| 后半块残差融合＋cast/amax＋自适应分组 | 14904.7141 | 101.8554930 | 12.737078 | 15.389946 | 447.511 |
| 完整残差融合＋权重scale缓存 | **15080.6001** | **100.6675458** | **12.736975** | **15.389946** | **448.278** |

本轮A/B吞吐提高4.4522%，时间减少约35.015 ms/微步。其中基线Muon阶段偏慢，差额约55.51 ms/更新、摊销6.94 ms/微步；因此不能把整段收益都归因于新融合。相对上一轮14603.2725 TPS为约3.27%，但它不是本轮同一时段A/B。native live峰值减少约0.00081%，设备级采样峰值增加约0.3694%；NVML属于设备采样且包含reserved，不等于进程allocated。

新增策略：

- RMSNorm反向在最后一条输入梯度边到达时，合并已存在的单条残差梯度。包括checkpoint输入叶子；保留FP32求和次序、原有dtype舍入和请求的中间梯度。复杂共享或未完成的输入边仍使用通用路径。
- FP32→BF16梯度转换同时统计实际BF16值的amax；保留统计缓冲的复用期，避免后续独立扫描。
- 每个BF16矩阵参数仅保留32字节的scale/倒数存储（本模型合计7168字节），不缓存完整FP8权重。优化器更新、Engine.copy、直接host上传和捕获的AdamW更新均使缓存失效。图构建时仍记录自己的scale计算，避免后续replay使用旧值。
- 连续文档分组最多8篇，按padding token数加每组4096 token等效固定成本做动态规划；不改变文档顺序和绝对ragged offsets。保留1024长度粒度和64个plan上限，160微步descriptor padded tokens由4213760降到3480576；48个plan、0淘汰、0fallback。固定成本是调度启发式，不能解释为GPU计时。

固定权重原始两批样本（每桶3次预热、3次无profiler计时，另各一次CUPTI）：

| 指标 | 基线短/长桶 | 全部融合短/长桶 |
|---|---:|---:|
| 墙钟中位数ms | 764.82 / 835.73 | 748.92 / 821.59 |
| 其他算子GPU ms | 196.17 / 196.19 | 183.31 / 182.71 |
| kernel数量 | 5402 / 5402 | 4314 / 4314 |
| gradient_sum数量 | 65 / 65 | 1 / 1 |

固定样本不含更新，权重scale在预热后持续命中；生产每次实际更新后必须刷新，所以不能把全部固定样本收益直接外推。CPU完整相关检查56项通过；缓存生命周期与执行选项22项通过（有重叠）。全部融合16个GPU loss/VJP用例通过，loss一致，最坏相对L2约2.71e-7。自适应attention的8个用例及文档/因果隔离通过，最坏相对L2约7.05e-4，小于用户已接受的0.0011；没有扩大阈值。

运行native_production_20261028_234102_36104已保存当前weights与完整resume；结果中的verified_current_state为true。上述三个生产运行均正常完成并保存核验。

- [生产基线](../../experiments/results/cublas/target15k_20261028/production_baseline.json) · [第一阶段](../../experiments/results/cublas/target15k_20261028/production_fusions_adaptive.json) · [15080完整结果](../../experiments/results/cublas/target15k_20261028/production_all_fusions.json)
- [固定样本剖面](../../experiments/results/cublas/target15k_20261028/fixed_comparison.json) · [融合GPU验证](../../experiments/results/cublas/target15k_20261028/all_fusions_parity.json) · [attention验证](../../experiments/results/cublas/target15k_20261028/adaptive_parity.json)
- [CPU融合检查](../../experiments/results/cublas/target15k_20261028/cpu_all_fusions.log) · [生命周期检查](../../experiments/results/cublas/target15k_20261028/cpu_cache_lifecycle.log)


### 扩大窗口与复现

相同配置再次独立运行，4个更新预热、32个更新测量：**3,055,038有效目标 / 200.8444416秒 = 15,210.9661 TPS**，共256测量微步、288总微步。其前128测量微步为15,081.4158 TPS，与上一轮15,080.6001 TPS基本一致；更长窗口桶组成不同，不能把15,211与15,081的差额当成额外代码收益。

该轮native live峰值12.736976 GiB；设备采样峰值15.421196 GiB、峰值时free约0.499702 GiB。Muon平均448.557 ms/更新。48个attention plans、0淘汰、0fallback；最大workspace仍433.5 MiB。整个运行324.105秒，启动、预热和最终保存/核验未计入15,211 TPS的窗口分母。

最终运行`native_production_20261028_234517_38424`的当前weights和完整resume均保存并逐数组核验通过，`status=complete`、`verified_current_state=true`。原生`enable_fused_ops`已默认启用本轮四项策略，保留显式实验覆盖；共享CLI/WebUI仍通过既有`fused_ops`开启，不增加第二份配置目录或修改用户面板状态。最终CPU/UI检查24项通过；checkpoint重新加载使scale缓存失效、Muon精确状态恢复、暂停边界保存4项通过。

- [扩大窗口完整结果](../../experiments/results/cublas/target15k_20261028/production_extended.json) · [本轮生产汇总](../../experiments/results/cublas/target15k_20261028/production_comparison.json)
- [最终CPU/UI检查](../../experiments/results/cublas/target15k_20261028/final_cpu_ui.log) · [续训与缓存恢复检查](../../experiments/results/cublas/target15k_20261028/final_resume.log)


## 原生与Torch模块耗时对照（15,211 TPS阶段）

本节只读取已完成的结果与trace，未重新启动训练或GPU剖面。当前生产记录为 **15,210.9661 TPS**：3,055,038有效预测目标 / 200.8444416秒，256个测量微步、32次累积更新，之前另有32个预热微步。结果状态为complete，当前完整状态保存核验通过。[生产原始记录](../../experiments/results/cublas/target15k_20261028/production_extended.json)

### 同输入固定样本的模块对照

单位为 **ms/微步**，一次微步包括前向和反向。短桶为4×3072，长桶为3×4096；两边读取同一份identical_batches.npz，有效预测目标分别为12,274和11,261。模型为32层、H1536、FFN5504、Q/KV16/4、head_dim96、词表6400的V2 Dense，GPU均为RTX 5070 Ti，保留真实优化器和梯度常驻内存。

**这是一份现有剖面的结构对照，不是15,211 TPS生产窗口的直接逐模块拆解，也不是完全同配置、同时段的严格A/B。** 原生采用10月28日最新all_fusions剖面，Torch采用10月27日已完成的编译版剖面。两边权重初始化实现不同；原生ckpt缓存为1280/1280 MiB，Torch为1792 MiB；attention实现和重算量不同。原生剖面开启了新梯度融合、cast/amax和权重scale复用，但其attention_group_penalty=0，**尚未开启生产记录中的4096自适应分组**。不将生产分组收益套入这张表。

| 模块/指标 | Torch短桶 | 原生短桶 | Torch长桶 | 原生长桶 |
|---|---:|---:|---:|---:|
| GEMM | 330.30 | 329.44 | 340.91 | 328.29 |
| Attention 合计，含 Q/K Norm、RoPE 和统计转换 | 263.59 | 215.38 | 348.09 | 291.20 |
| 其他算子：SwiGLU、量化、Norm、残差等 | 195.70 | 183.31 | 192.16 | 182.71 |
| memset | 0.04 | 5.62 | 0.04 | 2.82 |
| memcpy | 0.0063 | 0.0037 | 0.0062 | 0.0025 |
| **GPU 活动合计** | 789.63 | 733.76 | 881.20 | 805.02 |
| 计算窗口内空档 | 9.66 | 9.51 | 14.64 | 9.90 |
| 首个至最后一个计算 kernel 的跨度 | 799.29 | 743.27 | 895.84 | 814.92 |
| 无 profiler 墙钟中位数 | 796.31 | 748.92 | 889.80 | 821.59 |
| Kernel 数量（不含 memcpy/memset） | 4,234 | 4,314 | 4,297 | 4,314 |

GPU活动合计是trace内各活动持续时间之和；计算跨度和空档统一用首个计算kernel开始至最后一个计算kernel结束作为边界，排除不对称的首尾loss读取/控制段。边界外的少量copy仍计入完整活动合计，因此后者与空档相加可能有微小边界差。无profiler墙钟中位数来自独立测量，不能与另一轮单次trace强行相加。原生每桶3次预热、3次无profiler测量，另各1次CUPTI；不是每个模块都采样了3次。

Torch的FlexAttention模板可以融合Norm/RoPE，故主表将attention相关活动合并。当前原生对应阶段可进一步拆为：

| 原生attention阶段 | 短桶ms | 长桶ms |
|---|---:|---:|
| Q/K Norm＋RoPE | 16.13 | 16.12 |
| cuDNN attention 前向 | 77.75 | 72.21 |
| Attention 反向及统计格式转换 | 121.49 | 202.87 |

在这两份固定样本剖面中，原生“其他算子”已比Torch少约12.39/9.45 ms，GPU计算区间空档约9.51/9.90 ms。原生kernel数仅比Torch多80/17个；不能再沿用旧版本6426个kernel的差距。差异仍包含上述缓存、attention实现及运行时条件，不能全部归因于融合。

来源：[原生模块汇总](../../experiments/results/cublas/target15k_20261028/fixed_comparison.json)、[原生探针设置](../../experiments/results/cublas/target15k_20261028/all_fusions.json)、[原生短桶trace](../../experiments/results/cublas/target15k_20261028/all_fusions.window12.cupti.jsonl)、[原生长桶trace](../../experiments/results/cublas/target15k_20261028/all_fusions.window13.cupti.jsonl)、[Torch模块汇总](../../experiments/results/cublas/comparison_20261027/comparison.json)、[Torch探针设置](../../experiments/results/cublas/comparison_20261027/torch_actual_recipe.json)。Torch计算窗口空档由其bucket0/1.trace.json重读计算。

### 原生“其他算子”的内部组成

下面按原生kernel标签互斥归组，四行之和等于主表的其他算子耗时；不是额外耗时。Torch同类工作往往融合在同一kernel，现有trace不能无歧义地再拆成这些相同边界，故不制造逐小项Torch对照值。

| 原生内部工作 | 短桶ms | 长桶ms |
|---|---:|---:|
| FFN SwiGLU统计、前反向值计算及其量化 | 71.72 | 71.64 |
| Norm及与其融合的残差梯度 | 53.35 | 53.20 |
| 其他FP8准备：cast/amax、scale、量化与转置 | 42.20 | 41.96 |
| 其余逐元素、复制、清零、embedding和loss | 16.04 | 15.91 |

Norm组包含独立的Norm权重梯度归约；最后一组的清零是Triton fill kernel，主表memset是CUDA memcpy/memset活动，两者没有重复计数。固定权重预热后，权重scale缓存持续命中；真实训练每次更新后会失效并重新计算，所以该表的FP8准备成本不能直接代表所有生产微步。

### memset与生产优化器阶段

短桶cuDNN attention反向的32次memset合计 **5.5871 ms、4.5 GiB累计写入**，长桶为 **2.7848 ms、2.25 GiB累计写入**。短桶8篇文档×3072 padding长度，长桶3篇×4096；乘以32层、16头、96维和4字节，恰好解释两种清零量。这里是重复写入流量，不是同时常驻显存大小。其余cuBLAS memset只有约0.0373/0.0368 ms。原生copy为1024字节metadata H2D和4字节loss DtoH。[原始字节与归因记录](../../experiments/results/cublas/target15k_20261028/fixed_comparison.json)

当前15,211 TPS生产窗口可直接支持的阶段计时如下：

| 生产计时项 | 当前原生 | Torch现有参考 | 可比性 |
|---|---:|---:|---|
| 整个窗口平均墙钟 | 784.55 ms/微步 | 无同窗口数据 | 原生含loader、H2D、前后向、Muon和控制开销 |
| Muon/伴随参数更新阶段 | 448.56 ms/更新 | 384.52 ms/更新 | Torch为10月26日合成数据更新探针，非本窗口A/B |
| 更新阶段按累积8摊销 | 56.07 ms/微步 | 48.06 ms/微步 | 仅作各自摊销展示，不能加入上面的固定样本GPU合计 |
| 原生窗口扣除所记录更新阶段 | 728.48 ms/微步 | 无同窗口数据 | 仍含数据、前后向及CPU/同步等，不能称为纯GPU计算 |
| 数据加载独立耗时 | 本轮未单独计时 | 无同窗口数据 | 不从剩余时间反推loader耗时 |

生产测量包含212个短桶和44个长桶，文档长度组成与固定样本不同，不能用两个固定样本的均值还原生产窗口。生产已启用自适应分组，完整运行48个plan、0次淘汰和0次fallback。Torch优化器参考来自[历史完整更新汇总](D:/AI/Instinct/experiments/results/v2_14k/final_comparison.json)及[合成更新原始记录](D:/AI/Instinct/experiments/results/v2_14k/optimized_updates.json)；其口径不含loader和保存，测量方式也不同。

## 16k目标与有界选型（2026-10-29）

最终保持原有精确累计，生产256测量微步为 **15,596.7330 TPS**；同样本窗口旧版为15,210.9661，提升 **2.5361%**，节省 **19.4049 ms/微步**。**尚未达到16,000 TPS**：按本窗口有效目标数，仍需减少 **19.2848 ms/微步**。43 ms是验收目标预算，不是已经实现或可以直接相加的收益。

### 最终生产结果与保存

配方仍是32层V2、BF16隐藏参数/momentum、uint16 mantissa、FP8 auto、累积8、512 MiB Muon workspace。数据为同一生产Parquet，从group0末64批跨到group1；4更新预热后测32更新。使用已建Rust packed缓存，含loader、H2D、前后向和更新，不含启动/预热/最终保存；本轮未重建冷缓存。

| 项目 | 上一轮256微步 | 本轮256微步 |
|---|---:|---:|
| 有效预测目标 | 3,055,038 | 3,055,038 |
| 测量墙钟 | 200.844442 s | 195.876791 s |
| 有效TPS | 15,210.9661 | **15,596.7330** |
| 相同前128微步TPS | 15,081.4158 | **15,475.9458** |
| Muon/伴随更新阶段 | 448.5574 ms/更新 | **415.4570 ms/更新** |
| 更新阶段摊销 | 56.0697 ms/微步 | **51.9321 ms/微步** |
| native live峰值 | 12.736976 GiB | **13.281898 GiB（+4.2783%）** |
| NVML设备采样峰值 | 15.421196 GiB | **15.243004 GiB（−1.1555%）** |
| 设备采样峰值时free | 0.499702 GiB | **0.677895 GiB** |

native live与NVML device-used口径不同，后者是20 ms采样下界，不能把二者相减作为可分配余量。60个plan、0次淘汰、0次workspace fallback；启动预热30种描述符几何耗时3.0254秒。没有用单个全短桶窗口的16,256.6 TPS冒充混合桶验收。完整脚本含启动、预热、保存和逐数组核验共310.756秒，不等于表中的测量窗口。

共完成288微步、36次真实更新，正常退出；最终loss=5.3650336266。weights和完整resume逐数组核验通过，`verified_current_state=true`，`torch_imported=false`：

- `D:/AI/Instinct/out/native_cublas/native_production_20261029_012616_33952_1536.npz`
- `D:/AI/Instinct/checkpoints/native_cublas/native_production_20261029_012616_33952_1536_resume.npz`

[最终生产原始记录](../../experiments/results/cublas/target16k_20261029/production_precise_extended.json) · [同窗口比较](../../experiments/results/cublas/target16k_20261029/production_comparison.json)。此表核验性能和保存，**不表示下文完整模型梯度重复性问题已经解决**。

### 选择算法与采用项

1. CPU先筛资源和形状。cuBLAS保留前两个启发式推荐，再考虑workspace和waves不同的候选，最多4项；waves是占用率代理，不冒充延迟预测。cuDNN对A/B/fallback启发式去重，限制workspace，仅保留最多4个受支持plan。FFN按元素存活数量、转置存储宽度和数据流选择短名单。文档分组的CPU动态规划保持连续文档顺序，兼顾padding与调用成本。
2. 粗测每项2次，只将基线与领先项交替复测3轮，每轮3次。未超过1%的改善保留基线；仅一个支持项时跳过比较。选择结果随shape/precision/workspace的现有plan缓存复用；**没有新增跨进程持久选型缓存**。
3. cuDNN长度粒度512、每组最多4文档、等效调度成本4096、plan上限64。消费训练数据前预热合法的有界描述符目录，使用零合成输入，不改参数/优化器/数据cursor。实测只有30个单候选plan和30个双候选plan。邻近tile配置尝试没有新增可用plan，因此不宣称得到新attention核心内核。CUDA/cuDNN官方支持[按plan执行与实测选择](https://docs.nvidia.com/cudnn/latest/developer/overview/)。
4. Muon长矩阵的prepare/apply改为32×32分块转置，保持舍入点及mantissa拆位不变；只对SM120已测的1536×1536×1536和384×384×1536对称运算使用BK128三角内核，其余继续cuBLAS。五步NS链的三种生产几何对cuBLAS逐位一致。
5. 通用FP8双布局量化tile 32→64。ckpt4保持O/LSE优先，用剩余预算先缓存K/V，再缓存gate/up。两桶预算1792 MiB，实际32层O/LSE 1176 MiB＋32层K/V 576 MiB＝1752 MiB；未超过预算。BF16 K/V投影的实测时间/缓存字节优于大FFN Tensor Core投影，不能只按理论FLOP/byte排优先级。

### 同输入固定权重剖面

同旧版的两份生产输入、真实常驻优化器/梯度、6次warmup、4个无profiler测量样本及2个CUPTI样本。缓存策略是优化变量：旧1280、新1792 MiB。旧数据是历史测量，不能当成同时运行的严格配对，也不能代替生产窗口。

| 项目（ms/微步） | 旧短桶 | 新短桶 | 旧长桶 | 新长桶 |
|---|---:|---:|---:|---:|
| 无profiler墙钟中位数 | 748.92 | **726.19** | 821.59 | **813.43** |
| GEMM | 329.44 | **323.66** | 328.29 | **322.84** |
| Attention＋Q/K Norm/RoPE | 215.38 | **200.79** | 291.20 | 292.09 |
| 其他kernel | 183.31 | **181.61** | 182.71 | **181.46** |
| memset | 5.62 | **4.31** | 2.82 | **2.75** |
| 同首尾kernel边界的GPU空档 | 9.51 | 10.11 | 9.90 | 9.84 |
| kernel数 | 4314 | **4570** | 4314 | **4250** |

短桶分组增加了attention调用数，但前向从77.75降到62.13 ms，少做padding的收益超过额外调用；kernel更少不是独立验收目标。K/V缓存减少64次重算GEMM。通用quantize_dual约节省1.24/1.36 ms；FFN statistics＋quantized_values仍约71.7 ms，**本轮没有兑现FFN 15 ms预算**。Norm对齐/行分组没有稳定收益，未启用；Muon摊销节省4.14 ms，未达到8 ms预算。无法把这两个固定样本按桶数量加权后冒充整段生产的分项计时。

[固定样本分项报告](../../experiments/results/cublas/target16k_20261029/fixed_final_comparison.json)

### 未采用项与数值边界

- FFN单纯tile/warp、矩形转置tile、暂存BF16中间值均未胜出。基线前向/反向约0.747/1.591 ms每层；暂存变为0.787/2.155 ms，增加的读写抵消了少算一遍。完整输出逐元素一致，但不能因减少计算次数就采用更慢的版本。
- 更大的Muon batch/workspace基本按矩阵数线性扩展；128 MiB cuBLAS workspace及本机cuBLAS13.4.0 DLL没有显示可采纳收益，最终保留32 MiB cuBLAS workspace和原13.2.1库。
- fast accumulation两层生产形状曾通过，但32层检查未满足原门槛，最终**不启用**。通用模式最坏相对L2为0.297/0.194，限定形状的短桶为0.265。进一步关闭调优并重复原精确基线，仍出现0.160/0.133；在同一对照设置下，限定fast候选为0.155/0.132，loss一致。因此较大梯度差异不能全部归因fast accumulation，完整32层FP8/cuDNN基线的重复性仍未解释。没有提高容差，也没有宣称完整模型梯度验收通过。
- 用户之前接受的attention相对L2 0.0011仍仅用于既有attention对照。本轮小模型attention分组/选型最大约0.000786，混合缓存16个loss/VJP用例通过，最坏约2.38e-7；这些局部/小模型证据不覆盖上述32层问题。

[FFN短名单](../../experiments/results/cublas/target16k_20261029/ffn_shortlist.json) · [Muon转置逐位校验](../../experiments/results/cublas/target16k_20261029/muon_io.json) · [五步NS链](../../experiments/results/cublas/target16k_20261029/muon_chain.json) · [混合缓存VJP](../../experiments/results/cublas/target16k_20261029/auxkv_parity.json) · [完整模型重复对照](../../experiments/results/cublas/target16k_20261029/full_v2_precision_control.json)

CPU聚焦检查108 passed、4个CUDA用例按标记跳过；最终预设/选型/实际WebUI命令25 passed，另有上述独立CUDA验证。旧原生分支不存在主工作区的config模板三组测试及relay schema生成器，因此使用本分支native UI/CLI契约测试，不借用主工作区测试冒充验证。没有停止或丢弃任何未保存训练进度；所有生产候选均完成各自保存核验，未采纳候选的状态也保留在ignored输出目录。

## 16k第二轮执行（2026-10-30）

256微步完整生产窗口达到 **16,045.5191有效目标/s**。基线与候选同为3,055,038个有效预测目标、212个短桶和44个长桶；从group0末64批跨到group1，4次更新预热后测32次更新。仍使用已建Rust packed缓存、BF16隐藏参数及momentum、uint16 mantissa、FP8 auto、累积8、512 MiB Muon workspace、32 MiB Lt workspace，未启用fast accumulation。基准LR保持5e-4，配方LR仍是3e-4。

| 项目 | 上一轮256微步 | 本轮256微步 |
|---|---:|---:|
| 有效TPS | 15,596.7330 | **16,045.5191（+2.8774%）** |
| 窗口墙钟 | 195.876791 s | **190.398203 s** |
| 墙钟/微步 | 765.1437 ms | **743.7430 ms** |
| 更新阶段 | 415.4570 ms/更新 | **334.5147 ms/更新** |
| 更新阶段摊销 | 51.9321 ms/微步 | **41.8143 ms/微步** |
| native live峰值 | 13.281898 GiB | **14.108333 GiB（+6.2223%）** |
| NVML设备采样峰值 | 15.243004 GiB | **15.477634 GiB（+1.5393%）** |
| 设备采样峰值时free | 0.677895 GiB | **0.443264 GiB** |

实际节省21.4007 ms/微步，其中更新阶段摊销节省10.1178 ms；没有把原25 ms预算当作实测结果。计时包含loader、H2D、前后向、日志和真实更新，排除启动、预热及最终保存。NVML为20 ms采样下界，不能把设备used当作进程allocated。

[生产原始记录](../../experiments/results/cublas/target16k_round2/production_final2816.json)中`status=complete`、`verified_current_state=true`、`completed_requested_windows=true`、`torch_imported=false`。共288微步/36更新，最终loss=5.3673849106；weights及完整resume逐数组核验通过：

- `D:/AI/Instinct/out/native_cublas/native_production_20261030_004657_6800_1536.npz`
- `D:/AI/Instinct/checkpoints/native_cublas/native_production_20261030_004657_6800_1536_resume.npz`

### 采用的改动

- **cuBLAS对称分块Muon**：以device pointer-array将Gram的下三角块合为一次Lt调用，再镜像补齐；不依赖SM120不支持的BF16 fill-mode。FFN batch3采用64分块，batch2保持实测更快的512分块；1536方阵batch8的Gram和batch3/8的多项式也使用已验证分块。batch1保留旧路径，因为分块候选的五步链曾达到0.00964相对L2。最终五种生产形状的五步NS链与原路径逐位一致。动量、缩放、NS系数/次数及mantissa舍入不变，指针表复用且按stream隔离。
- **层边界融合**：原始checkpoint前向把层末FP32残差加法与下一层input RMSNorm合并，所有层共享一个BF16暂存区；VJP仍重建原操作。新token预算不再新增一套持久缓冲，使用普通路径。E4M3激活仅在重算阶段同时写出dW所需的转置布局；固定形状减少96次独立pack，单次链从0.1478降至0.1414 ms，未夸大为数毫秒的独立收益。
- **缓存生命周期**：`Optimizer.backward`完成该层反向后，释放只属于该层的selective输出；拒绝释放外部、视图或pinned参数地址。手动`Engine.vjp`保留原行为，已消费的optimizer图明确拒绝二次使用。两桶预算改为2816 MiB，实际32层O/LSE 1176 MiB＋32层K/V 576 MiB＋顶部4层gate/up 1032 MiB＝2784 MiB；复用池改为128 MiB。较大缓存通过缩短生命周期控制峰值，并非直接堆积所有缓存。
- **前向耗时模型**：最多9个描述符锚点，CPU拟合非负四项耗时模型；连续分组动态规划只在预热目录中选择，比预测基线改善不足1%就不改组。只改变前向分组；反向保留原分组。双向改组曾超过0.0011数值门槛，因此未采用。原有cuBLAS/cuDNN短名单仍最多4项，不穷举所有候选。

### 数值验证、未采用项及数据端

[层边界/量化回放](../../experiments/results/cublas/target16k_round2/boundary_parity.json)的16个BF16/FP8、ckpt3/4、长度137/512、缓存0/8 MiB用例通过；loss一致，最坏相对L2约2.38e-7。[缓存释放](../../experiments/results/cublas/target16k_round2/cache_release_parity.json)的四个双微步累积用例隐藏层梯度逐位一致；embedding沿用既有原子归约容差，最坏整体相对L2约2.06e-8。[选定Muon五步链](../../experiments/results/cublas/target16k_round2/muon_selected_chain.json)五种形状逐位一致。[前向分组](../../experiments/results/cublas/target16k_round2/attention_forward_cost_parity.json)含交错forward、文档和因果隔离，最坏相对L2约0.001008，保持用户已接受的0.0011门槛。

SM120 CuTe/FROST反向已完成Torch-free packed staging、执行和scatter对照，但未采用：短桶4.008→4.909 ms/层，长桶7.209→6.980 ms/层；dQ/dK相对L2约0.0032/0.0039，超过门槛。实验依赖只安装在ignored的`.cache/native-dsl`；生产路径不需要这些依赖。[原始结果](../../experiments/results/cublas/target16k_round2/frost_attention_staged.json)

完整32层基线的重复性仍未解决。同一context、固定参数与相同输入重复两次，FP8全局梯度相对L2为0.0960/0.0991，量化布局候选为0.0970；loss相同。关闭FP8后仍有0.00410全局相对L2、0.00674最大参数相对L2。结果支持FP8放大既有数值扰动的可能性，但尚未证明完整根因；**吞吐验收与局部一致性检查不等于完整模型数值验收已闭环**。[FP8重复记录](../../experiments/results/cublas/target16k_round2/repeatability_layout.json) · [BF16重复记录](../../experiments/results/cublas/target16k_round2/repeatability_bf16.json)

独立128微步诊断把生产时间拆为：墙钟764.27 ms/微步、前后向CUDA事件跨度712.08 ms、更新摊销51.76 ms，其余0.43 ms。loader.next主机时间0.514 ms、H2D staging主机时间0.103 ms，与GPU重叠，不能再加到CUDA跨度上。该诊断使用较早的量化布局候选，15,518.49 TPS，完整保存核验通过；用于定位数据端而非冒充最终吞吐。[端到端诊断](../../experiments/results/cublas/target16k_round2/e2e_audit.json)

此前`production_layout.json`只记录到224微步，后续发现进程已不存在且无匹配checkpoint，未计作完成/验收；不从部分JSON补造最终状态。本节两次已核验运行的状态与该中断运行分开保存。

最终聚焦CPU/实际WebUI检查122 passed、4个CUDA用例按标记跳过；另有上述独立GPU数值/性能验证。native bridge构建通过，保留ABI2旧入口，缺少pointer-array扩展时回退原Muon路径。共享配方已更新，实际UI启动命令检查覆盖2816 MiB桶预算与128 MiB复用池。

### 扩大窗口与选型稳定性

不能只用上面的最好256微步窗口宣布稳定达标。随后按共享默认配置测512微步，6,102,114个有效目标得到15,790.8017 TPS，相同前256微步为15,845.5778 TPS。更新阶段仍约335 ms/更新；主要回退来自两个FFN GEMM的选择：forward `[1536,12288,5504]` 和 dW `[5504,1536,12288]` 的短计时受到明显漂移影响。一次确认序列从0.898降至0.628 ms，仅靠三次中位数不足以排除该变化。此处不能断言漂移全部来自时钟，因为当时没有同步采集时钟证据。

有界选型仍最多筛4项，但多候选比较前增加约100 ms基线预热；确认样本极差超过5%则重新预热并复测一次，持续不稳定就保留启发式基线且输出警告。单候选不做无意义的比较；没有增加穷举或跨进程持久缓存。

修复后的独立512微步从空闲P8/202 MHz启动，结果为 **15,979.4566 TPS**（381.872435 s），相同前256微步为 **16,037.8873 TPS**。两个FFN选择恢复正常，更新阶段336.2546 ms/更新；512微步含419短桶、93长桶，不能用前256的桶比例代替。该结果距16k仍差约0.957 ms/微步。两次512运行均完成544总微步、保存并逐数组核验当前状态；没有把部分窗口计作完整结果。

后一轮native live峰值仍14.108333 GiB，但NVML设备采样峰值为16,954,400,768字节（约15.7900 GiB），当时free只有140,533,760字节（约0.1309 GiB）。设备采样含运行环境占用且是采样下界，不能把较早的0.443 GiB余量当作始终可用的缓存预算。

[扩大窗口原始记录](../../experiments/results/cublas/target16k_round2/production_shipped512.json) · [稳定选型原始记录](../../experiments/results/cublas/target16k_round2/production_stable512.json) · [选型修复后的Muon五步链](../../experiments/results/cublas/target16k_round2/stable_tuning_chain.json)

另外两类候选没有采用：FFN分阶段写回与插入barrier仍逐位一致，但约1.579 ms，比基线1.574 ms略慢；复用池128→160 MiB只多命中24次分配，固定样本也未加速。基准报告原子写盘约2.306 ms/更新（0.288 ms/微步），保持原测量与持久化方式，没有从分母扣除它来宣称16k。

临时池命中是纯主机元数据操作，已移除其中重复的`cuCtxSetCurrent`；实际CUDA分配、launch和cuDNN准备仍绑定对应context。两个live context交错测试以及四组梯度累积检查通过。固定样本暂未显示可归因的收益，生产验证单列，不把减少API次数自动换算成吞吐提升。

## 共享cuDNN运行时资源与扩大验收（2026-10-31）

去除冗余context绑定后的独立512微步结果是15,979.8448 TPS，完整保存核验通过；与15,979.4566几乎相同，不能归因吞吐收益。

随后256长度粒度、128 plan上限候选在相同固定生产输入上，短/长桶墙钟中位数由718.541/810.197降到713.941/807.536 ms；cuBLAS GEMM选择相同。前向仍用实测耗时模型，反向保留原分组代价模型。原生live峰值基本不变，但更多cuDNN描述符增加了运行时显存占用。

该候选的第一次生产测试按用户请求暂停在528总微步，即496测量微步；已测5,911,075目标/368.076006秒=16,059.3869 TPS。`status=paused_verified`且`completed_requested_windows=false`，当前weights和完整resume已核验，**不能将它计作完整512微步验收**。[暂停记录](../../experiments/results/cublas/target16k_round2/production_quantum256_512.json)

### 每个plan重复创建handle的常驻开销

代码原先为每个SDPA plan调用一次`cudnnCreate`。新可选ABI让同一backend、主机线程和执行流的plan共享handle，并保留context/stream匹配检查；不同流独立分配。plan与handle通过C++共享所有权关联，先释放所有图和plan，再释放handle。旧ABI继续使用原路径。该所有权方式符合[NVIDIA的handle生命周期建议](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html)，但节省多少必须以本机测量为准。

独立Torch-free进程预热同样120个合法描述符，临时池128 MiB、驱动异步保留阈值0，预热后同步再读取设备free：

| 项目 | 独立handle | 共享handle |
|---|---:|---:|
| plan数 | 120 | 120 |
| handle数 | 120 | 1 |
| 预热后设备常驻增量 | 608 MiB | 250 MiB |
| 相对减少 | — | **358 MiB，58.88%** |

这是同条件下目录预热的设备内存增量，不是模型native live或整段训练NVML峰值。没有把358 MiB直接当成总训练峰值下降。新bridge构建通过，旧创建/运行入口保留。共享handle的8组attention验证以及文档/因果隔离通过，最坏参数梯度相对L2为0.000574918，沿用0.0011门槛。

[独立handle显存](../../experiments/results/cublas/target16k_round2/handles_legacy.json) · [共享handle显存](../../experiments/results/cublas/target16k_round2/handles_shared.json) · [共享handle数值检查](../../experiments/results/cublas/target16k_round2/attention_shared_parity.json)

### 波动诊断及候选排除

首轮使用recipe学习率3e-4的共享handle生产诊断出现回退，已按保存流程暂停在312总微步，完整状态核验通过；它不是完整512微步结果。这次选型阶段与CPU测试并行，后续还观察到多个QQ GPU进程及空闲设备占用变化，控制条件不足，不能用它单独归因共享handle的吞吐变化。

计时记录同时暴露了有界选型的判定缺陷：两个大FFN dW候选的最慢确认样本，仍比基线最快样本快，但原规则因为样本漂移超过5%而强制退回较慢基线。修复后仍最多4候选、一次重试；漂移持续时，仅在`max(候选)<min(基线)×0.99`时接受候选，否则保留启发式基线。确认区间重叠及不重叠的测试均覆盖，不用增加穷举次数解决这个问题。

固定探针还修复了初始化顺序：candidate配置须在创建NativeAttention前应用，共享handle才实际生效。早期`fixed_handle_shared.json`的运行时字段仍为`shared_handles=false`，因此不能当作共享handle对照。

RMSNorm分组与多行反向短名单没有取得足够收益；未修改生产默认内核，实验补丁和原始数据保留在ignored结果目录。更低2560 MiB缓存候选也仅作为本机时段内的固定输入诊断，尚未据此更改共享recipe。

### 最终窗口

用户确认其他GPU应用保持空闲后，从相同数据起点独立测量512微步，使用recipe学习率3e-4、256长度粒度、128 plan上限及共享handle；记录loader/H2D主机耗时和主流CUDA跨度。验证期间不并行运行CPU测试或其他GPU基准。

| 项目 | 完整窗口结果 |
|---|---:|
| 测量微步/真实更新 | 512 / 64 |
| 有效预测目标 | 6,102,114 |
| 窗口墙钟 | 380.113418 s |
| 有效TPS | **16,053.4033** |
| 短桶/长桶数 | 419 / 93 |
| 平均墙钟 | 742.4090 ms/微步 |
| 前后向主流CUDA跨度 | 699.4979 ms/微步 |
| 更新阶段 | 338.3370 ms/更新，摊销42.2921 ms/微步 |
| 两阶段之外的墙钟 | 0.6190 ms/微步 |
| loader.next主机耗时 | 0.6826 ms/微步 |
| H2D staging主机耗时 | 0.0998 ms/微步 |
| native live峰值 | 14.108333 GiB，相比此前同窗口 **0.00%增量** |
| NVML设备采样峰值 | 15.483471 GiB，相比此前同窗口 **+0.038%** |
| NVML峰值时free | 0.437428 GiB |

loader和H2D主机耗时与GPU重叠，不能再加到CUDA跨度上；CUDA事件跨度包含GPU活动之间的空档。device-used是采样下界，且不同运行时段的后台占用可变，不等于进程allocated。显存参照是`production_context512.json`；其LR为5e-4，本次为recipe的3e-4，因此吞吐数值不冒充单一改动的严格A/B。两次都是同一419/93桶组成、6,102,114个有效目标。

该窗口使用已建Rust packed缓存，包含加载、H2D、前后向、日志和真实更新，排除启动、32微步预热及最终保存；没有重建冷缓存。完整脚本含启动、保存及逐数组核验共511.8681秒。共544总微步、68更新，最终loss=4.0522084236。

[最终窗口记录](../../experiments/results/cublas/target16k_round2/production_recipe_confirm512.json)与[计时/显存汇总](../../experiments/results/cublas/target16k_round2/final_verification_summary.json)均已完成：`status=complete`、`completed_requested_windows=true`、`verified_current_state=true`、`torch_imported=false`。120 plan、0次淘汰、0次workspace fallback；共享handle已实际启用，最终GEMM选型无持续不稳定回退警告。

- weights：`D:/AI/Instinct/out/native_cublas/native_production_20261031_001510_21700_1536.npz`
- 完整resume：`D:/AI/Instinct/checkpoints/native_cublas/native_production_20261031_001510_21700_1536_resume.npz`

新默认已接入`enable_fused_ops`，显式探针覆盖仍优先。最终策略、handle所有权/旧ABI回退、有界选型及实际WebUI命令聚焦检查54 passed；此前更广的CPU检查102 passed、4个CUDA用例按标记跳过。更新选型规则后，五种生产Muon几何的五步NS链再次逐位一致，[原始结果](../../experiments/results/cublas/target16k_round2/final_shared_muon_chain.json)。本次确认完整测量窗口超过16k；前文完整32层基线梯度重复性问题仍未闭环，局部数值检查及吞吐验收不能替代该项结论。
