# RecurrsiveInstinct：激活显存与深循环学习能力研究计划

日期：2026-10-19。状态：已完成代码初查、隔离激活重算原型，以及小模型/完整 RecurrsiveInstinct 配置的前向反向显存探针；正式训练消融待执行。

用户确认的症状是：**循环增加后不收敛或效果变差**。研究同时处理显存限制与深循环学习能力，不把“没有 NaN”当作成功。

## 1. 研究问题和当前证据

定义 `R` 为总前向循环次数，`K` 为保留反传的末尾循环次数，`L` 为 recurrent core 的独立层数。固定 `R` 改 `K` 与固定 `K` 改 `R` 是两组不同实验。

当前 `trainer/config_recurrsive_instinct.json` 是 Dense RecurrsiveInstinct：hidden=1248，P/R/C 物理层数为 2/4/2，目标循环次数 32，反传 8 轮，上限 256，夹心 RMSNorm、输入注入、随机深度，checkpoint mode=2。总前向深度在 R=32 时为 132 层，反传覆盖其中最后 32 个共享层调用及 prelude/coda 的相关路径。

代码初查基于 HEAD `49a06d5` 的当前工作副本；运行时另存源码哈希，不能仅凭 HEAD 代表有未提交修改的工作区。设备只读查询为 RTX 5070 Ti、16303 MiB、驱动 616.56；Python 为 E:/miniconda3/python.exe，PyTorch 2.13.0+cu132。该配置作为研究对象，**不能冒充用户失败运行的实际配置**：当前保存的 WebUI 顶层选择是 standard，仍残留 loop 字段。正式质量对照前需要从失败日志/检查点导出完整配置与数据游标。

| 已确认实现 | 研究含义 |
| --- | --- |
| `model/model_recurrsive_instinct.py` 的 `_resolve_steps` / `forward` 已使用 no-grad 前缀和末尾 K 轮反传 | 固定 K 后，R 的增加主要增加计算；在 standard residual、无缓存/中间输出条件下，不应导致保存的 CUDA 激活随 R 线性增长。若实测增长，应定位引用、编译池和缓存，而不是继续缩 K。 |
| mode=2 在每个有梯度的 core iteration 外做 non-reentrant checkpoint | 已丢弃轮内大部分中间值，但保留循环边界；不是尚未开启 checkpoint。 |
| 当前 mode=2 的 prelude/coda 经 `_run_block` 直接执行，`InstinctBlock.forward` 没有 mode=2 整块 checkpoint 分支 | side blocks 的激活仍跨越循环存活，是优先验证的改进点。 |
| `_compute_lm_loss` 先形成完整 logits，再移位和计算 CE | head/CE 的保存值、复制和瞬时峰值需单独归因。vocab=6400，不能假定它必然主导。 |
| 输入适配器为无约束 `Linear(2h,h)` | 状态传递增益与输入方向干扰是稳定性候选原因；尚未证实是本次故障根因。 |
| 改 `loop_iters` 会改变输出投影初始化标准差 | “相同种子、不同 loop_iters”并不保证相同初始权重；深度对照必须复用完全相同 state_dict，并固定初始化参考深度。 |

初步显存模型：`peak ≈ 参数/梯度/优化器 + side 激活 + K 相关边界/轮内保存 + 单轮重算工作区 + head/CE + 编译/分配器开销`。各项峰值不必同时发生，不能直接求和当成实测峰值。

## 2. 减少保存激活的方案分级

优先采用保留同一有限深度目标、同一截断位置的重算方案。激活 checkpoint 用前向重算换显存，non-reentrant 实现还会按需要提前停止重算；实际成本需测量。[PyTorch 方法说明](https://pytorch.org/blog/activation-checkpointing-techniques/)

| 方案 | 保存什么 / 重算什么 | 梯度语义 | 优先级 |
| --- | --- | --- | --- |
| 补 prelude/coda 整块 checkpoint | 保存 side block 输入，反向重算 attention/FFN/Norm | 保持当前截断目标；验证 RNG、参数与输入梯度 | A |
| LM head+CE checkpoint；必要时 chunked CE | 保存最终 hidden 和标签；反向再计算 logits/CE | checkpoint 可保持目标；分块必须保持 shift、mask 和全局 token reduction | A，取决于峰值归因 |
| 循环分段 checkpoint | 每 g 轮保留一个边界；反向重建段内计算图 | 保持目标，但段内重建可能抬高反向峰值 | B，必须测完整 backward |
| 嵌套 checkpoint | 外层每 g 轮存边界，内层每轮继续重算 | 保持目标；更多重算与 Python/编译开销 | B；优先 g=2/4，按 K 调整 |
| 选择性重算 | 按算子保存/重算，结合实际 Flash/SDPA、FFN 和 Norm 开销 | 保持目标，取决于实现正确性 | B；先拿 trace 再选 policy |
| 激活 CPU offload | 需要反传的张量暂存系统 RAM，反向拷回 GPU | 原 dtype 无损搬运可保持值；增加 PCIe 流量 | C；只对选定激活，避免盲目搬参数副本 |
| 更小 K / 单步梯度 | 只保留更短反传链 | 改变梯度估计与长期 credit assignment | 对照组，不作为默认修复 |
| FP8/INT8 激活压缩 | 保存量化激活、反向近似恢复 | 有损；可能进一步影响深循环学习 | 后置，与优化器低精度实验分开 |
| 可逆层、DEQ/隐式梯度 | 重构状态或固定点求导 | 大幅改变架构/目标及收敛假设 | 暂不作为第一阶段方案 |

不能把任意 saved tensor 丢弃而仍期望正确梯度。RMSNorm、门控 FFN、attention 等的 backward 需要相关输入/统计量，应重算、无损搬运或采用明确的近似方法。现有普通 Transformer block 也不能直接视为可逆变换。

MELT 等常量 KV 内存工作主要改变跨循环缓存和推理/适配训练机制。当前 teacher-forced 预训练默认 `use_cache=False`，因此 KV 压缩不是本阶段训练激活问题的首选。[MELT v3](https://arxiv.org/abs/2605.07721v3)

## 3. 已执行的隔离探针

脚本：`experiments/audit_loop_activation_memory.py`。仅使用随机 token ID 做前向/反向，不读取语料、不执行优化器更新、不改正式权重。原型直接复用当前 RecurrsiveInstinct 的实际模块，范围限定 dense + standard residual + 无 packing/KV + eager。

- 所有候选先通过 FP32 CPU loss 与全部参数梯度对照，覆盖 `(R,K)=(11,8)` 和 `(5,5)`，包括完整反传、截断反传及不整除分段。
- GPU 探针采用 FP32 参数、BF16 autocast；保存张量 hook 与性能/峰值测量分开运行。
- hook 返回 detached 引用避免记录过程保留原图；instrumented forward 后释放图并 GC，再测无 hook 的峰值。记录每个测量前 allocated baseline，检查是否一致。
- `saved_forward.nonparameter_unique_storage_bytes` 包含 autocast 权重副本等非原参数存储，**不是纯激活峰值**，也不能直接用它计算整步显存节省率。
- GPU loss 一致和梯度有限不等于已通过目标配置的全部 BF16 梯度或多步更新一致性验证。优化器、compile、真实 loader、packing 均不在本次探针内。

初始小模型 probe：hidden=384，core=2，side 各 1，batch=1，seq=512，R=16，K=8。结果文件：`experiments/results/loop_activation_probe_small.json`（本地忽略目录）。

| 策略 | 前向+反向 allocated 峰值 MiB |
| --- | ---: |
| 无 checkpoint | 394.83 |
| 原 mode=1 | 316.61 |
| 原 mode=2 | 193.12 |
| mode=2 + side checkpoint | 177.48 |
| 再加 head/CE checkpoint | 177.48 |
| 单层分段 g=4，含 side/head checkpoint | 250.34 |
| 嵌套分段 g=4，含 side/head checkpoint | 175.98 |

所有策略的测量前 allocated baseline 均为 115.04 MiB。该形状中 side checkpoint 有收益，head/CE checkpoint 未进一步降低总峰值；单层分段保存得更少，但反向峰值反而增加。短探针的 3 次计时不能作为生产吞吐排名。

完整 RecurrsiveInstinct 配置 probe：187,521,984 参数，batch=1，seq=1024，FP32 参数、BF16 autocast。每组 2 次预热、3 次测量；各组测量前 allocated baseline 均为 803.47 MiB。

| 策略 | R=32 / K=8 峰值 MiB | R=64 / K=8 峰值 MiB | R=32 / K=32 峰值 MiB |
| --- | ---: | ---: | ---: |
| 原 mode=1 | 3269.25 | 未测 | 未测 |
| 原 mode=2 | 2036.34 | 2036.34 | 2096.66 |
| mode=2 + side checkpoint | 1773.24 | 未测 | 1835.97 |
| 再加 head/CE checkpoint | 1778.57 | 未测 | 未测 |
| 单层分段 g=4，含 side/head | 2821.98 | 未测 | 未测 |
| 嵌套分段 g=4，含 side/head | 1765.12 | 1765.12 | 1782.70 |

证据支持的结论：

- R=32/K=8 的 side checkpoint 相对原 mode=2 降低峰值约 12.9%；嵌套 g=4 约 13.3%。主要收益来自补齐 side checkpoint，不能把全部收益归给嵌套本身。
- 固定 K=8，R 从 32 增至 64，两个已测策略的峰值均未增加；时间增加。这与 no-grad 前缀不保留每轮训练图的预期一致，不能外推为 compile/packing/其他 residual 也已验证。
- 在该 shape 下，嵌套方案完整反传 K=32 的峰值为 1782.70 MiB，低于原 mode=2 仅反传 K=8 的 2036.34 MiB；可将省出的显存用于增加 credit assignment 深度。
- 同样 K=32 时，嵌套方案相对原 mode=2 减少峰值约 15.0%，短探针中位时间从约 646 ms 增至 745 ms（约 +15%）；计时仅作成本线索，需暖态长窗口复核。
- FP32 CPU 梯度对照的最大绝对误差实测为 0；完整配置 GPU 各策略 loss 一致且梯度有限。所有输入/权重均为随机初始化，**没有学习质量、收敛或正式训练稳定性的证据**。

完整原始结果位于 `experiments/results/loop_activation_probe_v2_r32_k8.json`、`loop_activation_probe_v2_r64_k8.json`、`loop_activation_probe_v2_r32_k32.json`，均为本地文件。复现示例：

```powershell
$env:PYTHONUTF8='1'
python experiments/audit_loop_activation_memory.py --config trainer/config_recurrsive_instinct.json --seq 1024 --total 32 --backward 8 --warmup 2 --steps 3 --variants native1,native2,side_checkpoint,all_checkpoint,segment4,nested4 --output experiments/results/loop_activation_probe_v2_r32_k8.json
```

现阶段建议优先推进 **side checkpoint**，再按 K 和 shape 决定是否叠加嵌套重算；不要采用已测出反向峰值退化的单层四轮分段。

## 4. 阶段 P0：定位显存随什么增长

冻结同一份初始权重、dtype、attention backend、sequence length、micro-batch、packing 和优化器实现。第一组用固定深度，不把深度随机性混入显存归因。

| 探针 | 取值 | 回答的问题 |
| --- | --- | --- |
| 固定 K 改 R | K=8，R=8/16/32/64；必要时 128 | 增加无梯度前缀是否异常累积 CUDA 内存？ |
| 固定 R 改 K | R=32，K=2/4/8/16/32；超预算停止 | 真正限制长反传的是哪个保存项？ |
| 逐项 checkpoint | 原 mode=1/2、补 side、补 head/CE、嵌套 g=2/4 | 哪项降低完整 backward 峰值？ |
| 编译边界 | 最佳两项 eager / 当前 compile 配方 | 随深度重编译、CUDA graph 私有池、缓存是否放大峰值？ |
| 真实 shape | 先小模型；再完整 RecurrsiveInstinct，seq=1024；最后实际失败 shape | 小实验结论是否迁移？ |

分别测模型加载、第一次 optimizer step、暖态 forward/backward/optimizer 峰值，以及恢复后第一次 step。记录 allocated、reserved、驱动可用/DXGI budget、CPU RAM、参数/梯度/优化器状态与 workspace。不得把 reserved 缓存增长直接判为 tensor 泄漏。

固定 K 的 R 扫描使用同一模型和 `num_steps`，不通过重新初始化不同 loop_iters 模型比较。每个策略先独立进程隔离，再用 A/B/A 暖态比较；诊断 tensor hooks、memory snapshot 与吞吐测量分开。现有 `experiments/bench_checkpoint_memory.py` 面向 standard 模型，其历史阈值不能直接用于 RecurrsiveInstinct。

P0 交付：`memory_surface.json/csv`、峰值归因、2 个候选重算策略及适用 shape。只有相关代码实际需要修改时才添加针对性回归测试。

## 5. 阶段 P1：保留梯度，扩大可用反传深度

先补 side checkpoint；若 head/CE 是峰值来源，再补相应重算；若 K 的边界状态已经主导，再做嵌套 checkpoint。实验开关留在 loop 路径，保持原默认值作为对照。

正确性门槛：同一状态和 RNG 下检查 loss、所有参数梯度、prelude 梯度与 adapter 梯度；若扩展 MoE，必须检查 router/专家梯度、aux loss 与重算副作用。检查 dropout 非零时 RNG 重放、packing/位置/因果 mask、梯度累积、compile 和 resume。重算过程中不能再次抽深度、执行 optimizer step、改变模型持久状态，或把 prelude 输入 detach。

性能门槛：候选的最终比较至少 20 步预热和 100 步测量，重复 A/B/A；完整 optimizer 必须初始化完成。报告固定 `(R,K,tokens)` 的峰值/耗时，以及同一显存预算下可达到的最大 K。采用运行时 GPU/DXGI 预算并留安全余量，不能把标称 16 GB 全当可分配空间。

选择目标是显存—时间 Pareto 前沿：能否在相同训练目标下达到更长 K，而不是预设必须节省某个百分比。若参数/优化器/workspace 已主导，单独报告激活优化的上限，再考虑 optimizer workspace/offload；不同时混入未经验证的低精度更新。

## 6. 阶段 P2：识别“循环多了学不动”的机制

使用已审核的固定小训练集和独立验证集，保持 tokenizer、样本次序、有效 token/update 和 LR 按 token 的时间轴。新增/复用语料遵循项目 identity 清洗与 provenance 要求；本次随机 ID 探针仅用于数值/内存检查，不用于学习能力结论。

### P2a：先判定反传截断是否不足

在能完整反传的缩小 RecurrsiveInstinct 上固定 R=16，比较 K=2/4/8/16。先单种子短筛，每组 500 updates、每 update 4096 个有效 tokens（约 2.05M tokens）；对照与最多两个候选补到 3 个种子。固定同一 initialization reference depth 和相同 state_dict，固定 total R，不把更少计算误当成更好学习。

在少量固定 minibatch 上比较 K 截断梯度与 full BPTT 的分模块余弦、相对误差、范数；同时比较 held-out loss。不能仅凭梯度余弦认定质量改进。若扩大 K 明显恢复深循环收益，优先用 P1 的省显存换更长 K，再验证完整 RecurrsiveInstinct 的 R=32、K=8/16/32 中可运行的点。

### P2b：区分无效迭代、状态退化和读出不匹配

固定每个样本的初始 latent 噪声，测 R=4/8/16/32/64 的完整 coda+head 验证 loss；中间 logit lens 只作诊断，不能代替完整读出评估。生成评测采用相同 decoding 配方。

记录每轮归一化前 RMS、状态相对变化、与输入的余弦、attention entropy/QK logit 尺度、各参数组的 update/weight 比，以及按 R 分桶的 loss。只存标量到 CPU，不能把训练中间 GPU tensor 全挂在日志对象上。必要时离线估计状态转移的局部增益/Jacobian，避免把它放进每个正式训练 step。

区分以下证据：

- 增大 K 恢复效果：支持 credit assignment 不足。
- 相邻状态几乎不变且质量不提升：支持计算饱和/过早固定点，但不能据此断言答案正确。
- 状态方向振荡、输入分量漂移且质量下降：支持注入/动态系统问题。
- 只在偏离训练深度时下降：支持深度分布/读出匹配问题。
- 增 R 后同墙钟质量下降但同 token 不下降：可能主要是算力分配问题。

截断梯度和固定点近似的有效性依赖状态所处区域；新的局部去噪训练工作也专门研究短反传的 credit assignment，但目前主要是结构化推理任务，后置为探索路线。[Fixed Points](https://arxiv.org/abs/2610.06833)、[Looped Flows](https://arxiv.org/abs/2609.11801)

## 7. 阶段 P3：基于诊断选择稳定性干预

每次只修改一个因素；先在固定 R/K 下做结构消融，再改变随机深度策略。

| 证据 | 首选干预 | 必须保留的对照 |
| --- | --- | --- |
| 状态传递放大/注入漂移 | Parcae 式有界对角 carry；再单独增加 OrthoInj | 当前线性 adapter；相同 core/Norm/K |
| 不同深度导致梯度高方差 | 跨累积窗口的分层深度采样；可行时按样本采样 | 相同平均计算和边际深度分布 |
| 长 K 初始优化困难 | depth curriculum 与 LR warmup 分开测试 | 固定深度基线；按累计 token 对齐 schedule |
| QK 尺度或 attention entropy 异常 | 单独测试已有 qk_norm 开关 | 相同初始化与 optimizer，不同时更换注入 |
| 共享参数更新/残差比例异常 | 架构匹配的 residual scaling | 当前夹心 Norm 初始化，避免重复缩放 |

受约束 carry 的动机来自 [Parcae](https://arxiv.org/abs/2604.12946)，正交输入分量控制来自 [OrthoInj](https://arxiv.org/abs/2610.06833)。只替换 adapter 的本地消融应称为“借鉴 Parcae”，不是完整复现。当前 prelude 已有 post-residual Norm，额外加一层 Norm 需单独验证。

按样本深度采样若用 batch 内最大深度补齐，可能增加实际 FLOPs 和重算开销；batch=1 时没有批内多样性收益。跨累积窗口分层采样是工程候选，不宣称与原论文实现严格等价。初期不启用需要额外控制器的 learned prior。

Pre-Norm 的 `1/R` 规则与 Post-Norm DeepLoop 的 skip/init 联合缩放属于不同参数化；当前有输入 adapter、夹心 Norm 和 K<R，不能直接照搬展开深度系数。[Residual Scaling](https://arxiv.org/abs/2606.18524)、[DeepLoop](https://arxiv.org/abs/2607.13491)

## 8. 阶段 P4：联合验证与结论标准

完整 RecurrsiveInstinct 保留基线和最多两个候选。分别报告：

1. 同数据 token 预算下验证 loss、深度—质量曲线、目标代码/数学能力。
2. 同训练 FLOP/墙钟预算下质量，避免把多循环额外算力当免费收益。
3. 同显存预算下最大可训练 K、长序列/batch 能力和吞吐。

正式验证至少覆盖已知退化发生前后的训练区间，并超过首次出现问题的 token 位置；如果缺少该日志，先做预注册的小预算试验，不把它称为长期稳定性验证。预先固定主指标与可接受质量差异，依据基线多种子波动制定阈值，不在看见候选结果后放宽标准。

停止/降级条件：非有限 loss/gradient、OOM、数据/梯度正确性失败立即记录并停止该单元；质量连续多个验证点明显落后且超过基线波动则早停。不能把异常跳过后留下的好片段作为完整结果。深循环若没有质量收益，保留较浅配置并报告该计算区域的上限，而不以“能跑更多轮”判定成功。

工件按 run_id 写入 `experiments/results/loop_research/`：源码/配置/数据指纹、种子和 RNG、初始化哈希、R/K 每步记录、token 游标、attention backend、precision、optimizer、原始 metrics、失败信息、完整启动命令。研究输出和正式 out/checkpoints 分离。autotune 当前不支持该 loop 研究，不把本实验接入生产自动调优默认流程。

下一步顺序：先完成 P0 的 R/K 显存曲线和目标 shape 验证，再将最有效的重算方案实现为 opt-in，随后执行 P2a 的长反传对照。只有 evidence 支持注入问题时再进入 P3，避免一次叠加所有论文方案。
