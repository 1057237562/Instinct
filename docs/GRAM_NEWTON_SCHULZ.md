# Gram Newton-Schulz：数学等价的 Muon 正交化加速

## 结论

**已合入生产**（`trainer/batched_muon.py`，分支 `feat/gram-newton-schulz`）：
`_batched_zeropower` 现按形状分发——短边 ≥256 的矩形矩阵走 Gram 内核
（CUDA 上 fp16 迭代、第 3 步后单次重启），方阵/小矩阵/回退保持经典 bf16 迭代；
`INSTINCT_MUON_GRAM=0` 一键恢复全经典路径。
`trainer_utils.build_optimizer` 的 Muon workspace 默认 64 → 256 MiB（专家 chunk 4→16）。

数学上**精确算术逐迭代恒等**，浮点下仅结合顺序不同。5070 Ti 实测：生产分发路径
全 NS **204.1 → 122.0 ms（1.67×）**，投影端到端 tokens/s **+16%**；等价性由
75 项测试证明（fp64 ~1e-15 + 短训轨迹一致）。回退无需转换任何检查点。

来源：Tri Dao / Dao-AILab《Gram Newton-Schulz: A Fast, Hardware-Aware Newton-Schulz
Algorithm for Muon》（2026-03）。官方仓库自述："mathematically equivalent to and
faster than Newton-Schulz"。见文末参考链接。

## 动机：本分支的实测数据

运行 `pretrain_20260919_174858`（RTX 5070 Ti，hidden 512、32 层、8 专家 top-1 MoE，
BF16 参数 + Muon，稳定区间 steps 801–947）：

| 指标 | 数值 |
|---|---|
| step 时间 | 600 ms（forward 96 / backward 258 / optimizer 250） |
| Muon NS 执行量 | 8.09 TFLOP/step，占全部执行 FLOPs 的 45% |
| optimizer 阶段有效算力 | 32.8 TFLOPS = BF16 峰值（90.4）的 36% |

optimizer 占 41% 步时且吃不满 tensor core，瓶颈正是标准迭代的非对称矩形 GEMM
（每次迭代 2 个 m×n 乘 + 1 个 m×m 乘）。这是当前 MFU/HFU 分析给出的第一大杠杆。

## 数学推导

### 现有迭代（`_batched_zeropower`）

对（已转置到 m ≤ n 的）更新矩阵 X，记 A = X·Xᵀ，系数 (a, b, c)：

```
B_t   = b·A_t + c·A_t²                  # baddbmm(gram, gram, gram)
X_{t+1} = a·X_t + B_t·X_t               # baddbmm(x, polynomial, x)
A_{t+1} = X_{t+1}·X_{t+1}ᵀ
```

即 X_{t+1} = h(A_t)·X_t，其中 h(A) = aI + bA + cA²。

### Gram 重写

关键事实：Newton-Schulz 的多项式是奇函数 p(x) = ax + bx³ + cx⁵ = x·h(x²)，
且 h(A) 对称。于是：

```
A_{t+1} = X_{t+1}·X_{t+1}ᵀ = h(A_t)·A_t·h(A_t)
```

**A 的演化不依赖 X 本身**。因此整个迭代可以在 m×m 小矩阵上闭合：

```
A₀ = X·Xᵀ                    # 全程仅有的第 1 个矩形 m×n 乘法
Q₀ = I (m×m)
for t = 1..T:
    Z_t = a·I + b·A_{t-1} + c·A_{t-1}²      # 1 个 m×m 乘 + elementwise
    Q_t = Q_{t-1}·Z_t                        # m×m
    A_t = Z_t·A_{t-1}·Z_t                    # 2 个 m×m
X_T = Q_T·X                   # 全程仅有的第 2 个矩形 m×n 乘法
```

### 等价性边界

- **精确算术**：X_t = Q_t·X₀ 逐步恒等，返回的 Q_T·X 与标准迭代 T 次后的 X_T
  是同一个矩阵。系数、归一化（‖X‖_F + eps）、步数均不变。
- **浮点**：结合顺序不同，结果有微小舍入差异；上游在 Llama-430M / Qwen-600M /
  Gemma-1B / MoE-1B 的 Chinchilla-token FineWeb-Edu 验证中 perplexity 差异 < 0.01
  （You Jiacheng 与 Polar Express 系数均测过）。
- **稳定化（fp16 必需，bf16 建议保留）**：迭代第 3 步重启一次——
  X ← Q₂·X，重算 A = X·Xᵀ，Q 重置为 I——消除半精度下的负特征值漂移；
  a_t·I 项可折叠进 Q/R 更新以省一次加法。重启每次增加 3α−3 个 n³ 成本。

## FLOP 对比（本仓库形状，T = 5，n = 短边，α = 长边/短边）

每矩阵 FLOPs，n³ 单位：标准（现有 plain bmm）= 2T(2α+1)；Gram 纯 bmm ≈ 6T+4α；
Gram + 对称 GEMM kernel（上限，需自写 kernel）= 4T+3α−3。

| 形状（m×n） | 来源 | 数量/层 | α | 现有 | Gram 纯 bmm | Gram 对称 kernel |
|---|---|---|---|---|---|---|
| 512×1664 | 专家 gate/up/down ×8 | 24 | 3.25 | 10.07 G | 5.77 G（−43%） | 3.59 G（−64%） |
| 512×768 | q_proj | 1 | 1.5 | 5.37 G | 4.56 G（−15%） | 2.88 G（−46%） |
| 256×512 | k/v_proj | 2 | 2 | 0.84 G | 0.64 G（−24%） | 0.39 G（−54%） |
| 512×512 | o_proj | 1 | 1 | 4.03 G | 回退标准迭代（α=1 时两者相等） | 同左 |
| 8×512 | router | 1 | 64 | ~0 | 不改 | 不改 |

每层合计：252.7 G → 148.4 G（纯 bmm，naive，**−41%**）→ 93.9 G（对称 kernel，
**−63%**）；带 1 次稳定化重启的纯 bmm 版 ≈ 170 G（−33%）。
全程 NS：8.09 TFLOP/step → 约 4.8–5.4（纯 bmm）/ 3.0（对称 kernel）。

> 注：上表 "Gram 纯 bmm" 列是博客的 naive 公式（不含重启）。推荐配置
> （fp16 + 第 3 步后单次重启）的专家桶实测为 7.2 TFLOP vs 标准 8.0
> （FLOPs 仅 −10%）；其 1.18× 的算法收益主要来自工作形状（小 m×m 乘
> 替代矩形乘）而非 FLOP 总量，详见实验章节的收益分解。

## 小型实验结果（2026-09-22，RTX 5070 Ti，eager 模式）

脚本 `experiments/bench_gram_muon.py`（数值压力 + 分块扫描）；分块敏感性用
生产式 chunk 循环实测。专家桶 768×[512,1664] 中位耗时：

| 配置 | ms | 相对生产 |
|---|---|---|
| 标准 bf16 chunk4（**生产现状**） | 186.1 | 1.00× |
| 标准 bf16 chunk16 | 182.7 | 1.02× |
| 标准 fp16 chunk16 | 121.6 | 1.53× |
| Gram fp16 双重启 chunk16 | 120.8 | 1.54× |
| **Gram fp16 单重启(第3步后) chunk16** | **103.4** | **1.80×** |
| Gram fp16 无重启 chunk16 | （更快但数值不合格，见下） | — |

全 NS 投影：203.7 → 约 108.5 ms（专家 gram + 方阵/小矩阵回退标准 fp16），
optimizer 阶段 250 → 约 155 ms，step 600 → 约 505 ms，**tokens/s +19%**。

**1.80× 的构成分解**（相对生产 std-bf16-chunk4）：fp16 dtype 约 1.50×
（同 chunk16 下 182.7 → 121.6 ms；Blackwell 消费卡上 fp16 bmm 显著快于
bf16）+ chunk 调优约 1.02× + **Gram 算法本身约 1.18×**（121.6 → 103.4 ms，
FLOPs −10%、每 FLOP 效率 +6%）。即：fp16 是大头，Gram 是叠加项；方阵
桶只有 fp16 收益没有 Gram 收益。

实验裁决的三个关键事实：

1. **重启策略必须是"第 3 步后重启一次"（博客配置）**。无重启在重复行
   （真实梯度常见形态）下误差 1.7e-1，比生产差 10 倍；每 2 步重启（本实验
   初版误用）则多付 2 个矩形乘，FLOPs 反超标准迭代（8.6 vs 8.0 TFLOP），
   甜点配置下只有 1.54×。单重启 = 7.2 TFLOP，是速度与稳定性的正确平衡，
   数值全面优于生产 bf16（dup rows 2.5e-3 vs 1.8e-2；gaussian 4.2e-3 vs 1.4e-2）。
2. **chunk 是隐藏的关键变量**。生产 64 MB workspace 对专家形状只切出
   chunk≈4；实测甜点 chunk 16–32（更大反而因 L2 抖动变慢）。移植时需把
   workspace 上调到约 256 MB（训练峰值 allocated 7.97 GB / reserved 13.37 GB，
   16 GB 卡上安全）。
3. **fp16 优于 bf16（速度与精度同时）**。标准迭代单独换 fp16 就是免费的
   +5% 端到端（1.16× NS）且更准；方阵（α=1）与极小矩阵上 Gram 无优势，
   应回退标准 fp16（实测方阵 gram 5.93 vs 标准 fp16 4.35 ms）。

会话间绝对耗时漂移约 5–8%（时钟/状态），只比较同会话内数字。
端到端 +19% 为投影值：假设 optimizer 阶段随 NS 等比压缩（foreach 动量/
cast 开销不变）；按 PLAN.md 护栏，合入前需固定种子短训 A/B 确认 loss 曲线。

## 生产化结果（2026-09-22 合入，分支 `feat/gram-newton-schulz`）

- **分发规则**（`_batched_zeropower`）：`短边 ≥ 256 且非方阵 → Gram`，其余
  （q/o 方阵、k/v、router）保持经典迭代——实测这些形状 Gram 无收益。
  CUDA 上 Gram 迭代用 fp16（bf16 输入先归一化再 cast，元素 ≤1 无溢出风险），
  CPU 上 bf16。
- **workspace 默认 64 → 256 MiB**（`trainer_utils.build_optimizer`，环境变量
  `INSTINCT_MUON_WORKSPACE_MB` 仍可覆盖）：专家 chunk 4 → 16，甜点配置。
- **生产符号实测**（真实桶形状，chunk 按新 workspace）：
  全 NS 204.1 → 122.0 ms（**1.67×**；专家 193.7→112.9，其余形状按分发保持
  经典共约 9.7 ms）。投影 step 600 → 518 ms，tokens/s 17,947 → 约 20,800
  （**+16%**；略低于 +19% 上限是因为方阵/小矩阵未换 fp16——保守分发，
  避免为 <5 ms 收益扩大改动面）。
- **测试**：`tests/test_gram_newton_schulz.py`（含生产分发 bitwise、
  env-off 回退、GPU fp16 路径）+ `tests/test_gram_muon_short_training.py`
  （含生产分发短训）+ 既有 `tests/test_batched_muon.py`（与 native Muon
  对照）全绿。
- **兼容性**：momentum/param_groups/checkpoint 布局不变，resume 完全兼容；
  `INSTINCT_MUON_GRAM=0` 恢复经典迭代（bitwise 与旧版一致，有测试守护）。
- **真实训练验收**：恢复训练后预热两个 bucket、取 ≥20 步稳定区间比较
  tokens/s 与 optimizer 阶段占比（预期 250 → 约 168 ms）；训练中的进程
  不会热加载，需重启/resume 生效。注意：合入后首次启动会因代码变更触发
  torch.compile 缓存失效而重编译一次（约 1–2 分钟），属一次性成本。

## 实现方案（移植前设计）

> 注：本节为移植前设计；实际合入形态见上节"生产化结果"。

改动收敛在 `_batched_zeropower`（签名与返回值不变），要点：

1. **批处理结构不变**：`step()` 按 (device, dtype, shape) 分桶、workspace 分块的
   机制照旧；Gram 版的临时量是 [B, m, m]（比矩形 [B, m, n] 更小），
   `_chunk_size` 的 8·short·short 项估计天然更宽松。
2. **迭代体**：
   ```python
   A = torch.bmm(x, x.transpose(-2, -1))          # 矩形乘 1/2
   Q = torch.eye(m).expand(B, m, m).clone()
   for t in range(ns_steps):
       A2 = torch.bmm(A, A)
       Z = torch.baddbmm(a * I, b * A, A, ...)     # aI + bA + cA²
       Q = torch.bmm(Q, Z)
       A = torch.bmm(torch.bmm(Z, A), Z)
       if t == restart_at:                          # 稳定化，见上
           x = torch.bmm(Q, x); A = torch.bmm(x, x.transpose(-2, -1)); Q = I
   x = torch.bmm(Q, x)                              # 矩形乘 2/2
   ```
3. **α=1 回退**：`x.size(-2) == x.size(-1)`（o_proj 512×512）时走现有标准循环。
4. **dtype**：迭代内用 fp16（实验证实速度与精度同时优于 bf16：dup-rows
   误差 2.5e-3 vs 1.8e-2，且快 15%+），输出照旧 `to(params.dtype)`。
   保持 `autocast(enabled=False)` 不变。
5. **重启 = 第 3 步后一次**（`if t == restart_after: ...`，单次非周期）。
   周期性重启会多付矩形乘（实测降至 1.54×），无重启在病态输入上数值
   不合格（见实验章节）。
6. **chunk 上调**：Gram 的临时量是 [B,m,m] 而非 [B,m,n]，`_chunk_size`
   公式需按 Gram 改写并放宽（workspace 64 → 约 256 MB），目标 chunk 16–32。
7. **不改变任何状态布局**：momentum buffer、param_groups、checkpoint 序列化均不动，
   resume 检查点完全兼容。
8. **系数不在等价范围内**：现有 (3.4445, −4.7750, 2.0315)（You Jiacheng 系数）
   与 Polar Express 系数都可插入 Gram 形式，但换系数本身是另一个（非等价）决策，
   不与本方案捆绑验证。

## 短训一致性验证（2026-09-22，CPU，tiny MoE 模型）

`tests/test_gram_muon_short_training.py`（5 项，约 19 s）：通过 monkeypatch
替换 `_batched_zeropower`（生产符号，不改生产文件），走完整的
`build_optimizer`→`BatchedMuon`（动量/nesterov/foreach/分桶/chunk）+
`res.loss + res.aux_loss` 生产损失路径，周期 8 合成数据训练 25 步。

| 对比（vs 生产 bf16 轨迹） | loss 最大相对差 | 25 步后参数距离 |
|---|---|---|
| fp64 标准迭代（更精确的参考） | 0.025 | 0.484 |
| 标准 fp16（仅换 dtype） | 0.028 | 0.695 |
| Gram bf16 单重启 | 0.020 | 0.654 |
| **Gram fp16 单重启（推荐配置）** | **0.015** | **0.690** |

结论：

1. **fp64 下整条训练轨迹一致**（loss/参数差 <1e-6）：等价性贯穿训练动态，
   不止单次调用。
2. **推荐配置（Gram fp16 r3）的 loss 轨迹比 fp64 参考更贴近生产**
   （0.015 < 0.025），参数距离落在 dtype 级扰动的自然带宽内。
3. **参数级一致在工作精度下不可判**：top-1 离散路由把任何舍入差放大为
   路由翻转（连 fp64 标准版都发散 0.484）；正确判据是"Gram 的发散 ≤
   纯换 dtype 的发散"，实测 0.690 ≤ 0.695，成立并有专门测试守护此结论。
4. 测试同时验证：harness 确定性（两次生产运行逐位相等）、学习进度
   （两配置都从 5.53 降到 ≤5.15）。

## 验证与复现（对应 PLAN.md 护栏）

1. **CPU fp64 数值测试**（`tests/`，无需 GPU）：随机高斯、低秩、真实梯度形态
   （tall 1664×512 / 方 512×512）矩阵上，断言 Gram 版与标准版输出相对误差
   在 fp64 下 ~1e-12 量级、fp16/bf16 下与各自对 fp64 极分解的偏差同量级。
2. **GPU 基准**：`python experiments/bench_batched_muon.py`（已有脚本），
   对比标准 / Gram 两版在 (24×[512,1664])、(32×[512,512]) 真实桶形状上的耗时；
   排除首次编译，预热 ≥20 步。
3. **端到端**：固定种子短训，loss 曲线应与标准版在噪声内重合；
   恢复训练需重启进程（运行中的 Python 不热加载）。
4. 回退开关：环境变量（如 `INSTINCT_MUON_GRAM=0`）保留标准路径，便于 A/B。

## 预期收益与上限（实测优先）

- **第一版（纯 bmm eager，fp16 + 单重启 + chunk16，已实测）**：专家桶 1.80×，
  全 NS 203.7 → 约 108.5 ms；optimizer 250 → 约 155 ms；step 600 → 约 505 ms；
  **tokens/s +19%（17,947 → 约 21,300）**。前提是 workspace 上调到 chunk 16–32。
- **顺带的免费收益（可独立合入）**：仅把标准迭代从 bf16 换 fp16（一行改动）
  即 +5% 端到端且精度更好——即使不做 Gram 也值得。
- **对称 GEMM kernel（后续工作）**：理论 FLOPs 再降（−63% vs 标准），上游
  H100/B300 实测正交化最高 2×、Kimi K2 MoE 形状 2×。上游 CuTeDSL kernel
  面向 SM90/SM100，5070 Ti 是 SM120，需自写 Triton 三角调度 + 转置回写
  epilogue（参考上游约 160 行实现的思路）。本实验的 1.80× 不含此项。

## 参考

- 博客：<http://dao-lab.ai/blog/2026/gram-newton-schulz/>（原
  dao-ailab.github.io/blog/2026/gram-newton-schulz/）
- 代码：<https://github.com/Dao-AILab/gram-newton-schulz>
- Princeton 学位论文（J.C. Zhang, 2026）：Gram Newton-Schulz 章节
- 系数备选（非等价，单独决策）：Polar Express（ICLR 2026，
  github.com/NoahAmsel/PolarExpress）
- 背景：Keller Jordan《Muon》、Jeremy Bernstein《Deriving Muon》
