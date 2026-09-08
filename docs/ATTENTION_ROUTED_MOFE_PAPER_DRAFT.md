# Attention-Routed Mixture of Frozen Experts for Efficient Model Composition

> 论文草稿 v0.1（2026-09-09）。本文档严格区分已完成的工程验证与尚未进行的模型质量实验；所有标记为 **TBD** 的数值在正式实验前不得作为结论引用。

## 摘要

将多个已训练领域模型组合为统一语言模型，通常需要重新训练大规模稠密模型或可训练的混合专家模型，因而产生较高的显存、优化器状态与训练成本。本文提出 Attention-Routed Mixture of Frozen Experts（AR-MoFE）：从 (n) 个同构的已训练语言模型中逐层抽取前馈网络（FFN），组成不可训练的专家库；共享嵌入、注意力、归一化与语言模型头由一个基座权重初始化；一个可训练的 Attention Router 根据 token 表征动态激活 top-(k) 个冻结专家。与经典线性门控不同，Attention Router 在专家维度构造 token 内部的 Q/K/V 交互，使候选专家的分数可依赖其他候选专家。我们进一步实现了 router-only 与 router-shared 两种 post-pretraining 范式、负载均衡与 router z-loss、清单驱动的专家装配、紧凑增量检查点，以及面向 16 个冻结权重的可视化训练工作流。当前工程验证表明，FFN 可被精确移植并在优化步骤后保持位级不变，top-1 与 top-2 路由保留非零梯度，完整训练链路可生成不含专家副本的增量检查点。模型质量、吞吐与专家专门化收益仍需按照本文给出的对照实验测量。

## 1. 引言

稀疏混合专家模型通过只激活总参数的一小部分扩大模型容量 [1–3]，但常规 MoE 仍需联合训练所有专家。另一方面，现实中往往已经存在一组从同一基座派生的领域模型。MoFE 将这些模型的 FFN 作为冻结专家复用，只训练共享层和路由器，从而降低可训练参数量并保留已有领域能力 [4]。

这一思路有两个尚未解决的问题。第一，经典线性门控仅通过一次投影独立产生专家分数，无法显式表示候选专家之间的关系。第二，冻结 FFN 会阻止共享层与专家共同适配；MoFE 原论文甚至观察到 post-pretraining 可能降低特定医疗任务表现 [4]。因此，本工作不预设 post-pretraining 一定有益，而是研究如下问题：

1. Attention Router 能否比线性门控更有效地从冻结专家库中选择互补知识？
2. 只训练路由器能否避免共享层与冻结 FFN 的失配？
3. 在 16 个来源模型、top-2 激活的条件下，参数效率、吞吐、显存与质量之间存在怎样的折衷？

本文的贡献是：

- 提出将 Attention Router 与 MoFE 结合的 AR-MoFE 架构；
- 给出适用于已有 (n) 个完整模型权重的逐层 FFN 装配与冻结方法；
- 设计 router-only/router-shared post-pretraining、稳定化损失与紧凑检查点协议；
- 提供可复现的 16(n) 专家配置、预检、训练、暂停与续训 UI；
- 明确给出包含负面对照的实验矩阵，避免仅凭训练损失推断知识融合成功。

## 2. 相关工作

### 2.1 稀疏混合专家

GShard 将条件计算与自动分片用于超大规模稀疏模型 [1]。Switch Transformer 使用 top-1 路由简化计算，并以专家负载均衡目标缓解路由塌缩 [2]。ST-MoE 进一步提出 router z-loss 以改善训练稳定性和迁移表现 [3]。这些工作通常从头或联合训练专家，与本文复用现有冻结 FFN 的设定不同。

### 2.2 Mixture of Frozen Experts

Seo 等人将来源模型的 FFN 置入 MoE 层并冻结，只更新路由器与非 FFN 参数 [4]。该方法显著降低可训练参数和训练时间，但其 post-pretraining 实验出现性能下降；作者将其归因于知识注入需要全层协同，而冻结 FFN 可能造成层间失配。这一负面结果直接促使本文把 router-only 作为一等训练策略，并要求所有实验保留无 post-pretraining 对照。

### 2.3 Attention Router

Yuan 2.0-M32 使用 32 个专家、每 token 激活 2 个专家，并提出在专家空间进行 Q/K/V 交互的 Attention Router [5]。其报告显示相较经典路由器具有更好的训练或评测表现。本文将该路由思想用于冻结而非联合训练的专家，检验更有表达力的门控能否缓解异构领域专家的选择问题。

## 3. 方法

### 3.1 冻结专家装配

设有 (E=n) 个同构来源模型，第 (e) 个模型第 (l) 层的 FFN 为：

\[
F_{l,e}(x)=W^{down}_{l,e}\left(\mathrm{SiLU}(W^{gate}_{l,e}x)\odot W^{up}_{l,e}x\right).
\]

我们从每个来源权重抽取三组投影，并置为：

\[
\nabla W^{gate}_{l,e}=\nabla W^{up}_{l,e}=\nabla W^{down}_{l,e}=0.
\]

一个共享基座权重提供 token embedding、self-attention、RMSNorm 和 LM head。装配前要求所有权重具有一致的 tokenizer、词表、层数、隐藏维度、FFN 宽度及张量命名；系统对每层每个投影进行存在性与 shape 验证。

### 3.2 Attention Router

对于第 (l) 层 token 状态 (h\in\mathbb{R}^{d})，路由器计算：

\[
q=W_Qh,\quad k=W_Kh,\quad v=W_Vh,\qquad q,k,v\in\mathbb{R}^{E}.
\]

随后在专家空间形成：

\[
A=\mathrm{softmax}\left(\frac{qk^\top}{\sqrt{E}\tau}\right),\qquad z=Av,
\]

其中 (	au>0) 为温度，(z\in\mathbb{R}^{E}) 是最终专家 logits。路由概率和选中集合分别为：

\[
p=\mathrm{softmax}(z),\qquad S=\mathrm{TopK}(p,k).
\]

层输出为：

\[
y=\sum_{e\in S}\tilde p_e F_{l,e}(h).
\]

top-(k>1) 时可令 (	ilde p_e=p_e/\sum_{j\in S}p_j)。top-1 时若执行这一归一化，唯一权重恒为 1，语言模型损失无法通过混合权重向路由器传播；因此本文保留 (	ilde p_e=p_e)。路由器 Q/K/V 运算使用 FP32，以降低低精度 softmax 的不稳定性。

### 3.3 训练目标

总损失为：

\[
\mathcal{L}=\mathcal{L}_{LM}+\lambda_b\mathcal{L}_{balance}+\lambda_z\mathcal{L}_{z}.
\]

令 (f_e) 为 batch 内被路由到专家 (e) 的 token 比例，(P_e) 为该专家平均概率，则：

\[
\mathcal{L}_{balance}=E\sum_{e=1}^{E}f_eP_e.
\]

router z-loss 为：

\[
\mathcal{L}_{z}=\frac{1}{T}\sum_{t=1}^{T}\left(\log\sum_{e=1}^{E}\exp z_{t,e}\right)^2.
\]

默认设置为 (E=16)、top-2、(	au=1)、(lambda_b=5\times10^{-4})、(lambda_z=10^{-3})。这些值是初始实验点，而非已优化结论。

### 3.4 两种 post-pretraining 范式

- **Router-only：** 仅更新每层 Attention Router。该设置最大程度保持来源权重对齐，是主实验默认值。
- **Router-shared：** 更新 router、共享 attention、embedding、norm 与 LM head，但持续冻结全部 FFN。该设置容量更大，同时更容易出现共享层—冻结专家失配。

### 3.5 参数与检查点效率

在 Instinct 默认配置（8 层、(d=768)、FFN 宽度 2432、词表 6400）下，实测参数计数如下。

| 配置 | 总参数 | 每 token 活跃参数（top-2，估算） | 可训练参数 |
|---|---:|---:|---:|
| Dense Instinct | 63.91M | 63.91M | 63.91M |
| AR-MoFE-16E router-only | 736.61M | 109.03M | 0.295M |
| AR-MoFE-16E router-shared | 736.61M | 109.03M | 19.38M |

周期性检查点只保存可训练增量。每份增量携带专家清单指纹；恢复时若路径、文件大小或修改时间发生变化则拒绝加载，以防无意中对不同专家库继续训练。

## 4. 系统实现

实现基于纯 PyTorch Instinct 代码库。专家清单以 JSON 表示一个共享基座和任意数量的命名专家。WebUI 默认生成 16 行专家槽位，支持文件存在性检查和独立子进程中的 tensor topology 预检。训练器支持 JSONL post-pretraining 数据、sequence packing、混合精度、DDP、`torch.compile`、日志、暂停与精确续训。

当前已完成的工程验证包括：

| 验证项 | 状态 |
|---|---|
| 每层 gate/up/down 投影精确来自声明的专家权重 | 通过 |
| 所有专家 `requires_grad=False` | 通过 |
| 优化步骤后专家张量位级不变 | 通过 |
| top-1 Attention Router 获得非零 LM 路径梯度 | 通过 |
| 专家权重被替换后清单指纹改变 | 通过 |
| CPU 微型模型完成装配、一步训练和增量保存 | 通过 |
| 16 专家 GPU 质量/吞吐实验 | **TBD** |

## 5. 实验设计

### 5.1 研究问题

- RQ1：Attention Router 是否优于参数量更小的线性路由器？
- RQ2：router-only 是否比 router-shared 更能保持已有专家能力？
- RQ3：专家数量和 top-k 如何影响质量、吞吐、显存和利用率？
- RQ4：路由是否学习到可解释的领域专门化，而非仅偏好输出范数更大的专家？

### 5.2 对照组

1. 单个共享基座模型；
2. 最佳单领域来源模型；
3. 常规可训练 MoE；
4. MoFE，不进行 post-pretraining；
5. MoFE + 线性 router-only post-pretraining；
6. AR-MoFE + router-only post-pretraining；
7. AR-MoFE + router-shared post-pretraining。

消融实验覆盖 4/8/16 experts、top-1/top-2、去除 balance loss、去除 z-loss、不同温度，以及随机专家/领域专家组合。所有主要实验至少运行 3 个随机种子。

### 5.3 指标

报告 held-out perplexity、领域与通用下游准确率、平均/最差领域保持率、tokens/s、峰值显存、总/活跃/可训练参数、检查点大小、每层专家负载、利用率熵、跨领域路由互信息和运行方差。必须同时报告 wall-clock 时间；冻结参数减少不等同于前向计算或 I/O 免费。

### 5.4 待填结果表

| 方法 | PPL ↓ | 通用均分 ↑ | 领域均分 ↑ | 最差领域 ↑ | tok/s ↑ | 峰值显存 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Shared base | TBD | TBD | TBD | TBD | TBD | TBD |
| MoFE, no post-PT | TBD | TBD | TBD | TBD | TBD | TBD |
| Linear, router-only | TBD | TBD | TBD | TBD | TBD | TBD |
| Attention, router-only | TBD | TBD | TBD | TBD | TBD | TBD |
| Attention, router-shared | TBD | TBD | TBD | TBD | TBD | TBD |

## 6. 风险、局限与预期失败模式

首先，来源模型必须同构；当前方法不是任意架构模型的通用拼接器。其次，冻结专家仍占用设备显存，节省的是梯度和优化器状态而非总权重存储。第三，top-k 离散选择可能导致路由塌缩；均衡损失也可能牺牲真正合理的不均匀分配。第四，专家 FFN 依赖其原始 attention 表征，跨模型移植并不保证语义坐标完全对齐。第五，原始 MoFE 的负面 post-pretraining 结果意味着 router-shared 尤其可能退化。最后，Attention Router 的 (E^2) 关系随专家数量二次增长，尽管 (E=16) 时开销很小。

## 7. 结论

本文提出 AR-MoFE，用 Attention Router 在 token 级选择来自多个既有模型的冻结 FFN，并提供两种 post-pretraining 范式。当前成果证明了该方案的工程可行性和参数高效检查点路径，但尚未证明质量增益。后续工作的首要目标是完成有负面对照、多个随机种子和完整系统指标的 16 专家实验；只有这些结果才能判断更强路由器是否真正缓解冻结专家组合中的失配。

## 参考文献

[1] Lepikhin, D., et al. 2020. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*. arXiv:2006.16668. <https://arxiv.org/abs/2006.16668>

[2] Fedus, W., Zoph, B., and Shazeer, N. 2022. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR 23. <https://jmlr.org/papers/v23/21-0998.html>

[3] Zoph, B., et al. 2022. *ST-MoE: Designing Stable and Transferable Sparse Expert Models*. arXiv:2202.08906. <https://arxiv.org/abs/2202.08906>

[4] Seo, J., Kim, J., and Shin, H. 2025. *MoFE: Mixture of Frozen Experts Architecture*. NAACL 2025 Industry Track, 340–348. <https://doi.org/10.18653/v1/2025.naacl-industry.28>

[5] Wu, S., et al. 2024. *Yuan 2.0-M32: Mixture of Experts with Attention Router*. arXiv:2405.17976. <https://arxiv.org/abs/2405.17976>
