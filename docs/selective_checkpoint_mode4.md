# ckpt4：按训练桶选择缓存的 Dense selective checkpoint

日期：2026-10-23。入口：`--use_grad_checkpoint 4`。适用标准 Dense 主干与标准残差。
数值正确性与容量/性能验收是独立结论；以下静态缓存预算不保证整机不 OOM。

## 实际 recipe

V2 Dense：32 层，H=1536，FFN=5504，Q=16×96，KV=4×96，vocab=6400。
BF16 参数/计算、FP8 tensorwise + auto filter、Muon mantissa + BF16 momentum、
accumulation=8、compile=max-autotune-no-cudagraphs、token budget=12288。
实际桶是 4×3072 与 3×4096。原 WDDM allocator limit 为 14.37 GiB。
完整参数、梯度及优化器状态约 7.6012 GiB，另有 FP32 残差边界和临时张量。

Muon workspace 是更新阶段的临时工作目标，不是全程等量驻留的张量，也不是严格
上限。必须分别比较累积反向和 optimizer 更新阶段；优化器 workspace 变化也会
改变内存池布局与 reserved 峰值。

## 每层缓存收益表

每行按 T=12288、一次前向、FMA=2 FLOPs、GEMM 输出 BF16 计算。
GiB/MiB 使用二进制单位，GFLOPs 使用 10^9。缓存量仅指候选输出，不是总 allocated 增量。

| 候选结果 | 前向 GFLOPs | 输出 MiB | FLOPs/Byte | 32 层输出 GiB |
|---|---:|---:|---:|---:|
| Q | 57.982 | 36 | 1536 | 1.125 |
| K | 14.496 | 9 | 1536 | 0.28125 |
| V | 14.496 | 9 | 1536 | 0.28125 |
| O | 57.982 | 36 | 1536 | 1.125 |
| gate | 207.769 | 129 | 1536 | 4.03125 |
| up | 207.769 | 129 | 1536 | 4.03125 |
| gate + up（上两项合计） | 415.538 | 258 | 1536 | 8.0625 |
| Q/K/V/O 合计 | 144.955 | 90 | 1536 | 2.8125 |
| down | 207.769 | 36 | 5504（仅理论） | 1.125 |
| Attention 输出/LSE，完整 causal S=3072 | 116.002 | 36.75 | 3010.29 | 1.14844 |
| Attention 输出/LSE，完整 causal S=4096 | 154.657 | 36.75 | 4013.39 | 1.14844 |
| Attention 输出/LSE，片段长度均为 1024 | 38.692 | 36.75 | 1004.08 | 1.14844 |
| 输入 RMSNorm / FFN 前 RMSNorm，各一个 | 约 0.0755 | 72 | 约 1 | 2.25 |
| Q/K RMSNorm 合计 | 约 0.0949 | 45 | 约 2 | 1.40625 |
| Q/K RoPE 合计 | 约 0.0708 | 45 | 约 1.5 | 1.40625 |
| SwiGLU 逐元素部分 | 约 0.4058 | 129 | 约 3 | 4.03125 |
| 残差加法，各一个 | 0.0189 | 72 | 0.25 | 2.25 |

这些候选存在别名、融合和依赖，不能把整张表直接相加。RMSNorm 表中仅记输出，
实际低层 SAC 若同时保留 reduction 中间量还会增加字节。逐元素算术估计把 exp、
rsqrt、除法分别按一次操作计，不代表它们有相同执行成本。RoPE 只记乘加。
FP8 的转换、缩放、amax、padding 和重放临时量未加入 GEMM FLOPs。

对 GEMM `Y[T,b]=X[T,a]W[a,b]`：`F=2Tab`，`M=2Tb`，所以 `F/M=a`。
gate/up 占可省 GEMM 前向重放量的约 74.14%，单位输出字节收益与 Q/K/V/O 相同。
down 的表面比值最高，但当前逐层 non-reentrant checkpoint 的 early-stop 已跳过
其前向重放：CPU dispatch 计数确认 mode0/mode2 的 down-forward 形状调用次数相同。
因此不要把 207.769 GFLOPs 当成缓存 down 可新增省下的重算。
默认 early-stop 的语义亦见 [PyTorch checkpoint 文档](https://docs.pytorch.org/docs/2.14/checkpoint.html)。
层外 LM head 单次前向为 241.592 GFLOPs / 150 MiB，但不在逐层 checkpoint 中，
新增 SAC 缓存没有这部分重放收益。各 GEMM 的 dX/dW 仍必须执行。

Attention 有效因果 QKᵀ/AV 算术：`F=2*Qdim*sum(l*(l+1))`；
缓存输出/LSE：`M=T*(2*Qdim+4*Nheads)`。不包含 softmax、块取整以及 Flash backward
本身的内部重算；缓存 forward 输出不会消除整个 attention backward。

原失败运行第一块缓存的所有 metadata（110376 个 blocks）复算：
3072 桶的平均逻辑 Attention 为 57.873 GFLOPs / 层 / 4 条，即 1501.81 FLOPs/Byte；
4096 桶全部是单文档，trainer 丢弃不含多文档的 sequence_ids 后走完整 causal，
应使用 154.657 GFLOPs / 4013.39 FLOPs/Byte，不能按有效 token 的平方缩小。
前者包含 padding 段但不含块取整，单文档 batch 的 causal dispatch 也可能增加实际工作。
这只覆盖已经缓存的第一块语料，不能当作整个语料的统计。

## 实现策略

输出缓存的分析预算为 1.75 GiB；按实际 batch×sequence 和 autocast dtype 静态选层，
不逐层查询空闲显存、不改变 batch 或续训游标。默认计划：

| 桶 | 优先保留 | 额外保留 | 估算输出缓存 |
|---|---|---|---:|
| 4×3072 | 32 层紧凑 K/V | 顶部 4 层 gate/up | 1.5703125 GiB |
| 3×4096 | 32 层融合 Attention 输出/LSE | 顶部 2 层 gate/up | 1.65234375 GiB |

短桶重算 attention core；长桶重算 Q/K/V/O 投影，保留融合 attention 结果。
未选中的 gate/up、所有 down 和便宜中间量重算/由 early-stop 消去。
顶层大缓存优先在 backward 释放。若基础缓存本身超过预算，也只保留顶部可容纳的层。
较小 batch 自动增加可保留的 gate/up 层数。

该计划针对给定 recipe；4096 分界是由该语料桶形状选择的静态启发式，并不是任意
packing 分布下的普适最优。非零 dropout 强制 attention replay，避免跳过随机操作
改变后续 RNG；math attention 不缓存平方规模的 score/probability 矩阵。
Q/K/V 同宽时不依靠尺寸把 Q/O 错认作紧凑 K/V。现有 state_dict 和参数结构不变。

## 验证与测量入口

`tests/test_mfu_checkpoint.py` 覆盖 logits、loss、全部参数梯度、RNG、两次累积、LoRA、
单个 AOT trunk 与两个实际桶的缓存计划。GPU parity 位于
`tests/test_dense_selective_checkpoint.py`，包含 BF16/FP8 与 plain/packed attention。

`experiments/profile_checkpoint_policy.py` 是固定权重容量探针，可驻留真实状态/梯度，
不更新参数。初版全层 Q/K/V/O 缓存在 12k 下失败，即使不预留重叠 workspace 仍在
FFN backward OOM；不能因为 forward 成功就称该策略可用。

`experiments/profile_checkpoint_recipe.py` 用真实 Muon/Adam 更新比较 workspace，
保持 accumulation=8，分别记录反向与更新峰值；结束/SIGINT 都等完整更新后通过
save-and-pause 保存，再逐项核对模型、优化器、scaler、数据进度与对应输出权重。
它使用固定合成 batch，不代表真实数据加载吞吐或语料训练质量。

测量结果位于忽略目录 `experiments/results/ckpt4/`。实际吞吐和容量结论仅以 status=ok
的完整报告为准。allocated 增幅必须用 `(ckpt4 / ckpt2 - 1) × 100%`，在相同 batch、
精度、优化器驻留、warmup 与测量口径下比较。

### 已完成的固定权重容量探针

RTX 5070 Ti，PyTorch 2.13.0+cu132，TorchAO 0.18.0。32 层完整模型，驻留真实
Muon/Adam 状态和已有梯度共 7.6012 GiB；不在 activation 阶段重复预留 optimizer
临时工作区。5 次 warmup，3×10 次前向/反向，14.37 GiB allocator limit。

| 桶 / 测试 mask | 前向+反向中位 ms | peak allocated GiB | peak reserved GiB |
|---|---:|---:|---:|
| 4×3072，每条 3 个等长片段 | 800.620 | 12.5082 | 13.1289 |
| 3×4096，完整 causal | 1082.368 | 12.4701 | 13.0176 |

这是合成 mask 的稳态容量/前反向测量，尚不包含 optimizer update、数据加载或保存。
不能把上述 tokens/s 称为完整训练 MFU，也不能据此计算相对 ckpt2 的收益；后者需要
相同条件的 ckpt2 对照。报告：`kv_gate4_4x3072.json`、`attention_gate2_3x4096.json`。

对照候选：把 4×3072 也改为“全层 Attention + 顶部 2 层 gate/up”，第一轮前后向
成功，第二轮 forward 在 allocated 13.74 GiB、unused reserved 643.40 MiB 时无法
分配 130 MiB，因此不采用。其静态 forward 图保留量仅 3.9024 GiB，说明表中的
输出缓存预算不足以独立预测稳态 allocator 峰值；额外运行时存活量未逐项归因。
结果 `attention_gate2_4x3072.json` 为 error，不能当性能数据。静态去别名审计位于
`attention_gate2_3072_graph_audit.json`，没有未解析的返回张量或赋值。

### Muon workspace 联合测量

同机完整 32 层，交替 4×3072（三等长文档）/3×4096（完整 causal），8 次累积后
真实更新，包含 grad_clip=1、Muon mantissa 与 AdamW；LR=0.0005。
与 trainer 一致 `dynamic=False`。每档预热 1 个完整 update，再轮换顺序测量
3 个窗口，每窗口 1 update（8 个 micro-batches，98244 个预测目标 tokens）。
固定合成输入不含 loader、LR schedule 或保存耗时，也不代表语料训练质量。

| Workspace MiB | Muon+Adam/clip 中位 ms | 整步中位 ms | optimizer 阶段 allocated GiB |
|---:|---:|---:|---:|
| 128 | 397.49 | 7893.52 | 7.981 |
| 256 | 395.45 | 7897.37 | 8.011 |
| 512 | 384.12 | 7898.74 | 8.098 |
| 1024 | 383.85 | 7890.51 | 8.274 |
| 2048 | 382.75 | 7899.20 | 8.595 |
| 3072 | 382.12 | 7887.66 | 8.819 |

完整测量均成功。512→3072 仅减少约 2 ms 的优化器时间；整步中位差 0.14%，
落在窗口波动范围内，不能宣称大 workspace 有显著收益。大 workspace 预热曾使
reserved 达 14.2754 GiB；稳态约 13.9102 GiB，峰值 allocated 约 12.5708 GiB。
建议保留 512 MiB，给 allocator 留余量。该单进程 sweep 的 reserved 包含之前
更大工作区留下的池，不能归因于当前一档 workspace 的独立需求。

报告 `recipe4_workspace_static.json`：status=ok，24 updates / 192 microsteps。
通过 save-and-pause 保存并逐项验证了最新 weights、optimizer、scaler 和 cursor：
`checkpoints/ckpt4_recipe_20261023_202316_50492_1536_resume.pth`。

早先 `recipe4_workspace_sweep.json` 遗漏 `dynamic=False`，混合桶动态编译显著变慢，
已明确排除性能比较；该运行在第 11 次 update / 88 microsteps 后保存暂停，
匹配 checkpoint `ckpt4_recipe_20261023_201625_10356_1536_resume.pth` 已完整验证。

### ckpt2 / ckpt4 独立进程对照

保持前述模型、精度、静态编译与 50:50 两桶交替输入，均固定 Muon workspace=512 MiB，
每个进程各预热 2 次 update、测量 3 次 update，accumulation=8。
优化器状态在前向前预分配，测量稳态而非 lazy-state 冷启动分配顺序。
每个进程从相同 seed=307 初始化；固定合成 batch，不使用用户语料游标。

| 指标 | ckpt2 | ckpt4 | 相对变化 |
|---|---:|---:|---:|
| 整步中位 ms | 8311.152 | 7898.903 | −4.960% |
| 预测目标 tokens/s | 11820.745 | 12437.676 | **+5.219%** |
| peak allocated GiB | 11.320930 | 12.507867 | **+10.484%** |
| peak reserved GiB | 12.294922 | 13.707031 | +11.485% |
| optimizer/clip 中位 ms | 384.320 | 384.269 | 基本相同 |

同样的 useful FLOPs 口径下，计算窗口 MFU 相对增幅等于吞吐增幅；这里没有测量真实
语料加载后的端到端 MFU，也不是绝对 MFU 百分点提升。实际语料的两桶占比、padding、
片段长度和数据等待会改变结果。没有把固定合成 batch 的下降 loss 当作收敛/质量结论。
allocated 比例由未舍入数值计算：`(12.507867336273193/11.320930004119873-1)*100`。

两次运行均 status=ok、5 updates / 40 microsteps，当前状态与输出权重已完整核对：

- `recipe2_baseline_static.json` → `ckpt2_recipe_20261023_203800_45200_1536_resume.pth`
- `recipe4_final_static.json` → `ckpt4_recipe_20261023_204107_15808_1536_resume.pth`
- 汇总：`experiments/results/ckpt4/comparison.json`

使用时仅选择 `--use_grad_checkpoint 4`，Muon workspace 保持 512 MiB；CLI、Config WebUI、
relay schema、autotune 计划/搜索及 profile 路径均已接受该档位。现有模板默认值与
用户保存的 recipe 不自动迁移。

### 最终回归

- 模型/CLI/flag threading/autotune/WebUI AppTest 综合 CPU 回归：202 passed、1 GPU skipped；
  后续新增 3 个 batch/dtype 预算边界检查与 1 个 down early-stop 计数检查均通过。
- packed/causal attention 与 compile policy CPU 回归：16 passed、17 GPU skipped。
- WebUI 架构图/相关 preset 与 checkpoint selector：17 passed、6 GPU skipped；
  一个既有 serialized-config/preset 不一致用例不在本次选择范围内。
- relay mode3/mode4 command parity 通过，generated schema `--check` 一致。
- 最终 CUDA checkpoint 回归：**4 passed**，覆盖 BF16/FP8、137/4096 两种长度，每项均
  比较 plain 和 packed 的 loss 与全部参数梯度。小 GPU 模型强制一层 gate/up 缓存、
  一层 FFN 重算，以覆盖混合计划。BF16 梯度相对误差阈值 2%；FP8 cosine>0.99，
  norm ratio 在 0.95–1.05 内，loss atol=0.003/rtol=0.002。

所有本次真实更新诊断均保存了最新完整状态；结束时没有残留训练/基准进程。
