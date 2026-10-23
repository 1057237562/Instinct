# ckpt3：以提高 MFU 为目标的 Dense selective checkpoint

日期：2026-10-23。实现入口：`--use_grad_checkpoint 3`。

容量更新：真实 12,288-token、accumulation=8 运行已发生 OOM。故障编译图的
新增保留张量约 14.31 GiB，不能由下方 2k micro-batch 的约 12% allocated 增幅
外推可行性。详见 [OOM 重新核算与 ckpt4 规划](reports/2026-10-23-ckpt3-oom-ckpt4-plan.md)。

## 目标与当前基线

以根目录 `selective-checkpointing-v2.md` 的“昂贵运算保留、便宜中间量重算”
为蓝本，按用户确认的优先级设计：提高 useful MFU，允许增加显存，也允许手写算子。
第一版使用原生 PyTorch selective activation checkpoint（SAC），没有强制 3 GiB
激活上限，没有启用运行时预算规划，也没有改动模型结构、loss 或精度。

当前 V2 Dense 是 32 层、hidden=1536、FFN=5504、16 Q / 4 KV heads、head_dim=96、
vocab=6400、tied embeddings，共 1,010,277,888 参数。蓝本的 1554/12/128
是假设值，不用于本实现的估算。当前 ckpt1 仅重算 attention core，不再重算 FFN。

| 档位 | 反向时的主要重计算 | 长期保留的层内结果 |
|---|---|---|
| 0 | 无显式重计算 | autograd 所需激活 |
| 1 | Attention 核心 | 投影、norm、FFN 激活 |
| 2 | 整层，受 non-reentrant early-stop 和编译器优化影响 | 层输入；旧环境变量可额外保留 Flex 结果 |
| **3** | RMSNorm、RoPE、SwiGLU 等中间量；math Attention | **Dense GEMM、受支持融合 Attention 的输出** |
| **4** | 按桶/层选择重算；短桶重算 Attention | **预算内的 KV 或融合 Attention，以及顶部部分层 gate/up**；见 [ckpt4](selective_checkpoint_mode4.md) |

ckpt3 的档位编号表示新策略，不表示比 ckpt2 更省显存。
默认模板仍选 ckpt2，现有用户模板和训练状态不迁移。

## 保存策略

在原有逐层 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` 调用点
传入 `context_fn`，保持 Dynamo 可见的 checkpoint 高阶算子。每层、每次 forward
都有独立 SAC 缓存；不引入 `torch.compiler.disable` 边界。

- `aten.mm`、`aten.addmm`、`aten._scaled_mm`：`MUST_SAVE`。
  涵盖 Q/K/V/O、gate/up/down 投影，包括 TorchAO 使用的 FP8 GEMM。
- FlexAttention 高阶算子、原生 Flash/efficient/cuDNN SDPA 算子：零 dropout
  时 `MUST_SAVE`，保留算子返回的输出及辅助量。
- 其余：`PREFER_RECOMPUTE`，允许编译器对便宜运算进行融合/分区优化。
- 不笼统保存 `bmm`。eager/math Attention 的 QK 分数是二次大小的矩阵，还会被
  causal mask 原地修改；它继续重算，避免长上下文持有 O(S²) 的 SAC 结果。
- 非零 dropout 时，融合 Attention 也重算，防止跳过随机算子后改变后续 dropout
  的 RNG 消费顺序。checkpoint 保持 `preserve_rng_state=True`。

融合算子的内部反向可能仍会计算 softmax 等中间量。“不重算 Attention”在这里
仅指不额外重放其完整 forward。外部 FA4 的未知 custom op 不在保留白名单中，
不能据此宣称其 replay 已被消除；当前验证路径是 V2 Flex 和原生 SDPA。
FP8 的量化、scale 计算等可能继续重算；缓存 `_scaled_mm` 结果不等于整个 FP8
Linear 都没有重计算开销。

旧 `INSTINCT_DENSE_SELECTIVE_CHECKPOINT=attention` 仍只控制 ckpt2 的实验路径。
ckpt3 直接选择新策略，不依赖这个环境变量。

## 与蓝本各阶段的对应

- R1/R2：通过 SAC 丢弃 norm/SwiGLU 的便宜中间量，同时缓存投影结果。
  保留 PyTorch 原有反向、参数梯度和 autocast 语义。
- R0（chunked lm_head/CE）：暂不耦合到档位。它会额外重算 head GEMM，且改变
  loss-only / logits 接口契约；当前 MFU 优先且允许增加显存，无须默认付出这项成本。
- R3：复用已有的 packed/causal Flex 和 SDPA dispatch，保留原 mask、RoPE、label
  语义，不重写 eager Attention 反向。
- R4/R5：预算不足时可选择原有 ckpt2；本版不自动降档、不改 micro-batch 或
  accumulation，不据此改变已有 resume 的 data cursor。

手写算子可作为后续实测驱动的优化：若 trace 显示 RMSNorm/SwiGLU 的 replay 和
内存读写成为主瓶颈，再比较 fused RMSNorm、SwiGLU epilogue recompute；
现有 MoE Triton SwiGLU 可作参考，但不能直接替换 Dense FP8 down projection，
否则可能绕过 TorchAO、引入 graph break 或改变 BF16 舍入点。

## 显存与速度预期

对于 bf16 GEMM 输出，保留的主要输出账本约为：

`T × L × [2 × (2qd + 2kd + 2H + 2F) + residual_bytes × H]`

其中 `T=B×S`，`qd=Q_heads×head_dim`，`kd=KV_heads×head_dim`。
V2、T=2048、L=32、FP32 residual 的这部分约 **2.56 GiB**，另加 LSE、
mask、层外 head/loss、反向临时量及分配器保留空间。编译器可能消除不需要保存的
输出，因此这只是主要存储项的账本，**不是测得的激活峰值或总显存承诺**。
Muon mantissa 模式的 embedding/head/norm 保持 FP32，不能一律用 bf16 计算残差流。

理想化的 Dense GEMM 成本由 `forward + backward + replay ≈ 4F` 降至 `3F`，
对应计算受限情况下该部分至多约 4/3 加速。实际 ckpt2 的 early-stop、编译优化、
Attention、带宽、优化器、padding、数据等待都会缩小或改变收益，不承诺固定 MFU 增幅。
收益必须在相同参数/激活精度、attention backend、token batch、compile、优化器下比较。

useful MFU 的分子不能加入 checkpoint 重计算，否则更多无效计算会虚增指标。
同一模型与 batch 下，采用同一峰值算力口径时，MFU 相对变化等于 useful tokens/s
相对变化。FP8/BF16 混合路径需要保留原有分精度统计口径。

## 使用与兼容范围

Config WebUI：加载 **Instinct V2**，在 **梯度检查点模式** 选择 **3 Dense selective**。
命令预览会包含 `--use_grad_checkpoint 3`；relay schema 使用相同 CLI choices。
也可将现有已验证启动命令中的 `--use_grad_checkpoint 2` 改为 `3`。

第一版支持 standard Dense 主干、standard 残差。MoE、linear、RecurrsiveInstinct、
mHC、AttnRes 显式报错；避免把未实现的档位静默当作关闭 checkpoint。
训练时要求无 KV cache；eval 和 no-grad generation 正常绕过 checkpoint。
logits 返回格式不变，冻结基座的 LoRA 和 logits-only 目标有 CPU 梯度覆盖。
SAC 单次 forward 的图应只 backward 一次，不支持对同一张图反复 `retain_graph=True`
消费；多次 micro-batch forward/backward 的梯度累积正常。

参数 shape 与 state_dict key 不变；配置序列化保存 mode=3。
不重新计算既有 resume 的 batch/token budget。ckpt3 的显存更高，原有显存估算
没有经过该档位的整机校准；不能把 ckpt2 接近显存上限的 token budget 直接视为可用。
自动调优已接入 ckpt3：标准 Dense + standard 残差默认搜索 1/2/3；一键准备
保留已选的 3，试跑和 profile 应用支持 3，显存边界按策略分别记录。
自定义搜索可以只选 `[2, 3]`。其他拓扑仍排除 3；已有 autotune profile 的训练
指纹与旧测量不能视为 ckpt3 的验收结果，修改代码后需重新校准。

## 验证与复现

`tests/test_mfu_checkpoint.py` 检查 CPU FP32/BF16、packing、dropout、loss/logits、
全部参数梯度、RNG 状态、梯度累积、LoRA、AOT 编译主干，以及实际执行的 GEMM FLOPs。
`tests/test_dense_selective_checkpoint.py` 覆盖 V2 真实 head/FFN 形状的 CUDA
BF16/FP8 编译梯度对照（两层、普通/packed 序列）。

`experiments/profile_checkpoint_policy.py` 不更新参数，只测固定权重的 forward+backward。
它可在 CPU 计数执行的矩阵 FLOPs，或在 CUDA 记录热态耗时、allocated/reserved 峰值。
**该结果不包含优化器、真实 loader 或保存，不能当作整段训练的 MFU。**

GPU 空闲时，分别在独立进程运行 mode 2/3，例如：

```powershell
$env:PYTHONUTF8='1'
$env:INSTINCT_MUON_MANTISSA='1'
python experiments/profile_checkpoint_policy.py --mode 2 --device cuda --layers 32 --seq 2048 --compile --warmup 10 --steps 20 --repeats 3 --output experiments/results/ckpt3/ckpt2_bf16.json
python experiments/profile_checkpoint_policy.py --mode 3 --device cuda --layers 32 --seq 2048 --compile --warmup 10 --steps 20 --repeats 3 --output experiments/results/ckpt3/ckpt3_bf16.json
```

完整训练验收仍应采用同起点、同真实 batch 序列、同优化器/accumulation 的成对运行，
记录至少三个稳定窗口、useful tokens/s、端到端 useful MFU、两种显存峰值和 loss；
通过 trainer 的 save-and-pause 保存并核对最新完整 resume state 后结束。

## 本机证据（2026-10-23）

RTX 5070 Ti，PyTorch 2.13.0+cu132，V2 全部 32 层，batch=1、seq=2048，
单文档 causal Flex，BF16 GEMM/激活，`INSTINCT_MUON_MANTISSA=1` 使辅助参数及
残差流保持 FP32。`torch.compile` 默认模式；每个档位独立进程，10 次 warmup、
3×20 次计时，模型权重固定，无 optimizer update。

| BF16 前向＋反向指标 | ckpt2 | ckpt3 |
|---|---:|---:|
| 三个窗口 ms/次 | 248.51 / 249.16 / 249.25 | 204.50 / 204.84 / 205.43 |
| 中位数 ms/次 | 249.16 | 204.84 |
| 输入 tokens/s | 8,219.62 | 9,998.16 |
| Peak allocated GiB | 4.097 | 4.629 |
| Peak reserved GiB | 4.588 | 4.828 |
| 固定输入 loss | 9.07556534 | 9.07556534 |

该测量吞吐提高 **21.64%**、耗时减少 **17.79%**，peak allocated 相对 ckpt2
增加 **12.99%**（绝对增量 0.532 GiB）。显存增幅统一按
`(ckpt3_peak_allocated / ckpt2_peak_allocated - 1) × 100%` 计算，使用原始未舍入数值。
这些显存数包含本测试的权重、梯度和临时张量，**不含优化器状态**。
不能将其视为正式训练总显存；也不能把 21.64% 直接写成端到端 MFU 提升。
warmup（含冷启动/编译）分别 84.26 / 86.88 秒，不计入热态吞吐。
原始记录位于忽略目录 `experiments/results/ckpt3/ckpt{2,3}_bf16.json`。

同一形状、同一精度存储及计时设置下，TorchAO 0.18.0 tensorwise FP8（auto filter）
的结果如下；160 个 Linear 被转换，其余保留原计算路径：

| FP8 混合计算前向＋反向指标 | ckpt2 | ckpt3 |
|---|---:|---:|
| 三个窗口 ms/次 | 174.55 / 174.54 / 174.65 | 148.57 / 148.59 / 148.60 |
| 中位数 ms/次 | 174.55 | 148.59 |
| 输入 tokens/s | 11,732.82 | 13,782.93 |
| Peak allocated GiB | 4.096 | 4.588 |
| Peak reserved GiB | 4.598 | 4.738 |
| 固定输入 loss | 9.11071110 | 9.11221695 |

FP8 前向＋反向吞吐提高 **17.47%**，peak allocated 相对 ckpt2
增加 **12.02%**（绝对增量 0.492 GiB）。
warmup（含冷启动/编译）分别 158.06 / 131.82 秒；原始报告是
`experiments/results/ckpt3/ckpt{2,3}_fp8.json`。FP8 loss 差约 0.001506，
不能声称 bitwise 等价；两层 CUDA FP8 对照已通过 loss/梯度容差，完整真实训练
的成对质量验收仍未执行。转换层数也不是完整 kernel 执行覆盖率报告。

同尺寸的两层 CPU FP32 切片（1×128）中，执行的 Dense `aten.mm` FLOPs 从
67,243,081,728 降为 55,566,139,392；这是算子计数，不能换算成 GPU wall-time。

Focused CPU 检查 88 passed（排除已确认的旧预设断言）；CUDA BF16/FP8 编译
梯度检查 2 passed。新增 relay mode=3 CLI/模型快照传递检查及 schema `--check`
通过。另有两处已有检查问题，本次没有修改其预设或运行二进制：

- `test_current_serialized_configs_match_their_training_presets` 将 HEAD 中已经是 V2
  的 `trainer/config_pretrain.json` 断言为 V1 MoE。
- relay 现有完整 parity 测试将同值的 `muon_workspace_mb` 数字 `0` 和 CLI 字符串
  `0.0` 判为不同；使用 HEAD 版本的 command builder 也能重现。直接文件运行该
  unittest 另需正确的 Python import path，可用 `python -m unittest` 从仓库根运行。

API 依据：[PyTorch selective checkpoint 文档](https://docs.pytorch.org/docs/2.14/checkpoint.html#torch.utils.checkpoint.create_selective_checkpoint_contexts)。
本机实现与测试使用安装的 PyTorch 2.13.0+cu132，不能将在线文档版本当作本机版本。
