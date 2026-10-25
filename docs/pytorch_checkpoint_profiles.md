# PyTorch Dense 四档梯度检查点

## 结论

新接口是 `--checkpoint_profile ckpt1|ckpt2|ckpt3|ckpt4`，显存从高到低、重算从少到多。
旧 `--use_grad_checkpoint 0|1|2` 保留原语义；新字段未设置时使用 `legacy`，避免续训配置被静默重新解释。
`off` 明确关闭检查点；CLI 显式 profile 优先于 JSON，未指定 CLI 时保留 JSON 的 profile。
WebUI 的 **PyTorch Dense 显存档位** 已接入配置导出和启动命令。

| 新档位 | 保留的昂贵计算结果 | 重算内容 |
|---|---|---|
| ckpt1 | Q/K/V/O、FFN gate/up 投影、融合注意力输出与统计量 | RMSNorm、RoPE、激活、残差等 |
| ckpt2 | Q/K/V/O 投影、融合注意力输出与统计量 | 另重算 FFN gate/up |
| ckpt3 | 融合注意力输出与统计量 | 另重算 Q/K/V/O |
| ckpt4 | 层边界输入 | 整层 non-reentrant checkpoint，受 early-stop 提前终止 |

采用 PyTorch Selective Activation Checkpointing（SAC）。所有档位保留 RNG 状态，训练时使用 PyTorch SDPA，
以便按 ATen 算子缓存融合注意力结果；不走外部 FA4。推理和带 KV cache 的调用不启用新检查点。
不缓存 math-attention 的平方大小 scores/probabilities。当前四档针对 Dense；MoE、Loop、Linear 明确拒绝新档位，
这些架构及 cuBLAS 原生后端继续使用旧 0/1/2，WebUI 不向原生后端导出新字段。

一个有用的优化：最后的 FFN down projection 会被 non-reentrant checkpoint 的 early-stop 跳过，
因此缓存它不节省矩阵乘法 FLOPs。新策略不保存它。标准模型中以 GEMM 输入/输出维度识别投影；
更改 head_dim 或 FFN 维度后应重新运行分析器，而不是套用下表。

## 仿真条件与口径

本次环境：**PyTorch 2.8.0+cpu，无 CUDA GPU**。以下是执行真实 forward/autograd 路径的 FakeTensor 仿真，
不是 GPU 实测，不表示训练速度或 TFLOP/s。模型与 WebUI `instinct2` 预设一致：

- 16 层，hidden 768，FFN 2432，Q8/KV2，head_dim 96，vocab 6400，绑定 embedding，约 118.19M 参数。
- **序列长度固定 4096**；比较 microbatch 32、16、8、4。BF16 autocast；FP32 权重、梯度、AdamW 两份动量，foreach AdamW 与梯度裁剪。
- dropout 0，无 padding mask，SDPA 融合路径；包含 logits、完整交叉熵与 RoPE 缓冲区。
- 模拟稳态梯度累积：已有梯度和优化器状态；上一 microbatch 返回的 logits/hidden states 在下一次 forward 返回前仍存活，匹配当前 trainer。
- 内存按底层 storage 去重、跟踪 weakref 释放并作 512-byte 对齐，包含重算临时张量和 optimizer step。
- FLOPs 使用 PyTorch 公式：FMA=2，计 GEMM 和完整方阵 SDPA（包括 flash backward 内部 QK 重算）。
  不计逐元素、归约、优化器运算，也不减去因果注意力 kernel 跳过的 tile。因此是统一算法量比较，不能换算为耗时。
- 不模拟 CUDA allocator 缓存/碎片、opaque kernel workspace、CUDA context、其他进程、torch.compile/CUDA Graph、DDP 或 Early Exit 多头 loss。
  CPU SDPA 张量布局只近似 CUDA SDPA，显存门槛必须在目标 GPU 复核。

统计器避免使用 module backward hooks：本地 PyTorch 2.8 的 `MemTracker`/`FlopCounterMode` 模块归因 hooks
会延长 checkpoint 重算张量寿命，得到异常偏大的反向峰值。本分析器只用 operator dispatch 和 storage weakref，
复用官方 FLOP 公式而不安装模块 hooks。

## batch 32 × 4096 的结果

预算采用用户此前观察到的 **2 GiB 额外占用假设**，不是实测“PyTorch 固定开销”。另预留卡总显存的 10%：

`所需总容量 = (仿真张量峰值 + 2 GiB) / 0.9`。

| 档位 | 张量峰值 GiB | 预算总容量 GiB | 每步计算量 TFLOP | 比关闭多计算 | 比关闭少用显存 |
|---|---:|---:|---:|---:|---:|
| off 对照 | 118.79 | 134.21 | 185.285 | — | — |
| ckpt1 | 50.19 | 57.99 | 185.285 | 0.00% | 57.74% |
| ckpt2 | 31.19 | 36.88 | 200.953 | 8.46% | 73.74% |
| ckpt3 | 23.69 | 28.55 | 207.138 | 11.79% | 80.05% |
| ckpt4 | 20.63 | 25.15 | 233.526 | 26.04% | 82.63% |

**microbatch32 时，24 GiB 卡四档都不满足这个预算。** ckpt4 的张量峰值虽小于 24 GiB，
仍不能忽略额外占用与余量。仅换档位不能保证解决 OOM，需要减小 microbatch。

ckpt1 的“0%”仅指上述 GEMM/SDPA 计算量，仍会重算逐元素算子，不能宣称零耗时开销。

以显存最少的 ckpt4 为参照，设某档计算量为 F、张量峰值为 M：

- **总显存收益比** = `(F4 − F) / M`：每占用 1 GiB，总共省下多少 TFLOP/步。
- **增量显存收益比** = `(F4 − F) / (M − M4)`：比 ckpt4 多占 1 GiB，省下多少 TFLOP/步，适合决定是否升级档位。

| 档位 | 相比 ckpt4 节省 TFLOP/步 | 节省比例 | 总显存收益比 | 增量显存收益比 |
|---|---:|---:|---:|---:|
| ckpt1 | 48.241 | 20.66% | 0.961 | 1.632 |
| ckpt2 | 32.573 | 13.95% | 1.044 | 3.084 |
| ckpt3 | 26.388 | 11.30% | 1.114 | 8.617 |
| ckpt4 | 0 | 0% | 0 | 不适用（分母为 0） |

比值单位均为 **TFLOP/步/GiB**。相邻档位的边际收益：ckpt4→3 为 **8.617**，ckpt3→2 和 ckpt2→1 均为 **0.825**。
在 4096 长序列中，优先保留融合注意力的边际收益约为保留投影的 **10.45 倍**。
所以预算刚够时优先升到 ckpt3，随后再增加投影缓存。在本次比较的策略中，四个新档位均位于 GEMM/SDPA 计算量与显存的 Pareto 前沿。
旧 mode1 峰值 80.79 GiB、200.953 TFLOP/步，被新 ckpt2（31.19 GiB、同计算量）支配；旧 mode2 与 ckpt4 相同。

## 按显存选档

以下均含 2 GiB 额外占用与 10% 余量，保持 **seq=4096**；容量按 GiB 计算。
先固定 **microbatch8**，得到可用于常见显卡的四档：

| 档位 | 张量峰值 GiB | 预算总容量 GiB | 常见容量选择 | 每步 TFLOP |
|---|---:|---:|---|---:|
| ckpt1 | 14.05 | 17.84 | 24 GiB | 46.321 |
| ckpt2 | 9.30 | 12.56 | 16 GiB | 50.238 |
| ckpt3 | 7.43 | 10.47 | 12 GiB | 51.784 |
| ckpt4 | 6.66 | 9.62 | 10 GiB | 58.381 |

microbatch8 的容量区间为：`[9.62,10.47)` 选 ckpt4，`[10.47,12.56)` 选 ckpt3，
`[12.56,17.84)` 选 ckpt2，`>=17.84` 选 ckpt1。精确边界以 JSON 未四舍五入值为准。
同一序列长度下，不同 batch 的增量收益比相同；总显存收益比因固定参数/优化器占比不同而改变。

若优先容纳更大的 microbatch，可参考下面的预算组合。更大 batch 不等于实测更快，吞吐仍需 GPU 校准：

| 设备总容量 | 可行组合 | 含余量预算 GiB | 减少重算的另一选择 |
|---|---|---:|---|
| 8 GiB | batch4 + ckpt3 | 7.46 | ckpt4 预算 7.04，但额外重算更多 |
| 10 GiB | batch8 + ckpt4 | 9.62 | batch4 + ckpt2，预算 8.50 |
| 12 GiB | batch8 + ckpt3 | 10.47 | batch4 + ckpt1，预算 11.14 |
| 16 GiB | batch16 + ckpt4 | 14.80 | batch8 + ckpt2，预算 12.56 |
| 24 GiB | batch16 + ckpt2 | 20.67 | batch8 + ckpt1，预算 17.84 |

batch16 四档张量峰值为 **26.10 / 16.60 / 12.85 / 11.32 GiB**；batch4 为 **8.03 / 5.65 / 4.72 / 4.33 GiB**。
若原 microbatch32、accumulation_steps8，降为 batch16/8/4 时，可对应改为 accumulation_steps16/32/64，
维持相同有效 batch（同 GPU 数）。这不改变序列长度，仍是 4096。
推荐器没有可行档位时返回 null，不会假设 ckpt4 永远能避免 OOM。

**4096 序列尤其要检查普通注意力回退。** batch4 的 math 路径四档峰值为 **12.25 / 10.03 / 9.15 / 9.15 GiB**；
最小预算也需 **12.39 GiB**，不适用上面的 8 GiB 组合。batch32 math 峰值为 **85.41 / 67.59 / 60.56 / 60.56 GiB**，
off 达到 512.72 GiB。关闭 flash、padding mask 等可能进入此路径。
math 路径没有融合注意力可缓存，ckpt3 会与 ckpt4 合并为同一效果。

## 复现与 GPU 校准

```bash
# 完整预设仿真，含旧 0/1/2 对照，写入每个档位 FLOPs、峰值、两种比值和容量推荐
python experiments/profile_checkpoint_profiles.py --output checkpoint_profiles.json

# 8 GiB 预算：microbatch4、序列4096；开销和余量可按实际情况调整
python experiments/profile_checkpoint_profiles.py --batch 4 --seq 4096 --capacities 8 --overhead-gib 2 --reserve-fraction 0.1

# 自定义 Dense JSON / 普通注意力回退
python experiments/profile_checkpoint_profiles.py --config trainer/config_pretrain.json --batch 4 --seq 4096 --attention math

# 目标 GPU：两轮 warmup 后测量 allocated/reserved 峰值、耗时和吞吐；OOM 行明确标记
python experiments/profile_checkpoint_profiles.py --mode cuda --batch 8 --seq 4096 --steps 10 --output checkpoint_profiles_cuda.json

# 训练只需增加这一参数；模型/数据/形状继续使用原来的训练参数
python trainer/train_pretrain.py --checkpoint_profile ckpt2
```

`--config` 会覆盖预设字段；分析器当前限制 Dense、dropout=0、BF16 autocast、FP32 参数/AdamW。
若改变参数精度、优化器、batch、序列或编译模式，应重新校准，不能使用固定的 8/12/16 GiB 标签替代测量。

原始数据：[batch32 SDPA](checkpoint_profiles_b32_s4096.json)、[batch16 SDPA](checkpoint_profiles_b16_s4096.json)、
[batch8 SDPA](checkpoint_profiles_b8_s4096.json)、[batch4 SDPA](checkpoint_profiles_b4_s4096.json)、
[batch32 math](checkpoint_profiles_b32_s4096_math.json)、[batch4 math](checkpoint_profiles_b4_s4096_math.json)。

验证：四档的 FP32/BF16 输出、全部参数梯度、dropout RNG 与 off 一致；覆盖 mask、Early Exit loss、
eval/KV cache、配置往返、CLI/JSON 优先级、Dynamo eager backend 和内存/预算排序。
CUDA BF16 对照测试已提供，本环境因无 GPU 跳过；Inductor、真实 CUDA 显存与吞吐尚未测量。

参考：[PyTorch checkpoint API](https://docs.pytorch.org/docs/stable/checkpoint.html)、
[PyTorch activation checkpointing 技术说明](https://pytorch.org/blog/activation-checkpointing-techniques/)、
[PyTorch CUDA memory 统计边界](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)。
