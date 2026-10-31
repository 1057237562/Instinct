# cuBLAS 程序如何提高计算吞吐

> 调查对象：Instinct V2 Dense（1,010,277,888 参数）的原生 cuBLASLt/Triton 训练栈，以及主仓 PyTorch V2 路径中的 ATEN/cuBLAS GEMM 选择。
>
> 代码版本：原生栈位于 `C:/Users/L1bra/.codex/worktrees/cublas-throughput/Instinct`，分支 `codex/cublas-throughput`，HEAD `3f917b8`；主仓位于 `D:/AI/Instinct`，分支 `refactor/repo-layout`，HEAD `22c9d91`。
>
> 结论日期：2026-10-28（当前复测；旧数据保留原日期并标注“历史”）。

## 0. 先给结论

当前项目里有两条都叫“cuBLAS”的路线，必须先分开：

1. **原生 cuBLASLt/Triton 栈**：训练进程不导入 PyTorch，直接通过 CUDA Driver API、cuBLASLt 和 Triton 执行前向、反向、优化器和数据管线。它的吞吐提升不是“换用一个更快的 GEMM 库”，而是围绕 cuBLASLt 重写了整个 step：GEMM plan 预生成、直接转置描述符、FP8 与生产者融合、选择性重算、异步分配、packed attention 范围裁剪、按形状批处理的 Muon，以及 Rust/mmap 数据预取。
2. **主仓 PyTorch 路径**：模型仍由 PyTorch 执行，但 `model/compile_policy.py` 在匹配 V2 Dense、BF16 参数、tensorwise FP8、auto filter、Blackwell 等条件时，把 Inductor 的 GEMM 后端限定为 `ATEN`，让 FP8 GEMM 走 cuBLAS，而不是让 Inductor 自动选择的 Triton FP8 GEMM 执行。

最重要的实测结论是：

- **原生栈当前生产验证（2026-10-27）**：**12,968.49 targets/s**，1,539,764 个有效 targets / 118.7312 s，16 个测量更新窗口；比 2026-10-26 的 **11,398.30** 高 **+13.8%**。这是当前最可信的原生生产数字；配置为 hybrid attention、cuBLAS Muon、ckpt4 768 MiB + 按桶 1024/1280 MiB，`fused_residual_norm=false`。
- **主仓 PyTorch 当前复测（2026-10-28，当前文件夹）**：baseline **11,053.17**、optimized **11,105.07** targets/s；按总测量时间折算的聚合吞吐只高 **+0.47%**，不是旧报告的 `+15.62%`。复测时 GPU 上仍有 `RimWorld.exe`/`QQ.exe`，optimized 还复用了共享 compile cache 且缺少旧日志中的 TMA autotune 输出；fresh-cache 复测因 ptxas/OOM 失败。因此这组只能算“带干扰的当前观测”，不能当作干净的 TMA/ATEN 对照。
- **旧 PyTorch `14,399.67` 仍是历史值**：当前原生 12,968.49 比它低 **9.94%**；当前 PyTorch optimized 11,105.07 比它低 **22.88%**。旧 `+15.62%` 只描述 2026-10-26 那次运行，不能当成当前代码的复测结果。
- **固定权重当前只测了原生**：当前原生含 FRN 的 6 个窗口合计 **10,191.62 targets/s**（中位墙钟 **1,154.70 ms**；短桶约 1,243–1,244 ms，长桶约 1,065–1,066 ms），比旧原生 `1,063.79 ms / 11,078.29` 慢约 **8.0%**；短桶慢约 14.6%，长桶接近。当前 PyTorch 固定权重结果 `current_20261027_torch_fixed.json` 仍是 `running` 且 bucket 为空，不能用；旧 `1083.4 vs 796.3` 只能作历史对照。
- **不能得出“原生已超过 PyTorch”的结论**：原生当前生产 12,968.49 与当前 PyTorch 合成 A/B 11,105.07 的 scope 不同，不能直接比较；与旧 PyTorch 14,399.67 比仍低约 9.94%。当前固定权重原生 1,154.70 ms 也没有可用的当前 PyTorch 对手，不能据此判断固定权重谁更快。

一句话：**原生程序提高吞吐的核心，是把“GEMM 调度、布局、量化、重算、显存生命周期、注意力边界、优化器和数据供给”一起围绕固定形状训练做了系统级协同；单纯调用 cuBLASLt 并不足以提高吞吐。**

## 1. 口径与可比性

### 1.1 硬件与模型

| 项目 | 配置 |
|---|---|
| GPU | RTX 5070 Ti 16 GB，SM120，Windows |
| 模型 | Instinct V2 Dense：32 层，H=1536，FFN=5504，16 Q / 4 KV heads，head_dim=96，vocab=6400，tied embedding |
| 参数量 | 1,010,277,888 total / active |
| 原生精度 | BF16 隐藏权重和计算，FP32 embedding/head/norm，FP8 tensorwise，uint16 mantissa，BF16 momentum |
| 原生生产形状 | 4×3072（每行 3 文档）与 3×4096（每行 1 文档）交替，token budget 12288，accumulation 8 |
| 对照框架 | PyTorch 2.13.0+cu132，`max-autotune-no-cudagraphs`，ATEN/cuBLAS GEMM，长桶 Flex TMA backward |

### 1.2 吞吐定义

报告里的 **targets/s** 指“有效 next-token 预测目标数 / 墙钟时间”。它排除了 padding、文档起始位置和移位后没有目标的位置，因此不等于原始 token 数，也不等于模型 FLOPs 利用率。

两类测量不可混用：

- **完整更新**：包含前向、反向、梯度累积、裁剪、Muon/AdamW 更新。2026-10-25 的原生表是每组 5 次实际更新，前 2 次预热，后 3 次计时；每次累计更新包含 98,244 个有效 targets。2026-10-26 的旧生产验证是 160 个微步、20 次实际更新，其中 128 个微步用于吞吐统计；2026-10-27 的当前复测是 160 个微步、4 次预热 + 16 次计时更新，1,539,764 个有效 targets / 118.7312 s。
- **固定权重前反向**：保留真实优化器状态和梯度缓冲，但不执行参数更新，不计 loader、初始化、编译和保存。2026-10-27 的原生/PyTorch 旧对照属于这一类；当前只完成了原生固定权重复测（`current_20261027_fixed.json` / `..._no_frn.json`），当前 PyTorch 固定权重文件 `current_20261027_torch_fixed.json` 仍是 `running` 且没有 bucket 数据。
- **当前 PyTorch 复测**：2026-10-28 在 `D:/AI/Instinct` 用合成累积更新（不含 loader/save）重跑 baseline/optimized，5 个测量 update；这不是固定权重，也不等同于旧 `identical_batches.npz` 对照。

### 1.3 显存口径

原生栈同时报告 `peak live`、`peak arena` 和 CUDA async pool 的 reserved/used。`live` 是活跃张量字节，`arena` 还包含分配器空闲缓存及 cuBLASLt workspace；它们都不等同于 PyTorch 的 `allocated`，也不能把 native live 与 PyTorch allocated 直接相加或相减。

静态审计里的命令数、分配字节数来自形状/分配/命令模型，**不是 tok/s**。本报告会把“实测吞吐”和“静态审计”分开标注。

## 2. 原生 cuBLASLt/Triton 栈是什么

原生栈的入口是 `trainer/train_cublas.py`，实现集中在 `model/dense_cublas/`：

```text
trainer (不 import torch)
  └─ Engine / VJP 自动微分
       ├─ DenseOps / CublasDenseOps
       │    ├─ native/bridge.cpp：cuBLASLt plan、descriptor、workspace
       │    └─ backend.py：ScheduleConfig、ExecutionPlan、graph capture/replay
       ├─ Triton kernels：RMSNorm、RoPE、SwiGLU、量化、CE、梯度归约
       ├─ 可选 cuDNN SDPA：仅长桶 packed attention
       ├─ memory.py：arena、异步 allocator、allocation registry
       ├─ muon.py：split-weight Muon + 按形状分批
       └─ Rust dataset_compiler：Parquet 打包、mmap、行组缓存
```

cuBLASLt 在这个栈里只负责 **GEMM**：Linear、LM head、dX/dW、Muon 的 Gram/polynomial/update 矩阵乘法。attention 的 softmax 部分、归一化、RoPE、SwiGLU、量化、交叉熵和 AdamW 等仍由 Triton 自定义 kernel 完成。因此，只看 `native/bridge.cpp` 只能看到 GEMM 层，看不到完整的吞吐来源。

主仓 PyTorch 路径则是另一套：

```text
PyTorch V2 Dense
  └─ torch.compile / Inductor
       ├─ 满足条件时：max_autotune_gemm_backends = "ATEN"
       │    └─ FP8 GEMM 走 ATen/cuBLAS，而非 Inductor 自动选择的 Triton FP8 GEMM
       └─ 长桶 attention backward：Flex TMA（单独优化，不是 cuBLAS）
```

`model/compile_policy.py` 的选择是**按模型条件生效**的，不是全局关闭 Inductor 自动调优：

- 仅当 `mode` 是 `max-autotune` 或 `max-autotune-no-cudagraphs`；
- FP8 recipe 是 `tensorwise`，filter 是 `auto`；
- `param_dtype=bf16`，标准 residual、标准 Dense、非 MoE；
- hidden/FFN/layers/heads/head_dim 与 V2 Dense 完全匹配；
- 当前设备被识别为可用的 Blackwell 训练环境。

满足后，它把 `options['max_autotune_gemm_backends']` 设为 `ATEN`，并保留该 mode 的 graph capture 等其他选项。这样做的原因是：Inductor 的自动调优在 V2 的 1536/5504 宽投影上可能选中较慢的 Triton FP8 GEMM，而 ATen/cuBLAS 的厂商调优 kernel 在该形状上更快。

## 3. 原生栈提高吞吐的七个机制

### 3.1 GEMM plan 缓存与直接布局：减少“提交”和“搬运”

**问题**：训练 step 有大量固定形状 GEMM。如果每次调用都重新做启发式选核、重新打包转置矩阵，CPU 提交和显存搬运会吃掉 Tensor Core 的收益；在低精度小矩阵上尤其明显。

**做法**：

- `backend.py` 的 `ScheduleConfig` 默认给每个 stream 32 MiB cuBLASLt workspace，最多 8 个候选算法；`tune=False` 时用启发式选择，只有显式开启才计时候选。
- plan 按 stream、shape、布局缓存；构建/捕获阶段生成，重放时不再反复走 Python 侧的选核逻辑。
- 对齐的 BF16/FP16/FP32 直接使用原始连续存储和 transpose 描述符，计算 `Cᵀ = op(B)ᵀ @ op(A)ᵀ`；B 以 row-major `[N,K]` 存储时可直接视为 column-major `[K,N]`，不再物理打包、padding 或 crop。
- 只有 SM120 的 FP8 仍保留对齐后的 TN 打包；没有放宽精度或开启 fast accumulation。
- C++ bridge 升到 ABI 2；旧 ABI 仍可回退到 pack 路径。

**为什么能提高吞吐**：

1. 减少每次 GEMM 的 host 侧提交和 plan 选择；
2. 消除 dX/dW 的物理转置、padding 和 crop，减少整块显存读写；
3. 让 cuBLASLt 直接看到真实布局，避免为了“凑一个漂亮布局”而多一次拷贝。

**证据与边界**：

- 静态审计中，B8 S256 每次审计消除约 **515 MiB** 的 GEMM 显式读写。
- 该收益是“减少搬运”，不是“减少数学运算”；GEMM 次数和 TFLOP 基本不变。
- plan 缓存本身不保证选中全局最快算法；共享配方没有把 Lt autotune 作为默认，因为候选计时会引入启动成本和波动，且在小窗口内不一定稳定。

### 3.2 FP8：降低带宽，但必须把量化成本摊薄

原生栈使用：

- 前向输入/权重：E4M3；
- 反向 dY：E5M2；
- tensorwise current amax scaling；
- FP32 GEMM accumulation；
- `fp8_filter=auto` 对齐该机器 TorchAO 的过滤规则：排除共享 head、不满足 16 对齐的投影，跳过 `K≤4096 且 N≤1024` 的小矩阵。

仅把 GEMM 换成 FP8 并不自动更快。更早的 198M MoE / RTX 5060 Ti 基准里，原生 BF16 约 **2,320 tok/s**，原生 FP8 约 **2,056 tok/s**，FP8 反而更慢；这说明如果量化、打包和逐元素成本没有被摊薄，低精度 GEMM 的收益会被抵消。该结果与当前 1.01B V2 不是同一配置，不能外推，但足以说明“FP8 不是免费加速”。

原生栈真正有用的是把量化融进生产者：

- `ffn.py` 的 SwiGLU 前向同时计算激活、amax 和 FP8 量化值；
- amax 使用**实际写回 dtype 舍入后的值**，避免“统计的是高精度、写回的是低精度”造成 scale 偏差；
- 反向在生成 dY 时直接写 FP8，并为对齐的 E5M2 梯度同时写正常和转置布局，后续 dW 复用转置结果，不再单独打包；
- `projection_cast` 把 Q/K/V 或 gate/up 的批量 cast 合并，但保留独立 GEMM 的参数身份；
- 残差/RMSNorm 等融合 kernel 在写出归一化结果时直接写计算 dtype，减少一次 BF16→FP8 的中间张量。

这样 FP8 的收益来自“更少的 operand 字节 + 更少的量化 kernel + 更少的中间读写”，而不是只把 `cublasLtMatmul` 的 A/B 类型改成 FP8。

### 3.3 算子融合：把多次全局读写变成一次数据流

原生栈的融合不是“把所有东西塞进一个 kernel”，而是按数据流把生产者/消费者合并：

| 融合 | 位置 | 消除的中间步骤 |
|---|---|---|
| residual add + RMSNorm + cast + amax | `fused.py` | 残差结果、归一化结果、量化输入之间的多次全局读写 |
| Q/K RMSNorm + RoPE 前向/反向 | `fused.py` | 单独的 norm、RoPE、cast kernel |
| SwiGLU + amax + FP8 量化 | `ffn.py` | SwiGLU 输出、amax 扫描、量化写入 |
| 梯度求和/合并 | `tensor_kernels.py` | 多路梯度先物化再相加 |
| 梯度累积 + 范数 partials | `tensor_kernels.py`、`optim.py` | 整模型范数中间张量 |
| AdamW 清梯度 + 参数更新 | `optim.py` | 单独的 zero_grad 和更新 kernel |
| shifted CE + 有效目标计数 | `tensor_kernels.py` | 完整 FP32 log-softmax 张量 |

**为什么能提高吞吐**：每个被消除的中间张量都意味着至少一次写和一次读。对 32 层、每层多个投影的模型，这些逐元素 pass 的累计带宽很可观；融合还减少了 kernel launch 数和 Python 提交数。

**静态审计证据**（注意：是命令/分配审计，不是 tok/s）：

- Dense B1 S4096 eager：命令数 **2187 → 1311**（第一轮优化）；后续 `dense_schedule_optimization` 再把核心路径从 **1311 → 748**，相对 1311 为 **-42.9%**；graph 路径 **1321 → 717**，**-45.7%**。
- 同一核心路径：arena 峰值 **1.902 → 1.666 GiB**（**-12.4%**）；graph arena **2.183 → 1.765 GiB**（**-19.1%**）。
- 复制读写量 **6.446 → 2.860 GiB**（**-55.6%**）。
- cuBLASLt GEMM 次数 **227** 不变，统计算量 **2.054 TFLOP** 不变。这组数字说明收益主要来自调度、布局和中间流量，而不是偷偷减少了计算。

**边界**：2026-10-27 的对照指出，原生仍然没有达到 Torch 的跨算子融合范围。Torch 生成的 FFN kernel 可一次输出 SwiGLU 激活转置 FP8 以及 gate/up 梯度的正向和转置 FP8，共 5 路结果；原生仍会写出中间 BF16 梯度，再由独立 kernel 量化。原生每桶 SwiGLU+amax 约 **59 ms**、E5M2 双布局量化约 **35 ms**、梯度合并约 **29 ms**、残差 binary 约 **28 ms**、pack 约 **14 ms**。这些是当前成本，不能逐项都当作可省时间。

### 3.4 选择性重算：用可控缓存换掉重复前向

原生 `use_grad_checkpoint=4` 的语义不是“整层全部重算”，而是：

1. 整层建立重算边界；
2. 按实际 B×S、激活 dtype 和 `selective_cache_mib` 预算，保留部分投影输出或 fused attention 的 O/LSE；
3. 反向重新建立 VJP，但保留的 GEMM/attention 前向不重复执行；
4. 参数梯度仍按正常链路计算。

共享配方 `trainer/recipes/instinct_v2_native.json` 使用 `selective_cache_mib=768`：

- 4×3072：保留全部 32 层 K/V，不额外保留 gate/up；
- 3×4096：保留顶部 20 层 attention O/LSE，不保留 gate/up；
- 1792 MiB 的候选配置可保留更多 gate/up，但在实测中出现过 33.448 秒慢步和 15.1875 GiB CUDA pool reserved 高水位，因此共享配方选择更稳定的 768 MiB。这不是“吞吐最优”的证明，而是稳定性取舍。

**为什么能提高吞吐**：重算省的是显存，但重算本身要再跑前向 GEMM/attention。保留 O/LSE 或 K/V 后，反向不必重跑这些前向；保留预算越大，重算越少，但缓存占用越高。768 MiB 的配置试图在 16 GiB 卡上取得平衡。

**实测**：

- 同一最终后端上，ckpt4 相对 ckpt2 **+6.80%** 吞吐，native live 峰值增加 **5.88%**。
- 在已经具备范围裁剪的 ckpt2 上，范围裁剪与 ckpt4 配方合计相对最初 ckpt2 未裁剪为 **+22.89%**。
- 但 ckpt4 也有边界：原生 `Engine.checkpoint` 在反向重放时仍完整调用 `fn(leaf)`，每微步会重放 32 次 down projection；2026-10-27 估算短/长桶分别约 **20.82/20.70 ms**。这是“已识别、尚未实现”的优化，不能算进当前 12,968.49 的收益。

### 3.5 Packed attention：跳过确定无效的跨文档块

原生 `--packing --packed_attention` 用文档边界 sidecar 构造 segment IDs：

- 前向/反向都隔离不同文档；
- 按文档重置 RoPE；
- 屏蔽跨文档预测和 padding；
- 对 causal packed 序列，forward/dQ 从当前 query tile 的最早文档起点开始，dKV 在当前 key tile 的最晚文档终点结束；跨文档边界 tile 和尾部 padding 仍逐元素判定。

**为什么能提高吞吐**：标准 causal mask 已经能跳过未来 token，但 packed 序列里还有大量“文档 A 的 query 看文档 B 的 key”的无效块；范围裁剪把这些块整块跳过，而不是算完再 mask。它不改变有效 token、模型参数或 mask 语义。

**实测**：

- 4×3072 前反向从约 **1.81 s** 降到 **1.28 s**。
- 长桶 cuDNN 路径在 2026-10-27 对照中为 **328.7 ms**，快于 PyTorch 的 **348.1 ms**；短桶 Triton 为 **415.5 ms**，慢于 PyTorch 的 **263.6 ms**。这说明“packed attention”不是一个统一的快路径：长桶 cuDNN 有优势，短桶 Triton 仍是当前主要短板。

**边界**：

- 默认 `attention_backend=triton`；`hybrid` 才在 `seq>=4096` 使用 cuDNN，且需要可选的 C++ SDPA bridge。
- cuDNN 在 SM120 的 packed backward 不保证确定性/逐位重复；验证按 BF16 容差进行。默认 Triton 路径保留原语义。
- 这部分是 attention 优化，**不是 cuBLAS**。把它算进“cuBLAS 的吞吐提升”会混淆因果。

### 3.6 异步分配、arena 生命周期与 CUDA Graph：减少同步和碎片

**异步分配**：

- `--allocator_backend async` 使用 CUDA 流有序的 `cuMemAllocAsync`/`cuMemFreeAsync`，空闲缓存默认上限 256 MiB。
- 旧 driver 分配器在两次前反向中记录到 **4,980 次 `cuMemFree_v2`，合计 5.45 秒**，还有 **6,543 次 malloc**；这些同步/驱动开销会直接进入 step 墙钟。
- 固定权重 probe 中，ckpt4/1792 MiB 从约 **1,158 targets/s** 提升到约 **6,619 targets/s**，live 峰值不变。注意这是前反向 probe，不是完整 optimizer-step 的倍数。

**arena 生命周期**：

- 旧录制路径把所有前向重算、反向临时梯度和优化器中间值保留到图销毁；新版允许单流命令带复用地址，并在 VJP 消费完节点后回收临时梯度。
- 共享参数的梯度在最后一个来源完成时立即归并进优化器持久梯度，避免持有各层全部 dW。
- 静态审计中，Dense B1 S4096 graph 的 arena 从 **10.798 → 2.183 GiB**（**-79.8%**），命令数 **1926 → 1321**；MoE B8 S256 graph 从 **12.696 → 3.765 GiB**（**-70.3%**）；FP8 B8 S256 graph 从 **12.532 → 3.869 GiB**。

**CUDA Graph**：

- `graph.py` 支持固定形状的完整 fwd/bwd/accumulation/norm/clip/overflow/AdamW 捕获；LR、累积计数、force-update 等控制留在 device，重放时不重建 Python autograd 图，也不分配图内 buffer。
- 但它只适用于单 GPU、原生 AdamW、dropout=0、固定深度、监督 CE；第一个 batch 形状固定，尾批或其他 bucket 走 eager；不能与 `--copy_prefetch` 同时使用；图内地址固定，可能比 eager 使用更多显存。
- **重要**：共享配方 `instinct_v2_native.json` 没有设置 `cuda_graph`；2026-10-27 的 12,968.49 当前生产验证命令也没有 `--cuda-graph`。因此 CUDA Graph 是“已实现的可选能力”，不能算作当前生产数字的已证来源。2026-10-27 的固定权重对照没有启用 `--micro-graph`；kernel 数 6645/7034 反映的是逐 kernel 提交路径。

**Host 提交与数据预取**：

- 每次 kernel 只设置一次 CUDA context；缓存 kernel 默认 constexpr；graph 重放把输入和 optimizer controls 的上传合并为一组栅栏，loss/statistics 下载共用一次栅栏。
- 预训练不再要求每个微步下载 loss 并全局同步；日志吞吐改为整个报告间隔的有效 targets / 墙钟时间，覆盖加载、更新和保存开销。
- `--copy_prefetch` 使用两个 pinned host/device 槽和独立 copy stream，通过 ready event 连接计算流。
- 生产 Parquet 由 Rust `dataset_compiler` 的 `plan-pretrain`/`pack-pretrain` 并行分词打包；Python 只做进程调度和 mmap 批次解码，每个 row group 是有界缓存单元，后台准备下一块，保留文档边界、有效目标数和下一未读游标。

**剩余开销**：2026-10-27 追踪中，原生短/长桶仍有约 6 千次 `cuLaunchKernel`、约 2 千次 `cuMemAllocAsync`、约 1.58 万次 `cuCtxSetCurrent`。这些是 API 计数，不代表全部在关键路径，但说明“减少提交”远未结束。

### 3.7 Muon：按形状批处理，减少优化器的小 GEMM 提交

原生 Muon 使用 split-weight：

- 隐藏矩阵存 BF16 高位 + 原始 uint16 低位；
- momentum/Nesterov 在 FP32 计算，动量可存 FP32 或 BF16；
- embedding、共享 head、norm 等非矩阵参数仍由 AdamW 以 FP32 处理；
- Newton–Schulz 使用原 quintic 系数和 5 次迭代，每次矩阵乘法输出 BF16，更新按独立 decay 舍入及 FMA 后重新拆位。

吞吐上的关键不是“Muon 算法本身”，而是：

- 按形状把多个矩阵参数合并成 pointer table，按 `muon_batch_size` 和 `muon_workspace_mb` 分块；
- 用 cuBLASLt 的 strided-batched GEMM 执行 Gram、polynomial、update，支持原来的 alpha/beta 更新表达式，共用现有 Lt workspace；
- 避免为每个小矩阵单独发起一次 GEMM/launch，提高 Tensor Core 利用率和提交效率。

**实测**（固定随机 BF16 输入，无训练更新）：

| 生产形状 / 运算 | Triton | cuBLASLt |
|---|---:|---:|
| 2×1536×1536×5504 / Gram | 0.8001 ms | **0.5462 ms** |
| 2×1536×1536×1536 / polynomial | 0.1591 ms | **0.1571 ms** |
| 2×1536×5504×1536 / update | 1.0580 ms | **0.5300 ms** |

这些是单独 GEMM 计时，不等于完整优化器或训练吞吐。原生栈用 16 个 CUDA 更新用例、9 组生产矩阵形状和 20 次实际训练更新验证了该路径。

**边界**：共享配方 `muon_gemm_backend=triton`；cuBLASLt Muon 是可选路径，2026-10-27 的当前生产验证用了 `--muon-gemm-backend cublas`。不能把 Muon GEMM 的单次加速直接等同于完整 step 的加速。

## 4. 实测数据

### 4.1 原生受控完整更新（2026-10-25）

条件：固定合成批次交替 4×3072 与 3×4096，每次更新 98,244 个有效 targets，accumulation 8，5 次实际更新（前 2 次预热，后 3 次计时），包含 fwd/bwd、clip、Muon/Adam 更新；排除 loader、冷编译、保存。

| 配置（均为原生） | 平均秒/update | 有效 targets/s | peak live GiB |
|---|---:|---:|---:|
| ckpt2，未做 packed 范围裁剪 | 16.061 | 6,116.9 | 12.052 |
| ckpt4，768 MiB，未裁剪 | 15.182 | 6,471.0 | 12.761 |
| 最终 ckpt2，范围裁剪 | 13.959 | 7,037.9 | 12.052 |
| 最终 ckpt4，768 MiB + 范围裁剪 | **13.070** | **7,516.8** | 12.761 |
| ckpt4，1792 MiB，未裁剪 | 21.319 | 4,608.2 | 13.659 |

1792 MiB 行出现过 33.448 秒慢步，CUDA pool reserved 高水位 15.1875 GiB；共享配方因此选择 768 MiB，而不是把更大缓存当作吞吐最优。

同机同 bucket 的 PyTorch ckpt4 为 **7.891 秒/update，12,449.5 targets/s，allocated 12.508 GiB，reserved 13.707 GiB**。原生最终 7,516.8 是它的 **60.38%**，这与“原生已超过 PyTorch”相反。

### 4.2 原生生产验证（2026-10-26 历史，2026-10-27 当前复测）

条件：真实 `pretrain_coder_l2l3_24b.parquet` 的已建缓存窗口；当前复测为 160 个微步、4 次预热 + 16 次计时更新，128 个微步用于吞吐统计。配置为 hybrid attention（短桶 Triton、长桶 cuDNN）、cuBLAS Muon、Lt tune、ckpt4 768 MiB + 按桶 1024/1280 MiB，`fused_residual_norm=false`。旧验证为 20 次实际更新、128 个测量微步，作为历史基线保留。

| 指标 | 当前复测（2026-10-27） | 旧验证（2026-10-26，历史） |
|---|---:|---:|
| 测量窗口 | 128 个微步（16 个报告窗口） | 128 个微步（20 次更新） |
| 有效 targets | **1,539,764** | 1,518,127 |
| 测量窗口墙钟 | **118.7312 s** | 133.1889 s |
| 训练窗口有效 targets/s | **12,968.49** | 11,398.30 |
| peak native live | **12.8074 GiB** | 12.8315 GiB |
| 训练结束（自启动） | 210.6652 s | 228.3608 s |
| 进程总时间（含最终保存/校验） | 261.9553 s | 273.7363 s |
| 相对旧验证 | **+13.8%** | — |

当前复测的 12,968.49 是生产窗口；相对旧 11,398.30 的 +13.8% 包含数据窗口、缓存状态和配置差异，不能全部归因于某个单独优化。当前进程总时间 261.9553 s：setup 结束于 11.8345 s，首个测量窗口在 57.4177 s 开始，训练在 210.6652 s 结束，最后约 51.2901 s 用于最终保存/校验/关闭。旧验证总时间 273.7363 s：setup 结束于 10.6132 s，首个窗口在 58.1293 s 开始，训练在 228.3608 s 结束，最后约 45.3755 s 用于最终保存/校验/关闭。旧验证从启动到训练结束的短作业吞吐为 1,903,117 / 228.3608 = 8,333.82 targets/s（不含最终校验/关闭）；含最终校验/关闭的总进程吞吐为 1,903,117 / 273.7363 = 6,952.3 targets/s。不能把 8,333.82 当作含最终保存的长期吞吐；该窗口也没有达到 14,000 或 18,000 targets/s。

### 4.3 同输入固定权重旧对照（2026-10-27，历史）

**历史记录：这里的 `identical_batches.npz` 输入已按用户要求移除；当前复测不得再使用该文件。** 本节只保留旧对照的结论，不代表 2026-10-28 的当前运行。

条件：两边读取同一个 `identical_batches.npz`，token IDs、labels、document IDs 完全一致；短桶 4×3072 有 12,274 个有效预测目标，长桶 3×4096 有 11,261 个。固定权重前后向，不执行更新；不计 loader、初始化、编译和保存。原生使用 hybrid、cuDNN 长桶、cuBLASLt tune；PyTorch 使用 ATEN/cuBLAS 和 Flex TMA backward；两者不是完全相同的执行策略 A/B。

| 项目（ms/微步） | Torch 4×3072 | 原生 4×3072 | Torch 3×4096 | 原生 3×4096 |
|---|---:|---:|---:|---:|
| GEMM | 330.3 | 338.7 | 340.9 | 348.6 |
| Attention 及 Q/K norm、RoPE | 263.6 | 415.5 | 348.1 | 328.7 |
| 其他 kernel | 195.7 | 260.9 | 192.2 | 263.4 |
| memcpy/memset | 0.04 | 0.04 | 0.05 | 2.62 |
| GPU activity 合计 | 789.6 | 1015.1 | 881.2 | 943.3 |
| activity 跨度内空档 | 9.7 | 51.6 | 14.6 | 95.8 |
| 无 profiler 前后向墙钟 | **796.3** | **1083.4** | **889.8** | **1041.0** |
| kernel 数量 | 4234 | 6645 | 4297 | 7034 |

结论：

- 原生 GEMM 只比 PyTorch 慢约 **8 ms**（短桶 338.7 vs 330.3；长桶 348.6 vs 340.9）；差距不在“cuBLAS 本身”，而在短桶 attention、其他逐元素 kernel、以及运行时提交空档。
- 长桶原生 attention（328.7）已经优于 PyTorch（348.1）；短桶原生 attention（415.5）明显落后（263.6），是当前第一优先项。
- 按生产桶比例（短 101、长 27）加权，attention 相关差额约 **116 ms**，其他 kernel 约 **66 ms**，trace 空档约 **50 ms**，GEMM 约 **8 ms**。
- 要达到 14,000 TPS，原生需要平均每微步节省 **193.37 ms**（时间降低 18.58%，吞吐提高 22.83%）。

### 4.3.1 当前原生固定权重复测（2026-10-27，未使用旧 npz）

条件：当前 worktree，`experiments.bench_native_recipe`，32 层，4×3072×3 与 3×4096×1，FP8，accumulation 8，4 次 warmup + 6 个测量窗口；`--production-samples`，`--native-sdpa`，ckpt4 1792 MiB，按桶 3072:1024/4096:1280。它没有读取旧 `identical_batches.npz`；旧 npz 也不参与本节结论。

| 配置 | 4×3072×3 墙钟（ms） | 3×4096×1 墙钟（ms） | 6 窗口合计 targets/s | 中位墙钟（ms） |
|---|---:|---:|---:|---:|
| 当前原生 + FRN | 1,243.1 / 1,243.9 / 1,244.1 | 1,065.6 / 1,064.8 / 1,066.3 | **10,191.62** | **1,154.70** |
| 当前原生 no FRN | 1,247.4 / 1,252.4 / 1,250.2 | 1,071.4 / 1,069.7 / 1,073.0 | 10,138.27 | 1,160.21 |
| 旧原生（npz，历史） | 约 1,085.3 | 约 1,036–1,059 | 11,078.29 | 1,063.79 |

当前原生 +FRN 比旧原生慢约 **8.0%**；短桶慢约 **14.6%**，长桶接近（约慢 0.6%–2.9%）。当前 peak live 为 **12.8315 GiB**。这个差异说明当前固定权重短桶路径相对旧记录变慢；但 `native_sdpa` 现在因 `.cache/native-libs/nvidia/cudnn` 缺失无法加载，当前固定权重数据是在缓存丢失前测得的，`current_20261028_fixed_fresh.json` 又是失败后残留的 `running`，不能用来替代。

当前 PyTorch 固定权重文件 `current_20261027_torch_fixed.json` 仍是 `status: "running"`，`bucket_ms` 和 `trace_files` 都为空；因此现在没有可用的当前 PyTorch 固定权重对手。表中旧 `796.3 / 1083.4 / 889.8 / 1041.0 ms` 仍只作历史对照。

### 4.4 主仓 PyTorch ATEN/cuBLAS（2026-10-26，历史）

条件：V2 Dense，BF16 参数，tensorwise FP8，auto filter；PyTorch 2.13.0+cu132；`max-autotune-no-cudagraphs`；固定权重与完整更新分别测量。

| 配置 | 4×3072 固定权重 | 3×4096 固定权重 | 完整更新 | 有效 token/s |
|---|---:|---:|---:|---:|
| 原选核 | 802.10 ms | 1082.35 ms | 7,888.04 ms | 12,454.81 |
| GEMM 限定 ATEN/cuBLAS | **710.83 ms** | **989.35 ms** | **6,822.66 ms** | **14,399.67** |

两组峰值 allocated 都是 **12.507867 GiB**，没有用增加显存换吞吐。

拆解：

- 固定权重 4×3072 降低 **91.27 ms**（-11.38%），3×4096 降低 **93.00 ms**（-8.59%）。
- 下投影输入梯度 GEMM 从 Triton 的约 **2.06 ms/层** 降到 cuBLAS 的约 **0.67 ms/层**。
- 长桶 attention backward 从约 **10.67 ms/层** 降到 **8.21 ms/层**；独立 attention 前反向从 **13.32 ms** 降到 **11.38 ms**；但前向从 **2.66 ms** 退化到 **3.21 ms**，因此 TMA 只在 3×4096 的 backward 启用。
- 完整更新的 **+15.62%** 同时包含 ATEN/cuBLAS GEMM 与长桶反向 TMA，不应全部记在 cuBLAS 头上。

**这是 PyTorch 路径的数字，不是原生栈的数字。** 主仓报告明确说明它不是之前的 Torch-free native cublas worktree 吞吐。

### 4.5 主仓 PyTorch 当前复测（2026-10-28，带 GPU 干扰）

条件：`D:/AI/Instinct`，HEAD `22c9d91`；合成累积更新（不含 loader/save），4×3072×3 与 3×4096×1，accumulation 8；3 次 warmup + 5 个测量 update。baseline 设置 `INSTINCT_V2_GEMM_BACKEND=default`、`INSTINCT_V2_FLEX_TMA=0`；optimized 设置 `auto/auto`。两次都使用共享 `D:/AI/Instinct/.cache/torch_compile`，`verified_current_state=true` 并保存 resume。运行前后 GPU 上仍有 `RimWorld.exe`/`QQ.exe`（约 848 MiB、15–17% 利用率）；fresh-cache 探针因 ptxas 路径错误和 CUDA OOM 未完成，因此这不是干净 A/B。

| 指标 | baseline | optimized |
|---|---:|---:|
| 5 个测量 update 总时间 | 44,441.53 ms | 44,233.86 ms |
| 平均 update | 8,888.31 ms | 8,846.77 ms |
| 中位 update | 8,885.12 ms | 8,693.47 ms |
| **聚合 targets/s** | **11,053.17** | **11,105.07** |
| 每窗口中位 targets/s | 11,057.14 | 11,300.90 |
| allocated / reserved | 12.57818 / 13.70703 GiB | 12.57726 / 13.75391 GiB |
| 优化器均值 | 423.44 ms | 471.59 ms |
| 保存的 resume | `checkpoints/ckpt4_recipe_20261028_000924_28724_1536_resume.pth` | `checkpoints/ckpt4_recipe_20261028_001336_50976_1536_resume.pth` |

按总测量时间折算的聚合吞吐，optimized 只比 baseline 高 **+0.47%**（总时间 44,441.53 → 44,233.86 ms）；只有把每个窗口的 tps 取中位数才得到 **+2.20%**（11,057.14 → 11,300.90）。后者被 optimized 的 `9,243.68 ms` / `9,045.42 ms` 慢窗口影响，不是总时间口径，不能当作聚合增益。相对旧记录，当前 baseline 低约 **11.25%**，当前 optimized 低约 **22.88%**；这不能直接归因于代码，因为两次复测都带后台 GPU 负载和共享编译缓存。

当前 optimized 日志有 `[Compile GEMMs] V2 tensorwise FP8: ATEN/cuBLAS`，但没有旧 optimized 日志中的 TMA autotune 输出；共享缓存可能直接复用了旧 kernel，fresh-cache 复测又未完成。因此 **这组 +0.47% 不能证明当前 ATEN/cuBLAS 或 TMA 的真实收益**，也不能替代旧 `+15.62%` 的历史记录。

## 5. 差距在哪里，为什么原生还没有赢

### 5.1 差距不是“GEMM 库”

以下为 **2026-10-27 的历史分解**，不是 2026-10-28 的当前复测：原生 GEMM 比 PyTorch 慢约 8 ms，而总差距约 151–287 ms（4×3072 1083.4 vs 796.3；3×4096 1041.0 vs 889.8）。当前没有可用的 PyTorch 固定权重对手（`current_20261027_torch_fixed.json` 仍是 `running`），所以不能把这段差额当成当前结论。主要差距来自：

1. **短桶 attention**：原生 415.5 ms vs PyTorch 263.6 ms。前向（含重算）178.0 ms、反向核心/统计 183.7 ms、独立 Q/K norm/RoPE 53.8 ms；每线程 255 个寄存器，反向 Q/KV 为 252/254，资源压力明显，不能只靠增大 tile 解决。
2. **其他逐元素/量化 kernel**：原生 260.9/263.4 ms vs PyTorch 195.7/192.2 ms。原生已经融合了 amax，但还没有达到 Torch 的 FFN 五路输出融合范围。
3. **运行时提交空档**：原生 51.6/95.8 ms vs PyTorch 9.7/14.6 ms；约 6 千次 launch、2 千次 async alloc、1.58 万次 `cuCtxSetCurrent`。
4. **无效重算**：每微步 32 次 down projection 重放，约 20.8/20.7 ms。
5. **TMA 缺口**：原生启动 ABI 还不支持完整 TensorDescriptor/TMA，短桶 attention 的 Q/K/V 布局和载入仍有优化空间。
6. **GEMM 选型**：原生与 PyTorch 的 GEMM 已经接近，继续只在 cuBLASLt 候选算法上抠，收益有限；2026-10-27 报告明确说“两个版本的主要 GEMM 本来就都走 cuBLAS，直接使用 cuBLAS 本身不能证明 18k TPS 上限”。

### 5.2 优化优先级

按当前证据，优先级应是：

0. **先修复复测环境**：关闭后台 GPU 负载、使用独立 `TORCHINDUCTOR_CACHE_DIR`、设置可用的 `TRITON_PTXAS_PATH`，重跑当前 PyTorch A/B。当前 `+0.47%` 不能作为调优依据。
1. **专用 Dense replay + FFN 数据流融合**：去掉终端 down projection 重放，把 SwiGLU、amax、FP8 双布局、gate/up 梯度融合成更少的数据流。
2. **短桶 attention**：优化 packed block 调度、Q/K/V 布局与载入、反向分块、Q/K norm/RoPE 融合；保持长桶 cuDNN 的现有优势。
3. **提交与内存复用**：减少重复 context 设置、复用固定桶短期内存、批量提交或按 block 捕获稳定子图；同时看无 profiler 墙钟和峰值显存，避免恢复此前造成显存压力的无限持久缓冲。
4. 最后再评估 Muon 和剩余 GEMM 选型；单 GEMM 加速不等于完整更新加速。

## 6. 常见误解与边界

### 6.1 “用了 cuBLAS 就一定更快”？

不是。更早的 198M MoE / RTX 5060 Ti 基准里，原生 BF16 约 **2,320 tok/s**，PyTorch `reduce-overhead` 约 **10,300 tok/s**，原生慢约 4.5 倍；原生 FP8 约 **2,056 tok/s**，更慢。那套配置的瓶颈是 Python/逐 kernel 调度和布局，而不是 GEMM 库本身。该结果不能直接外推到当前 1.01B V2，但能说明“直接调用 cuBLASLt”不是充分条件。

### 6.2 “原生 12,968 已经接近 14k 了吗？”

当前原生生产是 **12,968.49** targets/s；要达到 14,000 还差约 **7.95%** 吞吐（时间约降低 7.37%）。旧 `193.37 ms` 预算属于 11,398.30 的历史口径，不能直接套到当前结果。当前原生比旧 PyTorch 生产窗口 14,150.9 TPS 低约 **8.36%**，比旧主仓 14,399.67 低约 **9.94%**；这仍不足以得出“已经达到 14k”的结论。

### 6.3 “CUDA Graph 已经在生产里”？

没有。`graph.py` 实现了完整 step 图，但共享配方没有 `cuda_graph` 字段，2026-10-27 的 12,968.49 当前生产验证命令也没有 `--cuda-graph`。CUDA Graph 是可选能力，适用于固定形状、单 GPU、原生 AdamW、dropout=0 的监督训练，不能与 `--copy_prefetch` 同用。

### 6.4 “cuDNN 长桶收益是 cuBLAS 的收益”？

不是。cuDNN 是 attention SDPA，不是 cuBLAS GEMM。2026-10-27 的历史固定权重分解里，长桶 328.7 vs 348.1 的收益属于 attention 路径；把它算进 cuBLAS 会混淆因果。并且 SM120 的 cuDNN packed backward 不保证确定性，默认仍用 Triton。

### 6.5 “静态审计的命令数下降就是 tok/s 提升”？

不是。命令数、arena 字节、copy 流量是结构性指标；它们能证明“减少了提交/搬运/重算”，但不能直接换算成吞吐。例如 Dense B1 S4096 的 Lt GEMM 次数 227 和算量 2.054 TFLOP 都没变，说明收益来自调度与流量，而不是少算了数学。真实吞吐必须看端到端墙钟。

### 6.6 “原生和 PyTorch 的 loss 可以直接比”？

不能。两套后端的 RNG 初始化、FP8 舍入、GEMM 归约顺序不同；2026-10-27 明确不用固定权重对照判断数值一致性。cuDNN 路径也不是 bit-exact。质量比较需要独立的长期训练/验证集曲线。

### 6.7 “14.4k 是原生 cuBLAS 的成绩”？

不是。14,399.67 是主仓 PyTorch + ATEN/cuBLAS + 长桶 TMA 的历史完整更新成绩。原生栈当前生产验证是 **12,968.49**（旧验证 11,398.30）。两者不是同一个程序，不能混写。

## 7. 一句话回答

原生 cuBLAS 程序提高吞吐的方式，是**围绕固定形状训练把 GEMM 计划、转置布局、FP8 量化、算子融合、选择性重算、异步分配、packed attention 边界、Muon 批处理和数据预取协同起来**，从而减少 CPU 提交、显存搬运、重复前向和同步等待；它把受控更新的原生栈从 6,116.9 提升到 7,516.8 targets/s，把当前生产验证推到 **12,968.49** targets/s，但相对旧 PyTorch 14,399.67 仍低约 **9.94%**。主仓旧记录达到 14,399.67 targets/s 的关键，是在 PyTorch 路径中把 V2 FP8 GEMM 限定到 ATEN/cuBLAS，并配合长桶反向 TMA；这不是单纯“换 cuBLAS”的结果。当前 PyTorch 复测受 GPU 后台负载和共享 compile cache 干扰，聚合只高 **+0.47%**，不能替代旧记录，也不能证明原生已经超过 PyTorch。

## 8. 复现与资料

### 8.1 原生

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'

# 生产窗口（已建缓存；会创建独立诊断权重并在更新边界保存）
python -u -m experiments.bench_native_production `
  --data D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet `
  --output experiments/results/cublas/current_20261027_production.json `
  --warmup-updates 4 --updates 16 --tune `
  --attention-backend hybrid --muon-gemm-backend cublas `
  --selective-cache-by-seq 3072:1024,4096:1280

# 固定权重前反向（当前使用 production samples；不再导出/读取 npz）
python -m experiments.bench_native_recipe --production-samples --native-sdpa `
  --output experiments/results/cublas/current_20261027_fixed.json
```

原生资料（worktree 内）：

- `docs/native_v2_recipe.md`：原生 V2 配方、ckpt4、Muon、packed attention、FP8、allocator 的完整说明。
- `docs/native_throughput_optimization.md`：第一轮命令/分配/流量审计。
- `docs/dense_schedule_optimization.md`：Dense 调度、融合和 arena 的第二轮审计。
- `docs/reports/2026-10-25-native-v2-throughput.md`：受控 5 次更新表、ckpt2/ckpt4、PyTorch 对照。
- `docs/reports/2026-10-26-native-validation.md`：历史 11,398.30 生产 Parquet 验证。
- `docs/reports/2026-10-27-native-vs-torch.md`：历史同输入固定权重分解、193.37 ms 预算、剩余优化项。
- `experiments/results/cublas/validation_20261026/production_hybrid_cublas.json`：历史生产验证记录。
- `experiments/results/cublas/comparison_20261027/comparison.json`：历史同输入对照汇总。
- `experiments/results/cublas/current_20261027_production.json`：当前生产 **12,968.49**，1,539,764 targets / 118.7312 s，`verified_current_state=true`。
- `experiments/results/cublas/current_20261027_production.log`：当前生产命令和验证日志。
- `experiments/results/cublas/current_20261027_fixed.json`：当前固定权重 +FRN，6 窗口合计 10,191.62 targets/s。
- `experiments/results/cublas/current_20261027_fixed_no_frn.json`：当前固定权重无 FRN，6 窗口合计 10,138.27 targets/s。
- `experiments/results/cublas/current_20261027_torch_fixed.json`：当前 PyTorch 固定权重文件，仍是 `running` 且 bucket 为空，不能用。
- `experiments/results/cublas/current_20261028_fixed_fresh.json`：`native_sdpa` 导出失败后的残留，不能用。

### 8.2 主仓 PyTorch

- `model/compile_policy.py`：ATEN/cuBLAS GEMM 选择的条件与实现。
- `docs/reports/2026-10-26-v2-throughput.md`：历史 14,399.67 token/s、固定权重表、TMA backward 拆解。
- `experiments/profile_torch_reference.py`：PyTorch 固定权重/剖面入口。
- `experiments/profile_checkpoint_recipe.py`：2026-10-28 当前 PyTorch 完整更新 A/B 入口。
- `experiments/results/cublas/current_20261028_pytorch/baseline.json`：当前 baseline，聚合 11,053.17 targets/s。
- `experiments/results/cublas/current_20261028_pytorch/optimized.json`：当前 optimized，聚合 11,105.07 targets/s。
- `experiments/results/cublas/current_20261028_pytorch/baseline.log` / `optimized.log`：当前 A/B 日志；optimized 有 ATEN/cuBLAS 编译行，但缺少旧日志中的 TMA autotune 输出。
- `experiments/results/cublas/current_20261028_pytorch/optimized_freshcache_probe.log`：fresh-cache 复测失败记录（ptxas/OOM）。

### 8.3 代码地图

| 文件 | 作用 |
|---|---|
| `model/dense_cublas/native/bridge.cpp` | cuBLASLt C ABI、plan、descriptor、候选算法、workspace |
| `model/dense_cublas/backend.py` | `ScheduleConfig`、每流 workspace、`ExecutionPlan`、graph capture/replay |
| `model/dense_cublas/layout.py` | host 侧布局/转置/对齐规划，避免物理 pack |
| `model/dense_cublas/ops.py` | `CublasDenseOps`、Linear、attention 调度 |
| `model/dense_cublas/fused.py` | residual+norm、Q/K norm+RoPE、projection cast |
| `model/dense_cublas/ffn.py` | SwiGLU + amax + FP8 量化、FFN 反向融合 |
| `model/dense_cublas/tensor_kernels.py` | CE、梯度求和、范数 partials、softmax 等 Triton 基础 kernel |
| `model/dense_cublas/selective.py` | ckpt4 候选输出缓存计划 |
| `model/dense_cublas/graph.py` | 固定形状完整 step CUDA Graph |
| `model/dense_cublas/memory.py` | arena、allocation registry、async allocator |
| `model/dense_cublas/muon.py` | split-weight Muon、按形状批处理、NS GEMM |
| `trainer/recipes/instinct_v2_native.json` | 共享原生配方（768 MiB、triton attention/Muon、无 cuda_graph） |
| `model/compile_policy.py` | 主仓 PyTorch 的 ATEN/cuBLAS GEMM 策略 |

## 9. 限制

- 本文的性能数据来自单机 RTX 5070 Ti 16 GB / SM120；不能外推到其他 GPU、其他 batch/sequence 组合或其他精度设置。
- 2026-10-25 的 5 次更新、2026-10-26 的 20 次更新、2026-10-27 的当前生产/固定权重、2026-10-28 的 PyTorch 复测是不同窗口，不能拼成一个“端到端长期吞吐”。
- 2026-10-26 和 2026-10-27 的生产验证使用的是诊断权重和抽样游标，不等于完整语料已经训练到该位置；也不建立长期收敛结论。
- 2026-10-28 的 PyTorch A/B 运行时有 `RimWorld.exe`/`QQ.exe`（约 848 MiB、15–17% 利用率），并复用了共享 compile cache；fresh-cache 探针因 ptxas/OOM 失败。聚合 `+0.47%` 不是干净 A/B，不能证明当前 ATEN/cuBLAS 或 TMA 的真实收益。
- 当前原生固定权重 `1,154.70 ms` 是在 `native_sdpa` 缓存丢失前测得的；现在 `native_sdpa` 因缺少 cudnn 无法加载，当前 PyTorch 固定权重文件仍是 `running` 且 bucket 为空。旧 `1083.4 / 796.3 / 889.8 / 1041.0 ms` 只能作历史对照。
- 旧 `identical_batches.npz` 已按用户要求移除；当前原生固定权重使用 production samples，本文不再用该 npz 生成新结论。
- 原生与 PyTorch 的 RNG、FP8 舍入和 GEMM 归约不同；本文不比较 loss 高低，也不声称数值等价。
- 静态审计的命令数、arena 字节和 copy 流量只证明结构变化，不直接证明 tok/s；所有吞吐结论都以完成的 GPU 窗口为准。
