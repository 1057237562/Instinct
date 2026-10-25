# 原生 cuBLASLt/Triton 吞吐与显存优化

基线：`1ed4740d1710d068470486639db80582a92e8365`。本轮按要求不运行实卡测试。以下数值来自真实模型、VJP、优化器和图构建代码的**形状/分配/命令审计**，不是 tok/s、CUDA 时序模拟或实测显存。原始输出见 [native_optimization_audit.json](native_optimization_audit.json)，审计器见 [profile_native_schedule.py](../experiments/profile_native_schedule.py)。

## 对照结果

共同配置：H=768、8 层、Q/KV heads=8/4、head_dim=96、FFN=2432、词表 6400；FP32 参数和 AdamW 状态、BF16 激活、原生 `use_grad_checkpoint=1`、默认单流、不拼接 QKV、不分块 LM loss。MoE 为 4 专家 top-1。4096 序列明确使用 **batch=1**；旧实测报告的配置为 batch=8、seq=256。这里的 checkpoint 数值是原生后端原有 0/1/2 语义，不是 PyTorch SAC 的四个档位。

| 模型与形状 | 执行方式 | 旧 arena 峰值 GiB | 新 arena 峰值 GiB | 降幅 | 旧→新命令数 |
|---|---|---:|---:|---:|---:|
| MoE B8 S256 | eager | 6.990 | 3.702 | 47.0% | 4195 → 2431 |
| MoE B8 S256 | graph | 12.696 | 3.765 | 70.3% | 3694 → 2441 |
| MoE B1 S4096 | eager | 7.773 | 3.989 | 48.7% | 4195 → 2431 |
| MoE B1 S4096 | graph | 17.381 | 4.294 | 75.3% | 3694 → 2441 |
| Dense B1 S4096 | eager | 3.760 | 1.902 | 49.4% | 2187 → 1311 |
| Dense B1 S4096 | graph | 10.798 | 2.183 | 79.8% | 1926 → 1321 |

补充 FP8 矩阵路径的 B8 S256 图审计：arena **12.532 → 3.869 GiB**，命令数 **4612 → 3698**。FP8 仍有量化与 TN 打包成本，不能仅凭乘法精度更低就认为整步更快；原始数据同时保留 eager 与 graph 结果。

口径限制：

- arena 是每次分配/回收后，仍由分配器持有的总字节数峰值，包含池缓存。还须另加默认 **32 MiB cuBLASLt workspace**、CUDA 上下文/模块、驱动开销及实际分配粒度；不是 `nvidia-smi` 读数，也不能直接等同 PyTorch allocated。
- eager 审计一次前反向加一次优化器更新；graph 审计完整固定命令带，含按累积状态条件执行的更新节点。两种模式内部做新旧对照，不将二者命令数当成等量执行时间。
- 命令数是 Triton launch / Lt matmul 的 Python 提交次数；Lt 内部可能使用多个 CUDA kernel。CTA 数包含固定容量 MoE 网格中条件退出的空 tile。
- `host_synchronizations` 记录审计后端执行到的模型/回收/优化器边界；不含真实 HtoD、首轮 cuBLAS 调优、JIT 和 trainer 计时栅栏。B8 S256 eager 此项 **204 → 1**，不是承诺整个训练进程只有一次同步。
- 没有减少参数、训练 token 或专家数量。GEMM 主体算量基本不变：B8 S256 的矩阵乘法统计约 **1.028 → 1.027 TFLOP**，微小差值来自取消高精度 padding。统计不含注意力内部 dot、归约和逐元素运算。

## 已落实的优化

### 1. 回收生命周期，而不是在图中保留所有临时张量

旧录制路径在 `release_since()` 直接返回，所有前向重算、反向临时梯度、优化器中间值都存活到图销毁。新版允许单流命令带复用相同地址，并在构建结束时固定所有被命令引用的地址，防止后续 eager fallback 借用它们。

VJP 消费完节点便回收其临时梯度；共享参数按最后一个梯度来源完成的时刻归并，立即累加进优化器的持久梯度，避免持有各层全部 dW。显式请求的 VJP 输出仍保留。多流重叠路径保留保守的同步与分配规则，以上显存收益针对默认单流。

真实 CPU 字节存储复用测试发现并修复了两个边界：图准备时的 host 常量必须使用独立且固定的地址；普通 HtoD 上传也不能直接覆盖可能仍被非阻塞计算流读取的池内存。固定图输入继续通过有明确栅栏的批量上传更新。

### 2. cuBLASLt 高精度 GEMM 不再搬运转置矩阵

BF16/FP16/FP32 使用原始连续存储和 transpose 描述符，直接计算 `Cᵀ = op(B)ᵀ @ op(A)ᵀ`。四种转置组合均覆盖，移除 dX/dW 的物理 pack、padding 和 crop。B8 S256 每次审计消除约 **515 MiB** 的 GEMM 显式读写。

C++ bridge 升至 ABI 2；Python 仍支持 ABI 1，并自动回退旧 pack 路径。SM120 FP8 保留对齐后的 TN 路径，没有放宽精度或开启 fast accumulation。该限制及矩阵描述符依据 [NVIDIA cuBLAS 12.8 文档](https://docs.nvidia.com/cuda/archive/12.8.2/cublas/index.html#cublasltmatmul)。

### 3. 梯度范数和优化器消除整模型中间副本

梯度直接在 kernel 中先反缩放再平方，每 1024 元素输出一个 FP32 部分和，再分层归约。避免先平方溢出；避免最终单 CTA 串行扫描约 20 万部分和。临时空间为 O(N/1024)，原 eager 的表达式会生成三个 O(N) 张量。

按张量读写计数、忽略缓存，旧 eager 范数约 32 字节/参数，新路径约 4 字节/参数；这只是该子过程的流量模型。AdamW 将清零梯度融合进更新；图中所有参数 step 计数通过一个 kernel 更新。标量常量不再各自分配和 fill；归约反向直接广播，不先生成整张量的 1；常量父节点不再计算无用梯度。

### 4. MoE 权重和 grouped GEMM

非 FP8 专家直接拼接到 BF16/FP16 计算精度，避免完整 FP32 拼接副本后再 cast。dW 仍按原参数精度累加。SiLU 与 gate/up 乘法融合。

大尺寸低精度 grouped forward/dX tile 从 16×32 改为 64×64，dW 输出 tile 从 16×32 改为 32×64；小矩阵和 FP32 保留原路径。B8 S256 审计中，forward/dX 的固定网格 CTA 总数 **1,949,696 → 243,712**，dW 为 **350,208 → 87,552**。

仅计算 tile 的两路输入读取、忽略输出和缓存，BF16 算术强度 `BM*BN/(BM+BN)` 从约 10.7 提升至 32 FLOP/byte，dW 从约 10.7 提升至 21.3。并不等价于 3×/2×实测加速；小专家负载、寄存器占用和尾部利用率仍会影响结果。

FP8 专家 amax 改成并行分块归约。2432×768 权重原来每专家只有一个 CTA、串行 7296 个 256 元素块；现在每专家 1824 个 1024 元素部分归约 CTA，最终 8 轮归约。增加少量标量 scratch，消除大权重扫描的串行长尾。

### 5. CE 与长序列注意力

普通 CE 融合行级 logsumexp、目标项和有效标签计数；反向重算 softmax，不保留整张 FP32 log-softmax。保持 ignore label 和全 ignore 时零 loss/梯度语义。

注意力反向预计算每行 `delta = sum(dO * O)`。dQ 与 dKV 共用小型 FP32 delta；dKV 不再每个 KV tile 重读整行 O 并重复归约。S4096、tile32 时，每行可能被访问上百次，新增的一次线性预处理减少这部分重复工作；不改变 causal、mask、GQA 和 dropout 索引。

### 6. Host 提交路径

每次 kernel 只设置一次 CUDA context，所有 tensor 仍逐个检查归属；缓存 kernel 默认 constexpr。图重放把输入及 optimizer controls 的上传合并为一组栅栏，loss/statistics 下载共用一次栅栏，trainer 不再重复等待已下载完成的图。日志公开 capture/replay/command 计数。

原报告“原生 MoE 每步重捕”的说法与源码不符；原生 dispatch 已为固定容量。旧图 arena 超过卡容量，是 WDDM 溢出退化的合理假设，仍需后续实卡 profiling 验证，不能把假设写成实测因果。

## 验证及使用

- 原生、Dense kernel、checkpoint/resume、分布式和 trainer 合同测试：159 passed，10 skipped。跳过项需要真实 CUDA 或解释器不支持的 FP8/BF16 舍入。
- 新增真实 CPU 内存复用专项最终复测 27 passed，覆盖共享梯度、三种原生 checkpoint 模式、累积更新、图输入/路由变化、host 常量、图外临时分配、范数和 CE；与综合测试有重叠，不将两者相加。Torch 只作为测试 oracle；生产代码不导入 Torch。
- SM120 离线编译：训练 primitives 66 个特化，grouped/图优化器 78 个，注意力 84 个；最高 shared memory 73,984 字节，均无不支持的 global scratch。BF16 grouped dot 的数值性能不能由 Triton 3.4 CPU interpreter证明；FP16 tile 数值有 oracle，BF16/FP8 路径有离线编译。
- bridge 使用真实 CUDA/cuBLAS 头文件完成 C++17 `-Wall -Wextra -Werror -fsyntax-only` 检查；没有在本环境链接或运行 cuBLASLt。

启用免搬运 GEMM 需重新构建 bridge：

```bash
cmake -S model/dense_cublas/native -B model/dense_cublas/native/build
cmake --build model/dense_cublas/native/build --config Release
```

复查不需要 GPU：

```bash
python -m experiments.profile_native_schedule --mode graph --batch 1 --seq 4096
python -m experiments.profile_native_schedule --mode graph --batch 1 --seq 4096 --dense
python -m experiments.compile_cublas_training --arch 120
python -m experiments.compile_native_remaining --arch 120
python -m experiments.compile_native_attention --arch 120
TRITON_INTERPRET=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/test_native_optimization.py -q
```

要复现旧版本对照，将基线放到独立 checkout，再从新版脚本传入 `--repo /path/to/baseline`。不要使用旧版脚本和新版脚本各跑一边。

## 能得出什么结论

本轮主要压缩存储、搬运、同步及重复归约，并改善 grouped tile 的复用率。训练 GEMM 主体算量仍由模型决定；纯命令数下降或 tile 算术强度上升不能推导真实训练吞吐倍数。

与 PyTorch 同精度、同语义地同时超过其速度和显存仍是验收目标，**尚未被实测证明**。剩余硬件相关选择包括 Lt 候选算法、grouped tile/流水级数、融合投影、FP8 收益以及 dW 多流重叠；不凭静态数值将这些开关全部打开。当前默认改动先消除了可明确识别的结构性浪费，再保留这些待实卡确定的选择。
