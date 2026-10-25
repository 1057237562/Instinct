# Dense 提交次数、搬运与吞吐优化

基线为上一轮优化后的 `e4eb938347b97a111d36456571e2a6af8e8cee11`。本轮继续优化原生 cuBLASLt/Triton Dense，不进行实卡验证。以下结果来自运行真实 Model/Engine/Optimizer/StepGraph 的**静态命令与分配审计**，不代表实测 tok/s 或 `nvidia-smi` 显存。原始对照见 [dense_schedule_audit.json](dense_schedule_audit.json)。

## 结果与口径

共同配置：H=768、8 层、FFN=2432、Q/KV heads=8/4、head_dim=96、词表 6400、共享 embedding/head；FP32 主参数及 AdamW 状态，BF16 激活，dropout=0，默认单流。使用默认独立投影 GEMM、不分块 LM loss。这里的 ckpt0/1/2 是原生后端原有语义，不是 PyTorch SAC 四个档位。

核心配置 **batch=1、seq=4096、ckpt1**：

| 指标 | 基线 | 本轮 | 降幅 |
|---|---:|---:|---:|
| eager Triton/Lt 提交数 | 1311 | 748 | 42.9% |
| graph 重放命令数 | 1321 | 717 | 45.7% |
| eager arena 峰值 | 1.902 GiB | 1.666 GiB | 12.4% |
| graph arena 峰值 | 2.183 GiB | 1.765 GiB | 19.1% |
| 统计到的复制/转换读写量 | 6.446 GiB | 2.860 GiB | 55.6% |
| Lt GEMM 次数 | 227 | 227 | 0% |
| Lt GEMM 算量 | 2.054 TFLOP | 2.054 TFLOP | 0% |

限定条件：

- eager 是一次前反向和一次更新；graph 是固定命令带，包含根据累积状态条件执行的更新节点。graph 原本就通过 CUDA Graph 重放，717 是图内 Triton/Lt 命令数，不是 717 次 host graph launch。
- cuBLASLt 一次 matmul 可能执行多个设备 kernel。审计不计库内部实现，也不计 HtoD/DtoH API 次数。
- graph 准备阶段另有一次初始化范数 scratch 的 fill，总计 718 次；该初始化不进入重放。审计器现在分别报告 `dispatches` 与 `preparation_dispatches`。
- arena 包含分配器持有的缓存，但不含默认 32 MiB Lt workspace、CUDA context/module、驱动及实际分配粒度。它不等于整进程显存，也不能直接与 PyTorch allocated 比较。
- 复制读写量只覆盖显式 cast、projection_cast、strided_copy 及 GEMM pack/crop，按元素读写字节相加。它不是全部显存流量，也不模拟缓存。GEMM 算量不包含 attention 内部 dot、归约和逐元素计算。

其他形状与检查点的同版本对照：

| 配置 | 执行方式 | 命令数：旧 → 新 | arena GiB：旧 → 新 |
|---|---|---:|---:|
| B1 S4096 ckpt0 | eager | 1103 → 612 | 2.929 → 2.425 |
| B1 S4096 ckpt0 | graph | 1113 → 581 | 3.153 → 2.503 |
| B1 S4096 ckpt2 | eager | 1447 → 764 | 1.836 → 1.531 |
| B1 S4096 ckpt2 | graph | 1457 → 733 | 2.113 → 1.666 |
| B8 S256 ckpt1 | eager | 1311 → 748 | 1.558 → 1.424 |
| B8 S256 ckpt1 | graph | 1321 → 717 | 1.632 → 1.424 |
| B1 S4096 FP8 ckpt1 | eager | 2480 → 1965 | 1.913 → 1.646 |
| B1 S4096 FP8 ckpt1 | graph | 2490 → 1934 | 2.273 → 1.856 |

## 已实现的优化

### 1. 根据反向算子需要的精度传递梯度

原 VJP 在每个节点把梯度提升到 FP32，而 linear、RMSNorm、attention 等反向算子紧接着又转换到激活精度。现在节点声明其反向输入精度，单一梯度来源可直接传递；reshape/cast 这类节点保留传入精度。

多来源梯度仍先在 FP32 累加，再进行一次目标精度转换。叶子参数梯度及显式请求的 VJP 输出仍是 FP32。共享 embedding/head、残差分支与 checkpoint 的累加语义保留。

Q/K/V 和 gate/up 的 FP32 主权重转换为计算精度时，使用一次批量 cast；GEMM 仍各自执行，不构造拼接的 FP32 权重或切片输出。主配置的转换命令由 **334 次 → 132 次普通 cast + 32 次 projection_cast**。这些转换没有被替换成持续占用显存的全模型 BF16 权重缓存。

### 2. 融合 Q/K RMSNorm + RoPE 与默认 Dense SwiGLU

Q/K 的 norm 和 rotary 前向合并，反向合并 RoPE 转置和 RMSNorm dX/dW 部分和，保留小型 dW 归约。实现显式保留原 RMSNorm 输出和反向 dY 的低精度舍入点；非重复 cos/sin 表也按真正的转置计算。

默认 Dense 的 SiLU 和 gate/up 乘法及其反向融合。旧默认路径先把 SiLU 结果舍入到 BF16/FP16，再相乘；新默认融合保留该顺序。已有 `fuse_projections=True` 路径继续保留自己的原有计算语义。对显式请求的 up 梯度保留 FP32，避免额外的提前舍入。

### 3. 去掉 Dense checkpoint 的空辅助损失打包

Dense 没有 MoE router aux loss，旧共用路径仍把一个零拼接进激活，再切片取回，并在层间累加。现在 Dense checkpoint 直接返回激活，复用持久零辅助损失。ckpt0/1/2 的重算边界保持不变。

语言模型 CE 直接寻址原 logits 中的预测位置和后移一位的标签；反向直接写完整梯度，并将每个序列最后位置置零。无需复制 `logits[:, :-1]` 或再 scatter 梯度。ignore label、全 ignore、early exit、显式 chunked loss 路径保持支持。主配置 eager 的 **83 次 strided_copy → 0 次**。

### 4. 一次提交全模型 AdamW 和范数部分和

同精度原生 AdamW 用持久指针表描述 parameter/gradient/m/v，按每 1024 元素映射到对应参数；无需拼接或复制参数。主配置 **90 次 AdamW → 1 次**，梯度清零仍融合在更新内。

eager 的范数部分和从 **90 次 → 1 次**，保留两级归约。它在 step 时重新读取当前梯度，因此仍能正确处理分布式同步或外部累加后的梯度。

混合参数精度、未使用参数、导入的参数组选项/调度器或 eager 不等 step 计数保留原单张量路径；Adafactor/Muon 不使用该批量路径。指针表引用既有持久参数与状态，checkpoint 恢复原地写入这些 buffer，避免恢复后引用旧地址。

### 5. 图内梯度累积同时计算范数，小参数集中提交

图内每个大参数在最后一个梯度来源结束后，融合梯度累积和范数部分和，省去随后扫描全部 FP32 梯度的一遍读取。小于等于 1024 元素的参数梯度暂存到反向末尾，通过一个 kernel 批量累积和计算范数。

主配置图中 **90 次累积 + 90 次范数部分和 → 57 次大参数融合 + 1 次小参数批量融合**，末尾两级归约保留。unused 参数的范数槽保持零，溢出时全局跳过更新，累积步数、loss scaling、force update 及 per-parameter step 行为保持一致。

## 对吞吐的含义与剩余成本

本轮减少提交开销、转换和中间结果搬运；矩阵乘法主体算量没有改变。假设每条提交成本完全相同，仅提交这一项的成本比例约为 eager **0.571**、graph 命令部分 **0.543**。这个假设不描述 cuBLAS 算法、图调度、并行重叠或实际内存带宽，不能推出整体 1.75×/1.84× 加速，更不能声称已经快于 PyTorch。

后续需要硬件证据决定的重点：

1. **Lt GEMM 算法与独立/融合投影取舍。** 本轮还保留 113 次 forward、57 次 dX、57 次 dW。融合投影虽然减少 GEMM 数，但会引入不同尺寸算法、拼接/切片或参数布局改变；不能只以提交数判定更快。
2. **残差及 checkpoint 边界。** 主配置仍有 57 次 `[1,4096,768]` BF16→FP32 梯度转换，涉及 FP32 梯度累加契约。继续融合时应消除中间读写，并保留原累加精度，不能直接把所有梯度改为 BF16。
3. **归约与 attention 的 GPU 占用率。** Q/K norm backward 的部分和分组、FlashAttention tile 和寄存器压力需要真实 profiler。离线编译成功不能判断最佳 CTA 数或实际 SM 利用率。
4. **FP8 全步成本。** FP8 图仍有 1934 条命令，明显多于高精度路径的 717 条；动态量化和打包成本仍须与 GEMM 加速一起评估，因此没有默认强制 FP8。

在当前静态配置下，ckpt2 相比 ckpt1 进一步降低 arena，但增加少量命令；ckpt0 减少重算但占用更多 arena。实际可用显存还要扣除 CUDA/Lt 等额外开销，不能只按此表设置显存上限。

## 验证与复现

- 完整 CPU/解释器回归：**187 passed，10 skipped**。覆盖原生 Dense、模型训练、attention、checkpoint/resume、内存复用、legacy 导入及 WebUI 合同。
- 新增专项包含 **27 个**测试，已计入上述 187 项：三种精度的融合舍入、非重复 RoPE 表、共享梯度 FP32 累加、批量投影、因果 shift CE、批量 AdamW、不同 step fallback、图中大小参数/unused/overflow/force update、持久指针表 checkpoint 恢复。
- **SM120 离线编译 43 个特化全部通过**，最高 shared memory 16 字节，无不支持的 global scratch 或 cluster launch；该检查通过拒绝 Torch 导入执行。覆盖 head_dim=96/160、FP32/BF16/FP16、shift CE、指针表 AdamW/范数与图累积。
- 十个跳过项需真实 CUDA/cuBLASLt 或解释器不支持的 FP8/BF16 dot/舍入路径。CPU 测试中的 GEMM 使用 Torch 作为 oracle，不能替代原生 Tensor Core 精度或性能测试。本轮未运行实卡、未修改 bridge ABI。

```bash
python -m experiments.profile_native_schedule --dense --batch 1 --seq 4096 --checkpoint 1 --mode eager
python -m experiments.profile_native_schedule --dense --batch 1 --seq 4096 --checkpoint 1 --mode graph
python -m experiments.compile_dense_schedule --arch 120
TRITON_INTERPRET=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/test_dense_triton.py tests/test_dense_cublas.py tests/test_cublas_training.py \
  tests/test_native_remaining.py tests/test_native_attention.py tests/test_native_loader.py \
  tests/test_native_optimization.py tests/test_dense_schedule.py tests/test_legacy_resume.py \
  tests/test_cublas_webui_contract.py -q --disable-warnings
```

复现基线时，将 `e4eb938` 放到独立 checkout，使用本轮 `profile_native_schedule.py --repo /path/to/baseline`，其余参数相同。使用同一审计脚本比较，尤其不要把基线图重放命令数与新版准备阶段总命令数混在一起。
