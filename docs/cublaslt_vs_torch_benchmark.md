# 基准报告：原生 cuBLASLt/Triton 训练 vs PyTorch reduce-overhead

分支：`experimental/dense-cublaslt-fp8` · RTX 5060 Ti 8GB（SM120）· Windows
日期：2026-10-25 · 附带驱动级显存审计（见"显存口径"）

## 实验条件（两侧严格对齐）

- 模型：`trainer/config_cublas_moe.json`（InstinctV1MoE，198M 总量 / 64M 激活，
  hidden 768 × 8 层 × 4 experts top-1，moe_intermediate 2432，head_dim 96）
- 数据：`dataset/instinct_moe_mini.jsonl`（4800 行，预训练格式）
- 设置：batch 8 × seq 256，accumulation 2，AdamW 5e-4，40 步短训，seed 42
- 精度：两侧均为 fp32 主权重 + bf16 激活计算；`use_grad_checkpoint=1`
  （torch 侧必须显式 `--use_grad_checkpoint 1`：`config_from_args` 中 CLI 覆盖 JSON）
- torch 基线：官方 `trainer/train_pretrain.py --use_compile 1 --compile_mode reduce-overhead`
- 吞吐计法：step 10→40 稳态段均值（不含编译/温机与数据加载热身）

## 吞吐（tokens/s，稳态）

| 方案 | tok/s | 秒/step | 备注 |
|---|---:|---:|---|
| **torch reduce-overhead（+动态图退出）** | **~10,300** | 0.20 | CUDA graph 仅覆盖静态子图 |
| torch eager | ~9,200 | 0.22 | 三段实测 9402/8872/9343 |
| native bf16 | ~2,320 | 0.88 | 四段 2317/2293/2331/2320 |
| native bf16 + `--fuse_projections` | ~2,220 | 0.92 | 无收益（误差内，略负） |
| native fp8 | ~2,056 | 1.00 | 比 bf16 慢 ~11%（每步 200+ 次量化内核） |
| native bf16 + `--cuda_graph` | ~222 | 9.1 | **10× 灾难性退化，见坑 ③** |

**结论：torch reduce-overhead ≈ 原生的 4.5 倍**；原生叠加全部缓和器也无法收敛差距。
torch reduce-overhead 相对 eager 增益只有 ~12%，因为 MoE 动态分派使大部分动态子图
被挤出 CUDA graph（见坑 ①②）。

## 为什么 cuBLASLt 后端比 PyTorch 慢

先用算术定位问题量级：本配置每步 2048 token × 63.9M 激活参数 ≈ 0.79 TFLOP
（fwd+bwd+重算 ≈ 6·N·T）。native 0.88s/step 折合有效算力 **~0.9 TFLOPS**，
torch eager 折合 ~3.6 TFLOPS —— 两者都远低于 5060 Ti 的 bf16 算力上限。
即**两侧瓶颈都不在 GPU 计算，而在每 op 的 CPU 调度成本**；差距来自调度成本的
一个数量级差异（每额外 op 开销 ≈ (0.88−0.22)s ÷ ~250 内核 ≈ 2.5ms）。
按因素分解（①②为实测，③④⑤为结构分析）：

1. **Python 驱动层逐内核调度（主因）**。原生引擎每个 kernel 经
   Python → ctypes → cuLaunchKernel 发射：参数打包、Tensor 元数据（shape/stride/
   dtype）构造、Triton launcher 元数据查找全部走 Python 运行时。每步 MoE
   堆叠出 24 组 expert GEMM + 32 组 attention GEMM，再加 norm/RoPE/量化/
   elementwise，合计 ~250 次发射 × 每次 ~1-3ms 的 Python 开销 ≈ 0.5-0.7s，
   正好补齐与 torch 的差距。torch 的等价调度都在 C++ dispatcher（eager）或
   编译产物里摊薄，单片 savable 状态也在 C++ autograd 引擎里回放。
2. **零算子融合。** torch eager 内部 RMSNorm/RoPE/CE 等已是融合 kernel，
   inductor 再把相邻 norm+cast+softmax 融合并配 CUDA graph 消发射；原生是
   逐 op 手写 kernel，每个逻辑变换（含 fp8 的 amax→scale→quantize→dequant
   四件套）都是独立发射，op 数量比 torch 同一路径多 2-3 倍。
3. **fp8 比 bf16 更慢的原因**：这些尺寸的 GEMM（如 2048×2432×768）本就吃不满
   tensor core，fp8 省下的算力远小于付出的量化会计成本——每投影每步都有
   weight scale 更新 + activation amax + cast 内核，纯粹是负优化（实测 −11%）。
   fp8 只有在大 GEMM 且量化 fuse 进 kernel 时才可能反转。
4. **反向引擎差异**：原生 VJP 引擎按 value 拓扑在 Python 里逐节点驱动；
   torch autograd 在 C++ 多线程引擎执行。小模型下反向的 CPU 侧调度占比
   与前向同量级，进一步放大差距。
5. **无 graph 兜底的静默代价**：CUDA graph 本可以把逐次发射成本一次性摊销
   （torch reduce-overhead 因此 +12%，且同配置下原生 graph 从 0.88s 崩到
   9.1s —— 重捕风暴反向证明发射开销确实是主要瓶颈），但动态 MoE dispatch
   让两条栈都吃不满 graph（见坑 ③）。

对应的提升方向（原生侧）：把每步调度从 Python 下沉到 C++/预编译调用计划
（固定形状 call plan），融合 norm+RoPE+quantize 链，MoE 换固定形状 dispatch
以解锁 graph；这些均未实施，当前原生栈定位在"零 torch 依赖"而非性能。

## 显存（同配置峰值，等效口径对比）

| 方案 | 分配峰值 (allocated 语义) | 保留池 (reserved/arena) | 8GB 卡可用性 |
|---|---:|---:|---|
| native bf16，ckpt=1 | 4,656 MiB | ~4,750 MiB | 整机峰值 ~6.2GB，余量 ~1.9GB ✅ |
| native bf16，ckpt=0 | 9,716 MiB | 9,804 MiB | 超物理显存 → WDDM 溢出系统内存 ⚠️ |
| torch eager，ckpt=1 | 4,109 MiB | 5,040 MiB | ✅ |
| torch reduce-overhead，ckpt=1 | 3,898 MiB | 5,270 MiB | ✅（图池抬高 reserve ~230MiB） |
| native `--fuse_projections --cuda_graph` | — | — | cuMemAlloc OOM ❌ |

**结论：显存维度 torch 略优、原生无明显优势。** allocated 口径 torch 少 ~550MiB；
reserved 两侧持平 ~5GB。原生的常驻项（稳态 3051MiB：AdamW 态 2271 + fp32 参数与
RoPE 781，workspace 仅 32MiB）属于亮点，但被激活/临时分配反超。

## 数值一致性交叉验证

40 步内两侧损失轨迹吻合（bf16：native 7.03→6.18 vs torch 7.17→6.01；
reduce-overhead 训练损失曲线一致），确认配置公平。

## 显存口径说明（驱动级审计）

- 原生 `live` 计账与物理一致；ckpt=0 峰值时 `cuMemGetInfo` 实测 `device_free=0`
  且分配仍成功 —— 为 Windows WDDM 将溢出部分映射系统内存（不 OOM，但 PCIe
  带宽惩罚显著，即 eager ckpt=0 慢的根因）。nvidia-smi 的 used 不含溢出部分，
  不能用它标定峰值。
- 原生 arena = live + pool 缓存（≤256MiB）+ cuBLASLt workspace（raw 旁路 32MiB）。
- torch 口径来自进程内采样 `torch.cuda.max_memory_allocated / max_memory_reserved`。

## 记录的坑（均为实测复现）

1. **torch reduce-overhead 直接跑会崩**：backward 报 `accessing tensor output of
   CUDAGraphs that has been overwritten by a subsequent run`（`model_instinct.py`
   RMSNorm 的 saved tensor 被覆写）。已修复：训练循环每步前调用
   `torch.compiler.cudagraph_mark_step_begin()`（commit `f27c21b`）。
2. **mark_step 修复不完整**：ckpt=1 的手动重算会 replay 编译图继续触发覆写；
   ckpt=0 也崩。根因是 MoE 专家分派的 `nonzero()` 产生 9 种动态尺寸（编译期
   警告），多图重录导致输出缓冲复用冲突。workaround：开启
   `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs`（动态子图退出
   graph 捕获），reduce-overhead 才可与现有 MoE 组合训练，代价是图覆盖面缩小。
3. **原生 `--cuda_graph` 与动态 MoE dispatch 组合 = 重捕风暴**：每步重新捕捉整图，
   40 步均值 ~222 tok/s（比不开 graph 慢 10×）；`fuse_projections` 叠加时直接 OOM。
   两条技术栈共同的结构性痛点：现有 MoE 实现需要固定形状 dispatch path 才能吃到
   CUDA graph 红利。
4. **原生 ckpt=0 在 8GB 卡上靠 WDDM 溢出而非 OOM 存活**：判别信号是
   `device_free==0`；显存规划应把"物理可分配"按 ~7GB（8151 − 上下文/桌面）估计。

## 总评

| 维度 | 更优者 | 幅度 |
|---|---|---|
| 吞吐 | torch reduce-overhead | ~4.5×（10.3k vs 2.3k tok/s） |
| 显存 | torch（小幅） | allocated 少 ~550MiB，reserved 持平 |
| 稳健性 | torch（需 1 行修复 + 1 个 inductor 开关）；native ckpt=1 默认可用 |

原生的价值定位在零 torch 依赖（autograd/torch 生态解耦、可审计的显存账本与
32MiB workspace），而非性能与显存。若要让两条路线都吃到 CUDA graph 收益，
候选方向是为 `model_instinct` 的 MoE 增加固定形状（padding 型）dispatch path。

## 复现

```bash
# 原生 bf16 / fp8
python trainer/train_cublas.py --train_type pretrain --config_path trainer/config_cublas_moe.json \
  --data_path dataset/instinct_moe_mini.jsonl --fp8 0 --batch_size 8 --max_seq_len 256 \
  --accumulation_steps 2 --max_steps 40 --seed 42
# torch reduce-overhead（需坑 ①修复 + 坑 ②inductor 开关）
python trainer/train_pretrain.py --config_path trainer/config_cublas_moe.json --use_moe 1 \
  --use_grad_checkpoint 1 --use_compile 1 --compile_mode reduce-overhead \
  --data_path dataset/instinct_moe_mini.jsonl --batch_size 8 --max_seq_len 256 \
  --accumulation_steps 2 --epochs 1 --dtype bfloat16 --param_dtype fp32
```

torch 侧基准脚本（时间戳包裹器 / 显存采样器 / inductor 开关注入）为临时文件，未随仓库提交。
