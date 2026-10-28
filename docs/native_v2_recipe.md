# Native V2：选择性重算、split-weight Muon 与 packed attention

基于远端 `experimental/dense-cublaslt-fp8` 的 `95db35e`，本次实现位于
`codex/cublas-throughput`。模型仍为 32 层、H=1536、FFN=5504、Q/KV=16/4、
head_dim=96、vocab=6400、共享 embedding/head，共 1,010,277,888 参数。

## 2026-10-28 当前已验证预设

当前共享原生预设采用cuDNN packed attention、cuBLAS Muon、FP8 scale/倒数融合、
attention自适应连续分组与组内长度上界、残差梯度融合、cast＋amax、权重scale缓存、
双1280 MiB checkpoint缓存、256 MiB复用池和512 MiB Muon workspace。
`--allocator_release_mib 13312` 控制CUDA异步池在同步时保留内存的阈值；它不是显存硬上限。
该参数也可在Config WebUI原生设置的Async pool retention MiB中调整，裸CLI省略时保留驱动设置。
16 GiB RTX 5070 Ti生产数据256微步窗口达到15,211有效目标/s；独立128微步为15,081，
扩大窗口的相同前128微步也为15,081。含loader和Muon更新，不含启动、预热及最终保存；
完整保存核验通过。权重scale在更新/加载后失效，只增加7168字节常驻标量。
分组保持1024长度粒度、最多8文档、4096 token等效调度成本和64 plan上限。
条件、显存和用户接受的attention 0.0011相对L2门槛见
[2026-10-28验证报告](reports/2026-10-28-native-14k-progress.md)。下面2026-10-26小节保留历史配置。

## 2026-10-26 代码更新与验证

后续验证已完成两个 native bridge 构建、融合/优化器数值回归和 160 微步生产数据训练。
已建缓存的训练窗口为 11,398 targets/s，尚未达到 14,000 或 18,000 targets/s。
完整条件、修复、确定性边界、保存证明和包含启动/保存的计时见
[本轮验证报告](reports/2026-10-26-native-validation.md)。

- `--fused_ops` 接通原生 Dense 融合策略，共享 V2 配方已启用。包括分块 FP8 转置、
  梯度合并、层内 RMSNorm 直接写计算 dtype，以及 SwiGLU 前反向同时生成后续量化的 amax。
  amax 使用实际写回 dtype 舍入后的值；重算和 VJP 回收保留尚待使用的统计缓冲。
  对满足对齐条件的 E5M2 梯度，量化同时写出正常与转置布局，后续 dW 复用转置结果。
- 预训练 loss 在日志、末批和保存退出边界读取；预取双缓冲通过 consumed/ready 事件连接
  计算流和拷贝流。预训练不再要求每个微步的前向后 loss 下载和全局同步。
  日志吞吐改用整个报告间隔的有效 targets 与墙钟时间，覆盖间隔内加载、更新和保存开销。
- `--selective_cache_by_seq 3072:768,4096:1024` 可分别设置两个输入桶的候选输出缓存预算。
  未指定的长度使用 `--selective_cache_mib`。示例数值只是配置语法，不是新的推荐最优点；
  共享配方仍保持 768 MiB，不能把增加预算本身当成已验证的加速。
- `--attention_backend hybrid --attention_cudnn_min_seq 4096` 接通分桶 attention：
  短桶使用 Triton 128×64 / 8-warps 前向，达到阈值的 packed BF16 输入使用原生 cuDNN。
  `cudnn` 模式对所有适用桶选择 cuDNN，`triton` 保留既有路径并作为默认。
  cuDNN 图只保留有上限的计划缓存；workspace 随当前操作释放，保存的 LSE 压回原 token 规模。
  文档元数据绑定到当前 mask，使 checkpoint 重放使用对应 forward 的文档边界。
  此路径不经过 Torch，也不导入 cuDNN Python frontend；需要可选的 C++ SDPA bridge。
  已验证候选采用 3072:1024 / 4096:1280 MiB；尚不代表全局最优配置。
  SM120 的 cuDNN packed 反向不支持确定性执行，验收按 BF16 容差进行，不能称为逐位重复。
- `--muon_gemm_backend cublas` 新增 cuBLASLt strided-batched NS 矩阵乘法路径，
  支持原来的 alpha/beta 更新表达式，共用现有 Lt workspace 和候选算法选择。
  momentum、mantissa、NS 次数和分块预算保持原语义。默认仍为 `triton`，新路径需要重新构建
  `model/dense_cublas/native` 的 bridge。两种后端的 16 个 CUDA 更新用例、
  9 组生产矩阵形状以及实际训练的 20 次优化器更新已通过对应验证。
- 新控制项通过同一原生 CLI 与 Config WebUI 传递。融合和 Muon GEMM 后端写入 resume 的
  execution options；恢复时拒绝未经声明的执行选项变化，旧 checkpoint 按原始选项解释。

可选 SDPA bridge 源码位于 `model/dense_cublas/native/attention.cpp`。CMake 的
`INSTINCT_BUILD_CUDNN=ON` 开关接收 `CUDNN_ROOT` 和 `CUDNN_FRONTEND_ROOT`，也可以直接指定
`CUDNN_INCLUDE_DIR` / `CUDNN_LIBRARY`。运行时 `CUDNN_PATH` 选择 SDK/runtime，
`INSTINCT_CUDNN_BRIDGE` 可指定已构建库。现有候选使用 cuDNN 9.27.0.42 与 Frontend 1.31.0；
SDK、下载缓存和编译产物不入库。本轮两个 Release bridge 均已构建成功。

生产 Parquet 现在可直接交给原生预训练入口。`dataset_compiler` 的 `plan-pretrain` /
`pack-pretrain` 命令用 Rust/Rayon 并行分词打包；Python 只做进程调度和 mmap 批次解码。
每个 Parquet row group 是有界缓存单元，后台准备下一块，保留文档边界、有效目标数和
下一未读 group/batch 游标。先在 `dataset_compiler/` 内构建 release CLI。
新数据路径不依赖 Torch/TorchAO，也不要求 HuggingFace Arrow 中间数据集。

```powershell
python -m scripts.native_v2_recipe --data-path D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet
```

上述命令仅打印原生命令；添加 `--run` 才启动。使用
`--sequence_buckets 3072,4096 --num_workers 4 --copy_prefetch` 可选择生产的两个桶、
Rust 预处理线程和设备双缓冲。原生 `batch_size × max_seq_len` 是这些桶的 token budget，
因此配方的 `3 × 4096` 对应 `4 × 3072` 和 `3 × 4096`。

## 已接通的运行路径

- 原生数值档位 `use_grad_checkpoint=4`：整层建立重算边界，按实际 B×S、
  激活 dtype 和缓存预算保留部分投影输出或 fused attention 的 O/LSE；
  反向重新建立 VJP，保留的 GEMM/attention 前向不重复执行，参数梯度仍正常计算。
  这是 native 的整数档位，不是该远端分支 PyTorch 的字符串
  `checkpoint_profile=ckpt4`（后者仍是整层重算）。
- 默认候选输出预算 1792 MiB 与原 V2 设计一致：4×3072 保留全部 K/V 和顶部
  4 层 gate/up；3×4096 保留全部 attention O/LSE 和顶部 2 层 gate/up。
  `--selective_cache_mib` 可调整预算。768 MiB 时分别保留全部 K/V、
  顶部 20 层 O/LSE，均不额外保留 gate/up。预算不包含层边界、优化器、量化临时量或驱动池。
  当前共享配方使用两桶各1280 MiB并在两桶缓存attention O/LSE；1792 MiB作为可显式选择的原始策略保留。
  ckpt3 也已提供；ckpt3/4 仅支持标准 Dense、独立投影，不与 LoRA/融合投影组合。
- `--muon_mantissa 1 --muon_momentum_dtype bf16`：
  隐藏矩阵为 BF16 高位 + 原始 uint16 低位；EMA/Nesterov 在 FP32 计算，
  动量存储可选 FP32/BF16。embedding、共享 head、norm 等伴随 AdamW 参数保留 FP32。
  残差流保留 FP32，Linear 计算输入为 BF16。NS 使用原 quintic 系数及 5 次迭代，
  每次矩阵乘法输出 BF16，更新按独立 decay 舍入及 FMA 后重新拆位。
  `--muon_batch_size`/`--muon_workspace_mb` 控制按形状分组的临时块，
  不改模型参数布局，也不写入影响续训精度的状态。
- `--packing --packed_attention`：使用文档边界 sidecar 构造 segment IDs，
  fused attention 前反向均隔离不同文档；按文档重置 RoPE，屏蔽跨文档预测与 padding。
  原有跨文档连续流 packing 保持为未指定 `--packed_attention` 时的行为。
  三字段 batch 可经 worker、裁切和预取路径传递。
  attention 的 forward/dQ 从本 query tile 最早文档的起点开始，dKV 在该 key tile
  最晚文档的终点结束，跳过确定无效的矩阵块；跨文档 tile 和尾部 padding 仍逐元素判定。
- `--fp8_filter auto` 对齐此机器 TorchAO tensorwise 过滤规则：
  排除共享 head 和不满足 16 对齐的投影，跳过 K≤4096 且 N≤1024 的小矩阵。
  原生 FP8 仍是 E4M3 前向 / E5M2 梯度；不承诺与另一套 GEMM 库逐 bit 相同。
  amax 已使用分级归约，支持该 recipe 超过 64M 元素的 FFN 激活。
- `--allocator_backend async` 使用 CUDA 流有序临时分配/释放，保留同步 host
  常量上传、跨 stream 的现有依赖及 graph 地址保护。释放 context 前先同步，再恢复
  普通 free 进行清理。默认通用后端仍是 `driver`，V2 配方显式选择 async。
  [CUDA Driver API](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html)
  定义了这些分配的访问顺序与生命周期要求。

原生 checkpoint 保留 uint16 低位、动量、梯度、优化器计数、scale、RNG 和数据进度，
恢复时拒绝不匹配的 mantissa/momentum 精度或 FP8 filter。原生保存/恢复已验证；
此前 PyTorch `.pth` 迁移器的旧 Muon 配方回归在远端原版也存在数值失败，
本轮不把跨后端精确续训列为已验证能力。

## 入口与配置

模型 JSON 为 `trainer/config_instinct_v2.json`，共享训练配方为
`trainer/recipes/instinct_v2_native.json`。Config WebUI 的
**Quick-select preset → Instinct V2 (native ckpt4)** 读取同一份文件，
同时应用 FFN=5504、精度、优化器、检查点和原生运行选项，不覆盖已有面板状态文件。

从仓库根目录预览命令；加 `--run` 才启动：

```powershell
python -m scripts.native_v2_recipe --data-path D:/data/pretrain.jsonl
python -m scripts.native_v2_recipe --data-path D:/data/pretrain.jsonl --run
```

额外 CLI 参数追加覆盖配方。例如 `--batch_size 4 --max_seq_len 3072`。
该原生数据入口目前仍使用分支已有的 JSONL loader；本次两 bucket 吞吐实验用固定的
4×3072（三文档）与 3×4096（单文档）合成批次交替，不能把它当成已经移植原仓库
Parquet/streaming/token-budget loader 的证明。

native CLI 的暂停请求（SIGINT 或 `checkpoints/.pause_request`）等待累计更新完成，
保存并检查 weights/resume 的 step 对齐后以 42 退出。有限 `--max_steps` 也等待更新边界。
完整性能实验的每一次实际参数更新都必须在退出前保存并验证当前状态。

## 验证及性能口径

检查覆盖：

- CPU 循环重用真实 storage：ckpt2/3/4 的 loss 与全部参数梯度；
  packed 与独立文档的 logits/loss/全部梯度；重复文档 ID、padding、边界标签；
  loader token/target 守恒及恢复；uint16 无损保存；暂停边界；实际 Streamlit AppTest 和 CLI 配方一致。
- Torch-free CUDA 进程：BF16/FP8、packed 与 4096 长度，ckpt4 对 ckpt2 的全部梯度；
  FP32/BF16 momentum 的更新、当前状态保存与恢复。driver 和 async 均运行。
- 原生 CUDA Muon 对独立 NumPy BF16 oracle：宽/高/方矩阵、多个 chunk、
  两种 momentum、两个 allocator。NS=0 时精确拆位更新逐 bit 对齐，
  NS=5 按声明容差验证 Tensor Core 归约与 BF16 中间舍入。

`experiments/bench_native_recipe.py` 默认只测固定权重前反向并保留真实优化器/梯度；
`--updates` 才包含累计、裁剪、Muon/Adam 更新，并在结束时写入且逐项检查完整状态。
合成计算窗口不含数据读取、分词、冷编译和保存时间，不能据此推断训练收敛或真实 loader 吞吐。

native `peak_live_gib` 是活跃张量字节数；`peak_arena_gib` 还含该 allocator 的空闲缓存
及 Lt workspace。async 的 CUDA driver pool reserved/used 另行报告，两种 reserved
不能相加：它们部分描述相同内存。原生 live 也不应直接冒充 PyTorch caching allocator
的 allocated。相对增量均在同一后端、相同统计口径内计算。

完整实测表与失败实验见 `docs/reports/2026-10-25-native-v2-throughput.md`。
