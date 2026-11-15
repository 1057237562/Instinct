# Native V2：选择性重算、split-weight Muon 与 packed attention

基于远端 `experimental/dense-cublaslt-fp8` 的 `95db35e`，本次实现位于
`codex/cublas-throughput`。模型仍为 32 层、H=1536、FFN=5504、Q/KV=16/4、
head_dim=96、vocab=6400、共享 embedding/head，共 1,010,277,888 参数。

## 2026-11-01 生产预设：MXFP8

共享配方 `trainer/recipes/instinct_v2_native.json` 已选择 `--compute_profile v2_mxfp8`。
CLI 和 Config WebUI 的 **Instinct V2 (native ckpt4)** 读取同一配置；不再需要实验脚本注入。

### Larger-memory full resume

The production wrapper accepts `--cache-mib 6144` (both 3072/4096 buckets)
and `--allocator-release-mib 24576`. These machine-specific overrides are
persisted in the registered run environment for later resumes; they do not
change the shared 16 GiB recipe. Cache budgets are execution settings: full
resume permits changes to `selective_cache_mib` and `selective_cache_by_seq`
only, with all other topology, checkpoint-mode and precision checks intact.
The production wrapper saves a complete checkpoint every 2000 microsteps;
manual pause and shard boundaries still force an immediate complete save.

An MXFP8 budget increase preserves the original compressed gate/up layer
selection in `cache_precision_reference` checkpoint metadata. Extra gate/up
outputs stay BF16, avoiding additional FP8 rounding. With the original
2684 MiB reference, 6144 MiB retains 7 FP8 and 13 additional BF16 gate/up
layers, plus attention O/LSE and K/V for all layers. Later resumes retain
this reference, including a fallback to the original budget; shrinking below
the reference is rejected. Checkpoints are read without rewriting their
weights, optimizer, RNG or data progress during this transition.
At 9216 MiB the same reference keeps 7 FP8 and 25 BF16 gate/up layers, covering
all 32 layers. The 5090 deployment uses this override and a 24576 MiB allocator
release threshold, while the shared recipe remains at 2684 MiB.

The 5090 fixed-weight screen resumed the complete production state at microstep
200808 with the original 1,010,277,888-parameter model and MXFP8/Muon precision.
It reused resident real-data samples of shapes 4x3072, 3x4096 and 1x10752,
discarded 9 warmup windows and measured 30 forward/backward windows per budget.
Useful throughput was 35609 / 36551 / 37578 tokens/s at 2684 / 6144 / 9216 MiB.
All sample losses matched. These windows include resident optimizer buffers but
exclude optimizer updates, loader time and checkpoint saves; the 5.5% gain is
not an end-to-end training claim. Native allocation peaks were 14.01 / 17.28 /
20.31 GiB and do not include every driver/library allocation.

`tests/test_native_cache_resume.py` audits complete restore and subsequent
updates on CPU. `experiments.validate_native_cache_resume` audits the actual
MXFP8 cached/recomputed CUDA paths with fixed weights and no optimizer updates.
Its two-layer audit is not a whole-model quality or throughput result.
`experiments.bench_native_recipe --shipped-recipe --resume-path ...` screens
cache budgets against a read-only full production checkpoint; it rejects
`--updates` with that checkpoint.
精度及算子选项集中在 `trainer/profiles/native_v2_mxfp8.json`，实验候选文件仅引用该档位。
通用裸 CLI 的默认值仍为 `tensorwise`。该 V2 档位检查 SM120、32 层 Dense 形状、BF16/Muon、
ckpt4、两个生产桶和必要 bridge；不兼容的配置直接报错。

| 项目 | V2 MXFP8 配置 |
|---|---|
| 隐藏层 Linear | 224 个投影的前向、dX、dW 使用 E4M3 MXFP8 / UE8M0，FP32 累加、BF16 输出 |
| Attention | cuDNN BF16 前向，FROST SM120 96 维 BF16 反向；softmax/LSE 等保留 FP32 |
| FFN / Norm | MX 量化融合；残差流 FP32；两桶各 2684 MiB ckpt4 缓存，最多 7 层压缩 gate/up |
| Muon | FFN 使用 FP16 Gram recurrence / restart3，仍为 5 次 NS；其余矩阵维持 BF16 NS |
| 数据 | Rust Parquet 编译打包；满长桶在原微批内按 512 对齐压缩，保留文档、目标数及源游标 |
| 存储容量 | 9728–12288 的已测 token 容量共享 12288 分配尺寸；罕见不足额尾批保留原形状 |

启动前运行无参数更新的形状预热，不前移训练数据游标或优化器计数。默认数据路径为
`./dataset/pretrain_coder_l2l3_24b.parquet`，WebUI 可编辑数据和 resume 路径；不会自动改写已保存的面板状态。
历史实验 MX checkpoint 的精度标记保持原值，正式入口可识别并严格核对；tensorwise checkpoint
不能作为同一精度配方直接续训。

**正式入口已完成独立 512 测量微步，达到 18,579.27 有效 TPS**：6,102,114 目标 / 328.436652 秒，
完整 544 微步 / 68 更新的 weights 和 resume 均与当前状态逐数组核验通过。
窗口包含生产 Parquet 缓存加载、H2D、前后向及真实更新，不含启动、预热和最终保存。
native live 峰值 13.910387 GiB，相对已验收 MX 实验增量 **0.00%**；相对 tensorwise 为 **−0.4626%**。
NVML 设备采样峰值 15.753025 GiB，峰值时空闲约 171.9 MiB，不等同于模型 allocated。
旧实验 MX checkpoint 的副本通过正式档位继续训练 72 微步：544→616、68→77 更新，
源游标 group=1/batch=480→552，待累积梯度为 0，完整保存核验通过，原文件未变。

首轮接线验证曾在第 48 微步之后发生 CUDA error 700 且无法保存；按用户明确授权才结束故障进程。
随后恢复历史实验的 attention 初始化顺序并补齐备用路径参数，上述完整窗口和独立续训均未复现错误。
没有隔离实验能证明具体根因，不能声称已定位该非法访问；长期稳定性和训练质量也不由短窗口证明。

已安装基础 `requirements-cublas.txt`、cuDNN SDK 并构建两个 native bridge 后，还需：

```powershell
python -m pip install --target .cache/native-dsl -r requirements-cublas-mxfp8.txt
# 首次安装且 .cache/cudnn-frontend 不存在时执行 clone；已有目录先检查本地改动。
git clone https://github.com/NVIDIA/cudnn-frontend.git .cache/cudnn-frontend
git -C .cache/cudnn-frontend checkout 51a3de73e122aeedafe68070acf3b7ff3970534e
cargo build --release --manifest-path dataset_compiler/Cargo.toml
```

正常入口命令预览如下，`--run` 才实际启动：

```powershell
python -m scripts.native_v2_recipe --data-path D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet
```

续训追加 `--from_resume 1 --resume_path <完整路径> --save_weight <输出前缀>`，并保持原数据、精度及缓存配置。
新建 tensorwise 对照需显式覆盖 `--compute_profile tensorwise --selective_cache_mib 2816
--selective_cache_by_seq 3072:2816,4096:2816`；这不是 MX checkpoint 的无缝回退。

## 2026-10-31 原生预设历史记录

当前共享原生预设采用cuDNN packed attention、cuBLAS Muon、FP8 scale/倒数融合、
attention自适应连续分组与组内长度上界、残差梯度融合、cast＋amax、权重scale缓存、
双2816 MiB checkpoint缓存、128 MiB复用池和512 MiB Muon workspace。
当前缓存顺序为全部O/LSE、K/V投影、再分配gate/up；V2两桶实际各缓存32层O/LSE、32层K/V和顶部4层gate/up，合计2784 MiB。
Optimizer.backward完成一层反向后释放该层私有selective缓存；手动Engine.vjp保持原来的保留行为。
层末残差与下一层input RMSNorm共享一个有界BF16暂存区，反向仍重建完整VJP。
重算时E4M3激活量化同时生成dW所需的转置布局，原始checkpoint前向不多存该布局。
Muon在SM120上对已测形状使用cuBLASLt pointer-array三角分块和镜像；不同batch采用64或512分块，其余保留原路径。
通用FP8双布局量化使用64 tile。精确累积保持原样，fast accumulation只保留实验开关，未启用。
`--allocator_release_mib 13312` 控制CUDA异步池在同步时保留内存的阈值；它不是显存硬上限。
该参数也可在Config WebUI原生设置的Async pool retention MiB中调整，裸CLI省略时保留驱动设置。
2026-10-28旧预设在16 GiB RTX 5070 Ti生产数据256微步窗口达到15,211有效目标/s；独立128微步为15,081，
扩大窗口的相同前128微步也为15,081。含loader和Muon更新，不含启动、预热及最终保存；
完整保存核验通过。权重scale在更新/加载后失效，只增加7168字节常驻标量。
当前分组为256长度粒度、最多4文档和128 plan上限。前向通过9个描述符锚点校准耗时模型并选择连续分组；
反向保持4096 token等效调度成本的原分组，避免采用已超过数值门槛的双向改组方案。
同一backend/线程/执行流的plan共享cuDNN handle；不同流独立，检查CUDA context和stream匹配。
重建`native_sdpa`可启用共享handle扩展；旧ABI继续运行原路径，但不具备该项显存收益。
同一生产256微步窗口为16,045.519 TPS，相对上一轮15,596.733提高2.8774%；
native live峰值14.1083 GiB（+6.2223%），NVML设备采样峰值15.4776 GiB，峰值时free 0.4433 GiB。
完整288微步状态已保存核验。测量仍含加载/H2D/更新，排除启动、预热和最终保存；基准固定LR为5e-4，配方LR仍为3e-4。
后续512微步确认出现15,790.802 TPS，两个FFN GEMM在冷启动计时漂移中选到较慢候选。
增加有界预热与漂移检查后，512微步达到15,979.457 TPS，其中相同前256微步为16,037.887 TPS；
该历史记录只能确认短窗口突破16k。
后一轮NVML采样峰值15.7900 GiB、free仅0.1309 GiB，显存余量随运行环境变化，不能只引用较低的一次峰值。

2026-10-31的新默认组合在**完整512测量微步达到16,053.403有效目标/s**：6,102,114目标/380.113418秒，
419个短桶和93个长桶，4次更新预热后测64次真实更新，使用生产recipe学习率3e-4。
native live峰值14.108333 GiB，相比此前同采样512微步为**0.00%增量**；NVML设备峰值15.483471 GiB（+0.038%），
峰值时free 0.437428 GiB。两个设备峰值来自不同时段，不用于单独归因共享handle；
独立120-plan对照的运行时常驻增量从608降到250 MiB，减少358 MiB（58.88%）。
生产窗口使用已建Rust packed缓存，包含加载、H2D、前后向、更新和日志；排除启动、预热和最终保存。
共544总微步/68更新的weights与完整resume逐数组核验通过，`torch_imported=false`；完整脚本共511.868秒。
`--tune`下，cuBLASLt先按启发式排序、workspace和waves筛到最多4项；cuDNN保留最多4个不同受支持plan。
粗测后仅交替复测基线和领先项，收益不足1%保留基线；单候选跳过比较计时。
多候选先做约100 ms的有界预热；确认样本漂移超过5%时只重试一次。持续波动时，
仅在候选最慢样本仍比基线最快样本快1%以上时接受候选，否则保留启发式基线并记录警告。
cuDNN在训练消费数据前预热合法形状目录，V2两桶共60种几何/120 plan，未超过128上限，避免首次建图落在测量窗口内。
当前硬件实际只有1–2个可用SDPA候选，不能将启用选型描述为已经获得大幅内核加速。
完整32层FP8模型的重复基线还出现较大梯度相对差异，详见报告中的重复性诊断；不以小模型测试代替该结论。
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
  历史 tensorwise 配方覆盖为两桶各2816 MiB，同时缓存attention O/LSE、K/V及顶部4层gate/up；新 MX 档位见本文开头。
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

额外 CLI 参数追加覆盖配方；`v2_mxfp8` 会拒绝超出已支持形状的覆盖。其他形状可显式选择 `--compute_profile tensorwise`。
当前原生入口支持 JSONL 和有界 Parquet 预训练；Parquet 要求 packed、文档隔离、单 GPU。
Rust 编译/分词、磁盘 packed 缓存、按 token budget 组桶、预取及完整数据游标已接入。
早期固定合成样本结果仅代表计算性能；生产端到端结果另用
`experiments/bench_native_production.py` 验证，并记录数据起点、有效目标、预热与保存边界。

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

## MXFP8 实验与精度参照

MXFP8 通过原生 cuBLASLt 使用 32 元素分块和 UE8M0 尺度；行/转置布局独立量化，
可融合 Norm、SwiGLU、反向梯度量化，并压缩 ckpt4 gate/up 缓存。
历史实验由 `experiments/native_candidate.py` 启用；现在由正式 `--compute_profile v2_mxfp8` 接线，状态见本文开头。
开启 MX 的 checkpoint 记录精度设置，普通 tensorwise 后端不能静默载入为同一计算配方。

精度参照为 BF16，不要求追随 tensorwise 的舍入误差。验收同时检查完整 loss、全部参数梯度、
有效位置 logits 分布与短训练稳定性；不能用局部相对误差代替长期模型质量验证。
2026-11-01 的 96 维 attention 候选已完成两个独立的 512 微步生产窗口，分别为
18,584.27 / 18,578.40 有效 TPS，均超过 18,500。相对同口径 tensorwise，
native live 峰值降低 0.4626%。精度代价、缓存字节核算及
失败候选见 [最新进展](reports/2026-10-28-native-14k-progress.md)。

显式实验配置为 `experiments/native_mxfp8_candidate.json`，继承共享 native fused policy，
只保存实验覆盖项。它启用 MXFP8 E4 前向/梯度、K/V 投影、压缩 gate/up、
矩形 FFN 的 FP16 Gram Muon，以及最多 24 文档的 96 维 FROST 反向。
压缩长桶共享 12288 token 存储容量，cache 预算为 2684 MiB（7 层压缩 gate/up）。
以上成绩产生于接入共享 recipe 和普通训练 CLI 之前，不能替代接线后的独立复验。

本机依赖：SM120、CUDA 13.1 工具链、cuBLASLt 13.2.1、Triton 3.4；
FROST 使用 `.cache/cudnn-frontend` 的 `51a3de73e122aeedafe68070acf3b7ff3970534e` 和
`.cache/native-dsl` 的 CuTe DSL 4.8.1。路径可由 `INSTINCT_CUDNN_FRONTEND_SOURCE` /
`INSTINCT_CUTE_DSL_PATH` 指定；源代码不匹配时适配器拒绝编译。

在 worktree 根目录复现有限验收窗口（结束会保存并核验当前完整状态）：

```powershell
$env:PYTHONUTF8 = '1'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
# 默认使用 dataset_compiler/target/release/dataset_compiler.exe。
$output = 'experiments/results/cublas/mxfp8-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json'
& E:/miniconda3/python.exe -m experiments.bench_native_production `
  --data D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet `
  --tokenizer D:/AI/Instinct/model --output $output `
  --candidate-config experiments/native_mxfp8_candidate.json `
  --start-group 0 --tail-batches 64 --warmup-updates 4 --updates 64 `
  --learning-rate 0.0003 --cache-mib 2684 `
  --selective-cache-by-seq 3072:2684,4096:2684 --compact-long-multiple 512 `
  --attention-backend cudnn --muon-gemm-backend cublas --muon-workspace-mib 512 `
  --tune --workspace-mib 32 --candidates 8 --save-interval 1000 `
  --async-release-mib 13312 --allocator-cache-mib 128 `
  --compact-gradients --aligned-pointwise --async-attention-metadata `
  --attention-plan-quantum 256 --attention-plan-limit 192 --attention-docs-per-plan 4 `
  --fused-residual-norm --fused-norm-backward --fused-scale-inverse `
  --fused-residual-backward --fused-cast-amax --cache-weight-scales `
  --attention-group-penalty 4096 --attention-local-maxima `
  --ffn-quant-tile 64 --ffn-flat-forward --norm-rope-rows 8 `
  --profile-stages --profile-pipeline --sample-device-memory
```

该窗口复用 Rust packed 缓存，包含加载、H2D、训练和真实更新；排除启动、预热和最终保存，
也不计作冷缓存构建吞吐。`tail-batches` 会跳过首块前缀且不会训练被跳过的数据，因此这是
指定数据位置的有限实验，不是完整预训练 epoch。MX/full-resume 必须通过相同精度配置恢复；
正式入口识别已验证的 MX 配置，tensorwise 后端会拒绝其精度合同。

验证正式入口时使用 `--shipped-recipe`，不要同时传 `--candidate-config` 或 `--compact-long-multiple`：

```powershell
python -m experiments.bench_native_production --shipped-recipe `
  --data D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet --tokenizer D:/AI/Instinct/model `
  --output experiments/results/cublas/production-profile.json `
  --start-group 0 --tail-batches 64 --warmup-updates 4 --updates 64 `
  --profile-pipeline --sample-device-memory
```

完整生产训练入口 `python -m experiments.run_native_1b_acceptance` 现在会注册到共享的
`checkpoints/native_cublas/training_run.json`。文件名保留历史的 `1b`，实际目标为整个
24B 语料的一轮训练。主仓库与 cuBLAS 工作树的 Config WebUI 都显示独立的
“原生 cuBLAS 生产训练”面板；不覆盖当前编辑的模型参数。

该面板读取原生状态和日志，按累计有效 token 显示进度；暂停按钮使用这一轮专属的
暂停标记，等待累计更新完成及完整保存。续训按钮先校验检查点的步数、token、优化器更新、
数据游标和零待累积梯度，再从原工作树运行同一原生入口。不会通过 PyTorch 入口恢复 NPZ。
运行进程以 PID、创建时间、Python 路径、工作目录和入口参数联合确认，避免 PID 复用误判。

## Exact continuation with Parquet shards

The production run can resume from a complete `dataset_compiler split` directory
using its existing NPZ checkpoint. The directory's `manifest.json` must contain
the original `source_identity`, original row-group sizes and contiguous shard
ranges. `verify-shards` hashes all physical shards with bounded Rust workers
before the native loader uses them. Original files are not required afterward.

`ParquetPretrain` keeps the original logical path, fingerprint, global row-group
index, packing seed and batch cursor. New cache builds translate a global group
into a physical shard/local group and normalize the packed cache metadata back
to the original logical corpus. Existing packed caches are reusable. Thus a
resume cursor inside a row group or at a shard boundary retains exactly the
same next batch and token totals.

`experiments.run_native_1b_acceptance --resume-path CHECKPOINT --data-path SHARD_DIR`
selects the directory explicitly. Config WebUI's registered native resume uses
the persisted `data_path` in `checkpoints/native_cublas/training_run.json` and
forwards it to that entry point. This changes the next resume's input; it does
not restart an active process or rewrite checkpoint tensors/progress.

Selecting an individual manifest-listed shard instead limits this run to that
file. At its end the trainer uses the normal final-batch gradient flush, saves
a complete zero-pending-gradient resume state and pauses. It keeps the current
epoch and global next-group cursor. The production wrapper registers the next
file only after confirming the save; the next Resume click starts that file.
The last shard completes the epoch normally. Flushing a partial accumulation
at a shard boundary creates an optimizer-update boundary there; batch order and
token accounting remain unchanged, but the update grouping may differ from an
uninterrupted whole-file run.

The manifest can carry `native_plans` keyed by maximum sequence length. With
the original global plan and selected file, the original large file and other
physical shards need not be present. Missing/wrong selected files, modified
hashes or a cursor outside that shard are rejected. A plain Parquet without a
matching parent split manifest retains the ordinary single-source contract.
