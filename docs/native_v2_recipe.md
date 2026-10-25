# Native V2：选择性重算、split-weight Muon 与 packed attention

基于远端 `experimental/dense-cublaslt-fp8` 的 `95db35e`，本次实现位于
`codex/cublas-throughput`。模型仍为 32 层、H=1536、FFN=5504、Q/KV=16/4、
head_dim=96、vocab=6400、共享 embedding/head，共 1,010,277,888 参数。

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
  本机完整累计更新的共享配方采用 768 MiB；1792 MiB 作为可显式选择的原始策略保留。
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
