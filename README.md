# Instinct

从零训练小型语言模型,同一套纯 PyTorch 代码覆盖 Dense、MoE 与 Instinct V2 latent
recurrent-depth 三种架构。当前仓库配置已在单张 RTX 5070 Ti 16GB、Windows 11 环境
完成预训练、SFT、断点续训与推理验证。

所有核心算法(Pretrain / SFT / LoRA / DPO / PPO / GRPO / CISPO / Agentic RL / 蒸馏)均用 PyTorch 原生实现,不依赖 `trl` / `peft` 等高层封装,每一行代码都可读、可改、可复现。

---

## 特性

- **单卡可训**: V1 MoE 16GB + Muon 配置为 678.7M 总参数 / 106.2M 激活参数（hidden 512, 32 层）
- **纯原生实现**: 无第三方训练框架抽象,从零手写 Transformer、LoRA、RL 算法
- **完整训练链路**: 预训练 → CPT → SFT → LoRA → DPO → PPO / GRPO / CISPO → Agentic RL → 蒸馏
- **多架构**: Dense、MoE 与 Instinct V2 recurrent depth；V1 MoE 为 8 experts / top-1 路由
- **Instinct V2 循环深度**: Prelude / latent recurrent core / Coda，在测试时用 `num_steps` 扩展隐空间推理计算
- **推理快路径**: 静态 KV cache + 解码步 CUDA graph + 融合 RMSNorm/SwiGLU 内核；MoE 分组 GEMM 按平台自动选择 native / triton / cached 后端
- **自适应 Sequence Buckets**: 实验性按长度分桶 packing，使用 wall-time DP、显存感知 batch、确定性混排与持久化编译缓存降低 padding 和训练耗时
- **双格式语料**: JSONL 与 Parquet 均可直接训练,配套 Rust 编译器(约 3 倍体积缩减),流式分块跨格式精确对齐
- **强制数据身份清洗**: 分诊 → 审计 → 清洗/修复 → 身份锚点 → 审阅交接的完整工具链,所有训练语料使用前必须过 identity 检查
- **现代运行时兼容**: 已适配 Transformers 5 / huggingface-hub 1，支持配置、tokenizer 与 Safetensors 往返保存
- **低内存数据流水线**: 支持 SFT 数据标准化、流式 replay 抽样、磁盘分块混洗、数据缓存 LRU 预算和 WebUI 自动发现数据集
- **生态兼容**: 权重可转换为 HuggingFace 格式,支持 `llama.cpp` / `vllm` / `ollama` / `sglang`
- **中文友好**: 自带 6400 词表 BPE tokenizer,支持工具调用(`<tool_call>`)与思考(`<think>`)标签

---

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

`requirements.txt` 不固定 PyTorch 构建版本，需要另行安装与显卡驱动兼容的 CUDA 版
PyTorch。无 GPU 也可进行 CPU smoke test，但完整训练建议使用 NVIDIA GPU。

#### 当前验证环境

以下是本仓库当前实际使用的本地环境（2026-09-06），不是最低配置要求：

| 项目 | 当前配置 |
|------|----------|
| 操作系统 | Windows 11 家庭中文版 64 位（build 26200） |
| CPU | AMD Ryzen 9 9950X，16 核 / 32 线程 |
| 内存 | 48GB（系统可用容量约 47.2GiB） |
| GPU | NVIDIA GeForce RTX 5070 Ti，16GB（16303MiB） |
| NVIDIA 驱动 | 616.56（驱动报告 CUDA 13.4） |
| Python | 3.12.8 |
| PyTorch | 2.13.0+cu132（CUDA runtime 13.2） |
| cuDNN | 9.2.0 |
| Transformers | 5.16.1 |

当前 GPU 为 Blackwell（compute capability 12.0）。Windows 下启用
`torch.compile` 时必须使用 UTF-8 模式；Config WebUI 启动训练时会自动设置
`PYTHONUTF8=1`。MoE 的 grouped-GEMM `auto` 后端在 Windows 上依赖 Triton
（`requirements.txt` 已包含 `triton-windows`，Linux 由 PyTorch 自带），缺失时会
退回逐层同步的 `cached` 路径，推理吞吐会大幅下降（见[解码快路径](#解码快路径chat-webui--eval_llm)）。

### 下载数据

将数据集放入 `./dataset/` 目录(推荐从 ModelScope 或 HuggingFace 单独下载所需文件,无需全部克隆;
数据本体不入库,仓库只保留说明文档)。语料支持 JSONL、gzip JSONL、Parquet 或分片目录:

```text
./dataset/
├── pretrain_t2t_mini.jsonl    # 轻量预训练数据(推荐,1.2GB)
├── sft_t2t_mini.jsonl         # 轻量 SFT 数据(推荐,1.6GB,含工具调用样本)
├── pretrain_t2t.jsonl         # 完整预训练数据(10GB)
├── sft_t2t.jsonl              # 完整 SFT 数据(14GB)
├── rlaif.jsonl                # RLAIF 强化学习数据(24MB)
├── agent_rl.jsonl / agent_rl_math.jsonl   # Agentic RL 数据
└── dpo.jsonl                  # DPO 偏好数据(53MB)
```

所有训练器的 `--data_path` 都接受单个文件、`.parquet`/`.pq` 编译产物,或一个目录
(自动按文件名排序展开为分片列表;同目录下 Parquet 优先于 JSONL,不混用)。
`*.report.json` 审计旁车文件会被自动忽略。语料来源、配比与重建命令见
[`dataset/dataset.md`](./dataset/dataset.md)。

### 准备与混合 SFT 数据

`scripts/data_builder/prepare_sft_data.py` 可将 CodeAlpaca、SmolTalk、BigCode Exec、Magicoder
和 No Robots 等数据统一转换为 Instinct 的 `conversations` JSONL：

```bash
python scripts/data_builder/prepare_sft_data.py codealpaca-local
python scripts/data_builder/prepare_sft_data.py smol-smoltalk --max-samples 100000
python scripts/data_builder/prepare_sft_data.py bigcode-exec-50k
```

混合大规模 Coding、Math 与原始 T2T replay 时可运行：

```bash
python scripts/data_builder/mix_sft_datasets.py
```

该脚本用 reservoir sampling 流式抽取大型 T2T 数据，并通过磁盘分块外部混洗限制峰值
内存；默认生成 `dataset/sft_magicoder110k_mathinstruct_t2t_replay20.jsonl`。更多数据源、
配比、继续 SFT 数据构建和校验方式见 [`dataset/dataset.md`](./dataset/dataset.md) 与
[`scripts/data_builder/README.md`](./scripts/data_builder/README.md)。所有输出
使用 `sft_` 前缀后，Config WebUI 会自动识别。

### Rust dataset 编译器(可选)

`dataset_compiler/` 提供一个 Rust 工具,把 JSONL 语料编译为 Parquet:产物约为源文件
1/3 体积(zstd),加载时无需 Python JSON 解析,并保留 `token_count` 列供有界流式规划。
训练器与 WebUI 直接读取 `.parquet`,与 `.jsonl` 视为同一语料(可跨格式断点续训):

```bash
cd dataset_compiler && cargo build --release

# 编译;--align-chunk-bytes 取训练的 --streaming_chunk_mb 值,保证流式分块跨格式精确对齐
./target/release/dataset_compiler compile \
  -i ../dataset/pretrain_x.jsonl -o ../dataset/pretrain_x.parquet \
  --align-chunk-bytes 1GiB

# 逐行核对行数与取值,零漂移才通过
./target/release/dataset_compiler verify ../dataset/pretrain_x.jsonl ../dataset/pretrain_x.parquet
./target/release/dataset_compiler inspect ../dataset/pretrain_x.parquet   # footer 摘要
```

Schema 规则与续训语义(对齐/未对齐编译、chunk 游标重放范围)详见
[`dataset_compiler/README.md`](./dataset_compiler/README.md)。

### 数据身份清洗(强制)

所有新增或复用的训练数据(预训练/CPT/SFT/LoRA/DPO/RLAIF/Agent RL/蒸馏)在训练前
必须过身份清洗:数据不得教模型自称 Qwen/通义千问等无关模型或厂商,禁止盲字符串替换,
DPO 的 `chosen`/`rejected` 两侧都要检查。工具链在 `scripts/data_builder/`:

```bash
# 1. 首轮分诊:产出候选行 + SHA-256 审计报告(候选需人工复核,不是自动删除)
python scripts/data_builder/filter_anomaly_candidates.py dataset/sft_t2t_mini.jsonl \
  --profile all --output dataset/review_candidates/t2t_mini_candidates.jsonl

# 2. 身份污染分层量化
python scripts/data_builder/audit_identity_contamination.py dataset/<corpus>.jsonl \
  --output dataset/review_candidates/<corpus>.identity_candidates.jsonl --emit review

# 3a. 修复式清理(逐行补丁,零删行)或 3b. 删除式清理,产物用 identity_clean/identity_repaired 后缀,不覆盖原始数据
python scripts/data_builder/apply_identity_repairs.py dataset/<corpus>.jsonl \
  --patches dataset/review_candidates/_repair_batches \
  --output dataset/<corpus>.identity_repaired.jsonl

# 4. 生成身份锚点,按 0.5%~2% 混入最终 SFT
python scripts/data_builder/build_identity_anchors.py --output dataset/identity_anchors_instinct.jsonl
```

每次清洗都必须产出审计信息(来源 revision、输入/输出行数、删除/改写数、关键词命中数、
输出 SHA-256)。完整流程与本项目实际执行记录见
[`scripts/data_builder/README.md`](./scripts/data_builder/README.md)。

### 推理

```bash
# 使用 Transformers 格式模型
python eval_llm.py --load_from ./instinct-3

# 使用原生 torch 权重(out/ 目录下)
python eval_llm.py --load_from model --weight full_sft

# 叠加 LoRA 权重
python eval_llm.py --weight full_sft --lora_weight lora_medical

# 启用自适应思考
python eval_llm.py --load_from ./instinct-3 --open_thinking 1

# 启用 Early Exit 动态推理
python eval_llm.py --weight full_sft --early_exit 1

# 逐层 Top-k 预测(logit lens)
python eval_llm.py --load_from ./instinct-3 --logit_lens 1

# Instinct V2 循环深度:增加隐空间推理计算
python eval_llm.py --model_architecture looped --num_steps 32
```

推理默认走 `--inference_compile auto` 档(融合内核 + 静态 KV cache + 解码 CUDA graph),
`full` 额外编译整个 Transformer 主干,`off` 为纯 eager,见[解码快路径](#解码快路径chat-webui--eval_llm)。

### 统一评测 WebUI

现已支持 [GSM8K 数学评测](docs/gsm8k.md)：官方 test 集、批量生成、accuracy / pass@K、严格数值答案评分与压缩分析报告。

运行 `python -m streamlit run scripts/eval_webui.py --server.address 127.0.0.1 --server.port 8503`，或在 Windows 双击 `start_eval_webui.bat`。
浏览器打开 `http://localhost:8503`，可配置 HumanEval、LiveCodeBench、GSM8K、推理自动测试和 ToolCall，启动/停止评测、查看日志并下载结果。详细说明见 [评测 WebUI 文档](docs/eval_webui.md)。

评测也可以直接用命令行脚本:`eval_gsm8k.py`、`eval_humaneval.py`、`eval_pass_k.py`
(离线 pass@K 计算),批量生成复用 `eval_batch.py`,报告汇总用 `eval_report.py`;
产物统一写入 `eval/`。

### LiveCodeBench 代码生成评测

`eval_llm.py` 可直接读取官方 `code_generation_lite` 数据，按官方通用提示生成代码，
并输出可交给 LiveCodeBench `custom_evaluator` 的 JSON：

```bash
# 先用 10 题做生成 smoke test；--lcb_limit 不能用于官方评分
python eval_llm.py --benchmark livecodebench --weight full_sft \
  --lcb_release_version release_v6 --lcb_limit 10 \
  --temperature 0.2 --max_new_tokens 2048

# 完整生成；官方标准常用每题 n=10、temperature=0.2
python eval_llm.py --benchmark livecodebench --weight full_sft \
  --lcb_release_version release_v6 --lcb_num_samples 10 \
  --temperature 0.2 --max_new_tokens 2048 \
  --lcb_output eval/livecodebench_release_v6.json
```

输出会在每完成一题后原子保存，默认自动续跑。若 Hugging Face 不可用，可用
`--lcb_dataset_path` 指向本地 JSON/JSONL。官方数据集仍使用加载脚本，因此请使用
`requirements.txt` 固定的 `datasets==3.6.0`。

计算 pass@k 需安装 [LiveCodeBench 官方仓库](https://github.com/LiveCodeBench/LiveCodeBench)，
再显式启用 evaluator：

```bash
python eval_llm.py --benchmark livecodebench --weight full_sft \
  --lcb_release_version release_v6 --lcb_num_samples 10 \
  --temperature 0.2 --max_new_tokens 2048 \
  --lcb_output eval/livecodebench_release_v6.json \
  --lcb_runner_path ../LiveCodeBench --lcb_evaluate

# 已有生成文件时，只运行评分，不加载模型
python eval_llm.py --lcb_evaluate_only \
  --lcb_output eval/livecodebench_release_v6.json \
  --lcb_release_version release_v6 --lcb_runner_path ../LiveCodeBench
```

官方 evaluator 会执行模型生成的 Python 程序，建议在隔离容器或专用评测环境中运行。

---

## 训练管线

所有训练脚本在仓库根目录运行(无需进入 `trainer/` 目录):

### 1. 预训练(Pretrain,必须)

```bash
# 单卡
python trainer/train_pretrain.py

# 多卡 DDP
torchrun --nproc_per_node N trainer/train_pretrain.py
```

输出权重:`out/pretrain_{hidden_size}{_moe}.pth`,MoE 配置自动加 `_moe` 后缀
(当前 MoE 配置下为 `out/pretrain_512_moe.pth`)。

### 2. 继续预训练(CPT,可选)

Config WebUI 的 `cpt (continual pretraining)` 模式复用 `train_pretrain.py` 的
next-token 目标，但要求加载已有 `pretrain_*` 或 `cpt_*` 权重。它自动发现
`pretrain*.jsonl`，优先选择 `dataset/pretrain_continue.jsonl`，默认 1 epoch、
`3e-5` 峰值学习率、前 3% micro-steps 线性 warmup，以及 4096 token 的自适应
Bucket packing(`muon` 优化器、FP32 主权重 + BF16 激活 + FP8 tensorwise GEMM、
有界流式加载),并将输出保存为独立的 `cpt_*` 权重。
CPT 完成后再从 CPT 权重执行 SFT。
该 re-warm/re-decay 形式参考 [Continual Pre-Training of Large Language Models: How to (re)warm your model?](https://arxiv.org/abs/2308.04014)。

### 3. 指令微调(SFT,必须)

```bash
python trainer/train_full_sft.py
```

SFT 必须基于预训练权重(`--from_weight pretrain`)。输出:`out/full_sft_{hidden_size}{_moe}.pth`。

Config WebUI 为 SFT 提供三套独立配置档案:`base`(从 pretrain 开始)、
`continue`(用已完成的 full_sft 权重在新数据上继续,低学习率 + Bucket packing)
与 `resume`(恢复中断训练的完整 checkpoint,不套用新默认)。下面是 Bucket
模式示例；其中 `batch_size` 和 `max_seq_len` 仅作为 non-packed/fixed 回退与 checkpoint
迁移参数，进入 Bucket 后实际 batch 由桶长度和 `bucket_gpu_memory_gb` 自动决定：

```powershell
python trainer/train_full_sft.py --config_path trainer/config_full_sft.json --batch_size 12 --max_seq_len 768 --sequence_packing 1 --sequence_packing_mode bucket --seq_bucket 2 --bucket_gpu_memory_gb 16 --accumulation_steps 1 --optimizer muon --dtype bfloat16 --param_dtype fp32 --fp8_training tensorwise --fp8_filter auto --use_compile 1 --compile_mode max-autotune --use_grad_checkpoint 1
```

SFT 的学习率较低，建议保留 FP32 master weights（`--param_dtype fp32`），
同时使用 BF16 activation 与 Tensorwise FP8 GEMM。首次 `max-autotune` 编译会明显较慢，
后续运行复用 `./.cache/torch_compile/` 中的编译缓存。

### MoE 路由迁移(可选)

`--moe_router_top_k` / `--moe_router_norm_topk_prob` 允许在 pretrain / CPT / full_sft
之间迁移 MoE 路由语义(例如从 top-1 迁移到 top-k),仅这三个训练器注册该参数。
迁移训练不能与 `--from_resume 1` 同用;top-1 会自动关闭概率归一化。
Config WebUI 提供对应控件并在不兼容组合时拒绝启动。

### 4. 进阶训练(可选)

| 阶段 | 脚本 | 说明 |
|------|------|------|
| LoRA | `train_lora.py` | 低秩微调,CPU 亦可跑,适合垂直领域适配 |
| DPO | `train_dpo.py` | 基于人类偏好对,离线偏好优化 |
| PPO | `train_ppo.py` | Actor-Critic + GAE 强化学习 |
| GRPO / CISPO | `train_grpo.py` | 分组相对策略优化,`--loss_type cispo` 切换 |
| Agentic RL | `train_agent.py` | 多轮 Tool-Use 场景,支持 torch / sglang rollout |
| 蒸馏 | `train_distillation.py` | 白盒蒸馏,CE + KL 混合损失 |
| Tokenizer | `train_tokenizer.py` | 自定义词表训练(一般不建议重训) |

`trainer/training_pipeline.py` 可把 Pretrain → SFT 等多阶段编排为一次运行(JSON plan,
`--validate-only` 预检);Config WebUI 内置流水线面板负责保存快照、导出/导入计划、
后台运行与逐阶段暂停。

### 断点续训

所有训练脚本支持检查点恢复:

```bash
python trainer/train_pretrain.py --from_resume 1
```

推理权重保存在 `./out/`,续训检查点保存在 `./checkpoints/`,命名
`<权重名>_<维度>{_moe}_resume.pth`,原子写入,跨 GPU 数量变化亦可恢复(流式
预训练续训要求相同 GPU 数)。

训练过程中也可以请求安全暂停。训练器会完成当前 step，原子保存普通权重和完整 resume
checkpoint，再以专用退出码 42 结束：

```powershell
# CLI：在仓库根目录创建暂停标记
New-Item checkpoints/.pause_request -ItemType File

# 之后恢复
python trainer/train_pretrain.py --from_resume 1
```

Config WebUI 可直接点击 **Pause Training**；页面刷新后会重新扫描训练进程、日志、暂停状态
与最终 checkpoint，从而区分 running / paused / success / failed。另有
`scripts/monitor_training_shutdown.ps1` 可在训练正常退出(退出码 0)后自动关机,
暂停(42)或失败时保持开机。

### 常用训练参数

| 参数 | 说明 |
|------|------|
| `--max_seq_len` | 最大截断长度(单位 token;中文约 1.5~1.7 字符/token)。轻量数据建议 768 |
| `--dtype` / `--param_dtype` | 激活精度(bfloat16/float16/fp32) / 参数精度(fp32=FP32 主权重,bf16/fp16=直接 cast) |
| `--kv_cache_dtype` | KV cache 精度:`fp32`(默认)/`bf16`/`fp16`/`fp8_e4m3`/`fp8_e5m2`(按 batch×head 分片量化,影响生成/RL rollout) |
| `--fp8_training` | TorchAO FP8 训练:`off`/`tensorwise`/`rowwise`/`rowwise_with_gw_hp`,见注意事项 |
| `--sequence_packing 0\|1` | Pretrain/SFT/LoRA/蒸馏启用完整样本 packing，减少 padding（别名 `--packing`） |
| `--sequence_packing_mode fixed\|bucket` | `fixed`=固定长度（默认、兼容旧流程）；`bucket`=实验性自适应长度桶 |
| `--seq_bucket` | Bucket 模式的桶数量，默认 2 |
| `--bucket_gpu_memory_gb` | Bucket 模式可用的单卡显存（GB），同时是分配器硬限制，默认 16 |
| `--bucket_max_seq_len` | Bucket 模式允许的最大单样本长度，默认 16384；SFT 超限样本整条丢弃 |
| `--bucket_large_threshold` | 大桶优先训练阈值，默认 8192 token；超阈值桶阶段结束后释放专属 GPU 内存 |
| `--packing_batch_size` | 首次构建 packing Arrow cache 时每批处理的原始样本数，默认 1000 |
| `--packing_num_proc` | token 统计和 packing 预处理进程数；0=自动，Windows 最多 4 |
| `--bucket_loader_workers` | Bucket 训练 DataLoader 进程数；-1=自动，Windows 自动为 0 以降低提交内存 |
| `--dataset_streaming auto\|on\|off` | 有界流式预训练:auto=语料 Arrow 体积超过 `--data_cache_max_gb` 预算才启用 |
| `--streaming_chunk_mb` | 流式分块大小,默认 1024 MB;编译 Parquet 时用 `--align-chunk-bytes` 对齐 |
| `--data_cache_max_gb` | 数据集 Arrow 缓存的磁盘 LRU 预算,默认 5 GiB |
| `--moe_router_top_k` / `--moe_router_norm_topk_prob` | MoE 路由迁移(见上节),不能与 `--from_resume 1` 同用 |
| `--use_compile 0\|1` | 启用 `torch.compile`；编译产物默认持久化在 `./.cache/torch_compile/` |
| `--compile_mode` | `default` / `reduce-overhead` / `max-autotune` / `max-autotune-no-cudagraphs` |
| `--use_moe 1` | 启用 MoE 架构 |
| `--use_looped 1` | 启用 Instinct V2 latent recurrent-depth 架构 |
| `--loop_iters` / `--mean_backprop_depth` | 目标平均递归次数 / 保留梯度的最后递归次数 |
| `--use_grad_checkpoint 0\|1\|2` | 梯度检查点(0=关闭, 1=选择性重算注意力QKᵀ/FFN, 2=整层checkpoint) |
| `--hidden_size` / `--num_hidden_layers` | 模型宽度 / 深度 |
| `--use_wandb` | 开启训练日志(默认 SwanLab,兼容 WandB 接口) |
| `--from_weight` | 基于哪个权重继续训练(`none` = 从头) |
| `--optimizer` | `adamw` / `adafactor` / `muon` |
| `--profile off\|timing\|torch` | 训练性能分析：低开销阶段计时或官方 PyTorch/Kineto trace |
| `--config_path` | 读取 JSON 训练配置 |

---

## 项目结构

```text
model/                # InstinctConfig / InstinctForCausalLM / LoRA / looped / linear 架构
                      #   + 推理运行时(inference_runtime, static_cache, grouped_mm, kv_cache_quant) + tokenizer 文件
trainer/              # 全部训练脚本(pretrain, SFT, LoRA, DPO, PPO, GRPO, Agent RL, KD, tokenizer)
                      #   + trainer_cli.py 共享 CLI / training_pipeline.py 流水线 / training_profiler.py
dataset/              # 数据说明文档(dataset.md);语料本体不入库,单独下载
dataset_compiler/     # Rust 工具:JSONL -> Parquet 语料编译(见其 README.md)
scripts/              # 推理、API 服务、Chat/Config/Eval WebUI、模型转换
scripts/data_loader/  # 训练侧数据加载:格式解析、packing、流式分块、缓存预算
scripts/data_builder/ # 离线数据收集/构建/清洗/审计工具(见其 README.md)
eval_llm.py           # CLI 推理 + LiveCodeBench 评测入口
eval_gsm8k.py / eval_humaneval.py / eval_pass_k.py / eval_batch.py / eval_report.py   # 评测 harness
eval/                 # 评测产物与运行记录(eval/runs/<时间戳>_<ID>/)
tests/                # pytest 套件;gpu 标记在无 CUDA 时自动跳过
docs/                 # 设计文档:training_pipeline / compact_dataset_cache / gsm8k / humaneval / eval_webui / grouped_moe_gemm ...
experiments/          # 性能/研究实验脚本(lambda sweep、分组 GEMM 基准、graph break 报告等)
```

---

## 模型架构

主线结构对齐 Qwen3 生态:Pre-Norm + RMSNorm、SwiGLU、RoPE、GQA。

仓库随附的 trainer 配置(`trainer/*.json`)当前为:

| 配置文件 | 架构 | 关键参数 |
|----------|------|----------|
| `config_pretrain.json` / `config_instinct_v1_moe.json` | Instinct V1 MoE | hidden 512、32 层、8 个 FFN=1664 专家 top-1 路由、16Q/4KV GQA(head_dim 32)、RoPE θ=1e6 + LongRoPE factor 8.0(原始 4096)、max_position 32768、tied embeddings |
| `config_full_sft.json`(= `config_instinct_v2.json` / `config_dpo.json`) | Instinct V2 looped | hidden 1248、物理层 (Prelude 2 / Core 4 / Coda 2)、13Q/13KV MHA(head_dim 96)、FFN 4224、`loop_iters 32`、`mean_backprop_depth 8`、RoPE θ=5e4 |

| 模型 | 参数量 | 说明 |
|------|--------|------|
| Instinct V1 | 152.4M | Dense，dim 768，20 层，8Q/4KV GQA；见 [architecture-v1.html](architecture-v1.html) |
| Instinct V1 MoE | 678.7M-A106.2M | 16GB + Muon deep-thin：hidden 512、32 层、16Q/4KV、8 个 FFN=1664 专家，top-1；见 [architecture-v1-moe.html](architecture-v1-moe.html) |
| Instinct V2 | 187.5M | latent recurrent depth，物理层 `(2,4,2)`，平均有效深度 132；见 [architecture.html](architecture.html) |

Instinct V1 Dense 的总参数和每 token 激活参数均为 152,406,528（约 152.4M）；V1 MoE 为 678,726,144 总参数 / 106,203,648 激活参数。V1 Dense 使用 20 层、hidden=768、FFN=2432，不能与 `InstinctConfig` 默认的 8 层旧版 `instinct-3` 混淆；加载和评测应以权重对应的配置为准。按训练者提供的信息，V1 MoE 使用了 34GB 数据集，V1 Dense 则使用两个 mini 数据集分别进行预训练和 SFT；文件大小不等于实际训练 token 数或代码数据量。

Instinct V2 参考 [Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171)：Prelude 将 token 映射到隐空间，共享的多层 recurrent core 每轮通过 `Linear([state; input])` 重新注入输入，Coda 解码最终状态。训练时递归次数使用 log-normal Poisson 采样，并只对最后 `k` 轮反传；推理可用 `eval_llm.py --model_architecture looped --num_steps 32` 增加隐空间计算。200M 级 V2 默认为 187,521,984 参数：`hidden=1248`、13 个 96 维 MHA heads、`FFN=4224`、物理层 `(Prelude, Core, Coda)=(2,4,2)`，平均递归 32 次（平均有效深度 132），只对最后 8 次递归保留梯度。另有 linear attention 主干(`model/model_instinct_linear.py`,经 `run_linear.py` 包装训练)。

---

## 部署

### 解码快路径(Chat WebUI / eval_llm)

推理侧 `optimize_inference()` 提供三档(环境变量 `INSTINCT_INFERENCE_COMPILE`,
Chat WebUI 默认 `auto`):

- **auto**: 加载保持瞬时——不编译主干,只装融合 RMSNorm/SwiGLU 内核,解码步被录制为
  一张 CUDA graph 复用(写入预分配的静态 KV cache)。512 维 MoE 实测约 3.7 ms/token
  (约 268 tok/s),dense 768 约 1.9 ms/token。
- **full**: 整个 Transformer 主干 `torch.compile`,约 2 倍 prefill、约 30% 解码吞吐,
  代价是加载时 ~50s 编译 + 首个真实 prompt 再编译一次。
- **off**: 纯 eager。

Windows 上启用编译必须 `PYTHONUTF8=1`(启动脚本已注入),否则会打印提示并退回 eager。
MoE 专家分组的 GEMM 后端由 `INSTINCT_GROUPED_MM_BACKEND` 控制(`auto`/`native`/`triton`/`cached`):
`auto` 在非 Windows SM90/SM100 用 native,否则装了 Triton 用 triton,再退 `cached`。
`cached` 后端每层做一次 `offsets.cpu()` 宿主同步且拒绝 CUDA graph 捕获——Windows 上
没有 `triton-windows` 时聊天解码实测只有约 8 tok/s,安装后恢复约 160+ tok/s(RTX 5060 Ti 8GB 实测)。
回归排查用 `python experiments/graph_break_report.py`(期望 graph count: 1),吞吐 A/B 用
`experiments/bench_chat_moe_decode.py`,加载分解用 `experiments/check_load_time.py`。
详见 [`docs/grouped_moe_gemm.md`](docs/grouped_moe_gemm.md)。

静态 KV cache 按模型精度存储;配置了 FP8 KV cache 时该快路径会打印提示并保持模型精度
(避免每步反量化)。批量评测另有 `--eval_kv_cache_dtype auto|configured` 策略。

### OpenAI 兼容 API

```bash
cd scripts && python serve_openai_api.py
# 端点: http://localhost:8998/v1/chat/completions
```

支持流式 SSE、`reasoning_content`、`tool_calls`、`open_thinking`(顶层字段或
`chat_template_kwargs.open_thinking`) 字段,可接入 FastGPT、Open-WebUI、Dify 等。
LongRoPE 权重需通过 `--config_path` 加载缩放向量。

### WebUI

Windows 可从仓库根目录直接启动三个界面：

```powershell
# Chat WebUI：http://localhost:8502
.\start_chat_webui.bat

# 训练配置 WebUI：http://localhost:8500
.\start_config_webui.bat

# 统一评测 WebUI：http://localhost:8503
.\start_eval_webui.bat
```

Chat WebUI 会自动扫描 `out/` 和 `checkpoints/` 下的原生 `.pth` 权重(排除 `_resume`),
并按 权重同名 JSON → `checkpoints/` 同名 JSON → 默认配置 的顺序解析架构;侧栏实时显示
Dense/MoE · 层数 · hidden · 头数 · 专家数摘要与解码吞吐(tokens/s,不含预填充)。界面提供
新对话、重新生成最后回复、每条回复复制按钮、中英切换、历史轮次/温度/重复性惩罚/最大
输出长度调节、思考模式(`<think>` 渲染为可折叠区域)、工具调用(8 个内置工具,最多选 4,
事件以卡片展示参数与结果)与 Logit Lens(逐层 Top-1 热力表,挂在真实生成上);流式输出使用
Markdown 渲染 + 打字机动画,图标为本地内联 SVG。Transformers 格式模型请通过
`eval_llm.py` 或 API 服务加载。

Config WebUI 会按文件名前缀扫描 `dataset/` 顶层 JSONL/gzip/Parquet，并只为当前训练器
显示兼容数据；提供 pretrain / cpt / full_sft / lora / dpo / ppo / grpo / agent /
distillation 九种训练模式与 SFT 三套启动档案、fixed / Bucket packing 选择、显存感知
batch 配置、MoE 路由迁移控件、FP8/compile 校验、参数量实时估算、训练日志跟踪以及
可恢复的暂停/续训状态。

### 模型转换

```bash
cd scripts && python convert_model.py
# torch (.pth) <-> transformers 格式互转,支持 LoRA 权重合并
```

除 Qwen3/Qwen3Moe 兼容格式外,还支持保留原生 `InstinctConfig` 的 auto-load 导出
(LongRoPE 语义精确保留;Qwen3 运行时不支持 LongRoPE 的部分语义时会拒绝导出)。

### 第三方推理框架

- **llama.cpp**: 转换 GGUF 后使用(需在 `convert_hf_to_gguf.py` 中补充 Instinct tokenizer 映射,可临时复用 `qwen2`)
- **vllm**: `vllm serve /path/to/model --served-model-name "instinct"`
- **ollama**: 通过 GGUF 文件创建本地模型,或 `ollama run 1057237562/instinct-3`

---

## 数据集格式

**预训练**(每行一个 JSON 对象):

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
```

**SFT / RL / LoRA**(OpenAI 对话格式):

```jsonl
{"conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
]}
```

**工具调用**(嵌入在 conversations 中):

```jsonl
{"conversations": [
    {"role": "system", "content": "# Tools ...", "tools": "[...]"},
    {"role": "user", "content": "帮我算一下 256 乘以 37"},
    {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"256 * 37\"}}]"},
    {"role": "tool", "content": "{\"result\":\"9472\"}"},
    {"role": "assistant", "content": "256 乘以 37 等于 9472。"}
]}
```

**DPO 偏好数据**:

```json
{"chosen": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "good"}],
 "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "bad"}]}
```

**Agent RL**: 每行 `{"conversations": [...], "gt": [...]}`,`gt` 为 ground-truth 命中
目标列表。以上格式均可被 Rust 编译器编译为 Parquet(对话列 `list<struct>`,`tools`/
`tool_calls` 保持 JSON 文本,任何行都不会因形状被丢弃,详见
[`dataset_compiler/README.md`](./dataset_compiler/README.md))。

---

## 测试

```bash
python -m pytest tests/             # 从仓库根目录
python -m pytest tests/ --skip-gpu  # 强制跳过 CUDA 测试
```

- 标记 `@pytest.mark.gpu` 的测试在无 CUDA 环境自动跳过;`slow` 测试需 `--run-slow` 显式开启。
- `tests/conftest.py` 强制 `datasets` 先于 `torch` 导入(Windows pyarrow/torch DLL 冲突
  的 workaround),与训练器一致,勿调整顺序。

---

## 注意事项

### 工作目录与平台

- **工作目录**: 训练脚本在仓库根目录运行,数据默认指向 `./dataset/`,权重输出到 `./out/`;API / WebUI 需在 `scripts/` 下运行
- **Windows**: 训练脚本先 import `datasets` 再 import `torch`,以规避 pyarrow/torch DLL 冲突,勿调整顺序;`torch.compile` 需要 `PYTHONUTF8=1`(WebUI 启动脚本已注入)
- **MoE 后端**: Windows 上 MoE grouped-GEMM 依赖 `triton-windows`(已在 requirements.txt);后端可用 `INSTINCT_GROUPED_MM_BACKEND` 强制为 `native`/`triton`/`cached`
- **日志工具**: WandB 在国内常不可直连,默认使用 SwanLab(API 兼容),需要时加 `--use_wandb`
- **奖励模型**: PPO / GRPO 使用的 `internlm2-1_8b-reward` 需放在仓库同级目录(非仓库内)
- **max_seq_len 是 token 数**: 中文约 1.5~1.7 字符/token,英文 4~5 字符/token,按数据分布调整
- **梯度检查点**: 选择性重算(1)在 eager 路径生效,flash 路径下仅省 FFN 中间量;序列越长(seq 大)节省显存收益越大

### TorchAO FP8 训练（可选）

```bash
python -m pip install -r requirements-fp8.txt
python trainer/train_pretrain.py --dtype bfloat16 --param_dtype bf16 \
  --use_compile 1 --fp8_training tensorwise --fp8_filter auto
```

- `tensorwise` 速度优先；`rowwise` 数值精度优先；`rowwise_with_gw_hp` 保留高精度权重梯度计算。
- 启动时会执行一次真实 FP8 前向/反向探测；当前设备不支持 rowwise 时自动回退到 tensorwise。
- `auto` 只转换预计有收益的 Linear；`eligible` 转换所有维度为 16 倍数的兼容 Linear。
- `lm_head`、LoRA adapter、Embedding、Norm、Attention softmax、残差拓扑以及优化器状态保持 BF16/FP32。
- FP8 包装不改变 state_dict 键名，普通权重、暂停检查点和恢复训练可在 FP8/BF16 间切换。
- Full SFT / DPO / PPO 等低学习率阶段应使用 `--param_dtype fp32` 保存可更新的主权重；
  `--dtype bfloat16` 与 TorchAO FP8 GEMM 仍然有效。直接更新 BF16/FP16 参数时，低于其量化间隔的
  optimizer update 会被舍入为零，训练器会对此类危险组合提前报错。

### 训练性能分析

持续比较 BF16 / FP8 时使用低开销 timing 模式：

```bash
python trainer/train_pretrain.py --profile timing \
  --profile_warmup 10 --profile_interval 100
```

日志中的 `[PROFILE]` 会报告真实 step 时间、物理/有效 tokens/s、前向、反向、优化器、
数据传输、step 间 host gap 和 CUDA 峰值显存。统计使用 `torch.cuda.Event`，仅在汇总间隔同步一次。

需要查看具体算子和 kernel 时，使用官方 `torch.profiler` 短窗口 trace：

```bash
python trainer/train_pretrain.py --profile torch \
  --profile_warmup 10 --profile_active_steps 5
```

trace 默认写入 `./profiler_traces/*.pt.trace.json`，可用 Chrome trace viewer、
Perfetto 或 TensorBoard Profiler 打开。`torch` 模式的采集窗口开销较高，不建议增加
`profile_active_steps` 后长期采集；窗口结束后仍保留低开销 timing 汇总。

### Pretrain / SFT / LoRA / 蒸馏 Sequence Packing

Packing 保证一条完整样本不会跨 block。Pretrain 保留每篇文档的 BOS/EOS；SFT、LoRA
和蒸馏同步保留 assistant-only `-100` loss mask。首次启动会按 `--packing_batch_size`
分批 tokenization，并在 Hugging Face cache 中生成 Arrow blocks；数据、tokenizer、packing
算法和桶边界等配置不变时，后续启动会复用 cache。

#### 固定长度模式（默认、稳定）

```bash
python trainer/train_pretrain.py --sequence_packing 1 \
  --sequence_packing_mode fixed --max_seq_len 1024

python trainer/train_full_sft.py --sequence_packing 1 \
  --sequence_packing_mode fixed --max_seq_len 768
```

所有 block 都使用 `--max_seq_len`，训练 batch size 使用 `--batch_size`。该模式保持原有
packing 行为，适合继续旧数据集和旧 fixed-packing checkpoint。

#### 自适应长度桶（实验性）

```bash
python trainer/train_full_sft.py --sequence_packing 1 \
  --sequence_packing_mode bucket --seq_bucket 3 \
  --bucket_gpu_memory_gb 16 --bucket_max_seq_len 16384 \
  --packing_num_proc 4 --bucket_loader_workers 0 \
  --use_compile 1 --compile_mode reduce-overhead
```

WebUI 的 Sequence Packing 选项中选择 **Bucket（实验性）** 后，可直接填写桶数量、单卡
显存、最大样本长度、预处理进程数和 DataLoader 进程数；Bucket 模式会接管 batch size，
无需手动填写每个桶的 batch。

Bucket 模式的流程如下：

1. 对样本 token 长度按 16 token 对齐（兼容 TorchAO FP8 scaled GEMM 的维度要求）并聚合
   直方图。例如 905,718 条数据可以缩减为约 153 个长度组，DP 不会直接在 905,718 个样本上运行。
2. 对每个候选区间用分组 Best-Fit Decreasing（BFD）估算 packed block 数；长度组超过
   256 时使用 token 体积与长样本数量下界，控制搜索内存和耗时。
3. 根据填写的显存计算 token budget。默认标定点是 16GB、`2048 × batch 12 = 24576`
   tokens，因此每桶 batch size 近似为 `floor(token_budget / bucket_max_seq_len)`，显存按
   GB 线性缩放。该值来自当前模型的实测标定，更换模型结构、精度、优化器或显卡后应预留余量。
4. DP 最小化预测的 epoch wall time，而不是只最大化填充率。单 step 成本包含固定开销、
   近似 `O(BL)` 的线性层成本和 `O(BL²)` 的注意力成本；初始时间标定点为
   `batch=28, seq=1024, step=0.56s`，其中固定开销 0.03s、注意力占可缩放部分的 20%。
5. 确定桶边界后，对每个桶按 `packing_seed + bucket_max_seq_len` 做确定性混排，再分块执行
   BFD，避免长度排序后每个 packing chunk 只包含相近长度。相同数据、seed 和配置会得到相同结果。

样本根据长度确定性地归属某个桶，不会随机“进桶”。训练时每个 step 只取同一个桶的
同形状 block，桶内 batch 和大桶/普通桶阶段分别做确定性 shuffle；每桶最后一个 batch
可能小于规划 batch size，但不会跨桶拼成一个 step。

DP 输出的是计算耗时最优解，不保证每个桶的填充率单独最高。启动训练前，日志会先打印
所有桶的 `max_seq_len`、预计 blocks、自动 batch size 和预计耗时，完成 packing 后再打印
实际 blocks 与填充率：

```text
[Packing DP] SFT: ... requested_buckets=3, cost_model=wall_time_bfd_v2
[Packing DP Bucket] SFT 1/3: max_seq_len=1024, estimated_blocks=..., batch_size=24, estimated_time=...
[Packing Bucket] SFT 1/3: max_seq_len=1024, raw_samples=..., packed_blocks=..., fill=...
```

训练第一次进入某个桶时会打印当前桶和实际 batch；周期日志也会包含桶编号、`L×B`、
预计剩余 `epoch_time` 和从本 epoch 开始累计的实际 `elapsed_time`：

```text
[Packing Bucket Active] epoch=1, step=1, bucket=1/3, max_seq_len=1024, batch_size=24 (planned=24)
Epoch:[1/2](100/...), ... bucket: 1/3 (1024x24), epoch_time: ...min, elapsed_time: ...min
```

超过 `--bucket_large_threshold` 的长序列桶会集中放在 epoch 开头训练；阶段结束后清理其
CUDA Graph/编译相关引用和 GPU cache，避免长桶的专属内存长期驻留。Windows 下建议保持
`--bucket_loader_workers 0`：`datasets.map` 的 packing 预处理仍可使用多进程，但训练期
DataLoader 子进程会重复加载 PyTorch 等模块，可能让每个 worker 额外占用大量提交内存。

#### `torch.compile` 持久化缓存与多桶编译

训练入口会在导入 PyTorch 前启用 FX Graph、AOTAutograd 和 Inductor 的磁盘缓存，默认目录为：

```text
./.cache/torch_compile/
```

可通过环境变量 `TORCHINDUCTOR_CACHE_DIR` 改为其他位置。Bucket 模式的每个 `L×B` 都是一个
独立 shape，因此启用 CUDA Graph/`torch.compile` 时，3 个桶通常需要分别完成 3 次冷编译；
缓存能让模型、PyTorch/Triton、编译参数和 shape 均未变化的后续启动快速复用产物，但不能
消除新 shape 的首次编译。`max-autotune` 对长桶的冷编译可能很慢且占用较多内存，优先从
`reduce-overhead` 或 `default` 开始验证。

#### Checkpoint 与 packing cache 兼容性

- 旧 fixed-packing checkpoint 继续使用相同 `fixed` 配置时可以正常续训。若日志再次显示
  packing，通常是在校验或重建 Arrow cache；只要数据、tokenizer 和 packing 参数未变，样本
  顺序与训练语义不变，但首次启动时间会增加。cache key 相关条件变化时必须重新 packing。
- 从 non-packed/fixed checkpoint 切换到 packing 时，会保留模型、优化器与 GradScaler，并
  还原当前 epoch 的 shuffle 顺序：未到 packing 分组边界的数据继续按原始 batch 训练，随后
  将同一 epoch 未训练的后缀构造成 packed blocks。迁移起点和游标会写入 checkpoint。
- Bucket checkpoint 的精确续训要求 packing-critical 配置一致，包括模式、桶数、桶边界算法、
  显存、最大长度和预处理配置。训练中途更换这些配置，或用新版桶算法读取旧版 Bucket
  checkpoint，不会静默重映射数据；应使用原配置/原代码续训，或在 epoch 边界完成迁移。

---

## License

Apache License 2.0。详见 [LICENSE](./LICENSE)。

## 致谢与引用

感谢开源社区与 MiniMind 原项目([jingyaogong/minimind](https://github.com/jingyaogong/minimind))的启发。

```bibtex
@misc{minimind,
  title = {MiniMind: Train a 64M-parameter LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind}
}
```
