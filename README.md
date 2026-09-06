# Instinct

从零训练一个约 123M 参数的小型语言模型。当前仓库配置已在单张 RTX 5070 Ti
16GB、Windows 11 环境完成预训练、SFT、断点续训与推理验证。

所有核心算法(Pretrain / SFT / LoRA / DPO / PPO / GRPO / CISPO / Agentic RL / 蒸馏)均用 PyTorch 原生实现,不依赖 `trl` / `peft` 等高层封装,每一行代码都可读、可改、可复现。

---

## 特性

- **单卡可训**: 当前 Dense 模型 122.9M 参数(vocab 6400, 16 层, dim 768),适配单张 5070 Ti 16GB
- **纯原生实现**: 无第三方训练框架抽象,从零手写 Transformer、LoRA、RL 算法
- **完整训练链路**: 预训练 → SFT → LoRA → DPO → PPO / GRPO / CISPO → Agentic RL → 蒸馏
- **双架构**: Dense 与 MoE(当前 16 层配置为 391.9M 总量 / 约 123.0M 激活,4 experts top-1 routing)
- **可扩展结构**: 支持循环深度架构(LoopUS)、Early Exit 动态推理、自蒸馏训练
- **自适应 Sequence Buckets**: 实验性按长度分桶 packing，使用 wall-time DP、显存感知 batch、确定性混排与持久化编译缓存降低 padding 和训练耗时
- **现代运行时兼容**: 已适配 Transformers 5 / huggingface-hub 1，支持配置、tokenizer 与 Safetensors 往返保存
- **低内存数据流水线**: 支持 SFT 数据标准化、流式 replay 抽样、磁盘分块混洗和 WebUI 自动发现数据集
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
`PYTHONUTF8=1`。

### 下载数据

将数据集放入 `./dataset/` 目录(推荐从 ModelScope 或 HuggingFace 单独下载所需文件,无需全部克隆):

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

### 准备与混合 SFT 数据

`scripts/prepare_sft_data.py` 可将 CodeAlpaca、SmolTalk、BigCode Exec、Magicoder
和 No Robots 等数据统一转换为 Instinct 的 `conversations` JSONL：

```bash
python scripts/prepare_sft_data.py codealpaca-local
python scripts/prepare_sft_data.py smol-smoltalk --max-samples 100000
python scripts/prepare_sft_data.py bigcode-exec-50k
```

混合大规模 Coding、Math 与原始 T2T replay 时可运行：

```bash
python scripts/mix_sft_datasets.py
```

该脚本用 reservoir sampling 流式抽取大型 T2T 数据，并通过磁盘分块外部混洗限制峰值
内存；默认生成 `dataset/sft_magicoder110k_mathinstruct_t2t_replay20.jsonl`。更多数据源、
配比、继续 SFT 数据构建和校验方式见 [`dataset/dataset.md`](./dataset/dataset.md)。所有输出
使用 `sft_` 前缀后，Config WebUI 会自动识别。

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
```

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
  --lcb_output out/livecodebench_release_v6.json
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
  --lcb_output out/livecodebench_release_v6.json \
  --lcb_runner_path ../LiveCodeBench --lcb_evaluate

# 已有生成文件时，只运行评分，不加载模型
python eval_llm.py --lcb_evaluate_only \
  --lcb_output out/livecodebench_release_v6.json \
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

输出权重:`out/pretrain_{hidden_size}.pth`(默认 768)。

### 2. 指令微调(SFT,必须)

```bash
python trainer/train_full_sft.py
```

SFT 必须基于预训练权重(`--from_weight pretrain`)。输出:`out/full_sft_{hidden_size}.pth`。

当前 5070 Ti 16GB 的 fixed 基准为 `max_seq_len=768, batch_size=12`。下面是 Bucket
模式示例；其中 `batch_size` 和 `max_seq_len` 仅作为 non-packed/fixed 回退与 checkpoint
迁移参数，进入 Bucket 后实际 batch 由桶长度和 `bucket_gpu_memory_gb` 自动决定：

```powershell
python trainer/train_full_sft.py --config_path trainer/config_full_sft.json --batch_size 12 --max_seq_len 768 --sequence_packing 1 --sequence_packing_mode bucket --seq_bucket 2 --bucket_gpu_memory_gb 16 --accumulation_steps 1 --optimizer muon --dtype bfloat16 --param_dtype fp32 --fp8_training tensorwise --fp8_filter auto --use_compile 1 --compile_mode max-autotune --use_grad_checkpoint 1
```

SFT 的学习率较低，当前配置保留 FP32 master weights（`--param_dtype fp32`），
同时使用 BF16 activation 与 Tensorwise FP8 GEMM。首次 `max-autotune` 编译会明显较慢，
后续运行复用 `./.cache/torch_compile/` 中的编译缓存。

### 3. 进阶训练(可选)

| 阶段 | 脚本 | 说明 |
|------|------|------|
| LoRA | `train_lora.py` | 低秩微调,CPU 亦可跑,适合垂直领域适配 |
| DPO | `train_dpo.py` | 基于人类偏好对,离线偏好优化 |
| PPO | `train_ppo.py` | Actor-Critic + GAE 强化学习 |
| GRPO / CISPO | `train_grpo.py` | 分组相对策略优化,`--loss_type cispo` 切换 |
| Agentic RL | `train_agent.py` | 多轮 Tool-Use 场景,支持 torch / sglang rollout |
| 蒸馏 | `train_distillation.py` | 白盒蒸馏,CE + KL 混合损失 |
| Tokenizer | `train_tokenizer.py` | 自定义词表训练(一般不建议重训) |

### 断点续训

所有训练脚本支持检查点恢复:

```bash
python trainer/train_pretrain.py --from_resume 1
```

检查点保存在 `./checkpoints/`,命名 `<权重名>_<维度>_resume.pth`,跨 GPU 数量变化亦可恢复。

训练过程中也可以请求安全暂停。训练器会完成当前 step，原子保存普通权重和完整 resume
checkpoint，再以专用退出码 42 结束：

```powershell
# CLI：在仓库根目录创建暂停标记
New-Item checkpoints/.pause_request -ItemType File

# 之后恢复
python trainer/train_pretrain.py --from_resume 1
```

Config WebUI 可直接点击 **Pause Training**；页面刷新后会重新扫描训练进程、日志、暂停状态
与最终 checkpoint，从而区分 running / paused / success / failed。SFT 另提供三种启动方式：
从 pretrain 开始、用已完成的 full_sft 权重在新数据上继续（重建 optimizer/scheduler/step），
或恢复中断训练的完整 checkpoint。

### 常用训练参数

| 参数 | 说明 |
|------|------|
| `--max_seq_len` | 最大截断长度(单位 token;中文约 1.5~1.7 字符/token)。轻量数据建议 768 |
| `--sequence_packing 0\|1` | Pretrain/SFT/LoRA/蒸馏启用完整样本 packing，减少 padding（别名 `--packing`） |
| `--sequence_packing_mode fixed\|bucket` | `fixed`=固定长度（默认、兼容旧流程）；`bucket`=实验性自适应长度桶 |
| `--seq_bucket` | Bucket 模式的桶数量，默认 2 |
| `--bucket_gpu_memory_gb` | Bucket 模式可用的单卡显存（GB），用于自动计算每桶 batch size，默认 16 |
| `--bucket_max_seq_len` | Bucket 模式允许的最大单样本长度，默认 16384；SFT 超限样本整条丢弃 |
| `--bucket_large_threshold` | 大桶优先训练阈值，默认 8192 token；超阈值桶阶段结束后释放专属 GPU 内存 |
| `--packing_batch_size` | 首次构建 packing Arrow cache 时每批处理的原始样本数，默认 1000 |
| `--packing_num_proc` | token 统计和 packing 预处理进程数；0=自动，Windows 最多 4 |
| `--bucket_loader_workers` | Bucket 训练 DataLoader 进程数；-1=自动，Windows 自动为 0 以降低提交内存 |
| `--use_compile 0\|1` | 启用 `torch.compile`；编译产物默认持久化在 `./.cache/torch_compile/` |
| `--compile_mode` | `default` / `reduce-overhead` / `max-autotune` / `max-autotune-no-cudagraphs` |
| `--use_moe 1` | 启用 MoE 架构 |
| `--use_looped 1` | 启用循环深度架构(LoopUS) |
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
model/          # InstinctConfig / InstinctForCausalLM / LoRA / 循环架构 / tokenizer 文件
trainer/        # 全部训练脚本(pretrain, SFT, LoRA, DPO, PPO, GRPO, Agent RL, KD, tokenizer)
dataset/        # 数据集加载类(PretrainDataset, SFTDataset 等)+ 数据文件
scripts/        # 推理、API 服务、WebUI、模型转换
eval_llm.py     # CLI 推理入口
```

---

## 模型架构

主线结构对齐 Qwen3 生态:Pre-Norm + RMSNorm、SwiGLU、RoPE(θ=1e6)、GQA(q_heads=8, kv_heads=4)。

`trainer/config_pretrain.json` 与 `trainer/config_full_sft.json` 当前保持一致：

| 配置项 | 当前值 |
|--------|--------|
| 架构 | Dense Transformer (`standard`) |
| 参数量 | 122,908,416（约 122.9M） |
| 词表 / 隐藏维度 / FFN 维度 | 6400 / 768 / 2432 |
| 层数 | 16 |
| Attention | 8 query heads / 4 KV heads / head dim 96 |
| 最大位置长度 | 32768 |
| RoPE | θ=1,000,000，YaRN factor=16，原始长度 2048 |
| 其他 | tied embeddings、SwiGLU、RMSNorm、dropout=0 |

| 模型 | 参数量 | 说明 |
|------|--------|------|
| instinct-3 | 122.9M | 当前 Dense 主线(dim 768, 16 层, max_pos 32768) |
| instinct-3-moe | 391.9M-A123.0M | 当前 16 层配置，4 experts / top-1 routing |

循环深度架构(LoopUS)可在固定参数下增加有效深度,配合 Early Exit 在推理时按 token 动态选择退出层,以控制推理成本。

---

## 部署

### OpenAI 兼容 API

```bash
cd scripts && python serve_openai_api.py
# 端点: http://localhost:8998/v1/chat/completions
```

支持 `reasoning_content`、`tool_calls`、`open_thinking` 字段,可接入 FastGPT、Open-WebUI、Dify 等。

### WebUI

Windows 可从仓库根目录直接启动两个界面：

```powershell
# Chat WebUI：http://localhost:8502
.\start_chat_webui.bat

# 训练配置 WebUI：http://localhost:8500
.\start_config_webui.bat
```

Chat WebUI 会自动扫描 `out/` 和 `checkpoints/` 下的原生 `.pth` 权重，并解析对应
JSON 配置。界面提供新对话、重新生成最后回复、重复性惩罚、思考模式、工具调用
与 Logit Lens；Transformers 格式模型请通过 `eval_llm.py` 或 API 服务加载。

Config WebUI 会按文件名前缀扫描 `dataset/` 顶层 JSONL，并只为当前训练器显示兼容数据；
同时提供 fixed / Bucket packing 选择、显存感知 batch 配置、预处理/加载 worker 设置、
FP8/compile 校验、训练日志跟踪以及可恢复的暂停/续训状态。

### 模型转换

```bash
cd scripts && python convert_model.py
# torch (.pth) <-> transformers 格式互转,支持 LoRA 权重合并
```

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

---

## 注意事项

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

- **工作目录**: 训练脚本在仓库根目录运行,数据默认指向 `./dataset/`,权重输出到 `./out/`;API / WebUI 需在 `scripts/` 下运行
- **Windows**: 训练脚本先 import `datasets` 再 import `torch`,以规避 pyarrow/torch DLL 冲突,勿调整顺序
- **日志工具**: WandB 在国内常不可直连,默认使用 SwanLab(API 兼容),需要时加 `--use_wandb`
- **奖励模型**: PPO / GRPO 使用的 `internlm2-1_8b-reward` 需放在仓库同级目录(非仓库内)
- **max_seq_len 是 token 数**: 中文约 1.5~1.7 字符/token,英文 4~5 字符/token,按数据分布调整
- **梯度检查点**: 选择性重算(1)在 eager 路径生效,flash 路径下仅省 FFN 中间量;序列越长(seq 大)节省显存收益越大

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
