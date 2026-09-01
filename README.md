# Instinct

从零训练一个 64M 参数的小型语言模型:单张 RTX 3090、约 2 小时、成本几块钱。

所有核心算法(Pretrain / SFT / LoRA / DPO / PPO / GRPO / CISPO / Agentic RL / 蒸馏)均用 PyTorch 原生实现,不依赖 `trl` / `peft` 等高层封装,每一行代码都可读、可改、可复现。

---

## 特性

- **极致轻量**: 主线 Dense 模型 64M 参数(vocab 6400, 8 层, dim 768),单卡 3090 即可训练
- **纯原生实现**: 无第三方训练框架抽象,从零手写 Transformer、LoRA、RL 算法
- **完整训练链路**: 预训练 → SFT → LoRA → DPO → PPO / GRPO / CISPO → Agentic RL → 蒸馏
- **双架构**: Dense 与 MoE(198M 总量 / 64M 激活,4 experts top-1 routing)
- **可扩展结构**: 支持循环深度架构(LoopUS)、Early Exit 动态推理、自蒸馏训练
- **生态兼容**: 权重可转换为 HuggingFace 格式,支持 `llama.cpp` / `vllm` / `ollama` / `sglang`
- **中文友好**: 自带 6400 词表 BPE tokenizer,支持工具调用(`<tool_call>`)与思考(`<think>`)标签

---

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

需要 PyTorch 与 CUDA 环境(训练 GPU 建议 3090 或以上;无 GPU 也可 CPU 训练,但速度较慢)。

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

### 常用训练参数

| 参数 | 说明 |
|------|------|
| `--max_seq_len` | 最大截断长度(单位 token;中文约 1.5~1.7 字符/token)。轻量数据建议 768 |
| `--sequence_packing 0\|1` | Pretrain/SFT 固定 block packing，减少 padding（别名 `--packing`） |
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

| 模型 | 参数量 | 说明 |
|------|--------|------|
| instinct-3 | 64M | Dense 主线(dim 768, 8 层, max_pos 32768) |
| instinct-3-moe | 198M-A64M | 4 experts / top-1 routing |

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

```bash
# 先将 transformers 格式模型复制到 scripts/ 下
cp -r instinct-3 ./scripts/instinct-3
cd scripts && streamlit run web_demo.py
```

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

### Pretrain / SFT Sequence Packing

```bash
python trainer/train_pretrain.py --sequence_packing 1 --max_seq_len 1024
python trainer/train_full_sft.py --sequence_packing 1 --max_seq_len 768
```

- 首次启动会按 `--packing_batch_size`（默认 1000）分批 tokenization，并在 Hugging Face
  cache 中生成定长 Arrow blocks；后续启动直接复用 cache。
- 使用 best-fit packing，完整样本不会被拆到两个 block；Pretrain 保留每篇文档的 BOS/EOS，
  SFT 同步保留 assistant-only `-100` loss mask。
- block 形状保持固定，因此兼容 `torch.compile`、DDP 和 profiler，不会因动态 batch 反复编译。
- Packing 会改变每个 epoch 的 step 数。从 non-packed checkpoint 使用 `--from_resume` 切换时，
  会保留模型、优化器与 GradScaler，并还原当前 epoch 的 shuffle 顺序：未到 packing 分组边界
  的数据行继续按原始 batch 训练，到达边界后将同一 epoch 尚未训练的后缀构造成 packed blocks。
  迁移起点和游标会写入 checkpoint，因此迁移途中暂停后仍可精确继续。

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
