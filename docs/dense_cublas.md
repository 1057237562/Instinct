# 原生 cuBLASLt / Triton 训练

分支：`experimental/dense-cublaslt-fp8`。
入口：`trainer/train_cublas.py`；WebUI 的 **Training backend → cublas**。
模型、反向传播、优化器和 GPU reward 推理均使用直接 CUDA Driver API、cuBLASLt 与
Triton，不在训练进程导入 PyTorch / TorchAO，也不调用 `torch.compile`。

代码已接通完整训练流程，可以进入实卡验收。当前环境没有 CUDA GPU：这里的测试
覆盖 CPU interpreter 数值、接口和离线编译，不能证明真实 GPU 正确性、FP8 收敛或 MFU。

## 实现范围

| 功能 | 实现位置与行为 |
|---|---|
| Dense / MoE | `model.py`：embedding、层堆叠、共享/独立 LM head、masked shifted CE；MoE top-k 只执行选中 token 的专家计算，并保留 router auxiliary gradient |
| Linear | `recurrent.py`：因果 depthwise conv、Gated DeltaNet、FP32 recurrent state、显式反向；每 16 token 重计算局部状态，支持流式 cache |
| Looped | encoder/body/output 分区、共享 body、选择性门、confidence head、阈值/上限、随机监督深度、单调性/深度/增益奖励、自蒸馏和 teacher stop-grad |
| 激活与位置 | SiLU / exact-erf GELU / ReLU、RMSNorm、RoPE / YaRN 全部导出参数、attention/residual/embedding dropout |
| Attention | head_dim≤256 的训练使用 fused causal GQA，支持 attention dropout、padding mask 和确定性前反向/重算；更大 head 及 decode 使用通用路径 |
| 重计算 | 0 关闭、1 attention/FFN、2 整层；RNG seed/counter 与 FP8 scale 在重算时重建 |
| Early Exit | 按 **1-based** 层号增加共享 head 的训练 CE |
| 精度 | FP32/BF16/FP16 参数与激活，FP32 optimizer state / gradient accumulation，FP16 动态 loss scale、overflow skip / 梯度清零 |
| FP8 | E4M3 前向输入/权重、E5M2 dY、tensorwise current amax scaling、FP32 GEMM accumulation；不依赖 TorchAO |
| KV cache | FP32/BF16/FP16、per-(batch,head) scaled E4M3/E5M2；扩展、重排、reset |
| 优化器 | fused AdamW、factored Adafactor、Muon 矩阵 Newton–Schulz + 非矩阵 AdamW；累积、裁剪、cosine LR |
| 八种入口 | pretrain、full_sft、LoRA、DPO、PPO、GRPO/CISPO、Agent RL、distillation |
| 数据 | Rust tokenizers + Jinja、索引 JSONL、spawn 多进程有界预取、按批次种子重现增强、pinned 上传缓冲；保留 label mask、chosen/rejected、工具消息 |
| RL | 原生采样、reference/value models、GAE、clipping/KL、组归一化、多轮工具观测零 policy mask；本地 InternLM2 reward safetensors 推理 |
| 保存恢复 | 原子 `.npz`，模型/LoRA、optimizer、待累积梯度、reference/teacher/critic、采样顺序、RNG、loss scale；原生精确续训要求配置与数据批次设置保持一致；旧 .pth 可迁移已保存的完整训练状态，缺失状态见下文 |
| WebUI | 原有 presets、模型 JSON 导入/导出、架构图、参数估计保留；新后端启动、日志/指标、刷新恢复、权重发现与续训已接通 |
| 调度 | workspace / heuristic plan cache、可选候选算法计时、独立 dW stream、event 依赖；原单 block 固定 buffer / CUDA Graph 实验保留 |

`config.py` 的 `implementation_ready` 与 `training_ready` 是不同字段。前者表示代码
已实现；后者保留为实卡与收敛验收门槛，因此当前 `plan_cublas --require_ready` 仍返回 2。
实际 runner 不因缺少远程 GPU 验收而禁止用户运行。

### Looped 的兼容边界

原 `model_instinct_loop.py` 是旧的 `prelude + loop_iters + coda` 模型，没有读取 WebUI
中的多数 LoopUS 控件。新后端实现这些控件，**不能将旧 Looped checkpoint 当作同一
结构直接载入**。可以加载相同层命名的 Dense/MoE 基座；新 gate / confidence 参数初始化，
然后训练并保存原生 checkpoint。

新 Looped 按 token 用严格 `q > q_threshold` 冻结状态，确保因果训练与 decode 可对齐；
KV cache 开启时仍填充所有循环深度的 cache。训练目标为最终 CE，加随机 K 个深度的
CE / confidence BCE / monotonicity，再加 survival-chain 深度成本、可选深度增益及 KD。
`n_supervision` 控制质量损失采样，body 的递归梯度仍贯通；长循环建议重计算模式 2。
这是一套明确的新实现，未宣称与旧 Looped 文件或论文实验逐数值相同。

### FP8 和显存

`--fp8 auto` 在 SM89+ 开启 FP8，SM80/86 使用所选高精度；`--fp8 1` 强制开启。
FP8 与 TorchAO 的目标是相同格式/缩放/累加策略及可接受的误差、收敛，**不是 bitwise 相同**。
Tensor Core reduction、cuBLAS 的反量化乘法和 BF16 rounding 均可能产生差异。

动态训练 arena 会回收中间 buffer，空闲缓存默认最多 256 MiB：
`--allocator_cache_mib 0` 可关闭空闲缓存，代价是更多 cudaMalloc/free。
报告同时包含 live / cached / Driver free bytes。CUDA context、Triton modules、cuBLAS
内部缓存不等于 tensor arena；移除 Torch 不能保证 Windows 上观察到的 2 GiB 全部消失。

## 安装与构建

从仓库根目录运行。需要 NVIDIA SM80+、CUDA Toolkit ≥12.8、C++17、CMake ≥3.21。
直接 launcher 固定 Triton **3.4.x ABI**；Windows 还需 MSVC 及兼容的 triton-windows。
本环境没有验证 Windows 的构建、加载或 kernel 执行。

```bash
pip install -r requirements-cublas.txt
cmake -S model/dense_cublas/native -B model/dense_cublas/native/build -DCMAKE_BUILD_TYPE=Release
cmake --build model/dense_cublas/native/build --config Release
```

自动查找 `native/build/libinstinct_cublaslt.so` 或 Windows `build/Release/instinct_cublaslt.dll`。
也可设置 `INSTINCT_CUBLASLT_BRIDGE` 指向产物。WebUI 继续使用项目原有 Streamlit 依赖。

## 建议的实卡验收顺序

先跑 FP32 单 stream 数值参考，再启用低精度、FP8、重计算及并发。
`--reference` 会启动隔离的 **CPU-only PyTorch 子进程**生成参考值；GPU 主进程仍拒绝 Torch。
该参考需要可用的 torch / datasets / transformers 测试依赖；常规 native 训练不需要它们。

```bash
python -m experiments.validate_cublas_training --fp8 0 --dtype fp32 --reference
python -m experiments.validate_cublas_training --fp8 0 --dtype bf16
python -m experiments.validate_cublas_training --fp8 1 --dtype bf16
python -m experiments.validate_cublas_training --fp8 1 --dtype bf16 --checkpoint 2 --dropout 0.1
python -m experiments.validate_cublas_training --fp8 1 --dtype bf16 --overlap_dw
python -m experiments.validate_cublas_training --architecture moe --fp8 0 --dtype fp32 --reference
python -m experiments.validate_cublas_training --architecture linear --fp8 0 --dtype fp32 --reference
python -m experiments.validate_cublas_training --architecture looped --checkpoint 2
python -m experiments.validate_cublas_training --kv_cache_dtype fp8_e4m3 --dtype bf16
```

FP8 需要 SM89+。FP32 CPU oracle 默认 atol=3e-4 / rtol=3e-3；BF16/FP8 与高精度对照需要
另外明确容差（`--atol`、`--rtol`），不能用放宽容差替代长期收敛检查。
验证脚本检查完整前反向/更新、非有限值、续训一致性、cache decode、live 内存是否增长，
输出 JSON 包含硬件架构、每步时间、内存和 cuBLAS 候选算法。首次 step 包含 JIT/tuning。

继续用 compute-sanitizer 检查越界；再用真实数据跑短训练，对照 loss、gradient norm、
若干百/千步验证集曲线。GRPO/PPO/Agent 还需本地 `../internlm2-1_8b-reward` 的完整
safetensors / config / SentencePiece 文件；真实 reward 模型和 tokenizer 的整机对照留待实卡，
这里仅验证了小尺寸 grouped-QKV reward 数学路径，没有下载 1.8B 权重。
Agent 的天气、时间、汇率等沿用原仓库的**模拟训练环境**，不是实时服务。

## 启动训练

```bash
python trainer/train_cublas.py --train_type pretrain --config_path trainer/config_pretrain.json --from_weight none --fp8 auto --batch_size 4 --max_seq_len 256 --accumulation_steps 4 --use_grad_checkpoint 2
python trainer/train_cublas.py --train_type full_sft --config_path trainer/config_full_sft.json --from_weight out/pretrain_768.npz
python trainer/train_cublas.py --train_type pretrain --config_path trainer/config_pretrain.json --from_resume 1
```

数据路径沿用原项目，其他入口用 `--train_type lora|dpo|ppo|grpo|agent|distillation`。
蒸馏指定 `--from_teacher_weight`，teacher/student 必须使用同一 tokenizer token-ID 语义。
LoRA 默认仅方阵 Linear，与原目标模块规则一致；保存完整模型及 `_adapter.npz`，
API 提供 adapter load 和 merge。RL rollout/current policy 概率统一禁用 dropout。

WebUI 中选择 `cublas` 后会禁用 compile 控件，其他原有配置继续生效。
`.pth` 基础权重首次加载会由独立 CPU 进程转换；也可预先转换，使用不安装 Torch 的训练环境：

```bash
python scripts/convert_cublas_checkpoint.py --input out/pretrain_768.pth --output out/pretrain_768.npz
python scripts/convert_cublas_checkpoint.py --input out/full_sft_768.npz --output out/full_sft_768.pth
```

上面的独立命令默认只转换权重。完整旧训练状态通过 `train_cublas.py --from_resume 1` 自动生成转换 manifest，并在独立 CPU 进程转换，详见下文。
输出命名为 `out/{save_weight}_{hidden_size}{_moe}.npz` 和
`checkpoints/{save_weight}_{hidden_size}{_moe}_resume.npz`。
Ctrl+C 请求会在当前 batch 完成后保存；强制杀进程只能恢复最近一次完整 checkpoint。

## 并行数据加载

默认 `--num_workers 8 --prefetch_factor 2 --pin_memory 1`，WebUI 提供对应控件。
使用 spawn 进程，不 fork 已初始化的 CUDA context；worker 只做 JSONL 读取、分词与组批。
最多排队 `num_workers × prefetch_factor` 个批次，按 sampler 顺序交付，错误传回主进程。
`--num_workers 0` 可用于排错。批次增强由 seed/epoch/样本游标确定；预取、worker 数量变化
和中断后重启不会提前消耗训练 RNG。旧版原生 checkpoint 则保留串行增强模式，以保持其 RNG。

上传端复用 CUDA Driver 分配的 pinned CPU 缓冲，最多 16 MiB；更大的上传继续走 pageable
路径，避免锁住过多主机内存。`memory()` 报告 `host_pinned_bytes`。HtoD 仍为同步复制，
CPU 数据准备可以提前运行；此处没有实现独立 GPU copy stream 的计算/传输重叠。

## 从旧 PyTorch checkpoint 续训

WebUI Resume 会发现 `.pth` 和 `.npz`，同名情况下 trainer 优先使用原生 `.npz`。
也可明确指定旧 checkpoint（模型 JSON、优化器类型与训练模式应与原训练保持一致）：

```bash
python trainer/train_cublas.py --train_type pretrain --config_path trainer/config_pretrain.json --from_resume 1 --resume_path checkpoints/pretrain_768_resume.pth --optimizer adamw --dtype bfloat16 --fp8 0 --batch_size 32 --accumulation_steps 8 --epochs 2 --legacy_batch_size 32 --legacy_accumulation_steps 8 --legacy_epochs 2
```

恢复模型、AdamW 的一二阶矩/step、Adafactor 的 row/col/variance、Muon 动量及其配方、
每个参数组的超参、GradScaler、CosineAnnealingLR、epoch/step，以及已保存的
reference/teacher/critic 和 critic optimizer/scheduler。兼容 tied weights、LoRA、
Dense/MoE/Linear；按源模型参数名映射 optimizer ID，不能直接套用新模型的参数注册顺序。
Muon 区分项目 fallback 和 `torch.optim.Muon` 的不同动量与更新规则；后者的 BF16
Newton–Schulz Tensor Core 数值仍需实卡验证。原训练未提供的 AMSGrad、自定义优化器等
不能默默转换；不匹配会报错，不覆盖源 `.pth`。

转换器在独立 CPU 进程使用 `torch.load(weights_only=True)`，需要可用的 torch/datasets；
输出原子写入同名 `.npz`。之后继续训练只需要原生依赖，训练进程不导入 Torch。
独立的权重转换命令仍然只转换权重。

**旧文件没有的信息无法补回来**：原保存函数把模型权重转成 FP16，未保存 RNG、worker
增强状态或未更新的累积梯度，因此只能恢复已保存的状态，不能逐 bit 接续原轨迹。
累积中途保存会报告梯度缺失并清空累积。原 batch size、累积步数、总 epoch 数也未存储，
`--legacy_*` 默认采用当前值；若不同必须传入原值。监督训练的最大 LR 可从保存 LR 与
原训练总步数推算，也可用 `--learning_rate` 明确指定。转换报告随原生进度一起保存。
旧 Looped 结构不兼容的限制仍然存在，本次没有改写其架构。

## 本次实卡检查

CPU 上已对照迁移后的下一次优化器更新，检查多进程预取/续训、共享权重/LoRA、附加
模型、scaler/scheduler，以及 Attention dropout 和 head_dim=160/192/256 的前反向。
新增 Attention 的 72 个特化已在 SM89、SM120 分别离线编译通过；最大 shared memory 为 90,240 bytes，低于 99 KiB。编译检查同时验证 direct-launch ABI；不能只看编译成功。
实卡从 FP32 开始，再测 BF16/FP16 与 FP8：

```bash
python -m experiments.validate_cublas_training --head_dim 256 --dtype fp32 --fp8 0 --reference
python -m experiments.validate_cublas_training --head_dim 192 --dtype bf16 --dropout 0.2 --checkpoint 2 --fp8 0
python -m experiments.validate_cublas_training --head_dim 192 --dtype bf16 --dropout 0.2 --checkpoint 2 --fp8 1
python -m experiments.compile_native_attention --arch 89
```

## 当前验证与后续 MFU 工作

此前实现完成核心回归与 WebUI/LoRA/oracle 检查；本次增加迁移和预取、融合 Attention 回归。
完成 CPU interpreter / PyTorch oracle 回归；原生 GPU、BF16 Tensor Core 和 FP8 cast
测试因无 GPU 或 interpreter 的 rounding 限制跳过。数值覆盖完整 Dense/MoE/Linear、
checkpoint/dropout、三个 optimizer、八种训练模式及续训、KD/DPO/GRPO/CISPO/PPO 梯度、
GAE、工具观测 mask、原生 tokenizer、reward 数学路径、内存/event 回收。

```bash
TRITON_INTERPRET=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q tests/test_cublas_training.py tests/test_cublas_webui_contract.py tests/test_dense_cublas.py tests/test_dense_triton.py tests/test_dense_model.py tests/test_checkpoint_ffn.py tests/test_flag_threading.py tests/test_native_attention.py tests/test_native_loader.py tests/test_legacy_resume.py
python -m experiments.compile_cublas_training --arch 89
```

既有训练基础 kernel 的 51 个 specialization 此前已分别离线编译 SM89 / SM90 / SM120（合计 153）；
这只检查 compiler / launcher ABI，不能代替 GPU 执行。

实卡优化优先级：

1. 固定模型、batch×seq、精度，记录 warmup 后 tokens/s、显存及 Nsight 时间线；把 JIT、
   数据准备、保存和 reward rollout 时间与训练 step 分开统计。
2. 比较 `--workspace_mib`、`--candidates`、`--tune`、`--overlap_dw`、`--dw_sm_target`。
   现通用 autograd 为保证依赖正确会及时 join dW；进一步跨层重叠应由时间线指导。
3. 将 MoE 路由索引的 host 同步换成 GPU prefix-sum / compact dispatch，增加 grouped GEMM。
4. 为 decode / head_dim>256 的通用 attention、depthwise conv、recurrent chunk、optimizer reductions
   加融合内核；当前通用 attention 会物化 S²，长序列应优先 fused 路径。
5. 缓存常量、减少临时 broadcast/copy，给动态训练建立固定 bucket 的 buffer/liveness 计划；
   再扩展整步 CUDA Graph。现有 CUDA Graph 仅覆盖 `smoke_dense_cublas` 的固定 block 实验。
6. 检查 FP8 相对 BF16 的精度/收敛，最后测 MFU。`--peak_tflops` 只给 Dense 近似
   `6 * parameters * tokens/s / peak`；MoE/Looped/RL 必须按实际激活参数、循环深度和阶段重新计数。

API/格式依据：
[NVIDIA cuBLASLt](https://docs.nvidia.com/cuda/archive/12.8.2/cublas/index.html#cublasltmatmul)、
[NVIDIA FP8 示例](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtFp8Matmul)、
[InternLM2 reward 官方实现](https://huggingface.co/internlm/internlm2-1_8b-reward/blob/main/modeling_internlm2.py)。

旧 Muon 配方迁移参考 [PyTorch v2.10 Muon 源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/_muon.py)。
