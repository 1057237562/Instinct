# cuBLASLt 分支：调度框架与 WebUI 功能契约

分支 `experimental/dense-cublaslt-fp8` 基于 `experimental/dense-triton-fp8`。
最终验收范围是 **`scripts/config_webui.py` 的全部功能，唯一排除项是
`use_compile` / `compile_mode`（torch.compile）**。CUDA Graph 是独立调度能力，仍在范围内。

当前交付是框架阶段：一个 Dense block 的 cuBLASLt Linear 前向/dX/dW、
Triton 非 GEMM 算子、固定缓冲区调度，以及完整配置范围的检查与缺口报告。
**尚不是完整 LLM trainer，也没有完成 WebUI 功能对等。** 原有 WebUI 仍启动原有
PyTorch trainer；不能用“接受配置”或回退到 PyTorch 来表示新后端已实现该功能。

## 已有底座

| 部件 | 实现 |
|---|---|
| CUDA 上下文和内存 | 继承直接 Driver API launcher；运行路径不导入 Torch/TorchAO |
| Linear 前向/dX/dW | 官方头文件构建的 C++ cuBLASLt shim；FP32/BF16/FP16 和 tensorwise FP8 |
| FP8 recipe | 激活/权重 E4M3，反向 dY E5M2，FP32 accumulation，关闭 fast accumulation；反量化 scale 在 GPU 上取倒数 |
| Layout | 统一 column-major TN；按 16 补齐，转置输入按需 pack，结果按需 crop；已对齐前向避免输入 pack |
| 计划缓存 | 按 shape、dtype、stream、workspace、SM hint 缓存候选/选中算法 |
| 默认调度 | 单 stream，32 MiB workspace，选择首个可执行 heuristic；不计时调优 |
| 可选调度 | 低优先级 dW stream，独立 workspace；event 依赖与步末 join；可选 dW SM hint |
| 重放 | `backend.build(callback)` 准备一次；记录 kernel/GEMM/event，固定地址重放；可选 CUDA Graph capture |
| 复用 | 同一 block 的 Q/K/V 输入量化和 gate/up 输入量化复用；scale/pack 缓存有跨 stream event 依赖 |

`dw_sm_target` 是 cuBLASLt heuristic 提示，stream priority 也不是硬件分区或抢占保证。
双 stream 默认关闭：两个 GEMM 争抢 Tensor Core/带宽时可能更慢，不能预设收益。
所有输入 pack 都是正确性优先的通用实现，尚未优化转置访存或融合。

arena 保留准备期间分配的所有 buffer；replay 本身不分配新 arena buffer。
这不是按张量存活期复用内存的 allocator。重复调用分配式 forward/backward 会继续增长，
训练循环应重放固定计划；改变 shape 应新建 backend/context。CUDA Graph、库内部内存不计入
`arena_bytes`，需同时观察 `device_free_bytes` 和系统显存统计。

cuBLAS 的反量化乘法与 Triton 原始除法、以及不同 GEMM reduction 顺序会引入舍入差异。
精度目标是相同 FP8 格式/缩放规则/累加策略与数值容差、收敛对照，**没有声明与 TorchAO
逐 bit 一致**。也没有证据说明 Windows 下观察到的 2 GiB 会全部消失；CUDA context、
cuBLAS 和驱动仍有自己的开销。

## 功能验收清单

机器可读契约在 `model/dense_cublas/config.py`。以下全部属于要求，不是可选删减项。
当前实现和计划要求分别标记，未接通的功能不可启动新后端训练。

| WebUI 功能 | 当前状态 | 新后端验收内容 |
|---|---|---|
| Dense 配置、层数、词表、权重共享 | 单 block 已有；完整模型待实现 | embedding、任意层堆叠、末层 norm、LM head、shifted/masked CE、共享梯度 |
| GQA/MHA、head 数和维度、RMSNorm | 部分已有 | 所有合法 UI shape；目前 block 限偶数 head_dim ≤ 128 |
| silu/gelu/relu | 仅 silu 已有 | 三种 gated FFN 前反向及专家复用 |
| dropout | 待实现 | attention/residual dropout，训练/评估切换，RNG 保存和重放 |
| RoPE/YaRN | rotary kernel 已有，配置表生成待接 | theta、最大长度、beta_fast/slow、factor、原长度、attention_factor |
| flash_attn | 因果 online GQA 已有，开关语义待接 | 开/关路径数值、mask、dropout 对等 |
| MoE | 待实现 | 专家数量/宽度、top-k、归一化、router auxiliary loss 及反向 |
| GatedDeltaNet (Linear) | 待实现 | depthwise conv、状态递推及反向、头/维度配置、full-attention interval |
| Looped (LoopUS) | 待实现 | encoder/body/output 分区、共享参数、q/最大步数、监督、beta、深度奖励、自蒸馏、teacher stop-grad |
| Early Exit | 待实现 | 中间层共享 head 的 CE、权重、共享梯度累加 |
| 参数/激活精度 | kernel 基础已有，训练策略待接 | FP32/BF16/FP16 组合、master weights、scaler/overflow、FP8 recipe |
| KV cache 精度 | 待实现 | FP32/BF16/FP16/E4M3/E5M2，per-(batch,head) scale、扩展/重排/清空 |
| 重计算模式 0/1/2 | 待实现 | 无重算/attention+FFN 选择重算/整层；RNG 与 FP8 scale 一致；实际降低显存 |
| AdamW/Adafactor/Muon | 待实现 | 更新公式、状态、梯度累积/裁剪、lr schedule、保存恢复 |
| Pretrain / Full SFT | 待实现 | tokenizer、数据批次、mini/normal、序列长度、label mask、完整 step |
| LoRA | 待实现 | 低秩矩阵反向、冻结基座、adapter 保存/加载/merge |
| DPO | 待实现 | chosen/rejected、参考模型、log-prob 与 loss 梯度 |
| PPO | 待实现 | rollout、reward/reference/value、GAE、policy/value 更新 |
| GRPO/CISPO | 待实现 | 多样本、组优势、ratio/clip、KL、mask 和训练变体 |
| Agent RL | 待实现 | 多轮工具交互、reward、边界/mask、更新 |
| Distillation | 待实现 | teacher/student 加载、KD 温度及 loss、stop-grad |
| 基础权重、保存、续训 | 待实现 | auto/显式选择、原子保存、optimizer/RNG/sampler/scaler 恢复；.pth 转换与 GPU 运行分离 |
| WebUI 操作和展示 | 原 UI 保留，新后端待接 | presets、导入/导出、架构图/参数估计、启动/状态、日志/历史/指标、权重发现 |
| torch.compile | 明确排除 | 配置报告列出被排除字段；不导入或调用 Torch 编译链 |

落地顺序：① 当前 Dense 调度底座；② 完整 Dense LM、精度/激活/dropout/RoPE、
重计算、优化器、数据/续训和 pretrain/SFT；③ MoE/Linear/Looped/Early Exit/KV/LoRA；
④ DPO/PPO/GRPO/Agent/KD；⑤ WebUI 新后端启动与组合回归。
非标准架构和训练目标都保留为明确模块边界，不应塞进 cuBLAS shim。

## 无 GPU 的配置检查

从仓库根目录运行，读取 WebUI 导出的模型 JSON：

```bash
python -m experiments.plan_cublas --config_path trainer/config_pretrain.json
python -m experiments.plan_cublas --config_path trainer/config_pretrain.json --require_ready
```

第一条只生成覆盖报告，不加载 Triton/CUDA/Torch。第二条检查能否训练，**当前返回 2**，
列出 blocker。非法或未知配置返回 1。不存在“忽略不支持配置并启动 Dense”的回退。

训练面板设置可单独放入 JSON，通过 `--training_path` 传入。例如：

```json
{
  "train_type": "full_sft",
  "dataset_size": "mini",
  "batch_size": 32,
  "max_seq_len": 768,
  "optimizer": "muon",
  "param_dtype": "fp32",
  "activation_dtype": "bfloat16",
  "kv_cache_dtype": "fp8_e4m3",
  "use_grad_checkpoint": 2,
  "from_resume": false,
  "base_weight": "auto"
}
```

契约测试直接执行 WebUI 的纯配置构造/导出函数，覆盖四种架构、MoE 组合、YaRN 和
Early Exit；校验所有导出字段都登记了归属。训练类型/优化器选项也与真实 UI 核对。

## 实卡构建和入口

要求：NVIDIA GPU、CUDA Toolkit ≥ 12.8、C++17/CMake ≥ 3.21、NumPy、Triton **3.4.x**。
BF16 路径需要 SM80+，FP8 需要 SM89+。Windows 还需可用且与 launcher ABI 相容的
Triton 安装；本环境没有验证 Windows 构建/执行。

```bash
cmake -S model/dense_cublas/native -B model/dense_cublas/native/build -DCMAKE_BUILD_TYPE=Release
cmake --build model/dense_cublas/native/build --config Release

# 单 stream 基线；奇数 token 数检查 padding/crop
python -m experiments.smoke_dense_cublas --fp8 0 --seq_len 17
python -m experiments.smoke_dense_cublas --fp8 1 --seq_len 17

# 在基线数值通过后，分别启用选项，避免混淆收益来源
python -m experiments.smoke_dense_cublas --fp8 1 --overlap_dw
python -m experiments.smoke_dense_cublas --fp8 1 --tune
python -m experiments.smoke_dense_cublas --fp8 1 --overlap_dw --cuda_graph
```

CMake 命令也适用于安装了 MSVC 和 CUDA 的 Windows shell。默认查找 Linux `.so` 或
Windows `build/Release/instinct_cublaslt.dll`；自定义构建路径用 `--library` 或
`INSTINCT_CUBLASLT_BRIDGE`。Windows 通过 `CUDA_PATH/bin` 查找 CUDA DLL。

smoke 入口检查一个 block 的输出/11 组参数梯度是否有限、重放是否稳定、arena 是否增长。
输出包含每条命令所属 stream、候选算法/workspace、显存统计。它没有 optimizer，
不是端到端训练，也不提供 MFU 结论。

`ExecutionPlan` 对 buffer 地址、shape、dtype 固定。要换输入/权重，用
`backend.copy_from_numpy(tensor, values)` 同步并原地更新已有 buffer；不能用新 Tensor
地址替换。BF16/FP8 上传要求对应 uint16/uint8 原始编码，此接口不进行量化。
准备/调优在 graph capture 前完成。双 stream
计划在步末汇合，外部消费结果需要同步或接在 main stream 上；不能并发重放同一计划。

## 验证边界与后续实卡工作

本线上环境没有 CUDA GPU：已做 C++ 官方头文件语法检查、Python 检查、Triton
interpreter 的 pack/crop/scale、完整 block 数值及调度依赖测试。测试 GEMM 使用 Torch
参考替身，**没有实际执行 cuBLAS**；替身只存在测试文件中。原生测试明确 skip，不能据此
认定 FP8 精度、Windows 支持、真实并发或 CUDA Graph 已通过。

```bash
TRITON_INTERPRET=1 python -m pytest -q tests/test_dense_cublas.py tests/test_cublas_webui_contract.py
# 实卡及 shim 构建完成后，去掉 TRITON_INTERPRET：
python -m pytest -q tests/test_dense_cublas.py
```

实卡阶段先做 native FP8/BF16 的前反向参考对照和 compute-sanitizer；随后在同一硬件、
形状和精度下测单 stream 基线，再分别比较候选算法、workspace、dW 并发/SM hint 和
CUDA Graph。记录 warmup 后稳态 step 时间、峰值/常驻显存与硬件/驱动/CUDA 版本。
完整 trainer 和收敛验证通过后再报告 tokens/s、模型 FLOPs 和 MFU，不能由孤立 GEMM
峰值或本环境离线编译推导训练提速。

API 依据：[NVIDIA cuBLAS 12.8 文档](https://docs.nvidia.com/cuda/archive/12.8.2/cublas/index.html#cublasltmatmul)
和 [NVIDIA FP8 cuBLASLt 示例](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtFp8Matmul)。
本环境仅用官方 cuBLAS 12.9 头文件进行了 C++ 语法检查，没有链接或执行原生库。

本次验证结果：新后端/配置契约加现有 Dense/Triton/checkpoint/flag 回归共 **71 passed、
10 skipped**；skip 涉及无 CUDA GPU 或 interpreter 的原生 FP8/BF16 限制。新增
pack/crop/scale kernel 在 SM89/90/120 上分别完成 9 个离线编译 specialization，
共 27 个，编译进程未导入 Torch。Python 编译检查和 `git diff --check` 通过。
