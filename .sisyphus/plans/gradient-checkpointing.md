# Gradient Checkpointing (选择性激活重计算) — Instinct

## TL;DR

> **Quick Summary**: 为 Instinct 训练框架设计梯度检查点机制：以选择性激活重计算（只重算注意力 QKᵀ/softmax 等"计算便宜、存储贵"的中间量）为主，整层 `torch.utils.checkpoint` 为兜底，覆盖 Dense/MoE/Loop/Linear 全部模型变体与 8 个 trainer，在不显著增加计算量的前提下节省训练显存。
>
> **Deliverables**:
> - `model/checkpointing.py` — 无参数注意力重计算 `autograd.Function`（diffusers 模式）+ FFN 区域 `torch.utils.checkpoint` 封装 + MoE aux_loss 返回修正
> - `--use_grad_checkpoint 0/1/2` flag（0=off, 1=选择性重算, 2=整层）贯穿 InstinctConfig(×3) + config_from_args + 8 个 trainer + WebUI
> - `tests/` pytest 套件（TDD）：梯度一致性、step 一致性、MoE router 梯度、flag 线程化
> - GPU 显存/吞吐基准脚本 `experiments/bench_checkpoint_memory.py`
> - 移除死 flag `--loop_grad_checkpoint`（train_pretrain.py + config_webui.py）
>
> **Estimated Effort**: Large（13 个实现任务 + F1-F4 终验）
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: T1 → T2/T3 → T7 → T10 → F1-F4 → user okay

---

## Context

### Original Request
> 设计一套 gradient checkpointing 机制，节省训练时显存占用，针对其中如 Attention 模块中 QK^T 这种计算容易、且结果占用显存大的模块进行重计算，在不提升大量计算量的前提下节省显存。

### Interview Summary
**Key Discussions**:
- **粒度**: 选择性重计算为主（自定义 autograd.Function 重算 QKᵀ/softmax）+ 整层 `torch.utils.checkpoint` 兜底（用户确认）
- **模型覆盖**: 所有变体 — Dense、MoE、LoopUS（循环）、Linear（GatedDeltaNet）（用户确认）
- **Flag 暴露**: 全部 8 个 trainer + WebUI（用户确认）
- **测试**: TDD 自动化测试（用户确认）
- **Flash 路径**: 保留 flash（flash 时 QKᵀ 不物化无可省）；选择性重算只在 eager 分支生效（用户确认）
- **Linear 变体**: Mode 2 整层 + 标准 Attention/FFN 部分 Mode 1；不深入 GatedDeltaNet FLA/Triton kernel（用户确认）
- **验收目标**: 用户引用"省 80% 显存、只增 5% 计算"方案 — 已调查（见下）

**Research Findings**:
- **Korthikanti et al. (MLSys 2022)** — 用户引用的"80%/5%"是论文数字的宽泛转述。论文真实数字：选择性重算省 **70%**（GPT-3）/ **65%**（MT-NLG）层激活，计算开销 **1.6-2.7% FLOPs**；整层重算 30-40% 墙钟开销。选择性重算的精确定义：只丢弃 QKᵀ scores + softmax 输出 + dropout mask + attn@V dropout 输出（`5as²b` 项），保存 Q,K,V + 全部 MLP + norm，从 QKV 投影之后开始重算
- **本仓库形状**（H=768, 8q/4kv, d=96, SwiGLU 2432, dropout=0 默认）: 每层激活 = 26880 + 16S bytes/token（dropout=0 时 softmax 输出是唯一二次项）。eager 路径选择性重算可省 **31%**（S=768）/ **55%**（S=2048）/ **71%**（S=4096）/ **95%**（S=32768）; 计算开销 ~5%（S=768）/ ~10%（S=2048）。**S≤768 时 MLP 中间量占主导** → 短序列 Mode 2 更有效
- **Flash 路径（默认开）**: 注意力已在 kernel 内重算（softmax_lse），选择性注意力重算零收益；只有 FFN/MLP 重算有效
- **死 flag**: `--loop_grad_checkpoint` 在 train_pretrain.py:115,166-167 + config_webui.py 声明/写入但模型代码从不读取（继承自 MiniMind 初始提交）
- **MoE aux_loss 侧信道**: `self.aux_loss` 模块属性（model_instinct.py:184-188）在 layer 后读取（247-248）— checkpoint forward 在 no_grad 下执行 → 梯度静默丢失，必须改为从 checkpointed 函数返回值返回
- **无测试基建**: 无 pytest/tests/CI；requirements.txt 无 pytest

### Metis Review
**Identified Gaps** (addressed):
- **FFN/MoE 参数梯度陷阱**: 自定义 `autograd.Function.forward` 在 no_grad 下运行，区域内参数（gate/up/down_proj、expert 权重）会得到零梯度 → **FFN/MoE 区域必须用 `torch.utils.checkpoint`**（backward 内部 enable_grad 重算）；自定义 Function 仅用于无参数注意力区域
- **Attention Function 边界**: 在 `repeat_kv` **之前**保存 q/k/v（pre-expansion），backward 内重算 transpose+repeat_kv+QKᵀ+softmax+dropout+@V，避免保存 2× 扩展副本
- **aux_loss 覆盖全部 3 个模型文件**: dense :247-248、loop :375-383、linear 相同模式
- **`hidden_states += residual`（:205）安全性**: 验证过——修改的是 attn 输出而非 block 输入，`use_reentrant=False` 无 in-place 冲突
- **dtype 重算**: backward 重算在保存张量自身 dtype 下进行（autocast 在 backward 不生效），需 fp16/bf16 梯度一致性测试验证
- **GatedDeltaNet**: FLA/Triton kernel 内部重算不可靠 → Linear 变体仅标准部分 Mode 1，整层 Mode 2

---

## Work Objectives

### Core Objective
在不显著增加训练计算量的前提下，通过选择性激活重计算（针对 QKᵀ/softmax 等 O(seq²) 大激活）和整层 checkpoint 兜底，显著降低 Instinct 训练的峰值显存占用。

### Concrete Deliverables
- `model/checkpointing.py` — 注意力重计算 autograd.Function + FFN checkpoint 助手 + aux_loss 返回封装
- 3 个模型文件（dense/loop/linear）接入 Mode 1（选择性）与 Mode 2（整层）
- `--use_grad_checkpoint` flag：3 个 InstinctConfig + `config_from_args` + 8 个 trainer argparse + WebUI
- `--loop_grad_checkpoint` 死 flag 移除（train_pretrain.py + config_webui.py）
- `tests/` pytest 套件 + `experiments/bench_checkpoint_memory.py` GPU 基准
- README 训练参数表新增 flag 文档

### Definition of Done
- [x] `pytest tests/ -x -q` 全绿（CPU 可跑）
- [~] GPU 基准显示 S=2048 eager：Mode 1 激活节省 ≥45%，Mode 2 ≥85%；计算开销 ≤15%（Mode 1）/ ≤50%（Mode 2）
- [x] 8 个 trainer `--use_grad_checkpoint 1` smoke 通过（loss 有限、无崩溃）
- [x] `grep -r loop_grad_checkpoint` 零残留

### Must Have
- Mode 0（默认）行为与现状完全一致（bitwise 级）——固定种子 step 一致性回归测试
- 自定义 Function 无参数陷阱：仅注意力区域（QKV 投影之后），保存 pre-repeat_kv q/k/v
- FFN/MoE 区域用 `torch.utils.checkpoint(use_reentrant=False)`（参数梯度安全）+ aux_loss 从返回值返回
- 所有 3 个模型文件 + 全部 8 个 trainer + WebUI 接入
- pytest 加入 requirements；GPU 测试 `@pytest.mark.gpu`（无 CUDA 跳过）
- 重算区域保持 dropout RNG 一致（preserve_rng_state）

### Must NOT Have (Guardrails)
- 不重构 3 个模型文件的类重复（不合并成共享基类）
- 不改 flash_attn_4.py / kv_cache_quant.py / eval_llm.py / dataset 代码
- 不重新实现 torch.utils.checkpoint
- 不做激活 CPU offloading（不同机制，未要求）
- 不加每层/每区域细粒度 flag、不做自动调优
- 不改种子处理、optimizer 状态、`y[0,0] += 0*sum(...)` expert 技巧
- 死 flag 不在 train_pretrain.py 和 config_webui.py 中半移除（同一 commit 清完）
- 不在 `self.training=False`（推理/rollout）时启用 checkpoint（纯开销）
- 不加 CI 基建；测试只本地跑
- 不引入 offloading/分片/模型并行

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO（仓库无 pytest/tests/CI — 需 T1 搭建）
- **Automated tests**: YES (TDD)
- **Framework**: pytest（加入 requirements.txt）
- **TDD**: 每个实现任务先写失败测试（RED）→ 最小实现（GREEN）→ 重构

### QA Policy
每个任务 MUST 包含 agent 执行的 QA 场景（见 TODO 模板）。证据存 `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`。
- **梯度一致性**: `python -c` 或 pytest 断言 `torch.equal`（CPU fp32）/ `torch.allclose(atol=1e-5, rtol=1e-4)`（fp16/bf16）
- **显存/吞吐**: `experiments/bench_checkpoint_memory.py` 用 `torch.cuda.reset_peak_memory_stats()` 对比 Mode 0/1/2
- **trainer smoke**: 短训练步（1-2 step）loss 有限、无崩溃

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - foundation, MAX PARALLEL):
├── T1: pytest 测试基建 [quick]
├── T2: checkpointing.py 注意力重计算 Function (TDD) [deep]
├── T3: checkpointing.py FFN checkpoint 助手 + aux_loss 封装 (TDD) [deep]
├── T4: flag 线程化: 3 configs + config_from_args + 8 trainers [unspecified-high]
├── T5: WebUI 死 flag 移除 + 新 flag 接入 [unspecified-low]
└── T6: README 文档更新 [writing]

Wave 2 (After Wave 1 - 模型接入, 3 路并行):
├── T7: Dense 变体接入 (model_instinct.py: Attention eager 分支 + InstinctBlock Mode2) [deep]
├── T8: Loop 变体接入 (model_instinct_loop.py: loop body + Attention) [deep]
└── T9: Linear 变体接入 (model_instinct_linear.py: 标准部分 Mode1 + 整层 Mode2) [deep]

Wave 3 (After Wave 2 - 集成 + 验证):
├── T10: 跨模型梯度/step 一致性测试套件 [unspecified-high]
├── T11: GPU 显存/吞吐基准脚本 + 测量 [deep]
├── T12: 8-trainer smoke 测试 [unspecified-low]
└── T13: flag 线程化验证 (config_path JSON / resume / 死 flag grep) [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T2 → T7 → T10 → F1-F4 → user okay
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 5 (Wave 1)
```

### Dependency Matrix (full)

| Task | Blocked By | Blocks |
|------|-----------|--------|
| T1 | — | T2, T3, T10, T12 |
| T2 | T1 | T7, T8, T9, T10 |
| T3 | T1 | T7, T8, T9, T10 |
| T4 | — | T7, T8, T9, T13 |
| T5 | — | T13 |
| T6 | — | — (可在 W1 并行) |
| T7 | T2, T3, T4 | T10, T11, T12 |
| T8 | T2, T3, T4 | T10, T11, T12 |
| T9 | T2, T3, T4 | T10, T11, T12 |
| T10 | T7, T8, T9 | F1-F4 |
| T11 | T7, T8, T9 | F1-F4 |
| T12 | T7, T8, T9 | F1-F4 |
| T13 | T4, T5 | F1-F4 |

### Agent Dispatch Summary

- **Wave 1**: T1 → `quick`, T2 → `deep`, T3 → `deep`, T4 → `unspecified-high`, T5 → `unspecified-low`, T6 → `writing`
- **Wave 2**: T7/T8/T9 → `deep`
- **Wave 3**: T10 → `unspecified-high`, T11 → `deep`, T12 → `unspecified-low`, T13 → `quick`
- **FINAL**: F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. 搭建 pytest 测试基建（TDD 前置）

  **What to do**:
  - 在 `requirements.txt` 追加 `pytest`（保持与现有 pin 风格一致，如 `pytest==8.3.4`）
  - 创建 `tests/conftest.py`：⚠️ **必须先 import `datasets` 再 import `torch`**（Windows pyarrow/torch DLL 冲突 workaround，见 AGENTS.md，不可乱序）；注册 `gpu` marker（`pytest.ini` 或 `pyproject.toml` 配置 `markers = ["gpu: needs CUDA"]` + `addopts`）；添加 `--skip-gpu` 处理（无 CUDA 时自动跳过 `@pytest.mark.gpu` 测试）
  - 创建 `tests/helpers.py`：提供 `make_tiny_config(use_moe=False, variant="dense")` — 返回极小 InstinctConfig（hidden_size=64, num_hidden_layers=2, num_heads=4/2kv, max_pos_len=512, vocab=256）供所有测试复用；提供 `assert_grads_equal(m0, m1, atol, rtol)` 助手（对比两模型 state_dict 中每个参数的 grad，`torch.equal` 或 `torch.allclose`）
  - 验证 pytest 可运行：`python -m pytest tests/ -x -q`（初始 0 测试也能 collect 通过）

  **Must NOT do**:
  - 不引入 CI 基建、不配置 coverage、不改现有 requirements pin 风格
  - 不把 `datasets`/`torch` import 顺序改掉

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 标准文件搭建任务，无算法复杂度
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T2, T3, T4, T5)
  - **Blocks**: T2, T3, T10, T12
  - **Blocked By**: None (start immediately)

  **References**:
  **Pattern References**:
  - `trainer/train_pretrain.py:8` — `import datasets` 在 `import torch` 之前（Windows pyarrow/torch DLL 冲突 workaround，AGENTS.md 强调不可乱序）
  - `requirements.txt:1-32` — 现有 pin 风格（`pkg==version`，注释掉的可用 `# pytest==8.3.4` 占位模式）
  **External References**:
  - 官方文档: `https://docs.pytest.org/en/stable/` - marker 配置与 conftest 约定

  **Acceptance Criteria**:
  - [ ] `requirements.txt` 含 pytest 条目
  - [ ] `tests/conftest.py` 存在，`datasets` 在 `torch` 之前 import
  - [ ] `tests/helpers.py` 存在，`make_tiny_config()` 返回可实例化的极小 config
  - [ ] `python -m pytest tests/ -x -q` → collect 成功（0 passed 或初始测试通过）

  **QA Scenarios**:
  ```
  Scenario: pytest collect 成功
    Tool: Bash
    Preconditions: tests/ 目录存在，conftest.py 就位
    Steps:
      1. 运行 `python -m pytest tests/ -x -q --collect-only`
      2. 断言 exit code 0
    Expected Result: pytest 正常收集，无 import 错误（验证 datasets-before-torch 顺序生效）
    Failure Indicators: ModuleNotFoundError / DLL load failed / pyarrow 冲突
    Evidence: .sisyphus/evidence/task-1-pytest-collect.txt

  Scenario: tiny config 可实例化
    Tool: Bash
    Preconditions: tests/helpers.py 就位
    Steps:
      1. 运行 `python -c "import sys; sys.path.insert(0,'tests'); from helpers import make_tiny_config; c = make_tiny_config(); print(c.hidden_size, c.num_hidden_layers)"`
      2. 断言输出包含 `64 2`
    Expected Result: 打印 `64 2`
    Failure Indicators: 异常 / 打印错误值
    Evidence: .sisyphus/evidence/task-1-tiny-config.txt
  ```

  **Evidence to Capture**:
  - [ ] task-1-pytest-collect.txt, task-1-tiny-config.txt

  **Commit**: YES
  - Message: `test: add pytest infrastructure for gradient checkpointing`
  - Files: requirements.txt, tests/conftest.py, tests/helpers.py
  - Pre-commit: `python -m pytest tests/ -x -q`

- [x] 2. 实现注意力选择性重计算 autograd.Function（TDD）

  **What to do**:
  - 创建 `model/checkpointing.py`，实现无参数注意力重计算：
    ```python
    class RecomputeAttention(torch.autograd.Function):
        # forward(ctx, q, k, v, attention_mask, is_causal, dropout_p, scale)
        #   - 输入为 pre-repeat_kv 的 q/k/v（shape [bs, seq, heads, head_dim]）
        #   - forward 内: transpose → repeat_kv → QKᵀ → causal mask → softmax.float() → dropout → @V → transpose/reshape
        #   - ctx.save_for_backward(q, k, v, attention_mask)
        #   - ctx 保存 is_causal/dropout_p/scale + RNG 状态（torch.cuda.get_rng_state / torch.random.get_rng_state）
        # backward(ctx, grad_out)
        #   - 恢复 RNG 状态（保证 dropout mask 一致）
        #   - 重算 scores/softmax/dropout/@V（在保存张量自身 dtype 下，autocast 在 backward 不生效）
        #   - 返回 grad_q, grad_k, grad_v（注意力区域无参数 → 只需 grad_input）
    ```
  - 严格复制 `model/model_instinct.py:140-144` eager 分支的计算语义（scores / sqrt(head_dim)、`is_causal` 的 `triu(1)` 掩码、`softmax(scores.float()).type_as(xq)`、`attn_dropout`）
  - TDD：先写 `tests/test_checkpoint_attention.py` — 用 `make_tiny_config()` 构造的随机 q/k/v，断言 `RecomputeAttention` 的输出与手写 eager 注意力完全一致（`torch.equal`，CPU fp32）；再实现
  - 测试覆盖：`is_causal=True/False`、`attention_mask` 有无、`dropout_p=0`（默认）与 `dropout_p>0`（验证 RNG 重放）、GQA `repeat_kv`（n_rep=2）

  **Must NOT do**:
  - 不在此 Function 内包含任何带参数的层（QKV 投影、o_proj 都在外面）— 参数梯度陷阱
  - 不实现 FFN 重算（那是 T3）
  - 不修改现有 model_instinct.py（接入在 T7）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 自定义 autograd 语义精确性要求高（RNG、dtype、掩码语义）
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T3, T4, T5)
  - **Blocks**: T7, T8, T9, T10
  - **Blocked By**: T1

  **References**:
  **Pattern References**:
  - `model/model_instinct.py:135-145` - eager 注意力分支：`scores = (xq @ xk.transpose(-2,-1)) / sqrt(head_dim)`、causal `triu` 掩码、`softmax(scores.float(), dim=-1).type_as(xq)`、`attn_dropout(...) @ xv` — 这是重算必须逐字节复制的语义
  - `model/model_instinct.py:87-96` - `apply_rotary_pos_emb`（q/k 在进入 Function 前已完成 RoPE）
  - `model/model_instinct.py:140` - `repeat_kv`（GQA n_rep=2）— 重算发生在 Function 内部
  - `model/model_instinct.py:112-114` - `attn_dropout`/`resid_dropout` 定义
  **API/Type References**:
  - `model/model_instinct.py:98-117` - `Attention` 类的 q/k/v shape：`[bs, seq, n_heads, head_dim]`，q_heads=8, kv_heads=4
  **External References**:
  - Korthikanti et al. 2022 实现模式: diffusers `attention_dispatch.py` (SHA 1b98ae1, L819-891) — `ctx.save_for_backward(query, key, value)` + backward 重跑 SDPA + `torch.autograd.grad`
  - Megatron-LM `tensor_parallel/random.py` L591-676 (SHA d96d76c) — `CheckpointFunction`: no_grad forward + `_get_all_rng_states` + enable_grad 重算 + `torch.autograd.backward`（RNG 处理参考）

  **Acceptance Criteria**:
  - [ ] `tests/test_checkpoint_attention.py` 先写（RED）：输出与手写 eager 注意力不一致则失败
  - [ ] `model/checkpointing.py` 的 `RecomputeAttention` 通过全部测试（GREEN）
  - [ ] CPU fp32: `torch.equal` 输出/梯度 vs 手写实现
  - [ ] dropout_p>0: 两次 forward（同 seed）mask 一致
  - [ ] `python -m pytest tests/test_checkpoint_attention.py -x -q` 全绿

  **QA Scenarios**:
  ```
  Scenario: 输出与手写 eager 注意力一致 (is_causal=True, dropout=0)
    Tool: Bash (pytest)
    Preconditions: tests/test_checkpoint_attention.py 就位
    Steps:
      1. 运行 `python -m pytest tests/test_checkpoint_attention.py::test_output_causal -x -q`
      2. 断言 pytest exit code 0
    Expected Result: 测试通过（输出 torch.equal）
    Failure Indicators: torch.equal 失败 → 重算语义与 eager 分支不一致
    Evidence: .sisyphus/evidence/task-2-output-causal.txt

  Scenario: dropout RNG 重放一致性 (dropout_p=0.1)
    Tool: Bash (pytest)
    Preconditions: 测试代码就位
    Steps:
      1. 运行 `python -m pytest tests/test_checkpoint_attention.py::test_dropout_rng -x -q`
      2. 断言 exit code 0
    Expected Result: 两次 forward 同 seed 输出完全一致（mask 重放成功）
    Failure Indicators: 不一致 → RNG 状态未正确保存/恢复
    Evidence: .sisyphus/evidence/task-2-dropout-rng.txt
  ```

  **Evidence to Capture**:
  - [ ] task-2-output-causal.txt, task-2-dropout-rng.txt

  **Commit**: YES
  - Message: `feat(checkpointing): selective attention recompute autograd function`
  - Files: model/checkpointing.py, tests/test_checkpoint_attention.py
  - Pre-commit: `python -m pytest tests/test_checkpoint_attention.py -x -q`

- [x] 3. 实现 FFN checkpoint 助手 + MoE aux_loss 返回封装（TDD）

  **What to do**:
  - 在 `model/checkpointing.py` 实现 FFN/MoE 区域 checkpoint 助手（**参数安全，用 torch.utils.checkpoint**）：
    ```python
    def checkpoint_ffn(ffn_module, hidden_states, use_reentrant=False):
        """包装 FeedForward/MOEFeedForward 的 forward。
        forward 时只保存 hidden_states 输入；backward 时 torch.utils.checkpoint
        内部 enable_grad 重算整个 FFN（参数梯度正确累积）。
        返回 (output, aux_loss)；aux_loss 从返回值取，不依赖模块属性侧信道。"""
        def run(h):
            out = ffn_module(h)
            aux = getattr(ffn_module, "aux_loss", None)
            return out, aux
        return torch.utils.checkpoint.checkpoint(run, hidden_states, use_reentrant=use_reentrant, preserve_rng_state=True)
    ```
  - 关键：`torch.utils.checkpoint` 的 forward 在 no_grad 下运行，backward 内 enable_grad 重算 → **参数（gate/up/down_proj、expert 权重）梯度正确**；`use_reentrant=False` 支持非张量/元组返回
  - TDD：`tests/test_checkpoint_ffn.py` — 构造带参数的极小 FeedForward（及 `MOEFeedForward` tiny 版），断言 `checkpoint_ffn` 后参数梯度非零且与手写 forward+backward 梯度 `torch.equal`（CPU fp32）；断言返回的 `aux_loss` 有 grad_fn 且梯度非零
  - 测试覆盖：dense FFN（SwiGLU）、MoE FFN（top-1 routing + aux_loss）、dropout>0 RNG 一致

  **Must NOT do**:
  - 不为 FFN 写自定义 autograd.Function（参数梯度陷阱）— 必须用 torch.utils.checkpoint
  - 不改 `model/model_instinct.py` 的 MOEFeedForward 内部（接入在 T7）
  - 不使用 `use_reentrant=True`（in-place 冲突 + 已弃用）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 涉及 torch.utils.checkpoint 内部语义 + 参数梯度正确性验证
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2, T4, T5)
  - **Blocks**: T7, T8, T9, T10
  - **Blocked By**: T1

  **References**:
  **Pattern References**:
  - `model/model_instinct.py:149-159` - `FeedForward`（SwiGLU: gate_proj/up_proj/down_proj, silu 乘积）
  - `model/model_instinct.py:161-189` - `MOEFeedForward`：router `gate` Linear + top-1 softmax + per-expert `index_add_` 循环 + `self.aux_loss` 属性（184-188）
  - `model/model_instinct.py:247-248` - `InstinctModel.forward` 读取 `layer.mlp.aux_loss` 累加（这是接入时需改为返回值读取的位置）
  **External References**:
  - PyTorch docs: `https://pytorch.org/docs/stable/checkpoint.html` - `torch.utils.checkpoint.checkpoint` 语义（use_reentrant、preserve_rng_state、非张量参数支持）

  **Acceptance Criteria**:
  - [ ] `tests/test_checkpoint_ffn.py` 先写（RED）
  - [ ] `checkpoint_ffn` 实现通过（GREEN）：参数梯度非零且与手写 `torch.equal`
  - [ ] MoE 场景：`aux_loss` 返回值有 grad_fn、`gate.weight.grad` 非零
  - [ ] `python -m pytest tests/test_checkpoint_ffn.py -x -q` 全绿

  **QA Scenarios**:
  ```
  Scenario: FFN 参数梯度正确 (dense SwiGLU)
    Tool: Bash (pytest)
    Preconditions: tests/test_checkpoint_ffn.py 就位
    Steps:
      1. 运行 `python -m pytest tests/test_checkpoint_ffn.py::test_ffn_param_grads -x -q`
      2. 断言 exit code 0
    Expected Result: gate/up/down_proj 梯度非零且 torch.equal 手写实现
    Failure Indicators: 零梯度 / 不相等 → 参数梯度陷阱未解决
    Evidence: .sisyphus/evidence/task-3-ffn-grads.txt

  Scenario: MoE aux_loss 梯度流动
    Tool: Bash (pytest)
    Preconditions: 测试代码就位
    Steps:
      1. 运行 `python -m pytest tests/test_checkpoint_ffn.py::test_moe_aux_loss_grad -x -q`
      2. 断言 exit code 0
    Expected Result: gate.weight.grad 非零，aux_loss.grad_fn 存在
    Failure Indicators: 零梯度 → aux_loss 侧信道梯度丢失
    Evidence: .sisyphus/evidence/task-3-moe-aux.txt
  ```

  **Evidence to Capture**:
  - [ ] task-3-ffn-grads.txt, task-3-moe-aux.txt

  **Commit**: YES
  - Message: `feat(checkpointing): FFN checkpoint helper + aux_loss return wrapper`
  - Files: model/checkpointing.py, tests/test_checkpoint_ffn.py
  - Pre-commit: `python -m pytest tests/test_checkpoint_ffn.py -x -q`

- [x] 4. flag 线程化: 3 个 config + config_from_args + 8 个 trainer argparse

  **What to do**:
  - `model/model_instinct.py:InstinctConfig.__init__`（:12-52 区域）追加 `self.use_grad_checkpoint = kwargs.get("use_grad_checkpoint", 0)`（跟随现有 `kv_cache_dtype` 的 `kwargs.get` 模式，加 `### Gradient Checkpointing configs` 注释头）
  - 同样追加到 `model/model_instinct_loop.py:InstinctConfig` 与 `model/model_instinct_linear.py` 的 config（3 个文件结构相同）
  - `trainer/trainer_utils.py:config_from_args`（:38-62）追加 `overrides.setdefault("use_grad_checkpoint", int(getattr(args, "use_grad_checkpoint", 0)))`（跟随 param_dtype/kv_cache_dtype 现有模式）
  - 8 个 trainer 的 argparse 块追加（跟随 `--use_compile` 模式，中文 help）：
    `parser.add_argument("--use_grad_checkpoint", default=0, type=int, choices=[0, 1, 2], help="梯度检查点模式（0=关闭, 1=选择性重算注意力/FFN, 2=整层checkpoint）")`
    文件: train_pretrain.py, train_full_sft.py, train_lora.py, train_dpo.py, train_ppo.py, train_grpo.py, train_agent.py, train_distillation.py
  - **移除死 flag**: 从 `train_pretrain.py` 删除 `--loop_grad_checkpoint` 全部 3 处（argparse :115、config 赋值 :166-167、Logger f-string :172），后续 T5 清理 WebUI
  - 测试：`tests/test_flag_threading.py` — 对 3 个 config 断言 `use_grad_checkpoint` 默认 0、可被 kwargs 覆盖；用 pytest monkeypatch 模拟 args 对象调用 `config_from_args` 断言 flag 传递

  **Must NOT do**:
  - 不在 config 里实现重算逻辑（那是 T2/T3/T7-T9）
  - 不添加除 `use_grad_checkpoint` 之外的任何新 flag
  - 不删除其他 trainer 参数

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 多文件机械修改（11+ 文件），需保持一致性，但无算法复杂度
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2, T3, T5)
  - **Blocks**: T7, T8, T9, T13
  - **Blocked By**: None (start immediately)

  **References**:
  **Pattern References**:
  - `model/model_instinct.py:12-52` - InstinctConfig 的 kwargs.get 模式（如 `self.kv_cache_dtype = kwargs.get("kv_cache_dtype", "fp32")`）
  - `trainer/trainer_utils.py:38-62` - config_from_args 的 overrides.setdefault 模式（param_dtype/kv_cache_dtype 现成例子）
  - `trainer/train_pretrain.py:115` - `--loop_grad_checkpoint` 死 flag（要删除）
  - `trainer/train_pretrain.py:166-167` - `model.config.loop_grad_checkpoint = True` 赋值（要删除）
  - `trainer/train_pretrain.py:108-126` - argparse 块现有模式（`--use_compile` 等，中文 help, choices=[0,1]）
  - `trainer/train_lora.py:169-171` - use_compile 自动禁用模式（后续 T7-T9 若需要可参考）

  **Acceptance Criteria**:
  - [ ] 3 个 config 类均有 `use_grad_checkpoint` 属性，默认 0
  - [ ] config_from_args 传递 flag（tests/test_flag_threading.py 覆盖）
  - [ ] 8 个 trainer 均有 `--use_grad_checkpoint` argparse（`grep -c use_grad_checkpoint trainer/*.py` → 8+）
  - [ ] `grep -r loop_grad_checkpoint trainer/train_pretrain.py` → 零匹配（T5 清 WebUI 后全仓零残留）
  - [ ] `python -m pytest tests/test_flag_threading.py -x -q` 全绿

  **QA Scenarios**:
  ```
  Scenario: config 默认值与覆盖
    Tool: Bash (pytest)
    Preconditions: tests/test_flag_threading.py 就位
    Steps:
      1. 运行 `python -m pytest tests/test_flag_threading.py::test_config_default_and_override -x -q`
      2. 断言 exit code 0
    Expected Result: 默认 0；kwargs 覆盖为 1/2 生效
    Failure Indicators: 断言失败 → config 未接线
    Evidence: .sisyphus/evidence/task-4-config.txt

  Scenario: 死 flag 移除验证（trainer 侧）
    Tool: Bash
    Preconditions: 本任务完成
    Steps:
      1. 运行 `grep -rn "loop_grad_checkpoint" trainer/ 2>/dev/null || echo "CLEAN"`
      2. 断言输出为 CLEAN
    Expected Result: trainer/ 下无残留
    Failure Indicators: 有匹配行 → 未清理干净
    Evidence: .sisyphus/evidence/task-4-deadflag-trainer.txt
  ```

  **Evidence to Capture**:
  - [ ] task-4-config.txt, task-4-deadflag-trainer.txt

  **Commit**: YES
  - Message: `feat(config): thread use_grad_checkpoint flag through configs and 8 trainers`
  - Files: model/model_instinct.py, model/model_instinct_loop.py, model/model_instinct_linear.py, trainer/trainer_utils.py, trainer/train_*.py
  - Pre-commit: `python -m pytest tests/test_flag_threading.py -x -q && grep -rn "loop_grad_checkpoint" trainer/ || echo CLEAN`

- [x] 5. WebUI 死 flag 移除 + 新 flag 接入

  **What to do**:
  - `scripts/config_webui.py` 中删除 `loop_grad_checkpoint` 全部引用（grep 确认 8 处：config dict :609, CLI 参数构建 :685-686, 持久化 :763/:856, UI checkbox :1520-1522）
  - 添加 `use_grad_checkpoint` 的 WebUI 支持：跟随现有 `loop_grad_checkpoint` checkbox 的接线模式（st.session_state → build_config_dict → gen_python_code → launch command）
  - UI 显示: 下拉/数字选择 0/1/2，label 类似"梯度检查点模式（0关闭/1选择性/2整层）"
  - 验证: 启动 WebUI 不崩溃（可 headless 检查 `streamlit run scripts/web_demo.py --server.headless true` 或仅 import 检查）

  **Must NOT do**:
  - 不重设计 WebUI 布局
  - 不修改 web_demo.py 其他逻辑

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: 机械的 UI 接线，与现有模式一致
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2, T3, T4)
  - **Blocks**: T13
  - **Blocked By**: None (start immediately; 建议与 T4 同步以免 config 键不一致)

  **References**:
  **Pattern References**:
  - `scripts/config_webui.py:1519-1527` - `loop_grad_checkpoint` UI checkbox（要删除/替换）
  - `scripts/config_webui.py:609, 685-686, 763, 856` - loop_grad_checkpoint 接线点
  - `scripts/config_webui.py` 现有 `use_moe`/`use_compile` 的 build_config_dict → gen_python_code → launch command 完整链路（新 flag 跟随）

  **Acceptance Criteria**:
  - [ ] `grep -rn "loop_grad_checkpoint" scripts/config_webui.py` → 零匹配
  - [ ] `use_grad_checkpoint` 出现在 build_config_dict / gen_python_code / launch command / session load 各处
  - [ ] `python -c "import scripts.config_webui"`（或等价 import 检查）无语法错误

  **QA Scenarios**:
  ```
  Scenario: WebUI 零残留死 flag
    Tool: Bash
    Preconditions: 本任务完成
    Steps:
      1. 运行 `grep -rn "loop_grad_checkpoint" scripts/ 2>/dev/null || echo "CLEAN"`
      2. 断言输出为 CLEAN
    Expected Result: scripts/ 下无残留
    Failure Indicators: 有匹配行
    Evidence: .sisyphus/evidence/task-5-webui-clean.txt

  Scenario: WebUI import 检查
    Tool: Bash
    Preconditions: 修改完成
    Steps:
      1. 运行 `cd scripts && python -c "import config_webui; print('OK')"`
      2. 断言输出含 OK
    Expected Result: import 无错误
    Failure Indicators: SyntaxError / ImportError
    Evidence: .sisyphus/evidence/task-5-webui-import.txt
  ```

  **Evidence to Capture**:
  - [ ] task-5-webui-clean.txt, task-5-webui-import.txt

  **Commit**: YES
  - Message: `refactor(webui): remove dead loop_grad_checkpoint, expose use_grad_checkpoint`
  - Files: scripts/config_webui.py
  - Pre-commit: `cd scripts && python -c "import config_webui"`

- [x] 6. README 文档更新

  **What to do**:
  - `README.md` 的"常用训练参数"表（含 `--use_moe`/`--use_looped` 的表）新增行：
    `| --use_grad_checkpoint 0\|1\|2 | 梯度检查点（0=关闭, 1=选择性重算注意力QKᵀ/FFN, 2=整层checkpoint） |`
  - 可选：在"训练管线"或"注意事项"补 1-2 句说明（eager 路径生效、flash 路径省 FFN 中间量、长序列收益更大）

  **Must NOT do**:
  - 不重写 README 其他内容

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: 文档任务
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (任何任务)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `README.md` 的"常用训练参数"表（`--use_moe 1 | 启用 MoE 架构` 行所在表格）

  **Acceptance Criteria**:
  - [ ] README 常用训练参数表含 `--use_grad_checkpoint` 行

  **QA Scenarios**:
  ```
  Scenario: README 文档存在
    Tool: Bash
    Preconditions: 本任务完成
    Steps:
      1. 运行 `grep -n "use_grad_checkpoint" README.md`
      2. 断言 exit code 0 且输出行包含 0\|1\|2 说明
    Expected Result: 文档行存在
    Failure Indicators: 无匹配
    Evidence: .sisyphus/evidence/task-6-readme.txt
  ```

  **Evidence to Capture**:
  - [ ] task-6-readme.txt

  **Commit**: YES
  - Message: `docs: document use_grad_checkpoint flag`
  - Files: README.md

- [x] 7. Dense 变体接入（model_instinct.py: Mode 1 选择性 + Mode 2 整层）

  **What to do**:
  - **Mode 1（选择性）**: 在 `Attention.forward`（model_instinct.py:119-147）的 eager 分支（:140-144）将 scores→softmax→dropout→@V 用 `RecomputeAttention`（T2 产出）替换：
    ```python
    if self.config.use_grad_checkpoint == 1 and self.training and not use_flash:
        # 用 RecomputeAttention.apply(xq_pre, xk_pre, xv_pre, mask, is_causal, dropout_p, scale)
        # 保存 pre-repeat_kv 的 q/k/v，Function 内部重算 transpose+repeat_kv+QKᵀ+softmax+dropout+@V
        output = RecomputeAttention.apply(xq, xk, xv, attention_mask, self.is_causal, self.attn_dropout.p, 1.0/math.sqrt(self.head_dim))
    ```
  - **Mode 2（整层）**: 在 `InstinctModel.forward` 的 layer 循环（:238-245）用 `torch.utils.checkpoint.checkpoint(block_forward, hidden_states, position_embeddings, use_reentrant=False, preserve_rng_state=True)` 包裹 `InstinctBlock.forward`；block forward 返回 `(hidden_states, present)`，用闭包函数处理
  - **MoE aux_loss 修正**: `InstinctModel.forward` 的 aux_loss 累加（:247-248）改为从 checkpoint 闭包返回值取（T3 的 `checkpoint_ffn` 返回 `(out, aux)`）；非 checkpoint 路径仍用 `layer.mlp.aux_loss`
  - 保留 flash 路径（:135-138）原样 — Mode 1 只在 eager 分支生效；Mode 2 与 flash 共存（flash 在 block 内被重算）
  - `self.training` 守卫: 推理/rollout 不启用 checkpoint
  - 测试: `tests/test_dense_model.py` — Mode 0/1/2 三态下 forward loss 一致（CPU fp32 `torch.equal`）、参数梯度一致；MoE 变体 `use_moe=1` 下 gate.weight.grad 非零

  **Must NOT do**:
  - 不改 flash 调用路径（:135-138）
  - 不改 `FeedForward`/`MOEFeedForward` 类内部（T3 的助手已封装）
  - 不修改 loop/linear 文件（T8/T9）
  - 不使用 `use_reentrant=True`

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 模型内部接入，需精确处理 autograd 语义、闭包返回、MoE aux_loss
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T8, T9)
  - **Blocks**: T10, T11, T12
  - **Blocked By**: T2, T3, T4

  **References**:
  **Pattern References**:
  - `model/model_instinct.py:119-147` - Attention.forward（eager 分支 :140-144 是 Mode 1 接入点）
  - `model/model_instinct.py:135` - flash 条件判断（`self.flash and seq_len>1 and ...`）— Mode 1 只在 flash 不满足时生效
  - `model/model_instinct.py:238-245` - layer 循环（Mode 2 接入点）
  - `model/model_instinct.py:247-248` - aux_loss 读取点（需改为返回值）
  - `model/model_instinct.py:199-207` - InstinctBlock.forward 签名 `(hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None)`
  - `model/checkpointing.py` - T2/T3 产出的 RecomputeAttention + checkpoint_ffn
  **Test References**:
  - `tests/helpers.py` - make_tiny_config(use_moe=True/False)
  - `tests/test_checkpoint_attention.py` / `tests/test_checkpoint_ffn.py` - T2/T3 的测试模式

  **Acceptance Criteria**:
  - [ ] Mode 0（默认）行为不变：`tests/test_dense_model.py::test_mode0_unchanged` — 与接入前 bitwise 一致（固定种子）
  - [ ] Mode 1: eager 分支走 RecomputeAttention；loss 与 Mode 0 `torch.equal`（CPU fp32）
  - [ ] Mode 2: block 走 torch.utils.checkpoint；loss 与 Mode 0 `torch.equal`
  - [ ] MoE: gate.weight.grad 非零（aux_loss 修正生效）
  - [ ] `python -m pytest tests/test_dense_model.py -x -q` 全绿

  **QA Scenarios**:
  ```
  Scenario: 三态 loss 一致性（dense）
    Tool: Bash (pytest)
    Preconditions: tests/test_dense_model.py 就位
    Steps:
      1. 运行 `python -m pytest tests/test_dense_model.py::test_mode012_loss_equal -x -q`
      2. 断言 exit code 0
    Expected Result: Mode 0/1/2 前向 loss torch.equal
    Failure Indicators: 不一致 → 重算语义偏差 / autocast dtype 问题
    Evidence: .sisyphus/evidence/task-7-loss-equal.txt

  Scenario: MoE router 梯度非零
    Tool: Bash (pytest)
    Preconditions: use_moe=1 测试就位
    Steps:
      1. 运行 `python -m pytest tests/test_dense_model.py::test_moe_router_grad -x -q`
      2. 断言 exit code 0
    Expected Result: gate.weight.grad 非零（Mode 1 和 Mode 2 都验证）
    Failure Indicators: 零梯度 → aux_loss 侧信道梯度丢失
    Evidence: .sisyphus/evidence/task-7-moe-grad.txt
  ```

  **Evidence to Capture**:
  - [ ] task-7-loss-equal.txt, task-7-moe-grad.txt

  **Commit**: YES
  - Message: `feat(dense): wire selective + full-layer checkpoint into dense model`
  - Files: model/model_instinct.py, tests/test_dense_model.py
  - Pre-commit: `python -m pytest tests/test_dense_model.py -x -q`

- [x] 8. Loop 变体接入（model_instinct_loop.py: loop body + Attention）

  **What to do**:
  - Loop 变体的 `Attention` 类（model_instinct_loop.py:100-149）与 dense 几乎相同 — 用与 T7 相同的方式接入 Mode 1（eager 分支 → RecomputeAttention）
  - **Mode 2**: loop body（:334-356）的 `self.loop_block(updated, position_embeddings, ...)` 调用用 `torch.utils.checkpoint` 包裹（**这就是原死 flag `loop_grad_checkpoint` 承诺的功能**）
  - **MoE aux_loss**: loop 模型的 aux_loss 收集（:375-383，从 unique_blocks 累计）改为从 checkpoint 闭包返回值取
  - `self.training` 守卫
  - 测试: `tests/test_loop_model.py` — 用 loop 变体 config（`model_instinct_loop.py` 中的 `InstinctConfig`，在 `trainer_utils.py:19` 别名为 `LoopedInstinctConfig`）极小配置验证 Mode 0/1/2 loss 一致、梯度一致、aux_loss 梯度非零（loop_iters 用小值如 2）

  **Must NOT do**:
  - 不改 prelude/coda 逻辑（:316-328, :359-371）
  - 不改 dense/linear 文件

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 与 T7 同复杂度，但 loop 共享 block 多次调用，checkpoint 语义需额外注意（同一 block 被 loop 多次重算）
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T7, T9)
  - **Blocks**: T10, T11, T12
  - **Blocked By**: T2, T3, T4

  **References**:
  **Pattern References**:
  - `model/model_instinct_loop.py:100-149` - Loop 变体 Attention（复制自 dense）
  - `model/model_instinct_loop.py:334-356` - loop body（prelude → loop_block → coda 结构；`self.loop_block(updated, position_embeddings, ...)` 在 :345-350 附近）
  - `model/model_instinct_loop.py:375-383` - aux_loss 从 unique_blocks 累计
  - `model/model_instinct_loop.py:193-210` - loop 的 InstinctBlock（与 dense 同构）
  - `model/model_instinct_loop.py:12-54` - loop 变体 `InstinctConfig`（trainer_utils.py:19 别名为 LoopedInstinctConfig；T4 已加 use_grad_checkpoint）

  **Acceptance Criteria**:
  - [ ] Mode 0 行为不变
  - [ ] Mode 1: loop Attention eager 分支走 RecomputeAttention；loss 与 Mode 0 `torch.equal`
  - [ ] Mode 2: loop_block 调用走 torch.utils.checkpoint（loop_iters=2 时每次迭代重算）
  - [ ] aux_loss 梯度非零
  - [ ] `python -m pytest tests/test_loop_model.py -x -q` 全绿

  **QA Scenarios**:
  ```
  Scenario: Loop 三态 loss 一致性
    Tool: Bash (pytest)
    Preconditions: tests/test_loop_model.py 就位
    Steps:
      1. 运行 `python -m pytest tests/test_loop_model.py::test_mode012_loss_equal -x -q`
      2. 断言 exit code 0
    Expected Result: Mode 0/1/2 loss torch.equal
    Failure Indicators: 不一致 → loop 内共享 block 重算语义问题
    Evidence: .sisyphus/evidence/task-8-loop-loss.txt

  Scenario: Loop + MoE aux_loss 梯度
    Tool: Bash (pytest)
    Preconditions: use_moe=1 loop 测试就位
    Steps:
      1. 运行 `python -m pytest tests/test_loop_model.py::test_moe_aux_loss_grad -x -q`
      2. 断言 exit code 0
    Expected Result: gate.weight.grad 非零
    Failure Indicators: 零梯度
    Evidence: .sisyphus/evidence/task-8-loop-moe.txt
  ```

  **Evidence to Capture**:
  - [ ] task-8-loop-loss.txt, task-8-loop-moe.txt

  **Commit**: YES
  - Message: `feat(loop): wire checkpoint into loop body (fulfills loop_grad_checkpoint)`
  - Files: model/model_instinct_loop.py, tests/test_loop_model.py
  - Pre-commit: `python -m pytest tests/test_loop_model.py -x -q`

- [x] 9. Linear 变体接入（model_instinct_linear.py: 标准部分 Mode 1 + 整层 Mode 2）

  **What to do**:
  - Linear 变体的标准 `Attention` 类（model_instinct_linear.py:278-330，与 dense 近复制）接入 Mode 1（eager 分支 → RecomputeAttention）— 与 T7 相同方式
  - **Mode 2**: linear 变体的 block forward（:374-385 附近）用 `torch.utils.checkpoint` 包裹整个 block（**不深入 GatedDeltaNet 内部 kernel**）
  - **MoE aux_loss**: linear 变体的 MOEFeedForward aux_loss 读取点（:366-371/:444）改为返回值
  - **GatedDeltaNet 区域**: 不在此模块内做选择性重算（FLA/Triton kernel 不可靠）；Mode 2 整层包裹时依赖 torch.utils.checkpoint 重跑整个 block forward（安全）
  - `self.training` 守卫
  - 测试: `tests/test_linear_model.py` — 通过 `model_instinct_linear` 直接实例化验证 Mode 0/1/2 loss 一致、梯度一致

  **Must NOT do**:
  - 不修改 GatedDeltaNet / FLA kernel 调用内部
  - 不改 dense/loop 文件
  - 不 promise Linear 变体有注意力选择性重算的额外显存收益（其标准 Attention 部分有）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 同 T7/T8 复杂度 + kernel 边界约束
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T7, T8)
  - **Blocks**: T10, T11, T12
  - **Blocked By**: T2, T3, T4

  **References**:
  **Pattern References**:
  - `model/model_instinct_linear.py:278-330` - Linear 变体标准 Attention（Mode 1 接入点）
  - `model/model_instinct_linear.py:374-385` - linear block forward（Mode 2 接入点）
  - `model/model_instinct_linear.py:182` - `torch.amp.autocast(enabled=False)` 的 GatedDeltaNet 特殊 dtype 处理（**不触碰**）
  - `model/model_instinct_linear.py:366-371` - MOEFeedForward aux_loss 读取点
  - `run_linear.py`（仓库根目录） - sys.modules hack 加载方式（`sys.modules["model.model_instinct"] = model.model_instinct_linear`，验证时用此路径；另见 config_webui.py:1956 注释）

  **Acceptance Criteria**:
  - [ ] Mode 0 行为不变
  - [ ] Mode 1: 标准 Attention eager 分支走 RecomputeAttention；loss 与 Mode 0 `torch.equal`
  - [ ] Mode 2: block 走 torch.utils.checkpoint；loss 与 Mode 0 `torch.equal`
  - [ ] GatedDeltaNet 内部未改动（git diff 验证）
  - [ ] `python -m pytest tests/test_linear_model.py -x -q` 全绿

  **QA Scenarios**:
  ```
  Scenario: Linear 三态 loss 一致性
    Tool: Bash (pytest)
    Preconditions: tests/test_linear_model.py 就位
    Steps:
      1. 运行 `python -m pytest tests/test_linear_model.py::test_mode012_loss_equal -x -q`
      2. 断言 exit code 0
    Expected Result: Mode 0/1/2 loss torch.equal
    Failure Indicators: 不一致 → kernel 边界问题 / dtype 特殊处理冲突
    Evidence: .sisyphus/evidence/task-9-linear-loss.txt

  Scenario: GatedDeltaNet 未改动
    Tool: Bash
    Preconditions: 本任务完成
    Steps:
      1. 运行 `git diff --stat model/model_instinct_linear.py`
      2. 断言 diff 不涉及 GatedDeltaNet 类内部行（如包含 kernel 调用的区域）
    Expected Result: 改动仅限标准 Attention/block forward 接入点
    Failure Indicators: diff 触碰 GatedDeltaNet 内部
    Evidence: .sisyphus/evidence/task-9-linear-diff.txt
  ```

  **Evidence to Capture**:
  - [ ] task-9-linear-loss.txt, task-9-linear-diff.txt

  **Commit**: YES
  - Message: `feat(linear): wire checkpoint into linear variant standard parts`
  - Files: model/model_instinct_linear.py, tests/test_linear_model.py
  - Pre-commit: `python -m pytest tests/test_linear_model.py -x -q`

- [x] 10. 跨模型梯度/step 一致性测试套件

  **What to do**:
  - 创建 `tests/test_cross_model_consistency.py`，覆盖 {dense, moe, loop, linear} × {Mode 0 vs 1, Mode 0 vs 2}:
    - **梯度一致性**: CPU fp32 `torch.equal`（参数 grads + 输入 grads）；fp16/bf16 `torch.allclose(atol=1e-5, rtol=1e-4)`
    - **step 一致性**: 固定 seed 42，3 个 optimizer step，断言 Mode 0 vs 1 loss 序列完全一致（防静默漂移）
    - **路径覆盖**: (a) flash 路径（flash_attn=True, 无 mask）, (b) eager + padding mask, (c) eager + KV cache（past_key_value）, (d) seq_len==1 — 每个路径梯度与 Mode 0 一致或明确 skip 文档化
  - 复用 T1 helpers、T2/T3 的测试基建

  **Must NOT do**:
  - 不引入 GPU 依赖（全部 CPU 可跑；GPU dtype 测试用 `@pytest.mark.gpu` 或跳过）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 综合测试套件，覆盖 4 变体 × 多路径
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T11, T12, T13)
  - **Blocks**: F1-F4
  - **Blocked By**: T7, T8, T9

  **References**:
  **Pattern References**:
  - `tests/helpers.py` - make_tiny_config + assert_grads_equal
  - `model/model_instinct.py:135` - flash 条件（构造 flash 路径测试：无 mask、seq>1、past_key_value=None）
  - `model/model_instinct.py:140-144` - eager 路径（构造 padding mask 测试）
  - `model/model_instinct.py:128-133` - KV cache 路径（past_key_value 传入）
  - `trainer/train_pretrain.py:35-46` - 训练 step 模式（loss.backward → optimizer.step）

  **Acceptance Criteria**:
  - [ ] 4 变体 × 2 模式梯度一致性测试全绿（CPU）
  - [ ] step 一致性：3 步 loss 序列 Mode 0 vs 1 完全一致
  - [ ] 4 条路径（flash/eager+mask/KV cache/seq1）覆盖
  - [ ] `python -m pytest tests/test_cross_model_consistency.py -x -q` 全绿

  **QA Scenarios**:
  ```
  Scenario: 全矩阵梯度一致性
    Tool: Bash (pytest)
    Preconditions: 测试套件就位
    Steps:
      1. 运行 `python -m pytest tests/test_cross_model_consistency.py -x -q -k "grad"`
      2. 断言 exit code 0
    Expected Result: 所有变体 × 模式组合梯度 torch.equal/allclose
    Failure Indicators: 任一失败 → 该变体重算语义偏差
    Evidence: .sisyphus/evidence/task-10-grad-matrix.txt

  Scenario: step 一致性回归
    Tool: Bash (pytest)
    Preconditions: 测试就位
    Steps:
      1. 运行 `python -m pytest tests/test_cross_model_consistency.py::test_step_equality -x -q`
      2. 断言 exit code 0
    Expected Result: Mode 0 vs 1 的 3 步 loss 序列完全一致
    Failure Indicators: 不一致 → 静默梯度漂移
    Evidence: .sisyphus/evidence/task-10-step.txt
  ```

  **Evidence to Capture**:
  - [ ] task-10-grad-matrix.txt, task-10-step.txt

  **Commit**: YES
  - Message: `test: cross-model gradient/step equality suite`
  - Files: tests/test_cross_model_consistency.py
  - Pre-commit: `python -m pytest tests/test_cross_model_consistency.py -x -q`

- [x] 11. GPU 显存/吞吐基准脚本 + 测量

  **What to do**:
  - 创建 `experiments/bench_checkpoint_memory.py`:
    - 参数: `--bs 8 --seq 2048 --layers 8 --hidden 768 --moe 0/1 --mode 0/1/2 --steps 5 --flash 0/1`
    - 每个 mode: `torch.cuda.reset_peak_memory_stats()` → 跑 5 步训练（前向+反向+opt step）→ 打印 `peak_allocated_MB`、`steps_per_sec`
    - 输出表格: Mode 0/1/2 × {peak MB, 节省 %, steps/s, 吞吐开销 %}
    - 两个序列档: S=768 与 S=2048；两个路径档: flash 开 / padding mask 强制 eager
  - **验收断言**（脚本内置，exit code 非零即失败）:
    - S=2048 eager: Mode 1 峰值显存节省 ≥45%（推导值 55%），Mode 2 ≥85%
    - 计算开销: Mode 1 ≤15%（推导 ~10%），Mode 2 ≤50%（论文 30-40%）
    - flash 路径: Mode 1 与 Mode 0 峰值差 ≤10%（注意力无收益，FFN 才有）

  **Must NOT do**:
  - 不修改训练脚本（这是独立基准工具）
  - 不 assert 推导值的精确命中（用保守阈值）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要精确的显存测量方法论（峰值统计、warmup、正确对比）
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T10, T12, T13)
  - **Blocks**: F1-F4
  - **Blocked By**: T7, T8, T9

  **References**:
  **Pattern References**:
  - `trainer/train_pretrain.py:35-46` - 训练 step 结构（autocast → forward → loss → backward → step）— 基准脚本复制此模式
  - `model/model_instinct.py:12-52` - InstinctConfig 参数（构造基准模型）
  - `experiments/verify_experiment.py` - 现有实验脚本风格（独立脚本 + 控制台输出）

  **Acceptance Criteria**:
  - [ ] 脚本存在且参数可解析（`--help` 正常）
  - [ ] 无 GPU 时优雅退出（打印 SKIP）
  - [ ] 有 GPU 时输出 Mode 0/1/2 对比表 + 断言（exit code 0 = 通过）
  - [ ] S=2048 eager: Mode 1 ≥45% 激活节省、Mode 2 ≥85%；计算开销 ≤15% / ≤50%

  **QA Scenarios**:
  ```
  Scenario: 基准脚本 GPU 测量（有 GPU 时）
    Tool: Bash
    Preconditions: CUDA 可用; experiments/bench_checkpoint_memory.py 就位
    Steps:
      1. 运行 `python experiments/bench_checkpoint_memory.py --bs 8 --seq 2048 --layers 8 --mode all --steps 5`
      2. 断言 exit code 0（内置断言通过）
    Expected Result: 打印 Mode 0/1/2 峰值显存对比表; Mode 1 节省 ≥45%; 计算开销 ≤15%
    Failure Indicators: 断言失败 → 显存节省不足 / 计算开销超标
    Evidence: .sisyphus/evidence/task-11-gpu-bench.txt

  Scenario: 无 GPU 优雅退出
    Tool: Bash
    Preconditions: CUDA 不可用（或 --cpu-only）
    Steps:
      1. 运行 `python experiments/bench_checkpoint_memory.py --cpu-only 2>&1 | tail -2`
      2. 断言输出含 SKIP
    Expected Result: 打印 SKIP，不崩溃
    Failure Indicators: traceback
    Evidence: .sisyphus/evidence/task-11-nogpu.txt
  ```

  **Evidence to Capture**:
  - [ ] task-11-gpu-bench.txt, task-11-nogpu.txt

  **Commit**: YES
  - Message: `bench: GPU memory/throughput benchmark script`
  - Files: experiments/bench_checkpoint_memory.py
  - Pre-commit: `python experiments/bench_checkpoint_memory.py --help`

- [x] 12. 8-trainer smoke 测试

  **What to do**:
  - 创建 `tests/smoke/test_trainer_smoke.py`（或独立 smoke 脚本）：
    - 对 8 个 trainer 各跑 1-2 个训练 step，`--use_grad_checkpoint 1`，极小配置（tiny data 或合成数据、max_seq_len 小值）
    - 断言: 无异常、loss 有限（非 NaN/inf）、正常退出
    - trainer 列表: train_pretrain.py, train_full_sft.py, train_lora.py, train_dpo.py, train_ppo.py, train_grpo.py, train_agent.py, train_distillation.py
    - 注意各 trainer 的依赖: pretrain 需要 --from_weight none；full_sft 需要 --from_weight pretrain 或用随机初始化 flag（如存在）；PPO/GRPO 需要 reward model（GRPO 用 `--rollout_engine torch` 默认值即可，train_grpo.py:245；PPO 无 rollout flag，用最小化配置或跳过 RL 特定部分）
  - 标记 `@pytest.mark.gpu` 或按环境 skip（训练需要 CUDA）

  **Must NOT do**:
  - 不跑完整训练（只 1-2 step）
  - 不下载真实数据集（用 tiny/合成数据）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: 机械的 smoke 脚本编写
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T10, T11, T13)
  - **Blocks**: F1-F4
  - **Blocked By**: T7, T8, T9

  **References**:
  **Pattern References**:
  - 8 个 trainer 的 argparse 块（T4 已加 --use_grad_checkpoint）
  - `trainer/train_pretrain.py:88-126` - 最小参数集合（batch_size、max_seq_len、epochs 等）
  - `README.md` - 各 trainer 的训练依赖（SFT 需 from_weight pretrain 等）
  - AGENTS.md 的 Windows 注意事项（datasets before torch）

  **Acceptance Criteria**:
  - [ ] 8 个 trainer 各 1-2 step smoke 通过（loss 有限）
  - [ ] 无 CUDA 时跳过（不阻塞 CI-less 本地）
  - [ ] smoke 脚本记录到 evidence

  **QA Scenarios**:
  ```
  Scenario: pretrain smoke (use_grad_checkpoint=1)
    Tool: Bash
    Preconditions: CUDA 可用; tiny 数据
    Steps:
      1. 运行 `python trainer/train_pretrain.py --use_grad_checkpoint 1 --max_seq_len 64 --epochs 1 --batch_size 2 --data_path dataset/pretrain_t2t_mini.jsonl 2>&1 | tail -3`（或 smoke 脚本等价命令）
      2. 断言 exit code 0 且 loss 为有限值
    Expected Result: 训练正常启动、loss 有限、无崩溃
    Failure Indicators: NaN loss / CUDA OOM / 参数解析错误
    Evidence: .sisyphus/evidence/task-12-pretrain-smoke.txt

  Scenario: 8-trainer 全量 smoke
    Tool: Bash (pytest -m smoke 或脚本)
    Preconditions: CUDA 可用
    Steps:
      1. 运行 `python -m pytest tests/smoke/ -x -q -m smoke`（或等价脚本）
      2. 断言 exit code 0
    Expected Result: 8 个 trainer 全部通过
    Failure Indicators: 任一 trainer 崩溃/NaN
    Evidence: .sisyphus/evidence/task-12-all-smoke.txt
  ```

  **Evidence to Capture**:
  - [ ] task-12-pretrain-smoke.txt, task-12-all-smoke.txt

  **Commit**: YES
  - Message: `test: 8-trainer smoke runner`
  - Files: tests/smoke/
  - Pre-commit: `python -m pytest tests/smoke/ -x -q -m smoke`

- [x] 13. flag 线程化验证（config_path JSON / resume / 死 flag 全仓 grep）

  **What to do**:
  - 创建 `tests/test_flag_threading_e2e.py`:
    - **config_path JSON**: 构造含 `"use_grad_checkpoint": 1` 的 JSON，经 `config_from_args(args, config_path=...)` 路径断言 model.config 正确读取
    - **resume**: 验证 checkpoint 保存/恢复不受新 flag 影响（flag 从 args 重建，非 checkpoint 存储 — 与现有 kv_cache_dtype 行为一致）
    - **旧 JSON 兼容**: 含残留 `loop_grad_checkpoint` 键的旧 config JSON 不崩溃（`**kwargs` 吸收）
  - 全仓 grep 验证死 flag 零残留

  **Must NOT do**:
  - 不修改 lm_checkpoint 的存储格式

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 验证型测试，无新逻辑
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T10, T11, T12)
  - **Blocks**: F1-F4
  - **Blocked By**: T4, T5

  **References**:
  **Pattern References**:
  - `trainer/trainer_utils.py:38-62` - config_from_args（config_path 分支: `cfg_dict.update(overrides)`）
  - `trainer/trainer_utils.py:288-349` - lm_checkpoint（保存 config.to_dict；resume 从 args 重建 config）
  - `scripts/config_webui.py` - WebUI 接线（T5 完成）

  **Acceptance Criteria**:
  - [ ] config_path JSON 路径传递 flag 正确
  - [ ] 旧 JSON 含 loop_grad_checkpoint 不崩溃
  - [ ] `grep -rn "loop_grad_checkpoint" --include="*.py" .` → 零匹配（全仓）

  **QA Scenarios**:
  ```
  Scenario: config_path JSON 传递
    Tool: Bash (pytest)
    Preconditions: 测试就位
    Steps:
      1. 运行 `python -m pytest tests/test_flag_threading_e2e.py::test_config_path_json -x -q`
      2. 断言 exit code 0
    Expected Result: JSON 中 use_grad_checkpoint=1 到达 model.config
    Failure Indicators: 断言失败 → config_path 分支未接线
    Evidence: .sisyphus/evidence/task-13-json.txt

  Scenario: 全仓死 flag 零残留
    Tool: Bash
    Preconditions: T4/T5 完成
    Steps:
      1. 运行 `grep -rn "loop_grad_checkpoint" --include="*.py" . 2>/dev/null || echo "CLEAN"`
      2. 断言输出为 CLEAN
    Expected Result: 全仓无 loop_grad_checkpoint
    Failure Indicators: 有匹配行
    Evidence: .sisyphus/evidence/task-13-deadflag-global.txt
  ```

  **Evidence to Capture**:
  - [ ] task-13-json.txt, task-13-deadflag-global.txt

  **Commit**: YES
  - Message: `test: flag threading verification (JSON config, resume, dead-flag grep)`
  - Files: tests/test_flag_threading_e2e.py
  - Pre-commit: `python -m pytest tests/test_flag_threading_e2e.py -x -q`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run pytest, grep). For each "Must NOT Have": search codebase for forbidden patterns (flash_attn_4.py untouched, no offloading, no shared-base refactor) — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `python -m pytest tests/ -x -q` + read all changed files for: `as any`/`@ts-ignore` (n/a Python), empty catches, print in prod code, commented-out code, unused imports, hardcoded shapes. Check AI slop: excessive comments, over-abstraction, generic names. Check the custom autograd.Function follows the no-param trap (no params inside forward region).
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (flag threading end-to-end: argparse → config → model). Run GPU benchmark if available: `python experiments/bench_checkpoint_memory.py` Mode 0/1/2 at S=768 and S=2048. Test edge cases: flag=0 (default behavior unchanged), flag=2 on short seq, MoE aux_loss gradient non-zero.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep: no offloading, no shared-base refactor, no flash kernel changes). Check "Must NOT do" compliance. Detect cross-task contamination (T7 touching T8's files). Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **T1**: `test: add pytest infrastructure for gradient checkpointing` — requirements.txt, tests/
- **T2**: `feat(checkpointing): selective attention recompute autograd function` — model/checkpointing.py
- **T3**: `feat(checkpointing): FFN checkpoint helper + aux_loss return wrapper` — model/checkpointing.py
- **T4**: `feat(config): thread use_grad_checkpoint flag through configs and 8 trainers` — model/*.py, trainer/*.py
- **T5**: `refactor(webui): remove dead loop_grad_checkpoint, expose use_grad_checkpoint` — scripts/config_webui.py
- **T6**: `docs: document use_grad_checkpoint flag` — README.md
- **T7**: `feat(dense): wire selective + full-layer checkpoint into dense model` — model/model_instinct.py
- **T8**: `feat(loop): wire checkpoint into loop body (fulfills loop_grad_checkpoint)` — model/model_instinct_loop.py
- **T9**: `feat(linear): wire checkpoint into linear variant standard parts` — model/model_instinct_linear.py
- **T10**: `test: cross-model gradient/step equality suite` — tests/
- **T11**: `bench: GPU memory/throughput benchmark script` — experiments/
- **T12**: `test: 8-trainer smoke runner` — tests/
- **T13**: `test: flag threading verification (JSON config, resume, dead-flag grep)` — tests/

---

## Success Criteria

### Verification Commands
```bash
python -m pytest tests/ -x -q    # Expected: all pass (CPU, no GPU needed)
python -m pytest tests/ -m gpu   # Expected: GPU tests pass (if CUDA available)
python experiments/bench_checkpoint_memory.py  # Expected: prints Mode 0/1/2 peak memory + steps/sec table
grep -r "loop_grad_checkpoint" --include="*.py" .  # Expected: no matches
```

### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] `pytest tests/ -x -q` all green
- [~] GPU benchmark: Mode 1 (S=2048 eager) activation savings ≥45%, Mode 2 ≥85%; compute overhead ≤15% / ≤50%
- [x] 8 trainer smoke passed
- [x] Dead flag fully removed
