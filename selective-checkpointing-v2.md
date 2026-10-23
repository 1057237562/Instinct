# Selective Checkpointing v2 — Instinct V2 Dense (1B), 3 GiB activation budget

Branch: `refactor/code-readability`. Supersedes the Mode 1 / Mode 2 design in `gradient-checkpointing.md`.

## 0. Read this first

**What "3 GiB" can mean for a 1B model.** Weights, gradients and optimizer state are fixed costs that checkpointing cannot touch (V2 Dense ≈ 1.035 B params, computed from the stated shapes):

| Precision setup | Fixed state |
|---|---|
| fp32 params + fp32 grads + fp32 AdamW (16 B/param) | 15.4 GiB |
| bf16 params + bf16 grads + fp32 Adam m,v (12 B/param) | 11.6 GiB |
| bf16 params + bf16 grads + bf16 Adam (8 B/param) | 7.7 GiB |
| bf16 weights, frozen (LoRA-style) | 1.9 GiB |

So a **total** GPU peak of 3 GiB is impossible for full training of this model, whatever the checkpointing strategy. This plan therefore treats **3 GiB as the budget for activations and recompute transients** (everything above the fixed state). Total GPU memory ≈ fixed state + 3 GiB + CUDA context. If you meant total 3 GiB, the only route is LoRA on a frozen bf16 base (≈ 1.9 GiB weights, leaving ≈ 0.8 GiB for activations ≈ 5k tokens per micro-batch at ≈ +33 % compute); tell me and I'll plan that separately.

**Assumptions I could not verify** (V2 config is not in this branch). Please confirm or correct:
- 32 layers, hidden 1554, FFN 5504 (stated by you); **12 q-heads × head_dim 128, 4 KV heads, vocab 6400, tied embeddings** (assumed; gives ≈ 1.04 B params).
- bf16 autocast, dropout 0, `flash_attn` as in the current code.
- If the vocab is much larger than 6400, the loss head grows proportionally (≈ 23 bytes × V per token) and dominates; R0 below becomes mandatory.
- **head_dim must be a multiple of 8.** `flash_attn_4.py:81` requires it for FA4, and the SDPA fallback likely loses its fused kernels otherwise, which would silently bring back the O(S²) attention memory. 1554 is not divisible by 8; make sure heads × head_dim is set explicitly, not derived as `hidden // heads`.
- Hidden 1554 is not a multiple of 8, so bf16 matmuls (K or N = 1554) may miss tensor-core-aligned kernels. A hidden of 1552 or 1568 would likely be faster. Worth a quick benchmark; separate from this plan.

## 1. Goal

For the 1B model: **activation + transient memory ≤ 3 GiB** per micro-batch, with the smallest recompute overhead the budget allows.

- Acceptance metric: `max_memory_reserved() − fixed_state ≤ 3 GiB` (run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
- Design target: **≤ 2.6 GiB estimated** (15 % headroom for model error and fragmentation).
- All numbers below come from a hand-built accounting model (Appendix A). It reproduces the measured 64M-model results in `.sisyphus/evidence/task-11-gpu-bench.txt` (Mode 0 within 1 %, Mode 1 within 9 %, Mode 2 within 12–29 %) and reproduces the 64M per-token byte counts exactly. **No 1B measurement exists and nothing here has been run on a GPU.** Phase 0 replaces the estimates with measurements.

## 2. Diagnosis — why Mode 1 is heavy and Mode 2 is expensive

Per token per layer at V2 shapes, S=2048 (derived from `model_instinct.py` / `checkpointing.py`):

| Policy | Saved bytes | × 32 layers | Recompute vs layer fwd |
|---|---|---|---|
| Mode 0 eager | 251,240 | 8.0 MB/token | 0 |
| Mode 0 flash | 99,736 | 3.2 MB/token | 0 |
| Mode 1 (current), eager / flash | 55,656 / 55,704 | 1.8 MB/token | 83 % / 73 % |
| Mode 2 | 3,108 (+ one-layer transient) | 0.1 MB/token | 100 % |

At 2048 tokens/micro-batch (bs=1, S=2048) that is ≈ 3.4 GiB of saved activations for Mode 1 *before* the loss head (≈ 0.3 GiB) and, on the eager path, the attention transient (≈ 0.6 GiB). **Mode 1 does not fit 3 GiB even at one sequence per micro-batch.** Mode 2 fits (≈ 1–1.5 GiB) but costs about +33 % compute.

Root causes (shape-independent, verified in the code):

1. **fp32 RMSNorm copies dominate what Mode 1 keeps.** `RMSNorm.forward` (`model_instinct.py:82-84`) retains an fp32 `x.float()` and an fp32 normalized tensor for backward, for four norms per layer (input LN, q_norm, k_norm, post-attn LN). That is 41,248 of Mode 1's 55,656 bytes (74 %) at V2 shapes. Mode 1 never touches them; they are elementwise and nearly free to recompute.
2. **Mode 1 recomputes matmuls to free elementwise tensors.** `checkpoint_ffn` re-runs the whole FFN. At V2 shapes the FFN is 67–73 % of layer forward MACs, so Mode 1 costs ≈ +27 % (eager) / +24 % (flash) train MACs, not the ~10 % the old plan assumed. What it frees (`silu(g)`, `silu(g)*u`) is elementwise.
3. **Eager attention recompute has a large transient** (≈ 12·NH·S bytes/token; ≈ 0.6 GiB at 2048 tokens, growing with S), and `_attention_core` also redoes `P@V`, which backward does not need.
4. **The loss head is a floor no layer checkpoint can lower:** logits + `_cross_entropy_loss` (`model_instinct.py:328-330, 387-389`) hold bf16 logits, a contiguous copy, an fp32 cast, the `log_softmax` output and their grads (≈ 150 KB/token at V=6400 from the 64M calibration).
5. **Which attention path runs matters.** `flash_attn` defaults to True and pretrain/SFT pass no `attention_mask`, so they take the fused path and `RecomputeAttention` is never reached; Mode 1 is then FFN-only. The eager path appears with `--flash_attn 0`, masked batches (RL trainers) or a head_dim that blocks the fused kernels.
6. **The old benchmark cannot judge compute cost.** Mode 0 ran 2.8× slower than Modes 1/2 at 11.4 GB on a 16 GB card with 3 timed steps: memory-pressure noise. Use MAC counts until Phase 0 fixes the bench.

## 3. Strategy — free bytes first, ordered by recompute cost per byte freed

A ladder of rungs, chosen per layer by a planner under the budget, instead of "save the layer" or "recompute the layer".

| Rung | What | Bytes freed per token/layer (V2, S=2048) | Extra recompute | Notes |
|---|---|---|---|---|
| R0 | Chunked fused `lm_head` + CE (recompute logits per 1024-token chunk) | whole-model head ≈ 150 KB → ≈ 15 KB per token | one extra head matmul ≈ +1 % train MACs | Only for trainers that don't read `logits` (§5) |
| R1 | Memory-efficient RMSNorm: custom Function saves the bf16 input + `rstd`, recomputes the fp32 copies | 30,864 (of 41,248) | ≈ 0 MACs | Same math, bit-identical |
| R2 | SwiGLU epilogue recompute: save `g`,`u`, recompute `silu(g)` and `p` | 22,016 | ≈ 0 MACs | Replaces `checkpoint_ffn` as the default FFN policy |
| R3 | Attention: (eager) query-chunked recompute-backward, recompute only `QKᵀ`+softmax; (masked batches) SDPA with `attn_mask` instead of eager | eager: all of `6·NH·S` + transient; flash: 0 | eager: `QKᵀ` ≈ 8 % of layer fwd; flash: 0 | Removes the quadratic term |
| R4 | Drop `g`,`u`, recompute gate/up matmuls from the FFN input | 22,016 | ≈ 45–49 % of layer fwd | Only when R0–R3 don't fit |
| R5 | Whole-layer checkpoint, placed at the **bottom** layers | down to 3,108 | 100 % per layer | Last resort; bottom placement means its transient runs after upper layers' saved activations are already freed |

Planner labels below: **A** = R1+R2+R3 (46.8 KB/token/layer), **B** = A + R4 (24.8 KB), **C** = R5 (3.1 KB).

## 4. Planner results for V2 Dense (activation budget 2600 MiB)

Cheapest per-layer mix by tokens per micro-batch (A/B/C = number of layers on each rung; overhead is train-MAC overhead vs Mode 0, including the head):

| Tokens / micro-batch | Eager attention | Flash attention |
|---|---|---|
| 2,048 (1 × 2048) | **+7.6 %** (A22 B10 C0) | **+4.8 %** (A23 B9 C0) |
| 4,096 (2 × 2048) | +23 % (A0 B21 C11) | +21 % (A0 B23 C9) |
| 8,192 | +30 % (A0 B6 C26) | +30 % (A0 B7 C25) |
| 16,384 | infeasible | +33 % (all 32 layers C) |

All rows assume R0 (chunked head). Without R0: 2,048 tokens → +10.6 % eager / +7.5 % flash; 4,096 → +26 % / +25 %; 8,192 → infeasible eager / +32.5 % flash.

**Key conclusion for 1B: at 32 layers × 5504 FFN, the budget is set by tokens per micro-batch.** Layer checkpointing alone makes 4k+ tokens cost +20–33 %, while gradient accumulation of 2,048-token micro-batches costs +5–8 % and no extra recompute. **Recommended operating point: micro-batch = 2,048 tokens (bs=1 at S=2048) with `accumulation_steps` for the effective batch**, with v2 rungs picking the layer mix. The planner reports this trade-off and will suggest a smaller micro-batch when the cheapest feasible plan exceeds a configurable overhead cap (default 15 %).

Total GPU ≈ fixed state (§0) + ≤ 3 GiB + CUDA context (≈ 0.4–0.6 GiB).

## 5. Implementation design

**New mode:** `use_grad_checkpoint = 3` ("budgeted selective"). Modes 0/1/2 remain unchanged and bitwise-compatible.

**New flags / config:** `--ckpt_act_budget_gib` (default 3.0, activations only), `--ckpt_headroom` (0.15), `--ckpt_max_overhead` (0.15, triggers the "reduce micro-batch" suggestion), optional `--ckpt_layer_plan` ("A,A,B,…,C" override, 32 entries). Thread through `InstinctConfig` (3 files), `trainer_cli.py:72`, `trainer_utils.py:51`, `config_webui.py`.

**Code** (`model/checkpointing.py` unless noted):

- `MemEfficientRMSNorm` — R1. Weight passed as a Function input so its grad is kept.
- `swiglu_down` — Function `(g, u, W_down)`; forward under no_grad saves `g`,`u`; backward recomputes `silu(g)` and `p`, returns grads for `g`, `u`, `W_down` (R2). Fallback if autocast/dtype handling is awkward: `torch.utils.checkpoint` with `create_selective_checkpoint_contexts` (PyTorch ≥ 2.5, prototype) and a shape-based policy.
- `chunked_attention` — query-block recompute-backward with saved `q,k,v` (+ per-row logsumexp), same math as `Attention.forward`; plus an SDPA `attn_mask` route in `Attention.forward` (`model_instinct.py:153-169`) for masked batches (R3).
- `chunked_lm_loss(hidden, lm_head_weight, labels)` — loops 1024-token chunks forward, recomputes logits per chunk in backward, accumulates `dW`/`dh`; handles the label shift without `.contiguous()` copies; tied weights fine because the weight is a Function input (R0).
- `plan_checkpoint(cfg, tokens, seq, budget)` — Appendix A model, returns a per-layer rung list (32 entries here) with C layers at the bottom; also returns the estimated peak and overhead so the trainer can log them.
- Integration points: `InstinctBlock.forward` (`model_instinct.py:224-239`), `InstinctModel.forward` layer loop (`:276-293`), `InstinctForCausalLM.forward` loss (`:387-389`). Mirror in `model_instinct_loop.py` (`:250-265`; loop body `:389-424`, planned per iteration over `total_effective_layers`) and `model_instinct_linear.py` (standard-attention layers only; GatedDeltaNet layers stay whole-layer).
- MoE: `aux_loss` keeps flowing through return values.

**Zero-code baselines to run first**: `torch.compile` with `torch._dynamo.config.activation_memory_budget` (PyTorch ≥ 2.4, experimental; applies a pareto-optimal SAC policy over the compiled region) and the SAC API above. If either already fits 3 GiB at acceptable speed, shrink the custom work accordingly.

## 6. Scope limits

- **R0 only helps trainers that read just the loss:** pretrain, full SFT, LoRA. DPO, GRPO, PPO, distillation and agent RL consume `out.logits` (`train_dpo.py:118-122`, `train_grpo.py:134`, `train_ppo.py:221`, `train_distillation.py:79`, `train_agent.py:436`). For those, skip R0 (budget the full head: ≈ 0.3 GiB at 2,048 tokens, much more at larger V) or add a chunked "per-token logprobs" variant later.
- The MoE variant is out of scope for this budget.
- The model assumes bf16 autocast, dropout 0, the stated shapes. Phase 0 re-derives the byte counts from the profiler.

## 7. Phases and acceptance

**Phase 0 — measure (first).** Add `experiments/profile_activations.py` using `torch.autograd.graph.saved_tensors_hooks` (saved bytes per module/op), `torch.utils.flop_counter.FlopCounterMode` (recompute MACs) and `torch.cuda.memory._record_memory_history` (peak snapshot). **Measure on a 2–4 layer slice of the V2 config with the real hidden/FFN/head sizes**: per-token-per-layer activation bytes do not depend on depth, so this runs on a small GPU, then scales to 32 layers. Fix `bench_checkpoint_memory.py` (≥ 10 timed steps after warmup, report `max_memory_reserved`, `expandable_segments:True`, compare Mode 0 at a smaller batch to avoid thrash; add `--hidden/--ffn/--heads/--head_dim/--vocab` so V2 shapes are reachable).
*Gate:* model within ±10 % of measured activation peak for Modes 0/1/2; otherwise recalibrate.

**Phase 1 — R1 + R2 (+ R0 for pretrain/SFT).** CPU tests as in `tests/`: grads vs Mode 0 (`torch.equal` fp32, `allclose` bf16), dense, tied weights.
*Gate:* flash path, 2,048 tokens, 32-layer-equivalent activation estimate ≤ 2.6 GiB.

**Phase 2 — R3.** Chunked recompute for eager; SDPA route for masked batches. *Gate:* eager attention transient < 0.3 GiB.

**Phase 3 — planner + flags + trainers + WebUI.** Smoke tests for pretrain/SFT/LoRA.

**Phase 4 — loop and linear variants, then logits-consuming trainers.**

**Done when:** (1) V2 Dense, 2,048 tokens/micro-batch: activation peak ≤ 3 GiB for flash and eager; (2) wall-clock overhead vs Mode 0 ≤ 10 % on the fixed bench (expect above the MAC estimate because R0–R2 add memory-bound elementwise passes: budget 2–5 % extra); (3) loss/grad parity with Mode 0 over a 20-step fixed-seed run; (4) Modes 0/1/2 unchanged and `pytest tests/` green.

**Fallback ladder if the cap is missed:** more R4 layers → R5 on bottom layers → smaller micro-batch with more accumulation.

## Appendix A — accounting model (bytes/token/layer; bf16 autocast, fp32 norm copies, dropout 0)

Let H=hidden, qd=NH·D, kd=NKV·D, F=FFN, NH=heads, S=sequence.
- fp32 norm copies kept by autograd: 16H + 8qd + 8kd. Norm outputs saved by linears: 4H.
- Eager attention, Mode 0: 8qd + 6·NH·S. Recompute modes: 4qd + 4kd. Flash: 4qd + 4kd + 4NH.
- FFN full: 8F. Gate/up only: 4F.
- Mode 0 = norms + 4H + attention + 8F. Mode 1 = norms + 4H + attention(recompute). v2-A = 8H + 6qd + 6kd + 4(2+NH+NKV) + 4F (bf16 norm inputs + rstd, norm outputs, q/k/v/o-input, gate/up); v2-B = v2-A − 4F; C = 2H.
- Validated: at H=768, qd=768, kd=384, F=2432, NH=8 this gives Mode 0 eager 50,176 + 48S, Mode 1 29,184, v2-A 22,840 (hand-derived from the 64M code, reproduced exactly by the script).

Layer forward MACs/token = H(2qd + 2kd) + 2·S·qd (×0.5 flash) + 3HF. At V2 shapes, S=2048 eager: 38.3 M (FFN 67 %, attention core 16 %). Training MACs ≈ 3× forward + recompute; the head adds 3·H·V per token.

Peak activation ≈ Σ saved × tokens + tokens × (≈150 KB head, or ≈15 KB chunked, + transient), with the transient coefficients fitted on the 64M bench and scaled by NH. The 1B transient and head coefficients are extrapolations; they are the first things Phase 0 checks.
