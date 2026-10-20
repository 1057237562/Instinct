# Compute & VRAM Savings Techniques for Instinct

Architecture-level and inference-level survey, mapped onto **Project Instinct**
(single RTX 5070 Ti 16 GB · Windows 11 · PyTorch 2.13 / CUDA 13.2 · Blackwell
SM120 · Triton 3.4). Model under study: **Instinct V1 MoE** (678.7 M total /
106.2 M active params, hidden 512, 32 layers, 16 Q / 4 KV GQA, head_dim 32,
8 experts top-1, vocab 6400, max_pos 32768). A 152.4 M dense variant and an
RecurrsiveInstinct latent-recurrent variant also ship in the repo.

## Classification legend

| Tag | Meaning |
|---|---|
| **R** | Requires retraining from scratch (or long re-pretrain) — architecture/math change. Viable because Instinct has its own training pipeline. |
| **F** | Retrofit + light finetune / calibration / one-time transform on the existing checkpoint. |
| **I** | Pure inference-time; no weight change. |
| ✅ | Already implemented in Instinct. |

## Already in the codebase

- Packed FlexAttention (document-aware `BlockMask`, compact GQA) — `docs/packed_flex_attention.md`
- Triton device-offset grouped MoE GEMM — `docs/grouped_moe_gemm.md`
- Preallocated static KV cache + CUDA-graph decode capture
- FP8 KV-cache config (`--kv_cache_dtype`) and TorchAO FP8 training (`--fp8_training`)
- Early exit (`--early_exit`), RecurrsiveInstinct test-time compute (`--num_steps`)
- Sequence packing (fixed + adaptive Bucket), selective gradient checkpointing

---

## (a) KV cache reduction

Decode on a single GPU is bandwidth-bound; the KV cache is the dominant
per-token read. Highest-leverage category for batch size / context length.

| Technique | Venue / Year | Mechanism | Reported savings | Code | Class |
|---|---|---|---|---|---|
| **MLA** (DeepSeek-V2) | arXiv 2405.04434 (2024) | Cache a low-rank latent instead of full K/V | 93.3% KV reduction; 5.76× generation tput; 42.5% training-cost cut vs 67B dense | `deepseek-ai/DeepSeek-V2` | R |
| **YOCO** | arXiv 2405.05254 (2024) | Compute global KV once in a self-decoder, reuse via cross-decoder | Cache computed once; constant memory | arXiv | R |
| **MLKV** | arXiv 2406.09297 (2024) | Share KV across layers (GQA across depth) | Up to 6× vs MQA; evaluated on **Pythia-160M** | GitHub | R |
| **KIVI** | ICML 2024 | Tuning-free asymmetric 2-bit: per-channel K, per-token V | 2.6× less peak memory; 4× larger batch | GitHub | I |
| **KVQuant** | NeurIPS 2024 | Per-channel pre-RoPE K, non-uniform, dense+sparse | 3-bit < 0.1 ppl; 4.8× compression; ~1.7× kernel speedup; 1M ctx LLaMA-7B on 1×A100-80G | `SqueezeAILab/KVQuant` | I/F |
| **SnapKV** | NeurIPS 2024 | Observation-window KV selection, tuning-free | 3.6× gen speed; 8.2× memory eff @16K; 380K ctx on A100-80G | `FasterDecoding/SnapKV` | I |
| **H2O** | NeurIPS 2023 | Heavy-hitter token eviction | ~20% KV budget, near-lossless | GitHub | I |
| **StreamingLLM** | ICLR 2024 | Attention sinks + rolling window | Bounded cache, 4M+ tokens | GitHub | I |
| **Quest** | ICML 2024 | Query-aware page selection at decode | 7.03× self-attn; 2.23× end-to-end latency | GitHub | I |
| **CaM / token merging** | ICLR 2025 / — | Merge/drop redundant decode tokens | 2–4× decode memory | GitHub | I |

**For Instinct.** The static-cache fast path deliberately ignores a configured
FP8 KV dtype (per-step dequant would break the CUDA graph). A KVQuant/KIVI-style
**pre-RoPE per-channel key quantization with offline calibration** fits the
preallocated cache and cuts KV bytes ~4–8× without giving up capture.
SnapKV/H2O/Quest are less invasive but interact with graph capture — measure.
MLKV's Pythia-160M evidence puts cross-layer sharing directly in Instinct's
size class.

---

## (b) Sparse attention (train + infer)

| Technique | Venue / Year | Mechanism | Reported savings | Code | Class |
|---|---|---|---|---|---|
| **NSA** (Native Sparse Attention) | ACL 2025 | Trainable compression + selection + sliding window, hardware-aligned | Large speedups at 64K, end-to-end trainable | DeepSeek | R |
| **MoBA** (Mixture of Block Attention) | arXiv 2502.13189 (2025) | Blocks-as-experts, top-k block routing; seamless train/infer | Long-context at sparse cost | Moonshot | R |
| **MInference 1.0** | NeurIPS 2024 | Training-free prefill sparse (A-shape / Vertical-Slash / Block) | Up to 10× prefill @1M (30 min → 3 min, A100); 1.8/4.1/6.8/10× @100K/300K/500K/1M | `aka.ms/MInference` | I |
| **FlexAttention** | MLSys 2025 | Compiler-driven attention with custom masks | ✅ used — Instinct microbench 13.3 ms → 1.7 ms vs SDPA | PyTorch | I |

**For Instinct.** At `max_seq_len` 32768, quadratic prefill only bites in
bulk-eval / long-document use. MInference is training-free and can be trialed
behind the packed-attention runtime; NSA/MoBA are the architectural options for
a v2.

---

## (c) Linear / SSM / hybrid attention

Most relevant if Instinct trains a new backbone — the repo already ships
`model_instinct_linear.py` and the RecurrsiveInstinct core.

| Technique | Venue / Year | Mechanism | Reported savings | Code | Class |
|---|---|---|---|---|---|
| **Mamba-2 / SSD** | ICML 2024 | State-space duality; 2–8× faster core than Mamba | Linear decode, constant memory | `state-spaces/mamba` | R |
| **GLA** | ICML 2024 | Gated linear attention, channel-wise decay | Linear | GitHub | R |
| **DeltaNet / Gated DeltaNet** | NeurIPS 2024 / 2024 | Error-correcting delta rule (content overwrite) | Linear | GitHub | R |
| **RWKV-7 "Goose"** | arXiv 2503.14456 (2025) | Generalized delta rule + vector gating | Constant mem/time per token; 3B SoTA multilingual on fewer tokens; 0.19B–2.9B models | `RWKV/RWKV-LM` | R |
| **Jamba** | arXiv 2403.19887 (2024) | Transformer–Mamba MoE hybrid | 256K ctx; ~8× smaller KV; 12B active / 52B total (v0.1) | ai21labs | R |
| **Samba** | arXiv 2406.07522 (2024) | Mamba + sliding-window attention, 4K pretrain | 3.73× prompt / 3.64× decode @128K/64K; 1M zero-shot | `microsoft/Samba` | R |
| **Kimi Linear** | arXiv 2510.26692 (2025) | KDA + MLA hybrid, 3B active / 48B | Up to 75% KV cut; up to 6× decode @1M | `MoonshotAI/Kimi-Linear` | R |
| **Qwen3-Next** | 2025 | Gated DeltaNet + sparse attention, 80B-A3B | 10×+ decode eff. at long context | Qwen | R |

**Small-scale evidence** (arXiv 2504.14366, *What Matters in Linearizing
Language Models?*, 140 M – 1.7 B): gated state-update models (xLSTM, **Gated
DeltaNet**) define the Pareto frontier; **Gated DeltaNet recovers 94.2%** of a
1.7 B Transformer teacher vs ~71% for plain LinearAttention, and is the only
family retaining long-context retrieval. Delta-rule updates scale best.
RWKV-7 ships 0.19 B models. This directly supports a small Instinct hybrid.

---

## (d) Speculative decoding

| Technique | Venue / Year | Mechanism | Reported savings | Code | Class |
|---|---|---|---|---|---|
| **EAGLE-3** | NeurIPS 2025 | Draft with multi-layer feature fusion + training-time test | Up to 6.5×; ~1.4× over EAGLE-2; 1.38× tput @batch 64 (SGLang) | `SafeAILab/EAGLE` | I (+ draft train) |
| **Medusa** | ICML 2024 | Multiple extra decoding heads, tree attention | 2.2–3.6× | `FasterDecoding/Medusa` | F |
| **Lookahead decoding** | ICML 2024 | Jacobi-style parallel n-gram draft | 1.5–2.3×, no draft model | HF | I |
| **LayerSkip self-speculation** | ACL 2024 | Early-exit + self-draft (see §g) | 1.8–2.16× | `facebookresearch/LayerSkip` | F |
| **Cascade** (MoE spec.) | arXiv 2506.20675 (2025) | Utility-driven speculation for MoE | Shows naive MoE speculation can be **1.5× slower**; gives the fix | vLLM | I |

**For Instinct.** Batch=1 WebUI decode is latency-bound with a 106 M-active
model, so speculation has limited upside there; the **batched API server** is
where EAGLE-3/Medusa pays. Read Cascade first — naive speculation on a top-1 MoE
can regress.

---

## (e) MoE inference efficiency

| Technique | Venue / Year | Mechanism | Reported savings | Code | Class |
|---|---|---|---|---|---|
| **MoE-Gen** | arXiv 2503.09716 (2025) | Single-GPU module-based batching + full KV offload | 8–31× vs FlexGen / MoE-Lightning / DeepSpeed | `EfficientMoE/MoE-Gen` | I |
| **MoE-Lightning** | arXiv 2411.11217 (2024) | CGOPipe CPU–GPU–I/O pipeline + paged weights | Up to 10.3× (Mixtral 8×7B, 1×T4 16 GB) | GitHub | I |
| **FloE** | arXiv 2505.05950 (2025) | Intra-expert contextual sparsity + ultra-low-bit + predictors | 9.3× per-expert compression; 11 GB VRAM; 48.7× vs DeepSpeed-MII (RTX 3090); 4.4–7.6% drop | GitHub | F/I |
| **REAP** (prune/merge) | 2025 | Router-weighted expert saliency; prune or fuse | 50% prune; INT4 combo 83.5% compression, 81–88% retention | `egesabanci/reap-cuda` | F |
| **Sub-MoE** | arXiv 2506.23266 (2025) | SVD subspace expert merging | 25% / 50% expert cut retains 96% / 86% (Mixtral) | `lliai/MoERazor` | F |
| **PreMoe** | arXiv 2505.17639 (2025) | Task-conditioned pruning + retrieval | DeepSeek-R1 671B 8/128 keeps 97.2% MATH500 | `JarvisPei/PreMoe` | F |
| **MergeMoE** | arXiv 2510.14436 (2025) | Least-squares output merging | Beats M-SMoE at equal ratio | GitHub | F |

**For Instinct.** With only 8 experts top-1 and a 678.7 M footprint, the model
already fits 16 GB — **offloading is unnecessary**. Expert merging/pruning is
only relevant to shrink the artifact or cut serving cost, and Sub-MoE's own curve
(25% cut ≈ 6% drop; 50% cut ≈ 13% drop) is a caution at 8 experts. REAP's
layerwise single-GPU calibration is the practical recipe.

---

## (f) Inference quantization

| Technique | Venue / Year | Mechanism | Reported savings | Code | Class |
|---|---|---|---|---|---|
| **GPTQ** | ICLR 2023 | Layer-wise weight-only 3–4 bit | ~4× weight memory | `IST-DASLab/gptq` | I |
| **AWQ** | MLSys 2024 | Activation-aware weight-only 4-bit | ~4×, minimal loss | `mit-han-lab/llm-awq` | I |
| **SmoothQuant** | ICML 2023 | W8A8 with channel smoothing | ~2× compute + memory | `mit-han-lab/smoothquant` | I |
| **QuaRot** | NeurIPS 2024 | Hadamard rotation → end-to-end 4-bit (incl. KV) | ≤ 0.47 WikiText-2 ppl on Llama-2-70B; 99% zero-shot; lossless 6/8-bit, no calibration | `spcl/QuaRot` | I/F |
| **SpinQuant** | ICLR 2025 | *Learned* rotations (Cayley SGD) | W4A4KV4 within 2.9 pts (LLaMA-2-7B); beats QuaRot by up to 45.1% gap cut (LLaMA-3-8B); online Hadamard ~8% latency; 30 min calibration (small models) | `facebookresearch/SpinQuant` | I/F |
| **FP8** (E4M3/E5M2) | — | 8-bit float weights/acts/KV | ✅ used in training | PyTorch / TorchAO | I |
| **LLM.int8 / bitsandbytes** | NeurIPS 2022 | 8-bit outlier-aware | ~2× | `bitsandbytes` | I |

**For Instinct.** FP8 training is in place; the easy inference win is
**weight-only INT4/FP8 for served weights** (already reachable via the
llama.cpp / vLLM / GGUF path). For a principled full-4-bit (weights + acts + KV)
artifact, QuaRot / SpinQuant are the state of the art and are post-hoc. MoE needs
per-expert quantization; attention/norm layers stay higher precision.

---

## (g) Test-time compute savings

| Technique | Venue / Year | Mechanism | Reported savings | Code | Class |
|---|---|---|---|---|---|
| **Early exit / adaptive depth** | — | Stop at an intermediate layer when confident | ✅ Instinct `--early_exit` | — | I/F |
| **LayerSkip** | ACL 2024 | Layer dropout + early-exit loss + self-speculative decode | 2.16× CNN/DM; 1.82× coding; 2.0× TOPv2 | `facebookresearch/LayerSkip` | F |
| **CALM** | NeurIPS 2022 | Confidence-based early exit with calibration | 2–3× compute | GitHub | I/F |
| **Mixture-of-Depths** | arXiv 2404.02258 (2024) | Router skips layers per token | Up to ~50% FLOPs at parity | GitHub | R/F |
| **Latent recurrent scaling** | arXiv 2502.05171 (2025) | Add test-time compute via recurrent depth | ✅ RecurrsiveInstinct `--num_steps` | — | R |

**For Instinct.** Both directions are already covered (early exit for *less*
compute, RecurrsiveInstinct for *more*). LayerSkip is the natural upgrade if you want
trainable early exits with self-speculation.

---

## (h) Efficient attention kernels

| Technique | Venue / Year | Mechanism | Reported savings | Code | Class |
|---|---|---|---|---|---|
| **FlashAttention-2** | 2023 | Tiling + work partitioning | ~2× over FA1 | `Dao-AILab` | I |
| **FlashAttention-3** | NeurIPS 2024 | Hopper async WGMMA + FP8 | 1.5–2.0× over FA2 (H100); BF16 840 TFLOP/s (85%); FP8 1.3 PFLOP/s; 2.6× lower FP8 error | `Dao-AILab` | I |
| **FlexAttention** | MLSys 2025 | PyTorch compiler attention | ✅ used by Instinct packing | PyTorch | I |
| **PagedAttention** | SOSP 2023 | KV paging for serving batching | 2–4× tput | vLLM | I |

**Platform note.** On this Windows/PyTorch build
`is_flash_attention_available()` is false; Instinct correctly routes packed BF16
through FlexAttention's Triton backend. FA-3 targets Hopper; FlexAttention
already covers the mask need here, and Triton 3.4 is available for custom
kernels.

---

## Synthesis — prioritized next steps for Instinct

### Pure inference-time (no retrain; try first)
1. **KV-cache compression on the static-cache path** — KVQuant-style pre-RoPE
   per-channel K + KIVI 2–3 bit, or SnapKV/H2O eviction. Biggest single lever
   for batch / context on 16 GB.
2. **Weight-only INT4/FP8 for served weights** (GGUF/AWQ) — cheap, orthogonal
   to FP8 training.
3. **Speculative decoding for the batched API server** (EAGLE-3); read Cascade
   first for the MoE caveat.
4. **Long-context prefill sparse attention** (MInference) only if pushing well
   past 32K.
5. Keep FlexAttention + grouped MoE GEMM + CUDA-graph decode (done).

### Retrofit + light finetune / calibration
- **QuaRot / SpinQuant** full W4A4KV4 for a genuinely 4-bit artifact.
- **REAP / Sub-MoE** expert reduction if shrinking the 678.7 M footprint is a
  goal (expect a real accuracy cost at only 8 experts).
- **LayerSkip**-style trainable early exit to replace the heuristic one.

### Requires retraining (only if building a v2 backbone)
- **MLA** or **Kimi Linear** (KDA + MLA) as the attention core; or a
  **Mamba-2 / Samba / Jamba** hybrid; or **MLKV** cross-layer KV sharing
  (validated at Pythia-160M, Instinct's size class). **Gated DeltaNet** is the
  strongest small-scale linear family (arXiv 2504.14366).
- **NSA / MoBA** trainable sparse attention.

---

## Reference index

| Short name | Link |
|---|---|
| DeepSeek-V2 (MLA) | https://arxiv.org/abs/2405.04434 |
| YOCO | https://arxiv.org/abs/2405.05254 |
| MLKV | https://arxiv.org/abs/2406.09297 |
| Quest | https://arxiv.org/abs/2406.10774 |
| KIVI | https://arxiv.org/abs/2402.02750 · https://proceedings.mlr.press/v235/liu24bz.html |
| KVQuant | https://arxiv.org/abs/2401.18079 |
| SnapKV | https://github.com/FasterDecoding/SnapKV |
| NSA | https://aclanthology.org/2025.acl-long.1126/ |
| MoBA | https://arxiv.org/abs/2502.13189 |
| MInference | https://arxiv.org/abs/2407.02490 |
| FlexAttention | https://arxiv.org/abs/2412.05496 |
| Mamba-2 | https://arxiv.org/abs/2405.21060 |
| RWKV-7 | https://arxiv.org/abs/2503.14456 |
| Samba | https://arxiv.org/abs/2406.07522 |
| Jamba | https://arxiv.org/abs/2403.19887 |
| Kimi Linear | https://arxiv.org/abs/2510.26692 |
| Linearization study | https://arxiv.org/abs/2504.14366 |
| EAGLE-3 | https://github.com/SafeAILab/EAGLE |
| Cascade (MoE spec.) | https://arxiv.org/abs/2506.20675 |
| MoE-Lightning | https://arxiv.org/abs/2411.11217 |
| MoE-Gen | https://arxiv.org/abs/2503.09716 |
| FloE | https://arxiv.org/abs/2505.05950 |
| Sub-MoE | https://arxiv.org/abs/2506.23266 |
| PreMoe | https://arxiv.org/abs/2505.17639 |
| MergeMoE | https://arxiv.org/abs/2510.14436 |
| REAP-CUDA | https://github.com/egesabanci/reap-cuda |
| QuaRot | https://arxiv.org/abs/2404.00456 |
| SpinQuant | https://arxiv.org/abs/2405.16406 |
| LayerSkip | https://aclanthology.org/2024.acl-long.681/ |
| Mixture-of-Depths | https://arxiv.org/abs/2404.02258 |
| Latent recurrent scaling | https://arxiv.org/abs/2502.05171 |

> Numbers are as reported by each paper on its own benchmarks (often much larger
> models / A100-H100-class GPUs). Treat them as upper bounds and re-measure on
> the RTX 5070 Ti / 512-hidden MoE before committing.
