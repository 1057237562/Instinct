# Instinct Agent Instructions

## What this is
Instinct — train InstinctV1Moe from scratch on a single GPU. The current V1 MoE configuration is about 678.7M total parameters / 106.2M active parameters. All core algorithms (Dense/MoE Transformer, Pretrain, SFT, LoRA, DPO, PPO, GRPO, CISPO, Agentic RL, KD) implemented in pure PyTorch — no `trl`/`peft` abstractions.

## Project layout

```
model/          # Model definition (InstinctConfig, InstinctForCausalLM, LoRA), tokenizer files
trainer/        # All training scripts (pretrain, SFT, LoRA, DPO, PPO, GRPO, Agent RL, KD, tokenizer)
                # + shared infra: trainer_cli.py, trainer_utils.py, training_pipeline.py,
                #   training_profiler.py, rollout_engine.py, compile_cache.py
dataset/        # Dataset corpora, data files, provenance, and audit reports
scripts/data_loader/ # Training dataset loaders, source resolution, packing, and caching
scripts/data_builder/ # Offline dataset collection, generation, cleaning, and audit utilities
dataset_compiler/  # Rust tool: compile JSONL corpora into Parquet (see its README.md)
scripts/        # Inference, API server, WebUI, model conversion, training launchers
eval_llm.py     # CLI inference script
eval_*.py       # Benchmark harnesses (gsm8k, humaneval, pass_k, batch, report); artifacts land in eval/
tests/          # Pytest suite; `gpu` marker = needs CUDA
experiments/    # Perf/research experiment scripts (lambda sweeps, synthetic data, plots)
docs/           # Design docs: training_pipeline.md, compact_dataset_cache.md, gsm8k.md, humaneval.md, eval_webui.md
```

## Essential commands

### Training (all run from repo root)
```bash
# Single GPU
python trainer/train_pretrain.py
python trainer/train_full_sft.py

# Multi-GPU (DDP)
torchrun --nproc_per_node N trainer/train_pretrain.py
torchrun --nproc_per_node N trainer/train_full_sft.py
```

### Precision knobs (all 8 trainers + WebUI)
- `--dtype bfloat16|float16|fp32` — activation compute precision (autocast; fp32 = no autocast)
- `--param_dtype fp32|bf16|fp16` — parameter precision (fp32 = master weights; bf16/fp16 = weights cast directly)
- `--kv_cache_dtype fp32|bf16|fp16|fp8_e4m3|fp8_e5m2` — KV cache precision (fp8 = per-(batch,head) quantized cache, half decode bandwidth; affects generation/RL rollouts)
- `--fp8_training` — FP8 training via torchao; optional dep: `pip install -r requirements-fp8.txt` (torchao==0.18.0)

### Testing
```bash
python -m pytest tests/             # from repo root
python -m pytest tests/ --skip-gpu  # force-skip CUDA-required tests
```
- Tests marked `@pytest.mark.gpu` auto-skip when CUDA is unavailable.
- `tests/conftest.py` enforces the `datasets`-before-`torch` import order (same Windows workaround as trainers) — do not reorder there either.

### Resume from checkpoint
```bash
python trainer/train_pretrain.py --from_resume 1
# Checkpoints: checkpoints/{weight}_{dim}_resume.pth
# Model weights: out/{weight}_{dim}.pth
```

### Pause / resume (暂停/续训)
- While training is running, the WebUI config has a "⏸ Pause Training" button: the trainer finishes its current step, saves `out/{weight}_{dim}{_moe}.pth` plus full resume state `checkpoints/{weight}_{dim}{_moe}_resume.pth`, and exits. To resume, tick "Resume from checkpoint" in the WebUI and press Start Training (or run with `--from_resume 1`).
- CLI equivalent: `touch checkpoints/.pause_request` pauses any running trainer at the next step boundary (same save + exit code 42); `--from_resume 1` resumes.

### Inference
```bash
# Raw torch weights (from trainer output)
python eval_llm.py --load_from model --weight full_sft
# Transformers-format weights
python eval_llm.py --load_from ./instinct-3
# With LoRA
python eval_llm.py --weight full_sft --lora_weight lora_medical
# With adaptive thinking
python eval_llm.py --load_from ./instinct-3 --open_thinking 1
```

Decode fast path (`optimize_inference`, used by the Chat WebUI and `eval_llm.py`):
- The Chat WebUI loads with `INSTINCT_INFERENCE_COMPILE=auto`: **no trunk compile**, so loading
  stays instant (0.05 s, vs ~50 s for `full`). Decode does not need it — the decode step is
  captured as a CUDA graph, which is what removes the per-token launch cost. Set
  `INSTINCT_INFERENCE_COMPILE=full` to compile the whole trunk instead: about 2x faster prefill
  and 30% more decode throughput, for ~50 s of compiling at load **plus a second compile on the
  first real prompt** (each `generate()` allocates a new static cache, and the compiled graph is
  guarded on those buffers).
- `generate()` on a single unpadded sequence decodes into a **preallocated KV cache**
  (`model/static_cache.py`) written in place, then **captures that step as one CUDA graph**, so
  per-token kernel launches and compiler guard evaluations cost nothing. Measured (mode=auto,
  the WebUI default): 512-dim MoE 3.7 ms/token (268 tok/s), dense 768 1.9 ms/token (515 tok/s);
  with mode=full: 2.7 ms/token and 1.5 ms/token. `python experiments/check_load_time.py` reports
  the load breakdown per mode.
- The static cache stores keys/values in the **model dtype**; a quantized `kv_cache_dtype`
  (e.g. `fp8_e5m2`) is not applied on that path, since it would need a full
  dequantize+requantize per step. It logs when it overrides such a config.
- Capture is skipped for batch > 1, padded batches, `return_kv`, logit lens
  (`layer_callback`), and any dispatch that still synchronizes with the host (per-expert
  loop, `cached` grouped backend) — those keep the previous eager/compiled loop.
- MoE dispatch keeps the `torch.compiler.disable` barrier for training (Blackwell TMA
  weight-gradient issue) and uses a traceable inference forward instead, which keeps the eager
  (mode=auto) path 1.85x faster by removing a graph break per layer; it matters most where
  capture cannot apply (logit lens, batch > 1, capture failure).
- Regressions show up as graph breaks: `python experiments/graph_break_report.py`
  (expect "graph count: 1"). Throughput A/B: `python experiments/bench_chat_moe_decode.py`.

### Model conversion
```bash
cd scripts && python convert_model.py
# Converts between torch (.pth) and transformers (HuggingFace) formats
```

### API server
```bash
cd scripts && python serve_openai_api.py
# OpenAI-compatible endpoint at localhost:8998
# Supports reasoning_content, tool_calls, open_thinking
```

### WebUI
```bash
# ⚠️ Must copy model folder into ./scripts/ first (e.g. cp -r instinct-3 ./scripts/instinct-3)
cd scripts && streamlit run web_demo.py
```

### Datasets: JSONL or Parquet
Every trainer accepts either format on `--data_path`; `scripts/data_loader/source_format.py`
resolves a path (or a directory of shards) to the right `datasets` builder. The
Config WebUI lists both, and ignores `*.report.json` sidecars.

```bash
# Compile a corpus (~3x smaller, no Python JSON parsing at load time).
# Pass --align-chunk-bytes = the trainer's --streaming_chunk_mb so a streaming
# chunk cursor keeps pointing at the same rows after switching to the parquet.
cd dataset_compiler && cargo build --release
./target/release/dataset_compiler compile \
  -i ../dataset/pretrain_x.jsonl -o ../dataset/pretrain_x.parquet \
  --align-chunk-bytes 1GiB
./target/release/dataset_compiler verify ../dataset/pretrain_x.jsonl ../dataset/pretrain_x.parquet
```

- Keep the source JSONL: `verify` compares row counts and values, and a compiled
  file is rebuilt from it whenever the corpus changes.
- `token_count` is preserved by the compiler, which is what lets bounded
  streaming plan chunks without tokenizing; a `--format auto` compile infers the
  preset from the first row.
- Loader contract: chat corpora are `list<struct<utf8...>>`, `tools`/`tool_calls`
  stay JSON text, and nested non-message objects become JSON text. See
  `dataset_compiler/README.md` for the full schema rules.
- Resume: `x.jsonl` and `x.parquet` count as the same corpus, so recompiling a
  corpus and continuing is allowed. Streaming chunk boundaries are rebuilt by
  that switch, so a mid-chunk cursor replays or skips part of one chunk and the
  trainer logs the new row offset; resuming at an epoch boundary is exact.

## Architecture and config

- **Instinct V1 Dense**: 20 layers, hidden=768, FFN intermediate size 2432, 8 q-heads / 4 kv-heads, head_dim=96, tied embeddings; 152,406,528 total / active parameters (about 152.4M). Verified against `checkpoints/pretrain_20260912_180004_768.json`, used by the `instinct-v1-0914` evaluation. Vocab 6400, max_pos 32768, SwiGLU, RMSNorm, RoPE θ=1e6.
- **Dense legacy/default config**: 8 layers, hidden=768, 8 q-heads / 4 kv-heads. This is the old `instinct-3` / constructor default, NOT the trained Instinct V1 Dense configuration. Historical 8-layer experiment reports describe their own baselines.
- **InstinctV1Moe**: hidden=512, 32 layers, 16 q-heads / 4 kv-heads, 8 experts, top-1 routing, MoE intermediate size 1664 (about 678.7M total / 106.2M active parameters; see `trainer/config_instinct_v1_moe.json`)
- Config in `model/model_instinct.py` → `InstinctConfig`. Defaults: `hidden_size=768`, `num_hidden_layers=8`, `use_moe=False`
- **V1 comparison context**: the user reports that V1 MoE was trained on a 34GB dataset, while V1 Dense used two mini datasets for pretraining and SFT. File size does not establish consumed token count or code-token coverage. Compare checkpoint configs and actual training/evaluation records; do not infer that MoE saw less total data, or equate its 678.7M total parameters with dense per-token capacity (V1 Dense: 152.4M active; V1 MoE: 106.2M active).
- **Alternate topologies**: `model/model_instinct_loop.py` (looped) and `model/model_instinct_linear.py` (linear attention). Run trainers through the wrappers `python run_loop.py trainer/train_x.py` / `python run_linear.py trainer/train_x.py`, which swap `sys.modules["model.model_instinct"]` — don't edit `model_instinct.py` to switch topology
- Aligned to Qwen3 ecosystem — compatible with `transformers`, `llama.cpp`, `vllm`, `ollama`

## Training pipeline (must respect order)

1. **Pretrain** (`train_pretrain.py`) — from scratch (`--from_weight none`), uses `pretrain_t2t(_mini).jsonl`
2. **Optional CPT** (`train_pretrain.py`) — continue from a completed `pretrain_*`/`cpt_*` weight on `pretrain*.jsonl`; use the Config WebUI `cpt` mode for separate defaults and `cpt_*` outputs
3. **SFT** (`train_full_sft.py`) — requires a completed pretrain/CPT base, uses `sft_t2t(_mini).jsonl`
4. **Optional**: LoRA (`train_lora.py`), DPO (`train_dpo.py`), PPO (`train_ppo.py`), GRPO/CISPO (`train_grpo.py`), Agent RL (`train_agent.py`), KD (`train_distillation.py`)

## Critical gotchas

### `__package__` hack
Training scripts and `scripts/` files set `__package__` + `sys.path.append` to resolve imports from the repo root. This is intentional — do NOT refactor away without understanding the import chain.

### Working directory matters
- **Training scripts**: MUST run from repo root. Data paths default to `./dataset/`, model paths to `./model/`, output to `./out/`, checkpoints to `./checkpoints/`. Reward model (`../internlm2-1_8b-reward`) lives as a sibling directory of the repo
- **API server / WebUI**: MUST run from `scripts/` directory

### Windows workaround
`trainer/` scripts import `datasets` before `torch` to work around a known pyarrow/torch DLL conflict on Windows. They also import `trainer.compile_cache` before `torch` to configure Inductor (compile cache). Do NOT reorder or remove either.

### Windows + torch.compile (`--use_compile 1`)
`torch.compile` on Windows **requires UTF-8 mode** — torch's own inductor template files (e.g. `torch/_inductor/kernel/mm_grouped.py`) crash with `UnicodeDecodeError: 'gbk' codec can't decode` otherwise. Set `PYTHONUTF8=1` in the environment of the training process (WebUI launches already inject it).

### Swarmed (WandB replacement)
WandB is blocked in China. Training scripts use `swanlab` by default; API is compatible with WandB calls. Set `--use_wandb` to enable logging.

### `max_seq_len` is in tokens, not characters
Chinese text: ~1.5–1.7 characters per token. English: ~4–5. Recommended values per dataset:
- `pretrain_t2t_mini.jsonl` → `max_seq_len ≈ 768` (was ~340 for full pretrain_t2t)
- `sft_t2t_mini.jsonl` → `max_seq_len ≈ 768`

### Checkpoint paths
- Simple saves: `out/{weight}_{hidden_size}.pth`
- Resume checkpoints: `checkpoints/{weight}_{hidden_size}_resume.pth`
- The `--from_resume 1` flag handles auto-detection; works across GPU count changes

### WebUI model discovery
`web_demo.py` auto-scans `./scripts/` for subdirectories containing model weight files. The model folder must be copied there before launching.

### Reward model location (RLAIF training)
For PPO/GRPO, the reward model (`internlm2-1_8b-reward`) must be placed **alongside** the instinct repo (sibling directory), not inside it:
```
parent/
├── instinct/
└── internlm2-1_8b-reward/
```

## Dataset formats

- **Pretrain**: `{"text": "..."}` (one row per line in JSONL)
- **SFT / LoRA / RLAIF**: `{"conversations": [{"role": "...", "content": "..."}, ...]}` (OpenAI chat format). Tool calls embedded in `conversations` with `tools`, `tool_calls`, `tool` roles.
- **DPO**: `{"chosen": [...], "rejected": [...]}`; **Agent RL**: `{"conversations": [...], "gt": [...]}`.
- The compiled `.parquet` form of any of these carries the same rows under the
  same key names (see `dataset_compiler/README.md` for the column types).

### Dataset identity hygiene (mandatory)

Every dataset added or reused for pretraining, CPT, SFT, LoRA, DPO, RLAIF,
Agent RL, or distillation MUST pass an identity-oriented data-cleaning step
before it is used for training. Do not assume that a public, filtered, or
Qwen-generated dataset is free of model-identity contamination.

- The target model identity is **InstinctV1Moe**. Training data must not teach
  the assistant to claim that it is Qwen, 通义千问, 千问, Alibaba Cloud, 阿里云,
  阿里巴巴, or another unrelated model/provider. Also remove or normalize
  self-referential answers that identify the assistant as a different named
  model, unless the sample is explicitly an evaluation-only example.
- For chat/SFT-style data, inspect both user prompts and assistant targets.
  Drop identity-centric samples with contaminated targets, or rewrite them to
  a canonical InstinctV1Moe answer. Apply the same check to both `chosen` and
  `rejected` responses in DPO data and to generated assistant turns in RLAIF,
  Agent RL, and distillation data.
- For pretraining/CPT text, do not blindly delete every factual mention of a
  third-party model. Preserve ordinary knowledge when appropriate, but remove
  self-identification passages, model-card-style promotional text, and
  question/answer examples that make the trained assistant adopt that identity.
- Never perform a blind string replacement such as `Qwen` -> `InstinctV1Moe`;
  this creates false facts. A Qwen/Alibaba factual answer should be removed,
  retained as non-identity knowledge, or rewritten by a reviewed transformation.
- Add a small, reviewed identity anchor set to the final SFT mix. It should
  consistently answer variants of “你是谁/What model are you?” with
  `我是 InstinctV1Moe，一个从头训练的语言模型。` and should state that the
  model has no subjective consciousness or personal experiences. Keep identity
  anchors small (normally about 0.5%--2% of the final SFT mix) so they do not
  dominate normal behavior.
- Every cleaning run MUST emit provenance and audit information: source
  repository and revision, input/output row counts, dropped/rewritten counts,
  identity-keyword hit counts, schema/length-filter counts, and output SHA-256.
  Keep the original source data; name the cleaned training artifact with an
  explicit `clean` or `identity_clean` suffix and do not overwrite raw data.
- Use `scripts/data_builder/filter_anomaly_candidates.py` as the first-pass triage
  CLI for suspicious questions and identity contamination. It supports JSONL,
  gzip JSONL, Parquet, and shard directories; it emits whole rows with source
  path, row number, rule hits, reviewer instructions, and a companion hash
  report. Treat its output as AI/human review candidates, never as automatic
  deletion decisions. Example:

  ```bash
  python scripts/data_builder/filter_anomaly_candidates.py dataset/sft_t2t_mini.jsonl \
    --profile all --output dataset/review_candidates/t2t_mini_candidates.jsonl
  ```

- Before training, audit every selected artifact, including generated JSONL,
  compressed JSONL, and compiled Parquet. Review keyword hits instead of
  treating a zero-hit grep as sufficient proof. A basic text audit can start
  with:

  ```bash
  rg -i -n "qwen|通义千问|千问|alibaba cloud|阿里云|阿里巴巴|通义万相|通义听悟" dataset/
  ```

  Matches in tokenizer vocabulary, compatibility code, or conversion configs
  do not establish model identity and must not be “cleaned” by changing the
  tokenizer or the Qwen3-compatible serialization path. The check is for
  trainable content and generated assistant behavior.

## Key arguments for `eval_llm.py`

| Flag | Purpose |
|------|---------|
| `--load_from model` | Use raw `.pth` weights |
| `--load_from ./instinct-3` | Use transformers-format model |
| `--weight full_sft` | Weight name prefix (only with `--load_from model`) |
| `--use_moe 1` | MoE model |
| `--lora_weight lora_medical` | Apply LoRA on top |
| `--inference_rope_scaling` | Enable YaRN length extrapolation |
| `--open_thinking 1` | Adaptive thinking mode |

## Model conversion
`scripts/convert_model.py` converts torch `.pth` → transformers format (and vice versa). Uses `Qwen3Config`/`Qwen3ForCausalLM` as the compatibility target. Supports LoRA merge.

## Where to read more
- `PLAN.md` — current performance-optimization plan for this branch (`codex/better_performance`): SDPA attention path, batch-level combined mask, selective gradient checkpointing, packing
- `docs/training_pipeline.md` — end-to-end pipeline; `docs/compact_dataset_cache.md` — dataset cache design
- `docs/gsm8k.md` / `docs/humaneval.md` / `docs/eval_webui.md` — benchmark harness and Eval WebUI details
