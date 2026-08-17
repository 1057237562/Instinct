# Instinct Agent Instructions

## What this is
Instinct — train a ~64M LLM from scratch in ~2h on a single 3090. All core algorithms (Dense/MoE Transformer, Pretrain, SFT, LoRA, DPO, PPO, GRPO, CISPO, Agentic RL, KD) implemented in pure PyTorch — no `trl`/`peft` abstractions.

## Project layout

```
model/          # Model definition (InstinctConfig, InstinctForCausalLM, LoRA), tokenizer files
trainer/        # All training scripts (pretrain, SFT, LoRA, DPO, PPO, GRPO, Agent RL, KD, tokenizer)
dataset/        # Dataset loading classes (PretrainDataset, SFTDataset, RLAIFDataset)
scripts/        # Inference, API server, WebUI, model conversion
eval_llm.py     # CLI inference script
```

## Essential commands

### Training (all run from `trainer/`)
```bash
cd trainer

# Single GPU
python train_pretrain.py
python train_full_sft.py

# Multi-GPU (DDP)
torchrun --nproc_per_node N train_pretrain.py
torchrun --nproc_per_node N train_full_sft.py
```

### Resume from checkpoint
```bash
cd trainer && python train_pretrain.py --from_resume 1
# Checkpoints: checkpoints/{weight}_{dim}_resume.pth
# Model weights: out/{weight}_{dim}.pth
```

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

## Architecture and config

- **Dense**: 8 layers, dim=768, 8 q-heads, 4 kv-heads, vocab 6400, max_pos 32768, SwiGLU, RMSNorm, RoPE θ=1e6
- **MoE**: Same base + 4 experts, top-1 routing (198M total, 64M active)
- Config in `model/model_instinct.py` → `InstinctConfig`. Defaults: `hidden_size=768`, `num_hidden_layers=8`, `use_moe=False`
- Aligned to Qwen3 ecosystem — compatible with `transformers`, `llama.cpp`, `vllm`, `ollama`

## Training pipeline (must respect order)

1. **Pretrain** (`train_pretrain.py`) — from scratch (`--from_weight none`), uses `pretrain_t2t(_mini).jsonl`
2. **SFT** (`train_full_sft.py`) — **requires** `--from_weight pretrain`, uses `sft_t2t(_mini).jsonl`
3. **Optional**: LoRA (`train_lora.py`), DPO (`train_dpo.py`), PPO (`train_ppo.py`), GRPO/CISPO (`train_grpo.py`), Agent RL (`train_agent.py`), KD (`train_distillation.py`)

## Critical gotchas

### `__package__` hack
Training scripts and `scripts/` files set `__package__` + `sys.path.append` to resolve imports from the repo root. This is intentional — do NOT refactor away without understanding the import chain.

### Working directory matters
- **Training scripts**: MUST run from `trainer/` directory. Data paths default to `../dataset/`, model paths to `../model/`, output to `../out/`
- **API server / WebUI**: MUST run from `scripts/` directory

### Windows workaround
`trainer/` scripts import `datasets` before `torch` to work around a known pyarrow/torch DLL conflict on Windows. Do NOT reorder or remove.

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

- **Pretrain**: `{"text": "..."}` (one text per line in JSONL)
- **SFT / LoRA / RLAIF**: `{"conversations": [{"role": "...", "content": "..."}, ...]}` (OpenAI chat format). Tool calls embedded in `conversations` with `tools`, `tool_calls`, `tool` roles.

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
