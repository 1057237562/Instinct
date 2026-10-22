# Instinct Agent Instructions

## What this is
Instinct is a single-GPU language-model training and inference project with V1/V2 Dense and MoE configurations, plus linear-attention and RecurrsiveInstinct backbones. All core algorithms (Dense/MoE Transformer, Pretrain, SFT, LoRA, DPO, PPO, GRPO, CISPO, Agentic RL, KD) are implemented in pure PyTorch — no `trl`/`peft` abstractions. Rust tools handle dataset compilation and the AutoDL data/training relay.

Last checked against the working tree on **2026-10-23**. Treat executable code,
checkpoint sidecars and the shared template catalog as the source of truth for
current behavior. Historical experiment numbers describe their recorded model,
hardware and precision; partial JSON reports do not establish a completed run.

## Before making changes

- **Never end a training process without saving its current progress first.**
  Requests to stop, pause, restart, or finish a diagnostic training run must use
  the trainer's save-and-pause flow. Wait for the accumulated optimizer update,
  save the full resume state (weights, optimizer, scaler and data cursor), and
  verify the successful save and matching checkpoint before reporting it stopped.
  An older checkpoint does not satisfy this rule. Do not use `Stop-Process`,
  `taskkill`, `kill`, `terminate`, or an exception-based benchmark exit to discard
  unsaved training progress. If saving fails or is blocked, keep the process
  alive when possible and report the problem instead of forcing termination.
- Inspect `git status --short` and relevant diffs first. Preserve existing local
  changes, including untracked source files; do not reset panel state or templates.
- Keep data, weights, resume states, runtime credentials and large generated
  traces out of commits. Inspect `.gitignore` before creating new output paths.
- Keep routine investigation and configuration checks CPU-only. Run GPU benchmarks
  with training paused, and record the actual model, batches, precision and warmup
  window when reporting performance.
- Update the smallest relevant surface and run its focused checks. Changes to
  trainer arguments or shipped presets also require WebUI/relay parity checks
  described below; do not copy defaults into a second catalog.

## Project layout

Four categories; root entry points (`eval_llm.py`, `eval_*.py`, `run_recurrsive.py`,
`run_linear.py`, `trainer/train_*.py`, `start_*.bat`) are stable public paths — do not
move them, and keep importable code under `scripts.*` / `trainer.*`.

```
# ── Training code ────────────────────────────────────────────────────
model/          # Model definition (InstinctConfig, InstinctForCausalLM, LoRA),
                # RecurrsiveInstinct/linear backbones, inference runtime, tokenizer files
trainer/        # All training scripts (pretrain, SFT, LoRA, DPO, PPO, GRPO, Agent RL, KD, tokenizer)
                # + shared infra: trainer_cli.py, trainer_utils.py, training_pipeline.py,
                #   streaming_pretrain.py, packing_transition.py, training_eta.py,
                #   training_profiler.py, rollout_engine.py, compile_cache.py
                # + batched_muon.py / muon_kernels.py, autotune*, config_*.json, recipes/, profiles/
run_recurrsive.py / run_linear.py   # Alternate-topology wrappers (root)
scripts/train_instinct_v2.ps1       # V2 Dense launcher

# ── Data acquisition & build ─────────────────────────────────────────
scripts/data_loader/   # Training dataset loaders, source resolution, packing, and caching
scripts/data_builder/  # Offline dataset collection, generation, cleaning, and audit utilities
                       # + l3_builder/ Rust corpus builder (cargo build inside that directory)
dataset_compiler/      # Rust tool: compile JSONL corpora into Parquet (see its README.md)
data_relay/            # Rust relay server: run download/cleaning scripts, auto-boot AutoDL in
                       # no-GPU mode, transfer via HTTP pull (default)/SFTP/Jupyter, disk quota;
                       # prebuilt Windows binaries live in data_relay/bin/ (ignored)
dataset/               # Dataset corpora (data ignored), provenance, and audit reports (tracked)

# ── Artifacts & checkpoints (git-ignored) ────────────────────────────
out/            # Inference weights out/{weight}_{dim}{_moe}.pth + matching .json
checkpoints/    # Full resume states checkpoints/{weight}_{dim}{_moe}_resume.pth + .json;
                # checkpoints/_archive/ keeps retired/duplicate artifacts (archive, never delete)
eval/           # Evaluation artifacts and runs
experiments/results|data/   # Experiment outputs
profiler_traces/ trainer/logs/ tmp/   # Traces, trainer + shutdown-monitor logs, scratch
.relay_training/ .relay_code_sync/    # Relay-managed training state/backups

# ── Tools & entry points ─────────────────────────────────────────────
scripts/        # Inference, API server, Chat/Config/Eval WebUI, model conversion, monitoring,
                # autotune panel, data tooling (importable as scripts.*)
eval_llm.py     # CLI inference script
eval_*.py       # GSM8K, HumanEval, HumanEval OJ, pass@K, batch and report helpers
                # LiveCodeBench generation/evaluation is in eval_llm.py; artifacts land in eval/
start_*.bat     # Root launchers (Config/Chat/Eval WebUI, API, data relay)

# ── Docs & tests ─────────────────────────────────────────────────────
tests/          # Pytest suite; `gpu` marker = needs CUDA
experiments/    # Perf/research experiment scripts (lambda sweeps, synthetic data, plots);
                #   probes/ keeps retired one-off diagnostics
docs/           # Pipeline/cache, packed attention, Muon precision, autotune and evaluation docs
                # + docs/architecture/ (HTML architecture/tour pages) + docs/reports/ (dated reports)
```

Put new files into the matching category: training code in `trainer/`/`model/`,
data code in `scripts/data_*`/`dataset_compiler/`/`data_relay/`, tools in `scripts/`,
and generated outputs only in the ignored artifact directories. See
`docs/repo_layout.md` for the full map and cleanup rules.

## Essential commands

### Implementation selection: multithreaded Rust by default

- Default to **multithreaded Rust** when selecting or building tools for CPU/I/O-intensive work, especially dataset collection, cleaning, merging, global shuffling, deduplication, format conversion, hashing, and large-file processing.
- **Inspect and reuse existing repository tools first.** Do not create duplicate Python scripts or wrappers for capabilities already provided by a Rust tool. In particular, JSONL-to-Parquet compilation, inspection, and verification must use the existing `dataset_compiler` CLI directly, with its supported thread controls; do not implement another Python conversion pipeline.
- Do not choose a single-threaded Python implementation merely because a reusable script already exists. Inspect the data size and bottleneck before launching a large job, and use bounded-memory Rust workers with an appropriate thread count.
- Verify actual parallelism in the hot path using active worker CPU time and throughput. A `--threads N` flag or an N-thread pool alone does not establish end-to-end parallel execution; identify serial compression/writing stages and report their limits accurately.
- Python may orchestrate native tools or provide interfaces that require Python. Keep existing pure-PyTorch model/training algorithms in their established stack; move standalone data-processing hot paths into multithreaded Rust.
- Preserve complete records and metadata, make random seeds reproducible, validate row/token conservation and file integrity, and retain progress logs. When replacing a slow running implementation, reuse validated intermediate files where possible.

### Training (all run from repo root)
```bash
# Single GPU
python trainer/train_pretrain.py
python trainer/train_full_sft.py

# Multi-GPU (DDP)
torchrun --nproc_per_node N trainer/train_pretrain.py
torchrun --nproc_per_node N trainer/train_full_sft.py
```

- Bare trainer defaults still select the legacy 8-layer Dense model. For a named
  model, use the Config WebUI command preview or a matching launcher; CLI defaults
  such as hidden size, layer count, precision and checkpointing can override JSON.
- V2 Dense launcher (PowerShell, from repo root):
  `./scripts/train_instinct_v2.ps1 -DataPath ./dataset/pretrain_x.parquet`.
  It reads the shared **Instinct V2** recipe, sets UTF-8 and starts a new run.
  It is not the separate fixed-budget recipe or a resume launcher.

### Precision knobs (all 8 trainers + WebUI)
- `--dtype bfloat16|float16|fp32` — activation compute precision (autocast; fp32 = no autocast)
- `--param_dtype fp32|bf16|fp16` — parameter precision (fp32 = master weights; bf16/fp16 = weights cast directly)
- `--kv_cache_dtype fp32|bf16|fp16|fp8_e4m3|fp8_e5m2` — cache precision on supported generation/RL paths; the static cache uses model dtype instead
- `--fp8_training off|tensorwise|rowwise|rowwise_with_gw_hp` — explicit TorchAO recipe, not a boolean flag; default `off`. Optional dep: `pip install -r requirements-fp8.txt` (torchao==0.18.0)
- `--fp8_filter auto|eligible` — skip small GEMMs expected to be slower, or convert all size-compatible Linear layers. Converted layer counts do not prove FP8 kernel coverage.
- `--muon_mantissa 0|1`, `--muon_momentum_dtype fp32|bf16`,
  `--muon_batch_size N`, `--muon_workspace_mb MiB` — shared Muon controls;
  see the precision/resume contract under Critical gotchas.

### Testing
```bash
python -m pytest tests/             # from repo root
python -m pytest tests/ --skip-gpu  # force-skip CUDA-required tests
python -m pytest tests/test_packed_attention.py --run-slow  # opt in to slow integration checks
```
- Tests marked `@pytest.mark.gpu` auto-skip when CUDA is unavailable.
- Tests marked `@pytest.mark.slow` are skipped unless `--run-slow` is supplied;
  `--skip-gpu` alone does not opt into them. PyTorch itself is environment-specific
  and is not pinned by an active requirement in `requirements.txt`.
- `tests/conftest.py` enforces the `datasets`-before-`torch` import order (same Windows workaround as trainers) — do not reorder there either.

Choose checks by the changed subsystem:

| Change | Focused checks (repo root) |
|---|---|
| Presets / command builders / template migration | `python -m pytest tests/test_config_templates.py tests/test_config_webui_presets.py tests/test_config_webui_app.py --skip-gpu` |
| Trainer flags / shared defaults | `python -m pytest tests/test_trainer_cli_defaults.py tests/test_flag_threading.py tests/test_flag_threading_e2e.py --skip-gpu`; `python data_relay/generate_training_schema.py --check` |
| Packing / streaming / pause / ETA | `python -m pytest tests/test_streaming_chunks.py tests/test_pretrain_pause_accumulation.py tests/test_training_eta.py tests/test_webui_training_status.py --skip-gpu` |
| Attention / compilation | `python -m pytest tests/test_packed_attention.py tests/test_packed_attention_v2.py tests/test_causal_attention_v2.py tests/test_compile_policy.py --skip-gpu`; rerun relevant CUDA cases for kernel changes |
| Muon precision / fused updates | `python -m pytest tests/test_muon_mantissa.py tests/test_muon_bf16_momentum.py tests/test_muon_v2.py tests/test_muon_kernels.py --skip-gpu`; CUDA parity/performance needs separate validation |

### Resume from checkpoint
```bash
python trainer/train_pretrain.py --from_resume 1
# Checkpoints: checkpoints/{weight}_{dim}{_moe}_resume.pth
# Model weights: out/{weight}_{dim}{_moe}.pth
```

### Pause / resume (暂停/续训)
- While training is running, the WebUI config has a "Pause Training" button: the trainer finishes its current step, saves `out/{weight}_{dim}{_moe}.pth` plus full resume state `checkpoints/{weight}_{dim}{_moe}_resume.pth`, and exits. To resume, tick "Resume from checkpoint" in the WebUI and press Start Training (or run with `--from_resume 1`).
- CLI equivalent: `touch checkpoints/.pause_request` pauses any running trainer at the next step boundary (same save + exit code 42); `--from_resume 1` resumes.
- PowerShell equivalent: `New-Item -ItemType File -Force checkpoints/.pause_request`.
  Pretraining honors pause after the accumulated optimizer update (or final
  gradient flush), because resume states do not contain pending gradients.
- The WebUI Training panel has a "训练结束后自动关机" toggle: ticking it arms `scripts/monitor_training_shutdown.ps1` (via `-ProcessId`) whenever a training run or the pipeline starts — a detached watchdog, so shutdown still fires with the browser closed. Only exit code 0 shuts down; pause (42) and failure keep the machine on. Un-tick before the run ends to disarm (the watchdog re-reads `trainer/webui_state.json` right before scheduling). The in-page countdown can be cancelled with one click (`shutdown /a`).
- Panel state auto-saves to `trainer/webui_state.json` on every interaction and is restored on the next open. Restores only re-apply a named preset when its definition fingerprint (`preset_fingerprint`) changed, so manual tweaks on top of a preset survive reloads.

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
- The Chat WebUI defaults to `INSTINCT_INFERENCE_COMPILE=auto`: fused kernels
  without compiling the whole trunk. `full` compiles and warms the trunk;
  `off` disables inference optimization. The CLI uses `--inference_compile`.
  Compilation cost and throughput depend on the model, shapes and environment.
- `model/static_cache.py` provides in-place KV buffers, reusable capacity buckets
  and CUDA-graph decode. Decode states survive successive `generate()` calls,
  preserving buffer addresses and captured graphs; new capacities/batch sizes can
  require allocation and warmup. Initial decode headroom is 256 tokens, then
  generation grows the cache as needed rather than reserving the entire allowance.
- Static decode supports single sequences and unpadded/left-padded batches.
  `INSTINCT_BATCH_STATIC_DECODE=0` disables the batched path. Right padding,
  interior mask holes, early exit and `return_kv` use the growing-cache path;
  memory guards can also trigger fallback.
- The static cache stores keys/values in the **model dtype**; a quantized `kv_cache_dtype`
  (e.g. `fp8_e5m2`) is not applied on that path, since it would need a full
  dequantize+requantize per step. It logs when it overrides such a config.
- Capture is skipped for diagnostic callbacks (such as logit lens), and dispatch
  that synchronizes with the host (per-expert loop or `cached` grouped backend).
  Static storage eligibility and graph-capture eligibility are separate checks.
- MoE dispatch keeps the `torch.compiler.disable` barrier for training (Blackwell
  TMA weight-gradient issue) and uses a traceable inference forward to avoid a
  graph break per layer. Keep those two dispatch policies separate.
- Diagnostics: `python experiments/check_load_time.py`,
  `python experiments/graph_break_report.py` and
  `python experiments/bench_chat_moe_decode.py`. Cache regressions are covered by
  `tests/test_static_cache.py`; compare warm runs with identical prompt/output lengths.

### Model conversion
```bash
# Inspect scripts/convert_model.py and set lm_config and input/output paths first.
# Its __main__ block is a legacy 8-layer Dense example, not an auto-detecting CLI.
cd scripts
python convert_model.py
```

### API server
```bash
cd scripts && python serve_openai_api.py
# OpenAI-compatible endpoint at localhost:8998
# Supports reasoning_content, tool_calls, open_thinking
```

### WebUI
```bash
cd scripts
streamlit run web_demo.py --server.port 8502
```

Windows root launchers: `start_config_webui.bat` (8500),
`start_chat_webui.bat` (8502), `start_eval_webui.bat` (8503), and
`start_openai_api.bat` (8998). Config/Chat run from `scripts/`; the Eval launcher
runs `python -m streamlit run scripts/eval_webui.py` from the repo root.
Chat directly scans raw `.pth` weights in `out/` and `checkpoints/` (excluding
`*_resume.pth`), supports a custom weight path, and defaults to the repo's
`model/` tokenizer. No model-directory copy is required.

HTTPS (Chat WebUI): `start_chat_webui.bat --https` from the repo root serves
over TLS using a self-signed pair auto-generated at `certs/chat_webui.crt|.key`
by `scripts/https_cert.py` (`cryptography` backend, openssl fallback). The same
effect manually: add `--server.sslCertFile <cert> --server.sslKeyFile <key>` to
any `streamlit run`. Self-signed = browser warning; local/trusted-intranet use
only (Streamlit's SSL option is not production-audited).

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
  --align-chunk-bytes 1GiB --threads 16
./target/release/dataset_compiler verify ../dataset/pretrain_x.jsonl ../dataset/pretrain_x.parquet
```

- Keep the source JSONL: `verify` compares row counts and values, and a compiled
  file is rebuilt from it whenever the corpus changes.
- `verify` checks the total row count but compares only **1,000 rows by
  default** (`--rows`, `--offset` select the window). Specify the intended full
  row count for a full-value audit; do not describe the default sample as one.
  `inspect FILE --json` reads footer/schema information without decoding all rows.
  `--threads` controls compilation parsing; compression/writing and verification
  have their own limits, so do not claim every stage uses that many workers.
- `token_count` is preserved by the compiler, which is what lets bounded
  streaming plan chunks without tokenizing; a `--format auto` compile infers the
  preset from the first row.
- Loader contract: chat corpora are `list<struct<utf8...>>`, `tools`/`tool_calls`
  stay JSON text, and nested non-message objects become JSON text. See
  `dataset_compiler/README.md` for the full schema rules.
- Resume: corresponding `x.jsonl` and `x.parquet` paths count as the same corpus.
  With matching `--align-chunk-bytes`, chunks retain the JSONL row ranges and the
  saved cursor remains exact. Without matching alignment, a mid-chunk resume can
  replay/skip part of a chunk and logs the new row offset; epoch-boundary resume
  has no drift. Aligned Parquet boundaries take precedence over later chunk-size
  requests. Keep corpus contents, packing and saved batch budgets consistent.

### Bounded pretraining and progress

- `--dataset_streaming auto|on|off` selects bounded chunk loading for pretraining;
  auto compares estimated Arrow size with `--data_cache_max_gb` (default 5 GiB),
  not GPU memory or compressed file size. `0` disables the cache size cap.
- `--streaming_chunk_mb 1024`, `--streaming_prefetch_chunks 0|1` and
  `--cache_build_mode inline|spawn` control staging. Budget checks reserve four
  chunk footprints without prefetch, five with it. Preserve the next unread
  chunk/step cursor and validate conservation when changing loader behavior.
- Streaming LR/progress uses consumed valid tokens; estimated step totals are
  not authoritative completion counts. `trainer/training_eta.py` estimates
  remaining **current-epoch minutes** from the last three reporting intervals
  (tokens for streaming, steps otherwise). Startup/resume/new epochs reset it;
  the first report is `estimating`. Later data waits, saves and recompiles count
  toward wall time. See `docs/training_pipeline.md` and its ETA tests.

### Data relay (download / clean / deliver to AutoDL)

`data_relay/` is a separate Rust binary (Axum + Tokio), normally deployed on a
Linux server with public/international bandwidth. Its WebUI is embedded from
`src/index.html`; it does not use the Python/Streamlit training environment.

```bash
cd data_relay
cargo build --release
cargo test
cargo clippy
./target/release/data_relay --config config.toml serve
# Read-only connectivity / instance / disk inspection (requires configured credentials):
./target/release/data_relay --config config.toml check
```

- `serve` is the default command; default listen address is `0.0.0.0:7800`.
  If the selected config file is missing, either command writes
  `config.example.toml` to that path and exits; configure it before restarting.
  Relative `server.data_dir` is resolved against the process working directory,
  so keep a stable deployment cwd (also when using `--config` elsewhere).
  Windows development can use the `stable-x86_64-pc-windows-gnu` toolchain;
  the commands above show the Linux deployment binary.
- Module map: `config.rs` (defaults/hot updates), `model.rs` (task states/path
  helpers), `store.rs` (file-backed task records), `runner.rs` (execution/upload
  lifecycle), `autodl.rs` (console/Pro API), `disk.rs` (quota planning),
  `pull.rs` / `sftp.rs` / `jupyter.rs` (transfer channels), `web.rs` (HTTP API).
- Server training manager: root `start_data_relay.bat` selects the newest of the
  `data_relay/bin/` manager/relay executables and `data_relay/target/release/data_relay.exe`.
  `training.rs` handles SSH/API operations and automatic
  no-GPU shutdown → GPU boot → SSH/CUDA readiness for WebUI start/resume clicks
  (`auto_gpu:true`, GPU billing). CLI `training --training-action start|resume`
  requires `--auto-gpu` or `allow_gpu_boot=true` for a mode switch. Delivery keeps
  its separate no-GPU boot policy. Startup operations return HTTP 202 and persist
  under `<data_dir>/training/operations`; poll `/api/training/operations`. GPU capacity
  errors queue and retry every `training.gpu_retry_secs` (60 by default), outside
  the boot timeout. Persist requests, training snapshots and pinned instance IDs;
  recover pre-dispatch waits on serve startup, never blindly replay an uncertain dispatch.
  Cancel through `/api/training/operations/{id}/cancel`. Each start/resume explicitly
  sends `shutdown_after_train:true` and a 60-second delay, independent of saved UI
  checkboxes. The remote supervisor invokes `/usr/bin/shutdown` only after exit 0
  and a fresh idle check; pause/failure/stop never trigger it. Preserve successful
  training status when shutdown fails. Queue lifecycle tests use local mock APIs.
  `training_remote.py` checks dataset paths (no mandatory identity review or audit
  report; explicit CLI/API `audit_path` can verify a SHA-256 report), derives model topology flags, imports full resume checkpoints, and supervises
  trainers independently of SSH under `<repo>/.relay_training`. One active job
  per repo; pause saves then exits 42, stop terminates the process group. Preserve
  full resume states, detach hard-linked output weights before overwriting, and
  block shutdown while training/mode transitions are active. Training config
  defaults to `/root/autodl-tmp`, `/root/miniconda3/bin/python`, 8 GiB free reserve
  and 1 GiB cache budget. Tests: `python data_relay/tests/test_training_remote.py`.
  The relay training panel edits model/training settings, auto-saves its own
  `<data_dir>/training/panel.json`, and previews commands without connecting to AutoDL.
  `training_panel.rs` compiles editable state through `training_cache.rs`; UI fields
  and presets in `training_schema.json` are generated by
  `python data_relay/generate_training_schema.py` from Config WebUI presets and trainer CLI
  definitions (use `--check` after changing them). No Streamlit/PyTorch is needed on the relay.
  Stage-specific fields/extra arguments are retained across stage switches. New and resumed
  jobs snapshot their model configuration. Resume no longer re-hashes the corpus.
  Training parameters can also load from Config WebUI's current `trainer/webui_state.json`
  through `training_cache.rs`, with `training.panel_state_path` relative to the
  selected config file. Prefer current top-level/stage fields over stale pipeline
  snapshots; preserve cached precision, optimizer, FP8, compile, packing and
  streaming settings. WebUI supports explicit cache import and persistent edits;
  no identity-review checkbox or audit path is required. For an optional CLI/API
  audit report, the remote helper accepts the top-level
  `output_sha256` or the dataset compiler's `output.sha256`.
  CLI uses `--from-panel-cache` / `--training-action panel_cache`. Cached model
  configurations are snapshotted per job; full resume checkpoints keep their saved
  topology. Reject changed training fingerprints before GPU boot, and never turn
  cached lifecycle/UI fields into trainer arguments. Test CLI parity against the
  Config WebUI builders with `python data_relay/tests/test_training_panel_cache.py`.
- Direct local delivery: root `upload_pretrain_coder_12b.bat` invokes
  `scripts/upload_pretrain_coder_12b.ps1` using
  `data_relay/bin/data_relay_delivery.exe` (separate binary so an active WebUI can
  keep running). It targets the configured
  instance's `/root/autodl-tmp/dataset`, cleans two named old SFT corpora and
  disposable caches, updates `master`, uploads Parquet plus available audit
  sidecars, and verifies SHA-256 before installing final files. The PowerShell
  script defaults to inspection; the BAT adds `-Execute`. CLI `deliver` defaults
  to inspection, accepts explicit `--remove-old`, and uses resumable SFTP
  `.incoming.rpart` files. `--init-repo` bootstraps Git metadata for an existing
  code copy; existing checkouts use `git pull --ff-only` and reject tracked
  changes or branch mismatch. `--clean-cache` is restricted to the documented
  cache/trash directories, preserves credentials and training outputs, and is
  opt-in on the CLI. `inspect-server` may boot in no-GPU mode to inspect the
  instance; both commands accept a temporary `--instance UUID` override.
  The PowerShell launcher fetches the selected branch locally and supplies a
  verified `--git-bundle` so the server can update via Git without GitHub access;
  original commit IDs, origin, and upstream are preserved.
  The BAT launches a hidden WMI worker through
  `scripts/start_pretrain_coder_12b_upload.ps1`, independent of the Codex process
  lifetime. `data_relay/relay_data/delivery-active.json` records PID, log and
  completion status; an exclusive file lock prevents duplicate workers. Restarting
  an interrupted delivery uses `-ResumeOnly` / CLI `--skip-git-update`, preserving
  the current remote revision and skipping cleanup. The launcher detects an
  existing worker and resumes stale/failed deliveries instead of starting a
  second upload. Resume requires an already initialized remote Git checkout.
  Delivery leaves the instance running. See the relay README for full flags;
  test the remote helper with `python data_relay/tests/test_delivery_remote.py`.
  CLI `remote-script --remote-script LOCAL_PYTHON_FILE` runs explicitly selected
  maintenance code on the instance using the same SSH credentials/host-key checks.
  `data_relay/src/cleanup_server_space.py` targets reviewed fixed names: remove
  an abandoned checkpoint `.tmp`, then SHA-256-verify every archive record and
  hard-link duplicate `out`/`checkpoints` weights. Preserve full resume states and
  training corpora. Hard-linked paths share in-place writes; atomic replacement
  separates them. Audit reports go to the remote `.cache/relay_storage_cleanup`.
  Test it with `python data_relay/tests/test_storage_cleanup.py`.
  `data_relay/src/check_server_python_env.py` diagnoses the instance env for the
  numpy/pyarrow ABI mismatch that breaks Config WebUI `st.dataframe` and
  `datasets` (`ImportError: numpy._core.multiarray failed to import`).
  Inspection is the default; `APPLY_REPAIR` / `--apply` upgrades pyarrow, or
  reinstalls a broken numpy pinned to its current metadata version, and refuses
  while a trainer runs. Audit reports go to the remote
  `.cache/relay_python_env`. Test it with
  `python data_relay/tests/test_server_python_env.py`.
- Uploaded `.py` / `.sh` scripts run **on the relay**, with the submitted
  filename and cwd `<data_dir>/tasks/<task_id>/`. Python uses `runner.python`
  (default `python3`); shell scripts use `bash`. `runner.env`, `RELAY_TASK_ID`,
  and `RELAY_TASK_DIR` are passed to the process. Dependencies must be installed
  on the relay. `task.json` persists state and `task.log` appends stdout/stderr.
  Exit code 0 plus `auto_upload=true` starts the upload pipeline; scripts can
  also be rerun or uploaded manually through `/api/tasks/{id}/start|upload`.
- Write deliverables to `output_dir` (default `out/`). Optional workspace
  `upload_paths.txt` lists **files relative to the task directory**, one per
  line, with blank lines / `#` comments ignored. Selected files are staged in
  `_relay_stage/` using hard links or copy fallback, preserving task-relative
  paths. The configured output directory must still exist. Apply the mandatory
  dataset identity cleaning/audit below on the relay before delivery; include
  cleaned artifacts and provenance/audit sidecars in the delivery selection.
- Browser data upload: `data_upload.rs` / `upload.js` accept 16 MiB SHA-256
  chunks (four browser workers), persist receipts for reload/restart resume,
  snapshot the target instance, and deliver verified files directly into
  `/root/autodl-tmp/dataset`. `dataset_upload_remote.py` verifies all hashes
  before atomic, non-overwriting installation from `dataset/.relay_incoming`.
  Dataset delivery never auto-cleans dataset or shuts down the instance;
  it serializes with training mode transitions. Tests: `cargo test` and
  `python data_relay/tests/test_dataset_upload_remote.py`. The browser smoke
  test requires an isolated local relay, Playwright, and `local-ui-test-only`.
  Transfer rate uses per-attempt `upload_transferred_bytes` excluding SFTP resume
  credits, plus `upload_phase` to separate connection, transfer and verification.
  UI samples every 3 seconds; pull rates are allocated-disk estimates, never
  sparse apparent-size throughput. Tests: `test_transfer_rates.cjs` and
  `test_transfer_rates_ui.cjs`.
- Remote dataset tab: `datasets.rs` / `datasets_remote.py` / `datasets.js`
  browse only `/root/autodl-tmp/dataset` on an explicitly selected instance,
  without booting it. Text preview is capped at 128 KiB. Deletion is limited
  to an individually confirmed unchanged regular file; reject symlink/hidden
  paths, serialize with delivery/training transitions, and check remote training
  idle first. Never recursively delete dataset directories or automatically
  remove audit sidecars. Tests: `test_datasets_remote.py`, `test_datasets_ui.cjs`.
- Training-code sync: `code_sync.rs` / `code_sync_remote.py` / `code_sync.js`
  snapshot the local working copy (`training.local_repo_root`, relative to the
  relay config), preview SHA-256 differences, then push only reviewed changes
  to the pinned instance/repo. Scope is training/model/scripts Python plus
  optional templates/tokenizer/requirements; exclude panel state, data, weights,
  checkpoints and secrets. Never prune remote-only files or install packages.
  Remote backups and transaction state live under `.relay_code_sync/<plan_id>`;
  partial installs leave ACTIVE and block managed training until recovery.
  Mutations use training_transition and the remote training control lock, reject
  active trainers/pipelines/autotune, verify staged bytes and syntax, and provide
  conflict-checked rollback. Neither preview nor apply boots an instance.
  Tests: `test_code_sync_remote.py`, `test_code_sync_ui.cjs`, Rust code_sync tests.
- **Default channel is `pull`**, requiring an instance-reachable
  `autodl.pull_base_url`. HTTP Range serves data through
  `/pull/{task_id}/{path}?t=<per-task-token>`; the instance uses aria2c
  (16 connections per file) or resumable curl and verifies file sizes.
  SSH/SFTP carries manifests, commands, quota checks, and progress only.
  `sftp` is the resumable push fallback (`.rpart`, same-size file skipping);
  `jupyter` uses base64 Contents API and defaults to a 64 MiB file limit.
  User download/cleaning scripts never run on AutoDL; pull runs a generated
  transfer script there. Manifest selection stages files under `_relay_stage`;
  `Relay::pull_root` serves that same root when `upload_paths.txt` is present.
- No-GPU boot uses the **console** API with `payload: "non_gpu"` and
  `console_token`. The Pro developer API cannot perform no-GPU boot; use an
  already running instance or console mode. Preserve the explicit GPU-boot
  guard (`allow_gpu_boot=false` by default); `boot_no_gpu` takes precedence.
  Select `instance_uuid` for multi-instance accounts (WebUI persists selection).
  SSH/Jupyter credentials are obtained from instance details, with an optional
  `ssh_password` override.
- Pull/SFTP preflight checks `used + incoming <= total * max_used_ratio -
  reserved_gb * GiB` (defaults: 0.92, 2 GiB). With `auto_clean=true`, it deletes
  oldest directories only under `remote_root` (default
  `/root/autodl-tmp/relay_uploads`), retaining the newest `keep_last_uploads=3`.
  Jupyter checks API disk figures without automatic cleanup. Preserve managed
  path boundaries and traversal checks when changing transfer/deletion code.
  Successful upload optionally requests shutdown (`power_off_after_upload=true`);
  failure/cancellation does not reach that shutdown step, and a shutdown error
  is logged but does not prevent task status `done`.
- API authentication uses `Authorization: <server.api_token>` (`Bearer` optional).
  Token push `/api/config/console_token?t=<api_token>` and pull use their own
  query-token checks. WebUI supports WeChat QR login, Tampermonkey push,
  bookmarklet push, and manual console-token refresh. `PUT /api/config` persists
  supported secret/instance/boot/shutdown fields; it is not a general config
  editor. Use strong access tokens, preserve credential redaction and host-key
  verification (`strict_host_key` defaults to false), and keep real secrets out
  of docs/tests. `config.toml`, `relay_data/`, `known_hosts.json`, and `target/`
  under `data_relay/` are ignored; `Cargo.lock` is tracked. Custom runtime/config
  locations need equivalent ignore protection.

## Architecture and config

- **Model configuration updates must reach Config WebUI.** `scripts/config_template_defaults.json` is the single catalog for built-in model presets and training recipes: `presets` holds model parameters, `defaults` holds shared recipe fields, `templates` holds per-model overrides, and `preset_training_keys` selects training fields applied during named-preset initialization. Config WebUI and the relay share the CPU-only loaders in `scripts/config_templates.py`; do not reintroduce model dictionaries or model-name-specific training branches in Python. When adding/changing a model, update the catalog and matching `trainer/config_*.json` / CLI launcher together. A standalone JSON or launcher is insufficient: the model must appear in **配置模板 → 已保存模板**, and loading it must apply the intended topology, precision, optimizer, checkpointing and batch settings. Account for CLI defaults that override model JSON fields.
- **Migrate existing template libraries.** New shipped templates must be added to already initialized `trainer/webui_templates.json` libraries on reload. Use default fingerprints to refresh only unchanged shipped templates; preserve custom templates, manual edits and intentional deletions. Do not reset the active `trainer/webui_state.json` or replace a user's saved recipe to expose a new model.
- **Verify all configuration surfaces.** Check the UI parameter count against the model, JSON/preset/template parity, template migration, and an actual Streamlit AppTest load plus generated-command check. Regenerate the relay schema with `python data_relay/generate_training_schema.py` and verify it with `--check` after changing presets or trainer arguments. Report where to select the model in Config WebUI; distinguish estimated memory from GPU measurements.
- **Instinct V2 Dense**: 32 layers, hidden=1536, FFN=5504, 16 Q heads / 4 KV heads, head_dim=96, tied embeddings, vocab=6400; **1,010,277,888 total / active parameters**. Native context 4096, RoPE theta=1e6, no inference scaling. `trainer/config_instinct_v2.json` and Config WebUI template **Instinct V2**. The shipped starting recipe uses BF16 parameters/compute, Muon mantissa with BF16 momentum, 128 MiB workspace, full-layer checkpointing (2), sequence length 2048, batch 1, accumulation 8, LR 3e-4, 2% warmup, compile/FP8 off.
- **V2 Dense fixed-budget recipe is separate**: `trainer/recipes/instinct_v2_16g_fixed.json` plus `.model.json` and `trainer/profiles/instinct_v2_16g_fixed/profile.json`. It uses bucket streaming, token budget 5344, tensorwise FP8, compile and different LR/workspace settings. Paths and fingerprints are local-machine/source/corpus specific. The profile remains `status: manual`, not an accepted autotune search result. Its `evidence` includes short warm throughput/memory and paired quality measurements; the older recipe `.report.json` records the initial unvalidated recommendation. Do not claim these finite windows establish long-run convergence or include cold compilation.
- **Instinct V2 MoE (standalone config)**: `trainer/config_instinct_v2_moe.json`; hidden=640, 32 layers, FFN/expert size=1728, 20 Q / 5 KV heads, head_dim=32, 8 experts, normalized top-2, tied embeddings, vocab=6400; **886,418,048 total / 249,408,128 active parameters**. Max position 32768 with LongRoPE from 4096. It currently has no built-in named preset/template in `scripts/config_template_defaults.json`; do not imply that it is selectable as a shipped WebUI template or has a validated training recipe. Wiring it into the UI requires the same catalog/migration/schema checks as other new presets.
- **Instinct V1 Dense**: 20 layers, hidden=768, FFN intermediate size 2432, 8 q-heads / 4 kv-heads, head_dim=96, tied embeddings; 152,406,528 total / active parameters (about 152.4M). Verified against `checkpoints/pretrain_20260912_180004_768.json`, used by the `instinct-v1-0914` evaluation. Vocab 6400, max_pos 32768, SwiGLU, RMSNorm, RoPE θ=1e6.
- **Dense legacy/default config**: 8 layers, hidden=768, 8 q-heads / 4 kv-heads. This is the old `instinct-3` / constructor default, NOT the trained Instinct V1 Dense configuration. Historical 8-layer experiment reports describe their own baselines.
- **InstinctV1Moe**: hidden=512, 32 layers, 16 q-heads / 4 kv-heads, 8 experts, top-2 routing with `norm_topk_prob=true`, MoE intermediate size 1664 (678,726,144 total / 187,992,576 active parameters; see `trainer/config_instinct_v1_moe.json`). Historical top-1 checkpoints and MFU experiments have 106,203,648 active parameters; use their saved routing configuration when loading or comparing them.
- Config in `model/model_instinct.py` → `InstinctConfig`. Defaults: `hidden_size=768`, `num_hidden_layers=8`, `use_moe=False`
- **V1 comparison context**: the user reports that V1 MoE was trained on a 34GB dataset, while V1 Dense used two mini datasets for pretraining and SFT. File size does not establish consumed token count or code-token coverage. Compare checkpoint configs and actual training/evaluation records; do not infer that MoE saw less total data, or equate its 678.7M total parameters with dense per-token capacity (V1 Dense: 152.4M active; historical top-1 V1 MoE: 106.2M; current top-2: 188.0M).
- **Alternate topologies**: `model/model_recurrsive_instinct.py` (RecurrsiveInstinct) and `model/model_instinct_linear.py` (linear attention). Shared trainers select them via `--model_architecture standard|linear|recurrsive_instinct` (`--use_recurrsive 1` is a compatibility alias). The wrappers `python run_recurrsive.py trainer/train_x.py` / `python run_linear.py trainer/train_x.py` also swap `sys.modules["model.model_instinct"]`. Do not edit the standard backbone to switch topology. Residual layouts `standard|mhc|attnres` are a separate config axis.
- Qwen3-compatible export is available for supported Dense/MoE configurations. Check the export path and runtime support for the actual topology and RoPE settings before claiming compatibility with `llama.cpp`, `vllm` or `ollama`; custom recurrent/linear/residual variants are not covered by that claim.

## Training pipeline (must respect order)

1. **Pretrain** (`train_pretrain.py`) — from scratch (`--from_weight none`), uses `pretrain_t2t(_mini).jsonl`
2. **Optional CPT** (`train_pretrain.py`) — continue from a completed `pretrain_*`/`cpt_*` weight on `pretrain*.jsonl`; use the Config WebUI `cpt` mode for separate defaults and `cpt_*` outputs
3. **SFT** (`train_full_sft.py`) — requires a completed pretrain/CPT base, uses `sft_t2t(_mini).jsonl`
4. **Optional**: LoRA (`train_lora.py`), DPO (`train_dpo.py`), PPO (`train_ppo.py`), GRPO/CISPO (`train_grpo.py`), Agent RL (`train_agent.py`), KD (`train_distillation.py`)

- A new stage from completed weights and a full-state resume are different
  operations. Restore the saved topology before applying permitted stage changes.
  `--moe_router_top_k` / `--moe_router_norm_topk_prob` explicitly migrate routing
  in new pretrain/CPT/SFT stages; combining migration with resume is rejected.
- `trainer/training_pipeline.py --plan PATH --validate-only` validates an exported
  plan without training. Pipeline stages have independent snapshots/processes and
  unique output prefixes under `trainer/pipeline_runs/`. Nonzero exits, including
  pause (42), stop later stages. Restarting the pipeline creates a new run; resume
  the paused stage through its trainer. Only stage-owned dataset caches are cleaned.

## Critical gotchas

### `__package__` hack
Training scripts and `scripts/` files set `__package__` + `sys.path.append` to resolve imports from the repo root. This is intentional — do NOT refactor away without understanding the import chain.

### Working directory matters
- **Training scripts**: MUST run from repo root. Data paths default to `./dataset/`, model paths to `./model/`, output to `./out/`, checkpoints to `./checkpoints/`. Reward model (`../internlm2-1_8b-reward`) lives as a sibling directory of the repo
- **API server / Config WebUI / Chat WebUI**: launch from `scripts/` or use the root BAT launchers. **Eval WebUI** uses the repo-root command in `start_eval_webui.bat`.

### Windows workaround
`trainer/` scripts import `datasets` before `torch` to work around a known pyarrow/torch DLL conflict on Windows. They also import `trainer.compile_cache` before `torch` to configure Inductor (compile cache). Do NOT reorder or remove either.

### Windows + torch.compile (`--use_compile 1`)
`torch.compile` on Windows **requires UTF-8 mode** — torch's own inductor template files (e.g. `torch/_inductor/kernel/mm_grouped.py`) crash with `UnicodeDecodeError: 'gbk' codec can't decode` otherwise. Set `PYTHONUTF8=1` in the environment of the training process (WebUI launches already inject it).

`model/compile_policy.py` sets per-region/accumulated specialization limits to
128/4096, configurable with `INSTINCT_COMPILE_RECOMPILE_LIMIT` and
`INSTINCT_COMPILE_ACCUMULATED_LIMIT`. Limit hits and compiler errors raise instead
of silently using eager training. Do not hide failures with `suppress_errors`,
reset caches per batch, or change shape policy just to avoid the limit.

### Packed and single-document attention

- `INSTINCT_PACKED_ATTENTION_BACKEND=auto|flex|sdpa` controls document-isolated
  packed attention; `INSTINCT_CAUSAL_ATTENTION_BACKEND=auto|flex|sdpa` controls
  plain causal dispatch. They are separate selectors. Preserve the exact
  same-document + causal predicate, label masks, RoPE resets and padding semantics.
- Build shared masks before checkpointed layers. For V2 Dense's 96-wide heads,
  FlexAttention stays traceable inside the outer compiled checkpoint; inserting a
  `torch.compiler.disable` boundary there can leave the whole layer eager.
- The loader intentionally drops `sequence_ids` for one document per row. V2's
  standard-residual training prepares a shared plain causal mask at model entry
  for that branch. Packed Flex alone does not cover the single-document bucket.
- Automatic 96-wide selection is gated on BF16, zero dropout, 16 Q / 4 KV heads
  and the validated PyTorch 2.13 / Blackwell SM12 stack with Triton. CPU, explicit
  caller masks and unsupported modes retain their existing paths. Explicit `flex`
  still checks eligibility and raises for missing dependencies; eligible runtime
  compilation errors are not silently swallowed. Use `sdpa` to select the fallback.
- See `docs/packed_flex_attention.md`, `model/packed_attention.py` and
  `model/flash_attn_4.py`. CUDA output/gradient/isolation and compiled-checkpoint
  tests are needed for dispatch/kernel changes; microbenchmarks alone do not
  establish full-training speed or quality.

### Muon split weights and momentum precision

- General defaults remain FP32 parameters/momentum. Explicit mantissa mode needs
  optimizer `muon` and BF16 parameters. BF16 high bits + uint16 low bits preserve
  the reconstructed FP32 weight update; forward sees truncated BF16 high bits,
  so exact update arithmetic does not imply FP32-master training equivalence.
- BF16 momentum is optional and requires mantissa mode. Momentum/update arithmetic
  stays FP32; the new EMA is stored in BF16 for the next step. This quantization
  changes the trajectory relative to FP32 momentum. Embeddings, tied heads, norms
  and biases assigned to the companion AdamW stay FP32.
- CLI controls override `INSTINCT_MUON_MANTISSA`,
  `INSTINCT_MUON_MOMENTUM_DTYPE`, `INSTINCT_MUON_BATCH_SIZE` and
  `INSTINCT_MUON_WORKSPACE_MB`; zero/omitted chunk limits retain defaults.
  Resume must preserve the saved precision and raw uint16 state. A precision
  change requires a new stage from weights, not silent resume-state conversion.
- Fused V1 and V2 shapes are explicitly gated in `trainer/batched_muon.py` and
  `trainer/muon_kernels.py`; other shapes retain the reference implementation.
  Report measured peak memory separately from per-element storage estimates.
  See `docs/muon_mantissa_fusion.md`, `docs/muon_bf16_momentum.md` and
  `tests/test_muon_v2.py` for evidence and its limits.

### Training CUDA fast paths (Muon + MoE)

Offline whole-step autotuning: `python trainer/autotune.py --config trainer/autotune_pretrain.json`.
Config WebUI exposes this in the **自动调优** tab (prepare/search, progress, results): snapshot the shared
training command/model, launch a detached CPU supervisor, monitor/cancel its owned
process tree, export/apply validated profiles. Profiles also propagate into the
pipeline's pretrain snapshot only. Tuning blocks concurrent panel training;
autotune trial processes must never be recovered as formal training. Keep UI
integration CPU-only and let trainer children enforce full compatibility/audits.
Use `--plan-only` to inspect the search without training. It runs isolated real
pretrain trials, keeps checkpointing enabled, gates on common validation and
training losses, and writes a machine/source/corpus-bound `profile.json` plus
the full launch command. Token-budget feasibility uses binary search separately
for each execution strategy; only OOM/driver-budget failures prune the interval.
Nearby smaller batches still compete on throughput, and nonmonotonic evidence
disables pruning. There is no required or hardcoded throughput target.
Production accepts `--autotune_profile PATH`; explicit
conflicts or changed fingerprints fail, and resume retains its saved batch cursor.
Search is bounded by trial count only, with no total or per-trial timeouts;
manual cancellation still stops the owned process tree.
Current scope is single-GPU standard Dense/MoE from-scratch bucket-streaming
pretraining; not SFT/RL/DDP. See `docs/training_autotune.md` for budgets, supported
search axes and cache behavior. Tests: `python -m pytest tests/test_autotune.py`.

On PyTorch 2.13 / Blackwell SM12 with Triton, shared training code automatically
uses validated Muon NS kernels/graphs, fused FP32 momentum/update operations,
expert BF16 packing and SwiGLU. Gradient checkpointing stays enabled. CPU and
unsupported shapes/dtypes/layouts retain their original paths. See `PLAN.md` for
measured MFU and correctness evidence; do not infer the same speed on other GPUs.

- `INSTINCT_MUON_NS_BACKEND=auto|torch|triton|triton_graph`; default `auto`.
- `INSTINCT_MUON_FUSED_UPDATES=auto|0|1`, `INSTINCT_MOE_FUSED_PACK=auto|0|1`,
  `INSTINCT_MOE_FUSED_SWIGLU=auto|0|1`; all default `auto`.
- Normalized top-2 local production baseline (2026-10-13): 444.821 ms/step,
  34.471% estimated useful BF16 MFU in the recorded 80-step warm window.
  It mixes 2x4096 and 4x2928 batches from the real pretrain_coder_12b Arrow loader.
  See PLAN.md for exclusions and paired 100-step training/validation evidence.
- New validated local defaults: `INSTINCT_MOE_FUSED_ROUTING=auto`,
  `INSTINCT_CAUSAL_ATTENTION_BACKEND=auto` (training causal Flex on Windows SM120),
  `INSTINCT_FLEX_AUTOTUNE=auto`, `INSTINCT_MOE_TRANSPOSED_DW=auto`;
  local CPU training threads default to 1 after initialization and FP32 Muon
  workspace to 256 MiB. Overrides are `INSTINCT_TRAINING_CPU_THREADS` (0 preserves
  platform settings) and `INSTINCT_MUON_WORKSPACE_MB`.
- `INSTINCT_MOE_SELECTIVE_CHECKPOINT`, `INSTINCT_MOE_DIRECT_WEIGHTS`, and
  `INSTINCT_MOE_INNER_CHECKPOINT` are opt-in experiments, default 0. Inner mode
  checkpoints gate/up inside the fused operator; never also replay the whole FFN.
- Inner checkpoint accepts `1` (retain the down-projection input) and `2`
  (recompute it inside backward, reusing an operator-owned temporary). Fixed
  tensor speedups did not transfer to the real larger-batch loader; keep both
  opt-in. `INSTINCT_MUON_PREALLOCATE=1` reserves zero momentum before activation
  allocation without taking an optimizer step; default 0.
- On the validated local Windows RTX 5070 Ti V1 top-2 path, automatic packing
  caps the memory-derived budget at 16,384 tokens for throughput. The final
  value can be lower when memory is tighter; explicit and saved resume budgets
  retain precedence. CLI allocator limits also respect a uniquely matched
  adapter's DXGI process budget. Unsupported/ambiguous adapters fall back.
  Do not assume fully occupied VRAM maximizes MFU or extrapolate this curve to
  unmeasured systems. See PLAN.md for the selective-checkpoint/batch sweep.
- FP8 conversion counts do not establish execution coverage: grouped MoE reads
  expert weights directly and currently executes BF16. Verify kernel traces before
  claiming FP8 speedups. Unsloth FP8 LoRA / NVFP4 inference results do not establish
  full-parameter pretraining support; local probe is experiments/probe_fp8_training.py.
- For the original NS/update/pack/SwiGLU comparison, set NS to `torch` and
  `INSTINCT_MUON_FUSED_UPDATES`, `INSTINCT_MOE_FUSED_PACK`,
  `INSTINCT_MOE_FUSED_SWIGLU` to `0`. Other attention/routing controls are separate.
  `INSTINCT_MUON_BACKEND=native` remains the native optimizer option.
- Preserve explicit FMA, BF16 intermediate rounding, FP32 decay rounding, and
  subnormal handling in these kernels. The symmetric NS GEMMs do **not** change
  the iteration into the rejected Gram recurrence. NS graphs contain only the
  pure transform: never capture a Python LR or replace grad=None with zero.
- Tests: `python -m pytest tests/test_muon_kernels.py`; full compiled parity:
  `python experiments/validate_paired_training_kernels.py --deterministic --output experiments/results/parity.json`.
  Ordinary compiled training has nondeterministic gradients even between
  reference repeats; use the deterministic control when diagnosing kernel drift.

### SwanLab logging
The legacy flag is `--use_wandb`, but the shared logger imports `swanlab as wandb`.
Remote experiment logging is disabled unless that flag is supplied and runs on
the main process. Preserve local logs and the saved run ID used on resume.

### `max_seq_len` is in tokens, not characters
Measure lengths with the repo's tokenizer; character/token ratios vary by
language and code. In bucket packing, also check `bucket_max_seq_len` and the
actual bucket shapes. Keep training lengths within the selected model's context
and distinguish native training context from inference-only RoPE extrapolation.

### Checkpoint paths
- Simple saves: `out/{weight}_{hidden_size}{_moe}.pth`
- Resume checkpoints: `checkpoints/{weight}_{hidden_size}{_moe}_resume.pth`
- Matching configuration: `checkpoints/{weight}_{hidden_size}{_moe}.json`.
  Preserve full resume states (model, optimizer, scaler, epoch/step and data state) alongside
  lightweight inference weights; the latter do not provide an exact resume.
- The `--from_resume 1` flag handles auto-detection; works across GPU count changes

### WebUI model discovery
`web_demo.py` scans `out/` and `checkpoints/` for non-resume `.pth` files and also
accepts a custom path. `scripts/web_demo_utils.py::resolve_model_config_path`
prefers a JSON beside the weight, then the matching JSON in `checkpoints/`, then
legacy `trainer/config_pretrain.json`. Check that fallback before diagnosing a
topology mismatch. The default tokenizer is the repo's `model/` directory.

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
| `--checkpoint_path PATH --config_path PATH` | Explicit raw weight and its matching topology/RoPE JSON |
| `--use_moe 1` | MoE model |
| `--lora_weight lora_medical` | Apply LoRA on top |
| `--inference_rope_scaling` | Enable default YaRN; use `--config_path` for saved LongRoPE settings |
| `--open_thinking 1` | Adaptive thinking mode |
| `--inference_compile auto\|full\|off` | Inference optimization mode |
| `--benchmark livecodebench` | LiveCodeBench generation/evaluation; see `--lcb_*` options |

## Model conversion
`scripts/convert_model.py` provides torch/Transformers conversion and LoRA merge.
The compatibility export selects Qwen3 Dense or Qwen3 MoE classes. Its functions
use the module-level `lm_config`; the example main block hardcodes the legacy
8-layer Dense model. Load the real checkpoint configuration and set paths before
running it. `convert_torch2transformers_instinct` preserves custom Instinct code
and LongRoPE semantics that the Qwen3 export explicitly rejects (retained-start
thresholds or separate short/long attention factors).

## Where to read more
- `data_relay/README.md` / `data_relay/config.example.toml` — relay deployment, AutoDL token refresh, transfer channels, and disk management (verify implementation when comments differ)
- `PLAN.md` — dated performance experiments, correctness gates and historical baselines; match the recorded topology/hardware before reusing a number
- `scripts/config_template_defaults.json` / `scripts/config_templates.py` — shared presets, recipes and template migration; `trainer/config_*.json` — standalone model configurations
- `docs/training_pipeline.md` — pipeline, stage/resume distinctions and ETA; `docs/compact_dataset_cache.md` — cache design; `scripts/data_loader/README.md` — loader entry points
- `docs/packed_flex_attention.md` — packed/causal Flex dispatch and V2 compiled checkpointing
- `docs/training_autotune.md` — isolated search, profiles and production compatibility
- `docs/muon_bf16_momentum.md` / `docs/muon_mantissa_fusion.md` — precision contracts and measured validation; `docs/muon_low_precision_audit.md` includes historical pre-fusion results
- `dataset_compiler/README.md` / `scripts/data_builder/README.md` — Rust compilation/verification and data preparation/audit entry points
- `data_relay/deploy/ONLINE.md` — monitor/proxy deployment; keep its private staging archives and credentials out of the repository
- `docs/gsm8k.md` / `docs/humaneval.md` / `docs/eval_webui.md` — benchmark harness and Eval WebUI details
- `docs/repo_layout.md` — repository map: training / data acquisition / artifacts & checkpoints / tools, tracked-vs-ignored status, stable-path contract and cleanup rules
