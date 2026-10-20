# Instinct Agent Instructions

## What this is
Instinct — train InstinctV1Moe from scratch on a single GPU. The current V1 MoE configuration uses normalized top-2 routing with about 678.7M total parameters / 188.0M active parameters. All core algorithms (Dense/MoE Transformer, Pretrain, SFT, LoRA, DPO, PPO, GRPO, CISPO, Agentic RL, KD) implemented in pure PyTorch — no `trl`/`peft` abstractions.

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
data_relay/        # Rust relay server: run download/cleaning scripts, auto-boot AutoDL in
                   # no-GPU mode, transfer via HTTP pull (default)/SFTP/Jupyter, manage disk quota
scripts/        # Inference, API server, WebUI, model conversion, training launchers
eval_llm.py     # CLI inference script
eval_*.py       # Benchmark harnesses (gsm8k, humaneval, pass_k, batch, report); artifacts land in eval/
tests/          # Pytest suite; `gpu` marker = needs CUDA
experiments/    # Perf/research experiment scripts (lambda sweeps, synthetic data, plots)
docs/           # Design docs: training_pipeline.md, compact_dataset_cache.md, gsm8k.md, humaneval.md, eval_webui.md
```

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
- While training is running, the WebUI config has a "Pause Training" button: the trainer finishes its current step, saves `out/{weight}_{dim}{_moe}.pth` plus full resume state `checkpoints/{weight}_{dim}{_moe}_resume.pth`, and exits. To resume, tick "Resume from checkpoint" in the WebUI and press Start Training (or run with `--from_resume 1`).
- CLI equivalent: `touch checkpoints/.pause_request` pauses any running trainer at the next step boundary (same save + exit code 42); `--from_resume 1` resumes.
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
  root manager/relay executables and `data_relay/target/release/data_relay.exe`.
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
  `scripts/upload_pretrain_coder_12b.ps1` using `data_relay_delivery.exe`
  (separate binary so an active WebUI can keep running). It targets the configured
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
- **Instinct V2 Dense**: 32 layers, hidden=1536, FFN=5504, 16 Q heads / 4 KV heads, head_dim=96, tied embeddings, vocab=6400; 1,010,277,888 total / active parameters. Native context 4096, RoPE theta=1e6, no inference scaling. `trainer/config_instinct_v2.json` and Config WebUI template **Instinct V2**. Local 16 GiB starting recipe: BF16 parameters/compute, Muon mantissa compensation with BF16 momentum, 128 MiB workspace, full-layer checkpointing (2), sequence length 2048, batch 1, accumulation 8, compile/FP8 off. Full GPU training memory/throughput is not yet measured.
- **Instinct V1 Dense**: 20 layers, hidden=768, FFN intermediate size 2432, 8 q-heads / 4 kv-heads, head_dim=96, tied embeddings; 152,406,528 total / active parameters (about 152.4M). Verified against `checkpoints/pretrain_20260912_180004_768.json`, used by the `instinct-v1-0914` evaluation. Vocab 6400, max_pos 32768, SwiGLU, RMSNorm, RoPE θ=1e6.
- **Dense legacy/default config**: 8 layers, hidden=768, 8 q-heads / 4 kv-heads. This is the old `instinct-3` / constructor default, NOT the trained Instinct V1 Dense configuration. Historical 8-layer experiment reports describe their own baselines.
- **InstinctV1Moe**: hidden=512, 32 layers, 16 q-heads / 4 kv-heads, 8 experts, top-2 routing with `norm_topk_prob=true`, MoE intermediate size 1664 (678,726,144 total / 187,992,576 active parameters; see `trainer/config_instinct_v1_moe.json`). Historical top-1 checkpoints and MFU experiments have 106,203,648 active parameters; use their saved routing configuration when loading or comparing them.
- Config in `model/model_instinct.py` → `InstinctConfig`. Defaults: `hidden_size=768`, `num_hidden_layers=8`, `use_moe=False`
- **V1 comparison context**: the user reports that V1 MoE was trained on a 34GB dataset, while V1 Dense used two mini datasets for pretraining and SFT. File size does not establish consumed token count or code-token coverage. Compare checkpoint configs and actual training/evaluation records; do not infer that MoE saw less total data, or equate its 678.7M total parameters with dense per-token capacity (V1 Dense: 152.4M active; V1 MoE: 106.2M active).
- **Alternate topologies**: `model/model_recurrsive_instinct.py` (RecurrsiveInstinct) and `model/model_instinct_linear.py` (linear attention). Run trainers through the wrappers `python run_recurrsive.py trainer/train_x.py` / `python run_linear.py trainer/train_x.py`, which swap `sys.modules["model.model_instinct"]` — don't edit `model_instinct.py` to switch topology
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
- For a full original-path reference, set NS to `torch` and all three fusion
  flags to `0`. `INSTINCT_MUON_BACKEND=native` remains the native optimizer option.
- Preserve explicit FMA, BF16 intermediate rounding, FP32 decay rounding, and
  subnormal handling in these kernels. The symmetric NS GEMMs do **not** change
  the iteration into the rejected Gram recurrence. NS graphs contain only the
  pure transform: never capture a Python LR or replace grad=None with zero.
- Tests: `python -m pytest tests/test_muon_kernels.py`; full compiled parity:
  `python experiments/validate_paired_training_kernels.py --deterministic --output experiments/results/parity.json`.
  Ordinary compiled training has nondeterministic gradients even between
  reference repeats; use the deterministic control when diagnosing kernel drift.

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
- `data_relay/README.md` / `data_relay/config.example.toml` — relay deployment, AutoDL token refresh, transfer channels, and disk management (verify implementation when comments differ)
- `PLAN.md` — current performance-optimization plan for this branch (`codex/better_performance`): SDPA attention path, batch-level combined mask, selective gradient checkpointing, packing
- `docs/training_pipeline.md` — end-to-end pipeline; `docs/compact_dataset_cache.md` — dataset cache design
- `docs/gsm8k.md` / `docs/humaneval.md` / `docs/eval_webui.md` — benchmark harness and Eval WebUI details
