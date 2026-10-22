# 仓库目录总览

本文件是 Instinct 仓库的完整目录地图：说明四类代码/资源分别放在哪里、哪些路径入库、
哪些是运行产物，以及哪些路径属于外部工具依赖的**稳定契约**。新增文件前请先读本文；
根目录的简版结构见 [README.md](../README.md) / [README_en.md](../README_en.md)。

仓库按以下五类组织：

1. **训练代码**（模型、训练器、训练配置）
2. **数据获取与构建**（加载、离线构建、语料编译、下载/交付中继）
3. **产物与检查点**（权重、续训状态、评测/实验输出，全部不入库）
4. **工具与运行入口**（根目录入口脚本、启动器、WebUI/API/转换/监控）
5. **文档与测试**（docs/、tests/、README 与项目规范）

---

## 1. 完整目录树

`# 入库` / `# 忽略` 标注该路径在 `.gitignore` 中的状态（"忽略"=运行产物，默认不提交）。

```text
Instinct/
├─ model/                    # 入库 · 训练：InstinctConfig / 模型主干 / LoRA / 推理运行时 + tokenizer
├─ trainer/                  # 入库 · 训练：全部 train_*.py 与共享训练基础设施
│  ├─ train_*.py             #   pretrain / SFT / LoRA / DPO / PPO / GRPO / Agent RL / KD / tokenizer
│  ├─ trainer_cli.py, trainer_utils.py, training_pipeline.py, streaming_pretrain.py,
│  │  packing_transition.py, training_eta.py, training_profiler.py, rollout_engine.py,
│  │  compile_cache.py       #   共享 CLI / 工具 / 流水线 / 流式加载 / ETA / 性能分析 / 编译缓存
│  ├─ batched_muon.py, muon_kernels.py, autotune*.py   # Muon 融合内核与离线自动调优
│  ├─ config_*.json          #   各模型/阶段的训练配置
│  ├─ recipes/               #   具名训练配方（含 .model.json / .report.json）
│  ├─ profiles/              #   autotune 产出的 profile（手动/已验证状态以内容为准）
│  └─ logs/                  # 忽略 · trainer 日志 + 关机监控日志
├─ run_recurrsive.py         # 入库 · 训练：RecurrsiveInstinct 包装入口（根目录，勿移动）
├─ run_linear.py             # 入库 · 训练：linear attention 包装入口（根目录，勿移动）
├─ scripts/train_instinct_v2.ps1  # 入库 · 训练：V2 Dense 启动器
├─ scripts/data_loader/      # 入库 · 数据：训练侧加载/格式解析/packing/流式分块/缓存预算
├─ scripts/data_builder/     # 入库 · 数据：离线收集/构建/清洗/审计（见其 README.md）
│  └─ l3_builder/            # 入库 · 数据：Rust 12B CPT 语料构建器（cargo 构建，target/ 忽略）
├─ dataset_compiler/         # 入库 · 数据：Rust JSONL→Parquet 编译/校验（Cargo.lock 入库）
├─ data_relay/               # 入库 · 数据：Rust 下载/清洗/交付中继（src/ tests/ deploy/ README）
│  └─ bin/                   # 忽略 · 数据：预编译 Windows relay exe
├─ dataset/                  # 忽略(数据)/入库(.md、审计说明)：语料本体、provenance、审计报告
├─ out/                      # 忽略 · 产物：推理权重 out/{weight}_{dim}{_moe}.pth + 同名 .json
├─ checkpoints/              # 忽略 · 产物：完整续训状态 *_resume.pth + 同名 .json
│  └─ _archive/              # 忽略 · 产物：退役/重复检查点归档（归档而非删除）
├─ eval/                     # 忽略 · 产物：评测产物；运行记录在 eval/runs/<时间戳>_<ID>/
├─ experiments/              # 入库(脚本/证据 json)/忽略(results/、data/)
│  ├─ probes/                #   一次性诊断脚本归档（_dense_*.py 等）
│  ├─ results/               # 忽略 · 实验输出
│  └─ data/                  # 忽略 · 实验用合成数据
├─ profiler_traces/          # 忽略 · 产物：torch profiler trace
├─ images/                   # 入库 · 文档：README/文档图片
├─ docs/                     # 入库 · 文档
│  ├─ architecture/          #   架构与导览 HTML：architecture.html、architecture-v1.html、
│  │                         #   architecture-v1-moe.html、architecture-diagram.html、code-tour.html
│  ├─ reports/               #   专题/验证报告：VERIFICATION_REPORT.md、REBUILD_STATUS.md
│  ├─ LongRoPE.md            #   长上下文 RoPE 说明
│  └─ *.md                   #   training_pipeline、compact_dataset_cache、packed_flex_attention、
│                            #   training_autotune、muon_*、gsm8k、humaneval、eval_webui 等
├─ tests/                    # 入库 · 测试：pytest 套件；gpu/slow marker；tests/smoke/ 冒烟用例
├─ scripts/                  # 入库 · 工具：Config/Chat/Eval WebUI、OpenAI API、模型转换、
│                            #   HTTPS 证书、关机监控、自动调优面板、数据工具（可导入为 scripts.*）
├─ eval_*.py                 # 入库 · 工具：根目录推理/评测入口（eval_llm、eval_gsm8k、
│                            #   eval_humaneval、eval_humaneval_oj、eval_batch、eval_pass_k、
│                            #   eval_report、eval_data）
├─ start_*.bat               # 入库 · 工具：根目录启动器（Config/Chat/Eval WebUI、OpenAI API、data relay）
├─ tmp/                      # 忽略 · 产物：临时脚本与缓存
├─ .relay_training/          # 忽略 · 产物：relay 托管训练的本地状态
├─ .relay_code_sync/         # 忽略 · 产物：code_sync 备份与事务状态
├─ requirements.txt / requirements-fp8.txt / pytest.ini  # 入库 · 工具：依赖与 pytest 配置
└─ README.md / README_en.md / AGENTS.md / PLAN.md / CODE_OF_CONDUCT.md / LICENSE  # 入库 · 文档
```

其他工具缓存（`.cache/`、`.codegraph/`、`.omo/`、`.sisyphus/`、`.playwright-mcp/`、
`__pycache__/`、`.pytest_cache/`、`certs/`、`trainer/webui_state.json`、
`trainer/webui_templates.json`、`data_relay/config.toml`、`data_relay/relay_data/`、
`data_relay/known_hosts.json`、仓库根 `relay_data/`）同样被忽略；具体规则以
[.gitignore](../.gitignore) 为准。

---

## 2. 产物生命周期与清理规则

- **out/**：轻量推理权重（`.pth` + 同名 `.json`）。可由训练重新产出，但对评测/WebUI 直接可用；
  删除前先确认有对应 checkpoint 或可重新推理的权重。
- **checkpoints/**：完整续训状态（模型、优化器、scaler、epoch/step 与数据游标）及匹配的
  `.json`。**训练尚未暂停保存前，严禁手动删除完整续训状态**。正确流程是发送暂停
  （WebUI"暂停训练"或创建 `checkpoints/.pause_request`），等待当前优化器更新完成、
  保存成功并以退出码 42 结束，再核对 `out/` 权重与 `checkpoints/*_resume.pth` 都存在。
- **归档而非删除**：退役或重复的检查点/权重移到 `checkpoints/_archive/`，文件名带
  日期或用途后缀（如 `..._duplicate.pth`），移动前用 SHA-256 核对，并保留同名 `.json`。
  不要原地覆盖/重命名仍在使用的续训文件。
- **硬链接**：relay 的清理/存储工具会在 `out/` 与 `checkpoints/` 之间创建硬链接。原地写入会
  同时影响两个路径；确需替换时用"临时文件 + 原子 rename"，让两路径分离后再处理。
- **eval/**、**experiments/results|data/**、**profiler_traces/**、**tmp/**：可重新生成的产物，
  可在无训练/评测任务运行时清理或归档；保留审计报告（`.md`/`_metrics.json` 等）便于复查。
- **trainer/logs/**：trainer 日志与 `scripts/monitor_training_shutdown.ps1` 的关机监控日志；
  保留最近一次运行的日志用于事后排查，其余可轮转/归档。
- **.relay_training/**、**.relay_code_sync/**：由 relay 托管，训练/同步进行中不要手改；
  部分安装会留下 ACTIVE 事务并阻塞托管训练，需按 relay 流程恢复。
- 清理优先级：**重新生成 > 归档 > 删除**。数据集本体、provenance/审计 sidecar、
  当前最佳权重与任何 `*_resume.pth` 不在常规清理范围内；远程清理使用
  `data_relay/src/cleanup_server_space.py` 的固定名单流程。

---

## 3. 稳定契约（不要移动/重命名）

- **根目录入口**：`eval_llm.py`、`eval_*.py`、`run_recurrsive.py`、`run_linear.py`、
  `start_*.bat` 是外部习惯用法与文档引用的公开路径。训练脚本固定在
  `trainer/train_*.py`，并且必须从仓库根目录运行（默认路径是 `./dataset/`、`./model/`、
  `./out/`、`./checkpoints/`）。
- **导入契约**：可导入代码保持在 `scripts.*` / `trainer.*`；文件头部的 `__package__` +
  `sys.path.append` 依赖仓库根目录布局，不要为整理目录新增一层包或移动模块。
- **配置源**：`scripts/config_template_defaults.json` + `scripts/config_templates.py` 是内置模型
  预设/训练配方的唯一目录；`trainer/config_*.json` 是与权重配套的模型配置 sidecar。
- **WebUI/权重发现**：Chat WebUI 直接扫描 `out/` 与 `checkpoints/` 下的非 `*_resume.pth`；
  `scripts/web_demo_utils.py::resolve_model_config_path` 依次查找权重旁 `.json`、
  `checkpoints/` 同名 `.json`、旧版 `trainer/config_pretrain.json`；默认 tokenizer 是仓库
  `model/` 目录。不要改输出前缀或移动 tokenizer。
- **检查点命名**：`out/{weight}_{dim}{_moe}.pth`、`checkpoints/{weight}_{dim}{_moe}_resume.pth`
  与 `checkpoints/{weight}_{dim}{_moe}.json` 构成一组，改名会破坏 `--from_resume 1` 的自动识别。
- **data_relay**：相对 `server.data_dir` 以进程工作目录为基准解析，因此保持固定的启动 cwd
  （根目录 `start_data_relay.bat` 已处理）；冻结的启动器候选路径是 `data_relay/bin/*.exe` 与
  `data_relay/target/release/data_relay.exe`；code_sync/deploy 按仓库根布局（`trainer/`、
  `model/`、`scripts/`、根 `run_*.py`/`eval_llm.py`）同步与部署，不要把这些入口移进子目录。
- **暂停哨兵**：`checkpoints/.pause_request` 是 trainer 的暂停约定文件，不要移动该目录。
- **语料路径**：训练配置与文档引用 `dataset/...`；`dataset_compiler --align-chunk-bytes`
  要与 trainer 的 `--streaming_chunk_mb` 对齐，否则流式游标在格式切换后不精确。
- **文档链接**：README 指向 `docs/architecture/*.html`；架构图/报告按 `docs/` 分类存放，
  不要再放回根目录。

---

## 4. 新文件放置规则

| 新增内容 | 放置位置 |
|---|---|
| 模型结构/推理运行时 | `model/`（并检查 `scripts/config_template_defaults.json` 与 `trainer/config_*.json`） |
| 训练脚本、共享训练基础设施、模型配置 | `trainer/`（`train_*.py`、`trainer_*.py`、`config_*.json`；配方进 `recipes/`） |
| 训练数据加载/格式/packing/缓存 | `scripts/data_loader/` |
| 离线数据收集/构建/清洗/审计（Python） | `scripts/data_builder/` |
| 大规模数据构建（Rust） | `dataset_compiler/` 或 `scripts/data_builder/l3_builder/` |
| relay 功能（Rust）与远程维护脚本 | `data_relay/src/`、`data_relay/tests/`、`data_relay/deploy/` |
| WebUI/API/模型转换/监控/面板等工具 | `scripts/`（保持 `scripts.*` 可导入） |
| 新 CLI/评测入口 | 根目录 `eval_*.py` / `run_*.py` / `start_*.bat`（仅当确实需要根路径） |
| 文档、架构页、报告 | `docs/`；架构/导览 HTML → `docs/architecture/`；专题报告 → `docs/reports/` |
| 测试 | `tests/`（标注 `gpu` / `slow`，冒烟用例放 `tests/smoke/`） |
| 运行产物/日志/权重/临时文件 | 只能写入已忽略目录：`out/`、`checkpoints/`、`eval/`、`trainer/logs/`、`experiments/results|data/`、`profiler_traces/`、`tmp/` |
| 新增被忽略的输出路径 | 先更新 `.gitignore` 再生成文件 |

---

## 5. 相关文档

- [README.md](../README.md) / [README_en.md](../README_en.md) — 快速开始、训练管线与部署说明
- [AGENTS.md](../AGENTS.md) — 协作规范、关键命令与易错点
- [dataset_compiler/README.md](../dataset_compiler/README.md) — JSONL→Parquet 编译/校验 CLI
- [scripts/data_builder/README.md](../scripts/data_builder/README.md) — 数据构建/清洗/审计入口
- [scripts/data_loader/README.md](../scripts/data_loader/README.md) — 训练侧加载器入口
- [data_relay/README.md](../data_relay/README.md) — relay 部署、传输通道与磁盘管理
- [docs/training_pipeline.md](training_pipeline.md) — 训练流水线、阶段/续训区分与 ETA
