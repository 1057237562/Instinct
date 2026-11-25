# instinctv2_sft_santitized_2b — 运行日志与评测归档（2026-11-25）

本次 SFT 运行 `native_instinctv2_sft_santitized_2b` 的训练/评测日志快照。所有文件都是从被
`.gitignore` 排除的 `tmp/`、`eval/` 目录原样复制，便于在仓库历史中留档；权重、resume 状态、
pth、生成样本等大文件未纳入。

## 运行身份

| 项 | 值 |
| --- | --- |
| 数据 | `dataset/instinctv2_sft_santitized_2b`（1,514,592 行 / 77,919 packed 16K blocks / 1,265,404,059 packed valid tokens） |
| 基座 | V2 Dense pretrain acceptance `checkpoints/native_cublas/cloud_step2004666/native_v2_1b_acceptance_1536.npz`（step 2,004,666） |
| 配置 | `tmp/native_sft_longrope128k/model_smooth.config.json`（smooth LongRoPE：`switch_mode=smooth`、smooth_start=4096、smooth_end=8192、factor=32、max_position=131,072） |
| 训练 | full_sft，16,384 packed，`v2_mxfp8_sft`，Muon mantissa/bf16 momentum，save_interval 512，async checkpoint，LR 2e-6 → 1e-5（warmup 128 updates）→ 2e-6，end_step 77,919 |
| 代码 | 分支 `codex/longrope-fix`，smooth 相位修复提交 `1a5fd1b` |

## 事件时间线（2026-11-25）

| 时间 | 事件 |
| --- | --- |
| 04:00 | 保存 65,536（weights + resume 一致） |
| 04:08 | 归档任务复制 resume npz，持有读句柄 |
| 04:08–04:16 | 权重推进到 66,048 并写出；随后 66,560 保存执行 `os.replace` 失败 —— `PermissionError(13, '拒绝访问。')`，训练退出 `failed` |
| 04:16 | 自动评测误用 resume 65,536 + weights 66,048，`run_eval.py` 抛 `Unexpected native inference checkpoint`，未产生任何生成 |
| 08:00 | 定时任务再次启动评测，同样失败；`REPORT.md` 未生成 |
| 20:12 | 修复后手动评测 **step 66,048**（weights npz 的 `progress` 作为唯一来源），EvalPlus + LCB 全部完成 |
| 21:00 | 写出 `eval/REPORT.md`（总表见下） |
| 20:58 | 从 **resume 65,536** 续训（66,048 无配对 optimizer 状态，不能续） |
| 21:09–21:10 | 收到暂停请求，在 step **66,052** 完成保存并退出，`status=paused` |

## Checkpoint 状态

| step | tokens | weights SHA-256 | resume SHA-256 | 说明 |
| ---: | ---: | --- | --- | --- |
| 35,840 | 180,355,072 | 见 `archive/index.json` | 见 `archive/index.json` | 首次中期归档 |
| 50,688 | 823,006,842 | `3f3e1501e18d3782a9343944a26b776545408e0670da0972bedf6439b5f3cf97` | 见 `archive/index.json` | 第一次完整评测（探针） |
| 65,536 | 1,064,208,249 | — | `c76ce01b805c90da83937b8964b2bc1eaa54f3a5a3c50e700926ae308f2d35be` | 66,560 失败后唯一一致的完整状态，续训起点 |
| 66,048 | 1,072,520,943 | `3adb083ad2b66b6f8076f1c68c7201ebba7df564aa0607e6d029dc7c5c647a71` | 未写出（失败） | 本次评测对象；转换权重 `step66048.pth` SHA-256 `0250ef7757b43425445e5b3d04b9f3174c426dff0dea7e7b1367ef7795527d7d` |
| 66,052 | 1,072,586,204 | `a18c70f4b997a6b8f618c5caee06383c5aa17a3f5abbe09a4447a6ad36595fa7` | `6bc3d5313d6b3d4b93959f289bbb47380614daab14008cfbf77e0ba34e40bf7b` | 保存并暂停的最终状态（status=paused） |

## 评测结果（记录协议）

EvalPlus 记录驱动（thinking off，164×10 HumanEval+ / 378×10 MBPP+，temp 0.8 / top_p 0.95 /
top_k 50 / seed 42 / max_new 512，batch 32）；LCB release_v6 latest-175，greedy，think ctx
**8192**、no-think ctx **4096**，MXFP8 E4M3 KV。

| 指标 | @66,048 | @50,688 | R1 pos @30,720 | v4 @3,000 |
| --- | ---: | ---: | ---: | ---: |
| HumanEval+ pass@1 / pass@10 | 0.3610 / 0.6585 | 0.3476 / 0.6463 | 0.4018 / 0.6768 | 0.4421 / 0.7317 |
| HumanEval base pass@1 / pass@10 | 0.3945 / 0.6951 | 0.3793 / 0.6951 | 0.4232 / 0.7073 | 0.4713 / 0.7622 |
| MBPP+ pass@1 / pass@10 | 0.4661 / 0.7143 | 0.4524 / 0.6878 | 0.4574 / 0.6931 | 0.4802 / 0.7090 |
| MBPP base pass@1 / pass@10 | 0.5323 / 0.7963 | 0.5209 / 0.7778 | 0.5291 / 0.7937 | 0.5463 / 0.7884 |
| LCB latest-175 think@8192 | 0.0800 (14/175) | 0.0971 (17/175) | 0.0857 (ctx 4096) | 0.0857 |
| LCB latest-175 no-think@4096 | 0.1086 (19/175) | 0.1257 (22/175) | 0.1257 | 0.1200 |

66,048 的 LCB think 细节：123/175 撞 8192 上限（no_code）、38 failed、52 closed 并给出代码、
14 passed；难度 easy 12/43、medium 2/52、hard 0/80。no-think：24 no_code、131 failed、
1 timeout、19 passed；easy 17/43、medium 2/52、hard 0/80。

## 目录

| 路径 | 内容 |
| --- | --- |
| `training/` | 训练进程 stdout/stderr（3 个会话）、`status.json`（暂停时的完整状态，含采样顺序）、启动参数 `start.process.json` / `detach_spec.json`、`lr_schedule.json` |
| `eval/` | 自动评测编排日志 `auto_eval.log` / `console.log` / `state.json` / `resume_watch.log`、`REPORT.md`、step 50,688 与 66,048 的 EvalPlus / LCB metrics、inference 与逐项 console log |
| `data_build/` | 数据集构建日志：`raw`（原始抽取）、`write`（分 lane 写出）、`finalize`、`pack`/`pack2`/`pack3`（JSONL→parquet→16K packed）、`backup_35840` |
| `archive/` | 每 2 小时归档任务的日志与 `index.json`（所有历史 checkpoint 归档记录与 SHA-256） |

未纳入（体积原因）：`*.npz` 权重/resume、`*.pth` 转换权重、`*.samples.jsonl`、
`responses.jsonl`、`latest.generations.json`、`write.report.json`（4.4 MB 的逐行配额报告）。
这些仍保留在 `out/`、`checkpoints/`、`eval/` 与 `tmp/`（本地，未跟踪）。
