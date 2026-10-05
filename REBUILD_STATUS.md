# Instinct seg2 语料重建 — 状态与续跑指引

快照时间：2026-10-05 03:40（关机前）。数据库目录：`D:\Workspace\Instinct`。

## 目标

组装 `pretrain_coder_12b_seg2`：与已训练的第一代 12B 语料**分片级不重叠**，
配方相同（60% 代码 / 10% 数学 / 30% 自然语言，12B token，DeepSeek-Coder-V2 风格）；
BASE/continue（约 21%）复用第一代作为 replay（用户确认）；
最终产物转 Parquet 保存并删除 JSONL（用户要求，约 34GB → 12GB）。

## 当前进度

### 段 1 边界推导（`dataset/coder_pretrain_12b/`，只用于确定"第一代用过的分片"）
| 组件 | 已采集 token / 目标 | 备注 |
|------|--------------------|------|
| code | 1,863,965,643 / 6,100,000,000 | 9 个分片报告完成 |
| math | 717,551,681 / 1,100,000,000 | |
| natural_en | 225,808,038 / 2,100,000,000 | 分片大（2.3GB）易被镜像断流，续跑请带 `--retries 30` |
| natural_zh | 739,927,988 / 1,100,000,000 | |

跑完即得到跳过清单（各组件 `reports/*.json` 的 `upstream_file`）。
**段 1 的 12B 组装不需要执行（已训练过，用户确认跳过）。**

### 基础语料链（段 2 的 replay 部分）
已完成：
- `dataset/pretrain_verified_submissions_115m.jsonl` — 75,444 行 / 114,999,931 token（cap=8）
- `dataset/pretrain_arxiv_abstracts_160m_tokens_4096.jsonl`（160M token）
- `dataset/pretrain_code_prompts_extra.jsonl`（143,685 行）
- `dataset/pretrain_math_reasoning_500mb.jsonl`（375MB）
- `dataset/pretrain_general_bilingual_substitute_330m.jsonl`（329,993,029 token，替代 t2t）
- `dataset/competitive-coding/data/{text_to_sql,exercism}.jsonl`
- `dataset/gsm8k/test.jsonl`、`dataset/math-instruct/MathInstruct.json`（262,039 行）
- codeforces 原始数据 36/36 submissions + 12/12 题面

未完成：
- 竞争编程 python/cpp 流式抽取（断点已保存于 `dataset/competitive-coding/data/*.r*.part` 与 `*.state`）
- stackv2 480M 的 raw 预取（`dataset/coder_pretrain_12b/code/raw/`，约 9/26 完成）
- `collect_stackv2_code_4096.py` 尚未运行（等预取）

## 恢复命令（Git Bash，仓库根目录）

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONIOENCODING=utf-8

# 1) 段 1 四组件边界采集（并行跑，可任意中断续跑）
python scripts/data_builder/collect_coder_pretrain_12b.py --component code --segment 1 &
python scripts/data_builder/collect_coder_pretrain_12b.py --component math --segment 1 &
python scripts/data_builder/collect_coder_pretrain_12b.py --component natural_en --segment 1 --retries 30 &
python scripts/data_builder/collect_coder_pretrain_12b.py --component natural_zh --segment 1 --retries 30 &

# 2) 竞争编程抽取（range 级断点续传）与 stackv2 预取
python scripts/data_builder/extract_competitive.py --language python &
python scripts/data_builder/extract_competitive.py --language cpp &
python scripts/data_builder/prefetch_stackv2.py &

# 3) 段 2 接力器：段 1 各组件达标后自动启动段 2 同组件采集
python scripts/data_builder/relay_seg2.py

# 4) 段 1 全部达标 + 预取完成后，构建基础语料：
python scripts/data_builder/collect_stackv2_code_4096.py        # 480M，单次不可续跑
python scripts/data_builder/build_codespecialist_4096.py        # 需 competitive 抽取完成
#   → dataset/codespecialist_4096.next.jsonl，校验后重命名为 codespecialist.jsonl
#   → 复制为 dataset/pretrain_codespecialist.jsonl（PRIOR：redo 基准）
python scripts/data_builder/collect_continue_pretrain_1b.py     # → pretrain_continue.jsonl
python scripts/data_builder/dedup_continue_vs_codespecialist.py # → .dedup.jsonl → 改名回 pretrain_continue.jsonl

# 5) 段 2 组装（四组件 + 基础语料就绪后）
python scripts/data_builder/build_coder_pretrain_12b.py --segment 2
#   → dataset/pretrain_coder_12b_seg2.jsonl（约 34GB）

# 6) 转 Parquet（zstd，约 12GB，含抽样校验）后删除 JSONL
python scripts/data_builder/jsonl_to_parquet.py \
  --input dataset/pretrain_coder_12b_seg2.jsonl \
  --output dataset/pretrain_coder_12b_seg2.parquet
# 校验通过后：rm dataset/pretrain_coder_12b_seg2.jsonl
```

## 关键说明

- 所有采集器/抽取器/预取器均支持断点续跑：重新执行同一命令即可。
- 段边界机制：段 N 采集跳过段 1..N-1 报告中的上游分片；段 N 要求段 N-1 已达标（内置护栏会拒绝提前启动）。
- 记录在案的替代项（最终报告需注明）：
  1. `general_bilingual`：原 `pretrain_t2t.jsonl` 在本机不存在且无公开镜像 → 用 fineweb-edu-dedup + chinese-fineweb-edu 切片替代（330M token）。
  2. `verified_competitive_submissions`：原组件只能从旧 mix 复制（不可复现）→ 由 open-r1/codeforces + codeforces-submissions 重建（cap=8/题/语言）。
  3. `competitive_programming_*`：上游单分片 24GB 放不进本地磁盘 → 流式抽取构建器会消费的行（cap=4/题，带 6-8M token 余量）。
  4. `pretrain_codespecialist.jsonl`（PRIOR）：无构建器 → 复用重建后的 codespecialist.jsonl（replay 语义更贴合）。
- 网络：03:00-03:40 观察到本机网络整体退化（清华源 11KB/s、百度 1.3-4.6s、hf-mirror 单流 <1KB/s），非镜像问题；恢复后重跑即可。
- 磁盘快照：77GB 可用（D: 400GB）。段 1 组件达标后接力器会自动清理其分片数据（报告保留）。

## 本次改动过的脚本（均为最小改动、默认行为不变）

- `collect_arxiv_pretrain.py`：`shard_url` 支持 `HF_ENDPOINT`；`CountingReader` 补上下文管理器。
- `collect_coder_pretrain_12b.py`：`--segment/--out-dir/--skip-reports-from`（段选择与跳过链）、`direct_dataset_download` 支持 `HF_ENDPOINT`。
- `collect_stackv2_code_4096.py`：优先读取本地 raw（`INSTINCT_STACKV2_RAW_DIR`，默认 12B 采集器的 raw 目录）。
- `build_coder_pretrain_12b.py`：`--segment/--collected-dir/--output/--report`。
- 新增：`fetch_parallel.py`（并行 range 下载）、`orchestrate_fetch.py`、`prefetch_stackv2.py`、`prefetch_seg1_next.py`、`extract_competitive.py`、`relay_seg2.py`、`build_verified_submissions.py`、`build_general_bilingual_substitute.py`、`fetch_raw_files.py`、`jsonl_to_parquet.py`。
