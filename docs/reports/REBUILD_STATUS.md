# Instinct seg2 语料重建 — 状态与续跑指引

快照时间：2026-10-06 01:10（关机前）。仓库：`D:\Workspace\Instinct`。磁盘 55GB 可用。

## 目标（用户确认的口径）

组装 `pretrain_coder_12b_seg2`：与已训练的第一代 12B 语料**分片级不重叠**
（配方相同：60% 代码 / 10% 数学 / 30% 自然语言，12B token，DeepSeek-Coder-V2 风格）；
BASE/continue（约 21%）复用第一代作为 replay；
**段 1 的 12B 组装不需要执行**（第一代已训练过，段 1 采集只用于确定分片边界）；
最终产物转 Parquet（约 34GB → 12GB）后删除 JSONL。

## 当前完成度

### 段 2 已完成/进行中（`dataset/coder_pretrain_12b_seg2/`）
- ✅ **math：采集完成**（5.7GB 分片，~1.5B token 超额垫备）
- ✅ **natural_zh：采集完成**（7.0GB 分片，~1.53B token）
- ⏳ code / natural_en：等待段 1 对应组件达标后由接力器自动启动

### 段 1 边界推导（`dataset/coder_pretrain_12b/`，reports 是跳过依据）
- code：~5.0B / 6.1B（82%，双 worker，shards 数据已按策略清理）
- natural_en：~1.19B / 2.1B（57%，双 worker）
- math / zh：✅ 达标（其分片数据与 raw 已清理，报告保留）

### 基础语料链（replay 部分）
- ✅ `pretrain_stackv2_code_480m_tokens_4096.jsonl`：469,914,737 token（与配方精确一致）
- ✅ `pretrain_verified_submissions_115m.jsonl`：114,999,931 token（cap=8 重建）
- ✅ `pretrain_arxiv_abstracts_160m_tokens_4096.jsonl`、`pretrain_code_prompts_extra.jsonl`（143,685 行）
- ✅ `pretrain_math_reasoning_500mb.jsonl`（375MB）、`pretrain_general_bilingual_substitute_330m.jsonl`（329,993,029 token）
- ✅ `competitive-coding/data/{text_to_sql,exercism,competitive_programming_python_00,cpp_00}.jsonl`（py_00 6GB / cpp_00 5.3GB 已合并）
- ⏳ py_01 + cpp_01 抽取：range parts/state 已清理，明天重跑（aifasthub 下每个分片约 30 分钟）
- ✅ `dataset/gsm8k/test.jsonl`、`dataset/math-instruct/MathInstruct.json`（262,039 行）

## 恢复命令（Git Bash，仓库根目录，按顺序）

```bash
export HF_ENDPOINT=https://aifasthub.com
export HF_FALLBACK_ENDPOINT=https://hf-mirror.com
export PYTHONIOENCODING=utf-8

# 1) 段 1 剩余边界（code 双 worker + en 双 worker；可并行）
python scripts/data_builder/collect_coder_pretrain_12b.py --component code --segment 1 --shard-workers 2 --worker-index 0 --retries 30 &
python scripts/data_builder/collect_coder_pretrain_12b.py --component code --segment 1 --shard-workers 2 --worker-index 1 --retries 30 &
python scripts/data_builder/collect_coder_pretrain_12b.py --component natural_en --segment 1 --shard-workers 2 --worker-index 0 --retries 30 &
python scripts/data_builder/collect_coder_pretrain_12b.py --component natural_en --segment 1 --shard-workers 2 --worker-index 1 --retries 30 &

# 2) 竞争编程抽取最后两个分片（aifasthub 下约 30 分钟/个）
python scripts/data_builder/extract_competitive.py --language python &
python scripts/data_builder/extract_competitive.py --language cpp &

# 3) 接力器（并行模式）：段 1 code/en 达标后自动启动段 2 同组件采集
python scripts/data_builder/relay_seg2.py

# 4) 抽取完成后：codespecialist 组装 → 改名 → PRIOR → continue 链
python scripts/data_builder/build_codespecialist_4096.py
#   --current 指向 dataset/pretrain_verified_submissions_115m.jsonl
#   --math dataset/pretrain_math_reasoning_500mb.jsonl
#   --general dataset/pretrain_general_bilingual_substitute_330m.jsonl
#   → codespecialist_4096.next.jsonl，校验后改名 codespecialist.jsonl/.report.json
cp dataset/codespecialist.jsonl dataset/pretrain_codespecialist.jsonl   # PRIOR
python scripts/data_builder/collect_continue_pretrain_1b.py
python scripts/data_builder/dedup_continue_vs_codespecialist.py        # .dedup.jsonl → 改名回 pretrain_continue.jsonl

# 5) 段 2 组装（--delete-consumed-inputs 控制磁盘峰值）
python scripts/data_builder/build_coder_pretrain_12b.py --segment 2 --delete-consumed-inputs
#   → dataset/pretrain_coder_12b_seg2.jsonl（约 34GB）

# 6) Parquet（zstd，约 12GB，含抽样校验）后删除 JSONL
python scripts/data_builder/jsonl_to_parquet.py \
  --input dataset/pretrain_coder_12b_seg2.jsonl \
  --output dataset/pretrain_coder_12b_seg2.parquet
```

## 注意事项

- **镜像**：hf-mirror 已对本机 IP 限速（单流 <2KB/s）；**主源已切换 aifasthub.com**
  （单流 0.5-1.1MB/s、8 路并行 13MB/s+，全部所需仓库+revision 已验证），hf-mirror 为自动回退。
  `collect_continue_pretrain_1b.py` 用 hf_hub_download，同样吃 `HF_ENDPOINT`。
- **段边界**：段 N 跳过段 1..N-1 报告中的上游分片；护栏要求前序段达标。
- **接力器为并行模式**：段 2 各组件同时跑，磁盘由清理循环（每 10 分钟）+ 段 1 数据释放平衡。
- **build --delete-consumed-inputs**：collected 分片边消费边删（BASE/CONTINUED 保留），把组装峰值从 ~140GB 压到 ~70GB。
- **替代项（最终报告须注明）**：t2t → fineweb/中文 fineweb 切片；verified_competitive_submissions → open-r1 重建（cap=8）；competitive → 流式抽取（上游 24GB/分片放不进磁盘）；PRIOR → 复用重建的 codespecialist。
- 磁盘 55GB 可用；明天恢复后清理循环会随段 1 完成释放 ~30GB（en/code 的 raw+shards）。
