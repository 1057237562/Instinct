# Instinct Datasets

将所有下载的数据集文件放置到当前目录.

Place the downloaded dataset file in the current directory.

## 数据集自动分类

Config WebUI 会扫描本目录顶层的 `.jsonl` / `.jsonl.gz` / `.parquet` 文件，
并按文件名前缀分类：

- `pretrain*.jsonl`：预训练数据，格式为 `{"text": "..."}`。
- `sft*.jsonl`：SFT 数据，格式为 `{"conversations": [...]}`。
- `lora*.jsonl`、`dpo*.jsonl`、`rlaif*.jsonl`、`agent*.jsonl`：对应训练器数据。

将新的指令或 Coding 数据命名为 `sft_<name>.jsonl` 后，它会自动出现在
`full_sft` 的 **Training dataset** 选项中。同一份数据的 `.jsonl` 与
`.parquet` 会同时出现在列表里，选择哪一个都可以；`.report.json` 报告文件
不会被识别为数据集。

## JSONL 编译为 Parquet

`dataset_compiler/` 是一个 Rust 工具，把训练语料编译成 Parquet。编译后的
文件是 `.jsonl` 的直接替代品：训练器、流式预训练、Config WebUI 都直接读取，
无需改动配置。

```bash
cd dataset_compiler && cargo build --release

# 编译（默认整文件预扫描，schema 覆盖每一行）
./target/release/dataset_compiler compile \
  -i ../dataset/pretrain_coder_12b.jsonl \
  -o ../dataset/pretrain_coder_12b.parquet

# 校验行数与取值完全一致
./target/release/dataset_compiler verify \
  ../dataset/pretrain_coder_12b.jsonl ../dataset/pretrain_coder_12b.parquet --rows 2000
```

收益与代价：

- 体积约为 JSONL 的三分之一（zstd），加载时不再需要 Python 逐行解析 JSON。
- 语料自带的 `token_count` 会被保留，因此有界流式（`--dataset_streaming`）
  可以按 token 规划分块而不做任何分词；分块边界对齐 row group，物化一块
  只需重编码这些 row group。
- 代价是一次性的编译步骤：JSONL 更新后需要重新编译（`verify` 用于确认
  编译结果没有丢行、没有改值）。
- 编译输出旁边会生成 `<output>.report.json`，记录源文件摘要、行数、列类型、
  压缩方式、吞吐与告警，便于审计。

Schema 约定（由 `scripts/data_loader/source_format.py` 与各 Dataset 类读取）：
- 预训练：`text` 为字符串，其余标量键（`token_count`、`license`、`source` 等）
  原样保留。
- 对话：`conversations` / `chosen` / `rejected` 为 `list<struct<...>>`，字段
  一律为字符串，字段顺序为 `role`、`content`、`reasoning_content`、`tools`、
  `tool_calls`，之后是语料里出现的其它键（额外字段会被保留，比 JSONL 路径更宽松）。
- 消息之外的嵌套对象（如 `metadata`）存为 JSON 文本；同一键在不同行出现
  互不兼容的类型时也退化为 JSON 文本，并在报告中列为告警，不会丢行。

### 与断点续训的关系

同一语料的 `.jsonl` 与 `.parquet` 被视作同一个数据集：编译后切换到 parquet
继续训练不会被续训校验拒绝（日志会打印 `[Resume] the corpus changed container`）。
分块边界是否一致取决于编译方式：

- **带 `--align-chunk-bytes` 编译**（数值取训练用的 `--streaming_chunk_mb`，例如
  `--align-chunk-bytes 1GiB`）：编译器在 schema 扫描时复现 JSONL 计划的分块规则，
  在每个分块边界处关闭一个 row group，于是「一个 row group = 一个流式分块」。
  两份计划覆盖完全相同的行区间，**游标精确对齐**，日志会打印
  `its chunks are aligned to the JSONL chunk boundaries, so the saved cursor
  points at the same rows`。这是推荐做法。
- **不带该参数**：chunk 边界会被重建，保存的 chunk 游标落在同一相对位置附近，
  最多有一个 chunk 的重放或跳过；日志会给出
  `[Dataset Streaming] resuming on a recompiled corpus: ... row X of Y` 提示。
  偏差随块号增大（每块约 1.2 万行），在 epoch 边界恢复则完全没有偏差。
- **非流式（整库 Arrow）路径**下，两种容器产生完全相同的 packed 数据，续训位置
  一直精确一致。
- 换成**别的**语料（文件名或目录不同）仍会被拒绝，请新建一个训练阶段。

详细参数与实测数据见 `dataset_compiler/README.md`。

## 收集开放许可的 arXiv 论文全文

从 `common-pile/arxiv_papers` 流式收集约 500 MB 的 arXiv 论文。该来源只包含
CC BY、CC BY-SA、CC0 / Public Domain 论文。构建器按全部时间分片均匀取样，
并严格保持“一篇论文一条记录”：不切块、不截断、不改写正文，同时保留 arXiv
ID、URL、作者和许可证元数据。

```bash
python scripts/data_builder/collect_arxiv_pretrain.py

# 例如扩充为约 2 GB
python scripts/data_builder/collect_arxiv_pretrain.py \
  --target-bytes 2000000000 \
  --output dataset/pretrain_arxiv_open_2gb.jsonl \
  --report dataset/pretrain_arxiv_open_2gb.report.json
```

默认输出 `dataset/pretrain_arxiv_open_500mb.jsonl`，可直接被 Config WebUI 识别为
预训练数据。旁边的 `.report.json` 记录来源分片、年份、许可证、SHA-256 和实际
字节数。数据文件保留完整论文，但训练时模型实际看到的长度仍由训练器的
`max_seq_len` 决定。

## 代码专家预训练混合

`dataset/codespecialist.jsonl` 使用按 Instinct tokenizer token 数控制的代码专家配方：约 60%
代码相关数据、20% 中英通用文本、10% 数学推理和 10% 学术摘要。代码部分
包含真实开放仓库代码与 Markdown 文档、竞赛题面/推理/答案、验证通过的 submission、
代码指令、Text-to-SQL 和 Exercism 软件任务。所有来源保持整条记录，不在合成阶段
切块或截断。含 BOS/EOS 超过 4096 tokens 的记录整条丢弃。语料约 16 亿 tokens，
训练两遍约 32 亿 token 呈现量；重复训练不等价于相同数量的新语料。
训练需设置 `max_seq_len=4096` 才能完整利用合格记录；更小长度仍会截断。

真实仓库组件可复现为：

```bash
python scripts/data_builder/collect_stackv2_code_4096.py
python scripts/data_builder/collect_arxiv_abstracts_4096.py
```

完整混合先构建为 `codespecialist_4096.next.jsonl`，校验后再替换正式文件：

```bash
python scripts/data_builder/build_codespecialist_4096.py
```

实际组成、许可证来源、行数、正文比例和 SHA-256 见
`dataset/codespecialist.report.json`。通过完整性验证后，比例不足的草稿、替换前旧版
和旧的 `pretrain_codespecialist` 派生混合均已清理；原始组件仍保留以支持重建。

## 12B Instinct Coder 预训练语料

面向 V1 MoE 编程模型的最终目标采用 DeepSeek-Coder-V2 的 token 配比：60% 源代码、
10% 数学、30% 自然语言，总计 12B 唯一 tokens。现有 `codespecialist.jsonl` 和
`pretrain_continue.jsonl` 中通过审计的新数据优先复用；不足部分由下列固定 revision
的开放数据补齐：

- 代码：`common-pile/stackv2_edu_filtered`（逐文件开放许可证）；
- 数学：`HuggingFaceTB/finemath` 的 `finemath-4plus`（ODC-By）；
- 英文：`HuggingFaceTB/smollm-corpus` 的 `fineweb-edu-dedup`（ODC-By）；
- 中文：`opencsg/chinese-fineweb-edu`（Apache-2.0 声明，分发前仍需复核上游条款）。

收集器按上游分片原子保存，可以安全中断后重跑：

```bash
python scripts/data_builder/collect_coder_pretrain_12b.py --component all
```

四个补充组件收齐后生成最终全局打乱语料：

```bash
python scripts/data_builder/build_coder_pretrain_12b.py
```

最终构建会按正文 SHA-256 全局精确去重，并用 13-word n-gram 筛除与 HumanEval、
sanitized MBPP test 和 GSM8K test 重叠的记录。所有样本均由项目 tokenizer 验证为
不超过 4096 tokens，不切分、不截断。完整来源、许可证计数、拒绝原因、token 配比
和输出 SHA-256 写入 `dataset/pretrain_coder_12b.report.json`。

当前成品 `dataset/pretrain_coder_12b.jsonl` 包含 13,267,273 条完整记录和
11,999,990,666 tokens：代码 7,199,997,236、数学 1,199,996,324、自然语言
3,599,997,106。文件 SHA-256 为
`2b2c76ff86e65c9849ad743690e1807cfc8421796e3459839c780fafbb60bc5b`。
各类别距名义配额的差值均小于单条 4096-token 上限，这是坚持不切断样本的结果。

## 推荐 SFT 数据准备

在仓库根目录运行：

```bash
# 使用已下载到 dataset/codealpaca/ 的 CodeAlpaca 20K
python scripts/data_builder/prepare_sft_data.py codealpaca-local

# 适合小于 1B 模型的通用指令数据（可先抽样 100K）
python scripts/data_builder/prepare_sft_data.py smol-smoltalk --max-samples 100000

# 执行过滤的 Coding 指令数据
python scripts/data_builder/prepare_sft_data.py bigcode-exec-50k

# 可选 Coding 补充
python scripts/data_builder/prepare_sft_data.py magicoder-75k
```

转换结果统一为 Instinct 所需的 `conversations` 格式，并使用 `sft_` 文件名前缀。

### 混合 Magicoder 110K、MathInstruct 与原始 T2T replay

```bash
python scripts/data_builder/mix_sft_datasets.py
```

默认读取：

- `dataset/magicoder-110k/data-evol_instruct-decontaminated.jsonl`
- `dataset/math-instruct/MathInstruct.json`
- `dataset/sft_t2t.jsonl`

默认输出 `dataset/sft_magicoder110k_mathinstruct_t2t_replay20.jsonl`。其中原始
T2T replay 占最终数据约 20%，使用固定随机种子均匀抽样；最终数据使用磁盘分块
随机混洗，不会把 14 GB 原始 T2T 或完整输出一次性载入内存。可用
`--replay-fraction`、`--seed`、`--chunk-rows` 和 `--output` 调整。

## 在新数据上继续 SFT

在 Config WebUI 选择 `full_sft`，然后将 **SFT start mode** 设为
**Continue a completed SFT on new data**。该模式加载最新或指定的已完成
`full_sft` 普通权重，并为新阶段重新创建 optimizer、scheduler 和训练 step；
它与恢复中断训练的 **Resume an interrupted run** 不同。

Auto sequence buckets 默认将单条上下文限制为 16,384 tokens。SFT 样本超过
上限时会整条丢弃（日志显示 `[Length Filter]` 统计），不会截断出残缺的代码或答案。
