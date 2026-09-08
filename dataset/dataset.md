# Instinct Datasets

将所有下载的数据集文件放置到当前目录.

Place the downloaded dataset file in the current directory.

## JSONL 自动分类

Config WebUI 会扫描本目录顶层的 `.jsonl` 文件，并按文件名前缀分类：

- `pretrain*.jsonl`：预训练数据，格式为 `{"text": "..."}`。
- `sft*.jsonl`：SFT 数据，格式为 `{"conversations": [...]}`。
- `lora*.jsonl`、`dpo*.jsonl`、`rlaif*.jsonl`、`agent*.jsonl`：对应训练器数据。

将新的指令或 Coding 数据命名为 `sft_<name>.jsonl` 后，它会自动出现在
`full_sft` 的 **Training dataset** 选项中。

## 推荐 SFT 数据准备

在仓库根目录运行：

```bash
# 使用已下载到 dataset/codealpaca/ 的 CodeAlpaca 20K
python scripts/prepare_sft_data.py codealpaca-local

# 适合小于 1B 模型的通用指令数据（可先抽样 100K）
python scripts/prepare_sft_data.py smol-smoltalk --max-samples 100000

# 执行过滤的 Coding 指令数据
python scripts/prepare_sft_data.py bigcode-exec-50k

# 可选 Coding 补充
python scripts/prepare_sft_data.py magicoder-75k
```

转换结果统一为 Instinct 所需的 `conversations` 格式，并使用 `sft_` 文件名前缀。

### 混合 Magicoder 110K、MathInstruct 与原始 T2T replay

```bash
python scripts/mix_sft_datasets.py
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
