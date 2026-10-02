# GSM8K 评测

WebUI 的评测类型选择 **GSM8K**，选择 `out` 权重、K、batch size 后运行。支持仅生成、仅评分、生成并评分。模型配置、编译和批次调度复用现有推理逻辑；结果保存在 `eval/`。

默认从 [OpenAI GSM8K 官方仓库](https://github.com/openai/grade-school-math) 下载 **test** 集的 1319 题，校验后缓存到 `dataset/gsm8k/test.jsonl`，后续无需联网。也可通过 `--problem_file` 指定含 `question`、`answer` 字段的 JSONL/JSONL.GZ；参考答案必须包含 `#### 数值`。训练集不参与默认评测。

```bash
# 原生权重：批量生成并评分
python eval_gsm8k.py --checkpoint_path out/full_sft_768.pth --config_path checkpoints/full_sft_768.json --mode all --batch_size 8

# 只对现有答案评分；不加载模型
python eval_gsm8k.py --mode evaluate --output eval/gsm8k_samples.jsonl --k 1

# pass@1 / pass@5 / pass@10
python eval_gsm8k.py --load_from ./instinct-3 --mode all --num_samples 10 --temperature 0.8 --batch_size 16 --k 1 5 10 --output eval/gsm8k_10.jsonl
```

提示协议是 **zero-shot step-by-step**，不提供示例或参考解答，要求最后一行为 `#### <number>`。评分只比较最终数值，支持千分位逗号、负数、小数；用 Decimal 精确比较，不执行表达式或生成代码。不从推理过程随便取最后一个数字，也不把缺少格式的答案当作正确答案。`\boxed{...}`、单位后缀等不符合该严格格式，会记为 `AnswerFormatError`。数值不同记为 `WrongAnswer`。

`accuracy` 是逐题样本通过比例的均值；每题一条贪心答案时即常规准确率，多样本时等于 pass@1 估计。pass@K 使用无偏公式，每题样本不足的 K 单独列出。此零样本、严格抽取协议应与相同协议的结果比较，不直接等同于 8-shot 或宽松抽取的报告。

生成后保存 `.gz` 压缩副本；评分后输出 `_results.jsonl`、`_metrics.json`、精简的 `_analysis.md/json` 和完整证据 `_analysis_full.jsonl.gz`。WebUI 可直接查看和下载；已有评分结果也支持“计算 pass@K”标签。`--resume` 保留已有样本并补足数量，需保持模型和生成配置一致；记录的题目哈希用于检测本地数据错配。
