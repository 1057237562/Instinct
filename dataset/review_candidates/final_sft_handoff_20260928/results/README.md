# 审阅结果

把各批次的 verdict JSONL 放在此目录，文件名应包含任务分类和批次号，避免不同审阅者覆盖彼此结果。例如 `P0_anchor_pairs_batch_001.verdicts.jsonl`。

每行必须包含输入里的 `review_id` 和 `record_sha256`。格式及判定要求见上级 `README.md`。不要生成占位判决；未读完、上下文不足或无法核实的记录用 `uncertain` 并说明原因。
