# 指令理解专项继续 SFT 数据

这套数据针对当前检查点“会做相似任务，但没有落实当前要求”的问题。训练记录全部由项目内确定性规则生成；公开的 IFEval、FollowBench 和 Conifer 仅用于任务分类设计，没有复制其评测或训练记录。详细来源见 `sources.json`。

## 数据文件

- `train.jsonl`：完整训练集。
- `validation.jsonl`：语义输入与训练集隔离，并使用训练中未出现的顶层措辞模板。
- `pilot_train.jsonl`：从训练集按完整对照组选出的约 100 万 token 试训集。
- `provenance.jsonl.gz`：每一行对应的任务族、规范化参数、答案、哈希、划分和 token 数，不进入训练。
- `full_audit.json`：独立实现对全部落盘记录的重算和代码执行结果。
- `overlap_audit.json`：与旧 SFT 的规范化精确提示重合及 HumanEval 13-token 重合审计。
- `luna_full_semantic_audit.json`：第三方审阅者对全部 49,758 条落盘记录（含 pilot 重复计数）的语义、题面、格式和分组复核。
- `luna_code_full_audit.json`：全部 9,340 条落盘代码记录的 57,908 次独立执行及错误变异测试。
- `luna_noncode_full_audit.json`：全部 40,418 条落盘非代码记录的独立复算和题面参数核对。

覆盖八类能力：同材料不同操作、实体与字段绑定、组合条件、严格输出协议、材料内嵌指令隔离、多轮要求修订、缺失与空结果辨析、Python 规格辨析。每个语义组的中英双语及全部对照操作都留在同一划分；pilot 也不会拆组。

完整重建和复核：

```powershell
python scripts/data_builder/build_instruction_understanding_sft.py
python scripts/data_builder/audit_instruction_understanding_sft.py
python scripts/data_builder/audit_instruction_overlap.py
```

建议先从当前 20 层检查点进行约 100 万 token 的试训：

```powershell
python scripts/run_instruction_sft.py --dry_run
python scripts/run_instruction_sft.py
```

默认参数为全参数继续 SFT、新优化器、1 epoch、学习率 `5e-6`、batch size 4、梯度累积 4、序列长度 768、packing、BF16 计算和 FP32 参数。只有试训在独立指令理解评测上改善、且代码和数学没有明显退化后，才使用 `--full`。启动器不会覆盖原权重，也不会自动开始训练。

启动器在运行前会重新计算所选数据文件的 SHA-256，并要求全量答案审计及重合审计为零错误，避免数据在审计后被修改却仍然开始训练。

所有标签在这些有限规则域内可以验证，但这不等于证明开放式理解能力。数据多数要求精确结构化输出；后续若增加自然回答，只有逐条完成语义审核的记录才能并入默认训练集。
