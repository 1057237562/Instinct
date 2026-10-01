# 全量可复算的短思维链 SFT

本数据包含库存流转、实体数量链、折扣与运费、单位产量、全程平均速度、时间预算、票务收入、地砖数量、比例分配和百分比变化十类多步问题。每个语义输入生成中英文两条记录。

训练集使用两种步骤衔接措辞，验证集使用第三种未见措辞；具体参数也按 semantic hash 隔离。每个回答包含两至三个明确步骤，每一步的算式放在反引号内，最后一行使用 `#### 数字`。

构建器逐条检查显示等式和终答；`audit_verified_cot_sft.py` 使用另一套公式和受限 AST 对全部记录重新复算。公开的 `verified-math-reasoning-3k` 仅用于参考任务分类，没有复制其记录，因为其说明只保证终答正确，不保证推理文本的教学质量。

重建和审计：

```powershell
python scripts/data_builder/build_verified_cot_sft.py
python scripts/data_builder/audit_verified_cot_sft.py
python scripts/data_builder/audit_verified_cot_overlap.py
```

这只能保证有限生成域内的关系、算式和终答一致，不能证明开放式推理泛化。
