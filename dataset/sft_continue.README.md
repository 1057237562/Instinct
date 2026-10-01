# sft_continue.jsonl

`sft_continue.jsonl` 是当前检查点的继续 SFT 混合文件，由以下部分组成：

- `quality_python_sft/continue_sft_reviewed_train.jsonl`
- `instruction_understanding_sft/train.jsonl`
- `verified_cot_sft/train.jsonl`
- 经结构筛选和逐条人工/代理复核后保留的 `sft_t2t_mini.jsonl` 重放

最终文件共 72,343 条、20,337,038 个项目 chat-template tokens，SHA-256 为 `5b0c038f3f81af5870b67e02ef5781d114bb9b9e2cbb75cbaab894e99c07bde5`。

| 来源 | 行数 | Tokens | 占比 |
| --- | ---: | ---: | ---: |
| continue_sft_reviewed | 14,795 | 6,727,863 | 33.08% |
| instruction_understanding | 41,158 | 10,603,797 | 52.14% |
| verified_cot | 14,154 | 2,051,986 | 10.09% |
| sft_t2t_mini_replay | 2,236 | 953,392 | 4.69% |

构建器会做确定性混排、按规范化 system/user 提示去重、按完整对话去重，并保存 `sft_continue.provenance.jsonl.gz`。旧 SFT 候选共 2,412 条，全部逐条审阅；70 条 reject 和 106 条 caution 均被排除，未用未审记录补齐。

原 `continue_sft_reviewed_train.jsonl` 中有 7 条精确重复，合并时只保留一份；其余记录均进入最终文件。新增思维链另有 1,474 条验证记录，不进入最终训练文件。15,628 条训练/验证 CoT 均经过独立公式复算、每步等式检查、题面参数核对和三位审阅者全量复核，错误为 0；与旧 SFT 无规范化精确题面重合，与 GSM8K test 无 13-token 题面重合。

完整重建与审计：

```powershell
python scripts/data_builder/build_sft_continue.py
python scripts/data_builder/audit_sft_continue.py
```

最终审计结果见 `sft_continue.audit.json`；逐行来源和原始行号见 `sft_continue.provenance.jsonl.gz`。思维链审计见 `verified_cot_sft/full_audit.json`、`luna_math_full_audit.json`、`luna_language_rules_full_audit.json` 和 `luna_generator_full_audit.json`。

最长记录为 2,875 tokens；训练时 `max_seq_len` 至少应为 3,072，避免丢弃来源文件中的长代码记录。混合包含 `continue_sft_reviewed_train` 的 CC-BY-NC-4.0 KodCode 成分，因此当前合并文件只适用于非商业用途。
