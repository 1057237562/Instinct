# 逐条验收的基础推理继续 SFT 课程

这是已全量校验的**基础能力补充课程**。用户进一步明确当前首要问题是检查点不能正确理解要求，因此不再把本课程作为解决该问题的主训练方案；现有启动器仍指向本课程，不能将它当作已经完成的指令理解专项训练入口。专项数据设计见 `instruction_understanding_plan.md`。根据用户“不再抽样”的要求，本集不包含未独立核验的外部题解或旧对话回放。此前收集的 Orca-Math、KodCode、GSM8K 及 v2 混合仍保留在原目录作为候选。

## 文件

| 文件 | 行数 | 渲染 tokens |
| --- | ---: | ---: |
| train.jsonl | 31736 | 2836944 |
| validation.jsonl | 1760 | 157879 |
| pilot_train.jsonl | 11302 | 999977 |

模板为当前项目的 OpenAI conversations 格式。最长单条169 tokens，采用512上下文可为训练器随机system前缀保留空间。pilot 是完整语义输入组构成的约100万token子集，不能将其当成独立验证集。

## 内容和验收依据

覆盖15类有限规则：库存增减、实体数量链、折扣和额外费用、小时分钟换算、面积周长、累计余额、滚动最大值、严格条件筛选、两种去重契约、中位数、绝对值个位乘积、词频、括号深度、多键排序、指定JSON字段提取。每组有中英文版本；两种去重契约共享输入并保持同一划分。

不是让外部教师模型随意生成答案：每条问题由明确的参数域和规则产生，构建器独立计算参考结果，并逐条验证显示的算式、终答和格式。空输入、零、负数、重复、严格大于、偶数个元素的中位数等边界在各自合法的任务域内出现。所有训练和验证记录都检查，而不是随机抽查。

三个 GPT-5.6 Luna 子代理分别完整审查数学生成分支、逻辑/格式分支和全量输出。审计文件为 `luna_math_rules_audit.json`、`luna_logic_rules_audit.json`、`luna_full_validation.json`（以该文件最终结果为准）。构建器曾缺少 gzip import，已修复并重新完整生成，不是数据中的标签修正。

`verification_records.jsonl.gz` 将每行对应的任务类型、参数、语言、输入哈希和token数单独保存，方便重算与追踪；这些校验信息不加入训练对话。

**范围限制**：这是有限基础课程，规则族相同而参数输入分离，验证不能代表未见模板或通用理解。全量参考检查也不能保证训练后效果。没有旧对话回放，尤其需要先做pilot并检查普通对话/代码能力是否退化，勿连续多轮反复灌同一模板。

## 训练设置

当前起点：`out/instinct-v1-0914.pth`；20层、hidden_size768，完整配置为 `checkpoints/full_sft_20260914_200306_768.json`。已在meta设备比对当前权重与模型定义，missing/extra/shape mismatch均为0。

- 全参数继续SFT，新建optimizer，`from_resume=0`；不是从随机权重开始，也不是恢复上一阶段旧数据的optimizer位置。
- AdamW，初始学习率1e-5，沿用现有余弦衰减至约0.1倍；没有虚构的warmup参数。
- 先1轮；batch_size4、accumulation_steps4，有效每次更新最多8192个packed token（4×4×512），实际非padding量另计。
- dtype=bfloat16，param_dtype=fp32；use_grad_checkpoint=1。
- 固定长度sequence_packing=1、max_seq_len=512；num_workers=0、packing_num_proc=1。
- use_compile=0、fp8_training=off，先确认效果再单独优化吞吐。
- grad_clip=1；每100个micro batch保存、每10个打印日志。周期保存会更新同名阶段权重，若需比较中间版本应另存副本。

仓库根目录运行（尚未代为启动训练）：

```powershell
# 只验证参数与路径，不训练
python scripts/run_reasoning_sft.py --pilot --dry_run

# 建议先做约100万tokens的小试
python scripts/run_reasoning_sft.py --pilot

# 小试验证改善后，从同一初始权重做完整一轮
python scripts/run_reasoning_sft.py

# 若出现遗忘或不稳定，从原始权重以更小学习率重试
python scripts/run_reasoning_sft.py --pilot --learning_rate 5e-6
```

启动器自动读取模型配置并显式传入20层，避免公共训练CLI默认8层覆盖配置的问题。每次使用带时间戳的新输出前缀，并设置PYTHONUTF8=1；原始权重不覆盖。

现有 trainer 不会自动使用本 validation.jsonl 计算功能正确率或早停。训练前后应在固定推理设置下检查：严格格式率、对照指令双正确率、数字答案、重复率以及日常对话保留能力。HumanEval 的提取与辅助函数问题应先修复，GSM8K 要区分格式失败和数值错误。

重建：`python scripts/data_builder/build_strict_reasoning_sft.py`。文件SHA-256及生成统计见 `report.json`。
