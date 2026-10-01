# 最终 SFT 审阅交接包

此目录用于审阅，不是训练集。模型架构仍为 InstinctV1Moe；外显身份按本次用户提供的审定文案使用 **Instinct**：

> 我是 Instinct，一个从头训练的语言模型，由 L1bra 个人独立开发和训练，不隶属于任何商业组织。

不要自动把 Instinct 改成 InstinctV1Moe，也不要把第三方公司/模型名称全局替换掉。若发现项目旧说明与这份最新身份口径不同，在结果中注明即可。

## 优先级与分工

| 顺序 | 子目录 | 工作内容 |
|---|---|---|
| P0 | `P0_t2t_residual/` | 129条现有审计残留：17条标为疑似污染、112条待复核；逐条检查正文、reasoning 和前文。审计标签不是最终判决。 |
| P0 | `P0_anchor_pairs/` | 4,500条锚点归并成225组唯一问答；重点纠正答非所问、否定对象不匹配和推理内容错配。 |
| P1 | `P1_other_identity/` | 剩余非T2T候选里的身份命中，区分代码中的虚构聊天机器人、事实介绍和模型真正的自述。 |
| P1 | `P1_python_semantics/` | 按10个原始分片各抽16条，共160条；核对题意、代码、测试三者一致性，不以“测试pass”代替语义判断。 |
| P1 | `P1_multiturn_quality/` | 80条完整多轮：追问指代、要求修改、约束保持、杜撰访问权限/个人经历、无依据具体事实。 |
| P2 | `P2_long_grounding/` | 48条完整长文档问答，每批1条；核对答案是否能由给定材料支持，是否遗漏关键限制。 |
| P2 | `P2_synthetic_wording/` | 16条完整合成长多轮，每类4条；检查任务描述歧义、轮次依赖和是否形成可用的对话训练信号。答案还会由程序全量复算。 |

P0应先完成。P1的结果用于发现系统性问题和决定是否扩大某一来源的审阅；抽样通过不能证明整份数据都正确。P2需要有足够上下文的模型；如果轻量模型放不下单条全文，请标记 `uncertain/context_limit`，交给更长上下文的审阅者，不要截断后判定通过。

可以给不同模型分配不同 `batch_*.jsonl`。每批是完整记录；长记录不切块。具体文件、条数、SHA-256 和待返回 ID 见 `manifest.json`。

## 已确认的检查点

- `train_ready` 的904,909行与审计报告一致，SHA-256为 `8dc5475ce99988709e06cf38a267f9d8f2135bf438d202d9898e5cbf090d85aa`。
- 17/112来自与这个**同一SHA**对应的审计报告，不是旧源文件的过期统计。
- 残留源行616使用“并不基于GPT”否定句，可能被误判为自认GPT；同时它又把GPT-3.5/GPT-4说成“开源模型”，这是另一种事实错误。应分别处理，不能只按污染标签删除。
- 锚点确有错配例子：`Are you conscious?` 配了只介绍名字和作者的回答；`Who are you?` 配了只解释没有意识、不报名字的回答；“训练数据来自哪里？”配了开发者介绍。内容可以本身真实，但不一定回答当前问题。
- 锚点的 `reasoning_variants` 也必须看。例如“不能透露身份”与目标不一致；和当前问题无关的解释应清除。不要编造新的内部思考过程。
- Python样本的通过记录来自上游；这次构建没有批量在本机执行这些外部程序。审阅时应给出具体反例或推导；不要在有凭据的宿主环境直接执行未经审阅的代码。
- 旧 `sft_t2t_mini.jsonl` 及旧混合中的T2T回放不会直接复用；无需重审这些旧版本。

## 每条结果的格式

每位审阅者输出 JSONL，一行对应一个 `review_id`。不要只给总体“通过”结论。将文件放在本目录的 `results/` 下，使用与输入批次对应的唯一文件名；不要改写输入包或原训练源文件。

```json
{"review_id":"原样复制","record_sha256":"原样复制","decision":"keep","reason":"结合题目与答案的具体依据","issue_tags":[],"patches":[],"counterexample":null,"reasoning_action":"not_applicable"}
```

允许的 `decision`：

- `keep`：当前记录符合本类标准；说明关键核对点。
- `repair`：问题局部、能确定修复内容；提供补丁，不能只写建议。
- `drop`：存在明确错误且不值得修复，或当前配方不适合；解释具体原因。
- `uncertain`：上下文、验证条件或把握不足；注明缺什么，不能冒充通过。

局部补丁格式：

```json
{"review_id":"原ID","record_sha256":"原哈希","decision":"repair","reason":"回答没有回应意识问题","issue_tags":["answer_question_mismatch"],"patches":[{"message_index":1,"field":"content","value":"我没有主观意识或个人经历。我是 Instinct，一个由 L1bra 个人从头训练的语言模型。"}],"counterexample":null,"reasoning_action":"clear"}
```

`message_index` 从0开始，指本条 `record.conversations` 中的位置。`field` 只允许 `content` 或 `reasoning_content`。`value` 是完整替换文本；删除 reasoning 可用 `null` 或 `reasoning_action: clear`。不改 user 要求来迁就错误答案。代码题修复应返回完整正确的 assistant content；不能只返回diff片段或自动修改测试以让错误实现通过。

锚点审阅以归并后的问答对为单位。它的 `source_row_numbers` / `source_row_sha256s` 列出全部原始位置；原始记录可能多一个system消息，因此不要直接把包里的消息下标套到所有源行。后续集成器会按已核对的问答内容和角色映射回原始行。检查 `reasoning_variants` 和 `system_variants` 后，给出保留已审推理、全部清空、或明确替换的决定。

建议 issue tags：`wrong_identity`、`unsupported_affiliation`、`false_personal_experience`、`answer_question_mismatch`、`reasoning_conflict`、`contract_violation`、`incorrect_algorithm`、`wrong_test`、`missing_edge_case`、`ungrounded_claim`、`lost_constraint`、`context_limit`、`false_positive`。

## 可直接派发的提示词

> 请先阅读这个交接包的 README.md，再审阅分配给你的 batch JSONL。逐条阅读完整 record 和附带的测试/审计证据，返回带原 review_id、record_sha256 的 verdict JSONL。不要把关键词命中或上游测试pass当最终结论。身份使用本包给定的 Instinct / L1bra 个人项目口径，保留普通第三方事实。repair 必须给完整可应用补丁；把握不足标 uncertain。不要修改输入或原始数据，不要训练模型。结果存入 results/，并报告已审条数和各decision计数。

## 集成边界

回填后会校验所有ID、哈希、重复/缺失结果、补丁字段，再应用到**新文件**并重新跑身份、schema、长度、去重及训练/验证隔离检查。审阅补丁确认后才会应用；抽检通过不代表未抽中的记录也全部经过人工语义审阅。身份锚点按最终样本呈现量0.5%–2%控制，不会因为文件有4,500行就全量重复叠加。

结果可以分批回填。在仓库根目录执行下列只读检查；去掉 `--allow-partial` 后，缺少任何结果也会返回失败：

```powershell
python -X utf8 dataset/scripts/validate_final_sft_review_results.py --allow-partial
```

本交接包已经通过完整性检查：677个唯一ID、116个批次及其原文哈希一致。校验器不会自动应用补丁，也不会把 `uncertain` 当作通过。

最终交付仍是一个混合训练JSONL和独立验证集，不按长度分片。64K是数据长度预算，不保证当前32K模型配置已能训练或有效使用64K；本次也不启动训练。
