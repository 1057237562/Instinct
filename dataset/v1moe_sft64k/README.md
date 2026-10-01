# InstinctV1Moe：多轮、Python、短指令与 64K SFT

目标模型：678.7M 总参数 / 106.2M 激活参数的 InstinctV1Moe。
最终主文件：`dataset/sft_v1moe_balanced_64k_identity_clean.jsonl`。
这是一套可复现的研究训练配方；没有启动训练，也没有声称 HumanEval 达到 60%。

最终训练集 **115,944 条 / 107,051,048 tokens**，独立验证集 **1,166 条**。训练集最长 **65,400 tokens**，有 **26,713 条多轮对话**。
抽样复核后额外隔离了 8,076 条多代码块 Magicoder 候选（训练+验证合计），避免当前评测器只提取末尾测试/示例块；另移出一条错误数学结论和一条冒充电商平台的回答。首版成品保存在 `pre_quality_review/`，仅供追溯，不用于训练。

第二轮复核另移出一条明显混杂乱码的 T2T 景色描写记录，记录于 `quality_pending_review_v2.jsonl`。中间版本保存在 `pre_quality_review_v2/`。

## 数据组成与证据

- `sft_t2t_mini.jsonl`：从完整源中固定种子均匀抽样，保留普通对话结构和工具字段；原身份描述另行隔离。
- `quality_python_sft/continue_sft_reviewed_train.jsonl`：复用之前做过抽样审阅的 KodCode / 基础数学 / T2T 配方，不能把全部记录宣称为人工验证或执行验证。
- Magicoder-Evol-Instruct：选择答案含可解析 Python 函数的记录；静态解析不是运行正确性证明。
- `instruction_understanding_sft/train.jsonl`：短指令、条件过滤、输出协议、信息绑定、多轮约束变更。
- UltraChat 200k：只读取 `train_sft`，选择至少两个用户轮次的完整对话，不采入 test 或 generation split。
- LongAlign-10k：固定 revision `12f17c4baff1001f0d44c4f8feab09ee2ee8c6dc`，用于真实长文档问答。它主要是单轮长文档，不应称为自然长多轮数据。
- 项目合成台账：128 个独立随机台账，包含同一台账三次输入、跨位置检索、修改已有记录、后续求和，六个用户轮次。目标长度覆盖 8K、16K、32K、48K、接近 64K；每条的答案由独立解析原始台账的校验器复算。这是有限的长上下文任务补充，不代表掌握一般自然长对话。
- 身份锚点：12 个审阅过的问答模板，约占样本呈现数 0.8%，明确重复加权。目标名称统一为 InstinctV1Moe，说明从头训练、没有主观意识或个人经历，不编造知识截止日期、自动记忆或工具权限。

实际行数、token 比例和分桶统计以 `build.report.json` / `package.report.json` 为准。配方按来源抽样，**没有强行声称实现某个预设 token 比例**。

| 训练来源 | 条数 | 模板 tokens 占比（约） |
|---|---:|---:|
| T2T | 41,658 | 17.06% |
| UltraChat 多轮 | 19,348 | 34.63% |
| Magicoder Python | 20,729 | 16.16% |
| 旧精选 Python / 数学 / 回放 | 14,410 | 6.74% |
| 短指令 | 17,829 | 5.07% |
| LongAlign 长文档问答 | 850 | 16.25% |
| 可验证长多轮台账 | 126 | 4.01% |
| 身份锚点（12模板重复加权） | 994 | 0.09% |

身份锚点占样本数约0.86%；它们很短，所以 token 占比更低。普通样本的 system 身份提示不是 assistant 监督目标。长文档 token 多数属于用户上下文，因此表内占比也不等于 assistant loss 占比。

## 长度与输出格式

每条是 `{"conversations":[...],"token_count":N}`，按当前本地 tokenizer 与 `_create_chat_prompt` 完整渲染后计数。保留整条对话，不截断正文、不把无关短问答拼成一条长会话。

64K 指 65,536 tokens 的总上下文预算，不是字符数或仅答案长度。成品上限 65,408 tokens，预留 128 tokens；近 64K 合成样本目标为 65,400。推理时还需为新生成答案另留空间。

所有记录加入统一身份 system 提示；既有非身份 system 内容保留。去掉源数据的 `reasoning_content` 字段，记录移除数量。本配方训练可见回答和简短分析，不用于保留教师冗长内部思考。

按用户要求，所有长度混合在同一个主训练文件中，不导出长度分片。`validation_identity_clean.jsonl` 是独立验证集，不参与训练；来源、审阅候选和归档文件也不应作为额外训练输入。

## 身份清洗和数据隔离

先执行项目指定的 `filter_anomaly_candidates.py --profile identity`。12 条命中已完整阅读，具体决策见 `identity_review.json`：包括真实身份污染和正常第三方知识两类，未把关键词命中一概判错。

扩展审计检查用户、助手、system 中的第三方名称、旧 Instinct/L1bra 描述、未证实知识截止日期，以及部分个人经历自述。存疑完整行进入 `identity_pending_review.jsonl`，不自动删除原文。当前保守配方仅选择无需这些争议内容的样本；正常第三方事实仍可经后续审阅重新加入。没有执行全局字符串替换。

T2T 抽样确认存在“Instinct3B 由阿里研发”“L1bra 是 Instinct Group 的云计算子公司”等错误，所以旧身份描述不直接继承。

按完整规范化用户提示去重；按首个用户提示分组划分约 1% 验证集，使共享首轮的不同对话落在同一侧。身份锚点重复仅进入训练。未完成语义聚类去重，因此不声称验证集实现完全的题目家族隔离。

HumanEval 仅作为排除依据：检查显式 benchmark 名称、题面、参考实现和测试的 13 词重合。没有用 HumanEval 题目或答案生成样本。启发式去污染不等于语义无污染证明，本次也未重新扫描此前预训练数据。

## 可复现与审核

在仓库根目录运行；脚本默认拒绝覆盖已完成输出。重建应先把旧成品归档到另一个明确命名的目录。

```powershell
python -X utf8 dataset/scripts/build_v1moe_sft64k.py prepare
python -X utf8 dataset/scripts/filter_anomaly_candidates.py dataset/v1moe_sft64k/candidates.jsonl --profile identity --output dataset/v1moe_sft64k/triage.jsonl
python -X utf8 dataset/scripts/build_v1moe_sft64k.py finalize
python -X utf8 dataset/scripts/refine_v1moe_sft64k.py
python -X utf8 dataset/scripts/build_v1moe_sft64k.py verify
python -X utf8 dataset/scripts/package_v1moe_sft64k.py
```

`sources.json` 固定输入 SHA-256，记录源仓库、已知 revision 和完整扫描计数。已有本地文件缺失的上游 revision 明确标为未知，不能用上游当前 HEAD 冒充下载版本。下载的 LongAlign 原始文件保存在 `dataset/v1moe_sft64k_sources/longalign/`。

`train.provenance.jsonl.gz` 与主文件逐行对应，保存源路径、源行、源仓库/revision、移除 reasoning 标记、token 数和提示哈希。合成数据记录 seed 与台账规模。旧精选源的详细测试和原始版本继续由 `quality_python_sft/reviewed_provenance_and_tests.jsonl.gz` 及其 README 提供。

`verification.report.json` 对成品全量重新分词、检查哈希与长度、核对 train/validation 分组隔离，并独立复算全部合成台账答案。`package.report.json` 另记录长度分布、多轮分布和真实 SFT label 逻辑抽检。

本次验证已通过：117,110 条成品全部重新分词，首轮提示 train/validation 交叉为0，128条台账答案独立复算通过；611条监督标签抽检通过。混合训练文件中有114,323条≤4K、1,123条4–16K、498条16–64K记录。最终身份审计见 `release_v2_identity_audit.jsonl.report.json`，两条存疑命中已在 `release_identity_review.json` 中判定为非身份污染。

`identity_review.json` 记录首轮完整身份审阅；`quality_pending_review.jsonl` 保存最终质量筛选移出的完整记录和理由。原有“reviewed”来源仍会出现错误，抽样不是全量语义正确性证明。

## 训练前需要处理的 64K 配置

现有 `trainer/config_instinct_v1_moe.json` 是 32,768 位置上限，LongRoPE 基于 4,096 原始窗口、factor=8；训练器 `bucket_max_seq_len` 默认 16,384。因此不能用原配置直接宣称完成 64K 训练。

建议先训练短指令、Python 与普通多轮，再逐步混入中长数据。扩展到 64K 需要单独校验位置编码、训练及推理缓存上限、分桶上限和显存。仅把 `max_seq_len` 改成 65,536 或把 factor 改成16，都不等于模型已经具备64K能力。不要在没有显存实测时直接让16GB显卡进行64K全参数训练。

上下文扩展阶段保留短指令和编程回放，避免只训练长文档；本次交付一个混合训练文件，没有改变模型配置或启动训练。

## 许可与效果边界

配方包含 KodCode 的 CC-BY-NC-4.0 成分，因此不能将整套数据当作可商用语料。UltraChat 数据卡标 MIT，Magicoder-Evol 数据卡标 Apache-2.0；原始上游条款继续适用。LongAlign 当前数据卡未明确声明数据许可，发布或商用前需要进一步确认。项目本地产物的历史来源不齐全处已标出。

没有批量执行外部代码，Python 语法通过不等于功能正确。最终 HumanEval、多轮指令遵循、身份稳定性和64K检索效果仍需独立训练后评测；不能保证 HumanEval 60%。
