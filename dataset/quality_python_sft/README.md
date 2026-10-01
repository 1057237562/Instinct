# Instinct 定向继续 SFT 数据

**2026-09-15 人工抽样后更新：优先使用 `continue_sft_reviewed_train.jsonl`（14802 条）和 `continue_sft_reviewed_validation.jsonl`（841 条）。** 初版混合的 40 条分来源审阅发现了语义冲突与低质量回放，BigCode 暂移出默认混合，修订版按 token 约为 70% KodCode、20% 基础数学、10% 清理后回放。完整证据见 [manual_review.md](manual_review.md)，统计以 `manual_review_report.json` 为准。下文的 17707 条、60/20/20 是保留用于对照的初版，不再是首选。

这份数据针对用户确认的训练历史制作：`pretrain_codespecialist.jsonl` 两轮预训练，`sft_t2t_mini.jsonl` 一轮 SFT。目的为继续强化短函数契约、循环状态、基础数学和简洁作答。原始训练文件和当前模型均未修改，也未启动训练。

## 直接使用的文件

| 文件 | 记录数 | 用途 |
| --- | ---: | --- |
| `continue_sft_train.jsonl` | 17707 | 本次建议的小规模续训混合 |
| `continue_sft_validation.jsonl` | 1173 | 与训练问题分组隔离的验证集，仅用于检查，勿作为训练数据 |
| `kodcode_train.jsonl` / `kodcode_validation.jsonl` | 10668 / 602 | 全量精选 KodCode，非商业许可 |
| `bigcode_train.jsonl` / `bigcode_validation.jsonl` | 5789 / 300 | 全量精选 BigCode，可单独使用 |
| `gsm8k_train_train.jsonl` / `gsm8k_train_validation.jsonl` | 6699 / 271 | 仅源自 GSM8K 官方 train，数学补强 |

所有训练文件采用项目已有的 `{"conversations":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}` 格式；回放记录可保留原有多轮、reasoning_content 和工具字段。

混合训练文件按当前 tokenizer 和实际 tools-aware chat template 计，共 **7111674 tokens**，最长一条 **2366 tokens**；验证集 **530395 tokens**，最长 **2040 tokens**。默认按 4096 上限设计并预留增强空间，不对原记录做截断或拆分。实际训练时随机 system/think 增强会轻微改变长度与比例。

## 续训构成

按渲染 token 的实验性起始配比：

- Python 新样本约 60%：KodCode 5295 条、BigCode 2927 条，共 8222 条。
- 基础数学约 20%：GSM8K train 6698 条；包括有意重新强调的已见训练题，其来源标记可查。
- 原 SFT 普通/工具对话回放约 20%：2787 条，用来保留原有对话能力。

该比例未经训练对照证明最优。混合中每条只出现一次；通过控制选入的 token 总量平衡，没有机械复制样本。先用当前已完成 SFT 的权重做独立续训实验并另存新权重；不要从随机初始化开始，也不要把本验证集或 HumanEval/GSM8K test 混入训练。

## 来源与许可

1. [KodCode-V1-SFT-4o](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-4o)，固定 revision `14f8782fb7787c7e31dd4a1372518bc10fedb66e`。下载全部 5 个 train shard，共 262659 条原始训练记录，只选择 Prefill、Algorithm、Data_Structure、Docs 的适合基础函数部分。**CC BY-NC 4.0，仅非商业使用**；因此包含它的默认混合也需遵循非商业限制。数据卡存于 `raw/kodcode_sft_4o/README.md`。
2. [BigCode self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k)，revision `356bb069eee815daa6e23e9a282eeefe1490ad44`，**ODC-BY**。复用 `dataset/bigcode-self-oss-instruct-50k/` 已有文件，SHA-256 与固定版本的上游 LFS OID 一致，未重复下载。来源记录在 `raw/bigcode_source.json`，数据卡在原目录。
3. [OpenAI GSM8K](https://github.com/openai/grade-school-math)，revision `3101c7d5072418e28b9008a6636bde82a006892c`，官方仓库 **MIT**。仅下载官方 train 7473 条；原数据、README、LICENSE 和 SHA-256 在 `raw/gsm8k/`。筛除与本地 test 的长文本重合及不符合本次算术筛选要求的记录。
4. 原 `sft_t2t_mini.jsonl` 回放继承原语料许可；本任务未重新确认其全部上游许可。不要把整个混合当作单一 MIT 或商业可用数据集。

## 质量与隔离检查

- KodCode 使用上游 4o_correctness=true、至少 2 次通过、easy 难度标签；要求配对测试可解析、至少 3 个 assert、测试导入的函数确实在答案中定义。easy 是上游模型通过率标签，不保证对 Instinct 容易。
- 选择完整 Python 函数代码，排除空实现、重复定义、不明确的多代码块、顶层示例执行、动态执行/交互输入、非基础依赖、超长样本与明显未定义全局名。答案改成一个完整代码块，避免继续强化冗长说明和多块输出。
- **没有在本机执行下载来的代码或测试。** 上游声称执行通过，本地只做静态检查；生成式测试本身也可能不完整或与题意不符。
- 从初筛候选中删除了 **32215 条**与实际 `pretrain_codespecialist.jsonl` 完整记录/围栏代码规范化精确匹配的记录；新增代码也检查与旧 SFT 用户问题及助手文本的精确重合。该检查不覆盖所有包装形式或语义改写。
- HumanEval：显式基准名、共享 13 词片段和参考实现 AST 指纹筛查。GSM8K test：共享 13 词片段筛查。没有直接采入两者的测试题，但启发式筛查不等于无污染证明。
- KodCode 按原 question ID 合并 instruct/complete 变体后分训练/验证；GSM8K 按规范化题目划分，旧 SFT 已见题不会进入新验证集。全量检查确认混合训练与验证没有精确用户消息或来源问题组交叉。

## 追溯与复现

- `downloads.json`：新下载文件的固定版本、大小、SHA-256（Parquet 已与上游 LFS 校验）。
- `prior_training_profile.json`：两份实际旧训练文件的行数、哈希和启发式分布。
- `build_report.json`：以 **final_continuation** 字段为最终混合统计；此前 75/25 字段仅为加入 GSM8K 诊断前的初筛阶段统计。
- `validation_report.json`：全量 JSON、长度、分组隔离和 provenance 数量检查。
- `provenance_and_tests.jsonl.gz`：混合与验证每一行对应的来源、题号、token 数、上游测试及验证声明；测试保存在旁文件，不出现在训练对话中。
- `all_curated_sources_metadata.jsonl.gz`：全部精选源样本的题号、分组及测试。

构建顺序：仓库根目录运行 `python scripts/data_builder/collect_quality_python_sft.py`，然后 `python scripts/data_builder/build_quality_python_sft.py`，最后 `python scripts/data_builder/finalize_continuation_sft.py`。GSM8K 的固定版本下载地址记录在 `raw/gsm8k/source.json`，最终步骤需要该目录已下载的数据。若要完整重建，应从初筛脚本重新运行，不要只对已完成的最终混合重复运行 finalizer。

详细 GSM8K 诊断见 `eval/gsm8k_20260915_weakness_report.md`。
