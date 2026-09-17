# 首次 Coding SFT 新增候选源

目标是首次 SFT：标准 `sft_t2t_mini.jsonl` 是基础组成，不是回放。
当前收集的是原始候选池，尚不是可直接训练的最终 75% Coding / 25% 通用混合。

## 新增来源

- [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft)：真实提交的修改前后代码与提交说明，覆盖 Python、JavaScript、TypeScript、Java、C++、Go、Rust、Shell、SQL。适合构造修改/修复任务；提交说明不保证是完整需求，提交存在也不等于通过本地测试。数据集卡标为 MIT，但原始仓库许可证必须按每条 `license` 字段处理，存在 unknown 和其他许可。
- [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback)：代码生成、执行反馈、多轮修改候选；数据集卡标为 Apache-2.0。可能与 Magicoder 等种子题重叠，不能视作全部独立新题。

版本、文件大小、行数和 SHA-256 由 `manifest.json` 记录。原始文件不执行、不截断。

## 标准集初步盘点

`sft_t2t_mini.jsonl`：905,718 条；多用户轮次 124,742 条；含 tools/tool_calls/tool 结构 84,832 条；含代码围栏 66,052 条。统计可重叠，代码围栏只是标记而非语义分类。尚未进行聊天模板长度过滤。

## 最终合成前必须完成

1. 按实际聊天模板计算长度；超过 4096 tokens 整条丢弃。
2. 与标准 SFT、当前预训练以及新增源对照题面、答案和代码重复；区分精确重复与近似重复。
3. 按任务和语言分类，优先 Debug、测试和重构；不只依据文件名分类。
4. 验证集按题目/仓库隔离；整合标准集的中文、多轮及工具调用样本。
5. 在过滤后的可用量基础上确定最终配比，并记录本地测试状态。

重现下载：在仓库根目录运行 `python dataset/scripts/collect_first_sft_sources.py`。
