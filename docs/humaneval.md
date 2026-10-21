# HumanEval 评测

推荐使用 **HumanEval-OJ** 评测完整程序：模型阅读题面、输入输出格式和公开样例，可先思考，再提交从 stdin 读取、向 stdout 写入的 Python 程序。WebUI 默认选择 HumanEval-OJ，并默认开启思考；原版 HumanEval 函数补全仍可单独选择。

```bash
# 首次使用会自动构建并验证全部 164 题；也可以先手动生成题库
python scripts/data_builder/build_humaneval_oj.py
python eval_humaneval_oj.py --load_from model --weight full_sft --mode all --open_thinking 1 --max_new_tokens 4096 --k 1 --output eval/humaneval_oj_samples.jsonl
```

默认题库为 `dataset/humaneval_oj/humaneval_oj_problems.jsonl`，缺失时自动准备官方缓存并转换，旧版不完整题库会自动重建。仅指定自定义路径时，缺失文件会明确报错。转换保留所有原始 `check(candidate)` 断言、循环测试和随机测试（固定种子 42），通过 stdin/stdout 适配执行，包括浮点容差及多解条件；不再只抽取可解析的字面量断言，也不因 None、字典或混合类型而漏题。常规 I/O 是标量逐行、数组长度加空格分隔元素；多行字符串、字典、元组和混合类型的具体协议写在各题题面中，全部是纯文本，不是 JSON。

最终代码从思考结束后的代码块提取；未结束的 `<think>` 内容不作为程序运行。提示只含题面、格式和公开样例，不含隐藏测试或参考程序。构建器只生成评测题库与 SHA-256 审计，不生成 SFT 训练数据。

指标明确标为 **HumanEval-OJ pass@K**。它与原版函数补全有不同的提示和作答协议，不能作为官方 HumanEval 分数混报。是否提升通过率需要在同一权重与采样设置下实测；开启思考并不保证提升。

`--mode all` 自动采用生成与评分流水线：每个批次的答案先写入 JSONL，再立即提交给 CPU 测试线程池，GPU 可继续生成下一批。`--workers` 控制评分并发，等待队列有上限，CPU 跟不上时会等待，避免无限堆积。逐样本结果及时写入 `_results.jsonl`，`_progress.json` 记录送评/完成数量；全部测试结束后才输出最终 pass@K 和压缩分析报告。失败中断保留已写入的答案和评分记录；`--resume` 不重复生成已有答案，但会重新评分已有答案以避免复用过期结果。

批量生成可加 `--batch_size 8`（默认 1），一次处理多个题目/样本；WebUI 中对应「生成 Batch size」。显存不足时减小。原生 Linear 主干仅对相同输入长度的任务组批。改变批次大小或从部分结果恢复可能改变随机采样的后续答案；严格复现需固定参数并从空文件运行。

从仓库根目录运行 `eval_humaneval.py`。复用 `eval_llm.py` 的模型参数，支持原生 `.pth`、Transformers 模型、LoRA 和 `--config_path`。默认从 OpenAI 官方 GitHub 下载 HumanEval，校验 164 道题的 ID 和字段后缓存到 `dataset/humaneval/HumanEval.jsonl.gz`；后续直接离线读取。HTTPS 保持证书验证，无需连接 Hugging Face。也可通过 `--problem_file` 指定本地 JSONL/JSONL.GZ，无需安装 human-eval。

生成完成自动保存 `<答案文件>.gz` 无损压缩副本。评分后额外输出 `_analysis.json`、`_analysis.md`（去重、截短的代表性失败输出及错误统计），以及 `_analysis_full.jsonl.gz`（完整题目、测试、参考答案、模型输出与错误信息）。HumanEval 记录异常类型、简短堆栈和是否达到生成 token 上限；后者仅为可能截断的线索。摘要用于定位待检查的能力，不证明训练数据缺失；参考答案与测试仅用于评测后分析，不用于生成提示，也不应混入训练集。

```bash
# SFT：生成单个答案（默认 greedy）；此步不执行生成代码
python eval_humaneval.py --load_from model --weight full_sft

# 计算 pass@1
python eval_humaneval.py --mode evaluate --k 1

# Transformers 模型：每题采样 10 次，生成并计算 pass@1、pass@10
python eval_humaneval.py --load_from ./instinct-3 --mode all --num_samples 10 --temperature 0.8 --k 1 10 --output eval/humaneval_10.jsonl

# 预训练模型使用原始函数前缀续写
python eval_humaneval.py --weight pretrain --prompt_style base --output eval/humaneval_base.jsonl

# 小规模验证；评测时需要指定同样的数据文件和 limit
python eval_humaneval.py --limit 3 --output eval/humaneval_smoke.jsonl
python eval_humaneval.py --mode evaluate --limit 3 --output eval/humaneval_smoke.jsonl
```

默认最多生成 512 tokens，可用 `--max_new_tokens` 修改。`--prompt_style auto` 对原生 pretrain 权重使用 base，其他使用 chat；外部 base 模型请显式选择 base。Chat 模式提取最后一个 Python 代码块，保留导入和辅助函数。Base 模式保留续写缩进并截断常见的下一题前缀。两种提示方式的分数应分别报告。

`--resume` 从已有 JSONL 继续，逐样本刷新文件；已有文件默认拒绝覆盖。恢复时须保持模型、数据、提示方式、随机种子和采样参数一致。若中断造成最后一行 JSON 不完整，先删除该残行再恢复。每题样本数不足的 k 会列入 `skipped_k`，不会伪报 pass@10/pass@100。`--limit` 的结果仅代表子集。

结果文件：`--output` 保存官方兼容的 `task_id`、`completion` 及额外的原始响应；`<output>_results.jsonl` 保存逐样本通过状态；`<output>_metrics.json` 保存 pass@k 和评测范围。仅评测本地文件不需要加载模型或导入 torch。

评分遵循 [OpenAI HumanEval](https://github.com/openai/human-eval) 的 `prompt + completion + test + check(entry_point)` 测试规则和无偏估计 `1 - C(n-c,k)/C(n,k)`，再对题目取均值。为兼容 Windows，每个答案启动独立 Python 子进程，`--timeout`（默认 3 秒）包含解释器启动时间，`--workers` 控制并发；执行时间边界与官方 Unix 信号实现有所不同。

`evaluate` / `all` 会执行模型生成的 Python。子进程和临时目录不是安全沙箱；应在没有敏感文件、凭据和网络权限的容器或虚拟机中评分。
