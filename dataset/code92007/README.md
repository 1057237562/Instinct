# code92007 博客 + 代码语料

Blogger **Code92007**（CSDN 昵称“小衣同学”，GitHub 同名）的公开博客文章与自有代码仓库整理成的训练语料。

- 博客：https://blog.csdn.net/Code92007 （1013 篇文章，其中原创 995 篇；内容为算法竞赛题解与知识点总结：思维题、dp、线段树、数论、组合数学等，2018–2026）
- 代码：https://github.com/Code92007 的非 fork 仓库（cpc 算法竞赛 C++ 题库、cf-tuxun、tijing、oj-submission-wall、qq-cf-bot、ai-math-frontier、dlut-cpc、xcpcboard、dlut-cpc-resources）

## 文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `code92007_blog_pretrain.jsonl` | `{"text": ...}` | 每篇原创文章一行（标题 + markdown 正文），转载(type=2)、过短(<200字)、正文重复的文章已剔除 |
| `code92007_blog_sft.jsonl` | `{"conversations": [user, assistant]}` | 同一批文章的题解 SFT 对：user 为标题生成的求解指令，assistant 为文章正文（正文 ≥300 字才收录） |
| `code92007_code_pretrain.jsonl` | `{"text": ...}` | GitHub 自有仓库的源码文件，每文件一行，首行为 `repo:`/`# repo:`/`// repo:` 路径注释 |
| `build.report.json` | — | 构建报告：来源、各阶段行数、剔除统计、身份污染扫描、输出 SHA-256 |
| `review_candidates_sft.jsonl` | 带元数据行 | `filter_anomaly_candidates.py` 对 SFT 输出的分诊候选（构建时 4 条 placeholder/模板类），仅供复核，非删除决定 |
| `raw/` | — | 原始缓存：文章清单、gzip 的文章 HTML、浅克隆的仓库（`fetch-*` 阶段可断点续抓） |

## 复现

```bash
python scripts/data_builder/build_code92007_corpus.py fetch-index
python scripts/data_builder/build_code92007_corpus.py fetch-articles
python scripts/data_builder/build_code92007_corpus.py fetch-repos
python scripts/data_builder/build_code92007_corpus.py build
```

## 许可与使用限制

- 博客文章与未附带 LICENSE 的仓库内容版权归 **Code92007（Zemin Yi）** 所有；仅 `ai-math-frontier` 为 MIT。
- 本语料仅限本项目内部训练研究使用，**不得对外再分发**。
- 正文为 HTML→markdown 的自动转换结果，公式保留 `$...$` LaTeX 原文，图片已剔除（数量记录在报告 `blog.row_stats.images_dropped`）。

## 身份污染审计

按 AGENTS.md 要求对全部输出行扫描模型身份关键词（qwen / 通义千问 / 千问 / alibaba cloud / 阿里云 / 阿里巴巴 / 通义万相 / 通义听悟）。命中情况见 `build.report.json` 的 `identity_scan` 与 `raw/identity_hits.json`，命中行为人工复核候选，不做自动改写。加入训练混合前请先用
`scripts/data_builder/filter_anomaly_candidates.py` 做二次分诊。
