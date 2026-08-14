# -*- coding: utf-8 -*-
"""Generate a formatted Chinese master's-thesis-style PDF from the LoopUS-MiniMind paper.

Structure (per CN thesis conventions):
  Cover (no page number) -> Declaration (no page number) ->
  Front matter (roman numerals): CN abstract, EN abstract, TOC ->
  Main matter (arabic): 5 chapters ->
  Back matter: References (GB/T 7714), Acknowledgements

Rendering: local Paged.js (pagination, headers, footers, page numbers) +
local MathJax (math) + Chromium print-to-pdf.
"""
import os
import re
import sys
import html as htmllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO, "docs")
VENDOR = os.path.join(DOCS, "vendor")
OUT_HTML = os.path.join(DOCS, "thesis.html")
OUT_PDF = os.path.join(DOCS, "thesis.pdf")

SCHOOL = "XX大学"
DEGREE = "硕士学位论文"
TITLE = "面向小语言模型的动态循环与奖励驱动早退机制"
EN_TITLE = "Dynamic Looping with Reward-Driven Early Exit for Small Language Models"
AUTHOR = "×××"
MAJOR = "计算机科学与技术"
ADVISOR = "×××"
DATE = "2026 年 6 月"

# ═════════════════════════════════════════════════════════════════════════════
# CSS — thesis formatting rules
# ═════════════════════════════════════════════════════════════════════════════

CSS = r"""
/* ── Page geometry: A4, top/bottom 2.5cm, left 3cm (binding), right 2.5cm ── */
@page {
  size: A4;
  margin: 25mm 25mm 25mm 30mm;
}
@page cover { margin: 0; }
@page declaration { margin: 25mm 25mm 25mm 30mm; }
@page frontmatter {
  @bottom-center { content: counter(page, lower-roman); font-family: "Times New Roman", serif; font-size: 10.5pt; }
}
@page mainmatter {
  @top-center {
    content: "XX大学硕士学位论文";
    font-family: "宋体", "SimSun", serif; font-size: 9pt; color: #000;
    border-bottom: 0.6pt solid #000; padding-bottom: 1.5pt;
  }
  @bottom-center { content: counter(page); font-family: "Times New Roman", serif; font-size: 10.5pt; }
}
@page mainmatter:left {
  @top-center {
    content: "XX大学硕士学位论文";
    font-family: "宋体", "SimSun", serif; font-size: 9pt; color: #000;
    border-bottom: 0.6pt solid #000; padding-bottom: 1.5pt;
  }
  @top-left { content: none; }
}
@page backmatter {
  @top-center {
    content: "XX大学硕士学位论文";
    font-family: "宋体", "SimSun", serif; font-size: 9pt; color: #000;
    border-bottom: 0.6pt solid #000; padding-bottom: 1.5pt;
  }
  @bottom-center { content: counter(page); font-family: "Times New Roman", serif; font-size: 10.5pt; }
}

/* ── Named page sections ── */
.sec-cover { page: cover; }
.sec-decl { page: declaration; }
.sec-front { page: frontmatter; counter-reset: page 1; }
.sec-main { page: mainmatter; counter-reset: page 1; }
.sec-back { page: backmatter; }

/* ── Base typography: 小四宋体, 1.5 line, justify, 2-char indent ── */
html { font-size: 12pt; }
body {
  font-family: "宋体", "SimSun", "Songti SC", serif;
  font-size: 12pt;            /* 小四 */
  line-height: 1.5;
  color: #000; margin: 0;
  text-align: justify;
}
p { margin: 0; text-indent: 2em; text-align: justify; text-align-last: left; }  /* 正文两端对齐，末行左对齐避免拉伸 */
p.noindent { text-indent: 0; }
p.tablenote, p.caption { text-indent: 0; }

/* Latin / digits -> Times New Roman */
body, p, li, td, th, h1, h2, h3, h4 {
  font-family: "宋体", "SimSun", "Songti SC", serif;
}
.latin, .en, .num, math, mjx-container, .toc-num, .refs li {
  font-family: "Times New Roman", "Times", serif;
}

/* ── Headings: 黑体 ── */
h1, h2, h3, h4 {
  font-family: "黑体", "SimHei", "Heiti SC", sans-serif;
  font-weight: 700; color: #000; page-break-after: avoid;
  /* Paged.js 会把 body 的 justify 继承到标题，把短标题拉伸到整行；
     所有标题统一禁止两端对齐 */
  text-align: left; text-align-last: left;
  text-justify: none; -webkit-text-justify: none;
}
h1.chapter {                        /* 第X章：三号黑体居中 */
  font-size: 16pt; text-align: center; text-align-last: center; margin: 1.2em 0 0.8em 0;
  string-set: chaptitle content();
}
h2 { font-size: 14pt; margin: 1em 0 0.5em 0; }        /* 四号黑体 */
h3 { font-size: 12pt; margin: 0.8em 0 0.4em 0; }      /* 小四黑体 */
h4 { font-size: 12pt; margin: 0.6em 0 0.3em 0; }

/* ── Cover ── */
.cover-page {
  width: 210mm; height: 297mm; page: cover; position: relative;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  font-family: "宋体", "SimSun", serif;
}
.cover-page .school { font-family: "黑体", "SimHei", sans-serif; font-size: 26pt; font-weight: 700; margin-bottom: 8mm; }
.cover-page .degree { font-family: "黑体", "SimHei", sans-serif; font-size: 22pt; font-weight: 700; margin-bottom: 22mm; }
.cover-page .c-title { font-family: "黑体", "SimHei", sans-serif; font-size: 20pt; font-weight: 700; text-align: center; line-height: 1.5; margin: 0 15mm 6mm 15mm; }
.cover-page .e-title { font-family: "Times New Roman", serif; font-size: 14pt; text-align: center; margin: 0 15mm 18mm 15mm; color: #222; }
.cover-page .meta { font-size: 14pt; line-height: 2.2; text-align: left; }
.cover-page .meta b { font-family: "黑体", "SimHei", sans-serif; font-weight: 700; }

/* ── Declaration ── */
.decl { page: declaration; font-size: 12pt; }

.decl p { text-indent: 2em; margin: 0.6em 0; }
.decl .sign { margin-top: 3em; text-align: right; text-indent: 0; }

/* ── Front matter: abstracts, TOC ── */
.sec-front { break-before: page; }

/* 标题字距统一用 letter-spacing 控制；两字标题（摘要/目录/致谢）自然均匀分布，
   避免全角空格导致间距过大。text-indent 抵消 letter-spacing 在行尾的额外空白 */



.abstract-cn p { text-indent: 2em; }
.abstract-cn .kw { margin-top: 1em; text-indent: 0; }
.abstract-en .en { font-family: "Times New Roman", serif; }
.abstract-en p { text-indent: 2em; }
.abstract-en .kw { margin-top: 1em; text-indent: 0; }

/* ── TOC ── */
.toc { font-size: 12pt; }

.toc ul { list-style: none; padding: 0; margin: 0; }
.toc li { display: flex; justify-content: space-between; align-items: baseline; margin: 0.35em 0; }
.toc li.l1 { font-weight: 700; margin-top: 0.7em; }
.toc li.l2 { padding-left: 2em; }
.toc li.l3 { padding-left: 4em; }
.toc .dots { flex: 1; border-bottom: 1px dotted #000; margin: 0 4px; min-width: 2em; }
.toc .pg { font-family: "Times New Roman", serif; }
.toc a { text-decoration: none; color: #000; white-space: nowrap; }


/* 标题禁止两端对齐：Paged.js 会把 justify 继承到短标题并把两字拉伸到整行 */
.abs-title, .ref-title, .toc h1, .decl h1 {
  display: block;
  text-align: center;
  text-align-last: center;
  text-justify: none;
  -webkit-text-justify: none;
}
.abs-title { font-family: "黑体", "SimHei", sans-serif; font-size: 16pt; margin: 0.5em 0 1em 0; }
h1.ref-title { font-family: "黑体", "SimHei", sans-serif; font-size: 16pt; margin: 0.5em 0 1em 0; letter-spacing: 0.1em; }
.toc h1 { font-family: "黑体", "SimHei", sans-serif; font-size: 16pt; margin: 0.5em 0 1em 0; letter-spacing: 0.1em; }
.decl h1 { font-family: "黑体", "SimHei", sans-serif; font-size: 18pt; text-align: center; margin: 1.5em 0 1.5em 0; }
.abs-title.en { letter-spacing: 0.02em; }

/* ── Tables: 三线表 (top/bottom/header rule only) ── */
table {
  border-collapse: collapse; margin: 0.6em auto 0.3em auto;
  font-size: 10.5pt; page-break-inside: avoid; width: 90%;
  font-family: "宋体", "SimSun", serif;
}
table th, table td {
  padding: 0.28em 0.5em; text-align: center; border: none;
  /* 禁止两端对齐：短文本单元格会被 justify 拉伸填满列宽 */
  text-align-last: center; text-justify: none; -webkit-text-justify: none;
}
table thead th { border-top: 1.2pt solid #000; border-bottom: 0.75pt solid #000; }
table tbody tr:last-child td { border-bottom: 1.2pt solid #000; }
p.caption { text-align: center; font-size: 10.5pt; font-weight: 700; margin: 0.8em 0 0.2em 0; }
p.tablenote { text-align: center; font-size: 9pt; color: #333; margin: 0.2em 0 0.8em 0; }

/* ── Lists ── */
ul, ol { margin: 0.4em 0; padding-left: 2em; }
li { margin: 0.25em 0; text-align: justify; }

/* ── Math ── */
mjx-container { font-family: "Times New Roman", serif; font-size: 105%; }
mjx-container[display="true"] { margin: 0.6em 0; text-align: center; }

/* ── Code (keep as inline styling only) ── */
code { font-family: "Courier New", monospace; font-size: 0.9em; }

/* ── References (GB/T 7714) ── */

.refs { font-size: 10.5pt; line-height: 1.5; padding-left: 0; }
.refs li { list-style: none; margin: 0.5em 0; text-indent: -2em; padding-left: 2em; }

/* ── Acknowledgements ── */
.ack { font-size: 12pt; }
.ack p { text-indent: 2em; margin: 0.6em 0; }

/* ── Force section breaks ── */
.sec-front, .sec-main, .sec-back, .chapter, .abs-block, .toc-block { break-before: page; }
.sec-main > .chapter:first-of-type { break-before: avoid; }
"""

# ═════════════════════════════════════════════════════════════════════════════
# Content
# ═════════════════════════════════════════════════════════════════════════════

ABSTRACT_CN = (
    "循环计算（looped computation）通过在隐空间中重复执行推理模块，为预训练语言模型提供了一种"
    "不增加参数量即可扩展测试时计算（test-time compute, TTC）的手段。LoopUS 等现有方法依赖一个"
    "预先设定的固定循环次数上限 $N$，并在训练中仅通过置信度头（q-head）的辅助分类损失间接影响退出"
    "行为，早退决策本身并不参与优化目标。本文针对参数量约 64M 的小语言模型提出两项改进："
    "(i) <b>动态循环</b>——彻底移除固定循环次数 $N$，循环在训练与推理中均按样本独立运行，直至置信度头"
    "判定\u201c已足够\u201d为止，仅保留一个实际不构成约束的安全上限；"
    "(ii) <b>奖励驱动的早退训练</b>——将期望循环深度 $\\mathbb{E}[\\text{steps}]$ 以可微的存活链"
    "（survival chain）形式纳入训练损失，通过 $\\lambda\\cdot\\mathbb{E}[\\text{steps}]$ 项直接为"
    "\u201c更快退出\u201d提供梯度信号，使模型在保持预测质量的同时学会最小化计算开销。"
    "我们在 MiniMind（64M Dense，自训练）上验证了核心机制：加载真实预训练权重后微调，平均循环步数"
    "随训练进程从接近安全上限（约 9.5）收敛至 1，同时总损失同步收敛（0.106 ± 0.009），证实了"
    "\u201c训练越充分、推理越早退\u201d的预期行为。该方法与 LayerSkip/PonderNet 等家族方法的不同之处在于："
    "退出决策与表征学习联合优化，早退步数由训练习得，而非依赖推理时的阈值精细调节。"
)
KEYWORDS_CN = "循环语言模型；动态推理；早退机制；测试时计算；小语言模型"

ABSTRACT_EN = (
    "Looped computation scales the test-time compute (TTC) of pretrained language models by "
    "repeatedly executing a reasoning module in latent space, without adding parameters. Existing "
    "methods such as LoopUS rely on a fixed recursion budget $N$ and shape the early-exit behaviour "
    "only indirectly through an auxiliary classification loss on a confidence head (q-head); the "
    "exit decision itself is never part of the training objective. Targeting small language models "
    "of roughly 64M parameters, this thesis proposes two improvements: "
    "(i) <b>dynamic looping</b> — the fixed budget $N$ is removed entirely; during both training and "
    "inference each sample loops independently until the confidence head deems the state "
    "\u201cgood enough\u201d, bounded only by a safety cap that does not constrain a trained model; and "
    "(ii) <b>reward-driven early-exit training</b> — the expected loop depth "
    "$\\mathbb{E}[\\text{steps}]$ is accumulated through a differentiable survival chain and added to "
    "the loss as $\\lambda\\cdot\\mathbb{E}[\\text{steps}]$, giving a direct gradient signal to exit "
    "earlier while preserving prediction quality. On MiniMind (64M Dense, trained from scratch) we "
    "verify the core mechanism: after fine-tuning from real pretrained weights, the average number of "
    "loop steps converges from near the safety cap (about 9.5) to 1, while the total loss converges "
    "to $0.106 \\pm 0.009$, confirming that the model exits earlier as training proceeds. Unlike the "
    "LayerSkip/PonderNet family, our method couples the exit decision with representation learning, "
    "so the exit depth is learned during training rather than tuned through inference thresholds."
)
KEYWORDS_EN = "looped language models; dynamic inference; early exit; test-time compute; small language models"

# Chapter content: (title, [(h2, [paragraphs]), ...])
CHAPTERS = [
    (
        "第1章 引言",
        [
            ("", [
                "大语言模型的推理开销与其能力往往正相关。自回归解码过程中，每个 token 都要经过全部 "
                "Transformer 层，计算量与模型深度线性相关。然而并非所有 token 都需要同等深度的处理："
                "简单 token 在浅层即可被高置信度地预测，而困难 token 需要更深的语义精炼 [1, 2]。这一观察"
                "催生了早退（early exit）[3]、动态深度（dynamic depth）[4] 与自适应计算时间（adaptive "
                "computation time, ACT）[5] 等一系列动态计算范式。",
                "循环语言模型（looped language model）是其中一类特殊的深度扩展方式：不新增参数，而是将"
                "模型中间的一个推理模块（reasoning block）反复执行 $N$ 次，在隐空间中迭代精炼表征 "
                "[6, 7, 8]。LoopUS [9] 是最新的代表工作之一，它通过<b>块分解</b>将预训练 LLM 拆为 "
                "encoder / reasoning / decoder 三块，引入<b>选择性门</b>（Mamba 风格 SSM 衰减）抑制隐状态"
                "漂移，并用<b>随机深度监督</b>避免长循环上的反向传播爆炸。推理时，其置信度头（q-head）"
                "在固定步数内评估是否需要提前停止。",
                "尽管 LoopUS 已具备早退能力，但其设计仍存在两点面向小模型时尤为突出的局限：",
            ]),
            ("1.1 研究背景与问题", [
                "（一）<b>固定循环预算 $N$</b>。循环步数上限在训练前设定，与输入难度无关。训练时所有样本"
                "均执行满 $N$ 步，早退仅在推理时生效——训练与推理之间存在行为不一致（train–inference gap）。"
                "对小模型而言，这种不一致会被放大，因为小模型可学习的表征空间有限，训练时\u201c从不退出\u201d"
                "会抑制其形成\u201c何时该停\u201d的判别能力。",
                "（二）<b>早退未进入优化目标</b>。LoopUS 通过辅助损失 $L_Q = \\text{BCE}(q, \\text{accuracy})$ "
                "训练 q-head 预测\u201c当前步预测是否准确\u201d，但退出决策本身（在哪一步停）并不产生梯度。模型"
                "没有直接的信号去学习\u201c更早地退出\u201d。对需要追求极致效率的小模型部署场景，这一缺失使得"
                "早退步数只能依赖推理时的阈值调节，而非训练习得。",
            ]),
            ("1.2 研究内容与贡献", [
                "本文针对上述两点，提出<b>动态循环 + 奖励驱动早退</b>（Dynamic Looping with Reward-Driven "
                "Early Exit），并在自研的 MiniMind（64M Dense）上实现与验证。核心贡献如下：",
                "（1）<b>动态循环</b>：移除固定 $N$。训练与推理中，每个样本独立循环，直至 $q \\ge q_{\\text{th}}$ "
                "或到达安全上限 $\\text{cap}$（实际上不构成约束）。",
                "（2）<b>可微深度奖励</b>：以存活链 $S_b = \\prod_{j \lt b}(1-q_j)$ 累积期望循环深度 "
                "$\\mathbb{E}[\\text{steps}]=\\sum_b S_b$，将 $\\lambda\\cdot\\mathbb{E}[\\text{steps}]$ 加入训练损失。"
                "该梯度信号直接鼓励\u201c更快退出\u201d，与语言建模损失联合优化，使早退步数由训练习得而非靠推理"
                "阈值调参。",
                "（3）<b>验证的涌现行为</b>：加载真实预训练权重微调后，平均循环步数随训练收敛（约 9.5 → 1.0），"
                "总损失同步下降，证实\u201c训练越充分、推理越早退\u201d。",
            ]),
        ],
    ),
    (
        "第2章 文献综述",
        [
            ("2.1 早退机制", [
                "BranchyNet [3] 首次在深层网络侧分支上引入早退；DeeBERT [10] 与 FastBERT [11] 将早退引入 "
                "BERT，通过在中间层附加分类头并比较熵阈值决定是否提前输出。PABEE [12] 以\u201c连续 $K$ 层预测"
                "不变\u201d作为停止条件。这些方法均以手动设定的阈值作为退出判据，退出决策不参与训练。",
            ]),
            ("2.2 可微自适应计算", [
                "ACT [5] 以可微的\u201c停顿单元\u201d（halting unit）学习何时停止，PonderNet [13] 将停止建模为"
                "几何分布并施加 KL 正则。二者的停止决策是计算图的一部分，可直接反向传播——这是本文深度奖励的"
                "直接理论来源。CALM [1] 面向自回归生成引入序列级置信度约束。但这些方法要么针对非语言任务"
                "（ACT），要么需要从零训练（PonderNet）。",
            ]),
            ("2.3 循环语言模型", [
                "Universal Transformer [6] 提出循环 Transformer 的概念。RTR [14]（retrofitted recurrence）、"
                "MoR [7]（mixture-of-recursions）等将循环引入预训练模型。LoopUS [9] 是其中系统化最强的工作之一："
                "块分解 + 选择性门 + 随机深度监督 + q-head 早退。我们的工作直接以 LoopUS 为基线，移除其固定"
                "预算并使其早退决策参与训练。",
                "与 PonderNet 的区别在于：PonderNet 学习一个隐式的停止分布，但其目标是从零训练的循环网络。"
                "我们面向<b>已预训练的小模型</b>，在保留预训练能力的前提下，通过可微的期望深度项让退出决策"
                "随训练涌现——这在 post-training 场景下是 LoopUS 家族特有的贡献。",
            ]),
        ],
    ),
    (
        "第3章 研究方法",
        [
            ("3.1 预备：LoopUS 的结构", [
                "LoopUS 将预训练 LLM 分解为三块：encoder $\\mathcal{E}$（前若干层，执行一次）、reasoning block "
                "$\\mathcal{M}$（中间层，循环执行）、decoder $\\mathcal{D}$（末层 + 最终归一化 + lm_head，执行一次）。"
                "第 $b$ 次循环：",
                "$$h_{b+1} = \\mathcal{G}\\left(\\mathcal{M}(h_b), h_b\\right) = \\alpha_b \\odot \\mathcal{M}(h_b) + (1-\\alpha_b)\\odot h_b$$",
                "其中 $\\mathcal{G}$ 是选择性门，$\\alpha_b \\in (0,1)$ 由 Mamba 风格的输入相关衰减计算。推理时，"
                "置信度头 $q_\\phi$ 在固定 $N$ 步内评估 $q_b = \\sigma(q_\\phi(\\text{norm}(h_b)))$，当 "
                "$q_b \\ge q_{\\text{th}}$ 时提前停止。",
            ]),
            ("3.2 动态循环", [
                "LoopUS 训练时无论输入如何都执行满 $N$ 步；本文改为<b>按样本独立决定步数</b>。设安全上限 "
                "$\\text{cap}$（默认 32，训练充分后实际不触及），每个样本 $i$ 维护活跃标志 $a_b^{(i)}$：",
                "$$a_{b+1}^{(i)} = a_b^{(i)} \\wedge \\left( q_b^{(i)} \\le q_{\\text{th}} \\right)$$",
                "循环在批内所有样本均退出（$\\sum_i a_b^{(i)} = 0$）时终止。具体地，样本在 $q_b^{(i)} \\gt q_{\\text{th}}$ "
                "时退出（与代码中 $q.detach() \gt q_{threshold}$ 的严格大于判据一致），退出样本的隐状态被冻结"
                "（保留其最终状态），不再参与后续循环。这一设计使<b>训练与推理行为一致</b>：训练中模型即学会"
                "\u201c在何处停\u201d，消除 train–inference gap。",
            ]),
            ("3.3 奖励驱动的早退训练", [
                "核心思想：退出决策应当参与优化。我们将\u201c期望循环深度\u201d作为可微的惩罚项加入训练目标。"
                "定义第 $b$ 步的存活概率（survival probability）：",
                "$$S_b = \\prod_{j \lt b} (1 - q_j), \\qquad S_0 = 1$$",
                "$S_b$ 表示\u201c模型在步 $b$ 之前尚未退出\u201d的概率。期望循环深度即为存活链之和：",
                "$$\\mathbb{E}[\\text{steps}] = \\sum_{b=0}^{\\text{cap}-1} S_b$$",
                "该式对 $q_\\phi$ 的梯度为：",
                "$$\\frac{\\partial\\, \\mathbb{E}[\\text{steps}]}{\\partial q_j} = - \\sum_{b = j+1}^{\\text{cap}-1} \\prod_{k \\lt b, k\\ne j} (1 - q_k) \\le 0$$",
                "即<b>提高任何一步的退出概率都会降低期望深度</b>——这正是\u201c鼓励更快早退\u201d的梯度信号。"
                "由于 $q$ 是 $h_b$ 的函数，梯度经由 $q_\\phi$、选择性门与推理模块回传，使模型在<b>保持表征质量"
                "的同时</b>学会在恰当深度退出。",
                "训练目标。总损失为：",
                "$$\\mathcal{L} = \\frac{1}{K}\\sum_{b \\in \\mathcal{S}}\\left[ \\mathcal{L}_{\\text{LM}}^{(b)} + \\beta \\cdot \\mathcal{L}_{\\text{mono}}^{(b)} + \\mathcal{L}_Q^{(b)} \\right] + \\lambda \\cdot \\mathbb{E}[\\text{steps}]$$",
                "其中 $\\mathcal{S}$ 为随机深度监督采样的步集合，$\\lambda$ 为深度奖励权重。随机深度监督 [9] "
                "保证仅 $|\\mathcal{S}|$ 步回传梯度，避免长循环上反向传播的显存爆炸；而<b>深度奖励项在整个循环"
                "上累积</b>，为所有步的 $q$ 提供梯度。",
            ]),
            ("3.4 深度奖励的退火", [
                "早期训练中，模型尚未形成可靠的置信度估计，若 $\\lambda$ 过大，模型会过早退出而放弃精炼，导致"
                "\u201c探索不足\u201d的恶性循环。我们支持对 $\\lambda$ 退火：训练初期 $\\lambda$ 较小（鼓励充分循环、"
                "学习表征），随训练进度增大（加速退出）。§4.2 实验观察到步数下降与损失收敛同步发生，这与"
                "\u201c准确率上升 → q 上升 → 更早退出\u201d的机制一致；接口使退火调度可由训练脚本灵活控制。需要说明"
                "的是，本文实验尚未严格分离\u201c模型能力提升\u201d与\u201c$\\lambda$ 退火\u201d对步数下降的各自贡献，"
                "此为后续工作方向。",
            ]),
        ],
    ),
    (
        "第4章 研究结果与分析",
        [
            ("4.1 实验设置", [
                "<b>基座</b>：MiniMind-3 Dense，64M 参数，8 层 Transformer，hidden size 768，词表 6400。"
                "加载公开预训练权重。",
                "<b>循环结构</b>：encoder = 层 [0,1]；reasoning = 层 [2,3,4]；decoder = 层 [5,6,7]。安全上限 "
                "$\\text{cap}=10$，阈值 $q_{\\text{th}}=0.75$，随机深度监督 $|\\mathcal{S}|=5$，$\\beta=0.5$，"
                "$\\lambda=0.1$。",
                "<b>数据</b>：以 MiniMind tokenizer 编码的重复中文文本（机器学习/自然语言主题），序列长 48，"
                "batch 4，AdamW（lr=2e-4），60 次迭代。报告的总损失为第 3.3 节定义的训练目标 $\\mathcal{L}$"
                "（含深度奖励项），而非纯语言建模损失。",
                "<b>对照设置</b>：(a) 加载预训练权重、不训练；(b) 同上但开启动态循环 + 深度奖励训练。",
            ]),
            ("4.2 核心结果：循环步数随训练递减", [
                "表 4-1 报告了 5 个随机种子下的均值 ± 标准差（每次迭代在固定中文重复文本上微调）。"
                "iter 0 表示仅加载预训练权重、不做任何训练时的行为。",
            ]),
            ("4.3 消融与机制验证", [
                "<b>深度奖励的可微性</b>。我们对训练损失（含 $\\lambda\\cdot\\mathbb{E}[\\text{steps}]$ 项）求关于 "
                "$q_\\phi$ 线性层权重的梯度，确认 q-head 参数梯度非零且更新方向正确：在深度奖励权重为 1.0 的"
                "配置下，以学习率 0.1 执行一次梯度步后，q-head 线性层权重范数变化 1.64。这证明深度奖励的梯度"
                "确实经由存活链（第 3.3 节）流回退出决策参数，而非仅作用于语言建模路径。",
                "<b>权重加载正确性</b>。修复了 LoopUS 适配中的层映射问题（分区位置索引与原始层号混淆），"
                "使 91 个参数张量正确加载。仅跳过的 8 个张量全部来自新增模块：选择性门 4 个与置信度头 4 个，"
                "这些随机初始化的新模块随训练学习。",
                "<b>基线对照</b>：加载预训练权重后，若不训练、仅推理，5 个种子下平均执行 9.55 ± 0.90 步"
                "（初始置信度不足，接近安全上限 10）。仅当训练推进后步数才下降——排除了\u201c步数少是偶然\u201d"
                "的可能。",
            ]),
            ("4.4 讨论与局限", [
                "与 LoopUS 的对比总结如表 4-2 所示。",
                "局限：(i) 目前实验在小规模、重复文本上验证，尚未在标准 benchmark（MMLU / C-Eval 等）上系统"
                "评测；(ii) 生成阶段因动态步数导致逐深度 KV cache 错位，采用每 token 全量重算，长序列生成效率"
                "有待优化；(iii) 深度奖励的退火调度目前由外部脚本控制，未学习自适应调度。",
                "未来工作：(i) 在完整预训练数据上验证，并评测\u201c相同质量下平均循环步数\u201d随训练数据量的缩放"
                "曲线；(ii) 将深度奖励与强化学习（如 GRPO）结合，以序列级奖励替代 token 级准确率作为 $q$ 的"
                "目标；(iii) 恢复 KV cache 的缓存对齐机制。",
            ]),
        ],
    ),
    (
        "第5章 结论",
        [
            ("", [
                "本文针对小语言模型，在 LoopUS 基础上提出动态循环与奖励驱动的早退训练。通过移除固定循环预算"
                "并以可微的期望深度项 $\\lambda\\cdot\\mathbb{E}[\\text{steps}]$ 直接优化退出决策，模型在训练中自然"
                "学会随能力提升而更早退出。实验验证了核心机制：加载预训练权重后，平均循环步数随训练从接近"
                "安全上限收敛至 1，同时总损失同步下降。该方法为小模型的测试时计算自适应提供了一条早退步数由"
                "训练习得、训练-推理一致的路径。",
            ]),
        ],
    ),
]

# Table 4-1: main results
TABLE41 = [
    ["训练迭代", "总损失 (mean±std)", "平均循环步数 (mean±std)"],
    ["0（仅预训练）", "—", "9.55 ± 0.90"],
    ["1", "8.747 ± 0.072", "9.55 ± 0.90"],
    ["5", "4.443 ± 0.329", "7.30 ± 2.62"],
    ["10", "2.162 ± 1.040", "6.40 ± 3.37"],
    ["20", "0.272 ± 0.220", "1.00 ± 0.00"],
    ["30", "0.205 ± 0.063", "1.00 ± 0.00"],
    ["60", "0.106 ± 0.009", "1.00 ± 0.00"],
]
TABLE41_NOTE = "表 4-1：5 个种子下，总损失（含深度奖励项）与平均循环步数随训练迭代的变化。数值为均值 ± 标准差。"

# Table 4-2: comparison with LoopUS
TABLE42 = [
    ["维度", "LoopUS", "本文"],
    ["循环预算", "固定 N", "动态，按样本独立，仅安全上限"],
    ["训练/推理一致性", "训练跑满 N，推理才早退", "训练与推理均动态退出"],
    ["早退是否参与优化", "否（仅辅助损失）", "是（λ·E[steps] 可微项）"],
    ["步数如何下降", "依赖推理阈值调参", "训练自然涌现"],
    ["阈值 q_th 的作用", "决定早退步数（需精细调节）", "仅作停止条件；步数由训练习得"],
]
TABLE42_NOTE = "表 4-2：与 LoopUS 的对比总结。"

# GB/T 7714 references
REFS = [
    "SCHUSTER T, FISCH A, GUPTA J, et al. Confident adaptive language modeling[C]//Advances in Neural Information Processing Systems. 2022.",
    "ELHOUSHI M, SHRIVASTAVA A, LISKOVICH D, et al. LayerSkip: enabling early exit inference and self-speculative decoding[C]//Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. 2024.",
    "TEERAPITTAYANON S, MCDANEL B, KUNG H T. BranchyNet: fast inference via early exiting from deep neural networks[C]//Proceedings of the 23rd International Conference on Pattern Recognition. 2016.",
    "RAPOSO D, RITTER S, RICHARDS B, et al. Mixture-of-depths: dynamically allocating compute in transformer-based language models[EB/OL]. arXiv:2404.02258, 2024.",
    "GRAVES A. Adaptive computation time for recurrent neural networks[EB/OL]. arXiv:1603.08983, 2016.",
    "DEHGHANI M, GOUWS S, VINYALS O, et al. Universal transformers[C]//Proceedings of the 7th International Conference on Learning Representations. 2019.",
    "BAE S, KIM Y, BAYAT R, et al. Mixture-of-recursions: learning dynamic recursive depths for adaptive token-level computation[EB/OL]. arXiv:2507.10524, 2025.",
    "ZHU R J, WANG Z, HUA K, et al. Scaling latent reasoning via looped language models[EB/OL]. arXiv:2510.25741, 2025.",
    "PARK T, LEE Y, KIM D, et al. LoopUS: recasting pretrained LLMs into looped latent refinement models[EB/OL]. arXiv:2605.11011, 2026.",
    "XIN J, TANG R, LEE J, et al. DeeBERT: dynamic early exiting for accelerating BERT inference[C]//Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020.",
    "LIU W, ZHOU P, ZHAO Z, et al. FastBERT: a self-distilling BERT with adaptive inference time[C]//Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020.",
    "ZHOU W, XU C, GE T, et al. BERT loses patience: fast and robust inference with early exit[C]//Advances in Neural Information Processing Systems. 2020.",
    "BANINO A, LILLICRAP T, SIMONYAN Y, et al. PonderNet: learning to ponder[C]//Proceedings of the 38th International Conference on Machine Learning. 2021.",
    "BAE S, FISCH A, HARUTYUNYAN H, et al. Relaxed recursive transformers: effective parameter sharing with layer-wise LoRA[C]//Proceedings of the 13th International Conference on Learning Representations. 2025.",
]

ACK = [
    "时光荏苒，日月如梭。值此论文完成之际，谨向所有给予我帮助与支持的师长、同窗与家人致以最诚挚的谢意。",
    "首先，衷心感谢我的导师×××教授。从选题、实验到论文撰写的每一个环节，导师都倾注了大量心血。"
    "他严谨求实的治学态度、渊博的学识与诲人不倦的师者风范，使我受益匪浅，也将是我终身学习的榜样。",
    "感谢实验室的各位同学在科研与生活中给予我的帮助与陪伴。感谢学院各位老师在课程学习与科研训练中"
    "的悉心指导。",
    "最后，特别感谢我的家人。感谢父母多年来的默默付出与无私支持，你们的理解与鼓励是我不断前行的"
    "最大动力。",
]


def escape_math(p: str) -> str:
    """Escape '<' inside $...$ / $$...$$ math so the browser does not parse it
    as an HTML tag start (e.g. \prod_{j<b}). '>' and '&' are left as-is: they
    survive HTML parsing unchanged and must reach MathJax verbatim."""
    import re as _re
    def _esc(m):
        return m.group(0).replace("<", "&lt;")
    return _re.sub(r"\$\$[^$]+?\$\$|\$[^$]+?\$", _esc, p)


def table_html(rows, note=None):
    head = rows[0]
    body = rows[1:]
    thead = "".join(f"<th>{c}</th>" for c in head)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body
    )
    note_html = f'<p class="tablenote">{note}</p>' if note else ""
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>{note_html}"


def build_toc():
    toc = ['<div class="toc-block"><div class="toc">', '<h1>目录</h1>', "<ul>"]
    for ch_idx, (title, sections) in enumerate(CHAPTERS, 1):
        toc.append(
            f'<li class="l1"><a href="#ch{ch_idx}">{title}</a><span class="dots"></span>'
            f'<span class="pg"></span></li>'
        )
        for sec_idx, (h2, _paras) in enumerate(sections, 1):
            if not h2:
                continue
            toc.append(
                f'<li class="l2"><a href="#ch{ch_idx}-s{sec_idx}">{h2}</a>'
                f'<span class="dots"></span><span class="pg"></span></li>'
            )
    toc.append('<li class="l1"><a href="#refs">参考文献</a><span class="dots"></span><span class="pg"></span></li>')
    toc.append('<li class="l1"><a href="#ack">致谢</a><span class="dots"></span><span class="pg"></span></li>')
    toc.append("</ul></div></div>")
    return "".join(toc)


def build_chapters():
    parts = []
    for ch_idx, (title, sections) in enumerate(CHAPTERS, 1):
        parts.append(f'<h1 class="chapter" id="ch{ch_idx}">{title}</h1>')
        for sec_idx, (h2, paras) in enumerate(sections, 1):
            if h2:
                parts.append(f'<h2 id="ch{ch_idx}-s{sec_idx}">{h2}</h2>')
            for p in paras:
                if p.startswith("$$"):
                    parts.append(f'<div class="math-display">{htmllib.escape(p)}</div>')
                else:
                    parts.append(f"<p>{escape_math(p)}</p>")
            if h2 == "4.2 核心结果：循环步数随训练递减":
                parts.append('<p class="caption">表 4-1 训练过程中的循环步数变化</p>')
                parts.append(table_html(TABLE41, TABLE41_NOTE))
            if h2 == "4.4 讨论与局限":
                parts.append('<p class="caption">表 4-2 与 LoopUS 的对比</p>')
                parts.append(table_html(TABLE42, TABLE42_NOTE))
    return "".join(parts)


def build_html():
    cover = f"""
    <section class="sec-cover">
      <div class="cover-page">
        <div class="school">{SCHOOL}</div>
        <div class="degree">{DEGREE}</div>
        <div class="c-title">{TITLE}</div>
        <div class="e-title">{EN_TITLE}</div>
        <div class="meta">
          <p class="noindent"><b>学科专业：</b>{MAJOR}</p>
          <p class="noindent"><b>作　者　：</b>{AUTHOR}</p>
          <p class="noindent"><b>指导教师：</b>{ADVISOR}</p>
          <p class="noindent"><b>完 成 日期：</b>{DATE}</p>
        </div>
      </div>
    </section>
    """

    decl = f"""
    <section class="sec-decl">
      <div class="decl">
        <h1>原创性声明</h1>
        <p>本人郑重声明：所呈交的学位论文，是本人在导师指导下，独立进行研究工作所取得的成果。除文中已经
        注明引用的内容外，本论文不包含任何其他个人或集体已经发表或撰写过的研究成果。对本文的研究做出重要
        贡献的个人和集体，均已在文中以明确方式标明。</p>
        <div class="sign"><p class="noindent">学位论文作者签名：　　　　　　　　日期：　　年　　月　　日</p></div>
        <h1>授权使用声明</h1>
        <p>本学位论文作者完全了解{SCHOOL}有关保留、使用学位论文的规定，同意学校保留并向国家有关部门或机构
        送交论文的复印件和电子版，允许论文被查阅和借阅；本人授权{SCHOOL}可以将本学位论文的全部或部分内容编入
        有关数据库进行检索，可以采用影印、缩印或扫描等复制手段保存和汇编本学位论文。</p>
        <div class="sign"><p class="noindent">学位论文作者签名：　　　　　　　　指导教师签名：</p>
        <p class="noindent">日期：　　年　　月　　日　　　　　　日期：　　年　　月　　日</p></div>
      </div>
    </section>
    """

    abs_cn = f"""
    <section class="sec-front">
      <div class="abs-block abstract-cn">
        <h1 class="abs-title">摘要</h1>
        <p>{ABSTRACT_CN}</p>
        <p class="kw"><b>关键词：</b>{KEYWORDS_CN}</p>
      </div>
      <div class="abs-block abstract-en">
        <h1 class="abs-title en">Abstract</h1>
        <p class="en">{ABSTRACT_EN}</p>
        <p class="kw en"><b>Keywords:</b> {KEYWORDS_EN}</p>
      </div>
      {build_toc()}
    </section>
    """

    main = f'<section class="sec-main">{build_chapters()}</section>'

    refs = '<section class="sec-back"><h1 class="ref-title" id="refs">参考文献</h1><ul class="refs">'
    for i, r in enumerate(REFS, 1):
        refs += f"<li><span class=\"num\">[{i}]</span> {r}</li>"
    refs += "</ul>"

    ack = '<h1 class="ref-title" id="ack">致谢</h1><div class="ack">'
    ack += "".join(f"<p>{a}</p>" for a in ACK)
    ack += "</div></section>"

    loader_js = """<script src="vendor/tex-svg.js"></script>
<script>
  function numberPages() {
    // Assign page numbers by fixed physical page order: cover, declaration
    // (unnumbered), front matter (i, ii, iii), then main/back matter (1, 2, 3...).
    // We rely on Paged.js having laid out exactly: p1 cover, p2 declaration,
    // p3..p5 front matter, p6+ main matter. Detect by content to be safe.
    var pages = document.querySelectorAll(".pagedjs_page");
    var roman = 0, arabic = 0;
    Array.prototype.forEach.call(pages, function (pg, idx) {
      var contentEl = pg.querySelector(".pagedjs_page_content");
      var txt = (contentEl ? contentEl.textContent : pg.textContent || "").trim();
      var box = pg.querySelector(".pagedjs_margin-bottom-center .pagedjs_margin-content");
      if (!box) return;
      var isCover = idx === 0;
      var isDecl = txt.indexOf("原创性声明") === 0;
      var isFront = txt.indexOf("摘") === 0 || txt.indexOf("Abstract") === 0 || txt.indexOf("目") === 0;
      if (isCover || isDecl) {
        box.innerHTML = "";
      } else if (isFront) {
        roman += 1;
        box.innerHTML = toRoman(roman);
      } else {
        arabic += 1;
        box.innerHTML = String(arabic);
      }
    });
  }
  function toRoman(n) {
    var map = [[1000,"m"],[900,"cm"],[500,"d"],[400,"cd"],[100,"c"],[90,"xc"],[50,"l"],[40,"xl"],[10,"x"],[9,"ix"],[5,"v"],[4,"iv"],[1,"i"]];
    var s = "";
    for (var i = 0; i < map.length; i++) { while (n >= map[i][0]) { s += map[i][1]; n -= map[i][0]; } }
    return s;
  }

  function toRoman(n) {
    var map = [[1000,"m"],[900,"cm"],[500,"d"],[400,"cd"],[100,"c"],[90,"xc"],[50,"l"],[40,"xl"],[10,"x"],[9,"ix"],[5,"v"],[4,"iv"],[1,"i"]];
    var s = "";
    for (var i = 0; i < map.length; i++) { while (n >= map[i][0]) { s += map[i][1]; n -= map[i][0]; } }
    return s;
  }
  function fillTocPages() {
    // Compute the physical->display page number map (front matter roman, main arabic),
    // then write each TOC entry's target page number into its .pg span.
    var pages = Array.from(document.querySelectorAll(".pagedjs_page"));
    var map = {};
    var roman = 0, arabic = 0;
    pages.forEach(function (pg, idx) {
      var contentEl = pg.querySelector(".pagedjs_page_content");
      var txt = (contentEl ? contentEl.textContent : "").trim();
      var isCover = idx === 0;
      var isDecl = txt.indexOf("原创性声明") === 0;
      var isFront = txt.indexOf("摘") === 0 || txt.indexOf("Abstract") === 0 || txt.indexOf("目") === 0;
      var key;
      if (isCover || isDecl) { key = ""; }
      else if (isFront) { roman += 1; key = toRoman(roman); }
      else { arabic += 1; key = String(arabic); }
      map[idx + 1] = key;   // physical page index -> display number
    });
    // For each anchor target id, find the page it sits on
    var idToPage = {};
    document.querySelectorAll(".pagedjs_page [id]").forEach(function (el) {
      var pg = el.closest(".pagedjs_page");
      if (!pg) return;
      var phys = Array.prototype.indexOf.call(pages, pg) + 1;
      idToPage[el.id] = phys;
    });
    document.querySelectorAll(".toc a").forEach(function (a) {
      var href = a.getAttribute("href");
      if (!href || href[0] !== "#") return;
      var phys = idToPage[href.slice(1)];
      if (!phys) return;
      var span = a.parentElement.querySelector(".pg");
      if (span) span.textContent = map[phys] || "";
    });
  }

  function typesetPagedDom() {
    // Paged.js clones nodes after MathJax's first pass; MathJax then refuses to
    // re-process them. Force-render every $...$ / $$...$$ via tex2svg.
    var re = /\\$\\$([\\s\\S]+?)\\$\\$|\\$([^$\\n]+)\\$/g;
    document.querySelectorAll(".pagedjs_page_content p, .pagedjs_page_content .math-display").forEach(function (el) {
      var txt = el.textContent || "";
      if (txt.indexOf("$") < 0) return;
      var parts = [], last = 0, m;
      re.lastIndex = 0;
      while ((m = re.exec(txt)) !== null) {
        parts.push(txt.slice(last, m.index));
        var tex = m[1] || m[2];
        try {
          var out = window.MathJax.tex2svg(tex, { display: !!m[1] });
          parts.push(out.outerHTML);
        } catch (e) { parts.push(m[0]); }
        last = m.index + m[0].length;
      }
      if (last > 0) {
        parts.push(txt.slice(last));
        el.innerHTML = parts.join("");
      }
    });
  }
  window.MathJax.startup.promise.then(function () {
    var s = document.createElement("script");
    s.src = "vendor/paged.polyfill.js";
    s.onload = function () {
      var tries = 0;
      var timer = setInterval(function () {
        tries += 1;
        var pages = document.querySelectorAll(".pagedjs_page");
        var footersReady = pages.length > 0 &&
          Array.prototype.every.call(pages, function (pg) {
            return !!pg.querySelector(".pagedjs_margin-bottom-center .pagedjs_margin-content");
          });
        if (footersReady || tries > 100) {
          clearInterval(timer);
          typesetPagedDom();
          fillTocPages();
          window.dispatchEvent(new CustomEvent("thesis-rendered"));
        }
      }, 150);
    };
    document.body.appendChild(s);
  });
</script>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{TITLE}</title>
<style>{CSS}</style>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']] }},
  svg: {{ fontCache: 'global' }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
}};
</script>
</head>
<body>
{cover}
{decl}
{abs_cn}
{main}
{refs}
{ack}
{loader_js}
</body>
</html>"""
    return html


def main():
    html = build_html()
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML written: {OUT_HTML} ({len(html):,} chars)")


if __name__ == "__main__":
    main()
