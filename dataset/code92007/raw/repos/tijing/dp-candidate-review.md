# DP 候选题二次筛选表

> 本表仅供审核，不会直接写入站点题库。编号 `N01`–`N82` 按本轮来源顺序固定，便于回复筛选结果。

## 口径

- **经典**：模型具有代表性，适合放到“经典题目”。
- **实战**：题目质量高，但更偏综合应用、状态设计或优化技巧，适合放到“实战训练”。
- **备选**：有价值，但与更好的候选重复、DP 不是主角，或实现/技巧过于偏门。
- **专题**：不放 DP 新分组，按现有专题处理。
- 同一道题可以进入多个子分组；插入 DP、乱序 DP、进位数位 DP、Convex Hull Trick 与决策单调性作为正式子分组，其余零散优化技巧继续作为标签。
- 原题链接优先使用 AtCoder、Codeforces、牛客、HDU、洛谷、QOJ 等比赛或官方题库入口；失效的老 OJ 才保留可用归档入口。

## 第一轮强推

我建议优先考虑进入“经典题目”的是：`N02 N04 N11 N13 N16 N18 N20 N21 N32 N36 N37 N40 N41 N43 N56 N60 N61 N64 N70 N73 N75 N77 N82`。

其中 `N16 HDU4055 Number String` 放在“计数 DP / 插入 DP”，并标注 `排列 DP / 插入法 / 前缀和优化 / 波浪排列`。资料中的“置置置换”与它是同类插入模型，只保留一条主记录并在备注中互相指向。

## 线性、计数、排列与优化 DP

| 编号 | 原题 | 建议分组与标签 | 去向 | 判断 | 题解 |
| --- | --- | --- | --- | --- | --- |
| N01 | [ARC170 C - Prefix Mex Sequence](https://atcoder.jp/contests/arc170/tasks/arc170_c) | 线性 DP；计数、MEX、状态压缩 | 实战 | MEX 状态设计有辨识度，适合作为综合计数训练。 | [CSDN](https://blog.csdn.net/Code92007/article/details/135738593) |
| N02 | [ABC221 H - Count Multiset](https://atcoder.jp/contests/abc221/tasks/abc221_h) | 线性 DP；计数、整数拆分、容斥、差分 | **经典** | 同一题串起拆分数、容斥与差分，知识密度高。 | [CSDN](https://blog.csdn.net/Code92007/article/details/135730490) |
| N03 | [CF1913D - Array Collapse](https://codeforces.com/contest/1913/problem/D) | 线性 DP；单调栈、计数 | 实战 | 单调栈维护 DP 转移，模型典型但更适合作为进阶练习。 | [CSDN](https://blog.csdn.net/Code92007/article/details/135447470) |
| N04 | [CF1860D - Balanced String](https://codeforces.com/contest/1860/problem/D) | 线性 DP；二维状态、逆序对 | **经典** | 状态含义直接、转移完整，适合做基础多维 DP 的代表题。 | [CSDN](https://blog.csdn.net/Code92007/article/details/133502695) |
| N05 | [ABC299 F - Square Subsequence](https://atcoder.jp/contests/abc299/tasks/abc299_f) | 线性 DP；子序列、序列自动机 | 实战 | 子序列 DP 与 next 数组结合自然，训练价值高。 | [CSDN](https://blog.csdn.net/Code92007/article/details/131875555) |
| N06 | [ECNA 2023 B - B Road Band](https://codeforces.com/gym/104757/problem/B) | DP 优化 / 决策单调性；分段、分治优化 | 实战 | 是决策单调性与分治优化的完整应用题。 | [CSDN](https://blog.csdn.net/Code92007/article/details/134636520) |
| N07 | [ICPC Jinan 2022 J - Skills](https://codeforces.com/gym/104076/problem/J) | 线性 DP；根号分治、松弛 | 实战 | 松弛思路值得保留，但仍归线性 DP，不单开“DP 优化”。 | [CSDN](https://blog.csdn.net/Code92007/article/details/133592838) |
| N08 | [ARC164 D - 1D Coulomb](https://atcoder.jp/contests/arc164/tasks/arc164_d) | 线性 DP；组合计数、括号/前缀状态 | 实战 | 状态与组合约束结合紧，适合训练抽象。 | [CSDN](https://blog.csdn.net/Code92007/article/details/131636416) |
| N09 | [ABC279 G - At Most 2 Colors](https://atcoder.jp/contests/abc279/tasks/abc279_g) | 线性 DP；计数、滑动窗口/前缀和 | 实战 | 转移优化自然，难度适合作为中档实战。 | [CSDN](https://blog.csdn.net/Code92007/article/details/128068982) |
| N10 | [CF1739E - Cleaning Robot](https://codeforces.com/contest/1739/problem/E) | 线性 DP；局部状态、分类讨论 | 实战 | 小状态但细节密集，适合检验状态完整性。 | [CSDN](https://blog.csdn.net/Code92007/article/details/127175595) |
| N11 | [CF1582F2 - Korney Korneevich and XOR](https://codeforces.com/contest/1582/problem/F2) | 线性 DP；值域状态、XOR、可达性 | **经典** | 用值域压状态消除子序列维度，是很好的状态设计范例。 | [CSDN](https://blog.csdn.net/Code92007/article/details/120950700) |
| N12 | [CF1542E2 - Abnormal Permutation Pairs](https://codeforces.com/contest/1542/problem/E2) | 线性 DP；排列计数、组合数学 | 实战 | 推导较重，适合作为高阶计数 DP。 | [CSDN](https://blog.csdn.net/Code92007/article/details/118496495) |
| N13 | [ABC134 F - Permutation Oddness](https://atcoder.jp/contests/abc134/tasks/abc134_f) | 计数 DP / 插入 DP；排列、贡献 | **经典** | 插入元素并维护贡献，是排列类 DP 的标准范例。 | [CSDN](https://blog.csdn.net/Code92007/article/details/113834827) |
| N14 | [CF1487G - String Counting](https://codeforces.com/contest/1487/problem/G) | 线性 DP；字符串计数、多项式/卷积 | 备选 | DP 主线可取，但多项式部分抢占重点，入门路径不够纯。 | [CSDN](https://blog.csdn.net/Code92007/article/details/113829262) |
| N15 | [BAPC 2018 E - Entirely Unsorted Sequences](https://codeforces.com/gym/102007/problem/E) | 线性 DP；计数、首个非法位置、多重集排列 | 实战 | “枚举第一个破坏位置”的补集计数思路很有训练价值。 | [CSDN](https://blog.csdn.net/Code92007/article/details/110296304) |
| N16 | [HDU4055 - Number String](https://acm.hdu.edu.cn/showproblem.php?pid=4055) | 计数 DP / 插入 DP；**排列、前缀和优化、波浪排列** | **经典** | 插入 trick 独特且可迁移；“置置置换”作为同构题备注，不重复建题。 | [CSDN](https://blog.csdn.net/Code92007/article/details/106976509) |
| N18 | [FZU2129 - 子序列个数](http://acm.fzu.edu.cn/problem.php?pid=2129) | 线性 DP；本质不同子序列、last 优化 | **经典** | 不同子序列计数的标准模型，状态和去重都很典型。 | [CSDN](https://blog.csdn.net/Code92007/article/details/89813655) |
| N19 | [CF1149B - Three Religions](https://codeforces.com/contest/1149/problem/B) | 线性 DP；多序列、在线子序列、next 数组 | 实战 | 动态维护三个串的组合状态，综合性强。 | [CSDN](https://blog.csdn.net/Code92007/article/details/89736427) |
| N20 | [CF1110D - Jongmah](https://codeforces.com/contest/1110/problem/D) | 线性 DP；局部计数、滚动状态 | **经典** | 将三元组选择压到相邻值域，模型简洁且代表性强。 | [CSDN](https://blog.csdn.net/Code92007/article/details/86776899) |
| N45 | [CF1101F - Trucks and Cities](https://codeforces.com/contest/1101/problem/F) | DP 优化 / 决策单调性；分段、分治优化、二分答案 | 实战 | 优化链条完整，但更适合高阶训练而非基础经典。 | [CSDN](https://blog.csdn.net/Code92007/article/details/86500021) |
| N52 | [CF2021D - Boss, Thirsty](https://codeforces.com/contest/2021/problem/D) | 线性 DP；前后缀最值、区间端点状态 | 实战 | 四类端点转移和增量维护很适合练状态拆分。 | [CSDN](https://blog.csdn.net/Code92007/article/details/142734368) |
| N53 | [ARC115 E - LEQ and NEQ](https://atcoder.jp/contests/arc115/tasks/arc115_e) | 线性 DP；容斥、单调栈、前缀和 | 实战 | 从朴素分段 DP 到奇偶压缩、单调栈优化，推导完整。 | [CSDN](https://blog.csdn.net/Code92007/article/details/135718686) |
| N54 | [ICPC Yinchuan 2020 M - Tower of the Sorcerer](https://codeforces.com/gym/104022/problem/M) | 线性 DP；数论分块、ST 表、单点更新 | 备选 | 技巧组合偏门且题解带“乱搞 AC”色彩，先放候补。 | [CSDN](https://blog.csdn.net/Code92007/article/details/134233208) |
| N56 | [ABC288 F - Integer Division](https://atcoder.jp/contests/abc288/tasks/abc288_f) | 线性 DP；划分、贡献拆分、前缀和优化 | **经典** | 从枚举最后一段到 O(n) 递推，适合展示优化过程。 | [CSDN](https://blog.csdn.net/Code92007/article/details/131875586) |
| N58 | [CF1453F - Even Harder](https://codeforces.com/contest/1453/problem/F) | 线性 DP；路径唯一性、后缀最小值 | 实战 | 状态设计不直观，适合进阶状态推导训练。 | [CSDN](https://blog.csdn.net/Code92007/article/details/110686771) |
| N59 | [2019 Nowcoder Multi-University 10 J - Wood Processing](https://ac.nowcoder.com/acm/contest/890/J) | DP 优化 / Convex Hull Trick；分组、斜率优化 | 实战 | 是斜率优化在分段 DP 中的正规赛题应用。 | [CSDN](https://blog.csdn.net/Code92007/article/details/108356003) |
| N60 | [CF868F - Yet Another Minimization Problem](https://codeforces.com/contest/868/problem/F) | DP 优化 / 决策单调性；分段、分治优化、双指针维护代价 | **经典** | 分治优化 DP 的代表题，模型和实现都具有模板价值。 | [CSDN](https://blog.csdn.net/Code92007/article/details/108212280) |
| N62 | [Nowcoder 91849 D - String Circle](https://ac.nowcoder.com/acm/contest/91849/D) | 线性 DP；计数、KMP、循环移位、字符串周期 | 实战 | 先筛合法旋转再计数非降切点，字符串与 DP 结合得很好。 | 直接题面 |
| N63 | [CF2109E - Binary String Wowee](https://codeforces.com/contest/2109/problem/E) | 线性 DP；组合数学、二进制串计数 | 实战 | 组合贡献与 DP 递推结合，适合作为新题实战。 | [知乎](https://zhuanlan.zhihu.com/p/2080679627098476953) |
| N64 | [CF659G - Fence Divercity](https://codeforces.com/contest/659/problem/G) | 线性 DP；矩形计数、前缀贡献 | **经典** | 状态短、贡献解释清晰，是很好的线性计数 DP。 | [知乎](https://zhuanlan.zhihu.com/p/2079985618810058207) |
| N65 | [CF2038D - Divide OR Conquer](https://codeforces.com/contest/2038/problem/D) | 计数 DP / 乱序 DP；二维偏序 | 实战 | 遍历顺序本身就是难点，适合训练偏序状态的组织方式。 | [知乎](https://zhuanlan.zhihu.com/p/2060737935792776329) |
| N67 | [CF1946F - Nobody is needed](https://codeforces.com/contest/1946/problem/F) | 线性 DP；离线、刷表法、Fenwick | 实战 | “刷表法”保留为标签，主分组仍是线性/计数 DP。 | [知乎](https://zhuanlan.zhihu.com/p/2057471713684805232) |
| N68 | [CF2237F - Paint the Array](https://codeforces.com/contest/2237/problem/F) | 线性 DP；数组染色、状态压缩 | 实战 | 新题且状态设计完整，适合放训练区观察反馈。 | [知乎](https://zhuanlan.zhihu.com/p/2053544212642239959) |
| N69 | [CF1832E - Combinatorics Problem](https://codeforces.com/contest/1832/problem/E) | 线性 DP；组合数递推、多阶前缀和 | 实战 | 能体现组合恒等式如何落成递推。 | [知乎](https://zhuanlan.zhihu.com/p/2053167799665701269) |
| N70 | [洛谷 P5999 / CEOI 2016 - kangaroo](https://www.luogu.com.cn/problem/P5999) | 计数 DP / 插入 DP；排列、相邻关系 | **经典** | 与 N13、N16 形成清晰的排列插入 DP 小专题。 | [知乎](https://zhuanlan.zhihu.com/p/2050743938642981229) |
| N71 | [CF2122E - Greedy Grid Counting](https://codeforces.com/contest/2122/problem/E) | 线性 DP；网格计数、组合递推 | 实战 | 偏比赛型状态设计，适合作为综合训练。 | [知乎](https://zhuanlan.zhihu.com/p/2047083081794954761) |
| N73 | [CF1715E - Long Way Home](https://codeforces.com/contest/1715/problem/E) | DP 优化 / Convex Hull Trick；最短路分层、斜率优化 | **经典** | DP、最短路与 CHT 的衔接很有代表性。 | [知乎](https://zhuanlan.zhihu.com/p/2006899847589011664) |
| N74 | [CF2176F - Omega Numbers](https://codeforces.com/contest/2176/problem/F) | 线性/计数 DP；GCD、经典容斥 | 备选 | 数论与容斥占比高，DP 分类辨识度稍弱。 | [知乎](https://zhuanlan.zhihu.com/p/1990905858247381851) |
| N76 | [CF2039E - Shohag Loves Inversions](https://codeforces.com/contest/2039/problem/E) | 线性 DP；组合数学、逆序对计数 | 实战 | 组合推导与递推结合自然，适合中高阶训练。 | [知乎](https://zhuanlan.zhihu.com/p/16504033556) |
| N78 | [Universal Cup Kunming G - GCD](https://codeforces.com/gym/105588/problem/G) | 线性 DP；DFS、GCD、数值状态 | 备选 | 小状态 DP 可学，但数论题味更重、可迁移性一般。 | [知乎](https://zhuanlan.zhihu.com/p/13958375418) |
| N79 | [HDU6289 - 寻宝游戏](https://acm.hdu.edu.cn/showproblem.php?pid=6289) | 线性 DP / 背包 DP；网格、换入换出、消后效性 | 实战 | `dp[i][j][out][in]` 的交换建模很独特，适合作为高阶实战。 | [知乎](https://zhuanlan.zhihu.com/p/12794712918) |

## 区间 DP

| 编号 | 原题 | 建议分组与标签 | 去向 | 判断 | 题解 |
| --- | --- | --- | --- | --- | --- |
| N37 | [ABC325 G - offence](https://atcoder.jp/contests/abc325/tasks/abc325_g) | 区间 DP；字符串消除、区间合并 | **经典** | 状态直观、转移有代表性，适合区间 DP 主线。 | [CSDN](https://blog.csdn.net/Code92007/article/details/133987855) |
| N39 | [CF1863F - Divide, XOR, and Conquer](https://codeforces.com/contest/1863/problem/F) | 区间 DP；XOR、端点性质 | 实战 | 位运算性质决定区间转移，综合性强。 | [CSDN](https://blog.csdn.net/Code92007/article/details/132660891) |
| N40 | [HDU4283 - You Are the One](https://acm.hdu.edu.cn/showproblem.php?pid=4283) | 区间 DP；出栈顺序、分割 | **经典** | 区间分割含义清楚，是标准入门到进阶题。 | [CSDN](https://blog.csdn.net/Code92007/article/details/105463727) |
| N41 | [CF1312E - Array Shrinking](https://codeforces.com/contest/1312/problem/E) | 区间 DP；合并、两阶段 DP | **经典** | 先判区间能否缩成一个值，再做最少段数，结构非常典型。 | [CSDN](https://blog.csdn.net/Code92007/article/details/105463735) |
| N42 | [HDU6212 - Zuma](https://acm.hdu.edu.cn/showproblem.php?pid=6212) | 区间 DP；消除、端点合并 | 备选 | 与 Blocks/Zuma 类模型重合，保留作补充即可。 | [CSDN](https://blog.csdn.net/Code92007/article/details/105283891) |
| N43 | [POJ1390 - Blocks](http://poj.org/problem?id=1390) | 区间 DP；消除、附加维状态 | **经典** | `dp[l][r][k]` 是区间 DP 消除类最具代表性的状态之一。 | [CSDN](https://blog.csdn.net/Code92007/article/details/105273397) |
| N44 | [Nowcoder NC201932 - 递增递增](https://ac.nowcoder.com/acm/problem/201932) | 区间 DP；填坑 DP、组合计数 | 实战 | 状态设计特别，适合训练非标准区间 DP。原赛为 Day6 D。 | [CSDN](https://blog.csdn.net/Code92007/article/details/104233527) |
| N61 | [ICPC Kunming 2021 C - Cities](https://ac.nowcoder.com/acm/contest/12548/C) | 区间 DP；分割、端点决策 | **经典** | 原题入口稳定，区间状态与转移具备专题代表性。 | 直接题面 |

## 树形与背包 DP

| 编号 | 原题 | 建议分组与标签 | 去向 | 判断 | 题解 |
| --- | --- | --- | --- | --- | --- |
| N21 | [HDU4003 - Find Metal Mineral](https://acm.hdu.edu.cn/showproblem.php?pid=4003) | **树形 DP / 背包 DP**；分组背包、返回/不返回状态 | **经典** | 树上分组背包模型清楚，值得在两个子分组同时出现。 | [CSDN](https://blog.csdn.net/Code92007/article/details/106241986) |
| N26 | [CF2021E2 - Digital Village](https://codeforces.com/contest/2021/problem/E2) | **树形 DP / 背包 DP**；Kruskal 重构树、闵可夫斯基和 | 备选 | 质量高但链路过长，适合作为高阶补充而非主线。 | [CSDN](https://blog.csdn.net/Code92007/article/details/142734416) |
| N27 | [CF1868C - Travel Plan](https://codeforces.com/contest/1868/problem/C) | 树形 DP；完全二叉树、组合计数、记忆化 | 实战 | `n` 极大但树结构隐式，训练结构化计数。 | [CSDN](https://blog.csdn.net/Code92007/article/details/132893371) |
| N28 | [HDU7401 - 流量监控](https://acm.hdu.edu.cn/showproblem.php?pid=7401) | **树形 DP / 背包 DP**；匹配计数、二维树上背包 | 实战 | 多维树上背包有代表性，但实现和推导都偏重。 | [CSDN](https://blog.csdn.net/Code92007/article/details/132397475) |
| N29 | [CCPC Guangzhou 2022 I - Infection](https://codeforces.com/gym/104053/problem/I) | **树形 DP / 背包 DP / 概率期望 DP**；感染分布 | 实战 | 三类交叉很鲜明，应在对应分组重复展示同一题。 | [CSDN](https://blog.csdn.net/Code92007/article/details/127894678) |
| N30 | [CF1695D2 - Tree Queries](https://codeforces.com/contest/1695/problem/D2) | 树形 DP；树上结构、询问/构造 | 实战 | 树结构观察强，适合树形 DP 综合训练。 | [CSDN](https://blog.csdn.net/Code92007/article/details/125353997) |
| N31 | [HDU6540 - Neko and tree](https://acm.hdu.edu.cn/showproblem.php?pid=6540) | 树形 DP；计数、子树状态合并 | 实战 | 属于正统树上状态合并，可作为训练题。 | [CSDN](https://blog.csdn.net/Code92007/article/details/107116642) |
| N32 | [HDU6035 - Colorful Tree](https://acm.hdu.edu.cn/showproblem.php?pid=6035) | 树形 DP；换根/补集贡献、颜色计数 | **经典** | 按颜色计算路径贡献的思路经典，树上贡献法代表性强。 | [CSDN 多题文中的 C 题](https://blog.csdn.net/Code92007/article/details/101801659) |
| N55 | [CF1856E2 - PermuTree (hard version)](https://codeforces.com/contest/1856/problem/E2) | **树形 DP / 背包 DP**；子树大小、bitset、根号优化 | 实战 | 树上背包优化很实用，主分组放树形和背包，优化作标签。 | [CSDN](https://blog.csdn.net/Code92007/article/details/132136500) |
| N57 | [力扣天池 04 - 意外惊喜](https://leetcode.cn/contest/tianchi2022/problems/tRZfIV/) | 背包 DP；分治优化、字典序/构造 | 实战 | 分治优化背包很少见，保留能扩展背包章节的形态。 | [CSDN](https://blog.csdn.net/Code92007/article/details/127484363) |

## 状态压缩与子集 DP

| 编号 | 原题 | 建议分组与标签 | 去向 | 判断 | 题解 |
| --- | --- | --- | --- | --- | --- |
| N17 | [CF11D - A Simple Task](https://codeforces.com/contest/11/problem/D) | 状态压缩 DP / 子集 DP；简单环计数 | **已有 T10** | 与当前保留题重复，不再新增。 | [CSDN](https://blog.csdn.net/Code92007/article/details/105942760) |
| N33 | [小米邀请赛 2020 决赛 J - Rikka with Book](https://ac.nowcoder.com/acm/contest/9328/J) | 状态压缩 DP；子集划分、集合收益 | 实战 | 原比赛链接已核实，适合作为状压实战。 | [CSDN](https://blog.csdn.net/Code92007/article/details/135035310) |
| N34 | [CF1392G - Omkar and Pies](https://codeforces.com/contest/1392/problem/G) | 状态压缩 DP / 子集 DP；置换状态、滑动区间 | 实战 | `k<=20` 的状态与超长操作序列结合很有挑战。 | [CSDN](https://blog.csdn.net/Code92007/article/details/108147908) |
| N35 | [CF1313D - Happy New Year](https://codeforces.com/contest/1313/problem/D) | 状态压缩 DP / 子集 DP；扫描线、超集转移 | 实战 | “每点覆盖不超过 8 次”转局部掩码，是很好的轮廓状压。 | [CSDN](https://blog.csdn.net/Code92007/article/details/107713295) |
| N36 | [CF1215E - Marbles](https://codeforces.com/contest/1215/problem/E) | 状态压缩 DP / 子集 DP；逆序对贡献、排列顺序 | **经典** | 将颜色排列的 `20!` 压成 `2^20`，是状压 DP 的标准模型。 | [CSDN](https://blog.csdn.net/Code92007/article/details/100880773) |
| N38 | [CF1870E - Another MEX Problem](https://codeforces.com/contest/1870/problem/E) | 状态压缩 DP / 子集 DP；MEX、集合状态 | 实战 | MEX 与子集状态结合紧，适合作为非模板状压训练。 | [CSDN](https://blog.csdn.net/Code92007/article/details/133253325) |
| N66 | [CF938F - Erasing Substrings](https://codeforces.com/contest/938/problem/F) | 状态压缩 DP；贪心、字符串删除 | 实战 | 贪心约束状态空间后再状压，综合性较强。 | [知乎](https://zhuanlan.zhihu.com/p/2057834466828104669) |
| N75 | [ABC391 G - Many LCS](https://atcoder.jp/contests/abc391/tasks/abc391_g) | 状态压缩 DP；LCS 数组性质、自动机式转移 | **经典** | 把整行 LCS 差分压成掩码，状态压缩技巧很漂亮。 | [知乎](https://zhuanlan.zhihu.com/p/21079011670) |

## 数位 DP

| 编号 | 原题 | 建议分组与标签 | 去向 | 判断 | 题解 |
| --- | --- | --- | --- | --- | --- |
| N22 | [CF1982E - Number of k-good subarrays](https://codeforces.com/contest/1982/problem/E) | 数位 DP；二进制分治、区间信息合并 | 实战 | 与传统逐位计数不同，能展示“数位块合并”的写法。 | [CSDN](https://blog.csdn.net/Code92007/article/details/139974616) |
| N23 | [CCPC Guangzhou 2022 M - XOR Sum](https://codeforces.com/gym/104053/problem/M) | **数位 DP / 背包 DP**；二进制、数位背包、余数 | 实战 | 明确的交叉模型，应同时挂在数位和背包分组。 | [CSDN](https://blog.csdn.net/Code92007/article/details/127840174) |
| N24 | [CF1734F - Zeros and Ones](https://codeforces.com/contest/1734/problem/F) | 数位 DP / 进位数位 DP；Thue-Morse、进位 | 实战 | 利用序列自相似性做数位递推，辨识度高。 | [CSDN](https://blog.csdn.net/Code92007/article/details/127040401) |
| N25 | [CF1487F - Ones](https://codeforces.com/contest/1487/problem/F) | 数位 DP / 进位数位 DP；高位到低位、借位、延迟贡献 | 实战 | 非典型数位状态，对进借位建模训练很好。 | [CSDN](https://blog.csdn.net/Code92007/article/details/113829187) |
| N72 | [洛谷 P3447 / POI 2006 - KRY-Crystals](https://www.luogu.com.cn/problem/P3447) | 数位 DP；进制表示、计数 | 实战 | 老竞赛题，模型扎实；先放实战比“经典”更稳妥。 | [知乎](https://zhuanlan.zhihu.com/p/2009081118369023725) |
| N81 | [CCPC Liaoning 2024 H - 划分数字](https://codeforces.com/gym/105481/problem/H) | 数位 DP；数字划分、计数 | 实战 | 只收整篇补题中的 H；I、K 不属于本次 DP 清单。 | [知乎](https://zhuanlan.zhihu.com/p/6078125026) |

## 概率与期望 DP

| 编号 | 原题 | 建议分组与标签 | 去向 | 判断 | 题解 |
| --- | --- | --- | --- | --- | --- |
| N46 | [CF1628D2 - Game on Sum](https://codeforces.com/contest/1628/problem/D2) | 概率 / 期望 DP；逆向递推、博弈过程 | 实战 | 状态小、推导有启发性，适合作为期望 DP 进阶题。 | [CSDN](https://blog.csdn.net/Code92007/article/details/135687656) |
| N47 | [CF1823F - Random Walk](https://codeforces.com/contest/1823/problem/F) | 概率 / 期望 DP；树上随机游走 | 实战 | 同时带树结构，但主知识点是随机过程期望。 | [CSDN](https://blog.csdn.net/Code92007/article/details/131507242) |
| N48 | [CF1753C - Wish I Knew How to Sort](https://codeforces.com/contest/1753/problem/C) | 概率 / 期望 DP；随机交换、逆序状态 | 实战 | 把全局排列过程压成错位数量，状态降维漂亮。 | [CSDN](https://blog.csdn.net/Code92007/article/details/127484327) |
| N49 | [CF1523E - Crypto Lights](https://codeforces.com/contest/1523/problem/E) | 概率 / 期望 DP；组合概率、停止时间 | 备选 | 推导质量可以，但与更具代表性的期望题相比优先级稍低。 | [CSDN](https://blog.csdn.net/Code92007/article/details/118417574) |
| N50 | [CF1392H - ZS Shuffles Cards](https://codeforces.com/contest/1392/problem/H) | 概率 / 期望 DP；随机过程、贡献法 | 实战 | 期望贡献拆分有代表性，可与状态递推题形成互补。 | [CSDN](https://blog.csdn.net/Code92007/article/details/108148714) |
| N51 | [Winter School 2014 G - A path to knowledge](https://codeforces.com/gym/100371/problem/G) | 概率 / 期望 DP；最短路 DAG、路径计数 | 备选 | 题目本质更偏最短路与路径概率，适合作为交叉补充。 | [CSDN](https://blog.csdn.net/Code92007/article/details/101350037) |
| N77 | [ABC231 G - Balls in Boxes](https://atcoder.jp/contests/abc231/tasks/abc231_g) | 概率 / 期望 DP；盒子模型、组合期望 | **经典** | 随机投球模型标准，适合概率 DP 章节作为代表题。 | [知乎](https://zhuanlan.zhihu.com/p/14748125814) |
| N82 | [CF1778D - Flexible String Revisit](https://codeforces.com/contest/1778/problem/D) | 概率 / 期望 DP；距离状态、**手动高斯消元** | **经典** | 递推形成线性方程，手动消元技巧独特且很适合专题讲解。 | [知乎](https://zhuanlan.zhihu.com/p/2116350159) |

## 现有专题与特殊处理

| 编号 | 原题 | 建议位置与标签 | 去向 | 判断 | 题解 |
| --- | --- | --- | --- | --- | --- |
| N80 | [CCPC Xiamen 2019 J - Zayin and Tree](https://qoj.ac/problem/9110) | **错解不优**；树、松弛、错误复杂度分析、改进 | **专题** | 按你的要求归入现有“错解不优”，不建立“DP 优化（松弛）”分组。 | [知乎](https://zhuanlan.zhihu.com/p/12793542634) |

## 二筛建议

- 若要先控制规模，我建议第一批只审核 23 道“经典”，再从“实战”里按每个子分组补 3–6 道。
- `N13 / N16 / N70` 可以组成“排列 DP 与插入法”小链路，其中 `N16` 的波浪排列插入 trick 最独特。
- 区间 DP 优先顺序建议为 `N43 > N41 > N40 > N37 > N61`；`N42` 与 `N43` 重合度较高。
- 树形/背包交叉优先 `N21`，概率/树形交叉优先 `N29`，数位/背包交叉优先 `N23`。
- DP 优化下设 `Convex Hull Trick / 决策单调性` 两个稳定子分组；其余 `松弛 / 单调栈优化 / 前后缀优化` 等继续作为标签。
