# A. All Closed 题解

## 题意

给定 \(n\) 个集合 \(S_1,S_2,\ldots,S_n\)，集合中的元素都是 \([0,2^m)\) 内的整数。

一次操作可以选择一个整数 \(x\)，然后把 \(x\) 同时加入所有集合。要求用最少的操作，使每个集合都满足：

\[
\forall x,y\in S_i,\qquad x\oplus y\in S_i.
\]

需要输出最少操作次数，以及一种最优操作方案。

## 一、把整数看成二进制向量

把一个 \(m\) 位整数看成 \(\mathbb F_2^m\) 中的向量，则按位异或就是向量加法。

一个非空集合 \(S\) 对异或封闭，当且仅当它是 \(\mathbb F_2\) 上的线性子空间：

1. 任取 \(x\in S\)，有 \(x\oplus x=0\)，所以 \(0\in S\)；
2. 题目已经保证它对向量加法（异或）封闭；
3. \(\mathbb F_2\) 中只有标量 0 和 1，因此数乘封闭自然成立。

所以题目实际上是在问：向所有集合同时补入哪些向量，才能让每个集合都变成线性空间？

记

\[
L_i=\operatorname{span}(S_i),
\]

即原集合 \(S_i\) 张成的线性空间。

## 二、操作集合需要满足什么条件

设进行操作的不同整数构成集合 \(A\)。重复加入同一个整数没有意义，所以只需要考虑集合，不需要考虑重复操作。

再记

\[
W=\operatorname{span}(A).
\]

操作后第 \(i\) 个集合为

\[
T_i=S_i\cup A.
\]

如果 \(T_i\) 是线性空间，那么它包含 \(S_i\) 和 \(A\)，因此也一定包含二者的张成空间：

\[
L_i+W=\{x\oplus y\mid x\in L_i,\ y\in W\}.
\]

但操作后真正存在的数只能来自 \(S_i\cup A\)，而 \(A\subseteq W\)。所以必须满足

\[
L_i+W\subseteq S_i\cup W.
\]

等价地，下面这个集合必须为空：

\[
D_i(W)=(L_i+W)\setminus(S_i\cup W).
\]

如果 \(x\in D_i(W)\)，那么：

- \(x\in L_i+W\)，最终的线性空间必须包含它；
- \(x\notin S_i\)，它原来不在第 \(i\) 个集合中；
- \(x\notin W\)，当前已知操作的张成空间也没有它。

因此 \(x\) 是一个新的必选操作。

## 三、为什么需要迭代

不能只对每个集合补一次 \(L_i\setminus S_i\)。某个数被加入所有集合后，可能会和另一个集合中的元素产生新的异或结果，从而迫使我们继续加入新数。

例如：

\[
S_1=\{1,2\},\qquad S_2=\{1\}.
\]

因为 \(1\oplus2=3\)，第一个集合迫使我们加入 3。把 3 同时加入第二个集合后，又有 \(1\oplus3=2\)，于是 2 也变成必选。

所以算法维护当前操作所张成的空间 \(W\)，反复寻找新的必选数：

```text
W = {0}

重复：
    记录 W 当前的维数

    对每个集合 S_i：
        U = W + span(S_i)
        D = U - W - S_i
        W = span(W 并 D)

    如果 W 的维数没有增加，则结束
```

这里初始的 \(W=\{0\}\) 是零空间。线性基数组为空，但它所表示的空间天然包含零向量。

## 四、如何得到最终操作集合

迭代结束后，对每个 \(i\) 都有

\[
W+L_i\subseteq W\cup S_i.
\]

并不是 \(W\) 中的所有数都必须操作。如果一个数本来就在每个 \(S_i\) 中，就没有必要再加入它。

因此最终答案为

\[
A=W\setminus\bigcap_{i=1}^{n}S_i.
\]

也就是枚举 \(W\) 中的每个数，只要至少有一个原集合没有它，就把它加入答案。

特别地，\(0\in W\)。如果有某个集合没有 0，那么 0 也会进入答案。这符合 XOR-closed 集合一定包含 0 的要求。

## 五、正确性证明

### 引理 1：算法发现的每个新向量都是必选操作

考虑任意可行操作集合 \(A'\)，记 \(W'=\operatorname{span}(A')\)。假设算法当前维护的 \(W\subseteq W'\)。初始时 \(W=\{0\}\)，该条件显然成立。

若算法发现

\[
x\in(W+L_i)\setminus(W\cup S_i),
\]

那么 \(x\in W'+L_i\)。操作后的第 \(i\) 个集合是线性空间，并且包含 \(A'\) 和 \(S_i\)，所以它必须包含 \(W'+L_i\)，进而必须包含 \(x\)。

又因为 \(x\notin S_i\)，所以只能通过操作加入，即 \(x\in A'\)。因此算法发现的每个新向量都属于任意可行方案。

把这些向量加入 \(W\) 并重新取张成空间后，仍然有 \(W\subseteq W'\)。由归纳法，该性质在整个算法中一直成立。

### 引理 2：算法输出的方案可行

算法停止时，对所有 \(i\) 都有

\[
W+L_i\subseteq W\cup S_i.
\]

令

\[
A=W\setminus\bigcap_i S_i.
\]

对于任意 \(w\in W\)：

- 如果 \(w\in\bigcap_i S_i\)，则它本来就在当前的 \(S_i\) 中；
- 否则 \(w\in A\)。

因此 \(W\subseteq S_i\cup A\)。另一方面，\(S_i\subseteq L_i\)，且 \(A\subseteq W\)，所以

\[
S_i\cup A\subseteq W+L_i.
\]

结合停止条件可得

\[
S_i\cup A=W+L_i.
\]

右侧是两个线性空间之和，仍然是线性空间。因此操作后每个集合都 XOR-closed。

### 引理 3：算法输出的方案最优

考虑任意可行方案 \(A'\)。由引理 1 的归纳过程，算法最终得到的 \(W\) 满足

\[
W\subseteq\operatorname{span}(A').
\]

任取算法答案中的 \(x\)。因为 \(x\notin\bigcap_i S_i\)，所以存在某个 \(i\) 使得 \(x\notin S_i\)。另一方面，操作后的第 \(i\) 个集合必须包含 \(\operatorname{span}(A')\)，从而包含 \(x\)。既然 \(x\notin S_i\)，就只能有 \(x\in A'\)。

所以算法的答案集合是任意可行答案集合的子集，操作次数一定最少。

## 六、代码中各变量的含义

| 变量 | 数学含义 |
| --- | --- |
| `sss[i]` | 原集合 \(S_i\) |
| `basi[i]` | \(L_i=\operatorname{span}(S_i)\) 的线性基 |
| `bas0` | 当前空间 \(W\) 的线性基 |
| `siz0` | \(\dim W\) |
| `bas1` | 临时空间 \(W+L_i\) 的线性基 |
| `siz1` | \(\dim(W+L_i)\) |
| `vis[x]` | 当前处理中，\(x\) 是否属于 \((W+L_i)\setminus(W\cup S_i)\) |
| `num[x]` | 对于 \(x\in W\)，原集合中缺少 \(x\) 的集合数量 |

输入中的实际 \(m\) 被读入 `tmp`，但代码统一按照最大值 `M=20` 处理。所有输入数的高位都是 0，因此不影响正确性，只是最后固定扫描 \([0,2^{20})\)。

## 七、异或线性基

下面是代码使用的线性基插入逻辑：

```cpp
for (int j = M - 1; j >= 0; --j) {
    if ((x >> j & 1) == 0)
        continue;

    if (bas[j]) {
        x ^= bas[j];
    } else {
        bas[j] = x;
        ++rank;
        break;
    }
}
```

`bas[j]` 保存一个最高位为 \(j\) 的基向量。插入一个数时从高位向低位消元：

- 如果对应最高位已经有基向量，就异或消掉这一位；
- 如果没有，就把当前向量作为新的基向量；
- 如果最后变成 0，说明它已经能被原来的基表示。

读入阶段，代码暂时借用 `bas0` 计算每个 \(L_i\) 的基，保存到 `basi[i]` 后再把 `bas0` 清空。全部读完后，`bas0` 才正式用于维护 \(W\)。

## 八、`vis` 的三次标记

处理第 \(i\) 个集合时，先令

\[
U=W+L_i.
\]

代码按下面的顺序修改 `vis`：

```text
枚举 U：    vis[x] = true
枚举 W：    vis[x] = false
枚举 S_i：  vis[x] = false
```

于是剩余的 `vis[x] == true` 恰好表示

\[
x\in U\setminus(W\cup S_i).
\]

代码随后再次枚举 \(U\)，把所有被标记的向量插入 `bas0`，也就是扩张 \(W\)。

如果构造出 \(U\) 后发现

```cpp
siz1 == siz0
```

说明 \(U=W\)，即 \(L_i\subseteq W\)，不可能产生 \(W\) 外的新必选向量，可以直接跳过。

## 九、如何枚举线性空间中的所有元素

设一个线性空间有 \(k\) 个基向量。选择基向量的任意子集并把它们异或起来，就能得到空间中的一个元素；因为基向量线性无关，\(2^k\) 个子集与空间中的 \(2^k\) 个元素一一对应。

原代码中的

```cpp
int t = 0, tx = 0;
for (int j = 0; j < M; ++j) {
    if (bas[j]) {
        t |= 1 << j;
        tx ^= bas[j];
        val[1 << j] = tx;
    }
}

for (int s = t, x = tx; s;
     x ^= val[s & -s], s = (s - 1) & t) {
    // 使用 x
}
```

就是在枚举基向量位置集合 `t` 的所有非空子集。

如果每次根据子集重新计算异或值，会多出一个 \(O(m)\) 因子。这里的 `val[lowbit]` 预处理了切换到下一个子集时需要翻转的若干基向量之异或，因此可以在 \(O(1)\) 时间从当前的 `x` 更新到下一个 `x`。

这个循环不会枚举零向量，所以最后统计答案时单独执行：

```cpp
num[0] = n;
```

## 十、为什么插入线性基后要恢复 `x`

代码中有下面的结构：

```cpp
int x0 = x;
if (vis[x]) {
    // 插入线性基，过程中会不断修改 x
}
x = x0;
```

这里的 `x` 不仅是待插入的向量，还是线性空间枚举器维护的当前状态。高斯消元会破坏它，因此必须先用 `x0` 保存原值，插入结束后恢复，否则下一次子集枚举得到的向量会出错。

`vis[x] = false` 使用的可能是消元后的余量。这不会影响正确性：原向量插入后已经属于扩张后的 \(W\)，以后真正读取 `vis` 前会在“枚举并清除 \(W\)”的步骤中被清除。

## 十一、终止性与复杂度

只要某一整轮扫描使 `siz0` 增加，\(W\) 的维数就至少增加 1。由于

\[
\dim W\le m\le20,
\]

维数增加最多发生 20 次，因此外层循环最多经历约 21 个完整轮次。

空间枚举看起来可能达到 \(n2^m\)，但可以进行摊还分析：

- 如果 \(D_i(W)\) 非空，这次处理后 \(W\) 的维数一定增加，这种事件总共最多 20 次；
- 如果 \(D_i(W)\) 为空但 \(W+L_i\ne W\)，则 \((W+L_i)\setminus W\subseteq S_i\)。因为 \(W\) 是 \(W+L_i\) 的真子空间，空间中至少一半元素位于 \(W\) 外，所以本次枚举规模至多是 \(O(|S_i|)\)；
- 题目保证 \(\sum |S_i|\le10^6\)。

因此结合 \(m\le20\)，所有枚举和线性基操作都能通过。内存主要为原集合、每个集合的基，以及大小为 \(2^{20}\) 的辅助数组：

\[
O\left(\sum_i|S_i|+nm+2^m\right).
\]

## 十二、完整代码

下面保留原实现的整体结构和变量名，只添加了少量注释。

```cpp
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

using std::cin, std::cout;

#define F(i, a, b) for (int i = (a); i <= (b); ++i)
#define F2(i, a, b) for (int i = (a); i < (b); ++i)
#define dF(i, a, b) for (int i = (a); i >= (b); --i)

void Solve();

int main() {
    cin.sync_with_stdio(false);
    cin.tie(nullptr);
    Solve();
    return 0;
}

const int MN = 200005;
const int M = 20;

int n;

// bas0：当前空间 W 的线性基；siz0：dim(W)。
int bas0[M], siz0;

// bas1：临时空间 W + L_i 的线性基；siz1：它的维数。
int bas1[M], siz1;

// sss[i] 保存原集合 S_i，basi[i] 保存 span(S_i) 的一组基。
std::vector<int> sss[MN];
std::vector<int> basi[MN];

void Solve() {
    int input_m;
    cin >> n >> input_m;

    // 预处理每个 L_i = span(S_i) 的线性基。
    // 此时暂时借用 bas0 作为工作数组，每处理完一个集合就清空。
    F(i, 1, n) {
        int c;
        cin >> c;

        std::vector<int> seq(c);
        for (int &x : seq)
            cin >> x;
        sss[i] = seq;

        for (int x : seq) {
            dF(j, M - 1, 0) {
                if ((x >> j & 1) == 0)
                    continue;

                if (bas0[j]) {
                    x ^= bas0[j];
                } else {
                    bas0[j] = x;
                    ++siz0;
                    break;
                }
            }
        }

        F2(j, 0, M) {
            if (bas0[j])
                basi[i].push_back(bas0[j]);
        }

        F2(j, 0, M)
            bas0[j] = 0;
        siz0 = 0;
    }

    // 从零空间开始，不断扩张 W，直到达到固定点。
    while (true) {
        int original_size = siz0;

        F(i, 1, n) {
            const auto &seq = sss[i];

            // bas1 = span(bas0, basi[i]) = W + L_i。
            F2(j, 0, M)
                bas1[j] = bas0[j];
            siz1 = siz0;

            for (int x : basi[i]) {
                dF(j, M - 1, 0) {
                    if ((x >> j & 1) == 0)
                        continue;

                    if (bas1[j]) {
                        x ^= bas1[j];
                    } else {
                        bas1[j] = x;
                        ++siz1;
                        break;
                    }
                }
            }

            // L_i 已包含在 W 中，此时 W + L_i = W，不会产生新向量。
            if (siz1 == siz0)
                continue;

            static bool vis[1 << M];
            static int val[1 << M];

            // 标记 W + L_i 中的所有非零向量。
            {
                int t = 0, tx = 0;
                F2(j, 0, M) {
                    if (bas1[j]) {
                        t |= 1 << j;
                        tx ^= bas1[j];
                        val[1 << j] = tx;
                    }
                }

                for (int s = t, x = tx; s;
                     x ^= val[s & -s], s = (s - 1) & t) {
                    vis[x] = true;
                }
            }

            // 去掉 W 中的向量。
            {
                int t = 0, tx = 0;
                F2(j, 0, M) {
                    if (bas0[j]) {
                        t |= 1 << j;
                        tx ^= bas0[j];
                        val[1 << j] = tx;
                    }
                }

                for (int s = t, x = tx; s;
                     x ^= val[s & -s], s = (s - 1) & t) {
                    vis[x] = false;
                }
            }

            // 再去掉原集合 S_i 中的向量。
            // 此时 vis[x] 为真当且仅当 x 属于
            // (W + L_i) \ (W 并 S_i)。
            for (int x : seq)
                vis[x] = false;

            // 把所有新发现的必选向量加入 W 的线性基。
            {
                int t = 0, tx = 0;
                F2(j, 0, M) {
                    if (bas1[j]) {
                        t |= 1 << j;
                        tx ^= bas1[j];
                        val[1 << j] = tx;
                    }
                }

                for (int s = t, x = tx; s;
                     x ^= val[s & -s], s = (s - 1) & t) {
                    int original_x = x;

                    if (vis[x]) {
                        dF(j, M - 1, 0) {
                            if ((x >> j & 1) == 0)
                                continue;

                            if (bas0[j]) {
                                x ^= bas0[j];
                            } else {
                                bas0[j] = x;
                                ++siz0;
                                break;
                            }
                        }

                        vis[x] = false;
                    }

                    // 消元修改了 x，恢复枚举器维护的原始状态。
                    x = original_x;
                }
            }
        }

        if (siz0 == original_size)
            break;
    }

    static int num[1 << M];

    // 对每个 x in W，先令 num[x] = n。
    {
        int t = 0;
        F2(j, 0, M) {
            if (bas0[j])
                t |= 1 << j;
        }

        for (int s = t; s; s = (s - 1) & t) {
            int x = 0;
            F2(j, 0, M) {
                if (s >> j & 1)
                    x ^= bas0[j];
            }
            num[x] = n;
        }

        // 上面的非空子集枚举不会得到零向量。
        num[0] = n;
    }

    // 对每个原本包含 x 的集合减一。
    // 因而 x in W 时，num[x] 表示有多少个集合缺少 x。
    F(i, 1, n) {
        for (int x : sss[i])
            --num[x];
    }

    std::vector<int> ans;
    F2(x, 0, 1 << M) {
        if (num[x] > 0)
            ans.push_back(x);
    }

    cout << static_cast<int>(ans.size()) << '\n';
    F2(i, 0, static_cast<int>(ans.size())) {
        cout << ans[i] << " \n"[i + 1 == static_cast<int>(ans.size())];
    }
}
```
