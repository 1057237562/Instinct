# 算法竞赛 C++ 编码风格

## 概述

本文档描述了一名算法竞赛选手的编码风格、约定和模式。将其作为参考，生成与该选手风格完全一致的代码。

---

## 文件结构

- 每道题解都是**单个自包含的 `.cpp` 文件**——没有 `.h` 头文件，没有多文件项目
- 文件按题目字母命名：`a.cpp`、`b.cpp`、`c.cpp` 等
- 多次尝试：`d.cpp`、`d2.cpp`、`d3.cpp`、`d4.cpp`
- 失败的尝试：`e_fail.cpp`
- 算法专项练习文件：`sais.cpp`、`ntt.cpp`、`silly_tree.cpp`
- 按平台组织比赛目录：`codeforces/`、`atcoder/`、`luogu/`、`nowcoder/` 等

---

## 标准模板（Boilerplate）

每个文件都以这个精确的头部开始。有两种变体：

### 变体 A — 完整模板（最常见，约 80% 的文件）

```cpp
//#include<bits/stdc++.h>
#include<iostream>
#include<cstdio>
#include<vector>
#include<map>
#include<set>
using namespace std;
#define rep(i,a,b) for(int i=(a);i<=(b);++i)
#define per(i,a,b) for(int i=(a);i>=(b);--i)
typedef long long ll;
typedef double db;
typedef pair<int,int> P;
typedef array<int,3> A;
#define fi first
#define se second
#define pb push_back
#define dbg(x) cerr<<(#x)<<":"<<x<<" ";
#define dbg2(x) cerr<<(#x)<<":"<<x<<endl;
#define SZ(a) (int)(a.size())
#define sci(a) scanf("%d",&(a))
#define pt(a) printf("%d",a);
#define pte(a) printf("%d\n",a)
#define ptlle(a) printf("%lld\n",a)
const int N=2e5+10;
```

### 变体 B — 精简模板（较简单的问题）

```cpp
#include<iostream>
#include<cstdio>
#include<vector>
#include<cmath>
#include<algorithm>
#include<random>
using namespace std;
#define rep(i,a,b) for(int i=(a);i<=(b);++i)
#define per(i,a,b) for(int i=(a);i>=(b);--i)
typedef long long ll;
typedef double db;
#define ull unsigned long long
#define fi first
#define se second
#define pb push_back
#define dbg(x) cerr<<(#x)<<":"<<x<<" ";
#define dbg2(x) cerr<<(#x)<<":"<<x<<endl;
#define SZ(a) (int)(a.size())
#define sci(a) scanf("%d",&(a))
#define pt(a) printf("%d",a);
#define pte(a) printf("%d\n",a)
#define ptlle(a) printf("%lld\n",a)
```

### 模板关键说明：
- `#include<bits/stdc++.h>` 被**注释掉**，下方是显式的 include
- `typedef pair<int,int> P` — 需要 64 位时用 `pair<ll,ll> P`
- `typedef array<int,3> A` — 需要时用 `array<ll,3> A`
- `N` 根据题目约束设定：`105`、`2e5+10`、`3e3+10`、`5e5+10`、`5e6+10`
- 始终 `+10` 作为 1-indexed 的安全余量
- 需要模运算时添加 `const int mod=998244353;`

---

## 宏参考表

| 宏 | 展开 | 用途 |
|---|---|---|
| `rep(i,a,b)` | `for(int i=(a);i<=(b);++i)` | 正向循环，闭区间 |
| `per(i,a,b)` | `for(int i=(a);i>=(b);--i)` | 反向循环，闭区间 |
| `ll` | `long long` | 64 位整数 |
| `db` | `double` | 浮点数 |
| `ull` | `unsigned long long` | 无符号 64 位 |
| `P` | `pair<int,int>` 或 `pair<ll,ll>` | 坐标对、边 |
| `A` | `array<int,3>` 或 `array<ll,3>` | 三元组 |
| `fi` / `se` | `.first` / `.second` | pair 访问器 |
| `pb` | `.push_back()` | vector 追加 |
| `dbg(x)` | `cerr<<(#x)<<":"<<x<<" "` | 调试输出到 stderr（无换行） |
| `dbg2(x)` | `cerr<<(#x)<<":"<<x<<endl` | 调试输出到 stderr（有换行） |
| `SZ(a)` | `(int)(a.size())` | 安全的 size 转换 |
| `sci(a)` | `scanf("%d",&(a))` | 快速 int 输入 |
| `pt(a)` | `printf("%d",a)` | int 输出 |
| `pte(a)` | `printf("%d\n",a)` | int 输出 + 换行 |
| `ptlle(a)` | `printf("%lld\n",a)` | long long 输出 + 换行 |

---

## I/O 模式

### 主要方式：scanf/printf（约 95% 的文件使用）

```cpp
int n, m;
sci(n), sci(m);  // 逗号运算符多输入
ll x;
scanf("%lld", &x);
pte(ans);         // int + 换行
ptlle(ans);       // long long + 换行
```

### 空格分隔输出 + 末尾换行技巧：

```cpp
rep(i, 1, n) {
    printf("%d%c", a[i], " \n"[i == n]);
}
```

### 快速 I/O（仅用于极端输入规模，N > 1e6）：

```cpp
namespace fastIO {
    static char buf[100000], *h = buf, *d = buf;
    #define gc getchar()
    template<typename T>
    inline void read(T &x) {
        int f = 1; x = 0;
        register char c(gc);
        while (c > '9' || c < '0') { if (c == '-') f = -1; c = gc; }
        while (c <= '9' && c >= '0') x = (x << 1) + (x << 3) + (c ^ 48), c = gc;
        x *= f;
    }
    template<typename T>
    void output(T x) {
        if (x < 0) { putchar('-'); x = ~(x - 1); }
        static int s[20], top = 0;
        while (x) { s[++top] = x % 10; x /= 10; }
        if (!top) s[++top] = 0;
        while (top) putchar(s[top--] + '0');
    }
}
using namespace fastIO;
```

### 交互题：

```cpp
int ask(int u, int v) {
    printf("? %d %d\n", u, v);
    fflush(stdout);
    int w;
    scanf("%d", &w);
    return w;
}
void out(int x) {
    printf("! %d\n", x);
    fflush(stdout);
}
```

### 多组测试数据模式：

```cpp
int t, n;
sci(t);
while (t--) {
    sci(n);
    // 处理一组数据
    // 在组间重置全局变量
}
```

---

## 变量声明

### 所有变量全局声明：

```cpp
const int N = 2e5 + 10;
int t, n, m, k, a[N], b[N], dp[N];
ll ans, sum;
vector<int> e[N];
```

### 1-indexed 数组（通用约定）：

```cpp
rep(i, 1, n) sci(a[i]);  // 始终 1-indexed
```

### 测试数据间重置模式：

```cpp
while (t--) {
    sci(n);
    rep(i, 1, n) dp[i] = 0, e[i].clear();
    // 求解
}
```

---

## 代码风格

### 缩进：4 个空格

### 大括号风格：K&R（左大括号在同一行）

```cpp
int main() {
    sci(t);
    while (t--) {
        if (ok) {
            // ...
        }
        else {
            // ...
        }
    }
    return 0;
}
```

### 命名约定：
- **单字母变量**：`n`、`m`、`k`、`t`、`x`、`y`、`u`、`v`、`w`、`p`、`q`、`l`、`r`
- **常用名称**：`ans`、`res`、`cnt`、`sum`、`dp`、`a[]`、`b[]`
- **复杂结构用描述性名称**：`segtree`、`CHT`、`SAIS`
- **数组按题目变量命名**：`a[N]`、`dp[N][N]`

### 运算符技巧：
- `mid = l + r >> 1`（无括号，依赖优先级）
- `p ^= 1`（XOR 翻转代替 `p = 1 - p`）
- `~las` 检查非负（等价于 `las != -1`）
- `x = 1ll * x * x % mod, n >>= 1`（循环中的逗号运算符）

---

## 常用数据结构

### 线段树（带懒标记）：

```cpp
struct segtree {
    int n;
    struct node { int l, r; ll c, v, mx; } e[N << 2];
    #define l(p) e[p].l
    #define r(p) e[p].r
    #define v(p) e[p].v
    #define c(p) e[p].c
    #define mx(p) e[p].mx
    void up(int p) {
        v(p) = v(p << 1) + v(p << 1 | 1);
        mx(p) = max(mx(p << 1), mx(p << 1 | 1));
    }
    void bld(int p, int l, int r) {
        l(p) = l; r(p) = r; c(p) = -1;
        if (l == r) { v(p) = 0; return; }
        int mid = l + r >> 1;
        bld(p << 1, l, mid); bld(p << 1 | 1, mid + 1, r);
        up(p);
    }
    void psd(int p) {
        if (~c(p)) {
            v(p << 1) = 1ll * (r(p << 1) - l(p << 1) + 1) * c(p);
            mx(p << 1) = c(p);
            c(p << 1) = c(p);
            v(p << 1 | 1) = 1ll * (r(p << 1 | 1) - l(p << 1 | 1) + 1) * c(p);
            mx(p << 1 | 1) = c(p);
            c(p << 1 | 1) = c(p);
            c(p) = -1;
        }
    }
    void init(int _n) { n = _n; bld(1, 1, n); }
    void upd(int p, int ql, int qr, int v) {
        if (ql <= l(p) && r(p) <= qr) {
            v(p) = 1ll * (r(p) - l(p) + 1) * v;
            mx(p) = v;
            c(p) = v;
            return;
        }
        psd(p);
        int mid = l(p) + r(p) >> 1;
        if (ql <= mid) upd(p << 1, ql, qr, v);
        if (qr > mid) upd(p << 1 | 1, ql, qr, v);
        up(p);
    }
    P cnt(int p, int ql, int qr) {
        if (ql <= l(p) && r(p) <= qr) return P(v(p), mx(p));
        int mid = l(p) + r(p) >> 1;
        ll res = 0, w = 0;
        psd(p);
        if (ql <= mid) {
            auto L = cnt(p << 1, ql, qr);
            res += L.fi;
            w = max(w, L.se);
        }
        if (qr > mid) {
            auto R = cnt(p << 1 | 1, ql, qr);
            res += R.fi;
            w = max(w, R.se);
        }
        return P(res, w);
    }
} seg;
```

### 图（链式前向星）：

```cpp
struct edge { int v, nex, id; } e[2 * N];
int head[N], cnt;
void add(int u, int v, int id) {
    e[++cnt] = edge{v, head[u], id};
    head[u] = cnt;
}
// 遍历：
for (int i = head[u]; i; i = e[i].nex) {
    int v = e[i].v;
    // ...
}
```

### 模运算结构体（Z）：

```cpp
const int N = 4096 + 10, P = 998244353, mod = P;
const int Primitive_root = 3;
struct Z {
    int x;
    Z(const int _x = 0) : x(_x) {}
    Z operator +(const Z &r) const { return x + r.x < P ? x + r.x : x + r.x - P; }
    Z operator -(const Z &r) const { return x < r.x ? x - r.x + P : x - r.x; }
    Z operator -() const { return x ? P - x : 0; }
    Z operator *(const Z &r) const { return static_cast<ull>(x) * r.x % P; }
    Z operator +=(const Z &r) { return x = x + r.x < P ? x + r.x : x + r.x - P, *this; }
    Z operator -=(const Z &r) { return x = x < r.x ? x - r.x + P : x - r.x, *this; }
    Z operator *=(const Z &r) { return x = static_cast<ull>(x) * r.x % P, *this; }
    friend Z Pow(Z, int);
};
Z Pow(Z x, int y = P - 2) {
    Z ans = 1;
    for (; y; y >>= 1, x = x * x) if (y & 1) ans = ans * x;
    return ans;
}
```

### 组合数预处理：

```cpp
const int N = 3e6 + 10, mod = 998244353;
int Finv[N], fac[N], inv[N];
int modpow(int x, int n, int mod) {
    int res = 1;
    for (; n; x = 1ll * x * x % mod, n >>= 1)
        if (n & 1) res = 1ll * res * x % mod;
    return res;
}
void init(int n) {
    inv[1] = 1;
    for (int i = 2; i <= n; ++i) inv[i] = 1ll * (mod - mod / i) * inv[mod % i] % mod;
    fac[0] = Finv[0] = 1;
    for (int i = 1; i <= n; ++i) fac[i] = 1ll * fac[i - 1] * i % mod, Finv[i] = 1ll * Finv[i - 1] * inv[i] % mod;
}
int C(int n, int m) {
    if (m < 0 || m > n) return 0;
    return 1ll * fac[n] * Finv[n - m] % mod * Finv[m] % mod;
}
```

### 模加辅助函数：

```cpp
void add(int &x, int y) {
    x = (x + y) % mod;
}
```

---

## 算法模式

### DP（线性，O(n²)）：

```cpp
dp[0] = 1;
rep(i, 1, n) {
    per(j, i - 1, 0) {
        if (condition) {
            add(dp[i], dp[j]);
        }
    }
}
```

### 二分答案：

```cpp
int l = 1, r = n;
while (l <= r) {
    int mid = l + r >> 1;
    if (check(mid)) r = mid - 1;
    else l = mid + 1;
}
// 答案是 l
```

### 单调栈（下一个更大元素）：

```cpp
int stk[N], x = 0, l[N], r[N];
x = 0;
rep(i, 1, n) {
    while (x && a[stk[x]] <= a[i]) x--;
    if (!x) l[i] = 1;
    else l[i] = stk[x] + 1;
    stk[++x] = i;
}
```

### Dijkstra（最大堆中取负距离）：

```cpp
priority_queue<pair<int, int>> q;
for (int i = 0; i < n; ++i) q.push({-dist[i], i});
while (!q.empty()) {
    int v = q.top().second;
    q.pop();
    if (used[v]) continue;
    used[v] = true;
    for (auto [u, w] : g[v]) {
        if (dist[u] > dist[v] + w) {
            dist[u] = dist[v] + w;
            q.push({-dist[u], u});
        }
    }
}
```

### 点分治模式：

```cpp
void getrt(int u, int fa, bool op) {
    f[u] = 0; sz[u] = 1;
    for (int i = head[u]; i; i = e[i].nex) {
        int v = e[i].v;
        if (v == fa || vis[v]) continue;
        getrt(v, u, op);
        f[u] = max(f[u], sz[v]);
        sz[u] += sz[v];
    }
    if (op) {
        f[u] = max(f[u], siz - sz[u]);
        if (f[u] < f[rt]) rt = u;
    }
}
```

### 01-Trie（XOR 最大值）：

```cpp
int tr[M][2], tot, num[M];
void add(int x) {
    int rt = 0;
    for (int i = 20; i >= 0; --i) {
        int v = x >> i & 1;
        if (!tr[rt][v]) {
            tr[rt][v] = ++tot;
            num[tot] = 0;
            memset(tr[tot], 0, sizeof tr[tot]);
        }
        rt = tr[rt][v];
        num[rt]++;
    }
}
int query(int x) {
    int rt = 0, ans = 0;
    for (int i = 20; i >= 0; --i) {
        int j = (x >> i & 1) ^ 1;
        if (num[tr[rt][j]]) {
            ans |= (1 << i);
            rt = tr[rt][j];
        }
        else rt = tr[rt][1 - j];
    }
    return ans;
}
```

---

## 注释风格

### 中文行内注释解释算法：

```cpp
//枚举是在哪位首次出现不卡上界的数（自由元）的，类似数位dp
//f[i][0/1][0/1]表示考虑前i个数时...
```

### 注释掉的调试 printf：

```cpp
//printf("i:%d k:%d x:%d left:%d ans:%d\n", i, k, x, left, ans);
```

### 文件末尾的块注释（测试用例 / 推理）：

```cpp
/*
1 2 3
1 [3 2]
[2 1] 3
[3 2 1]
*/
```

### 解法函数顶部的多行推理：

```cpp
/*
1. 不要考虑去限制m次操作次数是否超限...
【约束A】：<mid的棍子根数小于等于(n+m-1)/2根
【约束B】：最小化>=mid的(n+m+1)/2的长度s
*/
```

---

## 独特技巧与惯用法

1. **`" \n"[j==m]`** — 空格分隔输出 + 末尾换行
2. **`~las`** — 按位取反检查 `las != -1`
3. **`p ^= 1`** — XOR 翻转代替 `p = 1 - p`
4. **`mid = l + r >> 1`** — 无括号，依赖 `>>` 优先级高于 `+`
5. **循环中的逗号运算符**：`x = 1ll * x * x % mod, n >>= 1`
6. **`#define int long long`** — 全局需要 64 位时使用（少用，配合 `int32_t main()`）
7. **`assert()`** — 比赛文件中验证约束
8. **在 main 中预处理邻接关系**（当结构静态时，在处理测试数据前预计算）
9. **`dbg(x)` / `dbg2(x)`** — 调试宏，输出到 `cerr`
10. **`#include<bits/stdc++.h>` 被注释掉** — 为编译速度刻意选择

---

## 高级数据结构

### 凸包优化（CHT）：

```cpp
struct CHT {
    struct line {
        int k, b;
        line() {}
        line(int k, int b) : k(k), b(b) {}
        double intersect(line l) {
            double db = l.b - b;
            double dk = k - l.k;
            return db / dk;
        }
        long long operator()(long long x) {
            return k * x + b;
        }
    };
    vector<double> x;
    vector<line> ll;
    void init(line l) {
        x.push_back(-inf);
        ll.push_back(l);
    }
    void addLine(line l) {
        while (ll.size() >= 2 && l.intersect(ll[ll.size() - 2]) <= x.back()) {
            x.pop_back();
            ll.pop_back();
        }
        if (!ll.empty()) x.push_back(l.intersect(ll.back()));
        ll.push_back(l);
    }
    long long query(long long qx) {
        int id = upper_bound(x.begin(), x.end(), qx) - x.begin();
        --id;
        return ll[id](qx);
    }
};
```

### NTT / 多项式乘法：

参见 `xiaosai/ssdut2025/ntt.cpp` 和 `xiaosai/ssdut2025/silly_tree.cpp`。关键特性：
- 预计算单位根
- 展开的 DFT 循环（size <= 4 特判，然后 size 2、size 4，再一般情况）
- 使用 `ull` 数组 `F[]` 作为中间结果避免反复取模
- `IDFT` 通过 `reverse` + `DFT` + 缩放完成

### 后缀数组（SA-IS）：

参见 `codeforces/goodbye2025/sais.cpp`。包含：
- SA-IS 构造
- LCP/height 数组
- 稀疏表 RMQ
- 滚动哈希集成

---

## 并查集：

```cpp
int fa[N];
int find(int x) {
    return fa[x] == x ? x : fa[x] = find(fa[x]);
}
void mer(int x, int y) {
    x = find(x), y = find(y);
    if (x != y) fa[x] = y;
}
```

---

## KMP（失配函数）：

```cpp
void kmppre(int *nex, int *b, int m) {
    int i = 0, j = nex[0] = -1;
    while (i < m) {
        while (j != -1 && b[i] != b[j]) j = nex[j];
        nex[++i] = ++j;
    }
}
```

---

## 何时使用哪种模板变体

- **变体 A**（完整模板，含 `map`、`set`）：题目需要有序容器时
- **变体 B**（精简模板，含 `cmath`、`algorithm`、`random`）：数学或几何题
- **`#define int long long`** 变体：所有算术都需要 64 位时
- **`bits/stdc++.h` 不注释**：编译速度不是问题时（罕见）
- **Fast I/O 命名空间**：N > 1e6 且输入是瓶颈时

---

## 典型的 main() 结构

```cpp
int main() {
    sci(t);
    while (t--) {
        sci(n);
        // 读入
        rep(i, 1, n) sci(a[i]);
        // 重置全局变量
        rep(i, 1, n) dp[i] = 0;
        // 求解
        pte(sol());
    }
    return 0;
}
```

或单组测试数据：

```cpp
int main() {
    sci(n);
    rep(i, 1, n) sci(a[i]);
    // 直接求解
    pte(ans);
    return 0;
}
```

---

## 核心风格规则总结

1. **4 空格缩进**，不使用 tab
2. **K&R 大括号** — 左大括号在同一行
3. **全部全局变量** — 不使用局部大数组
4. **1-indexed** — 通用约定
5. **`scanf`/`printf`** 优先于 `cin`/`cout`（除非使用了 `#define int long long`）
6. **`rep`/`per` 宏** 用于所有循环
7. **`pb`**、`fi`、`se` 用于 STL 操作
8. **`dbg(x)`** 用于调试，提交时注释掉
9. **中文注释** 用于算法解释
10. **紧凑代码** — 不添加多余空白，自然时一行多条语句
11. **不引入多余头文件** — 只引入需要的（但 `bits/stdc++.h` 注释掉作为备用）
12. **`const int N`** 始终带 `+10` 余量
