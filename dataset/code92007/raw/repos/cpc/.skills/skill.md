# Competitive Programming C++ Coding Style

## Overview

This document describes the coding style, conventions, and patterns of a competitive programmer. Use this as a reference to generate code that matches their exact style.

---

## File Structure

- Every solution is a **single self-contained `.cpp` file** — no `.h` headers, no multi-file projects
- Files named by problem letter: `a.cpp`, `b.cpp`, `c.cpp`, etc.
- Multiple attempts: `d.cpp`, `d2.cpp`, `d3.cpp`, `d4.cpp`
- Failed attempts: `e_fail.cpp`
- Algorithm-specific practice files: `sais.cpp`, `ntt.cpp`, `silly_tree.cpp`
- Contest directories organized by platform: `codeforces/`, `atcoder/`, `luogu/`, `nowcoder/`, etc.

---

## Standard Boilerplate (Template)

Every file begins with this exact header. There are two variants:

### Variant A — Full Template (most common, ~80% of files)

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

### Variant B — Minimal Template (simpler problems)

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

### Key notes about the template:
- `#include<bits/stdc++.h>` is **commented out** with explicit includes below it
- `typedef pair<int,int> P` — use `pair<ll,ll> P` when the problem needs 64-bit
- `typedef array<int,3> A` — use `array<ll,3> A` when needed
- `N` is set per problem based on constraints: `105`, `2e5+10`, `3e3+10`, `5e5+10`, `5e6+10`
- Always `+10` buffer for 1-indexed safety
- Add `const int mod=998244353;` when modular arithmetic is needed

---

## Macro Reference

| Macro | Expansion | Usage |
|---|---|---|
| `rep(i,a,b)` | `for(int i=(a);i<=(b);++i)` | Forward loop, inclusive |
| `per(i,a,b)` | `for(int i=(a);i>=(b);--i)` | Reverse loop, inclusive |
| `ll` | `long long` | 64-bit integer |
| `db` | `double` | Floating point |
| `ull` | `unsigned long long` | Unsigned 64-bit |
| `P` | `pair<int,int>` or `pair<ll,ll>` | Coordinate pairs, edges |
| `A` | `array<int,3>` or `array<ll,3>` | Triplets |
| `fi` / `se` | `.first` / `.second` | Pair accessors |
| `pb` | `.push_back()` | Vector append |
| `dbg(x)` | `cerr<<(#x)<<":"<<x<<" "` | Debug to stderr (no newline) |
| `dbg2(x)` | `cerr<<(#x)<<":"<<x<<endl` | Debug to stderr (with newline) |
| `SZ(a)` | `(int)(a.size())` | Safe size cast |
| `sci(a)` | `scanf("%d",&(a))` | Fast int input |
| `pt(a)` | `printf("%d",a)` | Int output |
| `pte(a)` | `printf("%d\n",a)` | Int output + newline |
| `ptlle(a)` | `printf("%lld\n",a)` | Long long output + newline |

---

## I/O Patterns

### Primary: scanf/printf (used in ~95% of files)

```cpp
int n, m;
sci(n), sci(m);  // comma operator for multiple inputs
ll x;
scanf("%lld", &x);
pte(ans);         // int + newline
ptlle(ans);       // long long + newline
```

### Space-separated output with trailing newline trick:

```cpp
rep(i, 1, n) {
    printf("%d%c", a[i], " \n"[i == n]);
}
```

### Fast I/O (only for extreme input sizes, N > 1e6):

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

### Interactive problems:

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

### Multi-testcase pattern:

```cpp
int t, n;
sci(t);
while (t--) {
    sci(n);
    // solve one testcase
    // reset globals between cases
}
```

---

## Variable Declaration

### ALL variables declared globally:

```cpp
const int N = 2e5 + 10;
int t, n, m, k, a[N], b[N], dp[N];
ll ans, sum;
vector<int> e[N];
```

### 1-indexed arrays (universal convention):

```cpp
rep(i, 1, n) sci(a[i]);  // always 1-indexed
```

### Reset pattern between test cases:

```cpp
while (t--) {
    sci(n);
    rep(i, 1, n) dp[i] = 0, e[i].clear();
    // solve
}
```

---

## Code Style

### Indentation: 4 spaces

### Brace style: K&R (opening brace on same line)

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

### Naming conventions:
- **Single-letter variables**: `n`, `m`, `k`, `t`, `x`, `y`, `u`, `v`, `w`, `p`, `q`, `l`, `r`
- **Common names**: `ans`, `res`, `cnt`, `sum`, `dp`, `a[]`, `b[]`
- **Descriptive for complex structures**: `segtree`, `CHT`, `SAIS`
- **Arrays named by problem variable**: `a[N]`, `dp[N][N]`

### Operator tricks:
- `mid = l + r >> 1` (no parentheses, relies on precedence)
- `p ^= 1` (XOR toggle instead of `p = 1 - p`)
- `~las` for checking non-negative (equivalent to `las != -1`)
- `x = 1ll * x * x % mod, n >>= 1` (comma operator in loop)

---

## Common Data Structures

### Segment Tree (with lazy propagation):

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

### Graph (chain forward star):

```cpp
struct edge { int v, nex, id; } e[2 * N];
int head[N], cnt;
void add(int u, int v, int id) {
    e[++cnt] = edge{v, head[u], id};
    head[u] = cnt;
}
// traversal:
for (int i = head[u]; i; i = e[i].nex) {
    int v = e[i].v;
    // ...
}
```

### Modular arithmetic struct (Z):

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

### Combinatorics precomputation:

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

### Modular add helper:

```cpp
void add(int &x, int y) {
    x = (x + y) % mod;
}
```

---

## Algorithm Patterns

### DP (linear, O(n²)):

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

### Binary search on answer:

```cpp
int l = 1, r = n;
while (l <= r) {
    int mid = l + r >> 1;
    if (check(mid)) r = mid - 1;
    else l = mid + 1;
}
// answer is l
```

### Monotonic stack (next greater element):

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

### Dijkstra (with negated distances in max-heap):

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

### Centroid decomposition pattern:

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

### 01-Trie (XOR max):

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

## Comment Style

### Chinese inline comments for algorithm explanation:

```cpp
//枚举是在哪位首次出现不卡上界的数（自由元）的，类似数位dp
//f[i][0/1][0/1]表示考虑前i个数时...
```

### Commented-out debug printf:

```cpp
//printf("i:%d k:%d x:%d left:%d ans:%d\n", i, k, x, left, ans);
```

### Block comments at end of file for test cases / reasoning:

```cpp
/*
1 2 3
1 [3 2]
[2 1] 3
[3 2 1]
*/
```

### Multi-line reasoning at top of solution:

```cpp
/*
1. 不要考虑去限制m次操作次数是否超限...
【约束A】：<mid的棍子根数小于等于(n+m-1)/2根
【约束B】：最小化>=mid的(n+m+1)/2的长度s
*/
```

---

## Unique Tricks and Idioms

1. **`" \n"[j==m]`** — space-separated output with trailing newline
2. **`~las`** — bitwise NOT for checking `las != -1`
3. **`p ^= 1`** — XOR toggle instead of `p = 1 - p`
4. **`mid = l + r >> 1`** — no parentheses, relies on `>>` having higher precedence than `+`
5. **Comma operator in loops**: `x = 1ll * x * x % mod, n >>= 1`
6. **`#define int long long`** — for problems needing global 64-bit (used sparingly, then `int32_t main()`)
7. **`assert()`** — constraint validation in contest files
8. **Precomputed adjacency in main** before processing test cases (when structure is static)
9. **`dbg(x)` / `dbg2(x)`** — debug macros outputting to `cerr`
10. **`#include<bits/stdc++.h>` commented out** — deliberate choice for compilation speed

---

## Advanced Data Structures

### Convex Hull Trick (CHT):

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

### NTT / Polynomial multiplication:

See `xiaosai/ssdut2025/ntt.cpp` and `xiaosai/ssdut2025/silly_tree.cpp` for the full implementation. Key features:
- Precomputed roots of unity
- Unrolled DFT loops (size <= 4 special case, then size 2, size 4, then general)
- Uses `ull` array `F[]` for intermediate results to avoid repeated modulo
- `IDFT` done via `reverse` + `DFT` + scaling

### Suffix Array (SA-IS):

See `codeforces/goodbye2025/sais.cpp` for the full implementation. Includes:
- SA-IS construction
- LCP/height array
- RMQ via sparse table
- Rolling hash integration

---

## DSU / Union-Find:

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

## KMP (failure function):

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

## When to use which template variant

- **Variant A** (full template with `map`, `set`): when the problem needs ordered containers
- **Variant B** (minimal with `cmath`, `algorithm`, `random`): for math-heavy or geometry problems
- **`#define int long long`** variant: when all arithmetic needs to be 64-bit globally
- **`bits/stdc++.h` uncommented**: when compilation speed is not a concern (rare)
- **Fast I/O namespace**: when N > 1e6 and input is the bottleneck

---

## Typical main() structure

```cpp
int main() {
    sci(t);
    while (t--) {
        sci(n);
        // read input
        rep(i, 1, n) sci(a[i]);
        // reset globals
        rep(i, 1, n) dp[i] = 0;
        // solve
        pte(sol());
    }
    return 0;
}
```

Or for single-testcase problems:

```cpp
int main() {
    sci(n);
    rep(i, 1, n) sci(a[i]);
    // solve directly
    pte(ans);
    return 0;
}
```

---

## Summary of Key Style Rules

1. **4-space indentation**, no tabs
2. **K&R braces** — opening brace on same line
3. **All globals** — no local large arrays
4. **1-indexed** — universal convention
5. **`scanf`/`printf`** over `cin`/`cout` (except when `#define int long long` is used)
6. **`rep`/`per` macros** for all loops
7. **`pb`**, `fi`, `se` for STL operations
8. **`dbg(x)`** for debugging, commented out in final submission
9. **Chinese comments** for algorithm explanations
10. **Compact code** — no unnecessary whitespace, multiple statements on one line when natural
11. **No unnecessary includes** — only what's needed (but `bits/stdc++.h` commented out as backup)
12. **`const int N`** always with `+10` buffer
