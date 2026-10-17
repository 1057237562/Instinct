# OJ Submission Wall

OJ Submission Wall 是一个给算法训练队、社团或小团队使用的做题统计墙。成员绑定自己的 OJ 账号后，系统会同步公开提交记录，生成类似 GitHub contributions 的训练绿墙，并展示最新提交、解题数、连续训练天数和参赛统计。

如果只是想直接使用，可以访问公开站点：[https://oj-train-wall.wannafly.cn](https://oj-train-wall.wannafly.cn)。

如果你想自建一套给自己的队伍使用，可以 fork 本仓库并按下面的说明部署。

## 功能

- 支持游客模式和用户名密码注册，注册不需要邮箱验证。
- 支持成员昵称、真实姓名、分组管理和成员详情页。
- 支持绑定 Codeforces、AtCoder、牛客、洛谷、VJudge、LOJ、LeetCode（国际站与中国站）。
- 支持按年或近 10 年查看训练绿墙。
- 支持最新提交列表、平台/用户/语言/状态/时间范围筛选和分页。
- 支持 First AC、Unique AC、AC Submissions、Platform Activity 四种统计口径。
- 支持按成员和平台查看难度分布、官方 Rating 历史，并导出 PNG / SVG 图片。
- 支持比赛统计，按平台和常见比赛类型聚合。
- 支持 2–5 人循环排名、3v3 队伍对战和双人详细对战；统一按共同比赛名次统计胜负和比赛内做题速度。双人指数优先采用官方 Rating 变化，无 Rating 时使用名次百分位，并结合相对胜负；同队成绩不进入指数曲线；Codeforces VP/场外参赛会换算等价名次，赛后补题不参与比较。
- 支持本地 SQLite 持久化和 HTTP 缓存，平台接口临时失败时保留上次成功数据。
- 无前端构建依赖，后端只使用 Python 标准库，适合 Docker 轻量部署。

## 快速开始

本地运行：

```bash
python3 app.py
```

打开 `http://localhost:8000`。

Docker 运行：

```bash
cp .env.example .env
docker compose up -d --build
```

数据默认保存在 `./data/ojwall.sqlite3`。升级时保留 `data/` 目录即可。

## 自部署

推荐生产环境放在 Nginx 或 Caddy 后面，并使用 HTTPS。
可直接照着 [生产服务器部署清单](deploy/production.md) 操作；下面是配置要点。

1. 准备域名解析

   在你的 DNS 服务商处添加一条 A 记录：

   | 主机记录 | 记录类型 | 记录值 |
   | --- | --- | --- |
   | `oj` 或你喜欢的子域名 | `A` | `<你的服务器公网 IP>` |

   例如你的域名是 `example.com`，可以把服务挂在 `https://oj.example.com`。

2. 拉取代码

   ```bash
   git clone https://github.com/<your-name>/oj-submission-wall.git
   cd oj-submission-wall
   cp .env.example .env
   ```

3. 修改 `.env`

   ```bash
   PUBLIC_BASE_URL=https://oj.example.com
   BIND_ADDRESS=127.0.0.1
   PORT=8000
   COOKIE_SECURE=true
   OJ_USER_AGENT=OJSubmissionWall/1.0 (+https://oj.example.com)
   LUOGU_USER_AGENT=OJSubmissionWall/1.0 (+https://oj.example.com)
   ```

4. 启动应用

   ```bash
   docker compose up -d --build
   docker compose ps
   ```

5. 配置反向代理

   仓库里提供了两份示例配置：

   - [deploy/nginx.oj-train-wall.conf](deploy/nginx.oj-train-wall.conf)：Nginx 示例
   - [deploy/Caddyfile.oj-train-wall](deploy/Caddyfile.oj-train-wall)：Caddy 示例

   这些文件使用公开站点域名作为示例。自部署时请把 `server_name`、站点域名和反代端口改成你自己的配置。

后续更新：

```bash
git pull --ff-only
docker compose up -d --build
```

升级到包含“训练洞察”的版本后，建议每个成员点击一次“刷新同步”，开始拉取 LeetCode 的公开统计；AtCoder 后续同步的记录会自动带入难度元数据。

## 账号绑定格式

- Codeforces：填写 handle，例如 `tourist`。
- AtCoder：填写用户名，例如 `tourist`。
- 牛客：填写个人或团队 profile 数字 ID / 链接。绑定个人账号后会查询关联团队，可勾选全部或部分作为独立账号添加；也可在账号框输入个人 ID 后点击“查找牛客关联团队”。
- 洛谷：填写用户名、数字 UID 或用户主页链接。
- VJudge：填写 VJudge 用户名。
- LOJ：填写 LOJ 用户名。
- LeetCode 国际站：填写 `/u/` 后的用户名或个人主页链接。
- LeetCode 中国站：填写 `cn:用户名` 或 `leetcode.cn` 个人主页链接。

LeetCode 公开接口只提供完整活动日历、题量/难度汇总和近期 AC，不提供完整逐题历史。训练洞察会把这些数据分别标为 `Platform Activity` 和可验证的近期 AC，不会拼成虚假的完整提交历史。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PUBLIC_BASE_URL` | 当前 Host | 应用公网地址 |
| `BIND_ADDRESS` | `0.0.0.0` | Docker 对宿主机暴露的绑定地址；反向代理部署建议设为 `127.0.0.1` |
| `PORT` | `8000` | 宿主机端口 |
| `DATA_DIR` | `/data` | SQLite 数据目录 |
| `COOKIE_SECURE` | `false` | HTTPS 部署建议设为 `true` |
| `SYNC_INTERVAL_SECONDS` | `900` | 后台同步间隔，设为 `0` 可关闭后台同步 |
| `SYNC_MIN_AGE_SECONDS` | `120` | 同一账号最短同步间隔 |
| `FETCH_LOOKBACK_DAYS` | `3650` | 首次或强制同步时回看天数 |
| `FETCH_LIMIT` | `1000` | 支持分页的平台单页拉取数量上限 |
| `CODEFORCES_VP_RANKS_PER_SYNC` | `4` | 每轮最多补齐的 Codeforces VP/场外参赛名次数；公开榜单摘要会持久缓存 |
| `HTTP_TIMEOUT_SECONDS` | `15` | 外部 OJ 单次请求超时时间 |
| `HTTP_RETRY_COUNT` | `2` | 外部 OJ 超时或 5xx 时的额外重试次数 |
| `HTTP_RETRY_BACKOFF_SECONDS` | `0.8` | 外部 OJ 重试退避基准秒数 |
| `DISPLAY_TZ_OFFSET_HOURS` | `8` | 榜单日期、连续天数和提交时间展示的时区偏移 |
| `SYNC_INCREMENTAL_OVERLAP_SECONDS` | `7200` | 增量同步按最新本地提交时间戳向前重叠的秒数，用于防漏和去重 |
| `CACHE_DIR` | `DATA_DIR/cache` | HTTP 响应缓存和概览镜像目录 |
| `HISTORICAL_CACHE_AFTER_DAYS` | `30` | 距今超过多少天的历史页可直接使用缓存 |
| `HISTORICAL_CACHE_TTL_SECONDS` | `315360000` | 历史页缓存有效期，默认约 10 年 |
| `OVERVIEW_CACHE_TTL_SECONDS` | `20` | `/api/overview` 页面组装结果的内存缓存秒数 |
| `BATTLE_MEMORY_CACHE_LIMIT` | `128` | 对战短缓存的最大双人组合数，防止多人排名组合过多占用内存 |
| `OVERVIEW_FEED_LIMIT` | `1000` | `/api/overview` 返回的最近提交明细条数上限，统计仍基于库内全量历史记录 |
| `OJ_USER_AGENT` | `OJSubmissionWall/1.0` | 外部 OJ 请求的 User-Agent，生产环境建议包含你的站点地址 |
| `LUOGU_USER_AGENT` | `OJSubmissionWall/1.0` | 洛谷请求的 User-Agent，生产环境建议包含你的站点地址 |
| `LUOGU_CF_CLEARANCE` | 空 | 可选：服务器同出口浏览器合法通过 Cloudflare 后拿到的 `cf_clearance` 值，不是登录态 |
| `LUOGU_COOKIE` | 空 | 可选：管理员自己的洛谷 Cookie；公开部署不建议收集用户 Cookie |
| `LUOGU_CSRF_TOKEN` | 空 | 可选：和 `LUOGU_COOKIE` 配套的洛谷 CSRF token，会由主站随代理请求头动态透传 |
| `LUOGU_PROXY_URL` | 空 | 可选：洛谷海外 403 时，把洛谷请求转发到可信的国内出口代理 |
| `LUOGU_PROXY_TOKEN` | 空 | 可选：访问洛谷私有代理的 Bearer token |
| `LUOGU_THIRD_PARTY_FALLBACK` | `true` | 洛谷主源失败时，是否尝试第三方公开统计源兜底 |
| `LUOGU_FALLBACK_URLS` | 内置公开卡片接口 | 可选：逗号分隔的洛谷降级 URL 模板，支持 `{uid}`、`{handle}`、`{name}` |
| `LUOGU_RECORD_SYNC` | `true` | 是否同步洛谷 `record/list` 逐条评测记录 |
| `LUOGU_RECORD_RECENT_PAGES_PER_SYNC` | `10` | 每次增量同步最多抓取洛谷最新记录页数，遇到早于增量时间戳的页面会提前停止 |
| `LUOGU_RECORD_BACKFILL_PAGES_PER_SYNC` | `8` | 洛谷历史记录每次额外回填页数 |
| `LUOGU_RECORD_SLEEP_MIN_SECONDS` | `0.4` | 洛谷记录页分页请求的最小间隔秒数 |
| `LUOGU_RECORD_SLEEP_MAX_SECONDS` | `1.4` | 洛谷记录页分页请求的最大间隔秒数 |
| `LUOGU_RECORD_INCREMENTAL_OVERLAP_SECONDS` | `7200` | 洛谷 `record/list` 增量同步的秒级重叠窗口，默认继承通用增量窗口 |

## 洛谷海外访问

洛谷可能会拦截海外机房出口，表现为 `HTTP 403`、验证码或 `record/list` 返回登录页。项目提供三层处理：

1. 优先使用公开个人页和练习页数据。
2. 如果公开页失败，尝试第三方公开统计卡片接口兜底总题数。
3. 如果你需要更精确的洛谷逐条提交记录，可以部署一个只代理洛谷请求的国内出口。

国内出口代理脚本是 [deploy/luogu_proxy.py](deploy/luogu_proxy.py)，FRP 连接示例见 [deploy/luogu-frp.md](deploy/luogu-frp.md)。代理只允许访问 `https://www.luogu.com.cn` / `https://luogu.com.cn`，并要求 Bearer token。不要把代理无鉴权暴露到公网，也不要把 Cookie、token 或服务器 IP 提交到仓库。

如果国内代理由别人维护，不需要把洛谷小号 Cookie 写进代理服务。把 Cookie/CSRF 留在海外主站即可：配置 `LUOGU_COOKIE` / `LUOGU_CSRF_TOKEN`，或在触发同步时通过 JSON 字段 `luoguCookie`、`luoguCsrfToken`，也可以用请求头 `X-Luogu-Cookie`、`X-Luogu-CSRF-Token` 临时传入。主站只会把它们作为本次同步任务的请求头转发给国内代理，不写入数据库。

页面渲染只读取本地 SQLite 中的历史提交记录；绑定账号、重试、刷新同步都会先进入后台队列，接口立即返回。后台线程按 `SYNC_INTERVAL_SECONDS` 定期做增量同步，并用每个绑定的 `last_sync_at` 记录最后成功更新时间。

慢速回填洛谷历史记录：

```bash
docker compose exec -T oj-submission-wall python app.py luogu-backfill \
  --pages-per-round 20 \
  --recent-pages 1 \
  --sleep-min 0.8 \
  --sleep-max 2.0
```

`--pages-per-round` 是每轮每个账号最多回填的历史页数，不是总页数上限。命令会持续按游标向后补，直到历史完成或某页失败暂停。

## 数据源说明

- Codeforces 使用官方 `user.status` API，并分页拉取历史提交；比赛统计使用 `contest.list` 的主站和 Gym 数据。
- AtCoder 使用 AtCoder Problems 公开 API；比赛统计使用 AtCoder 官方用户参赛历史 JSON。
- 洛谷优先读取公开个人页、练习页和 `record/list`，海外出口受限时可使用私有国内代理或第三方公开统计兜底。
- 牛客从公开 profile 和参赛历史接口同步提交与比赛；Rating 使用官网 `rating-history`，个人和每个团队独立绘图，不混入未计分比赛。已有绑定需要同步一次以获取新的完整 Rating 历史。
- VJudge 使用公开 `solveDetail2` 和 `status/data`。
- LOJ 使用公开 `submission/querySubmission` API。
- LeetCode 使用公开 GraphQL 接口；国际站同步活动、题量、难度、近期 AC 和比赛 Rating，中国站同步公开可用的活动、题量、难度和近期 AC。

如果平台接口改版、风控或临时不可用，系统会保留上次成功同步的数据。前端会先显示浏览器里的上次概览，再后台刷新最新数据，避免打开页面时闪成空列表。

## 生产建议

- 使用 HTTPS，并设置 `COOKIE_SECURE=true`。
- 把 `OJ_USER_AGENT` / `LUOGU_USER_AGENT` 改成你自己的站点地址或联系方式。
- 不要提交 `.env`、数据库、Cookie、token、服务器 IP 等敏感信息。
- 如果成员较多，适当调大 `SYNC_INTERVAL_SECONDS`，减少对 OJ 的请求压力。
- 定期备份 `data/ojwall.sqlite3`。

## License

MIT
