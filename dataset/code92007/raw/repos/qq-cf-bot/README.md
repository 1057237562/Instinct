# CF Bot：QQ + Web Codeforces 训练场

同一套抽题、中文题面、远端代码判题、做法审核和 Rating 系统，同时支持 OneBot v11 QQ 群机器人与网站。QQ 和 Web 是两个接入层，共用 `ChallengeService` 业务服务和 SQLite 数据；移除任一接入层不会影响另一端。代码判题载具可选择 VJudge 或 Codeforces 直提。

网站入口直接由机器人进程提供，默认是 `http://127.0.0.1:8088/`；OneBot 反向上报地址仍是 `/onebot`，健康检查仍是 `/health`。

## 功能

### 网站

- 本地账号注册、登录和安全会话。
- 每个用户独立抽题、去重和难度区间，不会抢占其他人的当前题。
- 题目进行中隐藏题号、标题、实际 Rating 和标签，通过或放弃后揭晓。
- 支持口头做法审核与 C++、C、Java、Python 代码提交。
- 网站用户共享站内榜单；QQ 群继续使用各群自己的榜单。

### QQ 群

- `/new`：按本群默认难度推送一道题，默认 `1900-2600`。
- `/new 2100 2400`：临时按 `2100-2400` 推一道题，不修改群默认配置。
- `/new 1200`：临时按精确 `1200` rating 推一道题。
- `/new 1937 1956`：会规范到 `1900-2000`，范围下界向下取整、上界向上取整到 100 档。
- `/share 1704F`：分享指定 Codeforces 题目，不计入常规榜单。
- `/cfset rating 1900 2600`：设置本群默认题目难度。
- `/rating`：查看本群默认题目难度。
- `/cur`：重新发送当前题目。
- `/giveup`：放弃当前题目；默认新题发布 120 秒后才能放弃，该题仍计入已推送，不会重复。
- `/submitcode + 代码`：自动识别 C++/C/Java/Python 并提交当前题代码到 Codeforces，返回 AC/WA/TLE/CE 等 verdict。
- `/ranklist`：发送群内榜单图片。
- `/submit 做法`：大模型口头做法审核，默认开启。审核前会优先读取题解库，用题面、洛谷公开题解、CF 题解链接/内容和可选 AC 代码片段校对做法。

## 部署前准备

1. 准备一个 OneBot v11 网关，例如 NapCat、Lagrange.OneBot 或其他兼容实现。
2. OneBot HTTP API 示例：`http://127.0.0.1:3000`。
3. OneBot 反向 HTTP 上报地址填：`http://机器人服务器:8088/onebot`。
4. 准备一个专用 Codeforces 练习账号。不要用主号，避免账号风控影响正常使用。

## Docker 部署

```bash
cp .env.example .env
# 编辑 .env，填 ONEBOT_HTTP_URL、BOT_ALLOWED_GROUPS、CF_* 等配置
./scripts/deploy.sh
```

健康检查：

```bash
curl http://127.0.0.1:8088/health
```

浏览器打开 `http://服务器地址:8088/` 即可使用网站。公网部署建议只让 Nginx/Caddy 暴露 80/443，应用的 8088 端口保持在本机或内网。

网站单题训练带渐进提示。用户可在开题前设置算法标签与 Hint 1 的解锁时间、后续 Hint 间隔、Hint 总数（1-6 条），默认依次为 10、20、10 分钟和 6 条。未到常规解锁时间也可主动提前查看；提示只在点击后展示并记录。单题还可设置 20-240 分钟的自愿时限，到点后服务端会强制结束本轮。VP 同样按每道题首次打开时间和本场设置解锁提示，但不启用单题强制放弃时限。

## 部署到 cf-bot.wannafly.cn

1. 将 `cf-bot.wannafly.cn` 的 DNS A/AAAA 记录指向服务器。
2. 在 `.env` 中设置 `WEB_ENABLED=true`、`WEB_COOKIE_SECURE=true`、`BOT_PUBLISH_HOST=127.0.0.1`，并保留已有 QQ 与判题配置。
3. 执行 `./scripts/deploy.sh qq-cf-bot`。
4. 已使用 Caddy 的服务器把 `deploy/Caddyfile.cf-bot` 追加到 `/etc/caddy/Caddyfile` 并 reload；否则可参考 `deploy/nginx/cf-bot.wannafly.cn.conf.example` 配置 Nginx。

反向代理只需转发到 `http://127.0.0.1:8088`。网站与 `/onebot` 共用端口，因此原 NapCat 上报 URL 无需修改。

`oj-train-wall.wannafly.cn` 所在服务器可直接复用现有 Caddy：两个站点分别反代到 `127.0.0.1:8017` 和 `127.0.0.1:8088`，Caddy 会按域名自动申请和续期 HTTPS 证书。

## 架构

```text
QQ / OneBot adapter ─┐
                     ├─ ChallengeService ─ 抽题 / 判题 / Rating ─ SQLite
Web UI + JSON API ───┘                 └─ Codeforces / 洛谷 / LLM
```

网站账号拥有独立的题目作用域，站内榜单使用单独的共享作用域；QQ 数据仍按群号隔离。两个入口共用同一个远端判题账号和全局提交间隔，避免并发提交触发平台风控。

`scripts/deploy.sh` 会先执行 Tailscale preflight，再执行 `docker compose up -d --build`。如果你不需要 Tailscale，把 `.env` 里的 `TAILSCALE_REQUIRED=false` 留着即可。

OneBot 和机器人在同一台机器、但机器人跑在 Docker 容器里时，`ONEBOT_HTTP_URL` 通常填 `http://host.docker.internal:3000`；`docker-compose.yml` 已经映射了这个宿主机地址。如果 OneBot 在另一台机器，填 OneBot HTTP API 的可访问地址。

## 推荐稳定架构：国内 NapCat + 服务器 bot

如果服务器在海外，QQ/NapCat 容易被判定为异地或异常登录，表现为 NapCat 日志里出现 `KickedOffline`、`账号状态变更为离线`、`账号当前登录已失效`。这种情况重启 watchdog 只能帮你尽快发现和重登，不能从根上解决风控。

更稳的部署方式是把 QQ 登录环境从业务服务器里拿出来：

- 国内稳定 IP 机器：只运行 NapCat/QQ 登录态。
- 业务服务器：只运行 `qq-cf-bot`、SQLite 数据和 Codeforces/LLM 逻辑。
- 两台机器用 Tailscale 或内网互通，不把 NapCat API 暴露到公网。

服务器先查看自己的 Tailscale IP：

```bash
tailscale ip -4
```

在国内机器上运行 NapCat 后，WebUI 里建议这样配置：

- HTTP 服务器：Host `0.0.0.0`，Port `3000`，Token 填一个长随机串。
- HTTP 客户端：URL `http://<服务器Tailscale-IP>:8088/onebot`，消息格式 `Array`，Token 填同一个长随机串。
- 图片发送建议保持 `ONEBOT_IMAGE_MODE=base64`，两台机器不需要共享图片目录。

服务器 `.env` 配成访问国内 NapCat：

```env
ONEBOT_HTTP_URL=http://<国内NapCat的Tailscale-IP>:3000
ONEBOT_ACCESS_TOKEN=<NapCat HTTP服务器 Token>
ONEBOT_EVENT_ACCESS_TOKEN=<NapCat HTTP客户端 Token>
ONEBOT_IMAGE_MODE=base64
```

部署并检查链路：

```bash
cd ~/qq-cf-bot
./scripts/deploy.sh qq-cf-bot
./scripts/check-onebot-link.sh
```

`check-onebot-link.sh` 会检查 bot 健康状态、NapCat 登录账号、NapCat 是否能看到允许群，以及最近 bot/NapCat 日志里有没有收消息、拒绝 token 或掉线信号。

## 本地运行

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
playwright install chromium
python -m qq_cf_bot
```

如果系统 Python 低于 3.10，建议直接使用 Docker；Dockerfile 默认使用 Python 3.11。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ONEBOT_HTTP_URL` | `http://127.0.0.1:3000` | OneBot HTTP API 地址 |
| `ONEBOT_ACCESS_TOKEN` | 空 | OneBot access token |
| `ONEBOT_EVENT_ACCESS_TOKEN` | 空 | 可选，校验 NapCat HTTP 客户端上报到 `/onebot` 的 token；远端 NapCat 部署时建议设置 |
| `ONEBOT_IMAGE_MODE` | `base64` | 图片发送方式；`base64` 不要求 OneBot 读取本地文件 |
| `ONEBOT_SELF_ID` | 空 | 可选，机器人自己的 QQ 号；为空时自动调用 OneBot `get_login_info` 获取，用于先私聊自己再合并转发题面 |
| `BOT_HOST` | `127.0.0.1` | 机器人监听地址；Docker 中为 `0.0.0.0` |
| `BOT_PORT` | `8088` | 机器人监听端口 |
| `BOT_PUBLISH_HOST` | `0.0.0.0` | Docker 发布端口绑定地址；只经本机 Nginx 反代时设为 `127.0.0.1` |
| `WEB_ENABLED` | `true` | 是否同时提供网站与 JSON API |
| `WEB_REGISTRATION_ENABLED` | `true` | 是否允许网站创建新账号 |
| `WEB_COOKIE_SECURE` | `false` | HTTPS 生产部署必须设为 `true` |
| `WEB_SESSION_HOURS` | `720` | 网站登录会话有效小时数 |
| `WEB_MAX_BODY_BYTES` | `256000` | 网站 JSON 请求体上限 |
| `BOT_ALLOWED_GROUPS` | 空 | 允许使用的群号，逗号分隔；空表示所有群 |
| `BOT_DATA_DIR` | `data` | SQLite、题库缓存和图片输出目录 |
| `CF_MIN_RATING` | `1900` | 默认最低题目 rating |
| `CF_MAX_RATING` | `2600` | 默认最高题目 rating |
| `CF_BASE_URLS` | `https://codeforces.com,https://m1.codeforces.com,https://m2.codeforces.com,https://m3.codeforces.com` | Codeforces 主站和官方镜像候选；题库 API、题面抓取和提交状态查询会按顺序降级 |
| `BOT_DEDUP_SCOPE` | `group` | `group` 每群去重，`global` 全局去重 |
| `BOT_PREFETCH_ENABLED` | `true` | 是否在当前题发布后后台预制下一题，加快下一次 `/new` |
| `CF_RECENT_SELECTION_POOL_SIZE` | `500` | 推题时优先从较新 contestId 的候选池中随机，数值越大越分散 |
| `GIVEUP_MIN_SECONDS` | `120` | 新题发布后至少等待多少秒才能 `/giveup` |
| `FALLBACK_STATEMENT_SOURCE` | `codeforces` | 洛谷中文题面失败时，回退到 Codeforces 官方英文题面 |
| `CODE_SUBMIT_PROVIDER` | `auto` | 远端代码判题载具：`vjudge`、`codeforces` 或 `auto`；`auto` 优先选已配置的 VJudge |
| `CODE_SUBMIT_ENABLED` | `auto` | 是否开启 `/submitcode` 代码判定；`auto` 表示远端载具或模型兜底至少一项配置完整时自动开启 |
| `CODE_SUBMIT_LLM_FALLBACK` | `true` | 远端未配置、提交失败或 verdict 轮询超时时，使用 `JUDGE_*` 模型静态审核源码 |
| `VJUDGE_USERNAME` | 空 | VJudge 登录用户名；只放服务器 `.env` |
| `VJUDGE_PASSWORD` | 空 | VJudge 登录密码；只放服务器 `.env`，不要提交到 GitHub |
| `VJUDGE_COOKIE` | 空 | 可选，账号密码登录遇到 Turnstile 时，填写浏览器登录后的完整 `Cookie` 请求头 |
| `VJUDGE_LANGUAGE_ID` | 空 | 可选，强制使用 VJudge 提交页的语言 ID；默认从当前题提交配置自动识别 |
| `VJUDGE_BASE_URL` | `https://vjudge.net` | VJudge 服务地址 |
| `CF_SUBMIT_ENABLED` | `auto` | 旧版兼容开关；未设置 `CODE_SUBMIT_ENABLED` 时仍生效 |
| `CF_USERNAME` | 空 | Codeforces 登录账号或邮箱 |
| `CF_PASSWORD` | 空 | Codeforces 密码，只放在服务器 `.env`，不要提交到 GitHub |
| `CF_HANDLE` | `CF_USERNAME` | Codeforces handle，用于轮询提交记录 |
| `CF_SUBMIT_DEFAULT_LANGUAGE` | `cpp` | `/submitcode` 未指定语言时的默认语言 |
| `CF_SUBMIT_LANGUAGE_ID` | 空 | 可选，强制使用 CF 表单里的语言 ID |
| `CF_SUBMIT_MIN_INTERVAL_SECONDS` | `180` | 全局最小提交间隔，避免短期大量提交 |
| `CF_SUBMIT_POLL_INTERVAL_SECONDS` | `8` | 轮询 verdict 间隔 |
| `CF_SUBMIT_POLL_TIMEOUT_SECONDS` | `180` | 单次提交最长等待 verdict 时间 |
| `CF_AUTO_SUBMIT_DIRECT_CODE` | `false` | 是否自动识别群里的裸代码并提交，建议保持关闭 |
| `JUDGE_ENABLED` | `true` | 是否开启 `/submit` 口头做法审核 |
| `JUDGE_API_URL` | `https://api.openai.com/v1/chat/completions` | OpenAI-compatible 模型服务地址；可填完整 endpoint 或 provider base URL |
| `JUDGE_API_KEY` | 空 | 口头做法审核模型 key |
| `JUDGE_MODEL` | 空 | 口头做法审核模型名 |
| `JUDGE_WIRE_API` | 自动 | 模型接口协议；支持 `chat_completions`、`responses`、`responses_stream` 和 `responses_websocket` |
| `JUDGE_PROVIDERS` | 空 | 可选 JSON provider 队列；为空时使用 `JUDGE_API_*` 单 provider |
| `JUDGE_STATEMENT_MAX_CHARS` | `12000` | 单次判题传给模型的题面最大字符数 |
| `JUDGE_SOLUTION_CONTEXT_MAX_CHARS` | `10000` | 单次判题传给模型的题解库上下文最大字符数 |
| `JUDGE_CODE_MAX_CHARS` | `100000` | 代码兜底静态审核时传给模型的源码最大字符数 |
| `TAILSCALE_REQUIRED` | `false` | `true` 时 `scripts/deploy.sh` 会强制检查 Tailscale 已启动并已登录 |
| `TAILSCALE_AUTHKEY` | 空 | 可选，服务器自动 `tailscale up` 用的一次性 auth key；不要提交到 Git |
| `TAILSCALE_HOSTNAME` | `qq-cf-bot` | 服务器加入 tailnet 时显示的设备名 |
| `TAILSCALE_EXTRA_ARGS` | 空 | 可选，传给 `tailscale up` 的额外参数，例如 `--accept-routes` |
| `TAILSCALE_PING_HOST` | 空 | 可选，部署前必须能 ping 通的 tailnet 主机；为空时会从 `JUDGE_API_URL` 自动识别 `100.*` 或 `*.ts.net` |
| `TAILSCALE_PING_ATTEMPTS` | `4` | Tailscale reachability 检查最多尝试次数 |
| `TAILSCALE_PING_RETRY_DELAY_SECONDS` | `5` | Tailscale reachability 检查失败后的重试间隔秒数 |
| `TRANSLATE_ENABLED` | `true` | 是否用 OpenAI-compatible 模型把 CF 英文题面翻译成中文；未配置 key/model 时不会发起请求 |
| `TRANSLATE_API_URL` | `JUDGE_API_URL` | 翻译模型 API；为空时复用 `JUDGE_API_URL` |
| `TRANSLATE_API_KEY` | `JUDGE_API_KEY` | 翻译模型 key；为空时复用 `JUDGE_API_KEY` |
| `TRANSLATE_MODEL` | `JUDGE_MODEL` | 翻译模型名；为空时复用 `JUDGE_MODEL` |
| `TRANSLATE_WIRE_API` | `JUDGE_WIRE_API` | 翻译模型接口协议；为空时复用判题模型协议 |
| `TRANSLATE_PROVIDERS` | 空 | 可选 JSON provider 队列；为空时使用 `TRANSLATE_API_*` 单 provider |
| `TRANSLATE_TIMEOUT_SECONDS` | `60` | 单次翻译请求超时 |
| `TRANSLATE_MAX_CHARS` | `60000` | 单次翻译传给模型的题面最大字符数 |
| `SOLUTION_BANK_ENABLED` | `true` | 是否开启题解库缓存 |
| `SOLUTION_BANK_MIN_REFS` | `1` | 每题至少缓存多少条参考材料后不再主动抓取 |
| `SOLUTION_BANK_MAX_REFS` | `4` | 每题最多缓存/提供多少条参考材料 |
| `SOLUTION_BANK_MAX_REF_CHARS` | `5000` | 每条参考材料最大缓存字符数 |
| `SOLUTION_BANK_FETCH_LUOGU` | `true` | 是否抓取洛谷公开题解 |
| `SOLUTION_BANK_FETCH_CF_EDITORIAL` | `true` | 是否抓取 Codeforces 题解/教程链接和内容 |
| `SOLUTION_BANK_FETCH_CF_AC_CODE` | `false` | 是否抓取少量 CF AC 代码；需要 CF 登录态，默认关闭 |
| `SOLUTION_BANK_GENERATE_LLM` | `true` | 公开题解不足时，是否用判题模型生成内部参考解法并缓存 |
| `NAPCAT_WATCHDOG_ENABLED` | `false` | 是否启用宿主机 NapCat 掉线检测脚本 |
| `NAPCAT_WATCHDOG_ONEBOT_URL` | `http://127.0.0.1:3000` | watchdog 调用 NapCat OneBot HTTP API 的地址 |
| `NAPCAT_WATCHDOG_DOCKER_SERVICE` | `napcat` | 宿主机直连失败时，用这个 docker compose 服务名自动发现 NapCat 容器 IP |
| `NAPCAT_WATCHDOG_DOCKER_PORT` | `3000` | NapCat 容器内 OneBot HTTP API 端口 |
| `NAPCAT_WATCHDOG_INTERVAL_SECONDS` | `60` | watchdog 检查间隔 |
| `NAPCAT_WATCHDOG_RESTART_COOLDOWN_SECONDS` | `300` | 两次自动重启 NapCat 的最小间隔 |
| `NAPCAT_WATCHDOG_RESTART_CMD` | `./scripts/restart-napcat.sh` | watchdog 判定离线后执行的重启命令 |
| `NAPCAT_WATCHDOG_LOG_CHECK_ENABLED` | `true` | 是否额外扫描 NapCat 最近日志，识别账号离线、登录失败、安全风险等 API 假健康场景 |
| `NAPCAT_WATCHDOG_LOG_SCAN_SECONDS` | `600` | 每次扫描最近多少秒 NapCat 日志 |
| `NAPCAT_WATCHDOG_NOTIFY_EMAIL` | 空 | NapCat 离线/重启通知收件邮箱 |
| `SMTP_HOST` / `SMTP_PORT` | `smtp.qq.com` / `465` | watchdog 邮件通知 SMTP 服务 |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | 空 | SMTP 登录用户名和授权码 |
| `SMTP_FROM` | `SMTP_USERNAME` | 邮件发件人 |

## 远端代码判题与 `/submitcode`

服务器无法稳定访问 Codeforces 提交页时，推荐使用 VJudge 载具：

```dotenv
CODE_SUBMIT_PROVIDER=vjudge
CODE_SUBMIT_ENABLED=true
CODE_SUBMIT_LLM_FALLBACK=true
VJUDGE_USERNAME=你的_VJudge_用户名
VJUDGE_PASSWORD=你的_VJudge_密码
VJUDGE_COOKIE=
```

这里使用的是 VJudge 练习题提交接口和默认远端账号（`method=0`），题号仍为 `CodeForces-{contestId}{index}`。不需要在 VJudge 绑定 `toolist`，原来的 `CF_USERNAME`、`CF_PASSWORD` 可以保留，切换回 `CODE_SUBMIT_PROVIDER=codeforces` 时继续使用。

开启 `CODE_SUBMIT_LLM_FALLBACK` 后，远端载具未配置、请求失败或最终结果一直处于 `PENDING` 时，会改用 `JUDGE_*` 模型静态审核源码。只要远端已经返回 WA、CE、TLE 等终态 verdict，就不会调用模型改判。兜底结果固定标记为 `LLM_ACCEPTED` 或 `LLM_REJECTED`；前者会按现有规则结算题目和 Rating，但它不是实际运行结果，不能等同于 Codeforces AC。若不希望模型结果影响榜单，请把该变量设为 `false`。

修改服务器 `.env` 后执行：

```bash
chmod 600 .env
docker compose up -d --build qq-cf-bot
docker compose exec -T qq-cf-bot python -m qq_cf_bot.check_submit
```

如果自检提示 VJudge 要求人机验证：

1. 在自己的浏览器登录 [VJudge](https://vjudge.net/)。
2. 打开开发者工具的 Network，刷新任意 VJudge 页面，选择发往 `vjudge.net` 的请求。
3. 从 Request Headers 复制完整 `Cookie` 值，填入服务器 `.env` 的 `VJUDGE_COOKIE`，建议写成 `VJUDGE_COOKIE='JSESSIONID=...; ...'`。Cookie 与密码同等敏感，不要发到聊天或提交进 Git。
4. 再执行上面的重建和自检命令。登录会话还会持久化到 `data/vjudge-session/http-cookies.txt`。

推荐使用代码块：

````text
/submitcode
```cpp
#include <bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    return 0;
}
```
````

也支持不带代码块或不写语言：

```text
/submitcode
#include <bits/stdc++.h>
using namespace std;
int main() { return 0; }
```

注意：

- `/submitcode` 只提交当前群的当前题。
- 提交队列是全局单线程，默认至少间隔 180 秒。
- 语言会按代码风格自动识别，主要支持 C++、C、Java、Python；识别不出时按 `CF_SUBMIT_DEFAULT_LANGUAGE`。
- VJudge 或 Codeforces 可能触发验证码、二次验证或账号安全确认，此时远端提交会给出对应的登录/Cookie 提示。
- 大模型兜底只做静态审核，可能漏掉隐藏边界或运行时问题；页面和 QQ 消息会明确显示其非官方性质。
- 不要在正在进行的正式比赛中使用该机器人提交代码。

Docker 部署后先运行一次无提交登录自检：

```bash
docker compose exec -T qq-cf-bot python -m qq_cf_bot.check_submit
```

有远端载具时，自检只验证当前所选载具的登录，不提交代码；仅配置模型兜底时，自检会确认兜底配置完整。VJudge 会话保存在 `data/vjudge-session/`；Codeforces 直提的 HTTP Cookie 与持久浏览器资料仍保存在 `data/codeforces-session/`。后续容器重建会复用这些会话。

## `/new` 和 `/share`

`/new` 的 rating 参数会做严格校验，只接受单个整数、两个整数、逗号范围或短横线范围。两个端点会先按大小排序，再按 Codeforces 的 100 rating 档规范化：下界向下取整，上界向上取整。例如 `/new 1937 1956` 等价于 `/new 1900 2000`。带额外文本的参数会被拒绝，避免把聊天内容误当成参数。

`/share 1704F`、`/share CF1704F`、`/share https://codeforces.com/contest/1704/problem/F` 都会分享同一道指定题。分享题也会成为当前题，支持 `/cur`、`/giveup`、`/submit` 和 `/submitcode`，但通过后不会增加 solved 数、不会涨榜单 Rating，也不会占用普通 `/new` 的去重记录。

## `/ranklist` 榜单

榜单不是按单纯通过数量排序，也不再按传统 Elo 累加排序。每道通过题会按 rating 换算成指数分，默认约为“每差 200 rating，题目权重差 4 倍”：例如 1 道 2600 约等于 2 道 2500 或 4 道 2400。这样能奖励做高难题，也允许大量中高难题合理追分，不会被纯刷水题轻易顶掉。

## `/submit` 和题解库

`/submit` 适合群友提交口头做法、复杂度和关键边界处理。判定口径是：核心算法、关键状态/性质和复杂度说清楚，足以让熟练选手补出实现即可；不会要求代码级细节全部写完。机器人会：

1. 读取当前题的缓存题解库。
2. 如果题解库不足且之前没抓过，尝试抓取洛谷公开题解和 Codeforces 题解/教程。
3. 如果 `SOLUTION_BANK_FETCH_CF_AC_CODE=true` 且 CF 账号可登录，再限量抓取 AC 代码片段。
4. 如果公开材料仍不足且 `SOLUTION_BANK_GENERATE_LLM=true`，先让判题模型独立生成一份内部参考解法并缓存。
5. 把题面、受限长度的参考材料和本题历史口胡记录交给大模型一审。
6. 如果一审通过且题解库已有参考材料，再用参考材料做二审复核，只拦截明显错误，不因为表述不像官方题解而驳回。

题解库保存在 `data/bot.sqlite3`，一题可以对应多条题解，后续同一题不会重复抓取；模型生成的内部参考解法也会作为 `llm_generated` 缓存。模型只用于做法审核和内部参考解法生成，不用于 `/submitcode` 的最终 verdict。为避免提前暴露题目来源，当前题进行中不会公开题号或题解；`/giveup`、口胡通过或代码 AC 后才会解锁题目信息和已缓存参考材料。

### 接入大模型判题

`/submit` 调用的是统一的大模型 JSON 输出能力，当前支持四种 OpenAI-compatible 协议：

- `chat_completions`：传统 `/chat/completions` HTTP POST。
- `responses`：`/responses` HTTP POST；如果服务端返回 `426 WebSocket upgrade required`，机器人会自动改走 SSE 流式 Responses。
- `responses_stream`：Codex CLI 常用的 `/responses` HTTP SSE 流式接口，发送 `stream: true` 和 `Accept: text/event-stream`。
- `responses_websocket`：Codex 风格的 `/responses` WebSocket，首帧发送顶层 `response.create`。

普通 OpenAI-compatible Chat Completions 服务可以这样配：

```env
JUDGE_ENABLED=true
JUDGE_WIRE_API=chat_completions
JUDGE_API_URL=https://api.openai.com/v1/chat/completions
JUDGE_API_KEY=sk-...
JUDGE_MODEL=gpt-...
```

如果要复用 Codex 的模型服务配置，把 `~/.codex/config.toml` 里当前 provider 的 `base_url` 填到 `JUDGE_API_URL`，把 `wire_api` 填到 `JUDGE_WIRE_API`，把 `model` 填到 `JUDGE_MODEL`。例如 Codex provider 写的是 `wire_api = "responses"` 时：

```env
JUDGE_ENABLED=true
JUDGE_WIRE_API=responses
JUDGE_API_URL=http://<codex-provider-base-url>
JUDGE_API_KEY=<同一模型服务可接受的 token>
JUDGE_MODEL=<codex-model>
```

如果这个 provider 的 `/responses` 返回 `WebSocket upgrade required`，可以直接显式改成：

```env
JUDGE_WIRE_API=responses_stream
JUDGE_API_URL=http://<codex-provider-base-url>
```

`TRANSLATE_API_*` 默认复用 `JUDGE_API_*`，所以只要判题模型配好了，CF 英文兜底题面和洛谷英文标题也会自动走同一个模型翻译。若模型服务只暴露在 Tailscale 内网地址上，需要先让机器人服务器加入同一个 tailnet，再在服务器上验证 `curl http://<codex-provider-base-url>/health` 或模型服务自己的健康检查地址能通。

如果以后有多个模型 provider，可以用 JSON 数组配置 fallback；当前只有一个 Codex provider 时不用填：

```env
JUDGE_PROVIDERS=[{"name":"codex","api_url":"http://host.docker.internal:18080","api_key":"...","model":"gpt-5.5","wire_api":"responses_stream"}]
```

服务器接入 Tailscale 的典型流程：

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled
sudo tailscale up --authkey tskey-... --hostname qq-cf-bot
tailscale status
```

`tskey-...` 建议使用 Tailscale 后台生成的一次性 auth key，只放在服务器命令行或部署密钥里，不要写进 Git 仓库。

为了以后不忘记启动 Tailscale，服务器 `.env` 里建议加：

```env
TAILSCALE_REQUIRED=true
TAILSCALE_AUTHKEY=tskey-...
TAILSCALE_HOSTNAME=qq-cf-bot
TAILSCALE_PING_HOST=<模型服务的 100.x 地址或 ts.net 域名>
TAILSCALE_PING_ATTEMPTS=4
TAILSCALE_PING_RETRY_DELAY_SECONDS=5
```

之后部署统一跑：

```bash
./scripts/deploy.sh
```

脚本会自动启动 `tailscaled`，未登录时用 `TAILSCALE_AUTHKEY` 执行 `tailscale up`，并在模型服务不可达时按 `TAILSCALE_PING_ATTEMPTS` 重试；多次仍不可达才拒绝启动 Docker。

如果希望服务器重启后也自动按这个顺序拉起工程，可以使用 `deploy/systemd/qq-cf-bot.service.example`：

```bash
sudo cp deploy/systemd/qq-cf-bot.service.example /etc/systemd/system/qq-cf-bot.service
sudo sed -i 's#/opt/qq-cf-bot#'\"$(pwd)\"'#g' /etc/systemd/system/qq-cf-bot.service
sudo systemctl daemon-reload
sudo systemctl enable --now qq-cf-bot
```

## 题面兜底和缓存

`/new` 优先抓取洛谷中文题面。若洛谷返回 403 或暂时不可用，默认回退到 Codeforces 官方英文题面，避免出题失败。

如果 `TRANSLATE_ENABLED=true` 且提供 OpenAI-compatible 翻译模型，机器人会把 Codeforces 英文题面翻译成中文再渲染；洛谷题面若正文已是中文但标题仍是英文，也会只补翻译标题。若抓取到的洛谷题面章节标题是中文但正文仍是英文，也会触发整题翻译。成功生成的题面会缓存到 `data/bot.sqlite3` 的题面缓存中，当前题图片也会保存在 `data/assets`；后续同题复用缓存，不会重复请求翻译。

## 题面发送方式

`/new` 和 `/cur` 只发送题面图片，不暴露 Codeforces 题号、题名、难度、标签或链接。题面图片会优先先发到机器人自己的私聊窗口，再把这些原始私聊消息 ID 合并转发到群，避免多张图片在群里刷屏；“刷新了一道新题目~”会放在合并转发内，不额外刷一条群消息。若 OneBot 不支持按消息 ID 转发，才会退回自定义转发节点。`/giveup` 或通过判题后才会公开题号、难度、标签、链接和已缓存参考材料。

## 切换承载 QQ

业务代码不绑定具体 QQ 号。`ONEBOT_SELF_ID` 留空时，机器人会在运行时调用 OneBot `get_login_info` 自动识别当前 NapCat 登录账号，并用这个账号先私聊自己再合并转发题面。

如果要把机器人换到另一个 QQ，服务器执行：

```bash
cd ~/qq-cf-bot
./scripts/switch-napcat-account.sh
```

脚本会停掉 watchdog、停止 NapCat、备份旧 NapCat 账号资料到 `napcat/backup/`、清空 `.env` 里的 `ONEBOT_SELF_ID`、启动 NapCat 并导出新的二维码到 `/tmp/napcat-qrcode.png`。随后在 Mac 本地拉取二维码：

```bash
scp root@43.155.179.39:/tmp/napcat-qrcode.png ~/Desktop/napcat-qrcode.png
open ~/Desktop/napcat-qrcode.png
```

用新的 QQ 扫码登录后，确认 `get_login_info` 正常，再把新 QQ 拉进允许的群，并启动 watchdog：

```bash
sudo systemctl start napcat-watchdog
```

## NapCat Watchdog

NapCat 掉线后 QQ 消息不会再上报给机器人。可以在服务器宿主机启用 watchdog：它每分钟调用 NapCat 的 OneBot HTTP API，检测离线或 API 不可用时执行 `docker compose restart napcat`，并用邮件通知。watchdog 会依次尝试 `NAPCAT_WATCHDOG_ONEBOT_URL`、`ONEBOT_HTTP_URL`、`127.0.0.1:3000`，以及自动发现到的 NapCat Docker 容器 IP，避免宿主机无法解析 `napcat` 或端口映射不一致时误判。它还会扫描最近 NapCat 日志；如果出现“账号状态变更为离线”“Login Error”“安全风险”等信号，即使 `get_login_info` 仍然返回正常，也会判定为掉线。

先在 `.env` 里配置：

```env
NAPCAT_WATCHDOG_ENABLED=true
NAPCAT_WATCHDOG_DOCKER_SERVICE=napcat
NAPCAT_WATCHDOG_DOCKER_PORT=3000
NAPCAT_WATCHDOG_LOG_CHECK_ENABLED=true
NAPCAT_WATCHDOG_LOG_SCAN_SECONDS=600
NAPCAT_WATCHDOG_NOTIFY_EMAIL=1072805307@qq.com
SMTP_HOST=smtp.qq.com
SMTP_PORT=465
SMTP_SSL=true
SMTP_USERNAME=<你的QQ邮箱>
SMTP_PASSWORD=<QQ邮箱SMTP授权码>
SMTP_FROM=<你的QQ邮箱>
```

手动重启 NapCat：

```bash
./scripts/restart-napcat.sh
```

安装成 systemd 服务：

```bash
sudo cp deploy/systemd/napcat-watchdog.service.example /etc/systemd/system/napcat-watchdog.service
sudo sed -i 's#/opt/qq-cf-bot#'"$(pwd)"'#g' /etc/systemd/system/napcat-watchdog.service
sudo systemctl daemon-reload
sudo systemctl enable --now napcat-watchdog
sudo systemctl status napcat-watchdog --no-pager
```

## GitHub 上传注意

可以直接上传本工程，但不要上传以下内容：

- `.env`
- `data/`
- 任何真实账号、密码、access token

`.gitignore` 已经默认忽略这些文件。
