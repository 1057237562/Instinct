# 洛谷国内出口 FRP 部署指引

这个方案用于解决主站部署在海外时访问洛谷 `HTTP 403` 的问题。国内机器只做一个很轻的洛谷请求出口，不部署整套 OJ 训练墙。

## 架构

```text
OJ 主站 Docker 容器
  -> http://host.docker.internal:18787/
  -> 海外主站机 frps 18787
  -> FRP 隧道
  -> 国内机 127.0.0.1:8787 deploy/luogu_proxy.py
  -> https://www.luogu.com.cn
```

国内机只需要：

- Python 3
- `deploy/luogu_proxy.py`
- `frpc`

海外主站机需要：

- `frps`
- 主站 `.env` 增加 `LUOGU_PROXY_URL` / `LUOGU_PROXY_TOKEN`
- 云厂商防火墙只开放 `7000/tcp` 给国内机连入；`18787/tcp` 不对公网开放

## 约定

把下面两个 token 换成随机长字符串，不要公开：

```text
FRP_TOKEN=替换成frp隧道token
LUOGU_PROXY_TOKEN=替换成洛谷代理token
```

端口约定：

```text
海外主站机公网 IP: <你的服务器公网 IP>
frps 控制端口: 7000
frp 洛谷代理远端端口: 18787
国内机本地洛谷代理端口: 8787
```

## 一、海外主站机部署 frps

下载 frp 后，把 `frps` 放到 `/usr/local/bin/frps`。如果机器架构是普通 x86_64，选 `linux_amd64` 包。

创建配置：

```bash
mkdir -p /etc/frp
cat > /etc/frp/frps.toml <<'EOF'
bindAddr = "0.0.0.0"
bindPort = 7000
proxyBindAddr = "0.0.0.0"

auth.method = "token"
auth.token = "替换成frp隧道token"

transport.tls.force = true

allowPorts = [
  { single = 18787 }
]

log.to = "/var/log/frps.log"
log.level = "info"
log.maxDays = 7
EOF
```

创建 systemd 服务：

```bash
cat > /etc/systemd/system/frps.service <<'EOF'
[Unit]
Description=frp server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/frps -c /etc/frp/frps.toml
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now frps
systemctl status frps --no-pager
```

检查监听：

```bash
ss -lntp | grep -E ':7000|:18787'
```

正常情况下：

- `7000` 会监听在 `0.0.0.0`，给国内 frpc 连接。
- `18787` 需要让 Docker 容器通过 `host.docker.internal` 访问，所以会监听在宿主机网络上；不要在云厂商防火墙开放这个端口。

## 二、国内轻量机部署洛谷代理

创建目录，并把本仓库里的 `deploy/luogu_proxy.py` 上传到这个路径：

```bash
mkdir -p /opt/luogu-proxy
# 把 deploy/luogu_proxy.py 放到 /opt/luogu-proxy/luogu_proxy.py
chmod +x /opt/luogu-proxy/luogu_proxy.py
```

创建代理服务：

```bash
cat > /etc/systemd/system/luogu-proxy.service <<'EOF'
[Unit]
Description=OJ Wall Luogu proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=HOST=127.0.0.1
Environment=PORT=8787
Environment=LUOGU_PROXY_TOKEN=替换成洛谷代理token
# 可选：只有 record/list 分页仍被洛谷拦时再配置，不要公开到仓库或群里
# Environment=LUOGU_PROXY_UPSTREAM_COOKIE=__client_id=...; C3VK=...
# Environment=LUOGU_PROXY_UPSTREAM_CSRF_TOKEN=...
ExecStart=/usr/bin/python3 /opt/luogu-proxy/luogu_proxy.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now luogu-proxy
systemctl status luogu-proxy --no-pager
```

本机测试代理：

```bash
curl -sS http://127.0.0.1:8787/health

curl -sS http://127.0.0.1:8787/ \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer 替换成洛谷代理token' \
  --data '{"url":"https://www.luogu.com.cn/","headers":{"User-Agent":"Mozilla/5.0","Accept":"text/html"}}' \
  | head -c 300
```

如果第二个请求返回 JSON，且里面有 `"ok":true`，说明国内机访问洛谷出口可用。
`/health` 里的 `cookies` 只显示代理进程保存的洛谷临时 Cookie 数量，不显示内容。跑过几次同步后它通常会大于 0；如果一直是 0，后续分页请求可能会被洛谷打回登录页。

## 三、国内轻量机部署 frpc

下载 frp 后，把 `frpc` 放到 `/usr/local/bin/frpc`。

创建配置：

```bash
mkdir -p /etc/frp
cat > /etc/frp/frpc.toml <<'EOF'
serverAddr = "<你的服务器公网 IP>"
serverPort = 7000

auth.method = "token"
auth.token = "替换成frp隧道token"

transport.tls.enable = true

[[proxies]]
name = "luogu-proxy"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8787
remotePort = 18787
transport.useEncryption = true
transport.useCompression = true
EOF
```

创建 systemd 服务：

```bash
cat > /etc/systemd/system/frpc.service <<'EOF'
[Unit]
Description=frp client
After=network-online.target luogu-proxy.service
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/frpc -c /etc/frp/frpc.toml
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now frpc
systemctl status frpc --no-pager
```

## 四、海外主站接入代理

在主站项目目录的 `.env` 增加或修改：

```bash
LUOGU_PROXY_URL=http://host.docker.internal:18787/
LUOGU_PROXY_TOKEN=替换成洛谷代理token
# 可选：Cookie/CSRF 放在海外主站，由主站随每次代理请求动态透传
LUOGU_COOKIE='__client_id=xxx; _uid=xxx; C3VK=xxx'
LUOGU_CSRF_TOKEN=xxx
```

也可以不写入 `.env`，只在触发同步时临时传入：

```bash
curl -sS https://你的主站/api/sync \
  -H 'Content-Type: application/json' \
  -H 'Cookie: ojwall_session=你的登录态' \
  --data '{"force":true,"luoguCookie":"__client_id=xxx; _uid=xxx; C3VK=xxx","luoguCsrfToken":"xxx"}'
```

重新启动主站：

```bash
cd /path/to/oj-submission-wall
docker compose up -d --build
```

检查主站容器能不能通过 FRP 访问国内代理：

```bash
docker compose exec oj-submission-wall python - <<'PY'
import json
import os
import urllib.request

url = os.environ["LUOGU_PROXY_URL"]
token = os.environ["LUOGU_PROXY_TOKEN"]
payload = json.dumps({
    "url": "https://www.luogu.com.cn/",
    "headers": {"User-Agent": "Mozilla/5.0", "Accept": "text/html"},
}).encode()
req = urllib.request.Request(
    url,
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=20) as resp:
    print(resp.status)
    print(resp.read(300).decode("utf-8", "ignore"))
PY
```

看到 `200` 且 JSON 里有 `"ok":true`，再到网页点洛谷卡片的“重试”。

## 五、排查

如果主站仍然报洛谷 403：

```bash
docker compose exec oj-submission-wall sh -lc 'env | grep LUOGU_PROXY'
docker compose logs --tail=120 oj-submission-wall | grep -iE 'luogu|403|proxy|error'
```

如果海外主站机看不到 `18787`：

```bash
systemctl status frps --no-pager
tail -80 /var/log/frps.log
```

如果国内机 frpc 连不上：

```bash
systemctl status frpc --no-pager
journalctl -u frpc -n 80 --no-pager
```

如果国内机本地代理访问洛谷也 403，那说明这台国内机出口本身也被风控，需要换出口 IP 或稍后重试。
