# 生产服务器部署

推荐把训练墙部署到独立子域名，例如 `oj.example.com`。当前前端请求 `/api/*`、`/app.js` 和 `/styles.css`，如果直接挂到现有网站的 `/oj/` 二级路径，需要额外改写这些根路径；独立子域名不需要改代码。

## 1. 准备服务器

以下命令以 Ubuntu / Debian、Docker Compose 和 Nginx 为例。先把子域名的 A/AAAA 记录指向服务器，再确认 80、443 端口已经放行。

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 nginx certbot python3-certbot-nginx
sudo systemctl enable --now docker nginx
```

## 2. 启动应用

```bash
sudo mkdir -p /opt/oj-submission-wall
sudo chown "$USER":"$USER" /opt/oj-submission-wall
git clone https://github.com/<your-name>/oj-submission-wall.git /opt/oj-submission-wall
cd /opt/oj-submission-wall
cp .env.example .env
```

编辑 `.env`，至少修改下面几项：

```dotenv
PUBLIC_BASE_URL=https://oj.example.com
BIND_ADDRESS=127.0.0.1
PORT=8000
COOKIE_SECURE=true
OJ_USER_AGENT=OJSubmissionWall/1.0 (+https://oj.example.com)
LUOGU_USER_AGENT=OJSubmissionWall/1.0 (+https://oj.example.com)
```

启动并检查健康状态：

```bash
docker compose up -d --build
docker compose ps
curl -fsS http://127.0.0.1:8000/api/health
```

## 3. 接入现有网站

创建 `/etc/nginx/sites-available/oj-submission-wall`：

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name oj.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

启用站点并申请 HTTPS 证书：

```bash
sudo ln -s /etc/nginx/sites-available/oj-submission-wall /etc/nginx/sites-enabled/oj-submission-wall
sudo nginx -t
sudo systemctl reload nginx
sudo certbot --nginx -d oj.example.com
```

访问 `https://oj.example.com/api/health`，返回 `{"ok":true,...}` 后再打开主页。

## 4. 更新和备份

SQLite 数据保存在仓库目录的 `data/` 中。更新前先备份数据库：

```bash
cd /opt/oj-submission-wall
mkdir -p data/backups
cp data/ojwall.sqlite3 "data/backups/ojwall.sqlite3-$(date +%Y%m%d-%H%M%S)"
git pull --ff-only
docker compose up -d --build
docker compose logs --tail=100 oj-submission-wall
```

不要把 `.env`、`data/`、Cookie、代理 token 或服务器地址提交到公开仓库。洛谷在海外机房访问受限时，再按 [luogu-frp.md](luogu-frp.md) 配置可信的国内出口代理。
