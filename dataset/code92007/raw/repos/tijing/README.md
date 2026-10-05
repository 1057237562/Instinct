# 题径

面向算法竞赛学习的题单、教学内容与比赛收藏网站。

- 前端：纯 HTML/CSS/JavaScript，可同时部署到 GitHub Pages 和自己的 Caddy。
- 题单：侧栏支持知识点、专题与子分组的三级展开，多知识点归类和可检索的技巧标签。
- 导出：每个知识点及其全部子专题可导出为带版本号的 JSON 数据文件，或生成适合打印和群聊分享的 PDF 版题单；两种格式都会保留题目来源、难度、标签、备注与题解链接。
- 个人进度：完成状态和本周进度保存在当前浏览器，无需账号，也不会改动共享题单。
- 主数据：Supabase Postgres 中的单一、带版本号 JSONB 文档。
- 权限：所有访问者可读并可投稿题目或题解，只有 `catalog_admins` 中的 Supabase 用户可编辑和审核。
- 同步：Supabase Realtime 在不同域名和设备间推送最新版本。
- 容灾：浏览器保留离线副本和待同步修改；GitHub Actions 每日生成等价的 JSON/Markdown 快照。

## 1. 创建数据库

1. 创建一个 Supabase 项目。
2. 在 Supabase SQL Editor 中完整执行 [`supabase/schema.sql`](supabase/schema.sql)。已有项目升级时也要重新完整执行一次；脚本是幂等的，会补齐投稿表、审核策略和管理员权限函数。
3. 打开项目右上角的 Connect，复制 Project URL 和 Publishable key；也可以在 Settings > API Keys 中查看或创建 Publishable key。
4. 填写 [`config.js`](config.js)：

```js
window.TIJING_CONFIG = {
  supabaseUrl: "https://YOUR_PROJECT.supabase.co",
  supabasePublishableKey: "YOUR_PUBLISHABLE_KEY"
};
```

Publishable key 会随前端公开，这是 Supabase 的正常用法；真正的写权限由 RLS 控制。不要把 `service_role` key 写入 `config.js` 或提交到仓库。

未填写这两个配置项时，页面会保持只读并提示管理员登录尚未配置，不会退回到本地可编辑模式。

## 2. 配置管理员登录

在 Supabase Authentication > URL Configuration 中设置：

- Site URL：`https://tijing.wannafly.cn/`
- Redirect URLs：
  - `https://tijing.wannafly.cn/`
  - `https://code92007.github.io/tijing/`
  - `http://127.0.0.1:4173/`

部署并打开页面后，点击顶部“管理员登录”，或从“投稿 > 管理员登录”进入，用管理员邮箱接收魔法链接。首次登录创建用户后，在 Supabase SQL Editor 执行：

```sql
insert into public.catalog_admins (user_id)
select id from auth.users where email = '你的管理员邮箱'
on conflict (user_id) do nothing;
```

此后第一次新增或修改内容会创建 `catalog/main` 数据行。管理员可以在顶部“新建 > 投稿审核”处理题目和题解投稿；访客只能浏览、投稿，并修改保存在当前浏览器中的完成状态，不能直接改动共享题单。

题目投稿审核通过后会进入所选的经典例题或实战训练；题解投稿审核通过后会附在对应题目下。投稿使用独立的 `catalog_submissions` 表，公开端只能插入待审核记录，不能读取、修改或直接写入正式题单。

发布包含题单结构迁移的新版本后，管理员首次打开页面会用云端版本号做乐观锁校验，再把去重后的迁移结果写回 Supabase；普通访客只在内存中应用迁移，不具备写回权限。

个人进度不会在设备、浏览器或 `tijing.wannafly.cn` 与 GitHub Pages 两个域名之间同步；清除浏览器站点数据也会清除该进度。题目、专题和比赛等主数据仍按后续章节进行云端同步与备份。

## 3. GitHub Pages

仓库根目录可以直接发布，无需构建：

```bash
gh api --method POST repos/Code92007/tijing/pages \
  -f 'source[branch]=main' \
  -f 'source[path]=/'
```

发布地址：<https://code92007.github.io/tijing/>

## 4. 部署到 43.155.179.39

该服务器由 Caddy 统一处理 `443` 和自动 HTTPS，宿主机 `80` 已被现有的 `algo-buddy-nginx` 容器占用。题径仓库与发布文件都放在 `/root/tijing` 下，Caddy 从 `/root/tijing/public` 提供静态文件，不新增端口、容器、系统 Nginx 站点或 Certbot 配置。

先把 DNS 的 `tijing.wannafly.cn` A 记录指向 `43.155.179.39`。确认解析生效后登录服务器：

```bash
ssh root@43.155.179.39
```

首次部署：

```bash
git clone https://github.com/Code92007/tijing.git /root/tijing
chmod +x /root/tijing/deploy/publish.sh
/root/tijing/deploy/publish.sh

install -d -m 0755 /etc/caddy/sites
install -m 0644 /root/tijing/deploy/Caddyfile.tijing /etc/caddy/sites/tijing.caddy
grep -qxF 'import /etc/caddy/sites/*.caddy' /etc/caddy/Caddyfile || printf '\nimport /etc/caddy/sites/*.caddy\n' >> /etc/caddy/Caddyfile

caddy fmt --overwrite /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile
systemctl reload caddy
```

后续更新：

```bash
cd /root/tijing
git pull --ff-only
./deploy/publish.sh
caddy validate --config /etc/caddy/Caddyfile
systemctl reload caddy
```

Caddy 只发布 `/root/tijing/public` 中的 `index.html`、`styles.css`、`app.js`、`problem-set-export.js`、`config.js` 和 `.nojekyll`，不会暴露仓库根目录、数据库脚本或备份工作流。由于服务器全局关闭了 Caddy 的 HTTP 重定向，正式入口使用 `https://tijing.wannafly.cn/`。

## 5. GitHub 自动备份

在 GitHub 仓库 Settings > Secrets and variables > Actions 添加：

- `SUPABASE_URL`：项目 URL。
- `SUPABASE_SERVICE_ROLE_KEY`：Supabase 的 service role key，仅供备份工作流使用。

也可以通过 CLI 逐个安全输入：

```bash
gh secret set SUPABASE_URL --repo Code92007/tijing
gh secret set SUPABASE_SERVICE_ROLE_KEY --repo Code92007/tijing
```

数据库首次写入后，手动试跑备份：

```bash
gh workflow run backup.yml --repo Code92007/tijing
```

之后工作流每天北京时间约 03:17 执行。只有数据变化时才会产生 Git commit：

- `backups/catalog.json`：机器可读的完整快照。
- `backups/catalog.md`：可直接阅读，并内嵌同一份可恢复 JSON。

## 6. 恢复

推荐在 GitHub Actions 中手动运行 `Restore catalog`：

1. `backup_path` 填 `backups/catalog.json` 或 `backups/catalog.md`。
2. `confirmation` 填 `RESTORE_MAIN_CATALOG`。

也可在可信机器上执行：

```bash
export SUPABASE_URL='https://YOUR_PROJECT.supabase.co'
export SUPABASE_SERVICE_ROLE_KEY='YOUR_SERVICE_ROLE_KEY'
export RESTORE_CONFIRM='RESTORE_MAIN_CATALOG'
node scripts/restore-catalog.mjs backups/catalog.json
```

恢复会写入一个更高的数据库版本，所有已打开的页面会通过 Realtime 收到恢复后的数据。

## 本地预览

```bash
python3 -m http.server 4173 --bind 127.0.0.1
```

访问 <http://127.0.0.1:4173/>。
