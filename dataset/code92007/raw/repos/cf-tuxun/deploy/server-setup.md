# cf-tuxun.wannafly.cn 部署

先在腾讯云 DNSPod 增加 `cf-tuxun` 的 A 记录，指向 `43.155.179.39`。确认解析生效后，在服务器执行：

```bash
ssh root@43.155.179.39
git clone https://github.com/Code92007/cf-tuxun.git /root/cf-tuxun
cd /root/cf-tuxun
docker compose up -d --build
cp deploy/cf-tuxun.caddy /etc/caddy/sites/cf-tuxun.caddy
caddy validate --config /etc/caddy/Caddyfile
systemctl reload caddy
```

首次部署后，把自己的站内用户名设为超级管理员：

```bash
cd /root/cf-tuxun
docker compose exec app python scripts/make_admin.py 站内用户名 --super
```

之后由超级管理员在站内“权限管理”页面任免普通审核管理员；普通管理员只能审核投稿。

以后更新时执行：

```bash
cd /root/cf-tuxun
git pull --ff-only
docker compose up -d --build
```

账户、积分、投稿图片和审核记录保存在 `/root/cf-tuxun/data`。升级容器不会覆盖它们；迁移服务器前应备份这个目录。
