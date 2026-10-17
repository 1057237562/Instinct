# data_relay — 数据下载中继（AutoDL 无卡模式自动上传）

一个部署在「有公网/国际带宽的服务器」上的 Rust 单二进制服务，把 **下载数据集** 和
**搬进 AutoDL 数据盘** 之间的手工环节全部自动化：

```
你在 WebUI 上传下载脚本（.py/.sh）
        │
        ▼
中继服务器自动执行脚本（stdout/stderr 全量日志，可重跑/取消）
        │  脚本退出码 0
        ▼
AutoDL 实例自动 无卡模式开机（~0.1 元/时）          ← 控制台 API
        │
        ▼
数据盘配额预检（超限自动清理最旧的托管目录）
        │
        ▼
上传数据 ── SFTP 主通道（断点续传）/ Jupyter Contents API 备用通道
        │
        ▼
自动关机，任务标记完成；数据盘用量持续可视化管理
```

## 为什么这么设计

- **无卡模式开机必须走控制台内部 API**。AutoDL 官方 OpenAPI（`api.autodl.com`
  的 Pro API）明确不支持无卡开机（`power_on` 只接受 `payload:"gpu"`），但网页
  控制台可以——它的请求是 `POST https://www.autodl.com/api/v1/instance/power_on`，
  body `{"instance_uuid": "...", "payload": "non_gpu"}`（从控制台前端 JS 里确
  认过），鉴权用浏览器 LocalStorage 里的 `token`。所以 `api_mode = "console"`
  需要你粘贴一次控制台 token。
- **SSH/Jupyter 凭据不用手填**。实例 detail/snapshot 接口返回 `proxy_host` /
  `ssh_port` / `root_password` / `jupyter_domain` / `jupyter_token`，开机后直接
  用于上传。
- **SFTP 是主通道**：数据集动辄几十 GB，Jupyter Contents API（base64 塞 JSON）
  又慢又脆；SFTP 支持断点续传（`.rpart` 部分）、按大小跳过已传文件，还能顺手
  执行 `df`/`du`/清理。Jupyter 通道保留给小文件或 SSH 不可用的场景
  （`jupyter_max_file_mb` 限制单文件大小）。
- **配额管理只动自己创建的数据**。所有上传都落在实例的
  `/root/autodl-tmp/relay_uploads/<task_id>/` 下，清理策略只删除这个目录里的内
  容，绝不碰数据盘上的其他文件。

## 部署（Linux 服务器）

```bash
# 1. 安装 Rust（服务器上）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# 2. 把 data_relay/ 目录同步到服务器，然后
cd data_relay
cargo build --release
install -m755 target/release/data_relay /usr/local/bin/

# 3. 首次运行生成默认配置
data_relay serve
#   → 提示已生成 config.toml，编辑它（见下），然后再启动
data_relay serve   # 监听 0.0.0.0:7800
```

生产建议用 systemd 托管：

```ini
# /etc/systemd/system/data-relay.service
[Unit]
After=network-online.target

[Service]
WorkingDirectory=/opt/data_relay        # config.toml 和 relay_data/ 放这里
ExecStart=/usr/local/bin/data_relay serve
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### 获取 console_token（一次性，约 1 分钟）

1. 浏览器登录 [autodl.com 控制台](https://www.autodl.com/console/)
2. F12 → Application（应用）→ Local Storage → `https://www.autodl.com`
3. 复制键 `token` 的值，填进 `config.toml` 的 `autodl.console_token`
   （或直接在 WebUI 右上角 → 保存令牌区域下方的配置卡片里热更新）

> token 有效期视 AutoDL 会话策略而定，失效时任务会失败并显示
> `AuthorizeFailed`，重新抓一次即可。API 密钥类字段也可以通过
> `PUT /api/config` 在线更新，不必重启。

### console_token 自动续期（WebUI 内嵌）

`console_token` 是浏览器会话令牌，重新登录 AutoDL 后会更换。data_relay 提供三层续期方式
（WebUI →「AutoDL 控制台 Token 自动续期」卡片）：

0. **微信扫码登录（推荐，零扩展零密码）**：在 WebUI 续期卡片点「微信扫码登录」，
   微信扫二维码并确认，token 自动保存并切换 console 模式（密码登录被强制滑块验证时的首选）。
1. **全自动（推荐）**：浏览器装 [Tampermonkey](https://www.tampermonkey.net/)，在卡片里复制
   用户脚本安装。之后每次登录/打开 AutoDL 控制台，脚本自动把最新的 `localStorage.token`
   推送到中继并持久化，全程无感。
2. **一键手动**：无扩展时用卡片里的书签地址（bookmarklet），登录 autodl.com 后点一下书签即推送。
3. **手动粘贴**：卡片输入框粘贴新 token，或 `PUT /api/config`。

推送端点：`POST /api/config/console_token?t=<api_token>`，body 为 text/plain JSON
（text/plain 不触发跨域预检；仅放行来自 `https://www.autodl.com` 的调用）。用户脚本里已
预填中继地址与 api_token，由卡片自动生成。

### 配置要点（config.toml）

| 字段 | 说明 |
|---|---|
| `server.api_token` | WebUI/API 的访问令牌，**务必改掉默认值** |
| `autodl.api_mode` | `console`（支持无卡开机，推荐）/ `pro`（官方 API，仅查询/关机/上传） |
| `autodl.instance_uuid` | 多实例账号必填；单实例可留空自动选择 |
| `autodl.boot_no_gpu` | 下载完成后自动无卡开机（默认开） |
| `autodl.power_off_after_upload` | 上传完自动关机（默认开） |
| `autodl.strict_host_key` | 公网环境建议 `true`（首次连接记录指纹，之后强校验） |
| `disk.max_used_ratio` | 数据盘用量超过 `total×ratio` 拒绝上传（默认 0.92） |
| `disk.auto_clean` | 超限时自动删最旧的托管任务目录（保留最新 `keep_last_uploads` 个） |
| `runner.env` | 注入脚本的环境变量，如 `HF_ENDPOINT=https://hf-mirror.com` |

## 使用流程

1. 打开 `http://<服务器>:7800/`，右上角填入 `api_token` → 保存令牌
2. 「上传下载脚本」：选 `.py`/`.sh` 文件 → 提交即自动开始执行
   - 脚本以**提交时的文件名**放在独立工作区 `relay_data/tasks/<任务id>/` 下运行，
     cwd 就是该目录；把产物写到 `out/`（或表单里指定的 `output_dir`）
   - 页面可看实时日志、重跑、取消
3. 脚本退出码 0 且勾选「自动上传」→ 自动走 无卡开机 → 配额预检 → 上传 → 关机

**上传内容约定**：默认上传 `output_dir`（默认 `out/`）整个目录；若脚本在工作区写出
`upload_paths.txt`（每行一个相对任务目录的路径，`#` 为注释行），则只上传列出的文件——
适配「下载 → 清洗 → 精确交付」流程。清洗全程在中继服务器执行，用户脚本从不在 AutoDL
上运行；AutoDL 侧只发生数据落盘（pull 通道由实例内置下载器拉取，sftp/jupyter 为纯推送）。
4. 「AutoDL 实例」面板：状态、数据盘用量条、无卡开机/关机、托管目录列表与删除

脚本示例（下载数据到 out/）：

```python
# run_download.py
import os, pathlib
from huggingface_hub import snapshot_download
pathlib.Path("out").mkdir(exist_ok=True)
snapshot_download("open-r1/codeforces", repo_type="dataset",
                  local_dir="out/codeforces")
```

命令行手动触发同一流程：

```bash
curl -H "Authorization: $TOKEN" -F "script=@run_download.py" \
     -F "auto_upload=true" http://127.0.0.1:7800/api/tasks
```

## API 一览（全部要求 `Authorization: <api_token>`，`Bearer ` 前缀可选）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/tasks` | 任务列表 |
| POST | `/api/tasks` | multipart：`script`（文件）+ `name`/`output_dir`/`auto_upload`/`upload_mode`/`autostart` |
| GET | `/api/tasks/{id}` | 任务详情（含错误信息、上传字节数） |
| GET | `/api/tasks/{id}/log?bytes=N` | 日志尾部（默认 64KB，上限 1MB） |
| POST | `/api/tasks/{id}/start` | 重跑脚本 |
| POST | `/api/tasks/{id}/upload` | 手动触发上传管线（无卡开机→预检→上传→关机） |
| POST | `/api/tasks/{id}/cancel` | 取消执行/上传（SFTP 断点保留） |
| DELETE | `/api/tasks/{id}` | 删除本地任务（运行中需先取消） |
| GET | `/api/autodl/status` | 实例状态（凭据脱敏） |
| POST | `/api/autodl/boot_no_gpu` | 无卡模式开机 |
| POST | `/api/autodl/power_off` | 关机 |
| GET | `/api/autodl/disk` | 数据盘用量 + 托管目录明细 |
| GET/DELETE | `/api/autodl/remote` | 托管目录列表 / 删除其中一个任务目录 |
| GET/PUT | `/api/config` | 配置查看（密钥打码）/ 热更新密钥与开关 |

另有 `data_relay check`：命令行快速检查 AutoDL 连通性与数据盘用量。

## 上传通道：pull 高速通道（默认）

SSH 直传走 AutoDL 的 SSH 代理，大文件速度慢。data_relay 默认改走 **pull 高速通道**：
数据不经过 SSH——中继把产物目录做成带 Range 的 HTTP 文件服务，无卡实例从它
**aria2 16 连接多线程拉取**，SSH/SFTP 只承载 KB 级控制面（配额预检、清单、启动命令、进度）。

```
中继 (HTTP ServeDir + Range)  ←──控制面 (SFTP: 清单/脚本/进度/校验)──  无卡实例
        ↑ aria2c -x16 多连接 + 断点续传 (数据面, 不走 SSH 代理) ──┘
```

- 断点续传：中断重跑自动续传；逐文件大小校验后才算完成
- 进度：任务面板实时显示（du 轮询 + 清单比对）
- 取消：SFTP 发 pkill，已拉部分保留，可续传
- 实例侧优先 aria2c，没有则回退 curl -C -；AutoDL 镜像自带 aria2c

配置 `autodl.pull_base_url` = 本机对外可达地址（如 `http://<服务器IP>:7800`）；
留空时 pull 任务会失败并提示。通道用每任务一次性 `?t=<token>` 鉴权（32 位 uuid）。

备用通道：`sftp`（AutoDL SSH 代理直传，慢但无额外依赖）、`jupyter`（小文件）。
官方文档对大数据的推荐是阿里云盘中转（AutoPanel，5~20MB/s），需要网页授权，
不适合自动化——pull 通道就是为了在没有第三方账号依赖的前提下拿到相近速度。

## 数据盘容量管理

- **上传前预检**：`现有用量 + 本地待传` 不得超过 `total × max_used_ratio - reserved_gb`，
  否则任务失败并给出数字（不用等传一半爆盘）。
- **自动清理**（`disk.auto_clean`）：预检超限时，从最旧的托管任务目录开始删
  （任务 id 含时间戳，按名字排序即按时间排序），直到腾出空间；最新
  `keep_last_uploads` 个目录永不自动删除。
- **持续可视**：实例面板显示用量条；`/api/autodl/disk` 给出 API 与 SFTP 两侧
  的实测数字。

## 开发

```bash
cargo test            # 23 个单元测试（配置/路径安全/磁盘策略/API 解析/任务存储…）
cargo build --release
cargo clippy
```

Windows 下用 rustup 的 `stable-x86_64-pc-windows-gnu` 工具链即可编译（无需
MSVC）；部署目标一般是 Linux。本目录与 `dataset_compiler/` 平级，Cargo.lock
入库，构建产物与运行时数据不入库。

## 安全注意

- `api_token` 是唯一防线：公网部署请用强随机值，并用防火墙/反代限制 7800 端口。
- `config.toml`、`relay_data/`、`known_hosts.json` 均在 `.gitignore` 中，不会把
  密钥提交进仓库。
- 控制台 token 与实例 root_password 只存本机配置/实例快照，日志与 API 响应均
  已脱敏。
- 上传通道的删除操作被限制在 `remote_root` 之内（路径穿越会被拒绝）。
