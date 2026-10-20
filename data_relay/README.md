# data_relay — 数据中继与 AutoDL 服务器训练管理

一个 Rust 单二进制服务，支持下载、清洗、交付数据，并在 AutoDL 服务器上启动、
暂停、续训和管理 Instinct 训练。下载中继适合部署在有公网/国际带宽的服务器；
Windows 本机也可以通过 SSH 管理服务器训练。

## 直接上传数据（WebUI）

页面分为 **数据上传 / 远端数据集 / 脚本任务 / 服务器训练 / 连接设置**。数据上传不执行脚本，
直接把原文件名安装到 **`/root/autodl-tmp/dataset/`**。

1. 保存中继访问令牌，在“数据上传”选择目标 AutoDL 实例。
2. 拖入数据文件及对应审计/来源 sidecar（支持多选、移除），选择交付通道。
3. 点击“上传并交付到 dataset”。本地接收完成后，校验和交付在后台继续，页面可以关闭。

- 浏览器 → 中继：16 MiB 分块、最多 4 个并发请求、逐块 SHA-256、网络失败自动重试。
  分块直接写入暂存文件，不把整个大文件载入内存，也不保留额外的完整合并副本。
  可暂停；刷新页面或中继重启后，在记录中选择全部原文件续传。续传时重新核对已收分块的
  内容，只有缺失分块需要传输。浏览器本地文件句柄不会在刷新后保留。
- 中继 → AutoDL：配置实例可达的 `autodl.pull_base_url` 后可选 HTTP Range 拉取
  （aria2c 单文件最多 16 连接，没有 aria2c 时回退可续传 curl）；没有公网中继地址时选 SFTP。
  公开地址可在“连接设置”保存。实际速度由上下行带宽、磁盘和 SSH 通道决定。
- 数据传输记录和脚本任务都每 3 秒刷新速率与预计剩余传输时间，使用约 10 秒的滚动采样。
  SFTP 按本轮实际发送字节计速，不把已续传/跳过的字节计入速率；HTTP pull 显示落盘估算速率，
  使用已分配磁盘块而非稀疏文件的表观长度。连接准备、远端校验和断网分别显示对应状态；
  重试、计数回退或页面长时间暂停后重新采样，不沿用旧速率。ETA 不包含后续校验时间。
- 中继上必须有足够暂存空间；反向代理需允许至少 16 MiB 请求体，建议 `client_max_body_size 20m`。
  数据直传不使用 Jupyter 的小文件接口。单任务最多 256 个文件，单文件上限 1 TiB。
- 目标实例保存在任务中，切换页面选择不改变已有数据任务。交付使用现有无卡开机策略，完成后
  保持实例运行。与 GPU 模式切换及其他数据交付互斥，等待时显示“校验 / 等待交付”。
  不会清理 dataset；磁盘配额不足时停止并保留断点，续传预算扣除已分配的暂存数据块。
- 远端暂存于 `dataset/.relay_incoming/<task_id>`，全部文件 SHA-256 验证通过才逐个原子安装。
  同名同内容可重试；同名不同内容拒绝覆盖，需重命名后新建上传。失败可能留下已校验安装的
  部分文件，重试会验证已有文件。删除中继记录只删本地副本，不删远端数据或暂存断点。
- 传输校验不代替身份清洗和语料审核；训练入口继续执行原有审计检查。

本地验证：`cargo test`、`cargo clippy`、
`python data_relay/tests/test_dataset_upload_remote.py`（从仓库根目录）。
浏览器回归使用 `node data_relay/tests/test_upload_ui.cjs`，需要 Playwright + Edge，
`NODE_PATH` 指向 Playwright 所在 node_modules。必须先启动隔离的本地测试中继
（`127.0.0.1:17800`，令牌 `local-ui-test-only`，独立 `data_dir`）；
测试只向本地中继发送合成数据，AutoDL API 与交付提交由浏览器拦截模拟。

API（均需访问令牌）：

| 方法 | 路径 | 用途 |
|---|---|---|
| POST | `/api/data/uploads` | 创建任务，JSON 包含 `name`、`instance_uuid`、`upload_mode`、`files`（`name/size/fingerprint`） |
| GET | `/api/tasks/{id}` | 查询任务及已确认分块哈希；分块大小固定为 16 MiB |
| PUT | `/api/data/uploads/{id}/files/{file}/chunks/{chunk}` | 从 0 开始的文件/分块索引，二进制请求体，`X-Chunk-SHA256` 请求头 |
| POST | `/api/data/uploads/{id}/complete` | 完整性检查后返回 202，后台校验并交付；失败后也用此接口重试 |

`fingerprint` 是浏览器对“文件名 + 换行 + 字节数 + 换行 + 首尾各最多 64 KiB”的 SHA-256，
用于快速检查所选文件；服务端不把它当作完整文件校验。任务列表省略逐块哈希，只返回累计接收字节。

## 远端 dataset 管理

“远端数据集”页直接浏览所选实例的 `/root/autodl-tmp/dataset`，与脚本产物托管目录分开。
支持进入子目录、面包屑返回、文件名搜索、类型筛选、按名称/大小/修改时间排序，以及每页 50 项。
显示数据盘剩余容量和当前目录的文件数/文件总量（不递归计算子目录大小）。单目录最多加载
10,000 项，超出时提示截断。实例离线时不会自动开机；上传暂存与其他隐藏目录不展示。

- JSON/TXT/MD/YAML/CSV 可预览前 128 KiB，较大文件明确提示截断；内容按纯文本显示。
- 复制路径得到远端绝对路径。“用于训练”填写语料及匹配的审计报告，并清除身份清洗确认框，
  需要核对后重新确认；不会自动启动训练。
- 删除弹窗展示目标实例、完整路径和大小。只删除选中的普通文件，不递归删除目录，不自动
  删除关联审计报告或中继副本。服务端拒绝越界/隐藏路径、符号链接、已变化的文件版本；
  数据交付或训练切换期间阻止删除，并在删除前检查远端受管及独立训练进程。
- 浏览和删除都固定请求中的实例 UUID，切换实例会清空旧列表和待确认操作。

API 均需要 `Authorization`：`GET /api/datasets?instance_uuid=UUID&path=相对目录`，
`GET /api/datasets/preview?instance_uuid=UUID&path=相对文件`，
`DELETE /api/datasets` 接收 JSON `instance_uuid/path/version`（version 取自列表）。
测试：`python data_relay/tests/test_datasets_remote.py`；浏览器回归
`node data_relay/tests/test_datasets_ui.cjs` 使用与上传测试相同的隔离本地中继和 Playwright。

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
上传脚本产物 ── HTTP pull 高速通道 / SFTP 续传 / Jupyter 小文件通道
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
- **HTTP pull 是高速通道，SFTP 是备用通道**：数据集动辄几十 GB，Jupyter Contents API（base64 塞 JSON）
  又慢又脆；SFTP 支持断点续传（`.rpart` 部分）、按大小跳过已传文件，还能顺手
  执行 `df`/`du`/清理。Jupyter 通道保留给小文件或 SSH 不可用的场景
  （`jupyter_max_file_mb` 限制单文件大小）。
- **配额管理只动自己创建的数据**。脚本产物上传落在实例的
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

### 本地数据集直接交付到训练仓库

根目录 `upload_pretrain_coder_12b.bat` 使用独立的 `data_relay_delivery.exe`，
读取本目录 `config.toml`，将本地 `dataset/pretrain_coder_12b.parquet` 与存在的
两份 `.report.json` 交付到实例 `/root/autodl-tmp/dataset/`。
独立 exe 允许原有 WebUI 继续运行；它与普通服务由同一 Rust crate 编译。
BAT 通过 `scripts/start_pretrain_coder_12b_upload.ps1` 启动隐藏的 WMI 工作进程，
其父进程是 Windows WMI 服务，关闭终端或重启 Codex 不会结束上传。
`relay_data/delivery-active.json` 保存工作进程 PID、日志路径和完成/失败状态；
文件锁防止重复上传，重复双击会显示正在运行的任务。
中断后再次双击会自动仅续传，也可显式加 `-ResumeOnly`：不再清理数据或更新 Git。
该模式要求远端 Git 初始化已完成；初始准备阶段失败时可直接使用前台脚本执行完整流程。

```powershell
# 只查看清理/更新/交付计划，要求实例已经运行：
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/upload_pretrain_coder_12b.ps1
# 执行（也可双击根目录 BAT）：
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/upload_pretrain_coder_12b.ps1 -Execute
```

脚本默认清理这次调查发现的两份旧 SFT 文件：
`sft_competitive_t2t_ultrachat_16k_coding_heavy.jsonl` 和
`sft_easy_medium_competitive_clean_4k.jsonl`。同时清理仓库内的
`.cache/torch_compile`、`.cache/huggingface/datasets`、`.cache/huggingface/hub`
与 `.Trash-0/files`；保留 HF token、README、审计报告、权重及检查点。
PowerShell 参数可覆盖 `-RemoteRepo`、`-Branch`、`-Instance`、`-RemoveOld`，
`-SkipCacheCleanup` 关闭缓存清理。日志保存在 `data_relay/relay_data/delivery-*.log`。

该实例无法直接访问 GitHub，因此 PowerShell 启动脚本会先在本机
`git fetch origin <branch>`，创建含原始提交与历史的 Git bundle，交由 SFTP
传输并核验。服务器从 bundle 初始化仓库或 `fetch` + `merge --ff-only` 更新，
保留真实提交 SHA、GitHub origin 和分支 upstream；无需服务器访问 GitHub。
CLI 使用 `--git-bundle <本地bundle路径>` 启用该方式，bundle 应包含
`refs/remotes/origin/<branch>`。不提供 bundle 时才使用远端 clone/pull。

也可直接调用 CLI（以下在 `data_relay/` 工作目录执行）：

```bash
data_relay inspect-server --config config.toml
data_relay deliver --config config.toml --source ../dataset/pretrain_coder_12b.parquet \
  --repo /root/autodl-tmp --branch master --init-repo --clean-cache \
  --remove-old old_corpus.jsonl --execute
```

`inspect-server` 会按配置在必要时无卡开机，并输出仓库/代码拷贝、数据目录、
磁盘用量；`--instance UUID` 可临时覆盖实例选择。`deliver` 不带 `--execute` 时
只检查已经运行的实例，不开机、不清理、不更新代码、不上传。
执行模式先检查目标及空间，清理指定旧语料（目录选择只匹配 `.parquet`、
`.jsonl`、`.jsonl.gz`、`.arrow`，保留其他文件），然后通过
`git pull --ff-only` 更新当前分支。有本地受 Git 跟踪的修改时拒绝更新。
没有 `.git` 的 Instinct 代码拷贝需要显式 `--init-repo`：从指定分支浅克隆，
将受跟踪的代码检出到现有目录并安装 Git 元数据，保留数据与训练产物。
已有 Git 仓库不会自动切换分支，`--branch` 必须与当前分支相符。

交付走现有 SFTP 通道，使用 `.incoming.rpart` 断点续传；全部文件通过远端
SHA-256 校验后才替换最终文件，并写出 `.delivery.report.json`。
空间检查保留 1 GiB 余量。默认不关机，以免终止训练进程。
身份清洗仍需在训练前按根目录 AGENTS.md 单独完成，传输校验不能替代内容审计。

远端 helper 单元测试：`python data_relay/tests/test_delivery_remote.py`（仓库根目录）。

### 实例存储维护

`remote-script --remote-script <本地Python文件>` 使用配置中的实例和 SSH
凭据直接执行明确指定的维护脚本（必要时按配置开机）。它与 WebUI 上传的
下载/清洗脚本是不同入口；WebUI 脚本仍只在中继上执行。

在 `data_relay/` 工作目录下，本次已审核的存储维护可运行：

```powershell
..\data_relay_delivery.exe --config config.toml remote-script --remote-script src/cleanup_server_space.py
```

此脚本针对 `/root/autodl-tmp` 中固定名称的旧检查点临时文件和三对权重，
不是通用删除工具。它检查训练进程及临时文件占用，删除过期的中断保存
`.tmp`；对 `out/` 与 `checkpoints/` 中的权重，逐条 SHA-256 校验全部
PyTorch ZIP 记录（只忽略归档包装目录名），一致后原子替换为硬链接。
两条权重路径共享存储，原地写入会共享修改；原子替换保存会分离链接。
完整 `_resume.pth`、语料和报告都保留。操作记录写入
`/root/autodl-tmp/.cache/relay_storage_cleanup/`。重复运行会跳过已链接的文件。
校验测试：`python data_relay/tests/test_storage_cleanup.py`（仓库根目录）。

### 实例 Python 环境体检（numpy/pyarrow）

实例上 Config WebUI 的 `st.dataframe` 每次报
`ImportError: numpy._core.multiarray failed to import`，或训练经 `datasets`
加载语料时在 `import pyarrow` 处崩溃，说明 miniconda 环境里 numpy 与 pyarrow 的二进制版本不匹配（pip 单独升降级过
numpy，或 pip/conda 混装）。这是固定的环境状态问题，不会自愈，需要体检脚本诊断：

```powershell
..\data_relay_delivery.exe --config config.toml remote-script --remote-script src/check_server_python_env.py
```

默认只体检（只读）：实际导入 numpy / `numpy._core.multiarray`（按版本回退
`numpy.core`）/ pyarrow / pandas，并核对 dist 元数据版本、重复安装记录
（同名包多个 `.dist-info` 即混装）和 conda 管理痕迹；环境正常退出码 0，
发现问题时 1。报告打印一行 `ENV_REPORT` JSON，并写入
`/root/autodl-tmp/.cache/relay_python_env/`。

修复需显式授权：编辑脚本顶部 `APPLY_REPAIR = True` 后重跑（SSH 手动执行时
也可 `python3 check_server_python_env.py --apply`）。规则：numpy 本身正常时
只升级 pyarrow（或 pandas），不动 numpy；numpy 自身损坏时先卸载两者，再按
现有元数据版本精确重装 numpy（版本不明或有重复记录时用 `numpy<2`）并安装
最新 pyarrow，全部走 `python -m pip --no-cache-dir`。检测到 `train_*.py`
训练进程时拒绝修复。`NUMPY_SPEC` / `--numpy-spec` 可显式指定 numpy 约束。
修复后立即复测并给出最终状态。校验测试：
`python data_relay/tests/test_server_python_env.py`（仓库根目录）。

## 服务器训练管理器

### 训练链路代码同步

在“服务器训练”中的“训练链路代码同步”先点击 **比较本地与远端代码**，核对新增/替换文件，
再点击 **同步变更文件**。方向是本机仓库 → 所选 AutoDL 实例，默认远端仓库
`/root/autodl-tmp`。本机仓库由 `training.local_repo_root` 指定（默认 `..`，相对 relay
配置文件）；可在“服务器训练设置”修改。本地读取工作区实际内容，包含未提交和未跟踪的代码，
不需要 GitHub 连通或 Git 提交。比较会创建不可变的本地快照；同步安装的是比较时的内容。

- 代码范围：`trainer/**/*.py`、`model/**/*.py`、`scripts/*.py`、
  `scripts/data_loader/**/*.py`、`run_recurrsive.py`、`run_linear.py`、`eval_llm.py`、
  Config WebUI 启动器和训练关机脚本。可选包含训练配置模板、tokenizer 文本资产和 requirements。
- 排除 `trainer/webui_state.json`、数据、权重、检查点、缓存、日志和 relay 凭据。
  不删除远端额外文件，不自动安装 Python 依赖。缺少本地文件的目录不会被当成镜像删除。
- 列表按 SHA-256 显示一致、新增、替换；只传变更文件。远端代码在比较后有变化时拒绝安装，
  需要重新比较。暂存文件先做 SHA-256、Python 编译及 JSON 解析检查，再备份、逐文件原子替换，
  最后验证整个快照。保留原文件权限；安装异常会尝试回滚。
- 备份和清单保存在远端 `<repo>/.relay_code_sync/<plan_id>/`，本地快照/操作状态在
  `<data_dir>/code_sync/<plan_id>/`。记录里可以显式回滚；回滚只处理本次变更，后续其他修改会触发
  冲突而非被覆盖。被中断的安装留下 `ACTIVE` 标记，中继训练启动会被阻止，需先恢复/回滚。
- 同步/回滚与中继交付及训练模式切换互斥；远端训练管理锁和进程检查覆盖训练器、流水线与调优。
  比较和同步均不会自动开机。操作提交返回 202，后台运行，页面可关闭后再查看记录。
- 单文件上限 32 MiB、快照上限 128 MiB/4096 文件。该功能同步代码文件，不是双向合并工具。

API（均需访问令牌）：`POST /api/code-sync` 接收 `instance_uuid/include_assets` 创建比较快照；
`GET /api/code-sync` 查询最近 20 条记录，`GET /api/code-sync/{id}` 查询详情；
`POST /api/code-sync/{id}/apply` 同步，`POST /api/code-sync/{id}/rollback` 回滚。
目标实例和远端目录固定在计划中，不会随当前下拉框选择漂移。
测试：`cargo test`、`python data_relay/tests/test_code_sync_remote.py`、
`node data_relay/tests/test_code_sync_ui.cjs`（浏览器测试使用隔离本地中继与模拟远端响应）。

### 启动与续训

项目根目录双击 `start_data_relay.bat`，启动器从根目录的 `data_relay_manager.exe`、
`data_relay.exe` 和 `data_relay/target/release/data_relay.exe` 中按修改时间选择最新构建。
构建不会覆盖正在运行的进程；服务退出后重新运行启动器即可使用新版。
打开 `http://localhost:7800` 并填写访问令牌。“服务器训练”页自动读取
Config WebUI 缓存中的当前阶段、模型结构、数据路径和训练参数。可以直接修改中继的
配置控件、选择语料，预览命令后点击“启动训练（自动切换 GPU）”。无需身份审核勾选或审计报告。
支持 Pretrain、CPT、SFT、LoRA、DPO、PPO、GRPO/CISPO、Agent RL、蒸馏，及多卡 DDP。

**启动或续训会自动切换有卡模式并开始 GPU 计费**：无卡实例先检查语料路径、
模型配置、磁盘余量和现有训练进程，确认没有训练/中继上传后正常关机，等待关机
完成再有卡开机；已关机实例直接有卡开机，已在 GPU 模式运行的实例保持运行。
等待 SSH 和 CUDA 就绪后才启动训练，超时使用 `autodl.boot_timeout_secs`（默认 900 秒）。
WebUI 启动按钮通过单次 `auto_gpu:true` 请求明确授权切换，不需要另行手动开关机，
也不会改写数据交付的无卡开机配置。CLI 用 `--auto-gpu` 明确授权一次切换，或
配置 `autodl.allow_gpu_boot=true`。独立“有卡模式开机”按钮使用后者的配置许可。
GPU 库存不足时保留启动任务，显示“等待 GPU”，默认每 60 秒重试开机；等待库存的
时间不计入开机超时。可在服务器训练设置中修改 `training.gpu_retry_secs`（10–3600 秒）。
余额不足、凭据失效、驱动/显存错误等会显示失败原因，不当作库存不足无限重试。

启动/续训 API 立即返回 HTTP 202 和 `operation_id`，实际流程在后台执行。
页面每 4 秒读取启动阶段，关闭浏览器不影响流程。等待记录将训练请求、目标实例和
训练配置快照保存到 `<data_dir>/training/operations`；切换页面实例不改变已排队目标，
中继重启后自动恢复未下发的任务。已经进入命令下发阶段但结果不明的请求标记为
`interrupted`，先查看远端任务再处理，避免重启后重复训练。启动/续训 operation ID
同时用于远端幂等检查。当前只接收一个待启动请求；页面可取消排队，正在进行的开机
请求可能完成，但取消后不会下发训练。已下发的训练使用暂停/停止按钮。
训练一旦启动，由远端独立守护进程管理，SSH 断开、浏览器关闭、中继重启都
不影响训练。模式切换和训练启动期间重复请求被拒绝。

配置（可在 WebUI 的“服务器训练设置”保存）：

```toml
[training]
repo_root = "/root/autodl-tmp"
python = "/root/miniconda3/bin/python"
min_free_gb = 8.0
cache_max_gb = 1.0
panel_state_path = "../trainer/webui_state.json"
gpu_retry_secs = 60
```

`panel_state_path` 是**中继本机**的 Config WebUI 缓存文件路径，相对 `config.toml`
所在目录解析，支持绝对路径。Windows 项目内直接使用默认值；独立部署中继时
需复制或同步 Config WebUI 的缓存，并配置对应路径。读取不需要启动 Streamlit
或导入 PyTorch，也不会改写源缓存。

缓存使用当前面板顶层值，以及 `epochs_<阶段>`、`learning_rate_<阶段>`、
`warmup_ratio_<阶段>`、`data_path_<阶段>` 等阶段参数，不重新应用 preset，也不
使用可能过时的 `pipeline_stages` 快照覆盖当前设置。精度、优化器、FP8、编译、
packing、streaming、profiling 和模型拓扑（含 RoPE/残差/循环主干）一起读取。
本机仓库内的 Windows 数据路径转换为服务器仓库相对路径，保留原扩展名。
页面提供训练超参数、精度与计算、Packing/流式加载、模型结构、RoPE、Linear/RecurrsiveInstinct、
性能分析，以及每个 trainer 的专用参数。模型预设来自 Config WebUI，专用参数的类型和
选项来自 trainer argparse。修改自动保存到 `<server.data_dir>/training/panel.json`，
各阶段分别保留数据、基座、检查点、学习率、轮数和专用参数；公共模型与计算设置共享。
没有本机缓存时仍可使用默认预设直接配置。支持导入 `webui_state.json` 或面板导出的 JSON。
“导入 Config WebUI 当前配置”才会替换当前编辑内容；不会改写本机源缓存。
仅刷新服务器环境不会覆盖已编辑的参数。
“当前模型配置”面板显示主干、维度/层数、Q/KV 头数、词表/上下文、专家数和
每 token 激活专家数、FFN、RoPE 和路由设置，可展开完整 JSON。显示值来自实际
面板配置，手工修改的 top-k 等配置也会显示。选择服务器配置文件或完整续训检查点
时会标明实际配置来源，缓存预览不会冒充检查点的模型结构。

默认模型配置使用缓存快照，启动后写入任务目录的 `model_config.json`，避免
依赖服务器上较旧的同名配置文件；已有完整检查点续训仍优先使用检查点保存的
模型配置。启动请求使用提交时的配置快照；排队期间修改面板不会改变已提交任务。
CLI `--from-panel-cache` 仍可用参数指纹检测源缓存变更。
缓存里的续训勾选会显示提示，需要选择服务器上的完整检查点；它不会自动生成
`--from_resume`、覆盖输出路径或继承本机暂停/关机标记。已有受管任务续训保留
该任务实际启动时保存的参数。

- 所有训练从仓库根目录运行；配置 JSON 中的维度/层数/MoE 显式传给 trainer，
  避免退回旧的 768/8 层默认值。已有完整检查点续训强制使用保存的模型配置。
- 语料必须位于仓库内，格式为 JSONL、gzip JSONL 或 Parquet。启动只检查路径和文件；
  不要求身份审核或审计报告。CLI/API 若显式传入 `audit_path`，仍可选择验证报告中的
  `output_sha256` / `output.sha256`；受管任务续训不重复扫描文件哈希。
- CPT/SFT/后训练需要在 `out/` 中选择已经完成的基座权重；Pretrain 从头训练。
  可导入 `checkpoints/*_resume.pth`，对应 `.json` 配置必须存在，检查点阶段需
  与 trainer 匹配；续训保留原权重前缀，使用 `--from_resume 1`。
  管理任务续训保留原参数和同一语料，epochs 是训练的总目标轮数。
- `args` 为 trainer 参数的 JSON 对象，值为字符串、数字或布尔值；管理器控制
  `save_dir/save_weight/lora_name/pause_file/config_path/from_resume`，禁止用户覆盖。
  蒸馏支持 `student_*`/`teacher_*` 参数；基座选择映射到教师权重。
- 数据缓存放在数据盘 `.cache/relay_training/<job_id>/`，默认预算 1 GiB，
  预训练默认 streaming chunk 128 MiB、prefetch 0；预留空间分别在启动前、
  续训前以及分离历史共享权重后检查。管理器不主动删除训练产物或语料。
  这些是没有显式训练参数时的兜底；Config WebUI 缓存里的
  `data_cache_max_gb/streaming_chunk_mb/streaming_prefetch_chunks` 优先，保持用户配置。

“暂停并保存”通过每任务专属标记请求 trainer 在 step 边界保存完整检查点，
退出码 42 表示暂停；“停止（不保存）”终止进程组，会丢失未保存进度。
一个仓库同时只允许一个受管任务，也检查非受管的 `train_*.py` 进程。
管理器用 Linux boot ID 和进程启动时间辨认进程，避免 PID 重用后误杀。
有训练运行时，手动关机和数据上传后的自动关机均被阻止。
每次启动/续训请求均下发 `shutdown_after_train:true`、`shutdown_delay_seconds:60`，
不继承 Config WebUI 中未勾选的关机开关；这些生命周期字段不会作为 trainer 参数。
远端 supervisor 保存成功退出状态后，等待 60 秒、再次检查没有新训练/代码同步，
然后调用 AutoDL 支持的 `/usr/bin/shutdown`（[官方说明](https://www.autodl.com/docs/save_money/)）。
浏览器、中继或 SSH 断开不影响该关机流程。暂停 42、失败和手动停止不会自动关机；
关机命令错误记录在任务的 `shutdown_error`，不会把成功训练改成失败。
历史硬链接权重在同名输出恢复训练前会分离，避免覆盖其他备份路径。

远端状态及日志：`<repo_root>/.relay_training/<job_id>/job.json`、`attempt-N.log`、
`supervisor.log`；每次续训保留历史日志。中继缓存任务快照及启动阶段在
`<server.data_dir>/training/`，实例离线时仍可查看快照。

CLI 示例（在 `data_relay/` 工作目录执行，先修改示例中的路径和训练参数）：

```powershell
..\data_relay_manager.exe --config config.toml training --training-action inventory
..\data_relay_manager.exe --config config.toml training --training-action panel_cache
..\data_relay_manager.exe --config config.toml training --training-action start --from-panel-cache --request-json training_job.example.json --auto-gpu
..\data_relay_manager.exe --config config.toml training --training-action list
..\data_relay_manager.exe --config config.toml training --training-action pause --job-id r_ID
..\data_relay_manager.exe --config config.toml training --training-action resume --job-id r_ID --auto-gpu
..\data_relay_manager.exe --config config.toml training --training-action log --job-id r_ID
```

所有 API 使用原有 `Authorization` 访问令牌：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/training/environment` | 模型配置、语料、权重、完整检查点、GPU 和空间 |
| GET | `/api/training/panel_cache?trainer=pretrain` | 本机 Config WebUI 缓存转换后的参数与模型配置；省略 trainer 则使用缓存中的当前阶段 |
| GET / PUT | `/api/training/panel` | 读取 / 保存中继面板配置，GET 返回控件与预设定义 |
| POST | `/api/training/preview` | 编译 `panel_state` 并预览远端命令；不连接或启动实例 |
| GET / POST | `/api/training/jobs` | 列表 / 提交启动（请求与示例 JSON 一致，可加 `auto_gpu:true`） |
| GET | `/api/training/operations` | 排队、GPU 重试时间/次数及启动阶段（不返回完整请求快照） |
| POST | `/api/training/operations/{id}/cancel` | 取消尚未下发的训练启动任务 |
| GET | `/api/training/jobs/{id}` | 单个任务状态 |
| GET | `/api/training/jobs/{id}/log?bytes=262144&attempt=0` | 指定执行次数的日志尾部 |
| POST | `/api/training/jobs/{id}/pause` | 保存后暂停 |
| POST | `/api/training/jobs/{id}/resume` | 续训，JSON 可传 `{"auto_gpu":true}` |
| POST | `/api/training/jobs/{id}/stop` | 终止进程组 |
| POST | `/api/training/boot_gpu` | 独立有卡开机，需要配置许可 |

命令预览中的任务 ID、输出前缀和 DDP 端口在启动时生成，实际命令写入任务日志。
已有检查点或服务器配置文件的结构以远端文件为准。面板支持应用服务器已验收的
`autotune_profile`；自动调优搜索和多阶段流水线编排仍由 Config WebUI 提供。

维护控件定义：修改 Config WebUI 预设或 trainer 参数后，运行
`python data_relay/generate_training_schema.py`，并以 `--check` 检查生成文件。
`python data_relay/tests/test_training_panel_api.py` 在隔离本地服务上验证所有阶段的配置、
命令一致性、保存恢复和无效参数拦截，不启动 AutoDL。

远端 helper 测试：`python data_relay/tests/test_training_remote.py`。Linux 上使用
临时仓库和 CPU 模拟 trainer 检查启动、暂停 42、控制器重启、续训和日志；Windows
跳过两个 Linux 进程生命周期测试。测试不启动实际 GPU 训练。
缓存一致性测试：release 编译后运行 `python data_relay/tests/test_training_panel_cache.py`，
使用临时缓存和离线 CLI，与 Config WebUI 的实际模型/命令构造函数逐项比较。
启动 JSON 设置 `use_panel_cache:true` 时读取缓存，`args` 是临时覆盖值；模型
`config_path:"__panel_cache__"` 使用快照。可带 `panel_cache_sha256` 校验页面加载
时的参数指纹。显式不启用缓存的 CLI/API 请求仍可直接提交全部训练参数。

```bash
cargo test            # 配置/路径安全/磁盘策略/API/任务存储/训练模式切换等
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
