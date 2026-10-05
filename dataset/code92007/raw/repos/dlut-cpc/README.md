# DLUT CPC

大连理工大学程序设计竞赛队的成绩、成员、Rating 与训练记录站点。首版参考 [CVBB ICPC Team](https://cvbbacm.com/home) 的信息架构，采用与 `oj-submission-wall` 相同的轻量部署思路：Python 标准库后端、静态前端、Docker Compose 和反向代理。

> 本项目是队伍信息展示站点，不代表大连理工大学官方。校徽素材来自大连理工大学[学校 VI 页面](https://www.dlut.edu.cn/xxgk/dxwh/xxVI.htm)。

## 页面

- `/home`：本地已收录的历年金、银、铜成绩，可切换分组柱状图与累计柱状图，并展示最近参赛；铁牌仍保存于成绩库。
- `/honor`：按年份、成绩、关键词检索获牌及未获牌记录。
- `/rating`：从历年获奖名单归并出的完整成员目录，以及已确认的 Codeforces 账号。
- `/training`：去训练的入口，包含队伍 B 站讲课视频、历届校赛院赛、Hydro 训练团队和队伍 OJ。
- `/resources`：训练资料知识库，统一收录 PDF、GitHub 仓库和外部网页，可按关键词、类型、分类、难度与标签检索。
- `/admin`：管理员登录与成员、账号、姓名别名、历史参赛成绩补录。访客只读。
- `/pending`：获奖信息待确认成员，可按年份关键词、队伍及所属范围查找。

页面不依赖前端构建工具，图表使用原生 SVG 渲染。

公开参赛成绩表及首页最近参赛中的“铁牌”文字默认透明，选中文字时显示；保留原位置、复制内容及筛选。成员奖牌榜、待确认列表和管理员审核中的铁牌仍可见，数据库记录和计数不受样式影响。

## 本地运行

```bash
python3 app.py --check
python3 app.py
```

打开 `http://localhost:8000`。

运行测试：

```bash
python3 -m unittest discover -s tests -v
```

有 Node.js 的环境还可执行前端回归测试：`node --test tests/*.mjs`，包括排序、审核、柱状图及中文输入法组合输入。

## 数据

`data/site.json` 是可版本控制的公开数据快照，`runtime/dlut_cpc.sqlite3` 是运行时主数据库。首次启动会将快照导入 SQLite，之后自动同步只更新公开数据，不会删除人工录入的成员、账号关联或历史奖项。数据库支持同一奖项和成员关联多个来源，并用 CPC Finder 学生 UUID 区分同名成员。

当前成绩与队员名单来自 [CPC Finder 的大连理工大学学校页](https://cpcfinder.com/school/9c417252-c487-4eae-8822-fcd1e74b9329)、学校获奖 API、选手目录 API、选手参赛 API 和各赛事榜单 API，仅保留 2020 年及以后的成绩。同步脚本会逐项关联 `awardId`、`contestId`、`teamId` 与稳定的 `studentId`，核对学校后导入榜单中的三位队员；也可以用 [ICPC 参赛公示](https://icpc.pku.edu.cn/docs/20230202164632701013.pdf)、[2024 上海站结果](https://icpc.pku.edu.cn/docs/20250313164218706132.pdf)、XCPCIO、Gym 或经过核验的 QOJ 镜像榜补充或覆盖。

学校别名按队内指定范围处理：大连理工大学、本部、开发区校区及旧称软件学院统一为“大连理工大学”；城市学院、盘锦校区（盘锦学院）为两个独立维护范围。三者可在总览汇总，也可在成绩、成员和待确认页面分别筛选。保留原始榜单学校名，不因同名跨范围合并成员。CPC Finder 的校内奖牌汇总作为成员页公开基准，队内数据库仍可补录更早成员、账号和历史赛事。

队内已确认盘锦范围没有同名同姓的独立选手，因此仅在盘锦按规范化姓名合并重复成员。保留所有来源选手 UUID、参赛关联、账号、报名别名、人工资料、删除账号记录及审核历史，旧成员 ID 在 `member_redirects` 中映射到保留成员并归档被合并资料。更新后新的重复来源 UUID 也会复用同一人；本部、城市学院以及跨范围同名不合并。合并成员的奖牌次数从数据库中已确认的不同正式成绩计算，不相加两份来源汇总；CPC Finder Rating 保留已有资料中的最高值，不把两个人物页的分数相加。

### 一次性历史导入

`data/historical_honors.json` 保存从 [RankLand](https://rl.algoux.cn/search) 归档的 2010–2019 年 102 条明确获牌成绩（ICPC 81 条、CCPC 21 条），及 88 场榜单的检查报告。只接受 2020-01-01 前的区域赛、总决赛金银铜成绩，排除省赛、邀请赛、预选赛和女生赛。奖牌由官方 SRK 工具按来源榜单配置解析；18 场榜单的奖牌边界全为零、14 场赛事类型无法确认，记录在检查报告中而不猜测结果。这批明确获牌记录均属于本部/开发区；不能据此断言另两个范围没有历史获奖。

每条结果保存稳定赛事及队伍编号、原始校名、来源链接、分数和奖牌配置。首次更新启动从仓库中的本地归档导入 SQLite，并以 `historical_import:rankland-pre2020-v1` 标记完成。之后启动、公开同步均跳过此批次，不重新请求 RankLand，也不重新加入已完成的待确认项。日常运行不需要安装爬取依赖，来源网站下线不会影响已导入成绩和人工数据。归档仅收录获牌队，无法证明古早成员的完整铁牌次数；无完整公开统计的历史成员显示“铁待补”，不按 0 参与奖牌榜比较。

历史队名不能证明队员身份，因此先进入 `/pending`。10 条榜单保留的参赛姓名作为参考，不包含教练，也不自动关联到个人。管理员在 `/admin` 的“待确认成员”选择成绩后可直接填写队员姓名：同一所属范围内按姓名、显示名和报名别名匹配，唯一匹配时复用现有成员，无匹配时创建往届成员；多个同名或同别名候选时须从名单中选择具体成员 ID。榜单姓名会预填，但必须人工点击确认才写入。创建成员、人工成绩关联和完成状态在同一事务内提交，重复成员、无效输入或所属范围错误会全部回滚。确认只移出待确认列表，绝不删除比赛成绩；完整确认后才计入个人奖牌。本部与开发区可跨校区组队，城市学院、盘锦不能误关联到本部成绩。

已经由独立公开名单明确到具体比赛和队名的成员可放入 `rosterCorrections`，在历史成绩导入后一次性确认。修正只填补空名单；线上已经确认或后来修改的名单保持不变。2010 成都站校内获奖表与现场榜单可按本校仅有的银牌队和优胜奖队唯一对应，已确认 `dlut_Phoebus` 为孙崇林、王丰田、裴立，`dlut_Real Power` 为刘彬、冷骞、崔文锋；核验边界见 [2010 ICPC 成都站名单说明](docs/icpc-2010-chengdu-rosters-20260923.md)。2015 CCPC 仅举办南阳站，官方参赛队伍信息可与南阳榜单按队名直接对应，因此已确认 `EX咖喱棒` 为许思航、邹家树、刘庆周，`为了梦想` 为马少楠、熊昆、刘小坤；核验边界见 [2015 CCPC 南阳站名单说明](docs/ccpc-2015-nanyang-rosters-20260923.md)。

2012 年校内科技竞赛获奖统计可与历史榜单按赛站和奖项唯一对应三队：天津站 `Solo` 为孙木鑫、王琳、张璨，天津站 `H$H/乱码` 为胡骏、孙崇林、洪祈泽，成都站 `Gospel` 为孙木鑫、王琳、张璨。金华站存在两支同奖项队伍，证据不足，继续留在待确认队列；核验边界见 [2012 ICPC 天津成都名单说明](docs/icpc-2012-rosters-20260925.md)。

2014 北京站校内报道的搜索索引摘要明确写明 `临时变量` 由邵华、刘博、许思航组成，并与北京师大赛站的 Internet Archive 榜单在校名、队名、解题数和银牌上相符，已补录该队。原报道现为 404 且无 Wayback 正文快照，因此保留索引恢复限制；证据链见 [2014 ICPC 北京站名单说明](docs/icpc-2014-beijing-roster-20260925.md)，本轮 42 队逐项核查见 [旧赛季待确认名单 Archive.org 核查](docs/pending-roster-archive-audit-20260925.md)。

归档已经随仓库提供，**部署时不需要再执行爬取**。仅维护者需要创建其他归档时使用可选工具：

```bash
python3 -m venv /tmp/rankland-import-venv
/tmp/rankland-import-venv/bin/pip install -r tools/requirements-rankland.txt
/tmp/rankland-import-venv/bin/python tools/import_rankland.py
```

默认输出已存在时直接停止，不请求外站。下载失败保留缓存但不写不完整快照；缓存和数据库位于忽略的运行时目录。手工确认不会修改仓库里的历史归档，生产数据库必须持续备份。

### CCPC 官方补缺

`data/ccpc_official_honors.json` 归档了 [CCPC 榜单](https://ccpc.io/rank)与[官方获奖名单](https://ccpc.io/award)中的 50 条奖项明确的成绩，并保留逐场检查报告、原始校名、奖项、名单行及来源链接。与 2026-09-16 线上数据库比对，8 条缺失、42 条已有；新增详情、匹配 ID 和排名差异见 [补缺报告](docs/ccpc-merge-20260916.md)。其中 7 条新增在 2020 年之前，另一条为 2025 年举行的第十届总决赛。

来源取并集，人工补录也属于独立来源。同一 CCPC 赛季、赛区、维护范围和队伍合并；支持官方列明的中英文队名别名。队名不同但同场、同范围的完整三人名单一致时可匹配整队，绝不根据某次组队推断另一场的成员。多个候选、同名次不同队、奖项冲突均跳过并出具报告。匹配成绩只追加来源及独立来源 ID，不覆盖已有字段、人工成绩或确认名单；以后公开来源收录归档成绩时复用同一记录 ID，避免二次计数。

排除网络选拔、女生专场、高职专场、省赛、邀请赛及热身赛；开放区域赛中的女队仍收录。官方“优胜奖”映射为铁牌，但无奖项标记或截断榜单不用于猜测铁牌，不能据此断言某年已查全。部分原表名次与 CPC Finder 正式排名口径不同，两者都保留为证据，不自动覆盖排名。

新成绩进入待确认队列，榜单姓名仅预填供人工核对，不直接关联个人。确认仍可复用已有成员或创建古早成员，游客可提交、管理员可审核和纠错。已确认的个人成绩加入现有统计；无完整公开参赛统计的历史成员仍显示“铁待补”，不会把未知铁牌总数当作 0。

归档随仓库部署，首次启动以 `official_import:ccpc-official-union-20260916-v1` 标记导入，此后启动、公开同步不重新爬取或重复导入。数据库归档表保留各来源原始行，不依赖外站日常可用性。**不要上传本地 SQLite 覆盖生产数据库**。

生产更新时先拉取代码，再对服务器实时数据库备份并预览；此工具不会调用初始化快照，`--dry-run` 不修改成绩、名单或导入标记（可能创建新版本表结构）：

```bash
cd ~/dlut-cpc
git pull --ff-only
python3 tools/merge_ccpc.py --dry-run \
  --backup "runtime/backups/before-ccpc-$(date +%Y%m%d-%H%M%S).sqlite3" \
  --report runtime/ccpc-preview.json
# 核对预览后更新容器；启动会再次按生产数据库去重
docker compose up -d --build
docker compose exec -T dlut-cpc python tools/merge_ccpc.py --report runtime/ccpc-import.json
curl -fsS http://127.0.0.1:8021/healthz
```

保留 `runtime`、`.env` 和既有 Caddy 配置，端口仍为原来的 `127.0.0.1:8021`。默认数据库为 `runtime/dlut_cpc.sqlite3`；自定义路径需给预览命令传 `--database`。`ccpc-preview.json` 是预览，`ccpc-import.json` 是实际导入报告；线上新增补录可能使实际新增数比报告的 8 条更少。

仅维护者生成后续新归档时需要可选的 `pdfplumber`，部署无需安装。`tools/import_ccpc.py --channel 6 --pdfs` 及 `--channel 18 --pdfs` 缓存官方原文与 PDF 提取数据；`--build --baseline <api-site.json> --rankland-cache <cache-dir> --batch-id <new-id> --output <new-snapshot.json>` 生成新批次。已导入批次不可原地修改；后续批次必须使用新 ID。现有归档可通过 `--compare --baseline <api-site.json> --report <report.md>` 离线重新对照。未知 PDF 表格格式会报错，须人工核对并补充解析，不直接猜测列位置或奖项。

### RankLand 参赛记录补缺

`data/rankland_supplement_honors.json` 单独归档本次指定的 27 场 ICPC 榜单，共 69 条本校队伍成绩。对照生成时的线上公开快照，59 条新增、10 条只追加来源、0 条冲突；逐场队名、奖项状态、排名及已有记录 ID 见 [完整补缺清单](docs/rankland-merge-20260916.md)。其中新增记录有 2 条银牌、12 条铁牌、36 条奖项待确认、9 条打星，不将缺少奖牌边界的榜单猜成铁牌。2020 银川站、EC Final 实际举办于 2021 年，保留来源比赛日期。2011 大连站的 7 支本校队伍按队内确认全部标记打星。

原先明确获牌归档不修改，此补缺批次以 `official_import:rankland-supplement-20260916-v1` 一次性合并。自动启动只读取随仓库提供的归档，之后跳过完成批次，不重复爬取。人工补录与其他来源取并集，匹配时只追加来源，不覆盖已确认的字段或名单；未知奖项可由管理员确认成员时补齐，已补录名单中也可补齐。后续可靠来源收录同一结果时复用原 ID，只填空缺奖项，不替换已有奖项、排名或人工名单。

先对实时生产数据库备份并预览，再更新容器，禁止上传本地数据库覆盖线上补录：

```bash
cd ~/dlut-cpc
git pull --ff-only
python3 tools/merge_ccpc.py --snapshot data/rankland_supplement_honors.json --dry-run \
  --backup "runtime/backups/before-rankland-$(date +%Y%m%d-%H%M%S).sqlite3" \
  --report runtime/rankland-preview.json
# 核对后更新，启动会按当时的生产数据库再次去重并合并盘锦重复成员
docker compose up -d --build
docker compose exec -T dlut-cpc python tools/merge_ccpc.py \
  --snapshot data/rankland_supplement_honors.json --report runtime/rankland-import.json
curl -fsS --retry 10 --retry-connrefused --retry-delay 1 http://127.0.0.1:8021/healthz
```

`rankland-preview.json` 是启动前的预览；`rankland-import.json` 是实际导入报告，包含每条结果的处理状态及警告。线上已经补录的队伍会匹配旧记录，实际新增数可能小于 59。端口、`runtime`、`.env` 和既有 Caddy 配置保持不变。维护者生成后续归档可使用可选工具 `tools/import_rankland_supplement.py --rank-id <key> --batch-id <new-id> --output <new-file>`，部署无须安装爬取依赖。

2018 ICPC 徐州站的“顶呱呱（Dingguagua）”以独立后续批次 `rankland-icpc2018xuzhou-dingguagua-20260916-v1` 补入，保留正式排名、原始校名和榜单证据。该队为正式参赛且在金银铜奖牌段之外，按本站口径记为铁牌；吴将凯、胡小涛、钱昕予仅作为待管理员确认的建议名单，不自动关联成员。核验详情见 [徐州站补录清单](docs/rankland-xuzhou-2018-followup-20260916.md)。

2016 ICPC 青岛站另以 `icpc-official-qingdao-2016-honorable-mention-20260921-v1` 后续批次补充 SDUT 保存的 ICPC Global 官方榜单证据：大连理工大学 `winter is coming`，以及原始校名为大连理工大学软件学院的 `Innovator`、`lingering_sound` 均列入 Honorable Mention，站内按已确认但未获金银铜牌的正式参赛成绩记为铁牌。三队解题数、并列名次及总榜位置沿用 RankLand 原始榜单；成员均留空待人工确认。核验详情见 [青岛站补录清单](docs/icpc-qingdao-2016-followup-20260921.md)。

2009 赛季另以 `icpc-official-dlut-2009-recovery-20260924-v2` 补录校内页面列出的 7 条亚洲区域赛成绩：4 银、1 铜、2 个荣誉奖（按站内口径记铁牌）。页面发表于 2010-01-20，但五个赛站均由 ICPC Global 的 2009 赛季赛事元数据交叉确认；武汉站 `Bombee` 与已有 RankLand 记录合并，合肥站也由 ICPC Global 官方榜单确认队名为 `Bombee`，其余未恢复的队名和名次明确留作待考。名单全部进入人工确认队列，核验详情见 [2009 ICPC 补录清单](docs/icpc-2009-recovery-20260922.md)。

2006 西安站另以 `icpc-official-dlut-2006-xian-recovery-20260924-v1` 补录校内同期报道中的 1 条优胜奖（按站内口径记铁牌），队员为刘炯、张岳、张剑峰。ICPC Global 可确认比赛日期、承办校和赛事 ID，但官方登记的西电旧榜单已失效且没有历史快照，因此队名和名次明确留作待考，名单进入人工确认队列。核验详情见 [2006 ICPC 西安站补录清单](docs/icpc-2006-xian-recovery-20260924.md)。

铁牌指结果已确认但未获得金银铜牌的正式参赛成绩。打星队伍保留参赛记录与“打星 · 非正式”标记，但不计年度奖牌、个人奖牌、铁牌或获奖次数；来源明确注明金银铜时显示“打星银牌”等，否则成绩留空。没有奖项依据的正式记录显示“奖项待确认”，不按铁牌统计。铁牌次数来自选手逐场正式参赛记录，并与学校榜单核对；未查全的次数显示“铁待补”，不会当作 0。“奖牌榜顺序”依次按金、银、铜数量降序及铁牌数量升序排列；“获奖次数”不含铁牌。

同步 CPC Finder 并执行归一化去重：

```bash
python3 tools/sync_public_data.py --dry-run
python3 tools/sync_public_data.py
```

脚本使用 `日期 + 赛事 + 规范化队名` 作为去重键，处理全半角标点、空格和常见队名格式差异，同时更新 JSON 快照与 SQLite。来自 XCPCIO、Gym、QOJ 或官方名单的补充数据可以整理成 JSON 数组后合并：

```bash
python3 tools/sync_public_data.py \
  --supplement data/xcpcio.json \
  --supplement data/qoj.json
```

补充记录字段与 `site.json` 中的 `honors` 项一致。比赛名次等结果字段可优先采用官方或经过核验的独立榜单，但 CPC Finder 按 `awardId` 返回的成员名单不会被普通补充来源覆盖；只有显式标记 `memberRosterManual: true` 的人工修订可覆盖。每次公开同步会清除已失效的非人工公开快照镜像链接；人工补录及一次性官方归档来源始终保留。

### 训练入口

Training 已替换旧多校榜单，分为讲课视频、校赛与院赛、Hydro 团队、Online Judge 四个栏目。管理员在“训练入口”页添加或修改名称、链接、简介、年份 / 届别和排序，并可取消公开以隐藏入口。资料持久化保存在 SQLite，重建及公开同步不会覆盖管理员修改。首启一次性导入队伍 B 站、Hydro 团队和临时 DLUT OJ；校赛、院赛与团队链接确认后在后台补充。`trainingLinks` 的 `seedId` 是一次性导入标记，不应复用来覆盖线上条目。

### 人工补录

账号关联的可版本控制快照放在 `data/site.json` 的 `accountBindings` 中，以 CPC Finder UUID 对应选手；该来源没有收录的成员可显式设置 `createMember: true`。每人可保存任意多个 Codeforces 账号。主号取所有账号中 `maxRating` 最高者，与加入顺序无关；最高分相同则按当前分、账号名字稳定排序。成员页分别显示主号最高分、当前分和副号列表，账号搜索包含副号。

管理员在“账号”页可追加、修改或二次确认删除已有账号。修改为不同账号后重新获取该账号评分，不沿用旧账号分数；删掉主号后自动从剩余账号中选取主号。删除记录保存在数据库的 `removed_member_handles` 中，启动与公开数据同步不会重新导入被删除的绑定；管理员明确重新追加时可以恢复。`accountCorrections` 是带唯一 ID 的一次性纠错记录，用于更新已有部署中的错误绑定，不反复覆盖管理员后续维护。杨君泓的错误绑定 `Lance_J` 已纠正为 `Farewell`。

“已补录名单”列出已确认的历史成绩及本地人工补录成绩，支持按比赛、队伍、成员和独立维护范围筛选。管理员可修改参赛名单，已有姓名 / 别名合并，未知姓名新建成员；可同时修改队名，原比赛及已知奖项不变，空缺奖项可以一并补齐；队名修改不会被后续公开同步覆盖，参赛关联和本地补录奖牌统计随之更新。修改事务失败时保留原名单；本地确认和修改后的名单不会被启动快照覆盖。游客仅可提交待确认成绩的补录申请，不能直接修改已确认名单。

`memberOverrides` 保存稳定选手 UUID、中文显示姓名与报名别名。例如 `Fangyu Bu` 显示为“卜方昱”，英文名仍可检索；映射同时作用于成员页和参赛成绩。管理员后续修改的显示名不会被启动快照覆盖。

服务启动后自动在后台拉取全部已绑定 CF 主副号的当前分与历史最高分，之后每 3 小时刷新一次，不阻塞页面请求。自动刷新只写入持久化 SQLite，包含管理员在线补录的账号。通过 `CF_REFRESH_INTERVAL_SECONDS` 调整间隔（秒），设为 `0` 可关闭；来源失败保留旧值并记录日志。浏览器重新加载 Rating 页即可读取最新数据。

手动立即拉取所有已绑定账号的当前分与历史最高分：

```bash
python3 tools/sync_codeforces.py
# 生产容器只更新已挂载数据库，不改容器内的公开快照
docker compose exec -T dlut-cpc python tools/sync_codeforces.py --database-only
```

Rating 来自 Codeforces 官方 `user.info` API。同步按账号名匹配，不依赖返回数组顺序；接口失败保留旧分数，没有评级的账号保存为空，不伪造 0。重启时仅导入不早于数据库评级更新时间的快照。CPC Finder Rating 是该平台选手资料中的独立分数，只用于展示与对应排序，不参与我们自己的奖牌榜排序。

### 网站管理员

部署或更新后，生成管理员密码并重启服务：

```bash
docker compose exec -T dlut-cpc python tools/manage_data.py init-admin
docker compose restart dlut-cpc
```

命令仅首次创建 `runtime/admin_password`（权限 `0600`），显示生成的随机密码；已存在时拒绝覆盖。用户名默认 `admin`，可通过 `ADMIN_USERNAME` 配置。密码文件在持久化卷中，不提交 Git，不从网站提供下载。也可以通过 `ADMIN_PASSWORD` 环境变量配置至少 12 位密码；未配置或密码过短时禁止登录。公开站点应始终通过 HTTPS 访问。

访问 `/admin` 登录后可补录未被 CPC Finder 收录的古早成员、追加主副账号、设置中文名和别名、录入历史成绩（金银铜铁）。成员选择含独立 ID，同名选手不会被自动合并；补录姓名已存在时需显式确认是独立同名成员。账号归属不能同时绑定两人。人工数据写入 SQLite，重建容器和公开同步均不会删除。

### 独立资料库与 Git LFS

PDF 存放在独立公开仓库 [`Code92007/dlut-cpc-resources`](https://github.com/Code92007/dlut-cpc-resources)，不进入本代码仓库、应用服务器或 Docker 镜像。网站数据库只保存标题、简介、分类、难度、标签和资料仓库路径，访客打开 PDF 时直接跳转到 GitHub。GitHub Free 当前包含 10 GiB LFS 存储和每月 10 GiB 下载流量；没有付款方式或预算设为 0 时，超额会停止 LFS 服务而不是继续扣费。

首次在维护电脑克隆并初始化资料仓库：

```bash
brew install git-lfs
git lfs install
cd ~/Documents
git clone https://github.com/Code92007/dlut-cpc-resources.git
cd dlut-cpc-resources
git config core.hooksPath .githooks
```

添加课件只在资料仓库中操作：

```bash
mkdir -p resources/pdfs/graph
cp /path/to/network-flow.pdf resources/pdfs/graph/
python3 tools/check_resource_pdfs.py
git add resources/pdfs
git commit -m "Add network flow courseware"
git push origin main
```

资料仓库的预推送钩子会统计 `resources/pdfs/` 当前文件及 Git 历史中不同 PDF 对象的声明大小，超过 9,000,000,000 字节时拒绝推送。GitHub 的 LFS 配额按账号统计，其他仓库中的 LFS 对象仍会占用额度；删除或替换课件也不应视为立即释放远端历史对象。最终兜底是 GitHub 在没有付款方式或预算为 0 时停止超额使用。

推送 PDF 后，在 `/admin` 的“资料库”中选择 PDF，填写类似 `resources/pdfs/graph/network-flow.pdf` 的路径。后端只接受 `resources/pdfs/` 下的 `.pdf`，拒绝绝对路径和目录穿越。删除资料索引不会删除 Git 历史中的课件；需要删除文件时另行提交 Git 变更。

仓库和分支默认指向独立资料库，也可以在 `.env` 显式配置：

```dotenv
RESOURCE_GITHUB_REPOSITORY=Code92007/dlut-cpc-resources
RESOURCE_GITHUB_BRANCH=main
```

管理 API 使用服务端八小时会话、HttpOnly/SameSite Cookie、HTTPS Secure Cookie、同源校验与 CSRF 令牌。未登录访客不能修改正式成员或成绩，也不能查看私有审核队列；登录失败有限流，退出立即撤销会话，服务重启后需重新登录。内部备注不进入公开成员接口。

### 游客补录与审核

游客在 `/pending` 的成绩行点击“补录成员”，可填写姓名或选择已有成员，附上补录依据后“提交审核”。提交仅写入 SQLite 的 `roster_submissions`，不会创建正式成员、确认名单或增加个人奖牌；未审核姓名和说明只向管理员开放。公开列表仅显示待审核份数。

游客也可在 `/rating` 点击“补充 CF 账号”，选择带成员 ID 和所属范围的已有成员、填写账号及依据，提交到 SQLite 的 `account_submissions`。提交不会修改正式账号，也不触发外部 Rating 请求。管理员在“游客审核”中将“补录类型”切换为“CF 账号”，查看申请账号、现有主副号和依据后点击“通过”或“不通过”。通过后追加账号并拉取最高 / 当前 Rating，主副号仍按最高 Rating 排列；拉取失败不撤销已通过的绑定，可稍后刷新 Rating。审批再次核验账号归属，不能把他人的账号绑定给这位成员。游客不能删除、修改或转移已有账号，纠错仍由管理员在“账号”页处理。

管理员在 `/admin` 的“游客审核”页查看名单与依据，点击“通过”或“不通过”，不需要再手填一遍。通过时在同一事务内按该成绩所属范围复用或新建成员、确认参赛关联、写入审核人及时间；不通过仅保留审核记录，不改正式数据。名单已由其他提案或管理员确认后，同一成绩剩余提案自动失效，不能覆盖已确认名单。审核支持状态、所属范围筛选和每页 50 份的分页。

提交必须同源并使用 JSON，名单与 CF 账号申请共用每个来源 IP 十分钟最多 20 次的限流；每条成绩最多保留 5 份不同的待审核名单，每位成员最多保留 5 份不同的待审核账号申请，两类总待审核队列上限 1000 份。相同成员名单不计次序、空格、已知报名别名差异自动去重；相同成员的账号申请忽略账号大小写与两端空格去重。存在多个同名候选时要求指定成员 ID，城市学院与盘锦校区不能误合并到本部。限流计数保存在进程内，重启后重置；提案、审核历史和正式数据均持久化，重建容器或来源网站下线不会丢失。

Compose 默认 `TRUST_PROXY_HEADERS=1`，用于读取 Caddy 设置的真实客户端 IP，宿主机端口必须保持 `127.0.0.1` 绑定且只由可信反向代理访问。直接将服务端口暴露到外网时应设置 `TRUST_PROXY_HEADERS=0`，不接受客户端伪造的转发头。此次更新不需要新端口或更改 Caddy 配置；更新前备份生产 SQLite，启动会自动升级到 v11。

无法从公开网站找到的老成员直接写入 SQLite，不需要修改前端：

```bash
# 查看成员 ID
python3 tools/manage_data.py list-members --query 张三

# 新增老成员；默认允许同名成员独立存在
python3 tools/manage_data.py add-member \
  --name 张三 --entry-year 2007 --graduation-year 2011 \
  --source-name 队史补录

# 给成员绑定经本人或队内确认的 Codeforces 账号
python3 tools/manage_data.py set-handle \
  --member-id 76 --handle example_handle --rating 2100

# 再执行 set-handle 可追加副号，原账号会保留
python3 tools/manage_data.py set-handle --member-id 76 --handle another_handle

# 中文显示名与报名别名
python3 tools/manage_data.py set-name --member-id 76 --name 张三 --alias 'San Zhang'

# 新增历史奖项，并通过成员 ID 关联队员
python3 tools/manage_data.py add-honor \
  --event '2009 ICPC 亚洲区域赛' --series ICPC --date 2009-10-18 \
  --location 大连 --team 历史队伍 --medal 银牌 \
  --member-id 76 --member-id 77 --member-id 78 \
  --source-name 队史补录

python3 tools/manage_data.py list-missing
```

只有在确认补录对象就是现有同名成员时，`add-member` 才应添加 `--match-existing`。Codeforces handle 不根据姓名猜测，只收录人工或可靠来源确认的关联。

备份和导出：

```bash
python3 tools/manage_data.py backup --output backups/dlut-cpc-$(date +%F).sqlite3
python3 tools/manage_data.py export --output backups/site-merged.json
```

旧训练榜单已从页面和当前公开快照移除，历史版本仍可通过 Git 查阅。

## Docker 部署

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps
```

默认只监听宿主机 `127.0.0.1:8021`。`docker-compose.yml` 将宿主机 `./runtime` 挂载到容器 `/app/runtime`，重建容器不会丢失补录数据。仓库提供两份反向代理配置：

- `deploy/Caddyfile.dlut-cpc`：`dlut-cpc.wannafly.cn` 的 Caddy 示例，自动申请 HTTPS 证书。
- `deploy/nginx.dlut-cpc.conf`：Nginx HTTP 反代示例。

生产部署前需要先为 `dlut-cpc.wannafly.cn` 配置 DNS，并确认服务器上的 8021 端口未被占用。

## 更新

```bash
python3 tools/manage_data.py backup --output backups/dlut-cpc-before-update.sqlite3
git pull --ff-only origin main
docker compose up -d --build
curl -fsS http://127.0.0.1:8021/healthz
```

公开数据同步后需要提交更新后的 `data/site.json` 并重建容器。生产环境中的人工补录只保存在已挂载的 `runtime/dlut_cpc.sqlite3`，应随服务器备份一并保留。
