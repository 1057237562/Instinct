# 资料库备份

`2026-10-01.json` 保存 https://dlut-cpc.wannafly.cn/api/resources 的 55 条公开资料、14 个分类和标签，包含原始 ID、标题、简介、类型、难度、链接和创建/更新时间。备份仅含已公开条目，不包含管理员凭据或未公开草稿。

网页及 GitHub 资料保存的是索引和链接，不是外部网站正文。PDF 原件已由独立仓库 Code92007/dlut-cpc-resources 的 Git LFS 保存；本备份保留对应链接。

可以从 JSON 恢复到一份新的数据库（不要覆盖正在使用的生产数据库）：

```python
import json
from pathlib import Path
from database import Database

snapshot = json.loads(Path("data/resource_backups/2026-10-01.json").read_text())
db = Database(Path("runtime/resources-restored.sqlite3"))
db.initialize()
assert not db.list_resources(include_drafts=True), "只恢复到空资料库"
for item in snapshot["items"]:
    db.save_resource(item, created_by="GitHub backup restore")
assert len(db.list_resources()) == len(snapshot["items"])
```

恢复会生成新资料 ID 和时间；原始 ID 及时间仍可在备份中查阅。今后的新增、修改需要重新备份，本文件不是实时镜像。
