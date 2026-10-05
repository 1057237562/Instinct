# DLUT CPC 资料库

大连理工大学程序设计竞赛集训队的公开课件仓库。PDF 使用 Git LFS 保存，网站代码位于
[`Code92007/dlut-cpc`](https://github.com/Code92007/dlut-cpc)。

## 首次配置

```bash
brew install git-lfs
git lfs install
git config core.hooksPath .githooks
```

## 添加 PDF

按主题建立目录并放入文件，例如：

```bash
mkdir -p resources/pdfs/graph
cp /path/to/network-flow.pdf resources/pdfs/graph/

python3 tools/check_resource_pdfs.py
git add resources/pdfs
git commit -m "Add network flow courseware"
git push origin main
```

推送成功后，在 DLUT CPC 网站管理后台的“资料库”中新增 PDF，填写仓库内路径，例如：

```text
resources/pdfs/graph/network-flow.pdf
```

## 免费额度保护

`pre-push` 钩子会统计 `resources/pdfs/` 当前文件和 Git 历史中的不同 PDF 对象。累计超过
`9,000,000,000` 字节时会拒绝推送，为 GitHub Free 的 10 GiB LFS 存储额度留出余量。

同一 PDF 每次修改都会产生新的完整 LFS 对象，因此替换大文件也会增加存储用量。不要在 GitHub
网页端直接上传 PDF，因为网页上传不会执行本地的 9 GB 检查。
