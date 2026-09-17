# 紧凑数据缓存 v2

Pretrain/SFT 的 fixed 与 bucket packing 写出紧凑 Arrow：token ID 根据 tokenizer 全词表
及 special token 范围选择 uint16 或 uint32；序列 ID 改存有效片段长度；SFT labels
改存 Arrow boolean 监督掩码。训练读取时还原 long tensors，保留 padding 和跨样本 loss 屏蔽。
旧式逐 token sequence_ids / labels 的读取仍兼容。更改 schema 会生成新的缓存指纹。

配置 WebUI 与训练流水线自动启用独立进程预处理。CLI 可设置：

```powershell
$env:PYTHONUTF8='1'
$env:INSTINCT_MANAGED_DATA_CACHE='1'
python trainer/train_pretrain.py --sequence_packing 1
```

只有 packing 开启时使用该流程。非 packing 继续按原来的按需分词路径读取。

独立进程在专属 build 目录加载、分词、packing；bucket 模式分词结束后尝试释放原始
Arrow，packing 结束后释放分词 Arrow。Windows 仍占用的文件延迟到子进程退出后清理。
父进程只重命名最终分片到发布目录，不再复制整个最终数据集。manifest 最后随目录发布，
记录格式、源码和输入指纹、桶信息、样本数、分片顺序与大小。后续进程直接 mmap 最终分片。
数据内容采用完整 SHA256，因此首次检查大型输入仍有一次顺序读盘成本。

多个进程使用同一指纹锁；现有 DDP rank 0 构建屏障保持不变。数据子集、种子和所有构建
参数进入指纹，避免切换 packing 或断点恢复时误用不同的样本顺序。存储格式不改变模型接口。

共享目录为 `.cache/huggingface/datasets/instinct-packed/`；流水线使用阶段私有目录。
只清理新任务明确拥有的中间文件；历史缓存不自动删除。流水线阶段结束仍清理阶段全部数据缓存。
中断或机器断电遗留的 build 目录不自动扫描删除。缓存校验失败会报错而不会使用部分数据。

当前版本未调整分桶算法，也未采用有损数据转换。实际大数据集峰值、DDP 多卡及训练吞吐
需要在对应硬件上进一步验证；小数据子进程和已有 packing 测试不代表这些指标已验收。
