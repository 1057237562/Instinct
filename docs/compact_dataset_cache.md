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
默认对整个 `HF_DATASETS_CACHE` 应用 5 GiB 总预算（`--data_cache_max_gb 5`）。超限时按
LRU 顺序清理可重建内容：未完成 build、下载缓存、普通 Arrow，最后才是计算成本较高的
packed cache。训练进程会为正在 mmap 的 Arrow 发布 PID 租约，其他进程不能淘汰这些文件；
异常退出留下的租约会在下次检查时自动回收。设为 `0` 可关闭配额。

## 大型 JSONL 的有界流式展开

当单个预训练 JSONL 的整库 Arrow 预估超过缓存预算时，`--dataset_streaming auto`
会自动启用分片模式。训练器先顺序扫描一次 JSONL，生成仅包含换行对齐字节范围、行数和
token 总数的小型计划；包含 `token_count` 的 Instinct 数据不会在此阶段重复分词。随后每次
只物化一个源分片，完成 tokenization/packing 后立即训练；当前分片结束即释放 mmap。最终
packed Arrow 作为 LRU 分块缓存保留，可被重启和后续 epoch 直接复用；只有达到磁盘预算时
才淘汰最旧分块，因此总占用仍受 `--data_cache_max_gb` 严格限制。

```powershell
python trainer/train_pretrain.py `
  --sequence_packing 1 --dataset_streaming on `
  --streaming_chunk_mb 1024 --data_cache_max_gb 5 `
  --streaming_prefetch_chunks 1 --cache_build_mode inline
```

默认使用相邻分片双缓冲：GPU 训练分片 A 时，rank 0 的后台线程同时展开、
tokenize 并 packing 分片 B；A 结束后所有 rank 确认 B 已发布，再释放 A 的 Arrow。
`--streaming_prefetch_chunks 0` 可退回串行的逐片构建。开启预取时按最坏约
`5 × source chunk` 校验空间（关闭时为 `4×`），5 GiB 缓存建议保持默认
1024 MiB 分片。
分片边界只落在完整 JSONL 行之间，不丢行、不重复行。余弦学习率按全局实际消费 token
推进，不会在新分片开始时重启。checkpoint 额外保存 chunk/片内 batch/token 游标；续训要求
数据文件、分片大小、packing 配置和 GPU 数量不变。

`--cache_build_mode inline` 是默认安全路径：在当前训练进程内构建分片（可由上述单个后台线程执行），并在函数返回后
显式 GC/释放 Arrow mmap。流式预训练在该模式下会强制 `packing_num_proc=1`、
`bucket_loader_workers=0`、`num_workers=0`，因此数据路径不会创建任何 Python spawn 子进程。
PyArrow 仍可使用原生线程。需要用独立进程隔离预处理时可显式选择
`--cache_build_mode spawn`，但 Windows/Codex 环境不推荐。

`--packing_num_proc` 现在同时控制原始 JSON 分片读取、分词、fixed packing 和 bucket
packing。多个 JSONL 输入分片会并行读取；单个 JSONL 仍由 PyArrow 内部线程解析。预处理
进程直接写出彼此独立的 Arrow 分片，父进程仅重命名发布，因此并行化不会再复制一份完整
数据集，也仍受同一个 5 GiB 预算约束。

默认 pytest 不再启动昂贵的 Windows spawn 端到端缓存测试；worker 环境变量和配额行为由
快速单元测试覆盖。需要完整子进程验证时显式执行
`python -m pytest tests/test_compact_cache.py --run-slow`。

流水线阶段结束仍会删除该阶段全部数据缓存。缓存校验失败不会使用部分数据；如果单个活跃
数据集本身超过预算，训练会明确报错，需减小数据/packing 规模或提高预算，而不会删除原始数据。

分片内仍使用既有 fixed/bucket packing；全局最优分桶变为分片内最优分桶，这是有界缓存的
必要折中，不改变 token 内容。实际 12B-token 数据集的峰值、DDP 多卡及训练吞吐仍需要在
目标集群验证；小数据子进程和已有 packing 测试不代表这些指标已验收。

Auto buckets 的 `--bucket_gpu_memory_gb` 是每卡硬预算。batch 规划会先扣除全部参数（MoE
包括未激活专家）、梯度、Muon/AdamW 状态、0.5 GiB 安全余量，并按 bucket 数为编译图和
allocator 碎片保留空间，再把剩余显存换算成 activation token budget。因而增加专家数、
bucket 数或启用 CUDA Graph 都会自动降低每桶 batch；日志中的
`[Packing VRAM Budget]` 和 `[Packing Batch]` 会打印实际采用的预算与 batch_size。
此外训练启动时会把 PyTorch CUDA caching allocator 限制为“用户预算减 0.5 GiB”，给 CUDA
context、cuBLAS/NCCL 等非 allocator 显存留空间；日志以 `[Packing VRAM Limit]` 显示硬上限。
