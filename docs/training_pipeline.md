# 训练流水线

配置 WebUI 中展开「训练流水线 · Pretrain → SFT」：

1. 在训练面板选择 `pretrain`，设置模型、数据集、精度、学习率等，点击「保存当前配置到流水线」。
2. 选择 `full_sft` 并调整参数，再次保存。两份配置是独立快照。
3. 检查展示的配置，点击「启动训练流水线」。SFT 自动加载本次预训练输出。

支持导出/导入 JSON 预设。「刷新流水线进度」显示阶段状态及当前阶段日志。
「暂停当前阶段并停止后续阶段」请求训练器在 step 边界保存断点退出。
暂停后可通过原训练器加载所保存的 resume checkpoint；再次启动流水线会创建全新运行。

每个阶段通过独立进程运行。进程退出后释放其 CUDA 上下文，再删除该阶段独占的
Hugging Face datasets 缓存（加载、分词、packing 生成的 Arrow 文件）和临时文件。
清理失败会停止流水线。共享模型下载缓存、编译缓存、原始数据、权重、续训检查点、配置和日志保留。
只清理本次流水线阶段生成的独立缓存目录；已有历史数据集缓存不会被扫描删除。
阶段退出非零（包括暂停）时不会启动后续阶段。

配置及日志位于 `trainer/pipeline_runs/<运行ID>/`，权重位于 `out/`，
resume checkpoint 位于 `checkpoints/`。阶段使用唯一输出前缀。

也可从仓库根目录运行导出的预设：

```powershell
python trainer/training_pipeline.py --plan training_pipeline.json --validate-only
python trainer/training_pipeline.py --plan training_pipeline.json
```
