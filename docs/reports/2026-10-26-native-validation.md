# 2026-10-26 原生 cuBLASLt 优化验证

结论：本轮原生构建、融合算子、Muon、packed attention、数据游标和保存验证完成。
生产 Parquet 已建缓存的训练窗口为 **11,398.30 有效 next-token targets/s**；
尚未达到 14,000 或 18,000 targets/s。

所有源码改动位于 `codex/cublas-throughput`，基于 `5f3f388` 加当前工作区改动。
没有用 PyTorch 训练结果替代原生结果。实际训练进程的 Torch 导入检查为 false。

## 验证结果与修复

| 项目 | 结果 |
|---|---|
| cuBLASLt 和可选 cuDNN C++ bridge | 两个 Release 库构建成功 |
| CPU 广回归 | 初轮 123 passed / 1 failed；发现分桶预算漏出 JSON 导出白名单 |
| 导出修复及新增配置/缓冲依赖回归 | 29 passed，包含 WebUI 实际命令构建、预算导出、延迟读者保护 |
| 融合算子 GPU 对照 | 28 个 BF16/FP8 全模型 loss/VJP 用例通过；loss 与非嵌入梯度逐位一致，嵌入原子累加差异约 10⁻⁸ 相对 L2 |
| Muon GPU 更新 | Triton/cuBLAS × 两种 allocator × 两种动量 dtype × NS0/5，共 16 个用例通过 |
| Muon 生产矩阵形状 | 9 组 Gram / polynomial / update GEMM 与原生 Triton 输出逐位一致 |
| cuDNN 回放/隔离 | 8 个 ckpt2/4、BF16/FP8、交错 forward 用例按 BF16 契约通过；因果和文档隔离检查通过 |
| 完整原生训练 | 160 个微步、20 次实际优化器更新，未发生 OOM，结束时累计梯度为 0 |
| 保存与游标审计 | 当前权重、优化器、RNG、数据游标与保存文件逐项核对通过；独立重放数据批次后 token 计数与游标一致 |

cuDNN 原始严格梯度对照曾在 4096 长度出现 1 个 q_proj 元素不满足
`rtol=0.002, atol=2e-5`，绝对差为 3.0517578e-5，即该值的一个 BF16 量化步。
没有抹去该失败。cuDNN Frontend 1.31.0 的
`include/cudnn_frontend/node/scaled_dot_product_flash_attention.h` 明确拒绝 SM12X 的
deterministic ragged backward。重复参考也存在浮点累加波动。
最终单独检查点误差 `rtol=1/128, atol=2e-5`，并同时要求每个梯度张量相对 L2 < 0.001；
最终记录最大相对 L2 约 1.76e-6。**这条 cuDNN 路径不保证逐位重复**，不称为 bit-exact。
融合算子的原有逐位检查没有放宽。默认 attention 仍为 Triton。

生产大小 Muon 示例（固定随机 BF16 输入，无训练更新）：

| batch × M × N × K / 运算 | Triton ms | cuBLASLt ms |
|---|---:|---:|
| 2 × 1536 × 1536 × 5504 / Gram | 0.8001 | 0.5462 |
| 2 × 1536 × 1536 × 1536 / polynomial | 0.1591 | 0.1571 |
| 2 × 1536 × 5504 × 1536 / update | 1.0580 | 0.5300 |

上述是单独 GEMM 计时，不能当作完整优化器或训练吞吐。

## 生产训练条件

- RTX 5070 Ti 16 GB / SM120；CUDA Toolkit 13.1，Triton Windows 3.4.0.post20；
  cuDNN 9.27.0.42、Frontend 1.31.0。
- Instinct V2 Dense：32 层，H1536，FFN5504，Q/KV16/4，D96，vocab6400，
  共享 embedding/head，1,010,277,888 参数。
- BF16 隐藏权重和计算、FP32 embedding/head/norm、uint16 mantissa、BF16 momentum；
  tensorwise FP8 + auto filter，`fused_ops` 开启。
- 生产源：`D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet`。
  Rust 打包缓存已就绪；本次没有冷块构建事件。mmap 取批、H2D 和块切换仍在计时内。
- 两桶 4×3072 / 3×4096，token budget 12288，accumulation8。
  LR0.0005、weight decay0.1、grad clip1，Muon batch≤16、workspace512 MiB、5 次 NS。
- Muon GEMM 使用 cuBLASLt；Lt 候选调优开启，共享 Lt workspace32 MiB。
  attention 使用 hybrid：短桶 Triton128×64/8-warps，长桶原生 cuDNN。
  ckpt4 候选输出预算分别为 1024 / 1280 MiB。
- 从第 0 块末尾 64 批开始，跨入第 1 块前 96 批。省略的前缀没有训练；
  这是独立诊断权重和抽样游标，不是完整语料已训练到该位置的证明。
- 前 4 次累计更新预热，后 16 次更新计时。有效 targets 来自实际 labels，
  排除 padding、文档开头和移位后无目标的位置。所有窗口均计入，没有挑最快窗口。

## 吞吐和显存

| 指标 | 结果 |
|---|---:|
| 128 个测量微步有效 targets | 1,518,127 |
| 测量窗口墙钟时间 | 133.1889 s |
| 训练窗口有效 targets/s | **11,398.30** |
| 与旧运行相同的前 64 个测量微步 | 11,174.70 targets/s |
| 旧运行相同 64 微步 | 9,949.37 targets/s |
| 相同批次历史对照增益 | **12.32%** |
| 新 peak native live | 12.8315 GiB |
| 旧 peak native live | 12.5771 GiB |
| 同口径 live 增量 | **2.02%** |

历史对照使用完全相同的 755,988 个 targets，但旧运行有后台冷块准备，
本次缓存已就绪；它不是仅改变算子的严格同期 A/B。不能把全部 12.32% 归因于单个融合。
native live 是本后端的活跃张量字节数，不冒充 PyTorch allocated；
driver pool reserved 和 arena 也没有相加。

从进程启动到训练器完成最终保存共 **228.3608 s**，实际处理 **1,903,117** 个 targets，
因此这次短作业包含启动、预热、加载、更新和最终保存的吞吐是 **8,333.82 targets/s**。
其中预热在启动后 58.1293 s 结束，最后一次更新在 191.3182 s 完成。
随后独立保存文件核验另耗约 45.38 s，整项验证共 273.7363 s。
训练窗口不含启动和终止保存；本次未到 save_interval1000，不能据此声称长期含周期保存吞吐。
这轮也不建立长期稳定性或收敛结论。

## 证据与当前状态

以下文件位于 `experiments/results/cublas/validation_20261026/`：

- `build.log`、`options_cpu.log`、`fusions_gpu.json`。
- `muon_gpu.log`、`muon_shapes.json`。
- `attention_replay.log` 保留原始失败；`attention_repeat_diagnostic.json` 保留重复参考诊断；
  `attention_replay_bf16_contract.json` 是最终有明确数值契约的验收。
- `memory_preflight.json` 是无权重更新的显存预检，不作训练吞吐结论。
- `production_hybrid_cublas.json` 和 `.log` 是完成的训练记录；`.exit` 为 0。
- `production_audit.json` 独立验证所有测量 token 计数、20 次优化器更新、0 个未完成累计、
  最终游标和相同批次历史对照。

当前完整状态：
`D:/AI/Instinct/checkpoints/native_cublas/native_production_20261026_233626_44012_1536_resume.npz`

匹配权重：
`D:/AI/Instinct/out/native_cublas/native_production_20261026_233626_44012_1536.npz`

所有训练进程已正常保存退出，测试和生产验证均串行使用 GPU。
大数据、权重、resume、SDK 和生成的日志都保存在忽略目录。

复现本轮生产窗口：

```powershell
python -u -m experiments.bench_native_production `
  --data D:/AI/Instinct/dataset/pretrain_coder_l2l3_24b.parquet `
  --output experiments/results/cublas/production_recheck.json `
  --warmup-updates 4 --updates 16 --tail-batches 64 --tune `
  --attention-backend hybrid --muon-gemm-backend cublas `
  --selective-cache-by-seq 3072:1024,4096:1280
```

需要已构建的两个 native bridge、Rust dataset_compiler，以及正确的 CUDA_PATH/PYTHONUTF8。
该命令创建独立诊断权重，并在预定优化器更新边界保存当前完整状态后退出。
