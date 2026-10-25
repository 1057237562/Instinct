# 2026-10-25 原生 cuBLASLt V2 实现与吞吐验证

本轮基于远端 `experimental/dense-cublaslt-fp8` 的 `95db35e`，工作分支
`codex/cublas-throughput`。已移植原生 ckpt4、Muon split weights/BF16 momentum、
按形状分块的 workspace 控制和 packed 文档隔离/位置重置，并接通原生保存恢复、CLI 与 WebUI。
实现与运行入口见 `docs/native_v2_recipe.md`。

**结论：最终原生 ckpt4 相对同版原生 ckpt2 提高 6.80% 吞吐，native live 峰值增加
5.88%。它仍慢于这台机器上现有的 PyTorch ckpt4。**
配置选择为缓存预算 768 MiB、Muon workspace 512 MiB、async allocator、32×32 attention tile、
启用 packed 范围裁剪。Lt algorithm autotune 没有纳入共享配方。

## 条件与统计方式

- RTX 5070 Ti 16 GB / SM120，Windows；Triton Windows 3.4.0.post20。
  原生 bridge 使用本机 CUDA Toolkit 13.1；PyTorch 参考为 2.13.0+cu132。
- V2 Dense：32 层，H1536，FFN5504，Q/KV16/4，D96，vocab6400，1,010,277,888 参数。
- BF16 hidden weights/compute，FP32 embedding/head/norm，uint16 mantissa，BF16 momentum；
  tensorwise FP8 + auto projection filter，Muon batch≤16 / workspace512 MiB，accumulation8。
- 固定合成批次交替：4×3072，每行 3 个文档；3×4096，每行 1 个文档。
  每次累计更新包含 98,244 个有效 next-token targets。
- 每组 5 次实际更新：前 2 次预热，后 3 次计时。表中吞吐统一为
  **三个测量窗口有效 targets 总数 / 总墙钟时间**，保留慢步，不挑最快窗口。
  包含 forward/backward、clip、Muon/Adam 更新；排除 loader、冷编译、保存。
- GPU 实验串行。完成后保存 full resume + matching weights 并逐项检查；
  这些短窗口不建立真实语料吞吐或长期收敛结论。两套后端各自初始化，
  NumPy/PyTorch RNG 与原生 FP8/GEMM 舍入也不相同，不能跨后端比较 loss 高低推断质量。

## 完整累计更新结果

全部 native 行均启用 async allocator；`未裁剪` 指 attention 仍计算然后 mask 掉跨文档块。

| 配置 | 平均秒/update | 三窗口范围，秒 | 有效 targets/s | peak native live，GiB |
|---|---:|---:|---:|---:|
| 原生 ckpt2，未做 packed 范围裁剪 | 16.061 | 16.037–16.101 | 6,116.9 | 12.052 |
| 原生 ckpt4，1792 MiB，未裁剪 | 21.319 | 14.658–33.448 | 4,608.2 | 13.659 |
| 原生 ckpt4，768 MiB，未裁剪 | 15.182 | 15.180–15.183 | 6,471.0 | 12.761 |
| 最终原生 ckpt2，范围裁剪 | 13.959 | 13.944–13.988 | 7,037.9 | 12.052 |
| 最终原生 ckpt4，768 MiB + 范围裁剪 | 13.070 | 13.066–13.074 | 7,516.8 | 12.761 |

在已经具备这些新特性的原生 ckpt2 上，packed 范围裁剪与 ckpt4 配方合计提高
**22.89%** 吞吐；其中同一最终后端上的 ckpt4 相对 ckpt2 增益为 **6.80%**。
最终 ckpt4 的 peak arena 为 12.987 GiB。

1792 MiB 配方出现 33.448 秒的测量慢步，CUDA pool reserved 高水位达到 15.1875 GiB；
768 MiB 配方三窗口为 15.180 / 15.183 / 15.183 秒（裁剪前）。较大的池高水位与抖动同时出现；
本轮没有 page-fault trace，不能把慢步的原因直接定性为 WDDM 换页。最终选择依据是实测稳定性与吞吐。

同机、同 bucket/累计/精度设置的现有 PyTorch ckpt4：平均 **7.891 秒/update**，
**12,449.5 targets/s**，峰值 allocated **12.508 GiB**，
reserved **13.707 GiB**。最终原生吞吐为它的
**60.38%**，尚不支持以提高 MFU 为由替换现有 PyTorch 训练。
PyTorch 编译设置为 `max-autotune-no-cudagraphs, dynamic=False`，Muon workspace512。

### 显存口径

native live 按真实活跃张量字节记账；arena 加入自有空闲缓存及 Lt workspace。
CUDA async pool 的 reserved/used 另有统计，并不等于 arena，也不能与 arena 相加。
PyTorch allocated 包含其 allocator 粒度影响；表中的 native live 不能直接冒充它。
**5.88% 增量只在两个 native 配置之间计算**，没有混用跨后端计数器。

## 查到并验证的优化

1. **流有序分配。** 原始 native driver 分配器的两次前反向中，4,980 次
   `cuMemFree_v2` 合计耗时 5.45 秒；还有 6,543 次 malloc。固定权重 probe 中，
   ckpt4/1792 MiB 从约 1,158 targets/s 提升到约 6,619 targets/s，live 峰值不变。
   这是前反向 probe，不能当作完整 optimizer-step 的倍数。
2. **按文档范围跳过 attention 块。** forward/dQ 从 query tile 的最早文档起点开始，
   dKV 在 key tile 的最晚文档终点结束；跨边界 tile 仍使用逐元素 mask。
   4×3072 的前反向从约 1.81 秒降至 1.28 秒；无新增持久 GPU tensor。
   GPU 上不对齐文档、重复原始 ID、padding 的 mask-only 对照，loss 与全部梯度差异为 0。
3. **限制候选输出缓存。** 1792 MiB 仍可显式选择；本机共享配方用 768 MiB，
   3072 保留 32 层 K/V，4096 保留顶部 20 层 attention O/LSE。两者均重算 gate/up。
   理论缓存 FLOPs/bytes 不能代替包含临时量、driver pool 与优化器的实测。
4. **大张量 FP8。** 原实现上限 67,108,864 元素，小于 12,288×5,504=67,633,152；
   分级 amax 归约移除了这一限制。完整 V2 FP8 前反向和更新已实际覆盖该形状。

未采用的候选：64×64 attention tile 的初筛出现慢步，完整配方保留原 32×32；
Lt 候选算法计时仅在固定权重筛选中带来小幅变化，没有作为本轮推荐的完整训练设置。
所有候选的 JSON 留在 `experiments/results/cublas/`。

## 验证记录与边界

- 扩展 CPU/解释器回归：**198 passed、10 skipped、1 failed**。
  失败是旧 `test_legacy_full_state_next_update[muon-False-True]`，
  在从未修改的 `95db35e` 导出的独立源码上同样复现；未放宽容差。
- 后续变更的功能回归：`final_new_features.log` **29 passed**；
  loader 与真实 Streamlit AppTest 最终 **6 passed**；八种 trainer 的 stop/save 回归 **8 passed**。
- 原生 CUDA Muon：两种 allocator × 两种 momentum × NS0/5，**8 passed**；
  NS0 拆位更新与独立 oracle bit-exact，NS5 按 BF16/Tensor Core 容差检查。
- 独立 Torch-free CUDA 验收覆盖 BF16/FP8、packed/4096、ckpt2/4 全部梯度、
  两种 momentum 当前状态保存恢复；最终报告 `final_gpu_acceptance.json`。
  原生 npz mantissa 必须是 uint16，精度不匹配或浮点低位数组在写入模型前被拒绝。
- 原生 JSONL loader 的目标守恒、暂停累计边界、UI/CLI recipe 同源均有检查。
  **没有移植主仓库 Parquet/streaming/token-budget loader**；formal native 配方提供固定
  3×4096，可显式改为 4×3072。本次交替 bucket 是计算基准，不是原 corpus cursor 迁移。
- 首次会话中断留下的 `v2_ckpt4_async_updates.json` 标记为
  `interrupted_unverified`，无完成窗口/已验证保存，不计入结果。

## 证据与复现

所有下列 native 报告位于 `experiments/results/cublas/`：

- `native_ckpt2_final.json` → `native_v2_ckpt2_20261025_231204_17880_resume.npz`，5 updates / 40 microsteps，当前状态逐项校验通过。
- `native_ckpt4_1792_final.json` → `native_v2_ckpt4_20261025_230421_5948_resume.npz`，5 updates / 40 microsteps，当前状态逐项校验通过。
- `native_ckpt4_768_final.json` → `native_v2_ckpt4_20261025_230800_10876_resume.npz`，5 updates / 40 microsteps，当前状态逐项校验通过。
- `native_ckpt2_packed.json` → `native_v2_ckpt2_20261025_232425_18252_resume.npz`，5 updates / 40 microsteps，当前状态逐项校验通过。
- `native_ckpt4_packed.json` → `native_v2_ckpt4_20261025_232731_35820_resume.npz`，5 updates / 40 microsteps，当前状态逐项校验通过。

PyTorch 参考：`D:/AI/Instinct/experiments/results/cublas_reference_20261025.json`；
已验证 resume：`D:\AI\Instinct\checkpoints\ckpt4_recipe_20261025_231625_29200_1536_resume.pth`。
新的 `.npz/.pth`、权重和日志均在 ignored artifact 目录，没有写入源码提交。

最终 native 比较命令（分别选 2、4）：

```powershell
python -m experiments.bench_native_recipe --mode 4 --cache-mib 768 --allocator-backend async --updates --warmup 2 --steps 3 --output experiments/results/native_ckpt4.json
```

复现裁剪前的路径加入 `--no-packed-skip`。不加 `--updates` 是固定权重前反向，
它仍保留真实优化器状态和梯度。使用 `--tune` 或 `--attention-tile 64` 可重测未采用的候选。

GPU 验收：

```powershell
python -m experiments.validate_native_v2 --allocator async --output experiments/results/native_v2_validation.json
python -m pytest tests/test_native_muon_gpu.py -q
```

Windows 运行前设置 `PYTHONUTF8=1` 与正确的 `CUDA_PATH`，并先按分支原文档构建 native bridge。
与 native 默认 driver 分配器相比，async 的生命周期要求参见
[CUDA Driver API](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html)。
