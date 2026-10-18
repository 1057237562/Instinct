# 训练机自动调优

`trainer/autotune.py` 联合搜索整条预训练路径的性能参数。它像编译器 autotune 一样
试跑候选、缓存测量、选择通过验收的配置，但测量对象是完整训练 step 和真实数据
加载，不只是一条 GEMM。搜索有时间/次数上限，采用显存边界二分搜索及离散策略
对照，不宣称穷举得到全局最优。

当前支持：单 CUDA GPU、标准 Dense/MoE、从零预训练、JSONL/Parquet 单文件、
bucket packing、inline streaming。保持 gradient checkpoint 开启。SFT、RL、
DDP、Looped/Linear 和加载基座的 CPT 尚未接入，不能套用本档案。

## 执行

### Config WebUI

从仓库根目录运行 `start_config_webui.bat`，打开顶部 **自动调优** 页签。
侧栏只保留模型预设；训练参数、模型结构、流水线、监控与配置导出分别放在独立
页签。所有页签使用同一份自动保存的面板参数。

界面中的两种自动计算用途不同：

| 功能 | 使用方法 | 结果 |
|---|---|---|
| 动态显存估算 | 在「训练配置 → 数据加载与动态批量」开启 Sequence packing，选择 Auto sequence buckets，填写每卡显存，token budget 保持 0 | 启动时结合 GPU、模型、优化器和 checkpoint 校准每桶 batch；不代表吞吐最优 |
| 自动调优 | 在「自动调优」页预览计划后启动真实短训 | 搜索范围内通过质量门槛与独立复测的最佳有效 tokens/s 配置，保存为 profile |

第一次使用自动调优：

1. 在「训练配置」选择 `pretrain` 和已清洗审核的数据；在模型预设/模型结构中
   确认标准 Dense/MoE。关闭 Resume，选择好精度、优化器和学习率。
2. 进入「自动调优」，查看全部前置条件。可点 **一键准备调优设置**，补齐
   checkpoint、bucket packing、streaming=on 和 inline cache build；这一步
   保留模型、数据、学习率与精度，不启动实验，也不切换训练阶段或续训状态。
3. 确认清洗审计 JSON；Parquet 还需要来源 provenance。检查次数和时间预算
   （默认最多 24 次，20 步预热、60 步计时，总上限 3600 秒）。
4. 点击 **检查并预览搜索计划**，在进度日志中检查输出。这一步不使用 CUDA。
   确认计划后点 **开始自动调优**，实际占用 GPU 运行短训与验证。
5. 在 **查看并应用结果** 中选择通过验收的 profile，点击 **用于后续预训练**。
   回到「训练配置」，展开 **查看完整训练命令**，确认带有 `--autotune_profile`，
   没有兼容性报错后点击 **开始训练**。

「计划预览完成」不等于已有性能结果，只有真实试跑并通过最终复测才会发布 profile。
实测结果绑定当前训练机、代码、模型与语料，不能直接用于 CPT、SFT、RL 或 DDP。

数据源切换时自动更新默认审计路径，手工指定的报告保留。
Parquet 的同名 provenance 存在时自动填写。

- **检查并预览搜索计划**：通过实际 CLI 检查搜索范围；不初始化 CUDA 或训练。
- **开始自动调优**：冻结当前面板模型与共享训练命令中的参数，独立后台运行。
  可设置次数、预热/计时 steps、验证样本数和时间上限，支持缓存复用 / 强制重测。
- **高级调优选项**：可选计算 MFU 的非稀疏峰值、可选吞吐验收门槛、自定义
  搜索空间和性能环境参数。吞吐门槛默认 0，没有固定的 25,081 token/s 目标。
- **停止自动调优**：仅回收本次 supervisor 所拥有的 controller / 实验进程树。
  已完成的结果与编译 / packing 缓存保留，不保存正式模型权重。
- **已验收调优结果**：展示最佳与复测吞吐、MFU（提供峰值时）、显存、候选
  质量检查和选择参数，支持导出 profile、用于后续预训练。CLI 生成的 profile
  也可直接填写路径。性能参数由 profile 管理，面板的数学 / 数据设置继续校验。

实验期间禁止同时从该面板启动正式训练或流水线。浏览器刷新 / 关闭不会终止
实验，状态按 supervisor PID + 创建时间恢复；不会把调优子进程认成正式训练。
应用 profile 后，单次训练命令预览与实际启动共用相同构造器；保存流水线快照
时，仅预训练阶段携带 profile，SFT 继续使用自己的配置。代码或明确训练参数
不兼容会在命令预览阶段提示；GPU、完整训练合约和数据 SHA-256 在 trainer
初始化前再次校验。续训仍保留检查点的 batch/token budget。

普通机器直接运行既有入口。本机 Windows 仓库根目录被其他程序占用、目录枚举
返回 PermissionError 时，WebUI 的调优子进程自动使用
`scripts/autotune_bootstrap.py` 读取原文件；不替换训练实现或调整导入顺序。

从仓库根目录运行，使用正常训练的 Python 环境：

```bash
# 只检查配置、查看搜索范围，不启动训练
python trainer/autotune.py --config trainer/autotune_pretrain.json --plan-only

# 默认最多 24 次试跑，包含基线和最终复测；总时间预算一小时
python trainer/autotune.py --config trainer/autotune_pretrain.json

# 显式限制时间和试跑次数
python trainer/autotune.py --config trainer/autotune_pretrain.json --max-trials 12 --time-limit 2400

# 本机性能验收：20 步预热 + 80 步统计，未达到目标不发布新档案
python trainer/autotune.py --config trainer/autotune_pretrain.json --steps 80 --target-tokens-per-second 25081
```

Windows 在父进程启动前设置 `PYTHONUTF8=1`；调优器也会为子进程设置它。
例如 PowerShell：`$env:PYTHONUTF8='1'`。所有进程按 datasets-before-torch 和
compile_cache-before-torch 的既有顺序加载。

`trainer/autotune_pretrain.json` 是当前 V1 MoE 的配置模板。迁移训练机后检查其中
模型配置、数据路径、精度、优化器、LR、序列上限、packing/缓存/预取参数。
`bucket_gpu_memory_gb: "auto"` 读取目标卡物理容量，再由生产 CLI 的驱动预算和
安全余量限制；没有固定的 16 GiB 或 16,384-token 搜索上限。

`audit` 必须指向该训练文件的已审核 SHA-256 报告。Parquet 还需要 `provenance`
指向原清洗/来源审计。程序校验文件哈希，但不会用哈希校验替代身份污染审查。
验证集固定读取源文件第二数据块的前 N 条文本，所有候选使用相同 token；试跑
若进入第二块会失败，避免污染验证集。因此源文件必须能划分为至少两个数据块。

## 搜索范围

默认搜索以下参数，可用配置文件中的 `search` 完全替换默认搜索空间：

| 参数 | 搜索方式 |
|---|---|
| 每卡 token budget | 围绕基线已解析预算的 0.5–2 倍搜索，允许超出本机旧经验上限 |
| checkpoint | FFN/整层，以及可用的算子内 checkpoint；禁止关闭或双重重放 |
| Muon | workspace、矩阵分组大小、momentum 预分配 |
| CPU | PyTorch intra-op 线程数；在模型初始化后调整，保持初始化 RNG |
| compile | 编译模式；自定义搜索可增加 `args.use_compile: [0, 1]` |
| MoE Triton | forward/dW tile、warps/stages、每 SM 的持久调度规模 |

例如只搜索 batch 和 CPU 线程：

```json
"search": {
  "args.bucket_token_budget": [8192, 12288, 16384, 20480, 24576, 32768],
  "env.INSTINCT_TRAINING_CPU_THREADS": ["1", "2", "4"]
}
```

`environment` 可设置上述允许调优的环境变量作为基线。其他已有 backend 开关
沿用启动环境，并写入兼容性指纹。dtype、FP8 配方、模型结构、Muon 数学、LR、
有效梯度累积、语料与 packing 规则不在搜索白名单中。

默认先比较可复用同一 packing 缓存的主要矩阵内核，再对 token budget 二分。
每种 checkpoint/内核/workspace 策略分别记录可行区间；例如已知 16k 能运行、
候选上界 32k 时先测中点，OOM 后排除上半区。只有 OOM/驱动预算越界能收紧
显存上界，编译失败不会被误判为显存不足。出现非单调证据时停止剪枝、直接测量。
边界附近另测较小 batch，避免把“最大能装下”误当“吞吐最高”。tile/checkpoint
和 CPU 线程等吞吐不单调的参数仍使用有限候选，不对它们强行套二分。

程序默认没有吞吐目标值；`--target-tokens-per-second` 仅是可选的外部验收门槛，
不参与候选生成或打分，换训练机不需要填写本机的 25,081 数值。
数据预处理和 loader 参数由配置固定；它们仍参与真实测量，并不会通过关预取或
替换为随机张量来制造收益。标准库算法、Triton 版本和硬件兼容回退继续生效。

## 测量、筛选与隔离

- 一个候选一个新进程，避免 OOM、编译状态和优化器状态污染下一候选。
  调用真实 `train_pretrain.train_epoch`：真实 Arrow、预取、LR、前反向、裁剪、
  optimizer、zero_grad 和正常日志。超时只终止该调优器创建的进程树。
- 默认 20 步预热、60 步计时；计时区间含数据等待，排除初始读数/编译、验证和
  存盘。计时前等待首次后台预取完成，使各候选都在暖缓存条件下比较；这个额外
  等待单独记录，普通生产训练不执行该等待。开启预取时 warmup 至少为 2。
  若计时阶段出现未预热的形状，拒绝该候选并提示增加 warmup。
- 每次重新使用相同初始化。验证 token 和初始参数均记录 SHA-256；每 20 步
  训练 loss 均值恶化不得超过 2%，公共验证 NLL 恶化不得超过 1%，loss 必须有限。
  用户已允许扩大 batch；这些是等更新次数的短训门槛，不是等 token 的收敛证明。
- OOM/编译失败/超时不作为有效结果；基线在训练中 OOM 时会减半 token budget
  重试。显存预算越界的结果也不参与选择。基线模型本身装不下时需调整模型配置。
- 目标是最大有效 tokens/s，最高有效实测为主结果；最终候选必须独立复测且再次
  通过质量门槛。保留所有成功/失败记录和 GPU 快照，避免隐藏后台负载波动。
- 默认不猜测 GPU 峰值。需要 MFU 时显式给 `--peak-tflops`，确保是目标卡、实际
  计算精度、dense 口径。MFU 排除 checkpoint 重算、优化器和 padding FLOPs。
- 不运行正式 resume，不写正式模型权重、不推进正式游标，也不发送 SwanLab
  日志。只保留小型请求/结果/日志、空 scratch 权重目录及可复用的受预算管理缓存。

## 使用结果

结果位于 `.cache/autotune/<fingerprint>/`：

- `search.json`：候选列表、拒绝原因与测量；每次试跑另有 `trial.log`。
- `profile.json`：只有完成最终复测才发布的生产配置。
- `launch.json`：包含完整生产命令参数数组；终端也会打印可复制命令。

使用调优器打印的完整命令，或在相同训练配置上增加：

```bash
python trainer/train_pretrain.py ... --autotune_profile .cache/autotune/<fingerprint>/profile.json
```

显式 CLI/环境变量与调优档案冲突会报错，不会悄悄覆盖。生成的命令会省去由
档案提供的性能参数。档案绑定 GPU/驱动/CUDA/PyTorch/Triton/CPU/OS、代码和
tokenizer、模型/训练参数及语料 SHA；条件变化后重新调优。

恢复正式训练时仍保留断点保存的 token budget/游标，不能因新档案改变同一 epoch
的 batch 划分；恢复后的实际模型结构也必须匹配档案。切换 batch 应在新的训练
阶段执行并重新验收。

重复运行同一配置会复用已完成且指纹匹配的成功候选；失败候选可重试。
`--refresh` 强制重新测量。时间耗尽或中断时保存已完成结果，不发布未经复测的
配置。准备在专用训练机上完整调优前，应让其他 GPU 任务结束。

当前开发机另有 Windows 仓库目录 sharing lock。仅在该故障环境复现时使用：

```powershell
python tmp/run_locked_root.py trainer/autotune.py --config trainer/autotune_pretrain.json --worker-launcher tmp/run_locked_root.py
```

这是诊断启动器，不是正常机器的依赖；专用训练机使用前面的普通命令。

## 本机验证记录

2026-10-14 的默认候选搜索（4 次试跑，每次20步预热+80步计时，未设置吞吐
目标）得到 **26,420.710 有效 tokens/s / 38.779% MFU**，独立复测为
26,415.477 tokens/s。实际数据、checkpoint 和训练/验证门槛均保留；生产入口
已验证加载生成的档案。本次有限试跑覆盖内核选择，没有完成全部策略与容量
扫描。二分逻辑另通过边界、编译失败、OOM 和非单调回退测试。完整口径见
仓库 `PLAN.md`，生产档案和命令在
`experiments/results/autotune_default/05f5185e84de9778d84cb44b/`。
