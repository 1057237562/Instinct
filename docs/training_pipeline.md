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

## V1 MoE 修复与低预算验证（2026-10-01）

### 首次 SFT、继续 SFT 与断点续训的配置切换

WebUI 的 `SFT start mode` 在数据集和训练参数控件之前显示。点击模式旋钮时，
先保存当前模式的手动配置，再恢复目标模式的独立配置；首次进入目标模式采用下表起点。

| 模式 | 基础权重 | 峰值学习率 | Epoch | Warmup | 优化器/调度器状态 |
|---|---|---:|---:|---:|---|
| 首次 SFT | 已完成 pretrain/CPT | 1e-5 | 2 | 3% | 新建，从 step 0 开始 |
| 继续 SFT | 已完成 full_sft | 3e-6 | 1 | 5% | 新建，从 step 0 开始 |
| 断点续训 | 原运行 resume checkpoint | 当前运行配置 | 当前总目标 | 当前配置 | 恢复模型、优化器、scaler 和数据游标 |

继续 SFT 的较小 LR 和单轮曝光用于降低追加调优时的遗忘风险；这组具体数值是保守
实验起点，不是已证明最优，也不保证所有新领域数据均更好。
方向参考 [Continual Fine-tuning 遗忘研究](https://arxiv.org/abs/2308.08747)。
两种新阶段使用 FP32 主参数与 BF16 激活，继承当前优化器/编译/FP8/显存预算选择；
缺少运行设置时采用 Muon、4096 分桶 packing 和 default 编译。

独立配置记录学习率、epochs、warmup、数据集、基础权重、上下文/packing、精度、
优化器、编译和路由迁移开关。切换回来恢复手动修改，普通页面刷新不重置。
模式切换默认关闭显式路由迁移，保留所选基座的 top-k；需要迁移可在该模式独立开启。
模型结构与专家数量不随 SFT 模式旋钮改变。训练运行时禁用该旋钮。
已有 checkpoint 不改写，未启动训练；使用的训练文件仍须完成身份审核和基准去污染。

### WebUI CPT 默认配置 v4

切换到 CPT 时，旧版本默认值会一次性升级；以后保留手动调整，也可点击
「应用推荐 CPT 默认配置」恢复。正在运行、暂停或选择断点续训时不自动覆盖。
来源是 0925 预训练及 0930 CPT 日志和低预算路由适应需求，不是超参搜索所得的最优解。

| 项目 | 默认值 | 依据或限制 |
|---|---|---|
| Epoch | 1 | 不额外重复全语料；仍需先选择已审核的小子集控制成本 |
| 峰值 LR / warmup / 最低 LR | 3e-5 / 3% / 3e-6 | 比旧 CPT 5e-5、1% 更保守；针对路由迁移的启发式起点，尚无对照结果 |
| 优化器 | Muon | 延续 0925 和已完成的 0930 CPT 配方 |
| 精度 | FP32 参数、BF16 autocast、tensorwise FP8 | 延续已有运行配方；FP8 需要 torchao，稳定性诊断可显式关闭 |
| Packing / 长度 | 2 buckets / 上限4096 | 自动按显存规划实际 batch；非packing备用batch=1，累积=1 |
| 梯度检查点 | 1 | 选择性重算，沿用现有 MoE 配置 |
| 编译 | 开启，default | 降低短实验的首次调优成本；不保证稳态速度超过 max-autotune |
| 数据缓存 / streaming | 20GB / on，1024MiB chunks，预取开启 | 0930 曾在10GB缓存下因7.61+7.61GiB空间需求失败；20GB运行完成 |
| MoE 路由 | 显式应用当前面板 top-k，推荐 top-1 | top-1 自动关闭归一化；开启新阶段迁移，保留原权重 |
| GPU 显存预算 | 保留用户选择，缺省15.5GB | 本地16GB与远程32GB不能套用同一固定batch |

数据集和基础权重选择不自动替换，也不启动训练。建议先在独立验证集上比较原路由与
修复路由的短适应结果，再扩大训练量；评测题不得混入训练，所选数据须通过身份审核。
学习率重新预热、衰减和旧数据回放的方向参考
[Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763)；
具体的 3e-5 / 3% 是本项目保守选择，不能从论文直接推出最优。
`default` 模式优先控制编译成本，参考 [PyTorch compile 模式说明](https://docs.pytorch.org/docs/stable/generated/torch.compile)。

0925 预训练 checkpoint 记录约 12B token、35/35 streaming chunks 完成。
预计 step 总数不是完成率；按 token 和 chunk 游标判断。训练 loss 不能替代独立验证。

新 MoE 预设使用 `norm_topk_prob=false`：top-1 若将选中概率归一化为
`p/p=1`，router 几乎收不到任务损失梯度，只剩负载均衡信号。
模型构造器和历史 checkpoint 的默认语义保留，以免改变旧权重的推理输出。
训练加载旧配置时会明确打印警告，负载均匀不代表任务驱动的专家分工有效。

`train_pretrain.py`（含 CPT）和 `train_full_sft.py` 增加
`--moe_router_norm_topk_prob 0`，在恢复基座配置后显式迁移路由，并将新设置写入
后续 checkpoint。只用于 `--from_resume 0` 的新阶段，配合 `--from_weight`
和新的 `--save_weight`；不能直接修改旧 resume 状态或只切换推理设置。
迁移改变专家输出幅度，不是无损修复；先验证适应性，不能保证旧权重质量立即改善。
常规断点续训省略该选项，自动沿用 checkpoint 的路由设置。

WebUI：在 `pretrain` / `cpt` / `full_sft` 的训练设置中，找到基础权重选择框下方
的 **显式迁移 MoE 路由（新训练阶段）** 开关（仅 MoE 显示，普通模式默认关闭，CPT 推荐配置开启）。
开启后直接应用上方 MoE 面板的 `num_experts_per_tok`，范围为 1 到 `num_experts`，
不再单独限制为 top-1/top-2。例如 8 个专家可以迁移至 top-1 到 top-8：

- **修复 top-1**：传入 `--moe_router_top_k 1 --moe_router_norm_topk_prob 0`。
- **切换 top-2**：传入 `--moe_router_top_k 2 --moe_router_norm_topk_prob 1`。

top-1 自动关闭概率归一化；top-k>1 沿用面板的 `norm_topk_prob`，因此上面的 top-2
示例对应勾选归一化。专家总数与权重布局仍从基座恢复，迁移后的 k 必须不超过基座
的专家总数；此开关不增加/删除专家。降低面板专家总数时，激活专家数同步约束到有效范围。

面板展示实际传入参数。这些设置在基座配置恢复之后应用，覆盖基座保存的路由设置。
选择断点续训会禁用迁移，启动命令不传入迁移参数；CLI 也拒绝迁移和 resume 同用。
不启用开关则保持基座行为。新阶段沿用 WebUI 的新输出名称，原 checkpoint 不改写。
保存到 Pretrain→SFT 流水线的配置快照也携带迁移选项。Top-2 增加专家计算量，
属于额外实验，不是修复 top-1 梯度的必要条件。

按新增计算量从低到高安排，前一级失败就不要扩大预算：

1. **CPU 回归，无训练**：运行下方命令，检查任务梯度、旧配置兼容、迁移保存和
   packed attention 变长 mask。FlexAttention 保留原有 fullgraph 编译及默认形状策略。
   训练编译入口和 Flex 初始化统一设置 Dynamo 的局部上限 128、累计上限 4096，
   同时覆盖主干、mask 和 attention；不强制动态形状、不清空缓存。兼容新旧配置名。
   超过新上限明确报错，避免主干悄悄退回 eager 导致吞吐下降。
   代价是允许缓存更多形状及首次编译，不是免费消除所有编译开销；
   实际吞吐须用相同形状、充分预热的 GPU A/B 验证。
   启动器的非法 `OMP_NUM_THREADS` 在导入原生库前降为 1，合法设置保留。
2. **少量 GPU 前反向验证**：用小模型、两种长度及尾批次检查真实 CUDA/BF16、
   梯度检查点和 grouped MoE 路径。先关闭 FP8 与编译验证数值，再单独打开编译；
   最后才验证 FP8。不要一次改变路由、精度、优化器和数据配方。
3. **现有 checkpoint 推理对照**：0925 base 与 CPT 用相同 base 提示，SFT 用相同
   chat 提示。先每题单次 greedy，固定 KV 精度、生成长度和 batch；分开报告提示格式。
   结合独立 Python 验证集 loss 定位退化阶段，稳定后再做多样本 pass@k。
4. **旧基座短适应实验**：从同一个已完成预训练基座分别建立原路由对照和新路由分支，
   用同一份已审核的小语料、相同 token 预算和学习率。先约 100 万 token 验证能否稳定适应，
   再考虑 1000 万量级。该预算只是诊断起点，不足以证明最终质量；关闭编译适合极短冒烟，
   较长实验可用 `compile_mode=default`，先避免昂贵的 max-autotune。
5. **最后才考虑扩大 CPT/SFT 或从头重训**：只有独立验证和任务结果改善才增加预算。
   不默认重跑整份 12B 数据，也不直接增加 top-k、专家数或模型规模。

新增/复用训练子集都须完成身份审核、基准去污染及来源/计数/SHA-256 审计，
保留原始数据并使用 `identity_clean` 等明确后缀。HumanEval 题目及测试不能进入训练集。

```powershell
python -m pytest tests/test_moe_router_training.py tests/test_moe_dispatch.py tests/test_packed_attention.py tests/test_compile_cache.py tests/test_compile_policy.py tests/test_config_webui_presets.py --skip-gpu -q
```

上述 CPU 测试不证明 CUDA/Triton 内核性能或 HumanEval 提升；GPU 验证和实际训练效果需另行测量。

### Packed attention 编译策略的性能验证

用户允许训练途中因新形状多次重编译，因此保留原编译策略，扩大有界缓存额度；
没有强制动态形状，也不清空已编译缓存或静默退回 eager。
2026-10-01 本机 RTX 5070 Ti / PyTorch 2.13.0+cu132 实测：BF16、batch=4、
length=2880、16Q/4KV、head_dim=32，包含 mask 构建及 attention 前向/反向。
充分预热后交替测量各 40 次，原版中位数 2.101 ms，修正版 2.077 ms，
输出和 Q/K/V 梯度逐位一致。这一组件测试未发现稳态减速；不代表完整模型或其他 GPU
的吞吐保证。两方案可复用同一编译缓存，首次编译时间不作 A/B 对比；新形状仍可能
产生编译停顿和额外缓存占用。CPU 变长/尾批次及配置隔离测试通过。

后续排查确认仅给 Flex 单次 `torch.compile` 设置 128 不足：主干仍有默认 8 的限制，
同一 Python code object 的不同对象/区域及失效后的编译尝试仍受累计 256 的限制
（以本机 PyTorch 2.13 源码为准；这不是整个训练只能编译 256 个 GPU kernel）。
现通过 `model/compile_policy.py` 在编译前一次性设置两层限制，无每步配置包装。
上面的 GPU A/B 是先前仅调整局部限制版本的组件测量，不是新累计策略的完整训练基准。

可在启动训练前调整，必须为正整数且累计值不小于局部值：

```powershell
$env:INSTINCT_COMPILE_RECOMPILE_LIMIT = "128"
$env:INSTINCT_COMPILE_ACCUMULATED_LIMIT = "4096"
```

这些是上限，不会提前创建相应数量的图。大量形状仍会增加编译时间、缓存内存与 guard
查找开销；不是无条件吞吐保证。若触顶，检查 `TORCH_LOGS=recompiles` 的 guard 原因，
确认是合法新形状后再提高预算；不使用无限缓存或吞掉异常。
