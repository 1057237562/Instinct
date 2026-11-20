# Chat WebUI 的 NPZ / cuBLAS 推理

运行根目录的 `start_chat_webui.bat`。权重列表支持 PTH 和原生 NPZ，按修改时间倒序排列，
新会话优先选取最新 NPZ；已有会话保留当前选择。NPZ 递归扫描 `out/`、`checkpoints/`，
包括 `*_resume.npz`，跳过隐藏目录、`_archive` 和临时文件。也可填写自定义路径。

NPZ 必须带 `instinct-cublas-v1` 内嵌元数据。模型拓扑、RoPE 和上下文上限来自该文件，
不使用默认 pretrain JSON。完整 resume 文件只加载 `model/` 参数，不载入 optimizer、
梯度或训练游标，也不会修改检查点。侧栏显示保存步数及实际推理后端。

推理在独立的无 PyTorch 进程中复用 `model/dense_cublas`，矩阵乘法使用 cuBLASLt，
其他算子使用原生 Triton kernels。参数、激活和 KV cache 使用 BF16；MXFP8 检查点的
训练专用 FP8/Muon/激活重计算策略不用于推理。模型架构及位置编码保持不变。
这是推理精度选择，不是训练续训，也不承诺与 MXFP8 训练 logits 逐位一致。

运行时优先使用当前仓库已有的原生实现，否则从
`checkpoints/native_cublas/training_run.json` 获取 `runtime_root`、`python` 和环境。
可通过 `INSTINCT_NATIVE_RUNTIME_ROOT`、`INSTINCT_NATIVE_PYTHON` 覆盖位置。
需要该运行时已有的 CUDA Toolkit、cuBLASLt bridge 和兼容 Triton 依赖；缺失时显示错误，
不会自动安装依赖或转成 PTH。首次生成可能编译尚未缓存的 kernels。

支持单序列流式输出、温度/top-p/重复惩罚、EOS、多轮工具调用与取消生成。
上下文达到检查点上限时停止；输入已经占满上下文时提示缩短历史。逐层解释在原生模式禁用。
卸载会等待生成退出并关闭本会话专用推理进程，不操作训练进程。

CPU 回归：

```powershell
python -m pytest tests/test_chat_native.py tests/test_web_demo_utils.py tests/test_chat_generation.py --skip-gpu
```
