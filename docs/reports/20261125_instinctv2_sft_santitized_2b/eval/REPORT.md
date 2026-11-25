# native_instinctv2_sft_santitized_2b 评测报告 @ step 66048

- 生成时间：2026-11-25T21:00:00
- 训练状态：**manual**
- checkpoint：step **66048** / 1,072,520,943 tokens
- weights：`D:\AI\Instinct\out\native_instinctv2_sft_santitized_2b\native_instinctv2_sft_santitized_2b_1536.npz`  
  SHA-256 `3adb083ad2b66b6f8076f1c68c7201ebba7df564aa0607e6d029dc7c5c647a71`
- resume：`D:\AI\Instinct\checkpoints\native_instinctv2_sft_santitized_2b\native_instinctv2_sft_santitized_2b_1536_resume.npz`  
  SHA-256 `c76ce01b805c90da83937b8964b2bc1eaa54f3a5a3c50e700926ae308f2d35be`，progress step **65536** / 1,064,208,249 tokens（落后于 weights；仅作为续训起点）
- converted pth：`D:\AI\Instinct\eval\evalplus\native_instinctv2_sft_santitized_2b_step66048_lcb_provenance\step66048.pth`  
  SHA-256 `0250ef7757b43425445e5b3d04b9f3174c426dff0dea7e7b1367ef7795527d7d`
- 协议：EvalPlus 记录驱动（thinking off，164x10 HumanEval+ / 378x10 MBPP+，temp 0.8 / top_p 0.95 / top_k 50 / seed 42 / max_new 512，batch 32）；LCB release_v6 latest-175，greedy，think ctx **8192**、no-think ctx **4096**，MXFP8 E4M3 KV

## 主指标

| 指标 | @66048 | @50,688 | R1 pos @30,720 | v4 @3,000 |
| --- | ---: | ---: | ---: | ---: |
| HumanEval+ pass@1 | 0.3610 | 0.3476 | 0.4018 | 0.4421 |
| HumanEval+ pass@10 | 0.6585 | 0.6463 | 0.6768 | 0.7317 |
| HumanEval base pass@1 | 0.3945 | 0.3793 | 0.4232 | 0.4713 |
| HumanEval base pass@10 | 0.6951 | 0.6951 | 0.7073 | 0.7622 |
| MBPP+ pass@1 | 0.4661 | 0.4524 | 0.4574 | 0.4802 |
| MBPP+ pass@10 | 0.7143 | 0.6878 | 0.6931 | 0.7090 |
| MBPP base pass@1 | 0.5323 | 0.5209 | 0.5291 | 0.5463 |
| MBPP base pass@10 | 0.7963 | 0.7778 | 0.7937 | 0.7884 |
| LCB latest-175 think@8192 | 0.0800 (14/175) | 0.0971 (17/175) | 0.0857 (15/175, ctx 4096) | 0.0857 (ctx 4096) |
| LCB latest-175 no-think@4096 | 0.1086 (19/175) | 0.1257 (22/175) | 0.1257 (22/175) | 0.1200 |

## 细节

### LCB think@8192
- outcomes: `{"no_code": 123, "failed": 38, "passed": 14}`
- difficulty: `{"easy": {"tasks": 43, "passed": 12, "pass@1": 0.27906976744186046, "outcomes": {"no_code": 11, "failed": 20, "passed": 12}}, "hard": {"tasks": 80, "passed": 0, "pass@1": 0.0, "outcomes": {"no_code": 73, "failed": 7}}, "medium": {"tasks": 52, "passed": 2, "pass@1": 0.038461538461538464, "outcomes": {"no_code": 39, "failed": 11, "passed": 2}}}`

### LCB no-think@4096
- outcomes: `{"failed": 131, "no_code": 24, "passed": 19, "timeout": 1}`
- difficulty: `{"easy": {"tasks": 43, "passed": 17, "pass@1": 0.3953488372093023, "outcomes": {"failed": 26, "passed": 17}}, "hard": {"tasks": 80, "passed": 0, "pass@1": 0.0, "outcomes": {"failed": 58, "no_code": 21, "timeout": 1}}, "medium": {"tasks": 52, "passed": 2, "pass@1": 0.038461538461538464, "outcomes": {"failed": 47, "no_code": 3, "passed": 2}}}`

## 产物

- `D:\AI\Instinct\eval\evalplus\native_instinctv2_sft_santitized_2b_step66048_pass10`
- `D:\AI\Instinct\eval\livecodebench\native_instinctv2_sft_santitized_2b_step66048_release_v6_think_ctx8192_latest175`
- `D:\AI\Instinct\eval\livecodebench\native_instinctv2_sft_santitized_2b_step66048_release_v6_nothink_ctx4096_latest175`
- 运行日志：`D:\AI\Instinct\eval\auto_eval_native_instinctv2_sft_santitized_2b\auto_eval.log`

## 数据集与训练说明

- 数据：`dataset/instinctv2_sft_santitized_2b`，1,514,592 行 / 77,919 packed 16K blocks / 1,265,404,059 packed valid tokens（组装估算 1,311.1M，think 33.9%）
- 基座：V2 Dense pretrain acceptance（step 2,004,666）
- RoPE：smooth 模式（smooth_start=4096，smooth_end=8192，累加相位修复）
- 配额缺口：knowledge-think 1.8M（目标 35M）、tools 12.2M（目标 50M），总 token 为目标 1,382M 的 91.5%
