---
task_categories:
- question-answering
language:
- en
- zh
tags:
- Long Context
- sft
size_categories:
- 10K<n<100K
---
# LongAlign-10k

<p align="center">
  🤗 <a href="https://huggingface.co/datasets/THUDM/LongAlign-10k" target="_blank">[LongAlign Dataset] </a> • 💻 <a href="https://github.com/THUDM/LongAlign" target="_blank">[Github Repo]</a> • 📃 <a href="https://arxiv.org/abs/2401.18058" target="_blank">[LongAlign Paper]</a> 
</p>

**LongAlign** is the first full recipe for LLM alignment on long context. We propose the **LongAlign-10k** dataset, containing 10,000 long instruction data of 8k-64k in length. We investigate on trianing strategies, namely **packing (with loss weighting) and sorted batching**, which are all implemented in our code. For real-world long context evaluation, we introduce **LongBench-Chat** that evaluate the instruction-following capability on queries of 10k-100k length.

## All Models

We open-sourced the following list of models:

|Model|Huggingface Repo|Description|
|---|---|---|
|**LongAlign-6B-64k-base**| [🤗 Huggingface Repo](https://huggingface.co/THUDM/LongAlign-6B-64k-base) | **ChatGLM3-6B** with an extended 64k context window |
|**LongAlign-6B-64k**| [🤗 Huggingface Repo](https://huggingface.co/THUDM/LongAlign-6B-64k) | Chat model by LongAlign training on LongAlign-6B-64k-base|
|**LongAlign-7B-64k-base**| [🤗 Huggingface Repo](https://huggingface.co/THUDM/LongAlign-7B-64k-base) | **Llama-2-7B** with an extended 64k context window |
|**LongAlign-7B-64k**| [🤗 Huggingface Repo](https://huggingface.co/THUDM/LongAlign-7B-64k) | Chat model by LongAlign training on LongAlign-7B-64k-base|
|**LongAlign-13B-64k-base**| [🤗 Huggingface Repo](https://huggingface.co/THUDM/LongAlign-13B-64k-base) | **Llama-2-13B** with an extended 64k context window |
|**LongAlign-13B-64k**| [🤗 Huggingface Repo](https://huggingface.co/THUDM/LongAlign-13B-64k) | Chat model by LongAlign training on LongAlign-13B-64k-base|
|**ChatGLM3-6B-128k**| [🤗 Huggingface Repo](https://huggingface.co/THUDM/chatglm3-6b-128k) | **ChatGLM3-6B** with a 128k context window|