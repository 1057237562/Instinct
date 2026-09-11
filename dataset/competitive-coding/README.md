---
language:
- en
license:
- cc-by-4.0
- odc-by
- mit
task_categories:
- text-generation
configs:
- config_name: default
  data_files:
  - split: exercism
    path: data/exercism.jsonl
  - split: text_to_sql
    path: data/text_to_sql.jsonl
  - split: competitive_coding_cpp
    path: data/competitive_programming_cpp_*.jsonl
  - split: competitive_coding_python
    path: data/competitive_programming_python_*.jsonl
---


## Dataset Description:
Nemotron-Competitive-Programming-v2 is a large-scale synthetic coding and reasoning dataset designed to push LLM performance on challenging programming and systems tasks. It combines Python and C++ samples across unique competitive programming questions.  
In addition, the dataset includes samples targetting a subset of programming language exercises from [Exercism](https://exercism.org/).  

Beyond problem solving, the dataset includes Text-to-SQL, which aims at training LLMs to reason and perform tasks involving SQL.


This dataset is ready for commercial use.

### Competitive Coding
The competitive coding subset of Nemotron-Competitive-Programming-v2 is one of the largest reasoning-based synthetic dataset to date for coding, comprising 336k+ samples in Python and 332k++ samples in C++ across 34,799 unique competitive programming questions. 

This subset is designed for supervised fine-tuning (SFT) tasks of code completion and code critique.

- [Github Repo](https://github.com/NVIDIA/NeMo-Skills) - Access the complete pipeline used to generate the data and perform SFT.



### Python Exercises

The Exercism subset is a synthetic dataset derived from 115 Python Exercism problems (with problems from the Aider benchmark decontaminated) totalling 40K generated coding problems and their solutions.

### Text-To-SQL

The Text-to-SQL Dataset contains 96.5k high-fidelity synthetic records designed to train LLMs to reason and perform better on SQL tasks. Unlike standard benchmarks, every sample includes a reasoning trace, teaching models to plan and validate logic before writing code. Generated via NVIDIA Data Designer, it simulates messy enterprise environments with noisy data, distractor tables, and dialect-specific constraints (MySQL, PostgreSQL, SQLite).  
This dataset targets the "last mile" of enterprise AI, teaching agents to perform schema linking and write execution-ready SQL.



## How to use it - Competitive Coding

```
from tqdm import tqdm
from datasets import load_dataset

hf_datasets = {
    "taco": load_dataset("BAAI/TACO", trust_remote_code=True),
    "apps": load_dataset("codeparrot/apps", trust_remote_code=True),
    "code_contests": load_dataset("deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("open-r1/codeforces")
}


def get_question(ds_name, split, index):
    benchmark = hf_datasets[ds_name][split][int(index)]
    if ds_name == "code_contests":
        if not benchmark["description"]:
            return None
        return benchmark["description"]
    elif ds_name in ["taco", "apps"]:
        return benchmark["question"]
    elif ds_name == "open-r1/codeforces":
        if not benchmark["description"]:
            return None
        question = benchmark["description"]
        if benchmark["input_format"]:
            question += "\n\nInput\n\n" + benchmark["input_format"]
        if benchmark["output_format"]:
            question += "\n\nOutput\n\n" + benchmark["output_format"]
        if benchmark["examples"]:
            question += "\n\nExamples"
            for example in benchmark["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if benchmark["note"]:
            question += "\n\nNote\n\n" + benchmark["note"]
        return question

    return None


ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2")
for ocr2_ds in [ocr2_dataset["python"], ocr2_dataset["cpp"]]:
    for ocr2_ds_item in tqdm(ocr2_ds):
        assert ocr2_ds_item["dataset"] in ["taco", "apps", "code_contests", "open-r1/codeforces"]
        ds_name, ds_split, ds_index = ocr2_ds_item["dataset"], ocr2_ds_item["split"], int(ocr2_ds_item["index"])
        question = get_question(ds_name, ds_split, ds_index)
        assert question is not None
        assert ocr2_ds_item["question"] == "-"
        ocr2_ds_item["question"] = question
```

## Dataset Owner(s):
NVIDIA Corporation

## Dataset Creation Date:
Created on: 12/01/2025  
Last Modified on: 12/01/2025


## License/Terms of Use: 
This dataset is governed by the Creative Commons Attribution 4.0 International License (CC BY 4.0), except for Code Forces data, which is licensed under the Open Data Commons Attribution License (ODC-By).  
Additional Information: MIT License.

NOTICE FOR SCRIPTS: You may run the scripts below to pull datasets from their original source. The underlying datasets are available from the original sources subject to their own license terms.

## Intended Usage:
The Nemotron-Competitive-Programming-v2 dataset is intended to be used by the community to continue to improve open models on code writing and reviewing tasks.
The SQL subset is designed for researchers and developers fine-tuning Large Language Models to build enterprise-grade SQL agents.

## Dataset Characterization
**Data Collection Method**  
Hybrid: Automated, Synthetic  

**Labeling Method**
Hybrid: Automated, Synthetic  

## Dataset Format  
Modality: Text  
Format: JSONL  
Structure: Text + Metadata

## Dataset Quantification

| Subset | Samples |
|--------|---------|
| competitive_coding_cpp      | 332,559       |
| competitive_coding_python      | 336,568       |
| exercism       |   79,244      |
| text_to_sql      |   96,564      |
| Total       | 844,935        |


Total Data Storage: ~91 GB 


## Reference(s):

* [OpenCodeReasoning: Advancing Data Distillation for Competitive Coding](https://arxiv.org/abs/2504.01943)
* [OpenCodeReasoning-II: A Simple Test Time Scaling Approach via Self-Critique](https://arxiv.org/abs/2507.09075)
* [GenSelect: A Generative Approach to Best-of-N](https://arxiv.org/abs/2507.17797)
* [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)


## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/)
