---
license: mit
task_categories:
- text-generation
language:
- en
pretty_name: MathInstruct
size_categories:
- 100K<n<1M
tags:
- math
---

# 🦣 MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning

MathInstruct is a meticulously curated instruction tuning dataset that is lightweight yet generalizable. MathInstruct is compiled from 13 math rationale datasets, six of which are newly curated by this work. It uniquely focuses on the hybrid use of chain-of-thought (CoT) and program-of-thought (PoT) rationales, and ensures extensive coverage of diverse mathematical fields. 

Project Page: [https://tiger-ai-lab.github.io/MAmmoTH/](https://tiger-ai-lab.github.io/MAmmoTH/)

Paper: [https://arxiv.org/pdf/2309.05653.pdf](https://arxiv.org/pdf/2309.05653.pdf)

Code: [https://github.com/TIGER-AI-Lab/MAmmoTH](https://github.com/TIGER-AI-Lab/MAmmoTH)

Models:
|     | **Base Model: Llama-2**                                       | **Base Model: Code Llama**                                               |
|-----|---------------------------------------------------------------|--------------------------------------------------------------------------|
| 7B  | 🦣 [MAmmoTH-7B](https://huggingface.co/TIGER-Lab/MAmmoTH-7B)   | 🦣 [MAmmoTH-Coder-7B](https://huggingface.co/TIGER-Lab/MAmmoTH-Coder-7B)  |
| 13B | 🦣 [MAmmoTH-13B](https://huggingface.co/TIGER-Lab/MAmmoTH-13B) | 🦣 [MAmmoTH-Coder-13B](https://huggingface.co/TIGER-Lab/MAmmoTH-Coder-13B)|
| 34B | -                                                             | 🦣 [MAmmoTH-Coder-34B](https://huggingface.co/TIGER-Lab/MAmmoTH-Coder-34B)|
| 70B | 🦣 [MAmmoTH-70B](https://huggingface.co/TIGER-Lab/MAmmoTH-70B) | -                                                                        |

## **License**
Please check out the license of each subset in our curated dataset MathInstruct.
| Dataset Name 	| License Type   	|
|--------------|----------------|
| GSM8K        	| MIT            	|
| GSM8K-RFT    	| Non listed      |
| AQuA-RAT     	| Apache 2.0     	|
| MATH         	| MIT            	|
| TheoremQA    	| MIT            	|
| Camel-Math   	| Attribution-NonCommercial 4.0 International    	|
| NumGLUE      	| Apache-2.0          	|
| MathQA       	| Apache-2.0     	|
| Our Curated   | MIT             |


## **Citation**

Please cite our paper if you use our data, model or code. Please also kindly cite the original dataset papers. 

```
@article{yue2023mammoth,
  title={MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning},
  author={Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen},
  journal={arXiv preprint arXiv:2309.05653},
  year={2023}
}
```