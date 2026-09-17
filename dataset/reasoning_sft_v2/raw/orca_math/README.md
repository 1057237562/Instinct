---
dataset_info:
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 225322861
    num_examples: 200035
  download_size: 84248748
  dataset_size: 225322861
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: mit
task_categories:
- question-answering
language:
- en
tags:
- math
size_categories:
- 100K<n<1M
---
# Dataset Card

<!-- Provide a quick summary of the dataset. -->

This dataset contains ~200K grade school math word problems. All the answers in this dataset is generated using Azure GPT4-Turbo. Please refer to [Orca-Math: Unlocking the potential of
SLMs in Grade School Math](https://arxiv.org/pdf/2402.14830.pdf) for details about the dataset construction. 



### Dataset Description

- **Curated by:** Microsoft
- **Language(s) (NLP):** English
- **License:** MIT

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Repository:** [microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)
- **Paper:** [Orca-Math: Unlocking the potential of
SLMs in Grade School Math](https://arxiv.org/pdf/2402.14830.pdf)


### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

This dataset has been designed to enhance the mathematical abilities of language models. It aims to provide a robust foundation for language models to excel in mathematical problem-solving.


### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->
This dataset is not intended for use in educational systems or organizations.


## Dataset Structure

### Data Instances

A typical data entry in the dataset consists of a question and its corresponding answer. Below is an example from the dataset:

```python
{'question': 'In a highly contested election having multiple candidates, Mr. Jackson, one of the losing candidates, received 3,485,782 votes, which accounted for precisely 38.7 percent of all votes. To have achieved a victory, he would have needed to secure at least 51 percent of all votes. Approximately, what percent of the remaining unsecured votes would Mr. Jackson have needed to accumulate to reach this victory threshold?',
 'answer': "First, let's find out the total number of votes cast in the election. Since Mr. Jackson received 38.7% of all votes, and that amounted to 3,485,782 votes, we can set up the following equation to find the total number of votes (T):\n\n0.387 * T = 3,485,782\n\nNow, solve for T:\n\nT = 3,485,782 / 0.387\nT ≈ 9,000,467 votes (total number of votes cast)\n\nTo win, Mr. Jackson would have needed 51% of the total votes. Let's calculate that amount:\n\n0.51 * T = 0.51 * 9,000,467\n0.51 * T ≈ 4,590,238 votes needed to win\n\nNow, let's find out how many more votes Mr. Jackson needed to reach this winning threshold:\n\nVotes needed to win - Votes Mr. Jackson received = Additional votes needed\n4,590,238 - 3,485,782 = 1,104,456 additional votes needed\n\nNow, let's find out what percentage of the remaining unsecured votes this number represents. The remaining unsecured votes are the votes that were not for Mr. Jackson, which is 100% - 38.7% = 61.3% of the total votes.\n\n61.3% of the total votes is the remaining unsecured votes:\n\n0.613 * T = 0.613 * 9,000,467\n0.613 * T ≈ 5,514,686 votes were unsecured\n\nNow, we'll calculate the percentage of these unsecured votes that the additional votes needed represent:\n\n(Additional votes needed / Unsecured votes) * 100 = Percentage of unsecured votes needed\n(1,104,456 / 5,514,686) * 100 ≈ 20.03%\n\nSo, Mr. Jackson would have needed approximately 20.03% of the remaining unsecured votes to reach the victory threshold of 51%."}
```

### Data Fields

The dataset comprises the following fields:

- `question`: a string containing the question to be answered.
- `answer`: a string containing the answer to the corresponding question.

### Data Splits

The dataset is split into a training set. The number of rows in each split is as follows:

- `train`: 200,035 rows

The `DatasetDict` structure for the dataset is as follows:

```python
DatasetDict({
    'train': Dataset({
        features: ['question', 'answer'],
        num_rows: 200035
    })
})
```

Each split in the `DatasetDict` contains a `Dataset` object with the specified features and number of rows.


## Dataset Creation
Please refer to [Orca-Math: Unlocking the potential of
SLMs in Grade School Math](https://arxiv.org/pdf/2402.14830.pdf) for details about the dataset construction. 

### Source Data

- [Lila](https://huggingface.co/datasets/allenai/lila)

- [DMath](https://arxiv.org/ftp/arxiv/papers/2106/2106.15772.pdf)

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

Please refer to [Orca-Math: Unlocking the potential of
SLMs in Grade School Math](https://arxiv.org/pdf/2402.14830.pdf) for details about the dataset construction. 

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

Microsoft


#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

We expanded a seed set of questions using Azure GPT-4 Trubo. The answers to those questions are generated using Azure GPT-4 Trubo.



#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

None

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

This dataset is in English and contains only math word problems.



## Citation

If you find this work useful in your method, you can cite the paper as below:

```
@misc{mitra2024orcamath,
      title={Orca-Math: Unlocking the potential of SLMs in Grade School Math}, 
      author={Arindam Mitra and Hamed Khanpour and Corby Rosset and Ahmed Awadallah},
      year={2024},
      eprint={2402.14830},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


## Dataset Card Contact
[Arindam Mitra](armitra@microsoft.com)
