---
language:
- en
license: cc-by-4.0
size_categories:
- 1K<n<10K
task_categories:
- text-retrieval
dataset_info:
- config_name: Gemini-1.0_reason
  features:
  - name: query
    dtype: string
  - name: reasoning
    dtype: string
  - name: id
    dtype: string
  - name: excluded_ids
    sequence: string
  - name: gold_ids_long
    sequence: string
  - name: gold_ids
    sequence: string
  - name: gold_answer
    dtype: string
  splits:
  - name: biology
    num_bytes: 571840
    num_examples: 103
  - name: earth_science
    num_bytes: 569680
    num_examples: 116
  - name: economics
    num_bytes: 617171
    num_examples: 103
  - name: psychology
    num_bytes: 557169
    num_examples: 101
  - name: robotics
    num_bytes: 467620
    num_examples: 101
  - name: stackoverflow
    num_bytes: 652597
    num_examples: 117
  - name: sustainable_living
    num_bytes: 626085
    num_examples: 108
  - name: leetcode
    num_bytes: 1382908
    num_examples: 142
  - name: pony
    num_bytes: 327318
    num_examples: 112
  - name: aops
    num_bytes: 14182450
    num_examples: 111
  - name: theoremqa_questions
    num_bytes: 13144059
    num_examples: 194
  - name: theoremqa_theorems
    num_bytes: 393390
    num_examples: 76
  download_size: 5946613
  dataset_size: 33492287
- config_name: claude-3-opus_reason
  features:
  - name: query
    dtype: string
  - name: reasoning
    dtype: string
  - name: id
    dtype: string
  - name: excluded_ids
    sequence: string
  - name: gold_ids_long
    sequence: string
  - name: gold_ids
    sequence: string
  - name: gold_answer
    dtype: string
  splits:
  - name: biology
    num_bytes: 557025
    num_examples: 103
  - name: earth_science
    num_bytes: 558266
    num_examples: 116
  - name: economics
    num_bytes: 574237
    num_examples: 103
  - name: psychology
    num_bytes: 516517
    num_examples: 101
  - name: robotics
    num_bytes: 437209
    num_examples: 101
  - name: stackoverflow
    num_bytes: 622316
    num_examples: 117
  - name: sustainable_living
    num_bytes: 591735
    num_examples: 108
  - name: leetcode
    num_bytes: 1407501
    num_examples: 142
  - name: pony
    num_bytes: 361169
    num_examples: 112
  - name: aops
    num_bytes: 14149870
    num_examples: 111
  - name: theoremqa_questions
    num_bytes: 13121703
    num_examples: 194
  - name: theoremqa_theorems
    num_bytes: 382228
    num_examples: 76
  download_size: 5777474
  dataset_size: 33279776
- config_name: documents
  features:
  - name: id
    dtype: string
  - name: content
    dtype: string
  splits:
  - name: biology
    num_bytes: 21983744
    num_examples: 57359
  - name: earth_science
    num_bytes: 46952371
    num_examples: 121249
  - name: economics
    num_bytes: 22771374
    num_examples: 50220
  - name: psychology
    num_bytes: 23167414
    num_examples: 52835
  - name: robotics
    num_bytes: 20718385
    num_examples: 61961
  - name: stackoverflow
    num_bytes: 189733583
    num_examples: 107081
  - name: sustainable_living
    num_bytes: 24373723
    num_examples: 60792
  - name: pony
    num_bytes: 2365157
    num_examples: 7894
  - name: leetcode
    num_bytes: 456581333
    num_examples: 413932
  - name: aops
    num_bytes: 146564475
    num_examples: 188002
  - name: theoremqa_theorems
    num_bytes: 21124422
    num_examples: 23839
  - name: theoremqa_questions
    num_bytes: 146564475
    num_examples: 188002
  download_size: 465489179
  dataset_size: 1122900456
- config_name: examples
  features:
  - name: query
    dtype: string
  - name: reasoning
    dtype: string
  - name: id
    dtype: string
  - name: excluded_ids
    sequence: string
  - name: gold_ids_long
    sequence: string
  - name: gold_ids
    sequence: string
  - name: gold_answer
    dtype: string
  splits:
  - name: biology
    num_bytes: 326427
    num_examples: 103
  - name: earth_science
    num_bytes: 280741
    num_examples: 116
  - name: economics
    num_bytes: 343172
    num_examples: 103
  - name: psychology
    num_bytes: 286062
    num_examples: 101
  - name: robotics
    num_bytes: 366862
    num_examples: 101
  - name: stackoverflow
    num_bytes: 470365
    num_examples: 117
  - name: sustainable_living
    num_bytes: 336562
    num_examples: 108
  - name: pony
    num_bytes: 135009
    num_examples: 112
  - name: leetcode
    num_bytes: 1212640
    num_examples: 142
  - name: aops
    num_bytes: 13981802
    num_examples: 111
  - name: theoremqa_theorems
    num_bytes: 257842
    num_examples: 76
  - name: theoremqa_questions
    num_bytes: 12810785
    num_examples: 194
  download_size: 4590189
  dataset_size: 30808269
- config_name: gpt4_reason
  features:
  - name: query
    dtype: string
  - name: reasoning
    dtype: string
  - name: id
    dtype: string
  - name: excluded_ids
    sequence: string
  - name: gold_ids_long
    sequence: string
  - name: gold_ids
    sequence: string
  - name: gold_answer
    dtype: string
  splits:
  - name: biology
    num_bytes: 613511
    num_examples: 103
  - name: earth_science
    num_bytes: 618266
    num_examples: 116
  - name: economics
    num_bytes: 642234
    num_examples: 103
  - name: psychology
    num_bytes: 571504
    num_examples: 101
  - name: robotics
    num_bytes: 519720
    num_examples: 101
  - name: stackoverflow
    num_bytes: 704186
    num_examples: 117
  - name: sustainable_living
    num_bytes: 657382
    num_examples: 108
  - name: leetcode
    num_bytes: 1461063
    num_examples: 142
  - name: pony
    num_bytes: 423199
    num_examples: 112
  - name: aops
    num_bytes: 14332394
    num_examples: 111
  - name: theoremqa_questions
    num_bytes: 13262646
    num_examples: 194
  - name: theoremqa_theorems
    num_bytes: 442272
    num_examples: 76
  download_size: 6421256
  dataset_size: 34248377
- config_name: grit_reason
  features:
  - name: query
    dtype: string
  - name: reasoning
    dtype: string
  - name: id
    dtype: string
  - name: excluded_ids
    sequence: string
  - name: gold_ids_long
    sequence: string
  - name: gold_ids
    sequence: string
  - name: gold_answer
    dtype: string
  splits:
  - name: biology
    num_bytes: 478151
    num_examples: 103
  - name: earth_science
    num_bytes: 443792
    num_examples: 116
  - name: economics
    num_bytes: 493163
    num_examples: 103
  - name: psychology
    num_bytes: 407907
    num_examples: 101
  - name: robotics
    num_bytes: 340895
    num_examples: 101
  - name: stackoverflow
    num_bytes: 540771
    num_examples: 117
  - name: sustainable_living
    num_bytes: 475118
    num_examples: 108
  - name: leetcode
    num_bytes: 1305306
    num_examples: 142
  - name: pony
    num_bytes: 258002
    num_examples: 112
  - name: aops
    num_bytes: 14170933
    num_examples: 111
  - name: theoremqa_questions
    num_bytes: 13040517
    num_examples: 194
  - name: theoremqa_theorems
    num_bytes: 360282
    num_examples: 76
  download_size: 5202414
  dataset_size: 32314837
- config_name: llama3-70b_reason
  features:
  - name: query
    dtype: string
  - name: reasoning
    dtype: string
  - name: id
    dtype: string
  - name: excluded_ids
    sequence: string
  - name: gold_ids_long
    sequence: string
  - name: gold_ids
    sequence: string
  - name: gold_answer
    dtype: string
  splits:
  - name: biology
    num_bytes: 631132
    num_examples: 103
  - name: earth_science
    num_bytes: 622087
    num_examples: 116
  - name: economics
    num_bytes: 631657
    num_examples: 103
  - name: psychology
    num_bytes: 563987
    num_examples: 101
  - name: robotics
    num_bytes: 449342
    num_examples: 101
  - name: stackoverflow
    num_bytes: 641853
    num_examples: 117
  - name: sustainable_living
    num_bytes: 654690
    num_examples: 108
  - name: leetcode
    num_bytes: 1376032
    num_examples: 142
  - name: pony
    num_bytes: 315870
    num_examples: 112
  - name: aops
    num_bytes: 14183895
    num_examples: 111
  - name: theoremqa_questions
    num_bytes: 13157545
    num_examples: 194
  - name: theoremqa_theorems
    num_bytes: 405388
    num_examples: 76
  download_size: 5877403
  dataset_size: 33633478
- config_name: long_documents
  features:
  - name: id
    dtype: string
  - name: content
    dtype: string
  splits:
  - name: biology
    num_bytes: 19454314
    num_examples: 524
  - name: earth_science
    num_bytes: 41843262
    num_examples: 601
  - name: economics
    num_bytes: 20095594
    num_examples: 516
  - name: psychology
    num_bytes: 20541239
    num_examples: 512
  - name: robotics
    num_bytes: 18220587
    num_examples: 508
  - name: stackoverflow
    num_bytes: 184616744
    num_examples: 1858
  - name: sustainable_living
    num_bytes: 21200303
    num_examples: 554
  - name: pony
    num_bytes: 2098474
    num_examples: 577
  download_size: 104578765
  dataset_size: 328070517
configs:
- config_name: Gemini-1.0_reason
  data_files:
  - split: biology
    path: Gemini-1.0_reason/biology-*
  - split: earth_science
    path: Gemini-1.0_reason/earth_science-*
  - split: economics
    path: Gemini-1.0_reason/economics-*
  - split: psychology
    path: Gemini-1.0_reason/psychology-*
  - split: robotics
    path: Gemini-1.0_reason/robotics-*
  - split: stackoverflow
    path: Gemini-1.0_reason/stackoverflow-*
  - split: sustainable_living
    path: Gemini-1.0_reason/sustainable_living-*
  - split: leetcode
    path: Gemini-1.0_reason/leetcode-*
  - split: pony
    path: Gemini-1.0_reason/pony-*
  - split: aops
    path: Gemini-1.0_reason/aops-*
  - split: theoremqa_questions
    path: Gemini-1.0_reason/theoremqa_questions-*
  - split: theoremqa_theorems
    path: Gemini-1.0_reason/theoremqa_theorems-*
- config_name: claude-3-opus_reason
  data_files:
  - split: biology
    path: claude-3-opus_reason/biology-*
  - split: earth_science
    path: claude-3-opus_reason/earth_science-*
  - split: economics
    path: claude-3-opus_reason/economics-*
  - split: psychology
    path: claude-3-opus_reason/psychology-*
  - split: robotics
    path: claude-3-opus_reason/robotics-*
  - split: stackoverflow
    path: claude-3-opus_reason/stackoverflow-*
  - split: sustainable_living
    path: claude-3-opus_reason/sustainable_living-*
  - split: leetcode
    path: claude-3-opus_reason/leetcode-*
  - split: pony
    path: claude-3-opus_reason/pony-*
  - split: aops
    path: claude-3-opus_reason/aops-*
  - split: theoremqa_questions
    path: claude-3-opus_reason/theoremqa_questions-*
  - split: theoremqa_theorems
    path: claude-3-opus_reason/theoremqa_theorems-*
- config_name: documents
  data_files:
  - split: biology
    path: documents/biology-*
  - split: earth_science
    path: documents/earth_science-*
  - split: economics
    path: documents/economics-*
  - split: psychology
    path: documents/psychology-*
  - split: robotics
    path: documents/robotics-*
  - split: stackoverflow
    path: documents/stackoverflow-*
  - split: sustainable_living
    path: documents/sustainable_living-*
  - split: pony
    path: documents/pony-*
  - split: leetcode
    path: documents/leetcode-*
  - split: aops
    path: documents/aops-*
  - split: theoremqa_theorems
    path: documents/theoremqa_theorems-*
  - split: theoremqa_questions
    path: documents/theoremqa_questions-*
- config_name: examples
  data_files:
  - split: biology
    path: examples/biology-*
  - split: earth_science
    path: examples/earth_science-*
  - split: economics
    path: examples/economics-*
  - split: psychology
    path: examples/psychology-*
  - split: robotics
    path: examples/robotics-*
  - split: stackoverflow
    path: examples/stackoverflow-*
  - split: sustainable_living
    path: examples/sustainable_living-*
  - split: pony
    path: examples/pony-*
  - split: leetcode
    path: examples/leetcode-*
  - split: aops
    path: examples/aops-*
  - split: theoremqa_theorems
    path: examples/theoremqa_theorems-*
  - split: theoremqa_questions
    path: examples/theoremqa_questions-*
- config_name: gpt4_reason
  data_files:
  - split: biology
    path: gpt4_reason/biology-*
  - split: earth_science
    path: gpt4_reason/earth_science-*
  - split: economics
    path: gpt4_reason/economics-*
  - split: psychology
    path: gpt4_reason/psychology-*
  - split: robotics
    path: gpt4_reason/robotics-*
  - split: stackoverflow
    path: gpt4_reason/stackoverflow-*
  - split: sustainable_living
    path: gpt4_reason/sustainable_living-*
  - split: leetcode
    path: gpt4_reason/leetcode-*
  - split: pony
    path: gpt4_reason/pony-*
  - split: aops
    path: gpt4_reason/aops-*
  - split: theoremqa_questions
    path: gpt4_reason/theoremqa_questions-*
  - split: theoremqa_theorems
    path: gpt4_reason/theoremqa_theorems-*
- config_name: grit_reason
  data_files:
  - split: biology
    path: grit_reason/biology-*
  - split: earth_science
    path: grit_reason/earth_science-*
  - split: economics
    path: grit_reason/economics-*
  - split: psychology
    path: grit_reason/psychology-*
  - split: robotics
    path: grit_reason/robotics-*
  - split: stackoverflow
    path: grit_reason/stackoverflow-*
  - split: sustainable_living
    path: grit_reason/sustainable_living-*
  - split: leetcode
    path: grit_reason/leetcode-*
  - split: pony
    path: grit_reason/pony-*
  - split: aops
    path: grit_reason/aops-*
  - split: theoremqa_questions
    path: grit_reason/theoremqa_questions-*
  - split: theoremqa_theorems
    path: grit_reason/theoremqa_theorems-*
- config_name: llama3-70b_reason
  data_files:
  - split: biology
    path: llama3-70b_reason/biology-*
  - split: earth_science
    path: llama3-70b_reason/earth_science-*
  - split: economics
    path: llama3-70b_reason/economics-*
  - split: psychology
    path: llama3-70b_reason/psychology-*
  - split: robotics
    path: llama3-70b_reason/robotics-*
  - split: stackoverflow
    path: llama3-70b_reason/stackoverflow-*
  - split: sustainable_living
    path: llama3-70b_reason/sustainable_living-*
  - split: leetcode
    path: llama3-70b_reason/leetcode-*
  - split: pony
    path: llama3-70b_reason/pony-*
  - split: aops
    path: llama3-70b_reason/aops-*
  - split: theoremqa_questions
    path: llama3-70b_reason/theoremqa_questions-*
  - split: theoremqa_theorems
    path: llama3-70b_reason/theoremqa_theorems-*
- config_name: long_documents
  data_files:
  - split: biology
    path: long_documents/biology-*
  - split: earth_science
    path: long_documents/earth_science-*
  - split: economics
    path: long_documents/economics-*
  - split: psychology
    path: long_documents/psychology-*
  - split: robotics
    path: long_documents/robotics-*
  - split: stackoverflow
    path: long_documents/stackoverflow-*
  - split: sustainable_living
    path: long_documents/sustainable_living-*
  - split: pony
    path: long_documents/pony-*
tags:
- text-retrieval
- code
- biology
- earth_science
- economics
- psychology
- robotics
- math
---

# BRIGHT benchmark
BRIGHT is the first text retrieval benchmark that requires intensive reasoning to retrieve relevant documents. 
The queries are collected from diverse domains (StackExchange, LeetCode, and math competitions), all sourced from realistic human data.
Experiments show that existing retrieval models perform poorly on BRIGHT, where the highest score is only 22.1 measured by nDCG@10.
BRIGHT provides a good testbed for future retrieval research in more realistic and challenging settings. More details are in the [paper](https://brightbenchmark.github.io/).

## Dataset Structure
We unify all the datasets with consistent formats. They are organized into three subsets, examples, documents, and long_documents:
* `examples`:
  * `query`: the query for retrieval
  * `reasoning`: the gold reasoning steps annotated by humans (they help people understand the relevance between queries and documents, but are not used in any experiment in the paper)
  * `id`: the index of the instance
  * `excluded_ids`: a list of the ids (string) to exclude during evaluation (only for `theoremqa`/`aops`/`leetcode`)
  * `gold_ids_long`: a list of the ids (string) of the ground truth documents, corresponding to the ids of the `long_documents` subset
  * `gold_ids`: a list of the ids (string) of the ground truth documents, corresponding to the indices of the `documents` subset
* `documents`:
  * `id`: the index of the document
  * `content`: document content (short version split from the complete web page, blogs, etc., or a problem and solution pair)
* `long_documents` (not applicable to `theoremqa`/`aops`/`leetcode`):
  * `id`: the index of the document
  * `content`: document content (long version corresponding to the complete web page, blogs, etc.)

## Dataset Statistics
<img src="statistics.png" width="80%" alt="BRIGHT statistics">

## Data Loading
Each dataset can be easily loaded. For example, to load biology examples:
```
from datasets import load_dataset
data = load_dataset('xlangai/BRIGHT', 'examples')['biology']
```

## Citation
If you find our work helpful, please cite us:
```citation
@misc{BRIGHT,
  title={BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
  author={Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and Sun, Ruoxi and Yoon, Jinsung and Arik, Sercan O and Chen, Danqi and Yu, Tao},
  url={https://arxiv.org/abs/2407.12883},
  year={2024},
}
```