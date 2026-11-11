---
language:
- en
task_categories:
- instruction-retrieval
task_ids:
- image,text-to-image,text
config_names:
- query
- corpus
- qrels
tags:
- information-retrieval
- multimodal-retrieval
dataset_info:
- config_name: qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int8
  splits:
  - name: test
    num_examples: 402
- config_name: corpus
  features:
  - name: id
    dtype: string
  - name: modality
    dtype: string
  - name: image
    dtype: image
  splits:
  - name: corpus
    num_examples: 804
- config_name: query
  features:
  - name: id
    dtype: string
  - name: modality
    dtype: string
  - name: text
    dtype: string
  - name: image
    dtype: image
  splits:
  - name: test
    num_examples: 402
configs:
- config_name: qrels
  data_files:
  - split: test
    path: qrels-*
- config_name: corpus
  data_files:
  - split: corpus
    path: corpus-*
- config_name: query
  data_files:
  - split: test
    path: query-*
---
