---
dataset_info:
- config_name: corpus
  features:
  - name: corpus-id
    dtype: int64
  - name: image
    dtype: image
  splits:
  - name: test
    num_bytes: 90193755.0
    num_examples: 500
  download_size: 77165595
  dataset_size: 90193755.0
- config_name: qrels
  features:
  - name: query-id
    dtype: int64
  - name: corpus-id
    dtype: int64
  - name: score
    dtype: int64
  splits:
  - name: test
    num_bytes: 12000
    num_examples: 500
  download_size: 6729
  dataset_size: 12000
- config_name: queries
  features:
  - name: query-id
    dtype: int64
  - name: query
    dtype: string
  splits:
  - name: test
    num_bytes: 55717
    num_examples: 500
  download_size: 30417
  dataset_size: 55717
configs:
- config_name: corpus
  data_files:
  - split: test
    path: corpus/test-*
- config_name: qrels
  data_files:
  - split: test
    path: qrels/test-*
- config_name: queries
  data_files:
  - split: test
    path: queries/test-*
task_categories:
- document-question-answering
- visual-document-retrieval
---

BEIR version of [`vidore/arxivqa_test_subsampled`](https://huggingface.co/datasets/vidore/arxivqa_test_subsampled).
