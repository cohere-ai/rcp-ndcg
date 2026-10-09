---
annotations_creators:
- expert-annotated
language:
- ara
- ben
- deu
- eng
- fas
- fin
- fra
- hin
- ind
- jpn
- kor
- rus
- spa
- swa
- tel
- tha
- yor
- zho
license: cc-by-sa-4.0
multilinguality: multilingual
source_datasets:
- RSamoed/MIRACLRetrieval
task_categories:
- text-retrieval
task_ids: []
dataset_info:
- config_name: ar-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 1217782171
    num_examples: 2061414
  download_size: 547359941
  dataset_size: 1217782171
- config_name: ar-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 835497
    num_examples: 29197
  download_size: 297571
  dataset_size: 835497
- config_name: ar-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 192068
    num_examples: 2896
  download_size: 107179
  dataset_size: 192068
- config_name: bn-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 309278144
    num_examples: 297265
  download_size: 111041231
  dataset_size: 309278144
- config_name: bn-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 113824
    num_examples: 4206
  download_size: 39642
  dataset_size: 113824
- config_name: bn-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 55939
    num_examples: 411
  download_size: 24670
  dataset_size: 55939
- config_name: de-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 7018142949
    num_examples: 15866222
  download_size: 4154965117
  dataset_size: 7018142949
- config_name: de-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 105647
    num_examples: 3144
  download_size: 36113
  dataset_size: 105647
- config_name: de-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 19450
    num_examples: 305
  download_size: 14370
  dataset_size: 19450
- config_name: en-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 13936141152
    num_examples: 32893221
  download_size: 7975250425
  dataset_size: 13936141152
- config_name: en-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 242967
    num_examples: 8350
  download_size: 90601
  dataset_size: 242967
- config_name: en-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 41576
    num_examples: 799
  download_size: 28786
  dataset_size: 41576
- config_name: es-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 4490445463
    num_examples: 10373953
  download_size: 2570067226
  dataset_size: 4490445463
- config_name: es-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 215688
    num_examples: 6443
  download_size: 71449
  dataset_size: 215688
- config_name: es-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 43504
    num_examples: 648
  download_size: 29334
  dataset_size: 43504
- config_name: fa-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 1091928603
    num_examples: 2207172
  download_size: 474255337
  dataset_size: 1091928603
- config_name: fa-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 216792
    num_examples: 6571
  download_size: 71058
  dataset_size: 216792
- config_name: fa-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 58114
    num_examples: 632
  download_size: 32519
  dataset_size: 58114
- config_name: fi-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 739208952
    num_examples: 1883509
  download_size: 438759882
  dataset_size: 739208952
- config_name: fi-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 334260
    num_examples: 12008
  download_size: 114931
  dataset_size: 334260
- config_name: fi-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 65738
    num_examples: 1271
  download_size: 46897
  dataset_size: 65738
- config_name: fr-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 5507181331
    num_examples: 14636953
  download_size: 3090791707
  dataset_size: 5507181331
- config_name: fr-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 115836
    num_examples: 3429
  download_size: 38727
  dataset_size: 115836
- config_name: fr-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 21433
    num_examples: 343
  download_size: 14680
  dataset_size: 21433
- config_name: hi-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 490803869
    num_examples: 506264
  download_size: 175786968
  dataset_size: 490803869
- config_name: hi-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 111217
    num_examples: 3494
  download_size: 36608
  dataset_size: 111217
- config_name: hi-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 54122
    num_examples: 350
  download_size: 25503
  dataset_size: 54122
- config_name: id-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 537334548
    num_examples: 1446315
  download_size: 271115071
  dataset_size: 537334548
- config_name: id-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 263958
    num_examples: 9668
  download_size: 90280
  dataset_size: 263958
- config_name: id-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 47803
    num_examples: 960
  download_size: 30473
  dataset_size: 47803
- config_name: ja-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 2970531665
    num_examples: 6953614
  download_size: 1667392675
  dataset_size: 2970531665
- config_name: ja-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 236068
    num_examples: 8354
  download_size: 83800
  dataset_size: 236068
- config_name: ja-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 55179
    num_examples: 860
  download_size: 33901
  dataset_size: 55179
- config_name: ko-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 633206834
    num_examples: 1486752
  download_size: 361325767
  dataset_size: 633206834
- config_name: ko-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 83188
    num_examples: 3057
  download_size: 30628
  dataset_size: 83188
- config_name: ko-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 13875
    num_examples: 213
  download_size: 9844
  dataset_size: 13875
- config_name: ru-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 5921641619
    num_examples: 9543918
  download_size: 2767862757
  dataset_size: 5921641619
- config_name: ru-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 373821
    num_examples: 13100
  download_size: 134522
  dataset_size: 373821
- config_name: ru-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 116037
    num_examples: 1252
  download_size: 67086
  dataset_size: 116037
- config_name: sw-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 32766816
    num_examples: 131924
  download_size: 16589955
  dataset_size: 32766816
- config_name: sw-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 135409
    num_examples: 5092
  download_size: 44041
  dataset_size: 135409
- config_name: sw-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 24357
    num_examples: 482
  download_size: 15546
  dataset_size: 24357
- config_name: te-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 549992531
    num_examples: 518079
  download_size: 149672222
  dataset_size: 549992531
- config_name: te-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 43268
    num_examples: 1606
  download_size: 19278
  dataset_size: 43268
- config_name: te-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 95102
    num_examples: 828
  download_size: 38384
  dataset_size: 95102
- config_name: th-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 549881577
    num_examples: 542166
  download_size: 201796559
  dataset_size: 549881577
- config_name: th-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 208002
    num_examples: 7573
  download_size: 71074
  dataset_size: 208002
- config_name: th-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 98697
    num_examples: 733
  download_size: 44774
  dataset_size: 98697
- config_name: yo-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 10639617
    num_examples: 49043
  download_size: 5157335
  dataset_size: 10639617
- config_name: yo-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 35512
    num_examples: 1188
  download_size: 10613
  dataset_size: 35512
- config_name: yo-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 6697
    num_examples: 119
  download_size: 5615
  dataset_size: 6697
- config_name: zh-corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: dev
    num_bytes: 1718428423
    num_examples: 4934368
  download_size: 1092221644
  dataset_size: 1718428423
- config_name: zh-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: dev
    num_bytes: 129492
    num_examples: 3928
  download_size: 42650
  dataset_size: 129492
- config_name: zh-queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: dev
    num_bytes: 19261
    num_examples: 393
  download_size: 13803
  dataset_size: 19261
configs:
- config_name: ar-corpus
  data_files:
  - split: dev
    path: ar-corpus/dev-*
- config_name: ar-qrels
  data_files:
  - split: dev
    path: ar-qrels/dev-*
- config_name: ar-queries
  data_files:
  - split: dev
    path: ar-queries/dev-*
- config_name: bn-corpus
  data_files:
  - split: dev
    path: bn-corpus/dev-*
- config_name: bn-qrels
  data_files:
  - split: dev
    path: bn-qrels/dev-*
- config_name: bn-queries
  data_files:
  - split: dev
    path: bn-queries/dev-*
- config_name: de-corpus
  data_files:
  - split: dev
    path: de-corpus/dev-*
- config_name: de-qrels
  data_files:
  - split: dev
    path: de-qrels/dev-*
- config_name: de-queries
  data_files:
  - split: dev
    path: de-queries/dev-*
- config_name: en-corpus
  data_files:
  - split: dev
    path: en-corpus/dev-*
- config_name: en-qrels
  data_files:
  - split: dev
    path: en-qrels/dev-*
- config_name: en-queries
  data_files:
  - split: dev
    path: en-queries/dev-*
- config_name: es-corpus
  data_files:
  - split: dev
    path: es-corpus/dev-*
- config_name: es-qrels
  data_files:
  - split: dev
    path: es-qrels/dev-*
- config_name: es-queries
  data_files:
  - split: dev
    path: es-queries/dev-*
- config_name: fa-corpus
  data_files:
  - split: dev
    path: fa-corpus/dev-*
- config_name: fa-qrels
  data_files:
  - split: dev
    path: fa-qrels/dev-*
- config_name: fa-queries
  data_files:
  - split: dev
    path: fa-queries/dev-*
- config_name: fi-corpus
  data_files:
  - split: dev
    path: fi-corpus/dev-*
- config_name: fi-qrels
  data_files:
  - split: dev
    path: fi-qrels/dev-*
- config_name: fi-queries
  data_files:
  - split: dev
    path: fi-queries/dev-*
- config_name: fr-corpus
  data_files:
  - split: dev
    path: fr-corpus/dev-*
- config_name: fr-qrels
  data_files:
  - split: dev
    path: fr-qrels/dev-*
- config_name: fr-queries
  data_files:
  - split: dev
    path: fr-queries/dev-*
- config_name: hi-corpus
  data_files:
  - split: dev
    path: hi-corpus/dev-*
- config_name: hi-qrels
  data_files:
  - split: dev
    path: hi-qrels/dev-*
- config_name: hi-queries
  data_files:
  - split: dev
    path: hi-queries/dev-*
- config_name: id-corpus
  data_files:
  - split: dev
    path: id-corpus/dev-*
- config_name: id-qrels
  data_files:
  - split: dev
    path: id-qrels/dev-*
- config_name: id-queries
  data_files:
  - split: dev
    path: id-queries/dev-*
- config_name: ja-corpus
  data_files:
  - split: dev
    path: ja-corpus/dev-*
- config_name: ja-qrels
  data_files:
  - split: dev
    path: ja-qrels/dev-*
- config_name: ja-queries
  data_files:
  - split: dev
    path: ja-queries/dev-*
- config_name: ko-corpus
  data_files:
  - split: dev
    path: ko-corpus/dev-*
- config_name: ko-qrels
  data_files:
  - split: dev
    path: ko-qrels/dev-*
- config_name: ko-queries
  data_files:
  - split: dev
    path: ko-queries/dev-*
- config_name: ru-corpus
  data_files:
  - split: dev
    path: ru-corpus/dev-*
- config_name: ru-qrels
  data_files:
  - split: dev
    path: ru-qrels/dev-*
- config_name: ru-queries
  data_files:
  - split: dev
    path: ru-queries/dev-*
- config_name: sw-corpus
  data_files:
  - split: dev
    path: sw-corpus/dev-*
- config_name: sw-qrels
  data_files:
  - split: dev
    path: sw-qrels/dev-*
- config_name: sw-queries
  data_files:
  - split: dev
    path: sw-queries/dev-*
- config_name: te-corpus
  data_files:
  - split: dev
    path: te-corpus/dev-*
- config_name: te-qrels
  data_files:
  - split: dev
    path: te-qrels/dev-*
- config_name: te-queries
  data_files:
  - split: dev
    path: te-queries/dev-*
- config_name: th-corpus
  data_files:
  - split: dev
    path: th-corpus/dev-*
- config_name: th-qrels
  data_files:
  - split: dev
    path: th-qrels/dev-*
- config_name: th-queries
  data_files:
  - split: dev
    path: th-queries/dev-*
- config_name: yo-corpus
  data_files:
  - split: dev
    path: yo-corpus/dev-*
- config_name: yo-qrels
  data_files:
  - split: dev
    path: yo-qrels/dev-*
- config_name: yo-queries
  data_files:
  - split: dev
    path: yo-queries/dev-*
- config_name: zh-corpus
  data_files:
  - split: dev
    path: zh-corpus/dev-*
- config_name: zh-qrels
  data_files:
  - split: dev
    path: zh-qrels/dev-*
- config_name: zh-queries
  data_files:
  - split: dev
    path: zh-queries/dev-*
tags:
- mteb
- text
---
<!-- adapted from https://github.com/huggingface/huggingface_hub/blob/v0.30.2/src/huggingface_hub/templates/datasetcard_template.md -->

<div align="center" style="padding: 40px 20px; background-color: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); max-width: 600px; margin: 0 auto;">
  <h1 style="font-size: 3.5rem; color: #1a1a1a; margin: 0 0 20px 0; letter-spacing: 2px; font-weight: 700;">MIRACLRetrieval</h1>
  <div style="font-size: 1.5rem; color: #4a4a4a; margin-bottom: 5px; font-weight: 300;">An <a href="https://github.com/embeddings-benchmark/mteb" style="color: #2c5282; font-weight: 600; text-decoration: none;" onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">MTEB</a> dataset</div>
  <div style="font-size: 0.9rem; color: #2c5282; margin-top: 10px;">Massive Text Embedding Benchmark</div>
</div>

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages.

|               |                                             |
|---------------|---------------------------------------------|
| Task category | t2t                              |
| Domains       | Encyclopaedic, Written                               |
| Reference     | http://miracl.ai/ |




## How to evaluate on this task

You can evaluate an embedding model on this dataset using the following code:

```python
import mteb

task = mteb.get_tasks(["MIRACLRetrieval"])
evaluator = mteb.MTEB(task)

model = mteb.get_model(YOUR_MODEL)
evaluator.run(model)
```

<!-- Datasets want link to arxiv in readme to autolink dataset with paper -->
To learn more about how to run models on `mteb` task check out the [GitHub repitory](https://github.com/embeddings-benchmark/mteb).

## Citation

If you use this dataset, please cite the dataset as well as [mteb](https://github.com/embeddings-benchmark/mteb), as this dataset likely includes additional processing as a part of the [MMTEB Contribution](https://github.com/embeddings-benchmark/mteb/tree/main/docs/mmteb).

```bibtex

@article{10.1162/tacl_a_00595,
  abstract = {{MIRACL is a multilingual dataset for ad hoc retrieval across 18 languages that collectively encompass over three billion native speakers around the world. This resource is designed to support monolingual retrieval tasks, where the queries and the corpora are in the same language. In total, we have gathered over 726k high-quality relevance judgments for 78k queries over Wikipedia in these languages, where all annotations have been performed by native speakers hired by our team. MIRACL covers languages that are both typologically close as well as distant from 10 language families and 13 sub-families, associated with varying amounts of publicly available resources. Extensive automatic heuristic verification and manual assessments were performed during the annotation process to control data quality. In total, MIRACL represents an investment of around five person-years of human annotator effort. Our goal is to spur research on improving retrieval across a continuum of languages, thus enhancing information access capabilities for diverse populations around the world, particularly those that have traditionally been underserved. MIRACL is available at http://miracl.ai/.}},
  author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
  doi = {10.1162/tacl_a_00595},
  eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
  issn = {2307-387X},
  journal = {Transactions of the Association for Computational Linguistics},
  month = {09},
  pages = {1114-1131},
  title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
  url = {https://doi.org/10.1162/tacl\_a\_00595},
  volume = {11},
  year = {2023},
}


@article{enevoldsen2025mmtebmassivemultilingualtext,
  title={MMTEB: Massive Multilingual Text Embedding Benchmark},
  author={Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2502.13595},
  year={2025},
  url={https://arxiv.org/abs/2502.13595},
  doi = {10.48550/arXiv.2502.13595},
}

@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},
  year = {2022}
  url = {https://arxiv.org/abs/2210.07316},
  doi = {10.48550/ARXIV.2210.07316},
}
```

# Dataset Statistics
<details>
  <summary> Dataset Statistics</summary>

The following code contains the descriptive statistics from the task. These can also be obtained using:

```python
import mteb

task = mteb.get_task("MIRACLRetrieval")

desc_stats = task.metadata.descriptive_stats
```

```json
{
    "dev": {
        "num_samples": 106345647,
        "number_of_characters": 37176781172,
        "num_documents": 106332152,
        "min_document_length": 2,
        "average_document_length": 349.6241542163089,
        "max_document_length": 84930,
        "unique_documents": 106332152,
        "num_queries": 13495,
        "min_query_length": 5,
        "average_query_length": 36.49225639125602,
        "max_query_length": 176,
        "unique_queries": 13495,
        "none_queries": 0,
        "num_relevant_docs": 130408,
        "min_relevant_docs_per_query": 1,
        "average_relevant_docs_per_query": 2.3059651722860317,
        "max_relevant_docs_per_query": 20,
        "unique_relevant_docs": 119924,
        "num_instructions": null,
        "min_instruction_length": null,
        "average_instruction_length": null,
        "max_instruction_length": null,
        "unique_instructions": null,
        "num_top_ranked": null,
        "min_top_ranked_per_query": null,
        "average_top_ranked_per_query": null,
        "max_top_ranked_per_query": null
    }
}
```

</details>

---
*This dataset card was automatically generated using [MTEB](https://github.com/embeddings-benchmark/mteb)*