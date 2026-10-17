---
language:
- eng
multilinguality: monolingual
task_categories:
- text-retrieval
task_ids: []
config_names:
- corpus
tags:
- mteb
- text
dataset_info:
- config_name: default
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: float64
  splits:
  - name: train
    num_bytes: 3720942
    num_examples: 110575
  - name: dev
    num_bytes: 383427
    num_examples: 11385
  - name: test
    num_bytes: 415220
    num_examples: 12334
- config_name: corpus
  features:
  - name: _id
    dtype: string
  - name: title
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: corpus
    num_bytes: 5856698
    num_examples: 3633
- config_name: queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: queries
    num_bytes: 128355
    num_examples: 3237
configs:
- config_name: default
  data_files:
  - split: train
    path: qrels/train.jsonl
  - split: dev
    path: qrels/dev.jsonl
  - split: test
    path: qrels/test.jsonl
- config_name: corpus
  data_files:
  - split: corpus
    path: corpus.jsonl
- config_name: queries
  data_files:
  - split: queries
    path: queries.jsonl
---
<!-- adapted from https://github.com/huggingface/huggingface_hub/blob/v0.30.2/src/huggingface_hub/templates/datasetcard_template.md -->

<div align="center" style="padding: 40px 20px; background-color: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); max-width: 600px; margin: 0 auto;">
  <h1 style="font-size: 3.5rem; color: #1a1a1a; margin: 0 0 20px 0; letter-spacing: 2px; font-weight: 700;">NFCorpus</h1>
  <div style="font-size: 1.5rem; color: #4a4a4a; margin-bottom: 5px; font-weight: 300;">An <a href="https://github.com/embeddings-benchmark/mteb" style="color: #2c5282; font-weight: 600; text-decoration: none;" onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">MTEB</a> dataset</div>
  <div style="font-size: 0.9rem; color: #2c5282; margin-top: 10px;">Massive Text Embedding Benchmark</div>
</div>

NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval

|               |                                             |
|---------------|---------------------------------------------|
| Task category | t2t                              |
| Domains       | Medical, Academic, Written                               |
| Reference     | https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/ |


## How to evaluate on this task

You can evaluate an embedding model on this dataset using the following code:

```python
import mteb

task = mteb.get_tasks(["NFCorpus"])
evaluator = mteb.MTEB(task)

model = mteb.get_model(YOUR_MODEL)
evaluator.run(model)
```

<!-- Datasets want link to arxiv in readme to autolink dataset with paper -->
To learn more about how to run models on `mteb` task check out the [GitHub repitory](https://github.com/embeddings-benchmark/mteb). 

## Citation

If you use this dataset, please cite the dataset as well as [mteb](https://github.com/embeddings-benchmark/mteb), as this dataset likely includes additional processing as a part of the [MMTEB Contribution](https://github.com/embeddings-benchmark/mteb/tree/main/docs/mmteb).

```bibtex

@inproceedings{boteva2016,
  author = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
  city = {Padova},
  country = {Italy},
  journal = {Proceedings of the 38th European Conference on Information Retrieval},
  journal-abbrev = {ECIR},
  title = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
  url = {http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf},
  year = {2016},
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

task = mteb.get_task("NFCorpus")

desc_stats = task.metadata.descriptive_stats
```

```json
{
    "test": {
        "num_samples": 3956,
        "number_of_characters": 5786348,
        "num_documents": 3633,
        "min_document_length": 123,
        "average_document_length": 1590.783925130746,
        "max_document_length": 10090,
        "unique_documents": 3633,
        "num_queries": 323,
        "min_query_length": 3,
        "average_query_length": 21.764705882352942,
        "max_query_length": 72,
        "unique_queries": 323,
        "none_queries": 0,
        "num_relevant_docs": 12334,
        "min_relevant_docs_per_query": 1,
        "average_relevant_docs_per_query": 38.18575851393189,
        "max_relevant_docs_per_query": 475,
        "unique_relevant_docs": 3128,
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