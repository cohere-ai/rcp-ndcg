---
annotations_creators:
- human-annotated
language:
- eng
multilinguality: monolingual
task_categories:
- text-ranking
task_ids: []
dataset_info:
- config_name: corpus
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  - name: title
    dtype: string
  splits:
  - name: test
    num_bytes: 667643
    num_examples: 7220
  download_size: 274293
  dataset_size: 667643
- config_name: default
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: test
    num_bytes: 409115
    num_examples: 7220
  download_size: 55985
  dataset_size: 409115
- config_name: queries
  features:
  - name: _id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: test
    num_bytes: 25572
    num_examples: 361
  download_size: 16214
  dataset_size: 25572
- config_name: top_ranked
  features:
  - name: query-id
    dtype: string
  - name: corpus-ids
    sequence: string
  splits:
  - name: test
    num_bytes: 238286
    num_examples: 361
  download_size: 50286
  dataset_size: 238286
configs:
- config_name: corpus
  data_files:
  - split: test
    path: corpus/test-*
- config_name: default
  data_files:
  - split: test
    path: data/test-*
- config_name: queries
  data_files:
  - split: test
    path: queries/test-*
- config_name: top_ranked
  data_files:
  - split: test
    path: top_ranked/test-*
tags:
- mteb
- text
---
<!-- adapted from https://github.com/huggingface/huggingface_hub/blob/v0.30.2/src/huggingface_hub/templates/datasetcard_template.md -->

<div align="center" style="padding: 40px 20px; background-color: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); max-width: 600px; margin: 0 auto;">
  <h1 style="font-size: 3.5rem; color: #1a1a1a; margin: 0 0 20px 0; letter-spacing: 2px; font-weight: 700;">AskUbuntuDupQuestions</h1>
  <div style="font-size: 1.5rem; color: #4a4a4a; margin-bottom: 5px; font-weight: 300;">An <a href="https://github.com/embeddings-benchmark/mteb" style="color: #2c5282; font-weight: 600; text-decoration: none;" onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">MTEB</a> dataset</div>
  <div style="font-size: 0.9rem; color: #2c5282; margin-top: 10px;">Massive Text Embedding Benchmark</div>
</div>

AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar

|               |                                             |
|---------------|---------------------------------------------|
| Task category | t2t                              |
| Domains       | Programming, Web                               |
| Reference     | https://github.com/taolei87/askubuntu |


## How to evaluate on this task

You can evaluate an embedding model on this dataset using the following code:

```python
import mteb

task = mteb.get_tasks(["AskUbuntuDupQuestions"])
evaluator = mteb.MTEB(task)

model = mteb.get_model(YOUR_MODEL)
evaluator.run(model)
```

<!-- Datasets want link to arxiv in readme to autolink dataset with paper -->
To learn more about how to run models on `mteb` task check out the [GitHub repitory](https://github.com/embeddings-benchmark/mteb). 

## Citation

If you use this dataset, please cite the dataset as well as [mteb](https://github.com/embeddings-benchmark/mteb), as this dataset likely includes additional processing as a part of the [MMTEB Contribution](https://github.com/embeddings-benchmark/mteb/tree/main/docs/mmteb).

```bibtex

@article{wang-2021-TSDAE,
  author = {Wang, Kexin and Reimers, Nils and  Gurevych, Iryna},
  journal = {arXiv preprint arXiv:2104.06979},
  month = {4},
  title = {TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning},
  url = {https://arxiv.org/abs/2104.06979},
  year = {2021},
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

task = mteb.get_task("AskUbuntuDupQuestions")

desc_stats = task.metadata.descriptive_stats
```

```json
{
    "test": {
        "num_samples": 7581,
        "number_of_characters": 397127,
        "num_documents": 7220,
        "min_document_length": 15,
        "average_document_length": 52.49722991689751,
        "max_document_length": 148,
        "unique_documents": 7220,
        "num_queries": 361,
        "min_query_length": 17,
        "average_query_length": 50.13019390581717,
        "max_query_length": 148,
        "unique_queries": 361,
        "none_queries": 0,
        "num_relevant_docs": 7220,
        "min_relevant_docs_per_query": 20,
        "average_relevant_docs_per_query": 5.470914127423823,
        "max_relevant_docs_per_query": 20,
        "unique_relevant_docs": 7220,
        "num_instructions": null,
        "min_instruction_length": null,
        "average_instruction_length": null,
        "max_instruction_length": null,
        "unique_instructions": null,
        "num_top_ranked": 361,
        "min_top_ranked_per_query": 20,
        "average_top_ranked_per_query": 20.0,
        "max_top_ranked_per_query": 20
    }
}
```

</details>

---
*This dataset card was automatically generated using [MTEB](https://github.com/embeddings-benchmark/mteb)*