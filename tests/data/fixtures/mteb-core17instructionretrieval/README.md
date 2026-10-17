---
annotations_creators:
- derived
language:
- eng
license: mit
multilinguality: monolingual
source_datasets:
- jhu-clsp/core17-instructions-mteb
task_categories:
- text-ranking
task_ids: []
dataset_info:
- config_name: corpus
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: test
    num_bytes: 44843804
    num_examples: 19899
  download_size: 27837310
  dataset_size: 44843804
- config_name: instruction
  features:
  - name: query-id
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: test
    num_bytes: 13675
    num_examples: 40
  download_size: 7443
  dataset_size: 13675
- config_name: qrel_diff
  features:
  - name: query-id
    dtype: string
  - name: corpus-ids
    list: string
  splits:
  - name: qrel_diff
    num_bytes: 5632
    num_examples: 20
  download_size: 5646
  dataset_size: 5632
- config_name: qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: test
    num_bytes: 311980
    num_examples: 9480
  download_size: 53436
  dataset_size: 311980
- config_name: queries
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  - name: instruction
    dtype: string
  splits:
  - name: test
    num_bytes: 18225
    num_examples: 40
  download_size: 10300
  dataset_size: 18225
- config_name: top_ranked
  features:
  - name: query-id
    dtype: string
  - name: corpus-ids
    list: string
  splits:
  - name: test
    num_bytes: 498500
    num_examples: 40
  download_size: 210303
  dataset_size: 498500
configs:
- config_name: corpus
  data_files:
  - split: test
    path: corpus/test-*
- config_name: instruction
  data_files:
  - split: test
    path: instruction/test-*
- config_name: qrel_diff
  data_files:
  - split: qrel_diff
    path: qrel_diff/qrel_diff-*
- config_name: qrels
  data_files:
  - split: test
    path: qrels/test-*
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
  <h1 style="font-size: 3.5rem; color: #1a1a1a; margin: 0 0 20px 0; letter-spacing: 2px; font-weight: 700;">Core17InstructionRetrieval</h1>
  <div style="font-size: 1.5rem; color: #4a4a4a; margin-bottom: 5px; font-weight: 300;">An <a href="https://github.com/embeddings-benchmark/mteb" style="color: #2c5282; font-weight: 600; text-decoration: none;" onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">MTEB</a> dataset</div>
  <div style="font-size: 0.9rem; color: #2c5282; margin-top: 10px;">Massive Text Embedding Benchmark</div>
</div>

Measuring retrieval instruction following ability on Core17 narratives for the FollowIR benchmark.

|               |                                             |
|---------------|---------------------------------------------|
| Task category | t2t                              |
| Domains       | News, Written                               |
| Reference     | https://arxiv.org/abs/2403.15246 |

Source datasets:
- [jhu-clsp/core17-instructions-mteb](https://huggingface.co/datasets/jhu-clsp/core17-instructions-mteb)


## How to evaluate on this task

You can evaluate an embedding model on this dataset using the following code:

```python
import mteb

task = mteb.get_task("Core17InstructionRetrieval")
evaluator = mteb.MTEB([task])

model = mteb.get_model(YOUR_MODEL)
evaluator.run(model)
```

<!-- Datasets want link to arxiv in readme to autolink dataset with paper -->
To learn more about how to run models on `mteb` task check out the [GitHub repository](https://github.com/embeddings-benchmark/mteb).

## Citation

If you use this dataset, please cite the dataset as well as [mteb](https://github.com/embeddings-benchmark/mteb), as this dataset likely includes additional processing as a part of the [MMTEB Contribution](https://github.com/embeddings-benchmark/mteb/tree/main/docs/mmteb).

```bibtex

@misc{weller2024followir,
  archiveprefix = {arXiv},
  author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
  eprint = {2403.15246},
  primaryclass = {cs.IR},
  title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
  year = {2024},
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
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
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

task = mteb.get_task("Core17InstructionRetrieval")

desc_stats = task.metadata.descriptive_stats
```

```json
{
    "test": {
        "num_samples": 19939,
        "number_of_characters": 44471883,
        "documents_text_statistics": {
            "total_text_length": 44454438,
            "min_text_length": 7,
            "average_text_length": 2234.003618272275,
            "max_text_length": 2960,
            "unique_texts": 19143
        },
        "documents_image_statistics": null,
        "queries_text_statistics": {
            "total_text_length": 17445,
            "min_text_length": 198,
            "average_text_length": 436.125,
            "max_text_length": 1000,
            "unique_texts": 40
        },
        "queries_image_statistics": null,
        "relevant_docs_statistics": {
            "num_relevant_docs": 1744,
            "min_relevant_docs_per_query": 135,
            "average_relevant_docs_per_query": 43.6,
            "max_relevant_docs_per_query": 379,
            "unique_relevant_docs": 4739
        },
        "top_ranked_statistics": {
            "num_top_ranked": 40000,
            "min_top_ranked_per_query": 1000,
            "average_top_ranked_per_query": 1000.0,
            "max_top_ranked_per_query": 1000
        }
    }
}
```

</details>

---
*This dataset card was automatically generated using [MTEB](https://github.com/embeddings-benchmark/mteb)*