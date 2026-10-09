---
annotations_creators:
- derived
language:
- deu
- eng
- fra
- ita
- por
- spa
license: cc-by-4.0
multilinguality: multilingual
source_datasets:
- vidore/vidore_v3_finance_en
task_categories:
- visual-document-retrieval
- image-to-text
- text-to-image
task_ids: []
dataset_info:
- config_name: english-corpus
  features:
  - name: image
    dtype: image
  - name: id
    dtype: string
  splits:
  - name: test
    num_bytes: 1276313179
    num_examples: 2942
  download_size: 1260637509
  dataset_size: 1276313179
- config_name: english-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: test
    num_bytes: 404614
    num_examples: 8766
  download_size: 34487
  dataset_size: 404614
- config_name: english-queries
  features:
  - name: language
    dtype: string
  - name: id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: test
    num_bytes: 46355
    num_examples: 309
  download_size: 20788
  dataset_size: 46355
- config_name: french-corpus
  features:
  - name: image
    dtype: image
  - name: id
    dtype: string
  splits:
  - name: test
    num_bytes: 1276313179
    num_examples: 2942
  download_size: 1260637509
  dataset_size: 1276313179
- config_name: french-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: test
    num_bytes: 404614
    num_examples: 8766
  download_size: 34487
  dataset_size: 404614
- config_name: french-queries
  features:
  - name: language
    dtype: string
  - name: id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: test
    num_bytes: 46355
    num_examples: 309
  download_size: 23638
  dataset_size: 46355
- config_name: german-corpus
  features:
  - name: image
    dtype: image
  - name: id
    dtype: string
  splits:
  - name: test
    num_bytes: 1276313179
    num_examples: 2942
  download_size: 1260637509
  dataset_size: 1276313179
- config_name: german-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: test
    num_bytes: 404614
    num_examples: 8766
  download_size: 34487
  dataset_size: 404614
- config_name: german-queries
  features:
  - name: language
    dtype: string
  - name: id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: test
    num_bytes: 46355
    num_examples: 309
  download_size: 24090
  dataset_size: 46355
- config_name: italian-corpus
  features:
  - name: image
    dtype: image
  - name: id
    dtype: string
  splits:
  - name: test
    num_bytes: 1276313179
    num_examples: 2942
  download_size: 1260637509
  dataset_size: 1276313179
- config_name: italian-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: test
    num_bytes: 404614
    num_examples: 8766
  download_size: 34487
  dataset_size: 404614
- config_name: italian-queries
  features:
  - name: language
    dtype: string
  - name: id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: test
    num_bytes: 46355
    num_examples: 309
  download_size: 23164
  dataset_size: 46355
- config_name: portuguese-corpus
  features:
  - name: image
    dtype: image
  - name: id
    dtype: string
  splits:
  - name: test
    num_bytes: 1276313179
    num_examples: 2942
  download_size: 1260637509
  dataset_size: 1276313179
- config_name: portuguese-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: test
    num_bytes: 404614
    num_examples: 8766
  download_size: 34487
  dataset_size: 404614
- config_name: portuguese-queries
  features:
  - name: language
    dtype: string
  - name: id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: test
    num_bytes: 46355
    num_examples: 309
  download_size: 22622
  dataset_size: 46355
- config_name: spanish-corpus
  features:
  - name: image
    dtype: image
  - name: id
    dtype: string
  splits:
  - name: test
    num_bytes: 1276313179
    num_examples: 2942
  download_size: 1260637509
  dataset_size: 1276313179
- config_name: spanish-qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: test
    num_bytes: 404614
    num_examples: 8766
  download_size: 34487
  dataset_size: 404614
- config_name: spanish-queries
  features:
  - name: language
    dtype: string
  - name: id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: test
    num_bytes: 46355
    num_examples: 309
  download_size: 22966
  dataset_size: 46355
configs:
- config_name: english-corpus
  data_files:
  - split: test
    path: english-corpus/test-*
- config_name: english-qrels
  data_files:
  - split: test
    path: english-qrels/test-*
- config_name: english-queries
  data_files:
  - split: test
    path: english-queries/test-*
- config_name: french-corpus
  data_files:
  - split: test
    path: french-corpus/test-*
- config_name: french-qrels
  data_files:
  - split: test
    path: french-qrels/test-*
- config_name: french-queries
  data_files:
  - split: test
    path: french-queries/test-*
- config_name: german-corpus
  data_files:
  - split: test
    path: german-corpus/test-*
- config_name: german-qrels
  data_files:
  - split: test
    path: german-qrels/test-*
- config_name: german-queries
  data_files:
  - split: test
    path: german-queries/test-*
- config_name: italian-corpus
  data_files:
  - split: test
    path: italian-corpus/test-*
- config_name: italian-qrels
  data_files:
  - split: test
    path: italian-qrels/test-*
- config_name: italian-queries
  data_files:
  - split: test
    path: italian-queries/test-*
- config_name: portuguese-corpus
  data_files:
  - split: test
    path: portuguese-corpus/test-*
- config_name: portuguese-qrels
  data_files:
  - split: test
    path: portuguese-qrels/test-*
- config_name: portuguese-queries
  data_files:
  - split: test
    path: portuguese-queries/test-*
- config_name: spanish-corpus
  data_files:
  - split: test
    path: spanish-corpus/test-*
- config_name: spanish-qrels
  data_files:
  - split: test
    path: spanish-qrels/test-*
- config_name: spanish-queries
  data_files:
  - split: test
    path: spanish-queries/test-*
tags:
- mteb
- text
- image
---
<!-- adapted from https://github.com/huggingface/huggingface_hub/blob/v0.30.2/src/huggingface_hub/templates/datasetcard_template.md -->

<div align="center" style="padding: 40px 20px; background-color: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); max-width: 600px; margin: 0 auto;">
  <h1 style="font-size: 3.5rem; color: #1a1a1a; margin: 0 0 20px 0; letter-spacing: 2px; font-weight: 700;">Vidore3FinanceEnRetrieval</h1>
  <div style="font-size: 1.5rem; color: #4a4a4a; margin-bottom: 5px; font-weight: 300;">An <a href="https://github.com/embeddings-benchmark/mteb" style="color: #2c5282; font-weight: 600; text-decoration: none;" onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">MTEB</a> dataset</div>
  <div style="font-size: 0.9rem; color: #2c5282; margin-top: 10px;">Massive Text Embedding Benchmark</div>
</div>

Retrieve associated pages according to questions.

|               |                                             |
|---------------|---------------------------------------------|
| Task category | t2i                              |
| Domains       | Academic                               |
| Reference     | https://huggingface.co/blog/QuentinJG/introducing-vidore-v3 |

Source datasets:
- [vidore/vidore_v3_finance_en](https://huggingface.co/datasets/vidore/vidore_v3_finance_en)


## How to evaluate on this task

You can evaluate an embedding model on this dataset using the following code:

```python
import mteb

task = mteb.get_task("Vidore3FinanceEnRetrieval")
evaluator = mteb.MTEB([task])

model = mteb.get_model(YOUR_MODEL)
evaluator.run(model)
```

<!-- Datasets want link to arxiv in readme to autolink dataset with paper -->
To learn more about how to run models on `mteb` task check out the [GitHub repository](https://github.com/embeddings-benchmark/mteb).

## Citation

If you use this dataset, please cite the dataset as well as [mteb](https://github.com/embeddings-benchmark/mteb), as this dataset likely includes additional processing as a part of the [MMTEB Contribution](https://github.com/embeddings-benchmark/mteb/tree/main/docs/mmteb).

```bibtex
@misc{mace2025vidorev3,
  author    = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
  title     = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
  year      = {2025},
  month     = {November},
  day       = {5},
  publisher = {Hugging Face},
  journal   = {Hugging Face Blog},
  howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}}
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

task = mteb.get_task("Vidore3FinanceEnRetrieval")

desc_stats = task.metadata.descriptive_stats
```

```json
{}
```

</details>

---
*This dataset card was automatically generated using [MTEB](https://github.com/embeddings-benchmark/mteb)*