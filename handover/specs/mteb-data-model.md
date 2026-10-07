# MTEB's data model for retrieval-like tasks (evidence for workstream 10)

Read from the mteb 2.21.10 source, and diffed against 2.0.1 where behaviour changed. Real dataset cards and file
listings were checked on the Hub (2026-10-07). Citations are `file:line` in the mteb package:

| Prefix | File |
|---|---|
| L | `abstasks/retrieval_dataset_loaders.py` |
| R | `abstasks/retrieval.py` |
| A | `abstasks/abstask.py` |
| TM | `abstasks/task_metadata.py` |
| DL | `_create_dataloaders.py` |
| E | `_evaluators/retrieval_evaluator.py` |
| MET | `_evaluators/retrieval_metrics.py` |
| SW | `models/search_wrappers.py` |
| P | `models/models_protocols.py` |
| T | `types/_encoder_io.py` |
| TR | `results/task_result.py` |
| RC | `cache/result_cache.py` |
| MM | `models/model_meta.py` |
| EV | `evaluate.py` |

Re-verify every citation against the mteb version the workstream pins.

## 1. Hub layouts

**One loader for every retrieval-like task.**
- Retrieval, Reranking, InstructionRetrieval/Reranking, Any2Any and DocumentUnderstanding tasks all subclass
  `AbsTaskRetrieval`; the task `type` is only a label (TM:189-228).
- v2 reranking is retrieval plus `top_ranked`. `AbsTaskReranking` is deprecated, and the list of tasks still on the
  old format is empty (`abstasks/text/reranking.py:21-56`).

**Config names** (hf_subset `s`; `"default"` means no prefix, L:76):

| Part | Default subset | Subset `s` | Notes |
|---|---|---|---|
| corpus | `corpus` | `s-corpus` | required (L:155) |
| queries | `queries`, but a `query` config always wins | `s-queries` | L:171-176; the `query` check ignores the subset |
| qrels | `default`, else `qrels` | `s-qrels` | required (L:191-200) |
| top_ranked | `top_ranked` | `s-top_ranked` | optional, exact name (L:101-105) |
| instruction | `instruction` | `s-instruction` | optional (L:107-114) |

Every other config is ignored by the loader. That includes rcp-ndcg's `-excluded` and FollowIR's `qrel_diff`, which
the FollowIR task reads itself.

**Subsets and splits.**
- Subsets apply only when `eval_langs` is a dict; its keys are the config prefixes (TM:655-662, R:270-276).
  Cross-lingual keys look like `"en-de"`.
- Split resolution (L:123-135): the eval split if the config has it, else the config's only split, else an error.
  This is why the BEIR-style `corpus`/`queries` splits work.
- `dataset = {path, revision}` are both required (TM:413-429). `eval_splits` defaults to `["test"]` (TM:486).

**Columns and casts.**
- corpus:
  - `_id` is cast to string and renamed `id` (L:159-162); a column already named `id` is NOT cast.
  - `title` is optional and `text` is the body.
  - Media columns must be named exactly `image`, `audio` or `video` (R:282-296).
- queries:
  - The same id rule applies (L:177-180).
  - `text` is a string or a list of conversation turns.
  - `instruction` and media columns are optional.
- qrels:
  - Only `query-id`, `corpus-id` and `score` are kept, so extra columns (rcp-ndcg's `gain`, `theta`) are dropped
    (L:203).
  - The columns are cast to string, string and **int32** (L:205-213). A fractional score fails to load; pyarrow
    raises "truncated". Whole-number floats and negative scores pass.
  - A duplicate pair keeps the last value (L:215-220).
- top_ranked: `query-id` as a string and `corpus-ids` as a list of strings (L:229-247).
- instruction:
  - Rows are `(query-id, instruction)`, merged onto the queries by id.
  - A query with no instruction row raises an error.
  - When both the config and a column exist, the config wins (L:271-287).
- Query filtering:
  - At load, queries are cut to those with qrels (L:97-99).
  - At evaluation, queries with empty qrels are dropped (R:67-80, 372-377).
  - Queries whose qrels are all score 0 are kept.

**Repos that exercise each layout** (cards and API listings checked 2026-10-07):

| Repo | What it exercises |
|---|---|
| `mteb/nfcorpus` | v1/BEIR layout: `default` (qrels, float64 score), `corpus` (split `corpus`), `queries` (split `queries`), jsonl |
| `mteb/MIRACLRetrieval` | 54 `{lang}-{corpus,qrels,queries}` configs, split `dev`, parquet shards |
| `mteb/AskUbuntuDupQuestions` | v2 reranking with `top_ranked` |
| `mteb/Core17InstructionRetrieval` | an instruction config and an instruction column, plus an extra `qrel_diff` config |
| `mteb/blink-it2i` | Any2Any with the `query` config, the `qrels` config and image columns |
| `vidore/vidore_v3_finance_en_mteb_format` | multilingual image retrieval, category `t2i` |
| `mteb/arxivqa_test_subsampled_beir` | integer ids the task code remaps to `query-{split}-{id}` / `corpus-{split}-{id}` |
| `mteb/BRIGHT` | a raw layout; the task builds `top_ranked` as the corpus minus its exclusions |
| `fabianschmidt-cohere/rcp-ndcg-nanobeir` | MTEB v2 plus extra qrels columns and an extra config, both ignored |

**Custom loading.**
- 113 task files override `load_data` and 20 override `dataset_transform` (A:155-164).
- v1 dict attributes are converted at evaluation (R:113-229).
- A task's id space can therefore differ from its Hub repo's; ViDoRe v1 is an example.

## 2. In-memory model

- `RetrievalSplitData = {corpus, queries, relevant_docs, top_ranked | None}` (L:28-41).
- Relevant documents are `{qid: {did: int}}`; the run is `{qid: {did: float}}` (T:259-285).
- **Documents:** `text = (title + " " + text).strip()` when the title is non-empty, else `text.strip()`.
  `title` and `body` stay available as separate fields (DL:58-74). A `None` title raises an error.
- **Queries:** `text = query + " " + instruction`, so the instruction is appended (DL:77-89).
- **Conversations:** turns become `"role: content; ..."`, with the instruction prepended (DL:92-151).
- **Modalities:** each side's modalities come from `category`; for example `it2i` means image+text queries and
  image documents (TM:672-702). Other media columns are dropped (DL:303-308).

## 3. Model side and precomputed outputs

**Protocols.** All three are duck-typed (`runtime_checkable`):
- `SearchProtocol`: `index` and `search(..., top_k, top_ranked) -> {qid: {did: score}}`, plus `mteb_model_meta`
  (P:23-82).
- `EncoderProtocol` (P:85-188).
- `CrossEncoderProtocol` (P:191-247).

**Dispatch** (R:390-401):
- An encoder is wrapped for full-corpus search, or scores only the listed documents when `top_ranked` exists
  (SW:224-377).
- A cross-encoder requires `top_ranked` (SW:539-558).
- Late interaction is a plain `SearchProtocol` (`model_implementations/pylate_models.py:33-130`).
- `top_k = 1000` (R:102, 111).

**Scoring precomputed outputs.** There is no built-in way. Three routes:
1. A `SearchProtocol` class whose `search` returns the stored run. It needs a real `ModelMeta`, otherwise MTEB uses
   the name `no_model_name/available` (EV:78-88).
2. `convert_to_reranking(path, top_k)` (R:725-774), which turns a predictions file into `top_ranked`.
3. Pre-filling `CachedEmbeddingWrapper`'s cache. **Not recommended:** it is a raw float32 memmap keyed by hashes of
   the joined text, prompts are not part of the key, and it has no audio or video.

**Evaluator.**
- `ignore_identical_ids` removes `did == qid` from the results only (E:124-136); 32 task files set it, ArguAna among
  them.
- Metrics come from pytrec_eval, rounded to 5 decimals.
- MRR ties are broken by (score, doc id), descending (MET:37-55).
- A result for a query that has no qrels raises an error (MET:37-55).

## 4. Results and submission

**Predictions file.** `{prediction_folder}/{Task}_predictions.json` =
`{"mteb_model_meta": {model_name, revision}, subset: {split: {qid: {did: score}}}}` (A:277-326). It is written before
`ignore_identical_ids` is applied.

**TaskResult JSON** (TR:170-177).
- Top-level fields: `dataset_revision`, `task_name`, `mteb_version`, `scores`, `evaluation_time`,
  `kg_co2_emissions`, `date`, `evaluation_phases`.
- `scores` is `{split: [{...metrics, main_score, hf_subset, languages}]}` (TR:262-274).

**Folder layout.**
- `results/{org__model}/{revision}/{Task}.json`, plus `model_meta.json` and `run_settings.jsonl` (RC:263-433).
- **ModelMeta** forbids unknown fields and has many required fields; `name` must be `org/model` (MM:236-268,
  413-422).

**Reaching the leaderboard.**
- `ResultCache.submit_results` opens a PR on `embeddings-benchmark/results` (RC:1187-1303).
- The leaderboard shows only models whose `ModelMeta` is registered in mteb (`results/benchmark_results.py:233-247`).

## 5. Dataset upload: what `push_dataset_to_hub` writes (A:697-728, R:634-723)

Per subset:

| Config | Columns |
|---|---|
| `{s-}queries` | `id`, `text`, optional `instruction` |
| `{s-}corpus` | `id`, `title`, `text` |
| `{s-}qrels` | `query-id`, `corpus-id`, `score` (int64); the config is named `qrels`, not `default` |
| `{s-}top_ranked` | `query-id`, `corpus-ids` (list of strings) |

- Splits are the eval split names. Files are `{config}/{split}-0000k-of-N.parquet`.
- The README comes from `dataset_card_template.md` (TM:704-865).
- No `instruction` config is written.
- The optional `eval.yaml` path looks broken in 2.21.10 (A:743, 765).

## 6. Pitfalls to design for

- **Integer ids:** an integer `id` column is not cast, so every query is dropped.
- **Qrels:** scores are integers only, and duplicates keep the last value.
- **Duplicate corpus ids** collapse into one document.
- **Titles:** a `None` title raises an error.
- **Instructions:** appended to the query, not prepended.
- **`top_ranked`** turns every model into a reranker.
- **Splits:** each split is loaded separately.
- **`trust_remote_code`** is deprecated.
- **Revision pins** become `TaskResult.dataset_revision`.
- **Changes between 2.0.1 and 2.21.10:**
  - qrels moved from float-then-`int()` to an int32 cast;
  - `top_ranked`/`instruction` configs need exact names;
  - ties are broken by doc id;
  - new metrics: `hit_rate` and `accuracy`;
  - `TaskResult` gained `date` and `evaluation_phases`.
