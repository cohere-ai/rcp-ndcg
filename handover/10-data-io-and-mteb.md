# Workstream 10: data I/O and MTEB interoperability

Read these first:
- `handover/00-MASTER.md`, decisions 27-32;
- `handover/specs/mteb-data-model.md`, the evidence: MTEB's loader, model protocols, results and upload layout, with
  `file:line` citations.

When to run it:
- after 09, because both touch `data/`;
- before 06 and 07, because it changes core records, and they must change before the public-surface freeze in 07.

Staffing: one builder and two verifiers.

**Goal.** Any MTEB-formatted dataset loads with one URI. Every output (embedder, late-interaction, reranker, LLM
judge) can be scored by MTEB and submitted to it. Datasets we publish are exactly what MTEB's own upload helper
writes. Be maximally forward compatible with MTEB (decision 31).

## State at M3 (do not redo)

`rcp_ndcg.data.io` already exists:
- One reader/writer contract (`SourceReader`, `SinkWriter`) and the `READERS`/`WRITERS` tables.
- Readers: `beir`, `jsonl`, `hf` (column heuristics over `datasets`), `images`, `videos`, `frames`, `pdf`.
- Writers: `beir`, `jsonl`.
- `data convert`.

Elsewhere:
- `Rankings` reads and saves parquet, TREC, JSONL and CSV.
- `rcp_ndcg.eval.mteb` exposes the public suites as mteb tasks and adds `ndcg_float_at_k`.
- `runners` and `adapters` load plugins through entry points.

## Gaps

1. `hf://` reads only rcp-ndcg's own layout. The `hf` reader cannot open an MTEB repo, because MTEB keeps corpus,
   queries and qrels in separate configs. So `hf://mteb/nfcorpus` fails today.
2. The Hub path is special-cased inside `data/dataset.py` (`_load_hub*`, `_card_paths`, `_hub_*`). Only that path
   records the revision, candidates (`top_ranked`), exclusions and gains; readers cannot supply them.
3. Nothing is exported in MTEB's formats: no predictions file, no way to score our outputs inside mteb, no
   MTEB-exact dataset writer.
4. BEIR files compressed with gzip (`*.jsonl.gz`, `*.tsv.gz`) cannot be read.

## A. Reader contract and one dataset path

1. **Widen `SourceReader`.**
   - Optional `candidates()` (`top_ranked`), `excluded()`, `gains()`/`thetas()`.
   - A `provenance` property: source URI, resolved revision commit, `subset`, `split`.
   - The current derivations stay.
2. **`load_dataset` builds every `Dataset` from a reader through one function.**
   - The Hub code leaves `dataset.py`.
   - The `JsonlReader` special case in `_load_reader` becomes the reader's own `candidates()`.
3. **`Dataset` gains `subset`, `split` and an optional MTEB `task` name** (decision 29). Exports are keyed by
   `(task, subset, split)`.
4. **Entry-point groups `rcp_ndcg.readers` and `rcp_ndcg.writers`**, the same pattern as `runners`/`adapters`
   (`runners/registry.py`). A third-party format is one class in its own package. A plugin must pass the shared
   conformance suite (`tests/data/test_io_contract.py`), exposed to plugin authors through `rcp_ndcg.testing`.

## B. Ingest

1. **`data/io/hub.py`, the Hub reader behind `hf://<owner>/<repo>[/<subset>][@revision]`.**
   - Driven by the dataset card and following MTEB's own rules (spec section 1).
   - Lists the configs at a revision resolved to a commit.
   - Subsets: from the `{s}-corpus|queries|qrels` prefixes, else `default`.
   - Config resolution follows the table in the spec, including the `query` override and the `default`-then-`qrels`
     fallback.
   - Split: the requested one, else the config's only one.
   - Reads the data files directly through `huggingface_hub` and pyarrow (parquet, jsonl, jsonl.gz, tsv). It needs
     neither `datasets` nor `mteb`.
   - Normalises: `_id` becomes `id`, all ids are strings, `instruction` is merged by id (the config wins, as in
     mteb), `top_ranked` becomes `candidates`, media columns `image`/`video` (audio is deferred).
   - rcp-ndcg's extras are read where present: the qrels `gain`/`theta` columns and the `-excluded` config. rcp-ndcg's
     own layout is then simply this reader, and the bespoke `_load_hub*` code is deleted.
   - Carries over from the retired `hf` reader: `document_parts` (text / image / both, for corpora such as ViDoRe v3
     that ship OCR text and page images in one row) and content-addressed image persistence (`store_media`).
2. **`mteb:<TaskName>[/<subset>][@split]`, in the `[mteb]` extra.**
   - Loads the task through mteb itself (`task.load_data()`) and converts `task.dataset[subset][split]`.
   - This gives MTEB's exact ids and texts for the 113 tasks with custom loaders, for example ViDoRe v1's id
     prefixes and BRIGHT's exclusion-as-`top_ranked`.
   - Records the task name, subset, split and `dataset_revision`.
3. **Retire the `hf` column-heuristics reader** (decision 32). Non-MTEB Hub data is converted once: through
   `Dataset.from_records`, the MTEB writer, or a local BEIR/JSONL folder.
4. **BEIR:** read gzip-compressed files (`*.jsonl.gz`, `*.tsv.gz`), through `storage` so remote URIs keep working.
5. **Duplicates** (decision 30):
   - Exact duplicates are folded and counted in a note: the same pair with the same grade, or the same id with the
     same content.
   - Conflicting duplicates are refused, naming the rows, unless `duplicates: last` (MTEB's behaviour). That choice is
     recorded in the dataset's provenance with counts.
   - Measure how often the canonical repos contain duplicates, and report it.

## C. Data model (decision 27)

1. **`Document.title: str | None`** is an additive change to the core record. `text` is the body, and nothing joins
   at read time.
2. **One join, MTEB's.**
   - A model reads `(title + " " + body).strip()`, or the body alone when there is no title: byte-identical to mteb's
     dataloader (`_create_dataloaders.py:58-74`).
   - A model or recipe may take the title separately, as mteb's model integrations can.
   - This replaces today's blank-line join everywhere text is formatted: embedding, reranking and the judge's prompts.
     The paper's published runs used the blank line; say so in `REPRODUCIBILITY.md` (no compatibility is owed to the
     paper's layout, decision 31).
   - If formatting enters an identity or fingerprint (`rcp-fp/3`, run identities), bump it per the versioning page.
   - `run_all` must stay unchanged. If a number moves because it re-joins text, stop and report it to the owner.
3. **The instruction format is declared too.**
   - Today `Task: <instruction>\nQuery: <text>`.
   - `mteb` style: `query + " " + instruction`.
   - It stays a separate field on `Query`. Whether the default should follow mteb too is the owner's call: ask before
     changing it, because it changes every instruction-bearing prompt (BRIGHT).
4. **Deferred, both additive:** conversation-style queries and audio.

## D. Export

1. **Rankings to MTEB predictions.**
   - `Rankings.save(format="mteb")` writes `{Task}_predictions.json` in mteb's exact format.
   - Every filtered query is present; no query lacks qrels; there are no empty dicts; at most 1,000 documents per query.
   - TREC run files stay.
2. **Scoring inside mteb** (`rcp_ndcg.eval.mteb`).
   - A `SearchProtocol` that serves stored `Rankings` to `mteb.evaluate`, with a `ModelMeta` built from our model
     identity, its required fields declared by the caller.
   - This yields genuine `TaskResult` files in mteb's `ResultCache` layout; `submit_results` opens the results PR.
   - Embedders, late-interaction models, rerankers and LLM judges all produce `Rankings`, so one route covers all four.
   - Document: the leaderboard also needs the model's `ModelMeta` merged into mteb, which is outside this repository.
3. **The MTEB dataset writer** (`data/io/mteb.py`).
   - Writes exactly what `push_dataset_to_hub` writes (spec section 5): configs `{s-}corpus` (`id`, `title`, `text`),
     `{s-}queries` (`id`, `text`, `instruction`), `{s-}qrels` (`query-id`, `corpus-id`, `score` int64) and
     `{s-}top_ranked`.
   - Parquet files at `{config}/{split}-00000-of-0000N.parquet`, and a card from mteb's template.
   - rcp-ndcg's extras are added only where mteb ignores them: `gain`/`theta` columns on the qrels, and an `-excluded`
     config.
   - Exclusions are also folded into `top_ranked`, because mteb reads no other config.
4. **Fractional grades** (decision 28): exporting to MTEB refuses a non-integral `score` unless an integer grade is
   given. Continuous gains travel in the extra columns.
5. **Republishing** (decision 31).
   - A converter re-lays every published rcp-ndcg dataset in the writer's exact layout, with the eval split name
     (`test`), not `train`.
   - Validated by loading the result with mteb's own `RetrievalDatasetLoader`.
   - The owner pushes, combined with the move to a Hugging Face organisation (master section 13).
   - Paper configs pin commits, so they keep working.
   - Align the task definitions with the owner's local mteb PR (ask the owner for it).

## Gates

- The master's quality bar, `run_all` unchanged, contract snapshots and schemas regenerated, CHANGELOG entries.
- Offline fixtures of the nine canonical repos in the spec: their cards plus a few sampled rows each. Network-gated
  live loads at pinned revisions.
- **Round trips** (in the `[mteb]` extra test environment):
  - Our writer's output, loaded by mteb's `RetrievalDatasetLoader`, equals our `Dataset`.
  - `hf://` and `mteb:<Task>` give the same data for standard-layout tasks.
  - Stored `Rankings` scored through `mteb.evaluate` give the same integer nDCG@10 as our evaluator. Check the tie rule:
    mteb breaks ties by doc id.
- The join gate: `mteb` gives byte-identical text to mteb's dataloader for a sample of documents and queries.
- Report in `handover/reports/10-data-io-and-mteb.md`.
