# Lane l10a: workstream 10 A, B and C1 — data readers, one dataset path, the data-model fields

## Status

DONE. The gate passes on the merged tree (`GATE: PASS`, revision below), `run_all` is unchanged
(1022/987/35/0, 67/67, 82/82), and the two adversarial verifier rounds ended in PASS after the fixes below.

## Commits (lane branch `lane/l10a`)

| Commit | Subject |
|---|---|
| `a7aa9a15` | The data model carries provenance fields: `Document.title` as its own field (the body stays in `text`, nothing joins at read time), `Query.instruction` as the per-query instruction, and `Dataset.subset/split/task/task_instruction` with the `(task, subset, split)` export key |
| `4f21904b` | The Hub reader (`hf://`) reads mteb's card-driven layout through `huggingface_hub` and pyarrow, the `hf` column-heuristics reader retires (`document_parts` and media persistence carry over), and the reader/writer tables become `rcp_ndcg.readers`/`writers` entry points with the shared conformance suite in `rcp_ndcg.testing` |
| `14b1b2aa`, `abb87eec` | ruff import order / line length (mechanical) |
| `20faae0d` | The live canonical repositories load (the raw-binary media cells sniff their format; the multilingual corpus is not materialised in the test), gzip-compressed BEIR files read through `storage`, the duplicates measurement finds none, and the CHANGELOG carries the data-model and reader entries |
| `b224b569` | Verifier round-1 fixes: the `mteb:` reader converts v1-style tasks as mteb does, video media cells keep their container MIME, the duplicates counts cover every eager table and BEIR's provenance, the conformance suite counts a title as content and runs over the Hub and mteb readers, `document_parts` is corpus-only, an mteb media query keeps its instruction, exact duplicates fold in the sidecar and derived qrels too, `mteb:` URIs record their pinned revision, an unknown `hf` option is a typed `ConfigError`, dead code gone |
| `b23bdc5a` | Round-2 fixes: `duplicates=last` resolves the tables that can replace a row (labels, pools, exclusions) and refuses a streamed corpus/query conflict with the option's scope named; the conformance suite checks the qrels references its docstring promised (the nfcorpus and blink samples trimmed to match); `revision_payload` raises the typed errors; the unused `_Labels.counts` and the `corpu_id` message are gone; the docs state the policy precisely |
| `a0eb087b` | The duplicates docstrings and hints state the streamed-table scope exactly (an optionless source's hint names fixing the rows only), and the `mteb:` reader's import is a typed `DependencyError` like its revision accessor |
| `31f8b24d`, `7b5704f0` | Merge `rfc-0001` (the second brings lane w09's pipeline, which merges before this lane; CHANGELOG resolved by union, snapshots regenerated on the merged tree) |

## What changed (per brief item)

**A. Reader contract and one dataset path**

- `SourceReader` gains `candidates()` (`top_ranked` pools), `excluded()`, `gains()`/`thetas()` (the released
  calibrated values), `provenance` (the new `Provenance`, `DuplicateCounts`, `DuplicatesPolicy` and the
  `DuplicateFold` helper in `data/io/base.py`) and `task`/`task_instruction`; the existing derivations stay.
- `load_dataset` builds every `Dataset` from a reader through one function (`_from_reader`); the Hub code
  leaves `data/dataset.py`; `JsonlReader.candidates()` replaces the `_load_reader` special case.
- `Dataset` gains `subset` (`"default"`), `split` (`"test"`), `task`, `task_instruction`, `provenance` and
  `export_key`; `DocumentRow.title` and `from_records(..., subset=, split=, task=, task_instruction=)` follow.
- The reader/writer tables are the `rcp_ndcg.readers`/`rcp_ndcg.writers` entry-point groups (declared in
  `rcp-ndcg/pyproject.toml`, loaded on demand by `data/io/registry.py`); the shared conformance suite is
  public as `rcp_ndcg.testing.io_conformance` and runs over every built-in reader (the contract cases, six
  Hub fixtures, and `mteb:NFCorpus`).

**B. Ingest**

- `data/io/hub.py`: the Hub reader behind `hf://<owner>/<repo>[/<subset>][@revision]`, card-driven, mteb's
  own rules (config resolution including the `query` override and the `default`-then-`qrels` fallback, split
  resolution, the instruction config's precedence and its per-query refusal, `_id`→`id` with string ids,
  `top_ranked`→candidates, media columns→parts, queries cut to those with qrels), reading parquet, jsonl,
  jsonl.gz and tsv directly through `huggingface_hub` + pyarrow (no `datasets`, no `mteb`), with rcp-ndcg's
  extras (`gain`/`theta` columns, the `-excluded` config) and the retired reader's `document_parts` and
  content-addressed persistence. The offline absence semantics (`.no_exist`, the snapshot listing, the
  offline/provider hints) moved over unchanged.
- `data/io/mteb_task.py`: `mteb:<TaskName>[/<subset>][@split]` in the `[mteb]` extra, running mteb's own
  `load_data()` (plus the v1 conversion mteb's `evaluate` does), converting `task.dataset[subset][split]`,
  recording task/subset/split/revision and `TaskMetadata.prompt`.
- The `hf` column-heuristics reader is deleted; `datasets` leaves the `[data]` extra (the lock and the
  constraints file regenerated) and `EXTRA_FOR_MODULE`/the packaging gates map `datasets` to `[mteb]`.
- BEIR reads gzip-compressed files (`corpus.jsonl.gz`, `queries.jsonl.gz`, `qrels/<split>.tsv.gz`) through
  `storage`; the BEIR writer writes `Document.title` into the title column.
- Duplicates: exact duplicates fold (the labels, the pools, the exclusions, the jsonl sidecar and the derived
  ranking-shape qrels), conflicts refuse unless `duplicates="last"`, which resolves the tables that can
  replace a row and refuses a streamed corpus/query conflict with the option's scope named; the counts are in
  the provenance (labels/pools/exclusions on a load; corpus/query folds are logged as those tables are read).
  The measurement over the canonical repositories found none (numbers below).

**C1. Data model**

- `Document.title: str | None` (body in `text`; nothing joins at read time) and `Query.instruction` as the
  per-query instruction, never merged into the text at load. No formatting or join is added (lane l10c's job).

## Verification

**Round 1** (two fresh verifiers on the owner's verifier model (`deepseek-v4-1-flash`, xhigh), lenses A and B, in parallel):
both **FAIL**. Findings and what was done:

| Finding (severity) | Evidence | Fix |
|---|---|---|
| `mteb:` crashed on every v1-style custom-loaded task (blocker): `task.dataset is None` after `load_data` | `AttributeError: 'NoneType' object has no attribute 'get'` on `mteb:BrightBiologyRetrieval`; BRIGHT is the spec's named example | `convert_v1_dataset_format_to_v2(num_proc=None)` after `load_data()` (mteb's own conversion); test `test_a_v1_style_task_converts_like_mteb_does` (real task, 57 359 docs) |
| Video media cells failed for the `{"bytes","path"}` struct (mime `None` → `MediaError`) and for a decoded decoder object (bare `AttributeError`) (major) | `MediaError: unknown media type '.avi'`; `AttributeError: 'VideoDecoder' object has no attribute 'save'` | `_with_mime` computes the container MIME in every branch; a decoded object without `.save` is a typed `DataError`; tests `test_a_video_cell_in_the_struct_form_keeps_its_container_mime`, `test_a_decoded_video_object_is_refused_by_name` |
| Duplicates counts missing from the provenance (BEIR `None`; Hub labels only) (major) | folded BEIR label → `provenance.duplicates: None`; Hub corpus fold uncounted | both readers accumulate `DuplicateCounts` over every table read (first pass per table, no double count); tests `test_the_beir_provenance_records_its_label_duplicates`, `test_the_hub_provenance_records_every_eager_tables_duplicates` |
| The conformance suite did not run over `hf`/`mteb` and rejected title-only documents (major) | Core17's corpus has 580 title-only rows; `READER_CASES` had no hub/mteb case | a non-empty title counts as content; six Hub fixtures + `mteb:NFCorpus` run `io_conformance`; the fixtures were made internally consistent |
| `mteb:` media query dropped its instruction; `document_parts` filtered queries; docs named a nonexistent task; dead code (`join_title`, unreachable `_patterns` branch, registry duplicate warning, duplicate provenance read, `_conflicts`); the sidecar/derived qrels refused exact duplicates; `dataset_uri_revision` ignored `mteb:`; an unknown `hf` option was a bare `TypeError`; `cli.md`'s `data convert` summary was stale (minor) | verifier reproductions in the lane's scratch directory | each fixed with a test where the behaviour is observable; `join_title` removed (decision 27: no second join); `_load_hub` validates options through `get_reader("hf", ...)` |

**Round 2** (one fresh confirmation verifier, lens A+B): **FAIL** — every round-1 fix reproduced as fixed,
but one major remained: `duplicates="last"` on a conflicting *corpus/query* row yielded the id twice, the
`Dataset` then refused, and the provenance counted a resolution that never reached the data
(`rows: ['x','x']`, `provenance … resolved=1`, then `DataError corpus: corpu_id 'x' appears twice`). Fixed
by `DuplicateFold(replaceable=...)`: `last` resolves only where the caller can replace an already-read row
(labels, pools, exclusions — materialised); a streamed corpus/query table refuses with the option's scope
named and the count stays 0. Minors (unused `_Labels.counts`, the `corpu_id` message, the hint advertising an
option the optionless sources reject, `revision_payload`'s raw errors, the conformance qrels-reference
promise) were fixed too.

**Round 3** (one fresh confirmation verifier, lens A+B): **PASS**. Every F1–F6 fix reproduced independently
(corpus/query conflicts under `last` refuse with `resolved=0`; pool/exclusion/label conflicts take the last
row and count; exact duplicates fold once over re-reads; the conformance suite goes red on mutated qrels
references; `DependencyError`/`ConfigError` typed; docs match), the full bar green, and `tests/data` run
twice left `git status` clean. Two minors were reported and fixed (`a0eb087b`): stale public docstrings about
the `last` scope, and the optionless hint; plus the residual bare `import mteb` in `_load`, now a typed
`DependencyError`.

**Duplicates measurement** (network-gated `tests/data/test_io_hub_live.py::test_the_canonical_repositories_hold_no_duplicate_labels`,
pinned revisions; raw counts over the labels, queries and corpus of each measured repository):

```
mteb/nfcorpus: labels=0 queries=0 corpus=0
mteb/AskUbuntuDupQuestions: labels=0 queries=0 corpus=0
mteb/Core17InstructionRetrieval: labels=0 queries=0 corpus=0
mteb/blink-it2i: labels=0 queries=0 corpus=0
vidore/vidore_v3_finance_en_mteb_format/english: labels=0 queries=0 corpus=not measured (too large)
fabianschmidt-cohere/rcp-ndcg-nanobeir/NanoArguAnaRetrieval: labels=0 queries=0 corpus=0
```

`mteb/MIRACLRetrieval` and `mteb/BRIGHT` are excluded on purpose: MIRACL's corpus is 32.9M rows across 28
shards (a sampled corpus cannot cover the sampled qrels), and BRIGHT's raw layout has no labels for this
reader (`mteb:BrightBiologyRetrieval` serves it).

**Failing test first** (the red runs behind the fixes): the `Document.title` test failed with
`AttributeError: 'Document' object has no attribute 'title'`; the live blink load failed with
`'bytes' object has no attribute 'save'` (raw-binary media cells); the gzip tests failed with
`ValueError: Argument 'encoding' not supported in binary mode` (and the qrels split name `dev.tsv` vs `dev`);
the conformance tests failed with `candidates() names documents the corpus lacks` before the fixtures were
made consistent; the round-1 verifier's own reproductions (the v1 `AttributeError`, the `.avi` `MediaError`,
`provenance.duplicates: None`) and the round-2 corpus-conflict reproduction are the red evidence for those
fixes.

## Checks (last runs, merged tree)

```
bin/heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider
  -> 3282 passed, 93 skipped
uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider
  -> 289 passed, 52 skipped
uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider
  -> 570 passed, 225 skipped
uv run --no-sync ruff format --check . && uv run --no-sync ruff check . && uv run --no-sync basedpyright
  -> 538 files already formatted; All checks passed; 0 errors, 0 warnings, 0 notes
uv run --no-sync mkdocs build --strict -d <scratch>/site
  -> Documentation built
RCP_EXPERIMENTS_DATA=<experiments-data> uv run --no-sync python experiments/run_all.py
  -> leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
     human study: 67 checks, 67 match, 0 failed; external LLM judges: 82 checks, 82 match, 0 failed
RCP_NDCG_NETWORK_TESTS=1 RCP_NDCG_TEST_TIMEOUT=600 uv run --no-sync pytest tests/data/test_io_hub_live.py -q
  -> 10 passed
RCP_NDCG_NETWORK_TESTS=1 RCP_NDCG_TEST_TIMEOUT=900 the scratch venv with mteb 2.21.6 -m pytest tests/data/test_io_mteb_task.py -q
  -> 12 passed
/root/repos/rcp-ndcg-lanes/bin/gate lane/l10a
  -> GATE: PASS (see the gate SUMMARY for the revision)
```

The mteb reader's tests were run in a scratch venv (the lane's scratch venv,
mteb 2.21.6 + CPU torch, outside the repository and outside the lane's venv, as the gate builds its own
vllm venvs); the lane's own environment was never re-resolved beyond the `uv lock` the manifest change
required (regenerated lock + constraints, environment refreshed with `.github/scripts/cpu-env.sh dev docs`,
the same command the gate runs).

## Open questions

- `Dataset.from_records` still refuses *any* duplicate qrels row (strict in-memory validation), while a
  *load* folds exact duplicates (decision 30). The docs now say so explicitly; if the in-memory path should
  fold too, that is a small change for the owner to call.
- The Hub reader's provenance counts the tables a load reads (labels, pools, exclusions); a corpus's and a
  query table's folds join the reader's counts when those tables are read (they are lazy), and are logged.
  If the frozen `Dataset.provenance` must carry corpus counts too, the corpus would have to be read eagerly
  (rejected: it defeats lazy loading for 10^7-row corpora).
- `mteb:` video tasks hand back a decoded `VideoDecoder` (no bytes this reader can encode); the reader
  refuses it with a typed error and the hint points at the container-bytes route. If a task's video media
  must be judged, that is a media-preparation (frame policy) question for lane 09/06, not a dataset-reader
  one.
- The `hf://` reader is mteb's layout only; a repository in another layout (BRIGHT) is loaded through its
  task (`mteb:`). If a future Hub dataset needs a third layout, it is converted once (decision 32).

## CHANGELOG entry

Under `## Unreleased`, grouped as the repository asks (`### Public surface`, `### Fixed`, `### Changed`,
`### Removed`), the lane added exactly these bullets (see `CHANGELOG.md` for the full text; the merges kept
lane w09's and lane l08-cat's entries by union):

- **Public surface**: *The data model carries provenance* (decisions 27, 29, 33: `Document.title`,
  `Query.instruction`, `Dataset.subset/split/task/task_instruction/provenance/export_key`,
  `DocumentRow.title`, `from_records` arguments); *The reader contract widens and moves to entry points*
  (`candidates`/`excluded`/`gains`/`thetas`/`provenance`/`task`/`task_instruction`, `Provenance`,
  `DuplicateCounts`, `DuplicatesPolicy`, the `rcp_ndcg.readers`/`writers` groups,
  `rcp_ndcg.testing.io_conformance`); *`hf://` is mteb's layout, and `mteb:<Task>` loads a task through
  mteb* (decisions 28, 31, 32); *`load_dataset` takes `split=`*; *`WarningCode` gains `CARD_UNCACHED`*;
  *`rcp_ndcg.data.media` gains `media_extension`/`image_dimensions` and `store_media(..., mime=)`*.
- **Fixed**: *A raw-binary media column reads by its magic numbers* (plus the container MIME and the decoded
  video refusal, and `document_parts` as a corpus setting); *A v1-style `mteb:` task converts like mteb's own
  `evaluate` does*.
- **Changed**: *The Hub reader reads mteb's card-driven layout*; *Duplicates fold, conflicts refuse,
  `duplicates="last"` takes the last row* (with the streamed-table scope); *BEIR reads gzip-compressed files*.
- **Removed**: *The column-heuristics `hf` reader* (decision 32), with the `datasets` dependency leaving
  `[data]`.

## Public surface changes

- New public names: `Document.title`; `Dataset.subset/split/task/task_instruction/provenance/export_key`;
  `DocumentRow.title`; `from_records(subset=, split=, task=, task_instruction=)`; `load_dataset(split=)`;
  `SourceReader.candidates/excluded/gains/thetas/provenance/task/task_instruction`;
  `rcp_ndcg.data.io.{Provenance, DuplicateCounts, DuplicateFold, DuplicatesPolicy}`;
  `rcp_ndcg.testing.io_conformance`; `rcp_ndcg.data.media.{media_extension, image_dimensions}` and
  `store_media(..., mime=)`; the `rcp_ndcg.readers`/`rcp_ndcg.writers` entry-point groups and their
  `rcp_ndcg.data.io.registry` accessors; `MtebTaskReader`; `HubReader` (the `hf` entry point);
  `rcp_ndcg.errors.WarningCode.CARD_UNCACHED`; `dataset_uri_revision` resolves `mteb:` URIs.
- Removed public names: `rcp_ndcg.data.io.hf.HfReader` (and its module), `rcp_ndcg.data.io.base.join_title`.
- CLI: `rcp-ndcg data convert`'s help text now describes the entry-point format tables (no flag change);
  `rcp-ndcg data formats` lists `mteb` too.
- Schemas/snapshots: `tests/contract/snapshots/{python_api,packaging}.json` and
  `schemas/{cli,eval-report}.v1.json` regenerated (never hand-merged; `--update-snapshots` on the merged tree
  produced no further diff).
- Manifest: `rcp-ndcg` declares the two entry-point groups; `[data]` drops `datasets`; `uv.lock` and
  `requirements-constraints.txt` regenerated.

## Files outside scope

The lane's assignment covers `data/dataset.py`, `data/io/**` (except `data/io/mteb.py`), `data/_rows.py`,
`data/validate.py`, `data/revisions.py`, the core record fields, `rcp_ndcg.testing`'s io conformance, the
reader/writer entry points, `tests/data/**` and the data docs. These files outside it were touched, each
minimally and for a named reason:

- `rcp-ndcg/src/rcp_ndcg/data/media.py` — `store_media(..., mime=)`, `media_extension`, `image_dimensions`
  (the Hub reader's raw/container media cells need them; one home for media formats).
- `rcp-ndcg/src/rcp_ndcg/storage/io.py` — transparent `.gz` in the shared JSONL reader (the BEIR gzip item;
  the format readers decide whether to allow a compressed file, the jsonl reader still refuses one).
- `rcp-ndcg/src/rcp_ndcg/errors.py` — `WarningCode.CARD_UNCACHED`; `EXTRA_FOR_MODULE["datasets"]` → `mteb`.
- `rcp-ndcg/src/rcp_ndcg/eval/mteb/__init__.py` — its private `_hub_file` import followed the plumbing to
  `data/io/hub.py` (one line).
- `rcp-ndcg/src/rcp_ndcg/cli/data.py` — `data convert`'s docstring (the format tables are now entry points).
- `rcp-ndcg/pyproject.toml`, `uv.lock`, `requirements-constraints.txt` — the entry-point groups and the
  `datasets` dependency leaving `[data]` (generated files regenerated, per the master).
- `tests/cli/test_eval.py` — its staged hub cache gained the card (the reader resolves through it now).
- `tests/docs/test_packaging.py` — the `datasets` gate row moved to `mteb`.

## For the next lanes

- **l10b (MTEB export/scoring)**: `Dataset.export_key` is the `(task, subset, split)` key; `Provenance` and
  the duplicates counts are on every loaded dataset. The writer's config names must be `{s-}corpus`,
  `{s-}queries`, `{s-}qrels` (not `default`) per the spec's section 5, and the reader resolves exactly those
  (plus the `default`/`qrels` fallback for v1-style repos), so a round trip through the writer is readable.
- **l10c (the join and the instructions)**: `Document.title` and `Dataset.task_instruction` are fields and
  nothing joins at read time; the generic defaults the owner confirmed (task instruction prefixed, per-query
  appended) belong to the formatting stages. Note that a title-only document now reads as `text=""` plus a
  title (Core17 has 580 of them), and `io_conformance` counts a title as content.
- **Anyone adding a format**: `rcp_ndcg.readers`/`writers` + `io_conformance`; the built-ins are the worked
  examples, and `tests/data/test_io_hub.py` shows how to stage a repository offline.
- The mteb reader's tests need the `[mteb]` extra and network; the scratch venv used here is disposable and
  outside the repository.
