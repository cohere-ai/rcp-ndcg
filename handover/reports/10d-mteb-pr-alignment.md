# Lane l10d — published data exactly as mteb PR #5516 reads it (owner decision 40)

**Date:** 2026-10-09 · **Branch:** `lane/l10d` · **Base:** `rfc-0001` after l10b (`b67699c0`) · **Head:** `cf3604cd`
**Gate:** `bin/gate lane/l10d` → **PASS** (rev `cf3604cd`; merged `rfc-0001` @ `4fea88f4`, then `c5ccc851`, then
`75a341d9`)

## Status

DONE. Every brief item is implemented, tested (failing test first where behaviour changed) and gated on the
merged tree. Two verifier rounds ran: round 1, two fresh DeepSeek-V4.1-flash (xhigh) verifiers in parallel —
lens A (correctness) **FAIL** with one blocker and four minors, lens B (regressions/hygiene) **PASS** with six
minors; round 2, one fresh confirmation verifier (lenses A+B) **PASS** with one major and six minors. Every
blocker/major and every agreed minor is fixed and pinned; the two minors I did not act on are shown wrong or
inert below. `run_all` is unchanged (1022/987/35/0 · 67/67 · 82/82).

## Commits

| Hash | Subject |
|---|---|
| `b3630908` | The float-gain metric follows mteb PR #5516's own semantics (a query with no gains scores 0, every gain validated; the PR's function vendored as a test oracle) |
| `129ba58c` | The MTEB writer writes mteb's `Image`/`Video` columns (struct<bytes, path> + the parquet `huggingface` feature metadata; the round trip through mteb's dataloader and the PR's `load_float_gains`) |
| `b969ffa0` | The republish converter keeps the published task splits and refuses a mismatch (per-part split, task-definition validation, the card from the task metadata) |
| `39f55022` | Docs, CHANGELOG and the NOTICE for the mteb PR alignment (the vendored oracle attributed; `docs/data.md` also loses the committed merge-conflict markers the l10b merge left there) |
| `1a92f7ff` | The float-gain metric matches mteb PR #5516 bit-for-bit, nAUC keys included (core's `dcg` divides by `log2(rank+1)`; the full-dict comparison test) |
| `f748866d` | The MTEB writer shares a suite's corpus and refuses any video of frames (a one-part suite writes its part's config names) |
| `4c8091d1` | The republish converter writes every subset the repo and the task file share (the card's 48 ViDoRe v3 subsets; a symmetric subset check; a uniquely named validation view) |
| `db3d80b5` | Docs, CHANGELOG and NOTICE for the round-1 fixes |
| `427fa6c5` | The converter groups a shared corpus from the card and validates a relative `--out` (TREC-DL's two years share one corpus too) |
| `28e91e68` | The writer counts a shared corpus once and refuses a differing repeat |

Merges into the lane: `983296c4` (rfc-0001 @ `4fea88f4`, lane mrl-core), `dab714f8` (rfc-0001 @ `c5ccc851`,
scoring-fixes + sync-hardening + rf-engine), `cf3604cd` (rfc-0001 @ `75a341d9`, the decision-34 handover
wording).

## What changed (per brief item)

1. **Writer and converter reproduce the PR's layout.**
   - The writer already wrote the integer `score` plus the nullable float `gain`/`theta` columns, the
     `top_ranked` pools (exclusions folded out), the corpus/queries columns and a README whose `configs:`
     front matter `datasets.load_dataset` and mteb's `RetrievalDatasetLoader` read; a round-trip test loads the
     written `gain` column with the PR's own `load_float_gains` (vendored at the PR's head commit
     `595c8ecc…`) and the rest through mteb's loader, and compares to our `Dataset`.
   - `tools/republish_mteb.py` no longer re-lays every repository to `test`: each subset is written at the
     split its published task definition pins — NanoBEIR `train`, BRIGHT `standard`, ViDoRe v3 `test` — and
     the converter refuses a definition whose split does not match the data before writing anything.
   - The subset set comes from the published card's config names (`hub_subsets`), not from the paper's
     native-language `SUITES` view: all 48 ViDoRe v3 language subsets are written (the eight native-language
     ones are a scoring view), and a corpus shared by several subsets (ViDoRe v3's six languages of a domain,
     TREC-DL's two years) is written once, grouped from the card's `-corpus` entries. The task file's
     `eval_langs` subsets are checked against the card's in both directions, and every task definition
     matching a subset (the page-image task and its OCR view) has its split checked.
2. **`ndcg_float_at_k` is the PR's.** `rcp_ndcg.eval.mteb.ndcg_float_scores` equals the PR's
   `ndcg_float_scores` on a matrix of ties, tie blocks crossing `k`, unjudged documents, all-zero gains, a
   query whose gains are all null and empty rankings; the formula keeps its one home in
   `rcp_ndcg_core.metric` (`ndcg` with `ties="group_mean"`) and the eval module calls it. Core's `dcg` now
   divides by `log2(rank + 1)` (the PR's operation order), so the per-query values and the abstention nAUCs
   are identical, not merely equal to 5 decimals; the comparison test compares the full dict (NaN nAUCs
   included) against the vendored oracle. A query whose gains are all null now scores 0 and stays in the mean
   (the PR's rule) instead of raising.
3. **Media.** The writer's media refusal is lifted: a document's (or query's) `image`/`video` parts become
   mteb's `struct<bytes, path>` cells with the parquet's `huggingface` feature metadata, so
   `datasets.load_dataset` reads them as `datasets.Image`/`Video` and mteb's `create_dataloader` hands a model
   the decoded page image; `MediaRef` stays the internal representation and the bytes are resolved through
   the media resolver. One image and one video per row; a document with several images, or a video whose
   frames are extracted (with or without a container), is refused by name. `path` is null (declared: the
   internal ref is content-addressed; mteb reads the bytes). A round trip covers one image through mteb's
   dataloader and one video's feature and bytes (decoding a video needs torchcodec, mteb's own requirement).
4. **Task metadata.** The prompt rides in `TaskMetadata.prompt` (the published files already carry it; the
   product's `get_tasks` builds the task from them, and a test pins prompt and real split), and the
   converter's card renders from the published task metadata. The converter validates split and subset; the
   owner bumps the task file's `_REVISION`/`dataset.revision` to the pushed commit after uploading.
5. **l10b's open item closed.** The published `rcp_ndcg_tasks.py` files at their current revisions carry the
   PR's names (`*RCPReranking`), splits (`train`/`standard`/`test`), prompts and the PR's pinned data
   revisions; the network-gated test and the converter's live check verify them (nanobeir 13, bright 12,
   vidore 48 subsets; 8 groups share a corpus).

Decision 40 was already recorded in `handover/00-MASTER.md` (commit `937146ed`, "record owner decisions
34-41"); this lane implements it and needs no master edit.

## Verification

**Round 1** — two fresh DeepSeek-V4.1-flash (xhigh) verifiers in parallel, each waited for:

- *Lens A (correctness), VERDICT: FAIL.* Confirmed the oracle verbatim, the metric equal on 20 000 random
  matrices plus adversarial cases, the real pinned-revision round trips (nanobeir/bright/vidore) and the
  media encoding. Findings and what was done:
  - **A1 (blocker)** the converter wrote only 8 of ViDoRe v3's 48 language subsets. Fixed: the subset set
    comes from the card, the check is symmetric, all 48 are loaded and written, a shared corpus is written
    once; pinned by `test_the_converter_writes_every_subset_the_task_file_defines`,
    `test_a_shared_corpus_is_written_once_and_read_by_every_language` and the two-directional refusal tests.
  - **A2 (minor)** nAUC keys differed from the PR's by a 1-ulp per-query difference. Fixed: core's `dcg`
    divides (the PR's order), the full-dict comparison test includes the adversarial case; `run_all`
    unchanged.
  - **A3 (minor)** the writer writes `path: null` where the published cells carry the page file name. Fixed
    by declaring the deviation (module docstring, `_media_struct`, CHANGELOG): `MediaRef` is
    content-addressed and mteb reads the bytes.
  - **A4 (minor)** the converter's media validation was untested. Fixed:
    `test_the_converter_validates_media_bytes` corrupts the written image bytes and expects the refusal.
  - **A5 (minor)** a `VideoPart` with a container and frames silently lost the frames. Fixed: any extracted
    frames are refused by name (also B2's directory-ref case).
- *Lens B (regressions/hygiene), VERDICT: PASS* with six minors, all fixed: B1 dead `image`/`video`
  parameters dropped; B2 the frames refusal covers a directory ref; B3 the subset check is symmetric; B4 the
  oracle docstring/NOTICE say three adaptations (not two); B5 the card test asserts the fields, not a
  rendered prompt; B6 validation loads a uniquely named symlink view so a re-run cannot read a stale
  `datasets` build. Three mutations (media metadata, split check, missing-gains path) each turned the new
  tests red.

**Round 2** — one fresh confirmation verifier (lenses A+B), VERDICT: **PASS** with one major and six minors:

- **F1 (major)** TREC-DL's two subsets share one corpus but the grouping keyed only on `__`-named subsets,
  so it would have been written twice. Fixed: the groups come from the card's `-corpus` entries; the live
  check shows nanobeir 0 groups, bright 0, vidore 40 subsets in 8 groups, trecdl 1 subset in 1 group; pinned
  by `test_trec_dl_s_two_subsets_share_their_corpus`.
- **F2 (minor)** the B6 symlink view stored a relative link target, so a relative `--out` failed. Fixed:
  the view resolves its targets; pinned by `test_a_relative_out_directory_validates`.
- **F3 (minor)** `write_dataset(corpus_group=)` counted every part's rows and silently dropped a repeated
  group's differing corpus. Fixed: a group is counted once and a differing repeat is refused by name; pinned
  by `test_a_shared_corpus_is_written_once_and_counted_once` and
  `test_a_shared_corpus_that_differs_is_refused`.
- **F4 (minor)** the how-to page still said frames are refused only without a container. Fixed.
- **F5 (minor)** the `math.fsum` → `sum` half of A2 is unpinned. Not acted on: on Python 3.12 `sum` uses
  Neumaier compensated summation and is numerically indistinguishable from `fsum` for a 5-decimal mean (2
  million random vectors, including cancellation cases, found no rounded difference), so no value-based test
  can distinguish them; the operation-order half of A2 is pinned by the adversarial nAUC case. The code keeps
  the PR's `sum(values) / len(values)` expression verbatim.
- **F6 (minor)** the card-derived 48-subset list had no offline pin. Fixed:
  `test_repo_subsets_reads_the_card_not_the_paper_view` pins the delegation to the card parser.
- **F7 (minor)** a defensive branch for a hand-built `VideoPart` with neither ref nor frames. Kept and
  labelled: it guards `model_construct` (the validated model refuses that shape), so it cannot be dropped
  without losing the typed error for a hand-built record.

## Checks (last runs)

`bin/gate lane/l10d` on the merged tree (`cf3604cd`), every step exit 0:

```
rev lane/l10d = cf3604cd (slot 5)
ruff-check exit=0 All checks passed!
ruff-format exit=0 562 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 3432 passed, 98 skipped
contract-docs exit=0 295 passed, 52 skipped
mkdocs exit=0 Documentation built
test-pkg exit=0 570 passed, 225 skipped
recipes exit=0 no failure outside the baseline (34 baseline failures remain, 0 fixed)
vllm-pkg exit=0 9 passed
vllm-models exit=0 71 passed, 7 skipped
run_all exit=0 leaderboards 1022/987/35/0; human study 67/67; external judges 82/82
public-names exit=0 clean (2 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

mteb-gated tests in a scratch venv outside the repository (the gate's environment has no `[mteb]` extra):
`tests/eval/mteb tests/data` → 806 passed, 42 skipped; `RCP_NDCG_NETWORK_TESTS=1 tests/eval/mteb/test_tasks.py`
→ 8 passed (the published task names, splits, prompts and pinned revisions). The pinned-revision
verification against the published repositories (mteb's `RetrievalDatasetLoader` + the PR's
`load_float_gains`): nanobeir `NanoArguAnaRetrieval`/`train` 50 qrels/50 queries/3635 corpus/7479 gain
pairs; bright `aops`/`standard` 111/111/188002/12205; vidore `computer_science__english`/`test` 32250 qrels,
215 queries, 150-document pools, 32250 gain pairs.

Failing-test-first evidence: the metric alignment test failed on the missing-gains case
(`DataError: query 'q2' has no gains`) before the semantics change; the four media tests failed with the old
refusal (`document 'd1' carries media, which the MTEB text columns cannot express`) before the writer
change; the converter tests failed with the old two-argument `_card`/absent `_task_source` before the
converter change; the round-1/round-2 verifiers' own reproductions (the 8-of-48 subset gap, the TREC-DL
grouping, the relative-`--out` failure, the shared-corpus over-count, the 1-ulp nAUC case) are the red
evidence for the fix rounds.

## Open questions

- **Media `path`.** The writer writes `path: null`; the published `rcp-ndcg-vidore-v3` stores the original
  page file name there. mteb reads the bytes and ignores `path`; carrying the original name would need the
  Hub reader to record it on the `MediaRef` (a core field) and would put a repository-relative name in the
  export. Declared in the writer docstring, the CHANGELOG and here.
- **Video decoding.** mteb's dataloader decodes a `Video` cell with torchcodec (mteb's requirement, not a
  product dependency); the writer's video test checks the feature metadata and the bytes, and the image test
  goes through `create_dataloader`. If a video corpus ever ships, CI needs a torchcodec-enabled job for the
  decode.
- **The converter's task file is read at the repository's current revision**, while the data revision comes
  from `experiments/fetch_data.py`; the published files at `main` match the PR, and the owner bumps
  `_REVISION`/`dataset.revision` to the pushed commit after uploading. A later task-file edit before the
  push changes the check's input (it is the definition being checked).
- **`_corpus_entries` reads the product's private `_card_configs`** (the public `hub_subsets` names subsets
  but not their corpus entries). If another tool needs the same mapping, the product should grow a small
  public accessor; today the tool is its only caller.
- **`evaluate_abstention` raises `IndexError` on an empty `results` mapping**, in mteb and in the PR's
  function identically (a shared upstream edge, not a divergence); reachable through `task_specific_scores`
  only when a model scores no query at all.
- **The `sum` vs `fsum` expression** (F5): kept as the PR's `sum`; on Python 3.12 the two are numerically
  indistinguishable for the rounded mean, so no test pins the expression itself.

## CHANGELOG entry

Under `## Unreleased`, the lane added or extended exactly these bullets (full text in `CHANGELOG.md`):

- **`### Public surface`** — extended: *The MTEB dataset writer* (the `mteb` writer of `WRITERS`,
  `data convert --to mteb`) and *`tools/republish_mteb.py` re-lays the published rcp-ndcg datasets…* (all 48
  ViDoRe v3 language subsets, the card-derived splits, the shared corpus written once, the two-directional
  refusal); new: *The MTEB writer writes mteb's media columns* (`Image`/`Video` cells with the parquet
  `huggingface` metadata, one cell per row, frames refused, `path` null, `corpus_group=` writes and counts a
  shared corpus once and refuses a differing repeat).
- **`### Changed`** — *The float-gain metric matches mteb PR #5516 bit-for-bit, nAUC keys included* (core's
  `dcg` divides by `log2(rank + 1)`; the metric means and the paper's anchor numbers unchanged) and
  *`rcp_ndcg.eval.mteb.ndcg_float_scores` follows mteb PR #5516 on a query whose gains are all null* (scores
  0 and stays in the mean; every gain validated; a non-finite score still refused).
- **`### Fixed`** — *A one-part suite writes its subset's config names* (the single-dataset branch used the
  suite's `subset`).

## Public surface changes

- `rcp_ndcg_core.metric.dcg`: operation order changed (`gain / log2(rank + 1)`; same formula, the PR's
  order). No signature change; the anchor tests and `run_all` are unchanged.
- `rcp_ndcg.eval.mteb.ndcg_float_scores`: a scored query whose gains are all null now scores 0 and stays in
  the mean instead of raising; a non-finite model score is still refused.
- `rcp_ndcg.data.io.mteb.MtebWriter.write_dataset`: new keyword `corpus_group` (an internal module; no
  contract snapshot moves). `write_dataset` counts a shared group once and refuses a differing repeat.
- `tools/republish_mteb.py` (a tool): the full card-derived subset set, the published splits, the
  task-definition subset/split refusal, the card-derived corpus grouping and the unique validation view.
- No CLI command, flag, exit code or schema changed; `tests/contract/snapshots/` and `schemas/` were not
  regenerated (the contract tests pass without `--update-snapshots`).

## Files outside scope

- `docs/data.md` — the l10b merge had committed conflict markers in the data-conversion paragraph (the only
  such file in the tree); resolved to the union of both sides and the `--to mteb` sentence now names the
  media columns. Listed because the file is outside this brief's MTEB pages.
- `rcp-ndcg/NOTICE`, `rcp-ndcg-core/NOTICE`, `rcp-ndcg-test/NOTICE`, `rcp-ndcg-vllm/NOTICE` — the one merged
  NOTICE is byte-identical in all five copies; the vendored mteb oracle is attributed in the root file and
  copied.

## For the next lanes

- **l10c** (in flight; `data/io/hub.py`, `data/dataset.py`, `retrieval/_api.py`): this lane's only change in
  its files is a read-only import of `_card_configs` from `tools/republish_mteb.py` (no edit to `hub.py`),
  and `data/io/mteb.py`'s corpus sharing does not touch the title/instruction join.
- **The owner's republish**: `python tools/republish_mteb.py --out <dir>` now writes the full subset set with
  the published splits and one corpus per shared group; the vidore run reads the page images once per domain.
  After pushing, bump each task file's `_REVISION`/`dataset.revision` to the new commit.
- **The product's public surface**: `hub_subsets` names a repository's subsets but not their corpus entries;
  if a second caller needs the shared-corpus mapping, promote `_corpus_entries`'s read to a public accessor.
