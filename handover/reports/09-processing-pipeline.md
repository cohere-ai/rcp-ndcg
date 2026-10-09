# Report 09: one ordered processing pipeline (decision 23)

**Status:** DONE. All four workstream targets landed plus the topk image skip-rule fix; the full gate passes on
the merged head; `run_all` numbers unchanged (1022/987/35/0, 67/67, 82/82); two verifier rounds run (round 1:
FAIL with two blockers — fixed test-first; round 2: PASS, one fresh confirmation verifier, both lenses).

Base: `rfc-0001` tip `85aa44c5` (after the layout move). **Merged `rfc-0001` three times**: `511698ef` (before the
first gate), `a3e921ec` (lane l08-cat; handover files only, no product code — re-gated, PASS) — the final gate ran
on the merge `94087d52`.

## Commits

| hash | subject |
|---|---|
| `9c802391` | Split `data/preprocess.py` into one module per concept: census file I/O → `storage/census`, the judge text policy and chunking → `data/text_policy`, the cut record and census → `data/census`, the client fit → `data/text_budget`; `data/preprocess` stays the public aggregation facade |
| `2ec3f0cf` | The postprocess home: `data/postprocess` holds L2 normalisation, the chunk-score aggregation, the Matryoshka cut and the late-interaction skip rule |
| `f447fa18` | The one declared stage pipeline in RoleClient (`STAGES`, the base runners, role hooks; embed and pool run the shared runner, the rerank's pair fit composes the same order) + a test per stage-order invariant |
| `327a6983` | The topk image skip-rule fix: media documents are sent under `document_skip_token_ids`, the media vectors kept whole, the deviation recorded as `skip_unapplied`; a pool media item's text part carries the fitted content span (one frame on every route) |
| `76cff696` | Docs: the AGENTS one-home table, the preprocessing/text-budgets/inference/late-interaction pages |
| `345f5665` | The pipeline's media stage always sends the fitted contents (the re-entry only decides omissions; the rcp-ndcg-test media stage caught the regression) |
| `e5482da7` | Merge `rfc-0001` (`511698ef`) |
| `160e40ae` | The topk recorder test pins the sent behaviour (round-1 finding F2) |
| `f450ad7c` | Round-1 findings F1+F3: the empty re-entry works in the caller's coordinates; a mixed text/media batch under skip ids is refused upfront; `skip_keep_mask` loses the caller-less parameter and gains unit tests |
| `0754d2ff` | Round-1 findings F4–F7: the shipped recipe notes (topk, pplx-late) describe the shipped rule; the split's stragglers |
| `d8d22c77` | The CHANGELOG declares the facade's three exported-but-never-listed names |
| `94087d52` | Merge `rfc-0001` (`a3e921ec`) |

## What changed

Per the workstream prompt (`handover/09-processing-pipeline.md`):

1. **One declared stage order** — `rcp_ndcg.inference.clients._base.STAGES = (normalise, empty, media, render,
   budget, lower)`, with the runners as the stages' one homes (`_prepare_rows`, `_stage_normalise`,
   `_stage_media`, `_stage_budget`, `_stage_lower`, the shared `_apply_empty_documents`); embed and pool run the
   shared runner, the rerank's `_fit_pair` composes the same stage methods in the same order (its normalise is a
   `_stage_normalise` override; the instruction fold stays), and its docstring maps the stages. Nothing re-orders
   them: the base owns the composition. One deliberate reading, documented in the code and below (Open
   questions): the empty policy runs on the content AS GIVEN (before media is fetched — nothing is fetched for
   an omitted empty document) and again on the media fit's all-dropped outputs (a document whose every media
   item was dropped is empty — the pinned lane-H behaviour), so the policy always decides on the content as it
   will be sent, before any frame. Tests: `tests/inference/test_pipeline_stages.py` (the order itself via
   recording subclasses for all three roles, empty before the frame, the media-dropped re-entry in caller
   coordinates, one frame per route, the record for every change and only for a change).
2. **The record as the one output** — the pipeline emits `client.processing` (per changed row, with mechanisms,
   uncut/kept request totals and the budget) once per preparation; the census rows the fit wrote feed it, and the
   consumers (the equivalence harness's per-row gating, the recorder) read it and never recompute.
3. **The postprocess home** — `rcp_ndcg.data.postprocess`: `l2_normalize` (moved from `inference.types`,
   re-exported under the same name), `max_pool_scores_by_document`, `max_pool_rubric_window_by_document`,
   `document_id_for_chunk`/`document_ids_from_chunks` (moved), and the new `mrl_cut` and `skip_keep_mask` (the
   pool client's MRL cut and document skips read them; the rerank's chunk pooling does too).
4. **The split** — `data/preprocess.py` (1,934 lines) → `data/census.py` (CutCause, TextCutRecord,
   TextTruncationCensus), `data/text_policy.py` (TextPolicy, ChunkPolicy, apply_text_policy, token_prefix,
   chunking, Preprocessing), `data/text_budget.py` (TextBudget, fit, FitResult, ProcessingRecord,
   processing_records), `rcp_ndcg/storage/census.py` (drop_torn_last_line, census_sink_lock, append_census_rows,
   read_census_rows; exported from `rcp_ndcg.storage`). `data/preprocess.py` remains the re-export facade — every
   documented import path works; the contract snapshot records the moved `defined_in` homes. AGENTS.md's one-home
   table gained the rows.
5. **The topk image skip-rule fix** (MASTER section 9 open item) — `document_skip_token_ids` no longer refuses a
   media document: the skip rule at image positions keeps the media vectors (a media request's positions are the
   engine's chat-template render, which the client cannot tokenise), the deviation is on the row's
   ProcessingRecord as the new mechanism `skip_unapplied`, and a batch mixing text-only and media documents is
   refused upfront (one media item routes the batch through the messages wire, where the text ids cannot align).
   A pool media item's text part now carries the fitted content span — one frame on every route (the engine's
   chat template frames a media item once, exactly like the embed role's messages route). The recipe's media
   stage is unblocked (its pairs manifest no longer records the media check as failed), and the shipped
   network-gated recorder test pins the sent behaviour.

**Failing-test-first evidence** (red before green, both fixes):
- topk fix: `uv run --no-sync pytest "tests/inference/test_pool_client.py::TestDocumentSkipIds" -q` →
  `2 failed` (`test_media_documents_are_sent_and_their_positions_are_never_skipped`,
  `test_text_documents_in_the_same_batch_are_still_skipped`) with
  `rcp-ndcg/src/rcp_ndcg/inference/clients/pool.py:363: CapabilityError` on the unfixed code.
- the F1 re-entry coordinates fix:
  `tests/inference/test_pipeline_stages.py::TestTheDeclaredOrder::test_the_reentry_keeps_the_caller_coordinates`
  failed first (`prepared.positions == (1,)` against `omitted == (0, 2)`; the round-1 repro
  `repro_pool_misassign.py` misassigned vectors end to end), then passed after the fix.

## Verification

**Round 1** (two independent verifiers, GLM-5.3-flash, fresh context, in parallel):
- Lens A (correctness): **FAIL** — confirmed the four brief targets implemented and the split faithful (AST of
  every moved definition identical), but found **F1 (blocker)**: the empty re-entry mixed kept-relative and
  original indices → silent vector misassignment end to end (reproduced: a batch `[empty, kept, media-dropped]`
  shipped the sent row's vectors to the omitted row); **F2 (blocker)**: the network-gated topk recorder test
  still pinned the removed refusal (my edit had been lost by a later baseline `git checkout` and never committed);
  minors F3 (mixed batches degrade from upfront refusal to mid-flight failure on a real engine), F4 (stale
  recipe notes), F5 (`skip_keep_mask`'s `media_positions` had no caller and no test).
- Lens B (regressions/hygiene): **FAIL** — the same F1 (blocker) and the same recipe-notes finding (major);
  confirmed byte-identical moves for 30 symbols, one-home-per-concept holds, R30 clean, no scope creep, no AI
  attribution, mutation tests of the new guards red as expected.
- Fixes: all agreed findings fixed — F1 with the mixed-batch invariant test first (red), F2 by committing the
  recorder test (the round-1 range had accidentally not contained it — the file was restored by a baseline
  checkout; lesson: commit immediately), F3–F7 as proposed.

**Round 2** (one fresh confirmation verifier, lenses A+B): **PASS** — F1's fix verified for the general case
(120-check adversarial matrix: five batch shapes × both roles × batch splitting, including a mixed keep/drop
media batch where the kept image's media tokens must remap through `kept_positions`); all three new tests proven
non-vacuous by mutation in separate worktrees; the topk recorder test passes with network (21 passed); F3's
refusal matrix 12/12; all gates re-run green; numerics unchanged. Residuals (none blocking): `data.postprocess`
is not snapshot-pinned (not in `PUBLIC_MODULES` — deliberate); the mixed-batch refusal is batch-level (a mixed
corpus needs a batch_size that keeps batches pure — documented in the recipe note and the error hint); the
harness's per-row handling of `skip_unapplied` stays exercised by the GPU corpora.

## Checks

Last commands and results (on the final merged head `94087d52`, gate `gates/` run of `bin/gate lane/w09`):

- `bin/gate lane/w09` → **GATE: PASS**: ruff-check/format clean; basedpyright 0 errors; `heavy uv run --no-sync
  pytest tests/ -q -n 8` → 3203 passed, 82 skipped; contract+docs → 270 passed, 52 skipped; mkdocs --strict ok;
  rcp-ndcg-test → 570 passed, 225 skipped; the gate's new recipes step → "no failure outside the baseline";
  vllm-pkg + vllm-models green; run_all → 1022 checks, 987 match, 35 known deviations, 0 failed; human study
  67/67; external judges 82/82; clean tree.
- `RCP_NDCG_NETWORK_TESTS=1` recipe files, one at a time under `timeout 900`: topk 21 passed (the updated
  recorder test); jina-reranker-v3 7 passed; zembed 8 passed, 1 skipped; qwen3-embedding 9 passed;
  zerank-1/zerank-1-small 10/9 passed; ctxl-1b 12 passed; jina-v5 13 passed. Pre-existing reds on the `rfc-0001`
  tip **before** this lane (verified by running the same files on `85aa44c5` with the lane's own code checked
  out): pplx-late (14 — a stale recipes-root path from the layout move), qwen3-vl-embedding-2b (2), octen (1),
  qwen3-reranker-0.6b (1), -4b (1), -8b (1), zerank-2 (13, same stale path). They are the gate's "34 baseline
  failures" now; not touched (rfam/harness-fix territory), listed under Open questions.
- Conformance: `uv run --no-sync pytest rcp-ndcg-test/tests/conformance -q` → 63 passed, 1 skipped.
- Failing-test-first evidence shown above for both fixes.
- The failing-test-first runs for the topk fix and the F1 fix are quoted in What changed; the verifiers'
  reproduction scripts live in a scratch directory outside the repository.

## Open questions

1. **The stage order's empty/media position.** The prompt lists "content normalisation -> empty handling ->
   media preparation -> ..."; the pinned lane-H test (`test_a_document_whose_every_media_item_was_dropped_is_empty`)
   requires the empty policy to fire on the media fit's all-dropped contents. The declared order keeps the
   prompt's sequence (empty before media on the as-given contents, where an empty document has no media to
   prepare) and defines the media stage's re-entry into the same empty policy for its all-dropped outputs — the
   invariant (`empty_doc` decided before the frame, never a framed non-empty turn) holds either way. If the
   owner reads the prompt's order as the runner's literal call order, say so and I will re-land it.
2. **The pool media item's text part now carries the fitted content span** (not the rendered template). For a
   template-declared pool recipe with a text+media document this changes the served prompt (the declared head no
   longer rides inside the message text; the engine's chat template frames the content once). No shipped pool
   recipe pairs a text+media document today (topk's media documents are image-only), so no corpus or reference
   diverges; flagging it because it is a real behaviour change beyond the skip rule, pinned by
   `test_one_frame_per_route`'s messages-route tests.
3. **Scoring topk image documents** stays a declared approximation: the client keeps every returned vector (the
   engine's frame text included — a superset of the reference's keep-only-image-patches rule) until a corpus
   observes the kept positions. The recipe note states it; the wave can now SEND image documents (the media
   stage verifies geometry and tokens; the engine check needs the node).
4. **Pre-existing recipe-test reds on the tip** (zerank-2, pplx-late, octen, qwen3-reranker-0.6b/4b/8b,
   qwen3-vl-embedding-2b — 34 baseline failures in the gate's new recipes step): stale recipe-root paths and a
   moved `doc_prompt` from the layout move, plus media-request-set drift. Not this lane's files; the gate's
   baseline records them.
5. **The pair census's `kept_tokens`** still counts query+document concatenated without a separator
   (pre-existing, MASTER section 9) — untouched here; the record's request totals are the per-row truth.

## Docs updated

- `AGENTS.md` — the one-home table: `text_policy`, `census`, `text_budget` + `storage.census`, `postprocess`,
  the STAGES pipeline rows.
- `docs/concepts/preprocessing.md`, `docs/concepts/text-budgets.md` — a "where it lives" note (the new homes;
  the snippets import from the facade, still runnable).
- `docs/concepts/inference.md` — the pipeline (`STAGES`), the per-row record, the census sentence.
- `docs/concepts/late-interaction.md` — the skip bullet states the image-position rule and `skip_unapplied`.
- Grep commands run: `git grep -n -i "data.preprocess"` and `git grep -n -i "document_skip_token_ids"`,
  `git grep -n -i "l2_normalize"`, `git grep -n "max_pool"`, `git grep -n
  "census_sink_lock|drop_torn_last_line|append_census_rows|read_census_rows"` over docs/, skills/, examples/,
  README.md, REPRODUCIBILITY.md, experiments docstrings, docstrings and CLI help texts; every hit updated or
  verified still-true (facade references resolve). `mkdocs build --strict` clean.
- `CHANGELOG.md` — the workstream-09 entry under `## Unreleased` (quoted below).

## CHANGELOG entry

**### Public surface**

- **One processing pipeline, one postprocess home (workstream 09, decision 23)**: the split of
  `rcp_ndcg.data.preprocess` and the declared stage order of the role clients. Every public name keeps its import
  path (`rcp_ndcg.data.preprocess` is the aggregation facade), and the contract snapshot records the moved homes:
  the judge text policy, the chunk geometry and `Preprocessing` live in `rcp_ndcg.data.text_policy`; the cut
  record (`CutCause`, `TextCutRecord`) and the census (`TextTruncationCensus`) in `rcp_ndcg.data.census`; the
  served roles' `TextBudget` and `fit` in `rcp_ndcg.data.text_budget`; the census files' record I/O
  (`drop_torn_last_line`, `census_sink_lock`, `append_census_rows`, `read_census_rows`) in
  `rcp_ndcg.storage.census` (exported from `rcp_ndcg.storage`); and the postprocess of model output
  (`l2_normalize`, `max_pool_scores_by_document`, `max_pool_rubric_window_by_document`, the new `mrl_cut` and
  `skip_keep_mask`) in `rcp_ndcg.data.postprocess` (`l2_normalize` re-exported from `rcp_ndcg.inference.types` as
  before). `rcp_ndcg.inference.clients._base.STAGES` declares the one preparation pipeline every role composes
  (normalise -> empty -> media -> render -> budget -> lower), and the per-row `ProcessingRecord` is its one output.
  The facade's `__all__` grows by three names the old module carried at module level but did not export:
  `needs_tokenizer`, `require_tokenizer` and `census_sink_lock`.
- **`skip_unapplied`** joins the `ProcessingRecord` change mechanisms (`CHANGE_MECHANISMS`): a pooled document's
  declared `document_skip_token_ids` was not applied to a media item -- the image positions are exempt, the
  client keeps every returned vector, and the deviation is on the row's record, never silently unskipped.

**### Fixed**

- **topk-embed-v1-small can send images** (the MASTER open item, workstream 09): the pooling client refused every
  media document whenever `document_skip_token_ids` was declared, so the recipe's media stage failed on the node.
  The skip rule now has a rule at image positions (see `skip_unapplied` above), the media document rides the
  messages route, and its text part carries the fitted content span -- one frame on every route (the engine's
  chat template frames a media item once, exactly like the embed role's `messages` route).

## Public surface changes

- Module homes move (`defined_in` in `tests/contract/snapshots/python_api.json`): `data.preprocess`'s members now
  define in `data.text_policy`, `data.census`, `data.text_budget`; the census file I/O in `storage.census`
  (also exported from `rcp_ndcg.storage`); `l2_normalize` defines in `data.postprocess` (still exported from
  `rcp_ndcg.inference.types` and `rcp_ndcg.retrieval`).
- New public names: `rcp_ndcg.data.postprocess.{mrl_cut, skip_keep_mask}`, `rcp_ndcg.data.preprocess.{needs_tokenizer,
  require_tokenizer, census_sink_lock}`, `rcp_ndcg.inference.clients._base.STAGES`.
- `CHANGE_MECHANISMS`/`ChangeMechanism` gain `"skip_unapplied"` (last).
- Schemas: `document_skip_token_ids`'s description in `schemas/index.v1.json` and `schemas/run-config.v1.json`
  (the pooling configs) describes the image-position rule.
- No CLI command, flag or exit code changed.

## Files outside scope

- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/topk-embed-v1-small/recipe.yaml` and
  `.../pplx-embed-v2-late-0.6b/recipe.yaml` — the shipped contract notes described the removed refusal as current
  behaviour and the topk wave instruction directed the operator away from what this fix enables (round-1 finding
  F4/F2's artifact); changed minimally (notes only — no serve/client/reference field moved). rfam owns recipe
  files: re-apply or resolve on the family conversion if these notes conflict.
- `rcp-ndcg-test/tests/recipes/test_topk_embed_v1_small.py` — the network-gated recorder test pinned the removed
  refusal verbatim (round-1 finding F2); rewritten to pin the sent behaviour (probe ok, HTTP 200). It is a recipe
  test (rfam's tree), but leaving it red would fail CI's gated job.

## For the next lanes

- **rfam**: the topk recipe's notes are updated to the shipped rule; the recipe's `known_deviations` may want a
  declared media-superset entry when the wave scores image documents (the recipe's `known_deviations` vocabulary
  has no such value today — the note carries it for now). The pplx-late recipe's stale recipes-root path in its
  test file (`tests.conftest`-resolved `rcp_ndcg_vllm/recipes`) is one of the 34 baseline failures.
- **harness-fix**: the per-row handling of the new `skip_unapplied` mechanism in the equivalence harness has no
  in-repo test (the harness gates `record.changed` generically, which is correct); the GPU corpora will exercise
  it.
- **10 (data I/O)**: `data/text_policy.py` is the read-time policy home; the MTEB work touches `data/` next to it.
- The judge role's media path composes the same base stages (`_prepare_request`, `check_engine_media`); its
  per-window budget fit lives in `judging.judging` and was intentionally not folded into `_prepare_rows` (the
  judge has no per-request budget to fit).
- `data.postprocess` is internal-but-public-shaped (not in `PUBLIC_MODULES`); if the docs start presenting it as
  API, add it to the surface list so its signatures get pinned.
