# Lane `content-wire`: one lowering of Content to wire blocks, part order kept (pre/post-processing review A5, A7, A8)

**Status: DONE.**

Base: the `rfc-0001` tip after l10c, media-rules and qa-prep (`01f4b9be`). Before the report the current
`rfc-0001` (`afecce00`, 52 commits: harness-fix, ci-recipes, sz-qwen3) was merged with no conflicts; the
merge touched none of this lane's files and the full suite plus the network recipe tests were green on the
merged tree. Final gate on the merged head: **PASS**.

## Commits

| Hash | Subject |
|---|---|
| `1f29af4f` | A5/A8: the fit cuts and records per text part; the typed budget read, the chunk-mapping input id and the census accessor |
| `24df03c7` | A7: one Content-to-wire lowering, the judge's guards as hooks |
| `104bc8f0` | Docs, CHANGELOG and the contract snapshot for the A5/A7/A8 changes |
| `7d0ae52e` | A5 follow-up: the embeddinggemma-2 reference keeps every text part in its place |
| `fdadf252` | Round-1 verifier minors: refuse the normalisation+multipart-media combination, size a container only for a guard, strict pair parts, docs |
| `855426fb` | Merge branch `rfc-0001` into `lane/content-wire` (the merged `rfc-0001` tip is `afecce00`) |

## What changed

- **A5 — part order.** `Content.truncated` and the new `rcp_ndcg_core.content.split_text_across_parts` are the
  one walk that distributes a cut of the joined text over the parts; `RoleClient._with_text` places each
  part's own piece where it stands (media untouched) instead of joining every text part into the first slot.
  The fit's cut applies per part and the census records one row per part the cut shortened (each row's own
  original/kept counts, the request's totals repeated on every row; a part kept whole records nothing). The
  embed and pool `messages` routes and the rerank document body all pin it; the fixture media reference and
  the shipped embeddinggemma-2 reference keep each text segment in place too. A declared template
  normalisation beside a multi-part media content (whose normalised span cannot be distributed over the raw
  parts) is refused with a `ConfigError` before any fit runs, never silently hoisted.
- **A7 — one lowering.** `data/media.py:content_parts_payload` is the one lowering of a `Content` to wire
  blocks; the judge's `_blocks` delegates to it and passes its prepared-image check and inlined-container
  byte cap as `image_guard`/`video_guard` hooks. The lowering's declared mechanisms (an empty text part is
  dropped; a video part's frames win over its container; an already-inlined image is sent as it is) are in
  its docstring. A container's MIME is resolved by the same rule on every path, and its byte size is
  computed only when a guard asks for it (so an already-inlined `data:` container lowers on the served path).
- **A8 — minors.** `processing_records` groups a cut row under its input through the fit's `chunk_mapping`
  (an input id containing `#` is never mis-split; a row the mapping does not name stays its own id); the
  role-client base reads the declared budget fields typed (`_BudgetConfig`/`_DocumentCapConfig`) and refuses
  a config missing one instead of silently defaulting `on_overflow`/`aggregation`; the vendor path claims its
  one row per (corpus, limit) through the new public `TextTruncationCensus.claim_budget_row`; the
  `data/preprocess` docstring lists exactly the names it re-exports.

## Verification

Round 1 ran two fresh verifiers on DeepSeek-V4.1-flash (`:xhigh`), together: lens A correctness and lens B
regressions/hygiene, each told the other exists and not to duplicate it. Both returned **VERDICT: PASS**;
no blocker or major was found, so no round 2.

- **Lens A (correctness)** verified `split_text_across_parts`/`Content.truncated` differentially against the
  base algorithm (1482 structured cases + 20 000 random part sequences x every `max_chars`, zero diffs), the
  per-part census rows (counts, ids, request totals, chunks, pair, vendor, normalisation), the mapping, the
  typed budget read across all 30 shipped recipes, and the one lowering case by case (prepared/unprepared
  images, interleaved text, empty text, containers, frames-over-ref, over-cap). Five mutations in a scratch
  worktree went red: hoisting `_with_text` (7 `test_part_order` failures), `#`-re-splitting
  `processing_records`, ignoring `part_pairs` (2), restoring duck-typed budget defaults, and lowering
  `_blocks` without the guards (2). Findings: 4 minor.
- **Lens B (regressions/hygiene)** ran the full root suite, ruff, basedpyright, contract/docs, the test
  distribution and the network recipe tests; checked R30, one home per concept, the snapshot, the CHANGELOG,
  the docs and scope; five mutations went red (including the fixture media gate independently failing on a
  hoist). Findings: 6 minor.
- **What I did about them:** fixed 9 of 10 — the normalisation refusal, the guard-only container size, strict
  pair `parts`, and the docstring/CLI/harness/hint/comment/CHANGELOG wording. One is deliberately not fixed:
  `rcp_ndcg_core._records.Document.model_content` (`title="join"`) still collapses several text parts into
  the first slot; it is the MTEB title/body join from l10c (a different concept, declared in its docstring),
  no shipped reader builds a multi-part `Document`, and the file is outside this brief — recorded under
  **For the next lanes**.

Failing test first: the seven A5 order tests were red on the unfixed code (`assert ['text', 'image_url'] ==
['text', 'image_url', 'text']`); the A7/A8 tests were re-run against the pre-fix code by swapping the saved
files back (4 red: `TypeError: processing_records() got an unexpected keyword argument 'chunk_mapping'`,
`AttributeError: 'TextTruncationCensus' object has no attribute 'claim_budget_row'`, `DID NOT RAISE
ConfigError`, `DID NOT RAISE DataError`). The verifiers' mutation tables above are the independent red/green
evidence for the rest.

## Checks

Final tree `855426fb` (the merged head), the shared gate:

```
bin/gate lane/content-wire  ->  GATE: PASS
ruff-check exit=0; ruff-format exit=0 (583 files)
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 3650 passed, 102 skipped
contract-docs exit=0 304 passed, 55 skipped
mkdocs exit=0
test-pkg exit=0 902 passed, 222 skipped
recipes exit=0 (0 failures outside the rfc-0001 baseline)
vllm-pkg exit=0 40 passed; vllm-models exit=0 72 passed, 7 skipped
run_all exit=0: leaderboards 1022 checks / 987 match / 35 known deviations / 0 failed;
                human study 67/67; external LLM judges 82/82
public-names exit=0; clean exit=0
```

In the worktree with the network gate open: `RCP_NDCG_NETWORK_TESTS=1 pytest rcp-ndcg-test/tests/recipes -n 4`
-> 468 passed, 1 skipped. The gate also passed on the pre-merge head `7d0ae52e` (same steps).

## Open questions

- The `_with_text` normalisation refusal is a new `ConfigError` for a combination no shipped recipe hits
  (topk-embed-v1 declares `normalize: [strip]` with media, but its `max_images: 1` keeps its media set
  single-image). If a later recipe needs a normalising template beside interleaved media, the right fix is
  to distribute the normalised span through the same declared ops, not to drop the refusal.
- Per-part census rows are deliberately disabled for chunked inputs (the chunks are the units) and for
  normalising templates (the refusal above); a declared normalisation with a single-text-part media content
  keeps its one row.
- The settled rerank query is recorded both as the `<query>` settlement row and inside each pair's per-part
  query rows; the totals are identical, so `processing_records` is unaffected, but the census carries the
  same kind of duplicate it carried before.

## CHANGELOG entry

Under `## Unreleased`:

```markdown
### Public surface

- **The one Content-to-wire lowering and the per-part cut records (pre/post-processing review A5, A7, A8)**:
  the text-budget `fit` gains an optional `parts` (each input's text parts: one census row per part the cut
  shortened, the request's totals repeated on every row, a part kept whole recorded nothing),
  `processing_records` gains `chunk_mapping` (a chunk row groups under its input through the fit's own mapping,
  never a `#` re-split), and `TextTruncationCensus.claim_budget_row` is the public accessor for the vendor
  path's one-row-per-(corpus, limit) state. `rcp_ndcg.data.media.content_parts_payload` is the one Content
  lowering for every role (keyword-only `image_guard`/`video_guard` hooks carry the judge's prepared-image
  check and inlined-container cap), with `video_data_uri` and `VIDEO_CACHE_SIZE` in the same module;
  `rcp_ndcg_core.content` gains `split_text_across_parts` (the one distribution of a joined-text cut over the
  parts).

### Fixed

- **A served item's text parts keep their own places around its media (review A5)**: `[text A, image, text B]`
  was sent as `[A\nB, image]` -- every text part joined into the first slot, unrecorded. The fit's cut now
  applies to each part where it stands (the joined cut distributed over the parts) and is recorded per part;
  the embed and pool `messages` routes and the rerank document body all pin it, the media stage's fixture
  reference keeps each text segment in place too, and the shipped embeddinggemma-2 reference places the
  task prompt, the media and the body text the same way. A declared template normalisation beside a
  multi-part media content (whose normalised span cannot be distributed over the raw parts) is refused
  with a `ConfigError`, never silently hoisted.
- **The judge's wire and the served roles' wires lower Content through one function (review A7)**: the judge's
  `_blocks` and `rcp_ndcg.data.media.content_parts_payload` were two lowerings with different validation; the
  judge now delegates to the one lowering and adds its two guards as hooks, and the lowering's declared
  mechanisms (an empty text part is dropped; a video part's frames win over its container; an already-inlined
  image is sent as it is) are stated in its docstring. The served path now resolves a container's MIME instead
  of blindly sending `video/mp4`.
- **The text and interleaved minors of the review (A8)**: `processing_records` names a cut row's input through
  the fit's `chunk_mapping` (an input id containing `#` is never mis-split); the role-client base reads the
  declared budget fields typed and refuses a config missing one instead of silently defaulting
  `on_overflow`/`aggregation`; `data/preprocess`'s docstring lists exactly the names it re-exports; the vendor
  path claims its census row through `TextTruncationCensus.claim_budget_row`.
```

## Public surface changes

- `rcp_ndcg.data.fit` and `rcp_ndcg.data.preprocess.fit`: new keyword-only `parts`.
- `rcp_ndcg.data.preprocess.processing_records`: new keyword-only `chunk_mapping`.
- `rcp_ndcg.data.preprocess.TextTruncationCensus.claim_budget_row`: new method.
- `rcp_ndcg.data.media.content_parts_payload`: new keyword-only `image_guard`/`video_guard`; the module's
  `__all__` gains `video_data_uri` and `VIDEO_CACHE_SIZE`.
- `rcp_ndcg_core.content`: `__all__` gains `split_text_across_parts`.
- `tests/contract/snapshots/python_api.json` regenerated for the three tracked signature/member changes. No
  CLI, exit code or JSON schema change.

## Files outside scope

- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/embeddinggemma-2/reference.py` and
  `rcp-ndcg-test/tests/recipes/test_embeddinggemma_2.py`: the shipped reference's media placement had to
  follow the A5 fix (and its direct test); the gate's recipe step caught the stale placement.
- `rcp-ndcg-test/tests/test_media.py`: one docstring sentence about the placement.
- `rcp-ndcg-test/src/rcp_ndcg_test/equivalence/media.py`: one docstring sentence.

## For the next lanes

- `rcp_ndcg_core._records.Document.model_content` (title `join`) collapses several text parts into the first
  slot, unlike the served lowering; it is declared in its docstring and no shipped reader builds a multi-part
  `Document`, but a user-built multi-part `Document` with a title still reads hoisted. Give it the same
  distribution when a lane owns that file.
- No shipped recipe's media set carries an interleaved row (every recipe declares `max_images: 1`), so the
  part order is pinned by the product tests and the fixture recipe only; a recipe with `max_images >= 2`
  would exercise the media stage's placement against a real card.
- The census's per-part rows are new: a consumer that assumed one row per input for the `text_budget`
  mechanism (summing `original_tokens` over rows) now overcounts a multi-part content exactly as it already
  did for chunks.
