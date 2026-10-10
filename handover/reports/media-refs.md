# Lane `media-refs`: the media families' references compute their media outputs, so the media gate bites

**Status:** DONE. Branch `lane/media-refs`; final head `f73d7806` (the second merge of `rfc-0001`, at
`77710eeb`: the query-block-width test timeout fix; the first merge is `93cd10d0` at `7f3b94c1`:
runner-backends, the report scrub, the wave-grouping and Kubernetes test fixes). The gate on the merged tree
is **PASS** at `f73d7806`: ruff-check/ruff-format/basedpyright 0, root suite 3972 passed/102 skipped,
contract+docs 302 passed/55 skipped, mkdocs strict, test-pkg 1086 passed/227 skipped, recipes 0 baseline
failures, vllm-pkg 49 passed, vllm-models 92 passed/7 skipped, run_all leaderboards 1022 checks/987 match/35
known deviations/0 failed + human study 67/67 + external judges 82/82, public-names clean, checkout clean.
The first gate attempt on `93cd10d0` failed at its second step (`ruff format --check .`: one file,
`rcp-ndcg-test/tests/test_media.py`); `1fdb3f19` reformats that one line (no semantic change) and the gate
then passed. Nothing is pushed; the branch is not merged into `rfc-0001`.

## Commits

| Commit | Subject |
|---|---|
| `9115a78f` | qwen3-vl-embedding: the reference computes its media rows |
| `4c394c96` | qwen3-vl-reranker: the score mode reads the harness's media field |
| `fc99d2c5` | topk-embed-v1: the reference embeds image documents with the keep-mask |
| `e61a1a8f` | pplx-embed-v2-late: the reference embeds image documents |
| `797671ac` | embeddinggemma-2: the reference embeds image and video sides |
| `ad913bb5` | The harness gates a media document's kept token vectors per token |
| `1b6c9181` | Docs and CHANGELOG: the media families' references compute their media outputs |
| `93cd10d0` | Merge `rfc-0001` (`7f3b94c1`: runner-backends, the report scrub, the wave-grouping and Kubernetes test fixes) into `lane/media-refs` |
| `1fdb3f19` | ruff format: the ragged media reference helper's document_vectors line |
| `f73d7806` | Merge `rfc-0001` (`77710eeb`: the query-block-width test timeout fix) into `lane/media-refs` |

## What changed (per brief item)

**1. Each media family's reference reads the harness's `media` rows and computes the card's own outputs.**

- **qwen3-vl-embedding** (`9115a78f`): the embed mode decodes the `media` field -- an inline image to a
  loaded PIL image, a container through the card's own loader at the recipe's declared fps (2) capped at the
  engine's realised frame count -- into the card's `format_model_input`/`Qwen3VLEmbedder.process` input keys
  (media parts before the text, the card's order) and computes the card's own vectors for media rows exactly
  as for text rows. The interleaved forms the card's order cannot express (a text part before/between media,
  several images/videos, an image before its video) are refused loudly, never silently reordered.
- **qwen3-vl-reranker** (`4c394c96`): the score mode builds the card's own messages from each side's `media`
  entries (an inline image decoded to a PIL image); the retired `query_image`/`documents_images` columns (and
  the video siblings) are refused loudly, never scored as text.
- **topk-embed-v1** (`fc99d2c5`): the embed mode reads an image-only document as the loaded PIL image, which
  the wrapper's own `_image_row` renders and keep-masks (only the image-patch positions, the declared
  allowlist the plugin applies engine-side); a row's text and image documents ride separate encode calls, as
  the wrapper requires.
- **pplx-embed-v2-late** (`e61a1a8f`): the embed mode reads an image-only document as the loaded PIL image,
  which the checkpoint's own chat template renders with the trained `[D]` head and whose `MultiVectorMask`
  drops exactly the declared skip ids; text and image documents of one row ride separate encode calls, as the
  card's usage requires.
- **embeddinggemma-2** (`797671ac`): the embed mode reads the `media` field -- a media side becomes a
  one-user-turn conversation with the parts in order (the card injects the declared prompt as its system
  message), an image a loaded PIL image, a container a scratch file the checkpoint's processor samples at the
  declared video pin.

**2. `media_approximation` is removed from every family whose reference now computes its media rows.**

All five families (qwen3-vl-embedding, qwen3-vl-reranker, topk-embed-v1, pplx-embed-v2-late,
embeddinggemma-2) drop `reference.known_deviations: [media_approximation]`, so stage 2's media rows gate by
the same gates as the text rows. None of the five keeps the declaration: each card's own pipeline can run
the media inputs the harness's rows carry (the refused interleaved forms fail loudly instead of being
reported non-gating). The declaration and its non-gating path stay in the recipe schema and stay covered by
the fixture `fixture-vl-embed` (which keeps the declaration) and by
`test_stage2_reports_a_declared_media_approximation_non_gating`, for a card whose own pipeline genuinely
cannot run an input. The family notes and the nine affected goldens move with the declarations.

**3. The harness plumbing, the CPU tests and the generated files.**

- `ad913bb5` gates a media document's kept token vectors per token: a CPU test with a ragged fake client and
  a ragged fake reference requires the engine's kept count to equal the reference's and every kept position
  to gate (a dropped kept position is a count failure, never a silent skip); the test's own `(3, 4)` and
  `(4, 1)` cases must fail the stage.
- New CPU tests, no model weights: `test_the_reference_reads_media_rows_into_the_cards_own_inputs` (the
  qwen3-vl-embedding and embeddinggemma-2 row-reading contracts), `test_the_reference_media_embed_reads_
  documents_and_keeps_their_positions` (topk-embed-v1 and pplx-embed-v2-late, keep-mask/skip-id positions),
  `test_the_score_mode_reads_the_harness_media_field_and_refuses_the_retired_columns` (qwen3-vl-reranker),
  and the two stage-2 tests in `rcp-ndcg-test/tests/test_media.py` (the default gates a media row; a
  declaration reports it non-gating).
- The nine goldens (five families' variants) were regenerated the documented way,
  `pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py --update-goldens`; the diff is the recipe notes
  and `reference.known_deviations` only. `DELTAS.json` is unchanged.
- Docs and CHANGELOG (`1b6c9181`), below.

## Verification

Commands run on the final merged tree `f73d7806` (all offline, `uv run --no-sync`):

| Command | Result |
|---|---|
| `pytest tests -n 8 -q -p no:cacheprovider -o faulthandler_timeout=120` (via the `heavy` slot wrapper) | 3972 passed, 102 skipped |
| `pytest rcp-ndcg-test/tests -q -n 4 -p no:cacheprovider -o faulthandler_timeout=120` (via `heavy`) | 1086 passed, 227 skipped |
| `ruff check . -q` | pass |
| `ruff format --check .` | 610 files already formatted |
| `basedpyright` | 0 errors, 0 warnings, 0 notes |
| `bin/gate lane/media-refs` | **GATE: PASS** at `f73d7806` (slot 3) |

The failing gate step on the previous head, for the record: `ruff-format exit=1 - 1 file would be
reformatted, 609 files already formatted`, the file being `rcp-ndcg-test/tests/test_media.py` (one long
`document_vectors` comprehension); every later step of that run passed. `1fdb3f19` is that reformat and
nothing else.

## Checks

Gate `f73d7806` summary, verbatim:

```
rev lane/media-refs = f73d7806 (slot 3)
ruff-check exit=0 All checks passed!
ruff-format exit=0 610 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 3972 passed, 102 skipped in 54.14s
contract-docs exit=0 302 passed, 55 skipped in 40.66s
mkdocs exit=0 INFO    -  Documentation built in 1.43 seconds
test-pkg exit=0 ================ 1086 passed, 227 skipped in 708.70s (0:11:48) =================
recipes exit=0 recipes: no failure outside the baseline (0 baseline failures remain, 0 fixed; pytest exit 0)
vllm-pkg exit=0 49 passed in 3.69s
vllm-models exit=0 92 passed, 7 skipped in 56.94s
run_all exit=0   external_judges  ok
leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
human study: 67 checks, 67 match, 0 known deviations, 0 failed
external LLM judges: 82 checks, 82 match, 0 known deviations, 0 failed
public-names exit=0 public-names: clean (0 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

## Open questions

- **The media rows' GPU runs are E2's.** This lane is CPU-gated only (no model weights, no GPU): it proves
  the row reading, the keep-rule plumbing and the stage-2 gate on fakes. The real checkpoints' media
  outputs (and the engine's media count) must still be exercised by the E2 wave against a real engine.
- **qwen3-vl-embedding's per-clip pixel budget is not reconciled** (declared in the family's notes, open for
  a video wave): the engine and the client's container count use the checkpoint's video size
  (4096..25165824 px per clip), while the card script caps a clip at total_pixels 7864320 (786432 px per
  frame). The media stage gates a container's token count against the engine and the reference follows the
  card's own loader at the declared pin; the pixel-budget reconciliation itself is left to the video wave.

## CHANGELOG entry

Under `## Unreleased`, `### Fixed`:

> - **The five media families' references compute their media outputs, so stage 2's media rows gate**
>   (owner decision 35): qwen3-vl-embedding, qwen3-vl-reranker, topk-embed-v1, pplx-embed-v2-late and
>   embeddinggemma-2 read the harness's `media` field in their embed/score modes -- an inline image as a
>   loaded PIL image, a container through the card's own loader at the recipe's declared video pin, the
>   interleaved parts in order -- and compute the card's own vectors (per token, with the card's declared
>   keep-rule) or rerank scores for them, exactly as the model card's own pipeline does; qwen3-vl-reranker's
>   score mode now reads `media` instead of the retired `query_image`/`documents_images` columns, which are
>   refused loudly.  Stage 2 therefore compares the media rows by the same gates as the text rows, and the
>   five families drop `reference.known_deviations: [media_approximation]`; the media stage remains the
>   input gate (placement, geometry, tokens, the engine's count).

The existing "Stage 2 compares the media rows" bullet (the same section) was updated in the same commit to
point at this one instead of saying the five families declare the approximation until the E2 wave.

## Public surface changes

None: no `__all__`, CLI command or flag, exit code or exported schema changed. The recipe schema's
`reference.known_deviations` already accepted `media_approximation` (it lands with the earlier lane); this
lane removes the five families' use of it and regenerates their goldens and recipe notes only.

## Files outside scope

None. The diff touches the lane's assigned files (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/` for the five
families), their CPU tests and goldens under `rcp-ndcg-test/tests/`, `docs/how-to/` and `CHANGELOG.md`.

## Docs updated

- `docs/how-to/add-a-model.md` -- the media stage/stage 2 paragraph now says the reference reads the
  `media` field and computes the card's own vectors/scores, so the same gates apply; the declaration stays
  for a card whose own pipeline cannot run an input.
- `docs/how-to/validate-a-recipe.md` -- the T2 bullet and the equivalence-policy paragraph: a media
  recipe's image/video rows gate here too, with the five families' media paths and the declaration's
  residual meaning.
- `CHANGELOG.md` -- the new `### Fixed` bullet above plus the updated existing bullet.

Grep commands run (tracked files): `git grep -n "media_approximation" -- docs CHANGELOG.md rcp-ndcg-vllm/src
rcp-ndcg-test/tests`; `git grep -n "query_image\|documents_images" -- docs rcp-ndcg-vllm/src
rcp-ndcg-test/src`; `git grep -n "E2 wave" -- docs rcp-ndcg-vllm/src rcp-ndcg-test`. Every hit is either
the retained declaration mechanism (schema, fixture, its test, docs) or the retired columns' refusal.

## For the next lanes

- **E2 / the media wave:** run the five families' media rows against real checkpoints on the GPU; the CPU
  gate proves the row reading and the keep-rule plumbing, not the model outputs. The stage-2 gate is on by
  default now, so a media row that disagrees fails the wave.
- **A video wave:** the qwen3-vl-embedding per-clip pixel-budget reconciliation above.
- **A new media family:** declare no `media_approximation` once the reference's media path is wired; the
  fixture `fixture-vl-embed` shows the declaration path for a card that genuinely cannot run an input.
