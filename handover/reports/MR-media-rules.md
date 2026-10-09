# Report MR: the late-interaction image keep-rules, the video frame count and the media head (lane `media-rules`)

**Status:** DONE. The product mechanisms the brief and the operator addendum asked for are implemented, tested and
gated; the image keep-rules item ends in the brief's declared stop-and-report branch (the v0.31.0 pooling route
cannot return per-position token ids), with the options below for the owner and the recipe-fix lane.

Base: `rfc-0001` tip `681a8cea`; the branch merged `rfc-0001` (`89e7a3b6`, lane l10a) at `3c280aab`, and after the
recipe line landed it merged the moved `rfc-0001` (`247c3d53`: the family layout, l10b, mrl-core, rf-engine,
sync-hardening and scoring-fixes) at `426f1a18`, porting the lane onto it at `0c8467d2`. The final gate ran on
`0c8467d2`.

## Commits

| hash | subject |
|---|---|
| `4e22cf7e` | The engine's video sampling: the Qwen3-VL fps rule and its exact prompt count |
| `620a60c0` | The media side's fixed head as a system message |
| `13419bb6` | Docs, CHANGELOG and the regenerated public surface for the video rule and the media head |
| `3c280aab` | Merge `rfc-0001` (`89e7a3b6`) |
| `5e9ed3ca` | The pre/post-processing review's lane items: M13, the typed skip error and the video-pruning declaration |
| `900d9037` | Verifier round-1 findings: the mixed-batch head, the pruning family guard and the exact fit count |
| `1cb6d21a` | Round-2 confirmation findings: the pinned timestamp count, the fps family on `Preprocessing` and the client wiring test |
| `e23bbe12` | Round-3 confirmation findings: the deferred-processor fps form and the pinned qwen3_vl refusal |
| `926b034a` | Final confirmation minors: the resolved fps family and the cost estimate's tokenizer test |
| `426f1a18` | Merge `rfc-0001` (`247c3d53`: the family layout, l10b, mrl-core, rf-engine, sync-hardening, scoring-fixes) |
| `0c8467d2` | Port onto the moved `rfc-0001`: the family goldens, the pruning tests and the qwen3-vl fps recipe |

## What changed (per brief item)

1. **Image keep-rules (brief item 1) — established, stopped, options reported.** At the v0.31.0 tag the
   `/pooling` route cannot return the engine's per-position token ids: `PoolingParams.returned_token_ids` is the
   step pooler's *output-column* slice (`vllm/model_executor/layers/pooler/tokwise/methods.py:104-136`), and
   `requires_token_ids` is marked "Internal use only" (it only makes the worker materialise a CPU prompt-id copy,
   `vllm/pooling_params.py:62-69`); the HTTP response carries vectors/strings only
   (`vllm/entrypoints/pooling/pooling/protocol.py:102-106`), and `return_token_ids` exists on the generation routes
   only (`vllm/entrypoints/openai/chat_completion/protocol.py:416`). The engine-side
   `PoolingRequestOutput.prompt_token_ids` (`vllm/outputs.py:247-268`) never crosses the wire. So the client
   cannot compute the checkpoints' own image-position masks (topk: image-token positions only; pplx-late:
   drop the 32 skiplist ids and keep the rest) from the reply, and the current rule stays: a media document's
   vectors are kept whole, on record (`skip_unapplied`), which is still honest and tested. **Options** for the
   owner: (a) apply the checkpoint's mask engine-side in the plugin pooler (it already receives the prompt ids
   via `requires_token_ids`; no extra request, but the rule's home moves to the plugin); (b) ask the engine's
   chat `/tokenize` route for the same render's ids (`vllm/entrypoints/serve/tokenize/`, which returns the
   expanded prompt ids) and apply the declared mask client-side -- this keeps the rule in
   `rcp_ndcg.data.postprocess` but needs a transport path that reaches the engine root (the transport joins
   `base_url + path`, and `/tokenize` sits at the origin) and one extra request per media item; (c) keep the
   declared approximation. Recommendation: (a) if exactness with no extra request is wanted, (b) if the rule
   must stay client-side.
2. **Qwen3-VL video frames (brief item 2) — implemented.** `rcp_ndcg.data.resolution.qwen3_vl_video_frame_indices`
   ports vLLM v0.31.0 `Qwen3VLVideoBackend.compute_frames_index_to_sample`
   (`vllm/multimodal/video.py:360-400`: `fps = min(target, 30)`, `int(total/original*fps)`, clamped to
   `[4, 768, total]`, `.round()` indices). `VideoPolicy` gains `fps` (exactly one of `num_frames`/`fps` for
   `wire: video_url`), and `content_media_tokens` counts a `qwen3_vl` container from the clip's recorded frame
   count and rate: the chat template's outer vision pair once, then per temporal group the timestamp line, the
   vision pair and the merged patches, with the timestamp tokens counted exactly when the client's tokenizer is
   passed (the family's 10-token bound otherwise). The test reproduces E1's measured 98 and 458 tokens for the
   media set's two 64-frame/8 fps clips with the checkpoint's own vendored tokenizer. `prepare_request` and
   `fit_media_to_budget` take the caller's tokenizer, so the media fit's gate uses the exact count; the judging
   window budget and the cost estimate pass their tokenizer too. The engine's video-token pruning
   (`--video-pruning-rate`, EVS/VidCom2) is declared (`engine_video_pruning`,
   `engine_video_pruning_method`), counted with the engine's retention formula for the qwen3_vl layout, and the
   recipe loader refuses a serve pruning flag the client has not declared (and a declaration the serve args do
   not carry). The Qwen3-VL backend ignores `num_frames`, so a pinned `num_frames` policy on that family is
   refused at count time (declare `fps`); the recipe's fps pin is recipe-fix's.
3. **Media-side fixed head as a system message (operator addendum) — implemented.** A new declared client field,
   `PoolingEndpoint.media_head_as_system`, sends a media document's leading fixed template segments (resolved
   from the template, specials by name) as a leading `system` message; `PoolRequest.system_head` carries it and
   the pooling adapter renders it per media item only (a text item's fitted render already carries the head). The
   user turn keeps only the content span, the budget's fixed overhead already reserves the head, and the startup
   media probe and its baseline both carry it so the engine's media delta cancels it. For pplx-embed-v2-late
   this restores the card's `[D] ` render; the recipe declaration is recipe-fix's.
4. **Pre/post-processing review items — implemented.** M13: `max_duration_s` no longer refuses a prepared frame
   set whose container was dropped after the source was checked (the limit is a container limit).
   `skip_keep_mask` raises a typed `rcp_ndcg.errors.DataError` with a hint instead of a bare `ValueError` from
   inside a client. M14/A6: the video-pruning declaration and the recipe-side cross-check above. The fps-driven
   default wording ("the checkpoint's processor fps on the Qwen3-VL backend; 32 frames on the default loader
   elsewhere") is corrected in the docstrings and the docs.

## Port onto the moved `rfc-0001` (operator note 2026-10-09)

`rfc-0001` moved to `247c3d53` (the family layout, l10b, mrl-core, rf-engine, sync-hardening,
scoring-fixes). The merge (`426f1a18`) conflicted in `data/resolution.py`, the docs, the schemas and
`tests/data/test_resolution.py`; the resolutions were taken from the operator's `int/round2` merge
(`resolution.py` keeps both the `gemma4` processor and the fps rule; the docs keep mrl-core's `mrl_dim`
text; the schemas are re-exported), and the merged tree equalled `int/round2` before the port. The port
(`0c8467d2`):

- **The per-variant goldens (operator item 1).** `client.media_head_as_system` is a new request-shaping
  fingerprint input, so it moves the input set and the corpus key of the four pooling variants
  (`pplx-embed-v2-context-9b-preview`, `pplx-embed-v2-late-0.6b`, `topk-embed-v1-small`,
  `topk-embed-v1-xsmall`). Each declares two `DELTAS.json` entries (the new input, `null -> "false"`, and its
  `fingerprint.fingerprint` hash) with the reason (a new default-false field; no request bytes move) and
  evidence (`config.py`'s field, `fingerprint.py`'s classification, and no committed corpus for the four).
  The shrink-only delta check now walks the flat `fingerprint.inputs.*` keys, so a null-golden input delta is
  actually checked instead of falling through to `None` and reading as stale.
- **`test_recipe.py` (operator item 2).** The five video-pruning tests' helper copies `family.yaml` (the
  fixture families) instead of the deleted standalone `recipe.yaml`; the tests pass.
- **`qwen3-vl-embedding-2b` (operator item 3).** The family declares `client.video_policy {fps: 2, wire:
  video_url, engine_video_pinning: true}` with `--media-io-kwargs '{"video": {"fps": 2}}'` and notes restated
  to the backend's fps rule (the old `num_frames` pin is inert); its reference's media mode computes the
  engine's realised frame count (`int(total/original*fps)`, clamped to 4..768) from each pairs entry's own
  `num_frames`/`fps`; the media set writes an fps policy's clips at `VIDEO_FRAMES` (64, keeping the media set's
  existing clips); the equivalence media stage reconstructs the sent container with its rate, counts the
  realised frames and passes the recipe's tokenizer to the media count; and the stub engine honours
  `--media-io-kwargs`'s `video.fps` with the engine's own tokenizer. The recipe's golden differences (the
  policy fields, `serve.extra_args`, the argv, the two fingerprint inputs and the hash, and the notes) are
  declared in `DELTAS.json`; the recipe module's own contract pins moved with it.
  `test_the_media_stage_holds_the_client_to_the_card_on_every_image_bucket` and
  `test_the_engine_media_count_holds_under_the_pin_and_fails_unpinned` both pass.

## Verification

Round 1 (two independent verifiers, fresh context, lens A correctness and lens B regressions, model
DeepSeek-V4.1-flash at xhigh) found three majors and minors; round 2 (one fresh confirmation verifier) found a
pinned-`num_frames` major and a `Preprocessing` major; round 3 (one fresh confirmation) found the
deferred-processor form of that major and the pinned-residual major; the final confirmation (one fresh verifier)
returned **PASS** with three residual minors (two fixed, one handed to recipe-fix).

| round | verdict | findings and disposition |
|---|---|---|
| 1 | FAIL | F1 mixed text+media batch duplicated the system head on text items -- fixed per media item, test added; F2a `engine_video_pruning` silently ignored on qwen2_vl/qwen2_5_vl -- refused for families whose pruned layout is not ported, test added; F2b the media fit gated on the timestamp bound -- tokenizer threaded through `prepare_request`/`fit_media_to_budget`, client-level test added; minors (the stale `VideoPolicy` docstring/schema, the CHANGELOG section grouping, the unknown-count message, the `max_frames` clamp, a single-frame fps clip, `approx_media_tokens` for another family) fixed with tests. |
| 2 | FAIL | Major: a pinned `num_frames` qwen3_vl container still counted the timestamp bound -- the pinned count now uses the tokenizer's exact timestamps (386 for the E1 icon, not 514). Minors: the judging window budget and the cost estimate now pass their tokenizer; `Preprocessing` refuses fps beside an explicit non-qwen3_vl processor; the frames wire refuses a pruning method without a rate; stale docstrings and the CHANGELOG kwargs corrected. |
| 3 | FAIL | Major: the `Preprocessing` fps validator refused the documented deferred-processor form (the judge config fills qwen3_vl via `for_processor`) -- the validator now refuses only an explicit other family. Major residual: the pinned qwen3_vl count was exact for a layout the engine never renders -- the pinned `num_frames` policy is now refused at count time (the backend ignores it; declare `fps`), and the tests moved to the fps rule. Minors: the deferred form resolved to another family is refused in `_effective_preprocessing`; docs/CHANGELOG updated; tests added. |
| final | PASS | No blocker. Residual minors: the `cost.py` tokenizer wiring had no test -- now pinned by a recorder test; the deferred-to-another-family approximation path -- now refused in `_effective_preprocessing`; the shipped `qwen3-vl-embedding-2b` recipe still pins `num_frames` and now refuses at count time -- recipe-fix must switch it to `fps` (owner decision), the pre-fix behaviour was the silent miscount the round-3 major required refusing. |

Every fix landed test-first (the failing runs are quoted in the fix commits' evidence and the checks below).

## Checks

Last commands and results (on the ported final head `0c8467d2`, gate `bin/gate lane/media-rules`):

- `bin/gate lane/media-rules` -> **GATE: PASS**: ruff-check/format clean; basedpyright 0 errors;
  `pytest tests/` 3469 passed, 96 skipped; contract+docs 295 passed, 52 skipped; mkdocs `--strict` ok;
  `rcp-ndcg-test` 621 passed, 349 skipped; the recipes step "no failure outside the baseline (0 baseline
  failures remain, 34 fixed)"; vllm-pkg 40 passed and vllm-models 71 passed; run_all 1022 checks / 987 match /
  35 known deviations / 0 failed; human study 67/67; external judges 82/82; public-names clean (0 baselined
  hits); clean tree. The pre-port gate on `926b034a` was also PASS (3312 passed; recipes baseline 34); the
  first gate run on `3c280aab` failed one storage test on a lingering NFS temp file, which passed in
  isolation and on the re-run.
- The port's named recipe tests with `RCP_NDCG_NETWORK_TESTS=1`:
  `test_qwen3_vl_embedding.py::test_the_media_stage_holds_the_client_to_the_card_on_every_image_bucket` and
  `::test_the_engine_media_count_holds_under_the_pin_and_fails_unpinned` -> passed; the whole module 14 passed;
  the family golden guard 32 passed; `rcp-ndcg-test/tests/test_recipe.py` 32 passed.
- Failing-test-first evidence: the pinned-qwen3_vl refusal (`514 != 386`), the mixed-head test
  (`['system'] != ['user']`), the pruning-family refusal, the client fit (`media_drop`), the deferred-processor
  test and the frames-wire method test all failed on the pre-fix code before their fixes.
- Focused sets on the final head: `tests/data/test_resolution.py tests/data/test_prepare.py
  tests/data/test_postprocess.py tests/inference/test_client_budget.py tests/inference/test_pool_client.py
  tests/inference/test_pipeline_stages.py tests/judging/test_media_judging.py tests/judging/test_cost.py
  tests/contract` -> 439 passed, 1 skipped; `rcp-ndcg-test` recipe/fingerprint/media -> 64 passed.
- The lane worktree's own venv predates the `rcp-0001` reader entry points (lane l10a), so the data-reader and
  docs-example tests fail there on stale install metadata (`unknown dataset URI scheme 'jsonl'`); the
  authoritative gate environment is green. No `uv sync`/`uv lock`/`uv pip` was run.

## Open questions

1. **The image keep-rules (item 1) need an owner decision.** The pooling route cannot return the engine's
   per-position ids at v0.31.0 (evidence above); the options are the engine-side plugin mask, the `/tokenize`
   round-trip (which needs a root-relative transport path), or keeping the approximation. The recipe-fix lane
   cannot declare an exact rule until one is chosen; the pure mask functions (`skip_keep_mask` and a media
   sibling) are trivial to add once the ids route is fixed.
2. **The shipped `qwen3-vl-embedding-2b` recipe now declares the engine's fps rule** (`client.video_policy
   {fps: 2}`, `--media-io-kwargs '{"video": {"fps": 2}}'`) and its reference reports the realised frame
   count; the old pinned `num_frames` (inert on the Qwen3-VL backend) is gone. Resolved in the port
   (`0c8467d2`); E2 still validates the engine count on the node.
3. **The pplx-late media token count needs one definition.** The product counts the media block
   (`patches + 2`); the recipe's reference counts `patches + 3` (the `[D] ` head inside the image's tokens).
   With the head now sent as a system message, the reference's media mode should count `patches + 2`; the
   recipe declaration is recipe-fix's.
4. **The equivalence harness's fps arm was ported for this recipe**: `equivalence/media.py` now rebuilds the
   sent container with its rate, counts the realised fps frames and passes the recipe's tokenizer, and
   `stub_engine.py` honours `--media-io-kwargs`'s `video.fps` with the engine's own tokenizer. What remains
   with harness-fix (review A4) is the general per-recipe coverage and the pinned-count path's exact count
   where no tokenizer is resolvable.
5. **The media stage gates media geometry/tokens only, not scores** (review M5/A2, harness-fix): image-pair
   vectors/scores remain ungated; unchanged here.

## CHANGELOG entry

**### Public surface**

- **The engine's video sampling (owner decision 2026-10-09)**: `VideoPolicy` gains `fps`, the vLLM v0.31.0
  `Qwen3VLVideoBackend`'s own rule, and `num_frames` becomes optional: a `wire: video_url` policy declares
  exactly one of `num_frames` (a pinned uniform count) or `fps` (the engine's rate), and `wire: frames` still
  requires `num_frames`. The new `rcp_ndcg.data.resolution.qwen3_vl_video_frame_indices` ports the backend's
  rule (`int(total_frames / original_fps * fps)`, clamped to its 30 fps ceiling and 4..768 frame bounds), and
  `content_media_tokens` gains an optional `tokenizer`: a `qwen3_vl` container under `fps` is counted from the
  clip's recorded frame count and rate, its timestamp lines exactly when the client's tokenizer is passed (the
  family's 10-token bound otherwise), and the chat template's own vision pair around the placeholder is now
  included. `approx_media_tokens` counts the fps rule's frames too; `prepare_request` and `fit_media_to_budget`
  take the caller's `tokenizer` so the media fit's gate uses the exact count. `VideoPolicy` also gains
  `engine_video_pruning` and `engine_video_pruning_method`: a nonzero engine `--video-pruning-rate` retains a
  computed subset of the per-frame tokens (the EVS or VidCom2 formula, ported for the qwen3_vl family; a
  per-frame family's flat pruned run is refused), the client counts that layout, and the recipe loader refuses
  a serve pruning flag the client has not declared (and a declaration the serve args do not carry). The fps
  rule is likewise refused beside a non-qwen3_vl processor family, and a pinned `num_frames` on the qwen3_vl
  family is refused at count time (that backend samples by fps and ignores the pin; declare `fps`).
- **`PoolingEndpoint.media_head_as_system`** (a media document's fixed head as a system message, for a
  pass-through engine chat template) and **`PoolRequest.system_head`** (the field the pooling adapter renders
  it from).

**### Fixed**

- **The Qwen3-VL video token count**: the engine's prompt renders one timestamp line and one vision block per
  temporal group inside the chat template's own vision pair, and under the engine's fps rule the frame count
  follows the clip, not a declared `num_frames` (which the backend ignores). The count now reproduces E1's
  measured 98 and 458 tokens for the media set's two 64-frame/8 fps clips (the test loads the checkpoint's own
  vendored tokenizer); the timestamp lines are exact when the client has a tokenizer and the family's bound
  otherwise.
- **A media document's trained head can be sent as a system message**: a checkpoint whose engine chat template
  injects no frame of its own (pplx-embed-v2-late's pass-through template) otherwise renders an image-only
  document without the `[D] ` prefix its card's sentence-transformers path sends as a system message. The
  client now sends the shape's leading fixed template segments as a leading `system` message under
  `media_head_as_system: true`, keeps the user turn to the content span, and the startup media probe's
  baseline carries the same head (the media delta still cancels it).
- **`max_duration_s` no longer refuses a prepared frame set** for a duration its dropped container no longer
  carries (the source's duration was checked when it was sampled); `skip_keep_mask` raises a typed
  `rcp_ndcg.errors.DataError` with a hint instead of a bare `ValueError` from inside a client; and the engine's
  video-token pruning (`--video-pruning-rate`) is now declared, counted and cross-checked against the serve
  args instead of silently changing the prompt layout.

## Public surface changes

- `rcp_ndcg.data.resolution.qwen3_vl_video_frame_indices` (new public function); `content_media_tokens` gains
  the optional `tokenizer` keyword; `VideoPolicy` gains `fps`, `engine_video_pruning` and
  `engine_video_pruning_method`, and `num_frames` becomes optional (`wire`-dependent exactly-one-of).
- `rcp_ndcg.data.prepare.prepare_request` and `fit_media_to_budget` gain the optional `tokenizer` keyword.
- `PoolingEndpoint.media_head_as_system` (new field); `PoolRequest.system_head` (new field).
- `rcp_ndcg.data.postprocess.skip_keep_mask` raises `DataError` instead of `ValueError` on empty ids.
- Schemas regenerated: `schemas/index.v1.json`, `schemas/run-config.v1.json`; the contract snapshot
  `tests/contract/snapshots/python_api.json`. No CLI command, flag or exit code changed.

## Files outside scope

- `rcp-ndcg/src/rcp_ndcg/inference/config.py`, `inference/types.py`, `inference/adapters/pooling.py`,
  `inference/clients/_base.py`, `judging/tokens.py`, `judging/judging.py`, `judging/cost.py`,
  `data/prepare.py`, `data/text_policy.py` -- the minimum wiring the mechanisms need (the head field and the
  request field, the adapter's messages, the tokenizer passes, the fps-family checks, the M13/typed-error and
  pruning fixes); each listed in the per-fix commits.
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipe.py` -- the video-pruning cross-check the operator's M14/A6 item
  asked for.
- `rcp-ndcg-test/src/rcp_ndcg_test/fingerprint.py` -- one line classifying `media_head_as_system` as a
  request-shaping field (the unclassified-field guard refuses a new field otherwise).
- `rcp-ndcg-test/tests/test_recipe.py` -- the pruning cross-check tests (family fixture layout).
- `tests/_tokenizers.py` -- the vendored Qwen3-VL tokenizer loader shared by the E1 and fit tests.
- The port onto the moved `rfc-0001`: `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-vl-embedding/{family.yaml,
  reference.py}` (the fps policy and the reference's realised frame count);
  `rcp-ndcg-test/src/rcp_ndcg_test/{observe/media_set.py,equivalence/media.py}` and
  `rcp-ndcg-test/tests/stub_engine.py` (the fps arm and the tokenizer pass the operator's item 3 needs);
  `rcp-ndcg-test/tests/recipes/golden/DELTAS.json`, `test_family_goldens.py` (the flat-input walk) and
  `test_qwen3_vl_embedding.py` (the moved contract pins).
- Docs (`docs/concepts/preprocessing.md`, `judges.md`, `late-interaction.md`), `CHANGELOG.md`, `schemas/`,
  `tests/contract/snapshots/`.

## For the next lanes

- **recipe-fix**: declare `media_head_as_system: true` for `pplx-embed-v2-late-0.6b`; change its reference's
  media mode to count `patches + 2` (the head is a system message, not part of the media block); declare the
  image keep-rule once the owner picks an ids route (item 1). The `qwen3-vl-embedding-2b` fps switch and the
  pruning-test family layout are done here.
- **harness-fix**: the general per-recipe media-set/stub coverage the port's fps arm does not yet exercise,
  the pinned-count exact count where no tokenizer is resolvable, and the review's A4 gaps.
- **06/07**: the late-interaction page and the preprocessing page now describe the fps rule, the pruning
  declaration and the media system head; the media gate remains an input gate (A2/A9).
